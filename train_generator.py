# /home/machao/pythonproject/nfsba/train_generator.py

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import random
import os
import argparse
from tqdm import tqdm
import time
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.multiprocessing as mp
import math

# --- Project Imports ---
from utils.config import load_config
from utils.distributed import setup_ddp, cleanup_ddp, is_main_process, reduce_tensor, barrier
from utils.dct import block_dct, block_idct, get_ac_coeffs, set_ac_coeffs
from utils.stats import calculate_stat_loss, load_target_stats
from models.generator import MLPGenerator, CNNGenerator
from models.proxy import load_proxy_models
from attacks.nfsba import NFSBA_Attack # <--- 修改：只导入 NFSBA_Attack
from data.datasets import get_dataset
from utils.metrics import calculate_lpips
from utils.losses import perturbation_loss, compute_trigger_loss, calculate_generator_loss, \
    statistics_loss  # <--- 修改：从 losses 导入 compute_trigger_loss and calculate_generator_loss
from constants import CIFAR10_MEAN, CIFAR10_STD

# --- 定义反标准化函数 ---
def denormalize_batch(tensor, mean, std):
    mean = mean.to(tensor.device)
    std = std.to(tensor.device)
    tensor = tensor.clone()
    tensor.mul_(std).add_(mean)
    tensor = torch.clamp(tensor, 0, 1)
    return tensor
# ----------------------------------------------------

# --- Set Seed Function ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(rank, world_size, cfg_path):
    # --- Basic Setup ---
    cfg = load_config(cfg_path)
    try:
        setup_ddp(rank, world_size, cfg.master_port)
    except Exception as e:
        print(f"Rank {rank}: DDP Setup failed: {e}")
        return
    device = torch.device(f"cuda:{rank}")
    set_seed(cfg.seed + rank)
    if is_main_process(rank): print(f"NFSBA Generator Training Started - Rank {rank}/{world_size} on GPU {torch.cuda.current_device()}")

    # --- Load Data ---
    if is_main_process(rank): print("Loading data...")
    try:
        full_train_dataset = get_dataset(cfg.dataset, cfg.data_path, train=True, img_size=cfg.img_size)
        dataset_size = len(full_train_dataset)
        indices = list(range(dataset_size))
        split_generator = torch.Generator().manual_seed(cfg.seed)
        indices = torch.randperm(dataset_size, generator=split_generator).tolist()
        val_split_idx = int(np.floor(0.1 * dataset_size))
        train_indices = indices[val_split_idx:]
        val_indices = indices[:val_split_idx]
        train_subset = Subset(full_train_dataset, train_indices)
        val_subset = Subset(full_train_dataset, val_indices)

        train_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.seed)
        val_sampler = DistributedSampler(val_subset, num_replicas=world_size, rank=rank, shuffle=False)

        train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, sampler=train_sampler, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
        val_batch_size = cfg.batch_size * 2
        val_loader = DataLoader(val_subset, batch_size=val_batch_size, sampler=val_sampler, num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

        if is_main_process(rank):
            print(f"Data loaded: {len(train_subset)} train samples ({len(train_loader)} steps/epoch), "
                  f"{len(val_subset)} val samples ({len(val_loader)} steps)")
    except Exception as e:
        if is_main_process(rank): print(f"Error loading data: {e}")
        cleanup_ddp()
        return
    barrier()

    # --- Load Models ---
    if is_main_process(rank): print("Loading models...")
    try:
        # --- 根据配置选择生成器 ---
        if cfg.generator.type == "MLP":
            if is_main_process(rank): print("Using MLPGenerator")
            ac_dim = cfg.dct_block_size ** 2 - 1
            generator_model = MLPGenerator(ac_dim=ac_dim,
                                           condition_dim=cfg.condition_dim,
                                           hidden_dims=cfg.generator.hidden_dims,
                                           initial_scale=getattr(cfg.generator, 'initial_scale', 0.1),
                                           learnable_scale=getattr(cfg.generator, 'learnable_scale', False)
                                          ).to(device)
        elif cfg.generator.type == "CNN":
            if is_main_process(rank): print("Using CNNGenerator")
            generator_model = CNNGenerator(condition_dim=cfg.condition_dim,
                                           initial_scale=getattr(cfg.generator, 'initial_scale', 0.1),
                                           learnable_scale=getattr(cfg.generator, 'learnable_scale', False)
                                          ).to(device)
            # generator_model.generator_type = "CNN" # Can add this if needed
        else:
            raise ValueError(f"Unknown generator type: {cfg.generator.type}")
        # ----------------------------

        generator = DDP(generator_model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

        proxy_models = load_proxy_models(cfg.proxy_models.names, cfg.proxy_models.paths, device, cfg.num_classes)
        for model in proxy_models: model.eval()
        if is_main_process(rank): print(f"Loaded {len(proxy_models)} proxy models.")

        # --- Attacker Util ---
        attacker_util = NFSBA_Attack(
            generator.module,
            cfg.dct_block_size,
            cfg.condition_dim,
            cfg.num_classes,
            cfg.condition_type,
            device
        )
        if cfg.condition_type == 'embedding':
            emb_path = f'./checkpoints/cond_emb_{cfg.dataset}_{cfg.condition_dim}d.pt'
            if os.path.exists(emb_path):
                # Load embeddings onto the correct device directly
                attacker_util.condition_embeddings.load_state_dict(torch.load(emb_path, map_location=device))
                if is_main_process(rank): print(f"Loaded condition embeddings from {emb_path}")
            else:
                if is_main_process(rank):
                    print(f"Condition embeddings not found at {emb_path}, using random init and saving.")
                    os.makedirs(os.path.dirname(emb_path), exist_ok=True)
                    torch.save(attacker_util.condition_embeddings.state_dict(), emb_path)
            barrier() # Ensure all processes load or save before continuing

    except Exception as e:
        if is_main_process(rank): print(f"Error loading models: {e}")
        cleanup_ddp()
        return
    barrier()

    # --- Load Stats ---
    if is_main_process(rank): print("Loading target statistics...")
    try:
        # Load stats to CPU first to avoid duplicating on each GPU memory unnecessarily
        target_stats = load_target_stats(cfg.stats.target_path, device='cpu')
        fixed_beta_stats = target_stats.get('fixed_beta', 1.0)
        if is_main_process(rank): print(f"Target stats loaded. Using fixed beta = {fixed_beta_stats}.")
    except FileNotFoundError as e:
        if is_main_process(rank): print(f"Error: {e}. Run precompute_stats.py first.")
        cleanup_ddp()
        return
    barrier()

    # --- Optimizer, Scaler ---
    optimizer = optim.Adam(generator.parameters(), lr=cfg.generator.lr, betas=tuple(cfg.generator.betas))
    scaler = GradScaler(enabled=cfg.use_amp)
    if is_main_process(rank): print(f"Optimizer: Adam, LR: {cfg.generator.lr}, Betas: {cfg.generator.betas}")
    if is_main_process(rank): print(f"Automatic Mixed Precision (AMP): {'Enabled' if cfg.use_amp else 'Disabled'}")

    # --- Resume Logic ---
    start_epoch = 0
    best_val_loss = float('inf')
    # Use consistent naming for checkpoints
    ckpt_dir = os.path.dirname(cfg.generator.checkpoint_path)
    base_ckpt_name = os.path.splitext(os.path.basename(cfg.generator.checkpoint_path))[0]
    last_ckpt_path = os.path.join(ckpt_dir, f"{base_ckpt_name}_last.pt")
    best_ckpt_path = os.path.join(ckpt_dir, f"{base_ckpt_name}_best.pt") # Explicit best path

    if os.path.exists(last_ckpt_path) and getattr(cfg.generator, 'resume_generator', False):
        try:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank} # Map to current rank's device
            checkpoint = torch.load(last_ckpt_path, map_location=map_location)
            generator.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            # Load best_val_loss from checkpoint if resuming
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            # Load scaler state if using AMP and state exists
            if cfg.use_amp and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            if is_main_process(rank): print(f"Resuming generator training from epoch {start_epoch}")
        except Exception as e:
            if is_main_process(rank): print(f"Warning: Could not load resume checkpoint '{last_ckpt_path}': {e}. Starting from scratch.")
            start_epoch = 0
            best_val_loss = float('inf') # Reset best loss if starting over
    barrier()

    # --- 创建 DC 掩码 (在设备上) ---
    dc_mask = torch.ones(1, 1, cfg.dct_block_size, cfg.dct_block_size, device=device)
    dc_mask[..., 0, 0] = 0
    # ---------------------------

    # --- Training Loop ---
    if is_main_process(rank): print("Starting generator training...")
    num_blocks_h = cfg.img_size // cfg.dct_block_size
    num_blocks_w = cfg.img_size // cfg.dct_block_size
    ac_dim = cfg.dct_block_size ** 2 - 1 # Only used if generator.type is MLP

    for epoch in range(start_epoch, cfg.generator.epochs):
        epoch_start_time = time.time()
        generator.train()
        train_sampler.set_epoch(epoch) # Important for shuffling in DDP

        running_losses = {'total': 0.0, 'dist': 0.0, 'energy': 0.0, 'trigger': 0.0, 'perturb': 0.0, 'lpips': 0.0}
        num_steps = 0

        # Determine loss weights for the current phase
        is_phase1 = epoch < int(cfg.generator.epochs * cfg.generator.phase1_epochs_ratio)
        phase_idx = 0 if is_phase1 else 1
        current_weights = {
            'dist': cfg.generator.lambda_stat_dist[phase_idx],
            'energy': cfg.generator.lambda_stat_energy[phase_idx],
            'trigger': cfg.generator.lambda_trigger[phase_idx],
            'perturb': cfg.generator.lambda_perturb[phase_idx],
            'lpips': getattr(cfg.generator, 'lambda_lpips', [0.0, 0.0])[phase_idx]
        }

        if is_main_process(rank) and (epoch == start_epoch or epoch == int(cfg.generator.epochs * cfg.generator.phase1_epochs_ratio)):
            # Print weights at the start of each phase
            weights_str = ', '.join([f"L{k}={v:.3f}" for k, v in current_weights.items()])
            print(f"Epoch {epoch+1} [Phase {phase_idx+1}] Weights: {weights_str}")

        pbar = tqdm(train_loader, desc=f"Gen Epoch {epoch + 1}/{cfg.generator.epochs}", disable=not is_main_process(rank))

        for i, (images_normalized, labels) in enumerate(pbar):
            images_normalized = images_normalized.to(device, non_blocking=True)
            # Create target labels on the correct device
            targets = torch.full_like(labels, cfg.target_label).to(device, non_blocking=True)
            batch_size = images_normalized.shape[0]
            num_blocks_total_per_sample = images_normalized.shape[1] * num_blocks_h * num_blocks_w

            # Denormalize images to [0, 1] range on the device
            images_0_1 = denormalize_batch(images_normalized, CIFAR10_MEAN, CIFAR10_STD)

            optimizer.zero_grad(set_to_none=True) # More efficient zeroing

            with autocast(enabled=cfg.use_amp):
                # --- Forward Pass ---
                dct_blocks = block_dct(images_0_1, cfg.dct_block_size) # (B, C, Bh, Bw, 8, 8)
                N = dct_blocks.numel() // (cfg.dct_block_size ** 2) # More robust calculation: B*C*Bh*Bw

                condition_vec = attacker_util.get_condition_vec(targets, num_blocks_total_per_sample) # (N, cond_dim)

                if cfg.generator.type == "MLP":
                    ac_coeffs_flat = get_ac_coeffs(dct_blocks).view(N, ac_dim) # Use N
                    if ac_coeffs_flat.shape[0] != condition_vec.shape[0]:
                        raise RuntimeError(f"MLP: Dimension 0 mismatch ac_coeffs ({ac_coeffs_flat.shape[0]}) vs condition_vec ({condition_vec.shape[0]})")
                    scaled_perturbation_flat = generator(ac_coeffs_flat.contiguous(), condition_vec.contiguous())
                    loss_perturb = perturbation_loss(scaled_perturbation_flat)
                    dct_blocks_poisoned = set_ac_coeffs(dct_blocks, scaled_perturbation_flat)

                elif cfg.generator.type == "CNN":
                    dct_blocks_2d = dct_blocks.view(N, 1, cfg.dct_block_size, cfg.dct_block_size) # Use N
                    scaled_perturbation_2d = generator(dct_blocks_2d.contiguous(), condition_vec.contiguous())
                    final_perturbation_2d = scaled_perturbation_2d * dc_mask # dc_mask is already on device
                    loss_perturb = F.mse_loss(final_perturbation_2d, torch.zeros_like(final_perturbation_2d))
                    final_perturbation_blocks = final_perturbation_2d.view(batch_size, images_normalized.shape[1], num_blocks_h, num_blocks_w, cfg.dct_block_size, cfg.dct_block_size)
                    dct_blocks_poisoned = dct_blocks + final_perturbation_blocks

                # --- Calculate all losses ---
                # Move target stats needed for this batch to the device
                temp_target_stats_dev = {
                    'dist_params': [p.to(device, non_blocking=True) for p in target_stats['dist_params']],
                    'energy_ratios': [r.to(device, non_blocking=True) for r in target_stats['energy_ratios']]
                }
                # Use statistics_loss wrapper (which calls calculate_stat_loss)
                loss_stat_dist, loss_stat_energy = statistics_loss(
                    dct_blocks_poisoned, temp_target_stats_dev, fixed_beta=fixed_beta_stats
                )

                # Use trigger_effectiveness_loss wrapper (which calls compute_trigger_loss)
                loss_trigger = compute_trigger_loss(dct_blocks_poisoned, proxy_models, targets, cfg)

                # IDCT and Clamp needed for LPIPS calculation
                x_poisoned = block_idct(dct_blocks_poisoned, cfg.dct_block_size)
                x_poisoned = torch.clamp(x_poisoned, 0.0, 1.0)
                loss_lpips = calculate_lpips(images_0_1, x_poisoned, device)

                # Check and handle NaN LPIPS
                is_lpips_nan = False
                current_lpips_val = 0.0
                if torch.is_tensor(loss_lpips):
                    is_lpips_nan = torch.isnan(loss_lpips).item()
                    if not is_lpips_nan: current_lpips_val = loss_lpips.item()
                elif isinstance(loss_lpips, float):
                    is_lpips_nan = math.isnan(loss_lpips)
                    if not is_lpips_nan: current_lpips_val = loss_lpips

                if is_lpips_nan:
                    loss_lpips = torch.tensor(0.0, device=device) # Use zero tensor if NaN
                    if rank == 0 and i % 100 == 0: print("Warning: LPIPS NaN detected. Setting loss_lpips to 0.")


                # Store calculated losses in a dict
                current_losses = {
                    'dist': loss_stat_dist,
                    'energy': loss_stat_energy,
                    'trigger': loss_trigger,
                    'perturb': loss_perturb,
                    'lpips': loss_lpips
                }

                # Calculate total weighted loss using the function from utils.losses
                total_loss = calculate_generator_loss(current_losses, current_weights)

            # --- Backward Pass & Optimize ---
            scaler.scale(total_loss).backward()
            # Optional: Gradient Clipping
            # scaler.unscale_(optimizer) # Unscale before clipping
            # torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # --- Log losses ---
            # Make sure to handle non-tensor losses if calculate_stat_loss returns floats
            running_losses['total'] += total_loss.item()
            running_losses['dist'] += current_losses['dist'].item() if torch.is_tensor(current_losses['dist']) else current_losses['dist']
            running_losses['energy'] += current_losses['energy'].item() if torch.is_tensor(current_losses['energy']) else current_losses['energy']
            running_losses['trigger'] += current_losses['trigger'].item() if torch.is_tensor(current_losses['trigger']) else current_losses['trigger']
            running_losses['perturb'] += current_losses['perturb'].item() if torch.is_tensor(current_losses['perturb']) else current_losses['perturb']
            running_losses['lpips'] += current_lpips_val # Use the already checked float value
            num_steps += 1

            if is_main_process(rank) and i % 50 == 0:
                 pbar.set_postfix({
                     'Loss': f"{total_loss.item():.4f}",
                     'Ltrig': f"{current_losses['trigger'].item():.4f}", # Make sure Ltrig is tensor
                     'LPIPS': f"{current_lpips_val:.4f}"
                 })
            # ---------------------------------

        # --- End of Epoch ---
        barrier()

        if num_steps > 0:
            avg_losses = {k: v / num_steps for k, v in running_losses.items()}
            loss_tensor = torch.tensor(list(avg_losses.values())).to(device)
            reduced_losses_tensor = reduce_tensor(loss_tensor, world_size)

            if is_main_process(rank):
                epoch_duration = time.time() - epoch_start_time
                reduced_losses_list = reduced_losses_tensor.cpu().numpy()
                loss_keys = list(avg_losses.keys())
                log_str = ', '.join([f"{key.capitalize()}={reduced_losses_list[idx]:.4f}" for idx, key in enumerate(loss_keys)])
                print(f"Epoch [{epoch+1}/{cfg.generator.epochs}] Completed in {epoch_duration:.2f}s.")
                print(f"  Avg Train Losses: {log_str}")
        else:
            if is_main_process(rank): print(f"Epoch [{epoch+1}/{cfg.generator.epochs}] had no training steps.")

        # --- Validation Step ---
        if (epoch + 1) % 5 == 0 or epoch == cfg.generator.epochs - 1:
            generator.eval()
            val_losses = {'total': 0.0, 'dist': 0.0, 'energy': 0.0, 'trigger': 0.0, 'perturb': 0.0, 'lpips': 0.0}
            val_steps = 0
            if is_main_process(rank): print("Running validation...")

            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Gen Validation Epoch {epoch+1}", disable=not is_main_process(rank))
                for images_normalized, labels in val_pbar:
                    images_normalized = images_normalized.to(device, non_blocking=True)
                    targets = torch.full_like(labels, cfg.target_label).to(device, non_blocking=True)
                    batch_size_val = images_normalized.shape[0]
                    num_blocks_total_per_sample_val = images_normalized.shape[1] * num_blocks_h * num_blocks_w
                    N_val = batch_size_val * num_blocks_total_per_sample_val

                    images_0_1 = denormalize_batch(images_normalized, CIFAR10_MEAN, CIFAR10_STD)

                    with autocast(enabled=cfg.use_amp):
                        # --- Validation Forward Pass (similar to training) ---
                        dct_blocks = block_dct(images_0_1, cfg.dct_block_size)
                        condition_vec = attacker_util.get_condition_vec(targets, num_blocks_total_per_sample_val)

                        if cfg.generator.type == "MLP":
                            ac_coeffs_flat = get_ac_coeffs(dct_blocks).view(N_val, ac_dim)
                            scaled_perturbation_flat = generator(ac_coeffs_flat.contiguous(), condition_vec.contiguous())
                            loss_perturb = perturbation_loss(scaled_perturbation_flat)
                            dct_blocks_poisoned = set_ac_coeffs(dct_blocks, scaled_perturbation_flat)

                        elif cfg.generator.type == "CNN":
                            dct_blocks_2d = dct_blocks.view(N_val, 1, cfg.dct_block_size, cfg.dct_block_size)
                            scaled_perturbation_2d = generator(dct_blocks_2d.contiguous(), condition_vec.contiguous())
                            final_perturbation_2d = scaled_perturbation_2d * dc_mask
                            loss_perturb = F.mse_loss(final_perturbation_2d, torch.zeros_like(final_perturbation_2d))
                            final_perturbation_blocks = final_perturbation_2d.view(batch_size_val, images_normalized.shape[1], num_blocks_h, num_blocks_w, cfg.dct_block_size, cfg.dct_block_size)
                            dct_blocks_poisoned = dct_blocks + final_perturbation_blocks

                        # --- Compute Validation Losses ---
                        temp_target_stats_dev = {
                           'dist_params': [p.to(device, non_blocking=True) for p in target_stats['dist_params']],
                           'energy_ratios': [r.to(device, non_blocking=True) for r in target_stats['energy_ratios']]
                        }
                        loss_stat_dist, loss_stat_energy = statistics_loss(
                           dct_blocks_poisoned, temp_target_stats_dev, fixed_beta=fixed_beta_stats
                        )
                        loss_trigger = compute_trigger_loss(dct_blocks_poisoned, proxy_models, targets, cfg)

                        # IDCT and Clamp needed for LPIPS
                        x_poisoned = block_idct(dct_blocks_poisoned, cfg.dct_block_size)
                        x_poisoned = torch.clamp(x_poisoned, 0.0, 1.0)
                        loss_lpips = calculate_lpips(images_0_1, x_poisoned, device)

                        # Handle NaN LPIPS for validation
                        current_lpips_val_val = 0.0
                        if torch.is_tensor(loss_lpips):
                            if not torch.isnan(loss_lpips).item(): current_lpips_val_val = loss_lpips.item()
                        elif isinstance(loss_lpips, float):
                            if not math.isnan(loss_lpips): current_lpips_val_val = loss_lpips

                        if current_lpips_val_val == 0.0 and math.isnan(loss_lpips): # Check if it was NaN
                            loss_lpips = torch.tensor(0.0, device=device)
                            if rank == 0: print("Warning: LPIPS NaN in validation. Setting loss_lpips to 0.")


                        # Store calculated losses
                        current_val_losses = {
                            'dist': loss_stat_dist, 'energy': loss_stat_energy,
                            'trigger': loss_trigger, 'perturb': loss_perturb,
                            'lpips': loss_lpips
                        }
                        # Calculate total weighted validation loss
                        total_loss = calculate_generator_loss(current_val_losses, current_weights)

                    # Accumulate validation losses
                    val_losses['total'] += total_loss.item()
                    val_losses['dist'] += current_val_losses['dist'].item() if torch.is_tensor(current_val_losses['dist']) else current_val_losses['dist']
                    val_losses['energy'] += current_val_losses['energy'].item() if torch.is_tensor(current_val_losses['energy']) else current_val_losses['energy']
                    val_losses['trigger'] += current_val_losses['trigger'].item() if torch.is_tensor(current_val_losses['trigger']) else current_val_losses['trigger']
                    val_losses['perturb'] += current_val_losses['perturb'].item() if torch.is_tensor(current_val_losses['perturb']) else current_val_losses['perturb']
                    val_losses['lpips'] += current_lpips_val_val # Accumulate the float value
                    val_steps += 1
                # ---------------------------------------------

            # --- Aggregate and Log Validation Losses ---
            barrier()
            if val_steps > 0:
                avg_val_losses = {k: v / val_steps for k, v in val_losses.items()}
                val_loss_tensor = torch.tensor(list(avg_val_losses.values())).to(device)
                reduced_val_losses_tensor = reduce_tensor(val_loss_tensor, world_size)

                if is_main_process(rank):
                    reduced_val_losses_list = reduced_val_losses_tensor.cpu().numpy()
                    loss_keys_val = list(avg_val_losses.keys())
                    log_str_val = ', '.join([f"{key.capitalize()}={reduced_val_losses_list[idx]:.4f}" for idx, key in enumerate(loss_keys_val)])
                    current_val_loss = reduced_val_losses_list[0] # Total loss is the first element

                    print(f"  Avg Val Losses: {log_str_val}")

                    is_best = current_val_loss < best_val_loss
                    if is_best:
                        best_val_loss = current_val_loss
                        print(f"  New best validation loss: {best_val_loss:.4f}")

                    # --- Save Checkpoint ---
                    save_dict = {
                        'epoch': epoch,
                        'model_state_dict': generator.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_loss': best_val_loss, # Save the current best loss
                        'scaler_state_dict': scaler.state_dict() if cfg.use_amp else None,
                        'config': vars(cfg)
                    }
                    os.makedirs(ckpt_dir, exist_ok=True) # Ensure directory exists

                    # Save last checkpoint
                    torch.save(save_dict, last_ckpt_path)

                    # Save best checkpoint if current is best
                    if is_best:
                        torch.save(save_dict, best_ckpt_path)
                        print(f"  Best generator model saved to {best_ckpt_path}")
            else:
                if is_main_process(rank): print("  Validation loader was empty or validation step failed.")

        barrier() # Sync before starting next epoch

    if is_main_process(rank): print("Generator training finished.")
    cleanup_ddp()


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        print(f"Warning: Could not set multiprocessing start method to 'spawn': {e}")

    parser = argparse.ArgumentParser(description='Train NFSBA Generator with DDP')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    args = parser.parse_args()

    try:
        main(local_rank, world_size, args.config)
    except Exception as e:
        print(f"Rank {local_rank}: An error occurred during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if dist.is_initialized():
            cleanup_ddp()