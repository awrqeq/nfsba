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
import torch.nn.functional as F # Ensure F is imported

# --- Project Imports ---
from utils.config import load_config
from utils.distributed import setup_ddp, cleanup_ddp, is_main_process, reduce_tensor, barrier
from utils.dct import block_dct, block_idct, get_ac_coeffs, set_ac_coeffs
from utils.stats import calculate_stat_loss, load_target_stats
from models.generator import MLPGenerator
from models.proxy import load_proxy_models
from attacks.nfsba import NFSBA_Attack, compute_trigger_loss
from data.datasets import get_dataset

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

        train_loader = DataLoader(train_subset, batch_size=cfg.batch_size, sampler=train_sampler,
                                  num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
        val_batch_size = cfg.batch_size * 2
        val_loader = DataLoader(val_subset, batch_size=val_batch_size, sampler=val_sampler,
                                 num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

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
        ac_dim = cfg.dct_block_size**2 - 1
        generator = MLPGenerator(ac_dim=ac_dim,
                                 condition_dim=cfg.condition_dim,
                                 hidden_dims=cfg.generator.hidden_dims).to(device)
        generator = DDP(generator, device_ids=[rank], output_device=rank, find_unused_parameters=False)

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
                attacker_util.condition_embeddings.load_state_dict(torch.load(emb_path, map_location=device))
                if is_main_process(rank): print(f"Loaded condition embeddings from {emb_path}")
            else:
                 if is_main_process(rank):
                     print(f"Condition embeddings not found at {emb_path}, using random init and saving.")
                     os.makedirs(os.path.dirname(emb_path), exist_ok=True)
                     torch.save(attacker_util.condition_embeddings.state_dict(), emb_path)
            barrier()

    except Exception as e:
        if is_main_process(rank): print(f"Error loading models: {e}")
        cleanup_ddp()
        return
    barrier()

    # --- Load Stats ---
    if is_main_process(rank): print("Loading target statistics...")
    try:
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
    resume_ckpt_path = cfg.generator.checkpoint_path.replace('.pt', '_last.pt')
    if os.path.exists(resume_ckpt_path) and getattr(cfg.generator, 'resume_generator', False):
        try:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            checkpoint = torch.load(resume_ckpt_path, map_location=map_location)
            generator.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            if cfg.use_amp and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
                 scaler.load_state_dict(checkpoint['scaler_state_dict'])
            if is_main_process(rank): print(f"Resuming generator training from epoch {start_epoch}")
        except Exception as e:
             if is_main_process(rank): print(f"Warning: Could not load resume checkpoint '{resume_ckpt_path}': {e}. Starting from scratch.")
             start_epoch = 0
    barrier()

    # --- Training Loop ---
    if is_main_process(rank): print("Starting generator training...")
    num_blocks_h = cfg.img_size // cfg.dct_block_size
    num_blocks_w = cfg.img_size // cfg.dct_block_size

    for epoch in range(start_epoch, cfg.generator.epochs):
        epoch_start_time = time.time()
        generator.train()
        train_sampler.set_epoch(epoch)

        running_losses = {'total': 0.0, 'dist': 0.0, 'energy': 0.0, 'trigger': 0.0, 'perturb': 0.0}
        num_steps = 0

        is_phase1 = epoch < int(cfg.generator.epochs * cfg.generator.phase1_epochs_ratio)
        phase_idx = 0 if is_phase1 else 1
        lambda_dist = cfg.generator.lambda_stat_dist[phase_idx]
        lambda_energy = cfg.generator.lambda_stat_energy[phase_idx]
        lambda_trig = cfg.generator.lambda_trigger[phase_idx]
        lambda_pert = cfg.generator.lambda_perturb[phase_idx]

        if is_main_process(rank) and epoch == start_epoch: # Print weights only once per phase start
             print(f"Epoch {epoch+1} [Phase {phase_idx+1}] Weights: Ltrig={lambda_trig:.2f}, Ldist={lambda_dist:.2f}, Lener={lambda_energy:.2f}, Lpert={lambda_pert:.2f}")

        pbar = tqdm(train_loader, desc=f"Gen Epoch {epoch+1}/{cfg.generator.epochs}", disable=not is_main_process(rank))

        for i, (images, labels) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            targets = torch.full_like(labels, cfg.target_label).to(device, non_blocking=True)
            num_blocks_total_per_sample = images.shape[1] * num_blocks_h * num_blocks_w # C * Bh * Bw

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=cfg.use_amp):
                # --- Forward Pass ---
                dct_blocks = block_dct(images, cfg.dct_block_size)
                ac_coeffs_flat = get_ac_coeffs(dct_blocks).view(-1, ac_dim) # Shape: (B*C*Bh*Bw, ac_dim)

                # --- CORRECTED get_condition_vec call ---
                # Pass the original batch targets, let the function handle repeat
                condition_vec = attacker_util.get_condition_vec(targets, num_blocks_total_per_sample)
                # Expected shape: (B*C*Bh*Bw, condition_dim)

                # --- Size Check ---
                if ac_coeffs_flat.shape[0] != condition_vec.shape[0]:
                    print(f"Rank {rank} ERROR: Size mismatch before cat!")
                    print(f"ac_coeffs_flat shape: {ac_coeffs_flat.shape}") # Should be (12288, 63) for batch 256, C=3, 8x8 blocks
                    print(f"condition_vec shape: {condition_vec.shape}") # Should also be (12288, 16)
                    # Raise error or skip step
                    raise RuntimeError("Dimension 0 mismatch between ac_coeffs and condition_vec")

                delta_f_flat = generator(ac_coeffs_flat.contiguous(), condition_vec.contiguous())

                ac_coeffs_poisoned_flat = ac_coeffs_flat + delta_f_flat
                dct_blocks_poisoned = set_ac_coeffs(dct_blocks, ac_coeffs_poisoned_flat)

                # --- Compute Losses ---
                loss_perturb = F.mse_loss(delta_f_flat, torch.zeros_like(delta_f_flat))

                # Ensure target stats tensors are on the correct device for loss calculation
                temp_target_stats = {
                    'dist_params': [p.to(device, non_blocking=True) for p in target_stats['dist_params']],
                    'energy_ratios': [r.to(device, non_blocking=True) for r in target_stats['energy_ratios']]
                }

                loss_stat_dist, loss_stat_energy = calculate_stat_loss(
                    dct_blocks_poisoned, temp_target_stats, fixed_beta=fixed_beta_stats
                )

                loss_trigger = compute_trigger_loss(dct_blocks_poisoned, proxy_models, targets, cfg)

                total_loss = (lambda_dist * loss_stat_dist +
                              lambda_energy * loss_stat_energy +
                              lambda_trig * loss_trigger +
                              lambda_pert * loss_perturb)

            # --- Backward Pass & Optimize ---
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # --- Log losses ---
            running_losses['total'] += total_loss.item()
            running_losses['dist'] += loss_stat_dist.item() if torch.is_tensor(loss_stat_dist) else loss_stat_dist
            running_losses['energy'] += loss_stat_energy.item() if torch.is_tensor(loss_stat_energy) else loss_stat_energy
            running_losses['trigger'] += loss_trigger.item() if torch.is_tensor(loss_trigger) else loss_trigger
            running_losses['perturb'] += loss_perturb.item() if torch.is_tensor(loss_perturb) else loss_perturb
            num_steps += 1

            if is_main_process(rank) and i % 50 == 0:
                 pbar.set_postfix({
                     'Loss': f"{total_loss.item():.4f}",
                     'Ltrig': f"{loss_trigger.item():.4f}",
                     # ... Add other loss components if needed ...
                 })

        # --- End of Epoch ---
        barrier()

        if num_steps > 0:
            avg_losses = {k: v / num_steps for k, v in running_losses.items()}
            loss_tensor = torch.tensor([avg_losses['total'], avg_losses['dist'], avg_losses['energy'],
                                        avg_losses['trigger'], avg_losses['perturb']]).to(device)
            reduced_losses = reduce_tensor(loss_tensor, world_size)

            if is_main_process(rank):
                 epoch_duration = time.time() - epoch_start_time
                 print(f"Epoch [{epoch+1}/{cfg.generator.epochs}] Completed in {epoch_duration:.2f}s.")
                 print(f"  Avg Train Losses: Total={reduced_losses[0].item():.4f}, "
                       f"Dist={reduced_losses[1].item():.4f}, Energy={reduced_losses[2].item():.4f}, "
                       f"Trigger={reduced_losses[3].item():.4f}, Perturb={reduced_losses[4].item():.4f}")
        else:
            if is_main_process(rank): print(f"Epoch [{epoch+1}/{cfg.generator.epochs}] had no steps.")


        # --- Validation Step ---
        if (epoch + 1) % 5 == 0 or epoch == cfg.generator.epochs - 1:
            generator.eval()
            val_losses = {'total': 0.0, 'dist': 0.0, 'energy': 0.0, 'trigger': 0.0, 'perturb': 0.0}
            val_steps = 0
            if is_main_process(rank): print("Running validation...")

            with torch.no_grad():
                 val_pbar = tqdm(val_loader, desc=f"Gen Validation Epoch {epoch+1}", disable=not is_main_process(rank))
                 for images, labels in val_pbar:
                     images = images.to(device, non_blocking=True)
                     targets = torch.full_like(labels, cfg.target_label).to(device, non_blocking=True)
                     num_blocks_total_per_sample = images.shape[1] * num_blocks_h * num_blocks_w

                     with autocast(enabled=cfg.use_amp):
                         # --- Repeat forward pass logic for validation ---
                         dct_blocks = block_dct(images, cfg.dct_block_size)
                         ac_coeffs_flat = get_ac_coeffs(dct_blocks).view(-1, ac_dim)
                         # --- CORRECTED get_condition_vec call ---
                         condition_vec = attacker_util.get_condition_vec(targets, num_blocks_total_per_sample)

                         # Use generator.module in eval if needed, but DDP should handle it
                         delta_f_flat = generator(ac_coeffs_flat.contiguous(), condition_vec.contiguous())

                         ac_coeffs_poisoned_flat = ac_coeffs_flat + delta_f_flat
                         dct_blocks_poisoned = set_ac_coeffs(dct_blocks, ac_coeffs_poisoned_flat)

                         # --- Compute Validation Losses ---
                         loss_perturb = F.mse_loss(delta_f_flat, torch.zeros_like(delta_f_flat))
                         temp_target_stats = {
                              'dist_params': [p.to(device, non_blocking=True) for p in target_stats['dist_params']],
                              'energy_ratios': [r.to(device, non_blocking=True) for r in target_stats['energy_ratios']]
                         }
                         loss_stat_dist, loss_stat_energy = calculate_stat_loss(
                              dct_blocks_poisoned, temp_target_stats, fixed_beta=fixed_beta_stats
                         )
                         loss_trigger = compute_trigger_loss(dct_blocks_poisoned, proxy_models, targets, cfg)
                         total_loss = (lambda_dist * loss_stat_dist + lambda_energy * loss_stat_energy +
                                       lambda_trig * loss_trigger + lambda_pert * loss_perturb)

                     # Accumulate validation losses locally
                     val_losses['total'] += total_loss.item()
                     val_losses['dist'] += loss_stat_dist.item() if torch.is_tensor(loss_stat_dist) else loss_stat_dist
                     val_losses['energy'] += loss_stat_energy.item() if torch.is_tensor(loss_stat_energy) else loss_stat_energy
                     val_losses['trigger'] += loss_trigger.item() if torch.is_tensor(loss_trigger) else loss_trigger
                     val_losses['perturb'] += loss_perturb.item() if torch.is_tensor(loss_perturb) else loss_perturb
                     val_steps += 1

            # --- Aggregate and Log Validation Losses ---
            barrier() # Ensure all ranks finish validation loop
            if val_steps > 0:
                avg_val_losses = {k: v / val_steps for k, v in val_losses.items()}
                # Reduce validation losses across GPUs
                val_loss_tensor = torch.tensor([avg_val_losses['total'], avg_val_losses['dist'], avg_val_losses['energy'],
                                                avg_val_losses['trigger'], avg_val_losses['perturb']]).to(device)
                reduced_val_losses = reduce_tensor(val_loss_tensor, world_size)

                if is_main_process(rank):
                    current_val_loss = reduced_val_losses[0].item()
                    print(f"  Avg Val Losses: Total={current_val_loss:.4f}, "
                          f"Dist={reduced_val_losses[1].item():.4f}, Energy={reduced_val_losses[2].item():.4f}, "
                          f"Trigger={reduced_val_losses[3].item():.4f}, Perturb={reduced_val_losses[4].item():.4f}")

                    # --- Save Checkpoint (only on main process) ---
                    is_best = current_val_loss < best_val_loss
                    best_val_loss = min(current_val_loss, best_val_loss)

                    save_dict = {
                        'epoch': epoch,
                        'model_state_dict': generator.module.state_dict(), # Save unwrapped model
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_loss': best_val_loss,
                        'scaler_state_dict': scaler.state_dict() if cfg.use_amp else None,
                        'config': vars(cfg) # Convert DictToObject back for saving if needed, or save cfg object directly
                    }
                    # Ensure checkpoint directory exists
                    ckpt_dir = os.path.dirname(cfg.generator.checkpoint_path)
                    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)

                    last_ckpt_path = cfg.generator.checkpoint_path.replace('.pt', '_last.pt')
                    torch.save(save_dict, last_ckpt_path)

                    if is_best:
                        best_ckpt_path = cfg.generator.checkpoint_path
                        torch.save(save_dict, best_ckpt_path)
                        print(f"  Best generator model (Val Loss: {best_val_loss:.4f}) saved to {best_ckpt_path}")
            else:
                 if is_main_process(rank): print("  Validation loader was empty or validation step failed.")

        barrier() # Synchronize all processes at the end of epoch/validation


    if is_main_process(rank): print("Generator training finished.")
    cleanup_ddp()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train NFSBA Generator with DDP')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    args = parser.parse_args()

    # --- Run Main Training Function ---
    try:
        main(local_rank, world_size, args.config)
    except Exception as e:
         print(f"Rank {local_rank}: An error occurred during training: {e}")
         import traceback
         traceback.print_exc()
    finally:
         # Ensure cleanup happens even if there's an error on some ranks
         if dist.is_initialized():
              cleanup_ddp()