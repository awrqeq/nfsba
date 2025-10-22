# /home/machao/pythonproject/nfsba/train_victim.py

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
import torch.multiprocessing as mp

# --- Project Imports ---
from evaluate import calculate_asr_ba_eval # Assuming this function exists and works
from utils.config import load_config
from utils.distributed import setup_ddp, cleanup_ddp, is_main_process, reduce_tensor, barrier
from data.datasets import get_dataset, PoisonedDataset
from models.victim import load_victim_model
# --- 修改：同时导入两种 Generator ---
from models.generator import MLPGenerator, CNNGenerator
from attacks.nfsba import NFSBA_Attack

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
    try: # Add try-except for DDP setup
        setup_ddp(rank, world_size, cfg.master_port)
    except Exception as e:
        print(f"Rank {rank}: DDP Setup failed: {e}")
        return # Exit if DDP fails
    device = torch.device(f"cuda:{rank}")
    set_seed(cfg.seed + rank)
    if is_main_process(rank): print("Victim Training Setup:")

    # --- Load Generator for Poisoning ---
    if is_main_process(rank): print("Loading generator for poisoning...")
    try: # Add try-except for model loading
        # --- 修改：根据 cfg.generator.type 加载正确的 Generator ---
        if cfg.generator.type == "MLP":
            if is_main_process(rank): print("Loading MLPGenerator...")
            # 检查 hidden_dims 是否存在
            if not hasattr(cfg.generator, 'hidden_dims'):
                 raise AttributeError("Config missing 'generator.hidden_dims' required for MLPGenerator.")
            ac_dim = cfg.dct_block_size ** 2 - 1
            generator_model = MLPGenerator(ac_dim=ac_dim,
                                           condition_dim=cfg.condition_dim,
                                           hidden_dims=cfg.generator.hidden_dims,
                                           initial_scale=getattr(cfg.generator, 'initial_scale', 0.1), # Load scale params if defined
                                           learnable_scale=getattr(cfg.generator, 'learnable_scale', False)
                                           ).to(device)
        elif cfg.generator.type == "CNN":
            if is_main_process(rank): print("Loading CNNGenerator...")
            generator_model = CNNGenerator(condition_dim=cfg.condition_dim,
                                           initial_scale=getattr(cfg.generator, 'initial_scale', 0.1), # Load scale params if defined
                                           learnable_scale=getattr(cfg.generator, 'learnable_scale', False)
                                          ).to(device)
        else:
            raise ValueError(f"Unknown generator type in config: {cfg.generator.type}")
        # -----------------------------------------------------------

        # --- Checkpoint loading logic (remains similar) ---
        gen_ckpt_path = cfg.generator.checkpoint_path # Default path from config
        # Prefer loading the best checkpoint if it exists
        ckpt_dir = os.path.dirname(gen_ckpt_path)
        base_ckpt_name = os.path.splitext(os.path.basename(gen_ckpt_path))[0]
        best_gen_ckpt_path = os.path.join(ckpt_dir, f"{base_ckpt_name}_best.pt")

        if os.path.exists(best_gen_ckpt_path):
             load_path = best_gen_ckpt_path
             if is_main_process(rank): print(f"Using best generator checkpoint: {load_path}")
        elif os.path.exists(gen_ckpt_path): # Fallback to default path in config
             load_path = gen_ckpt_path
             if is_main_process(rank): print(f"Using generator checkpoint from config: {load_path}")
        else:
             # Fallback to last checkpoint if others don't exist
             last_gen_ckpt_path = os.path.join(ckpt_dir, f"{base_ckpt_name}_last.pt")
             if os.path.exists(last_gen_ckpt_path):
                  load_path = last_gen_ckpt_path
                  if is_main_process(rank): print(f"Warning: Best/Config generator checkpoint not found. Using last generator checkpoint: {load_path}")
             else:
                  if is_main_process(rank): print(f"Error: Generator checkpoint not found at expected paths derived from {cfg.generator.checkpoint_path}. Cannot poison data.")
                  cleanup_ddp()
                  return

        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint_gen = torch.load(load_path, map_location=map_location)

        # Handle both dictionary and direct state_dict saving
        state_dict_gen = checkpoint_gen['model_state_dict'] if isinstance(checkpoint_gen, dict) and 'model_state_dict' in checkpoint_gen else checkpoint_gen

        # Handle DDP module prefix if present
        if "module." in list(state_dict_gen.keys())[0]:
            state_dict_gen = {k.replace("module.", ""): v for k, v in state_dict_gen.items()}

        # Load the state dict (allow missing keys if generator structure changed slightly, but warn)
        missing_keys, unexpected_keys = generator_model.load_state_dict(state_dict_gen, strict=False)
        if is_main_process(rank):
            print(f"Generator weights loaded from {load_path}")
            if missing_keys: print(f"  Warning: Missing keys in generator state_dict: {missing_keys}")
            if unexpected_keys: print(f"  Warning: Unexpected keys in generator state_dict: {unexpected_keys}")
        # -----------------------------------------------

        generator_model.eval()  # Generator is only for inference here
    except Exception as e: # Catch broader exceptions during loading
        if is_main_process(rank): print(f"Error loading generator model: {e}")
        cleanup_ddp()
        return
    barrier()

    # --- Initialize Attacker ---
    try: # Add try-except for attacker init
        attacker = NFSBA_Attack(
            generator_model, # Pass the loaded model
            cfg.dct_block_size,
            cfg.condition_dim,
            cfg.num_classes,
            cfg.condition_type,
            device
        )
        # Load embeddings if needed (should already be on device)
        if cfg.condition_type == 'embedding':
            emb_path = f'./checkpoints/cond_emb_{cfg.dataset}_{cfg.condition_dim}d.pt'
            if os.path.exists(emb_path):
                # Ensure embeddings are loaded onto the correct device
                attacker.condition_embeddings.load_state_dict(torch.load(emb_path, map_location=device))
            else:
                if is_main_process(rank): print(f"Warning: Condition embeddings not found at {emb_path}. Attacker might fail if embeddings are crucial.")
        barrier()
        if is_main_process(rank): print("Attacker initialized.")
    except Exception as e:
        if is_main_process(rank): print(f"Error initializing attacker: {e}")
        cleanup_ddp()
        return

    # --- Load Data ---
    if is_main_process(rank): print("Loading data...")
    try: # Add try-except for data loading
        train_dataset_clean = get_dataset(cfg.dataset, cfg.data_path, train=True, img_size=cfg.img_size)
        test_dataset_clean = get_dataset(cfg.dataset, cfg.data_path, train=False, img_size=cfg.img_size)

        # Create Poison Indices - Use torch generator for consistency
        num_train = len(train_dataset_clean)
        indices = list(range(num_train))
        g = torch.Generator()
        g.manual_seed(cfg.seed) # Ensure consistent poison indices across ranks
        indices = torch.randperm(num_train, generator=g).tolist()
        split = int(np.floor(cfg.poison_rate * num_train))
        poison_indices = set(indices[:split]) # Use a set for faster checking in PoisonedDataset
        if is_main_process(rank): print(f"Poisoning {len(poison_indices)} out of {num_train} training samples ({cfg.poison_rate * 100:.2f}%).")

        # Create Poisoned Training Dataset (on-the-fly poisoning)
        train_dataset_poisoned = PoisonedDataset(train_dataset_clean, attacker, poison_indices, cfg.target_label, mode='train')
        # Test sets using the SAME attacker instance
        test_dataset_clean_for_ba = PoisonedDataset(test_dataset_clean, attacker, set(), cfg.target_label, mode='test_clean') # Empty poison set
        # Poison all test samples for ASR evaluation
        test_poison_indices_asr = set(range(len(test_dataset_clean)))
        test_dataset_poison_for_asr = PoisonedDataset(test_dataset_clean, attacker, test_poison_indices_asr, cfg.target_label, mode='test_attack')

        # DDP Samplers and Loaders
        train_sampler = DistributedSampler(train_dataset_poisoned, num_replicas=world_size, rank=rank, shuffle=True, seed=cfg.seed) # Add seed here too
        test_ba_sampler = DistributedSampler(test_dataset_clean_for_ba, num_replicas=world_size, rank=rank, shuffle=False)
        test_asr_sampler = DistributedSampler(test_dataset_poison_for_asr, num_replicas=world_size, rank=rank, shuffle=False)

        train_loader = DataLoader(train_dataset_poisoned, batch_size=cfg.batch_size, sampler=train_sampler, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
        # Use larger batch size for validation if desired, but keep consistent for loss calculation
        val_batch_size = cfg.batch_size * 2
        test_ba_loader = DataLoader(test_dataset_clean_for_ba, batch_size=val_batch_size, sampler=test_ba_sampler, num_workers=cfg.num_workers, pin_memory=True)
        test_asr_loader = DataLoader(test_dataset_poison_for_asr, batch_size=val_batch_size, sampler=test_asr_sampler, num_workers=cfg.num_workers, pin_memory=True)
        if is_main_process(rank): print("DataLoaders created.")
    except Exception as e:
        if is_main_process(rank): print(f"Error loading data: {e}")
        cleanup_ddp()
        return
    barrier()

    # --- Load Victim Model ---
    if is_main_process(rank): print("Loading victim model...")
    try: # Add try-except
        victim_model_base = load_victim_model(cfg.victim_model.name, cfg.num_classes, cfg.victim_model.pretrained).to(device)
        victim_model = DDP(victim_model_base, device_ids=[rank], output_device=rank, find_unused_parameters=False) # find_unused=True might be needed if parts aren't used
    except Exception as e:
        if is_main_process(rank): print(f"Error loading victim model: {e}")
        cleanup_ddp()
        return
    barrier()

    # --- Optimizer, Scaler, Criterion, Scheduler ---
    try: # Add try-except
        optimizer = optim.SGD(victim_model.parameters(), lr=cfg.victim_model.lr,
                              momentum=cfg.victim_model.momentum, weight_decay=cfg.victim_model.weight_decay)
        scaler = GradScaler(enabled=cfg.use_amp)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.victim_model.epochs)
    except Exception as e:
        if is_main_process(rank): print(f"Error setting up optimizer/scaler/criterion/scheduler: {e}")
        cleanup_ddp()
        return

    # --- Resume from Checkpoint (Optional) ---
    start_epoch = 0
    best_ba = 0.0
    best_asr = 0.0
    victim_ckpt_path = cfg.victim_model.checkpoint_path # Path for saving/loading victim
    # Standardize checkpoint naming
    vic_ckpt_dir = os.path.dirname(victim_ckpt_path)
    vic_base_name = os.path.splitext(os.path.basename(victim_ckpt_path))[0]
    last_vic_ckpt_path = os.path.join(vic_ckpt_dir, f"{vic_base_name}_last.pt")
    best_vic_ckpt_path = os.path.join(vic_ckpt_dir, f"{vic_base_name}_best.pt") # Explicit best path

    if getattr(cfg, 'resume_victim', False) and os.path.exists(last_vic_ckpt_path):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        if is_main_process(rank): print(f"Attempting to resume victim training from {last_vic_ckpt_path}")
        checkpoint = torch.load(last_vic_ckpt_path, map_location=map_location)
        try:
            # Load model state, allow missing keys for flexibility
            missing, unexpected = victim_model.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if is_main_process(rank):
                print(f"Loaded victim model state from {last_vic_ckpt_path}")
                if missing: print(f"  Warning: Missing keys: {missing}")
                if unexpected: print(f"  Warning: Unexpected keys: {unexpected}")
        except Exception as e:
            if is_main_process(rank): print(f"Warning: Could not load victim model state: {e}")

        # Try loading optimizer state
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if is_main_process(rank): print("Loaded optimizer state.")
        except Exception as e:
            if is_main_process(rank): print(f"Warning: Could not load optimizer state: {e}. Re-initializing optimizer.")
            # Reinitialize if loading fails
            optimizer = optim.SGD(victim_model.parameters(), lr=cfg.victim_model.lr,
                                  momentum=cfg.victim_model.momentum, weight_decay=cfg.victim_model.weight_decay)

        # Try loading scheduler state
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if is_main_process(rank): print("Loaded scheduler state.")
        except Exception as e:
            if is_main_process(rank): print(f"Warning: Could not load scheduler state: {e}. Re-initializing scheduler.")
            # Reinitialize if loading fails
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.victim_model.epochs)

        # Load epoch, BA, ASR using .get with defaults
        start_epoch = checkpoint.get('epoch', 0) + 1 # Start from next epoch
        best_ba = checkpoint.get('ba', 0.0)
        best_asr = checkpoint.get('asr', 0.0)

        # Try loading scaler state
        try:
            if cfg.use_amp and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                if is_main_process(rank): print("Loaded scaler state.")
        except Exception as e:
            if is_main_process(rank): print(f"Warning: Could not load scaler state: {e}. Re-initializing scaler.")
            scaler = GradScaler(enabled=cfg.use_amp) # Reinitialize

        if is_main_process(rank): print(f"Resuming victim training from epoch {start_epoch}")
    else:
         if is_main_process(rank) and getattr(cfg, 'resume_victim', False):
              print(f"Resume requested but checkpoint not found at {last_vic_ckpt_path}. Starting from scratch.")
    barrier()

    # --- Training Loop ---
    if is_main_process(rank): print("Starting victim training...")
    for epoch in range(start_epoch, cfg.victim_model.epochs):
        epoch_start_time = time.time()
        victim_model.train()
        train_sampler.set_epoch(epoch) # Important for DDP shuffling
        running_loss = 0.0
        num_processed = 0

        pbar = tqdm(train_loader, desc=f"Victim Train Epoch {epoch + 1}/{cfg.victim_model.epochs} [Rank {rank}]",
                    disable=not is_main_process(rank))

        for i, batch_data in enumerate(pbar):
            # Handle potential variations in dataset return format
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                images, labels = batch_data
            else:
                 # Assume images is the only element if format is unexpected
                 images = batch_data
                 # Create dummy labels if needed by loss, adjust as necessary
                 labels = torch.zeros(images.size(0), dtype=torch.long)
                 if is_main_process(rank) and i == 0: # Print warning only once
                      print("Warning: Unexpected data format from DataLoader. Assuming first element is image.")

            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.use_amp):
                outputs = victim_model(images)
                loss = criterion(outputs, labels)

            # Check for NaN loss before backward pass
            if torch.isnan(loss):
                if is_main_process(rank): print(f"Warning: NaN loss detected at epoch {epoch+1}, step {i}. Skipping batch.")
                continue # Skip optimizer step if loss is NaN

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            num_processed += images.size(0)

            if is_main_process(rank) and i % 50 == 0:
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        # --- Aggregate and Log Epoch Loss ---
        if num_processed > 0:
            epoch_loss = running_loss / num_processed
            epoch_loss_tensor = torch.tensor([epoch_loss], device=device) # Put scalar in a tensor
            # Use all_reduce to get the sum, then divide by world_size for average
            dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
            reduced_avg_loss = epoch_loss_tensor.item() / world_size
        else:
            reduced_avg_loss = 0.0 # Handle case where no batches were processed

        scheduler.step() # Step the scheduler once per epoch

        if is_main_process(rank):
            epoch_duration = time.time() - epoch_start_time
            current_lr = scheduler.get_last_lr()[0] # Get current LR
            print(f"Epoch [{epoch+1}/{cfg.victim_model.epochs}] Completed in {epoch_duration:.2f}s. LR: {current_lr:.6f}")
            print(f"  Avg Train Loss: {reduced_avg_loss:.4f}")

        # --- Validation ---
        if (epoch + 1) % 5 == 0 or epoch == cfg.victim_model.epochs - 1:
            victim_model.eval()

            # Calculate BA (Benign Accuracy) and BA Loss
            correct_clean_local = 0
            total_clean_local = 0
            val_loss_clean_local = 0.0
            if is_main_process(rank): print("  Calculating BA...")
            with torch.no_grad():
                val_ba_pbar = tqdm(test_ba_loader, desc=f"Victim BA Val [Rank {rank}]", disable=not is_main_process(rank))
                for images, labels in val_ba_pbar:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    with autocast(enabled=cfg.use_amp):
                         outputs = victim_model(images)
                         loss = criterion(outputs, labels)
                    val_loss_clean_local += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total_clean_local += labels.size(0)
                    correct_clean_local += (predicted == labels).sum().item()

            # Calculate ASR (Attack Success Rate) and ASR Loss
            correct_poison_local = 0
            total_poison_local = 0
            val_loss_poison_local = 0.0
            # Pre-create target labels tensor for efficiency
            target_labels_template = torch.full((val_batch_size,), cfg.target_label, dtype=torch.long, device=device)
            if is_main_process(rank): print("  Calculating ASR...")
            with torch.no_grad():
                val_asr_pbar = tqdm(test_asr_loader, desc=f"Victim ASR Val [Rank {rank}]", disable=not is_main_process(rank))
                for images, _ in val_asr_pbar: # Ignore original labels for ASR
                    images = images.to(device, non_blocking=True)
                    current_batch_size = images.size(0)
                    # Adjust target labels tensor if last batch is smaller
                    if target_labels_template.size(0) != current_batch_size:
                         current_target_labels = target_labels_template[:current_batch_size]
                    else:
                         current_target_labels = target_labels_template

                    with autocast(enabled=cfg.use_amp):
                        outputs = victim_model(images)
                        loss = criterion(outputs, current_target_labels) # Use target labels for ASR loss
                    val_loss_poison_local += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total_poison_local += images.size(0)
                    correct_poison_local += (predicted == cfg.target_label).sum().item()

            # --- Aggregate Validation Results Across GPUs ---
            # Aggregate counts and losses using a tensor
            val_results_local = torch.tensor(
                [correct_clean_local, total_clean_local, val_loss_clean_local,
                 correct_poison_local, total_poison_local, val_loss_poison_local],
                dtype=torch.float64, device=device # Use float64 for sum precision
            )
            dist.all_reduce(val_results_local, op=dist.ReduceOp.SUM)

            reduced_correct_clean = val_results_local[0].item()
            reduced_total_clean = val_results_local[1].item()
            reduced_val_loss_clean = val_results_local[2].item()
            reduced_correct_poison = val_results_local[3].item()
            reduced_total_poison = val_results_local[4].item()
            reduced_val_loss_poison = val_results_local[5].item()
            # Extract aggregated results on rank 0
            if reduced_total_clean > 0:
                final_val_ba = 100 * val_results_local[0].item() / val_results_local[1].item()
                avg_val_loss_clean = val_results_local[2].item() / val_results_local[1].item()
            else:
                final_val_ba = 0.0
                avg_val_loss_clean = 0.0

            if reduced_total_poison > 0:
                final_val_asr = 100 * val_results_local[3].item() / val_results_local[4].item()
                avg_val_loss_poison = val_results_local[5].item() / val_results_local[4].item()
            else:
                final_val_asr = 0.0
                avg_val_loss_poison = 0.0

            # Calculate an overall validation loss (e.g., average)
            avg_val_loss = (avg_val_loss_clean + avg_val_loss_poison) / 2.0 if (reduced_total_clean > 0 and reduced_total_poison > 0) else avg_val_loss_clean + avg_val_loss_poison

            # -----------------------------------------------

            if is_main_process(rank):
                print(f"Validation Epoch [{epoch + 1}]: BA: {final_val_ba:.2f}%, ASR: {final_val_asr:.2f}%, AvgValLoss: {avg_val_loss:.4f} (Clean: {avg_val_loss_clean:.4f}, Poison: {avg_val_loss_poison:.4f})")

                # --- Checkpoint Saving Logic ---
                # Example: Save if BA improves while ASR is high, or always save last
                save_criterion_met = (final_val_ba > best_ba and final_val_asr > 80.0) # Adjust ASR threshold as needed

                if save_criterion_met or epoch == cfg.victim_model.epochs - 1:
                    if save_criterion_met:
                         best_ba = final_val_ba # Update best BA if criterion met
                         best_asr = final_val_asr

                    save_dict = {
                        'epoch': epoch,
                        'model_state_dict': victim_model.module.state_dict(), # Save unwrapped model
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'ba': final_val_ba,
                        'asr': final_val_asr,
                        'avg_val_loss': avg_val_loss,
                        'scaler_state_dict': scaler.state_dict() if cfg.use_amp else None,
                        'config': vars(cfg) # Store config
                    }
                    os.makedirs(vic_ckpt_dir, exist_ok=True) # Ensure directory exists

                    # Save last checkpoint
                    torch.save(save_dict, last_vic_ckpt_path)

                    # Save best checkpoint based on criterion
                    if save_criterion_met:
                        torch.save(save_dict, best_vic_ckpt_path)
                        print(f"Best victim model saved to {best_vic_ckpt_path} (BA: {best_ba:.2f}%, ASR: {best_asr:.2f}%)")
                    # Save last separately if it wasn't the best but is the final epoch
                    elif epoch == cfg.victim_model.epochs - 1:
                         print(f"Last epoch victim model saved to {last_vic_ckpt_path}")

            barrier() # Synchronize before next epoch or finishing

    if is_main_process(rank): print("Victim training finished.")
    cleanup_ddp() # Clean up DDP resources


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        print(f"Warning: Could not set multiprocessing start method to 'spawn': {e}")

    parser = argparse.ArgumentParser(description='Train NFSBA Victim Model with DDP')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    args = parser.parse_args()

    try:
        main(local_rank, world_size, args.config)
    except Exception as e:
         # Print error specific to the rank
         print(f"Rank {local_rank}: An error occurred during training: {e}")
         import traceback
         traceback.print_exc() # Print full traceback for debugging
    finally:
        # Ensure cleanup happens even if errors occur
        if dist.is_initialized():
            cleanup_ddp()