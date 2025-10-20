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

from evaluate import calculate_asr_ba_eval
from utils.config import load_config
from utils.distributed import setup_ddp, cleanup_ddp, is_main_process, reduce_tensor, barrier
from data.datasets import get_dataset, PoisonedDataset
from models.victim import load_victim_model
from models.generator import MLPGenerator  # Needed to load for poisoning
from attacks.nfsba import NFSBA_Attack  # Needed for poisoning


# from utils.metrics import calculate_asr_ba # Or use the internal calculation

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
    setup_ddp(rank, world_size, cfg.master_port)
    device = torch.device(f"cuda:{rank}")
    set_seed(cfg.seed + rank)
    if is_main_process(rank): print("Victim Training Setup:")

    # --- Load Generator for Poisoning ---
    if is_main_process(rank): print("Loading generator for poisoning...")
    generator_model = MLPGenerator(ac_dim=cfg.dct_block_size ** 2 - 1,
                                   condition_dim=cfg.condition_dim,
                                   hidden_dims=cfg.generator.hidden_dims).to(device)
    if os.path.exists(cfg.generator.checkpoint_path):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint_gen = torch.load(cfg.generator.checkpoint_path, map_location=map_location)
        state_dict_gen = checkpoint_gen['model_state_dict'] if 'model_state_dict' in checkpoint_gen else checkpoint_gen
        if "module." in list(state_dict_gen.keys())[0]:
            state_dict_gen = {k.replace("module.", ""): v for k, v in state_dict_gen.items()}
        generator_model.load_state_dict(state_dict_gen)
        if is_main_process(rank): print(f"Generator weights loaded from {cfg.generator.checkpoint_path}")
    else:
        # Cannot proceed without generator
        if is_main_process(rank): print(
            f"Error: Generator checkpoint not found at {cfg.generator.checkpoint_path}. Cannot poison data.")
        cleanup_ddp()
        return
    generator_model.eval()  # Generator is only for inference here
    barrier()

    # --- Initialize Attacker ---
    attacker = NFSBA_Attack(
        generator_model,
        cfg.dct_block_size,
        cfg.condition_dim,
        cfg.num_classes,
        cfg.condition_type,
        device
    )
    # Load embeddings if needed
    if cfg.condition_type == 'embedding':
        emb_path = f'./checkpoints/cond_emb_{cfg.dataset}_{cfg.condition_dim}d.pt'
        if os.path.exists(emb_path):
            attacker.condition_embeddings.load_state_dict(torch.load(emb_path, map_location=device))
        else:
            if is_main_process(rank): print(f"Warning: Condition embeddings not found at {emb_path}")
    barrier()
    if is_main_process(rank): print("Attacker initialized.")

    # --- Load Data ---
    if is_main_process(rank): print("Loading data...")
    train_dataset_clean = get_dataset(cfg.dataset, cfg.data_path, train=True, img_size=cfg.img_size)
    test_dataset_clean = get_dataset(cfg.dataset, cfg.data_path, train=False, img_size=cfg.img_size)

    # Create Poison Indices
    num_train = len(train_dataset_clean)
    indices = list(range(num_train))
    g = torch.Generator()
    g.manual_seed(cfg.seed)  # Ensure consistent poison indices across ranks
    indices = torch.randperm(num_train, generator=g).tolist()  # Use torch.randperm for better shuffling
    split = int(np.floor(cfg.poison_rate * num_train))
    poison_indices = indices[:split]
    if is_main_process(rank): print(
        f"Poisoning {len(poison_indices)} out of {num_train} training samples ({cfg.poison_rate * 100:.2f}%).")

    # Create Poisoned Training Dataset (on-the-fly poisoning)
    train_dataset_poisoned = PoisonedDataset(train_dataset_clean, attacker, poison_indices, cfg.target_label,
                                             mode='train')
    # Test sets using the SAME attacker instance
    test_dataset_clean_for_ba = PoisonedDataset(test_dataset_clean, 'eval', None, cfg.target_label,
                                                mode='test_clean')  # Bypass poison
    test_dataset_poison_for_asr = PoisonedDataset(test_dataset_clean, attacker, list(range(len(test_dataset_clean))),
                                                  cfg.target_label, mode='test_attack')  # Poison all test

    # DDP Samplers and Loaders
    train_sampler = DistributedSampler(train_dataset_poisoned, num_replicas=world_size, rank=rank, shuffle=True)
    test_ba_sampler = DistributedSampler(test_dataset_clean_for_ba, num_replicas=world_size, rank=rank, shuffle=False)
    test_asr_sampler = DistributedSampler(test_dataset_poison_for_asr, num_replicas=world_size, rank=rank,
                                          shuffle=False)

    train_loader = DataLoader(train_dataset_poisoned, batch_size=cfg.batch_size, sampler=train_sampler,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    test_ba_loader = DataLoader(test_dataset_clean_for_ba, batch_size=cfg.batch_size, sampler=test_ba_sampler,
                                num_workers=cfg.num_workers, pin_memory=True)
    test_asr_loader = DataLoader(test_dataset_poison_for_asr, batch_size=cfg.batch_size, sampler=test_asr_sampler,
                                 num_workers=cfg.num_workers, pin_memory=True)
    if is_main_process(rank): print("DataLoaders created.")
    barrier()

    # --- Load Victim Model ---
    if is_main_process(rank): print("Loading victim model...")
    victim_model = load_victim_model(cfg.victim_model.name, cfg.num_classes, cfg.victim_model.pretrained).to(device)
    victim_model = DDP(victim_model, device_ids=[rank], output_device=rank,
                       find_unused_parameters=False)  # Check find_unused if needed
    barrier()

    # Optimizer, Scaler, Criterion, Scheduler
    optimizer = optim.SGD(victim_model.parameters(), lr=cfg.victim_model.lr,
                          momentum=cfg.victim_model.momentum, weight_decay=cfg.victim_model.weight_decay)
    scaler = GradScaler(enabled=cfg.use_amp)
    criterion = torch.nn.CrossEntropyLoss()
    # Example scheduler: Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.victim_model.epochs)

    # --- Resume from Checkpoint (Optional) ---
    start_epoch = 0
    best_ba = 0.0
    best_asr = 0.0
    victim_ckpt_path = cfg.victim_model.checkpoint_path
    if os.path.exists(victim_ckpt_path) and cfg.get('resume_victim', False):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(victim_ckpt_path, map_location=map_location)
        victim_model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_ba = checkpoint.get('ba', 0.0)
        best_asr = checkpoint.get('asr', 0.0)
        # scaler.load_state_dict(checkpoint['scaler_state_dict']) # If saved
        if is_main_process(rank): print(f"Resuming victim training from epoch {start_epoch}")
        barrier()

    # --- Training Loop ---
    if is_main_process(rank): print("Starting victim fine-tuning...")
    for epoch in range(start_epoch, cfg.victim_model.epochs):
        epoch_start_time = time.time()
        victim_model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        num_processed = 0

        pbar = tqdm(train_loader, desc=f"Victim Train Epoch {epoch + 1}/{cfg.victim_model.epochs} [Rank {rank}]",
                    disable=not is_main_process(rank))

        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.use_amp):
                outputs = victim_model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            num_processed += images.size(0)

            if is_main_process(rank) and i % 50 == 0:
                pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        epoch_loss = running_loss / num_processed
        # Reduce loss across GPUs for logging
        epoch_loss_tensor = torch.tensor(epoch_loss).to(device)
        reduced_loss = reduce_tensor(epoch_loss_tensor, world_size)

        scheduler.step()  # Step the scheduler after optimizer step

        if is_main_process(rank):
            epoch_duration = time.time() - epoch_start_time
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch [{epoch + 1}/{cfg.victim_model.epochs}] Completed in {epoch_duration:.2f}s. LR: {current_lr:.6f}")
            print(f"  Avg Train Loss: {reduced_loss.item():.4f}")

        # --- Validation ---
        if (epoch + 1) % 5 == 0 or epoch == cfg.victim_model.epochs - 1:  # Validate periodically
            victim_model.eval()

            # Use the internal evaluation function
            val_ba, val_asr = calculate_asr_ba_eval(victim_model, test_ba_loader, test_asr_loader, cfg.target_label,
                                                    device)

            # Need to gather results from all ranks for accurate BA/ASR
            ba_tensor = torch.tensor(val_ba).to(device)
            asr_tensor = torch.tensor(val_asr).to(device)
            # This simple gather might be incorrect if batches aren't perfectly divisible
            # Correct DDP validation requires gathering correct/total counts before calculating rates
            # Reusing the logic from the end of the original train_victim main loop for proper reduction:

            # Calculate BA counts locally
            correct_clean_local = 0
            total_clean_local = 0
            with torch.no_grad():
                for images, labels in test_ba_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = victim_model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total_clean_local += labels.size(0)
                    correct_clean_local += (predicted == labels).sum().item()

            # Calculate ASR counts locally
            correct_poison_local = 0
            total_poison_local = 0
            with torch.no_grad():
                for images, _ in test_asr_loader:
                    images = images.to(device)
                    outputs = victim_model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total_poison_local += images.size(0)
                    correct_poison_local += (predicted == cfg.target_label).sum().item()

            # Reduce results across GPUs
            counts = torch.tensor(
                [correct_clean_local, total_clean_local, correct_poison_local, total_poison_local]).float().to(device)
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)

            reduced_correct_clean, reduced_total_clean, reduced_correct_poison, reduced_total_poison = counts.tolist()

            final_val_ba = 100 * reduced_correct_clean / reduced_total_clean if reduced_total_clean > 0 else 0
            final_val_asr = 100 * reduced_correct_poison / reduced_total_poison if reduced_total_poison > 0 else 0

            if is_main_process(rank):
                print(f"Validation Epoch [{epoch + 1}]: BA: {final_val_ba:.2f}%, ASR: {final_val_asr:.2f}%")
                # Save best model based on combined metric or BA
                # Example: save if BA improves and ASR is high enough
                save_criterion = final_val_ba > best_ba and final_val_asr > 80.0  # Example criterion
                if save_criterion:
                    best_ba = final_val_ba
                    best_asr = final_val_asr  # Store best ASR corresponding to best BA
                    # Save model checkpoint (unwrap DDP model)
                    save_dict = {
                        'epoch': epoch,
                        'model_state_dict': victim_model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'ba': final_val_ba,
                        'asr': final_val_asr,
                        'config': vars(cfg)
                    }
                    os.makedirs(os.path.dirname(victim_ckpt_path), exist_ok=True)
                    torch.save(save_dict, victim_ckpt_path)
                    print(f"Best victim model saved to {victim_ckpt_path} (BA: {best_ba:.2f}%, ASR: {best_asr:.2f}%)")

            barrier()  # Synchronize before next epoch

    if is_main_process(rank): print("Victim training finished.")
    cleanup_ddp()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train NFSBA Victim Model with DDP')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    # DDP arguments handled by launcher
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    args = parser.parse_args()
    main(local_rank, world_size, args.config)