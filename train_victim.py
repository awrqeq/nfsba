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

from utils.config import load_config
from utils.distributed import setup_ddp, cleanup_ddp, is_main_process, reduce_tensor, barrier
# --- 修改：导入新的 PoisonedDataset ---
from data.datasets import get_dataset, PoisonedDataset
from models.victim import load_victim_model
from models.generator import MLPGenerator, CNNGenerator
from attacks.nfsba import NFSBA_Attack
from constants import CIFAR10_MEAN, CIFAR10_STD  # 导入常量


# (假设 calculate_asr_ba_eval 在 evaluate.py 中，或者您可以将其移至 utils.metrics)
# from evaluate import calculate_asr_ba_eval

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
        if is_main_process(rank): print(f"Rank {rank}: DDP Setup failed: {e}")
        return
    device = torch.device(f"cuda:{rank}")
    set_seed(cfg.seed + rank)
    if is_main_process(rank): print("Victim Training Setup:")

    # --- Load Generator for Poisoning ---
    if is_main_process(rank): print("Loading generator for poisoning...")
    generator_model = None
    gen_ckpt_path_to_load = None
    try:
        generator_type = getattr(cfg.generator, 'type', 'MLP').upper()
        if is_main_process(rank): print(f"Loading generator type: {generator_type}")

        if generator_type == "MLP":
            if not hasattr(cfg.generator, 'hidden_dims'):
                raise AttributeError("Config missing 'generator.hidden_dims' required for MLPGenerator.")
            ac_dim = cfg.dct_block_size ** 2 - 1
            generator_model = MLPGenerator(ac_dim=ac_dim,
                                           condition_dim=cfg.condition_dim,
                                           hidden_dims=cfg.generator.hidden_dims,
                                           initial_scale=getattr(cfg.generator, 'initial_scale', 0.1),
                                           learnable_scale=getattr(cfg.generator, 'learnable_scale', False)
                                           ).to(device)
        elif generator_type == "CNN":
            generator_model = CNNGenerator(condition_dim=cfg.condition_dim,
                                           initial_scale=getattr(cfg.generator, 'initial_scale', 0.1),
                                           learnable_scale=getattr(cfg.generator, 'learnable_scale', False)
                                           ).to(device)
        else:
            raise ValueError(f"Unknown generator type in config: {generator_type}")

        # --- Checkpoint loading logic ---
        gen_ckpt_path_config = cfg.generator.checkpoint_path
        ckpt_dir = os.path.dirname(gen_ckpt_path_config)
        base_ckpt_name = os.path.splitext(os.path.basename(gen_ckpt_path_config))[0]
        # 优先加载 _best.pt
        best_gen_ckpt_path = os.path.join(ckpt_dir, f"{base_ckpt_name}_best.pt")
        last_gen_ckpt_path = os.path.join(ckpt_dir, f"{base_ckpt_name}_last.pt")

        if os.path.exists(best_gen_ckpt_path):
            gen_ckpt_path_to_load = best_gen_ckpt_path
        elif os.path.exists(last_gen_ckpt_path):
            gen_ckpt_path_to_load = last_gen_ckpt_path
            if is_main_process(rank): print(
                f"Warning: Best generator checkpoint not found. Using last generator checkpoint: {last_gen_ckpt_path}")
        else:
            if is_main_process(rank): print(
                f"Error: No generator checkpoint (_best or _last) found at paths derived from {cfg.generator.checkpoint_path}. Cannot poison data.")
            cleanup_ddp()
            return

        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint_gen = torch.load(gen_ckpt_path_to_load, map_location=map_location)
        state_dict_gen = checkpoint_gen['model_state_dict'] if 'model_state_dict' in checkpoint_gen else checkpoint_gen
        if "module." in list(state_dict_gen.keys())[0]:
            state_dict_gen = {k.replace("module.", ""): v for k, v in state_dict_gen.items()}
        generator_model.load_state_dict(state_dict_gen, strict=False)
        if is_main_process(rank): print(f"Generator weights loaded from {gen_ckpt_path_to_load}")

    except Exception as e:
        if is_main_process(rank): print(f"Error loading generator model: {e}")
        cleanup_ddp()
        return
    generator_model.eval()
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
    if cfg.condition_type == 'embedding':
        emb_path = f'./checkpoints/cond_emb_{cfg.dataset}_{cfg.condition_dim}d.pt'
        if os.path.exists(emb_path):
            attacker.condition_embeddings.load_state_dict(torch.load(emb_path, map_location=device))
        else:
            if is_main_process(rank): print(f"Warning: Condition embeddings not found at {emb_path}")
    barrier()
    if is_main_process(rank): print("Attacker initialized.")

    # --- Load Data (!!! 关键修改部分 !!!) ---
    if is_main_process(rank): print("Loading data...")
    try:
        # 加载原始的（已归一化）训练集和测试集
        train_dataset_clean = get_dataset(cfg.dataset, cfg.data_path, train=True, img_size=cfg.img_size)
        test_dataset_clean = get_dataset(cfg.dataset, cfg.data_path, train=False, img_size=cfg.img_size)

        num_train = len(train_dataset_clean)
        num_test = len(test_dataset_clean)

        # --- 1. 创建训练集毒化索引 (仅非目标类) ---
        target_label = cfg.target_label
        non_target_train_indices = [
            i for i, (_, label) in enumerate(train_dataset_clean) if label != target_label
        ]

        num_to_poison = int(np.floor(cfg.poison_rate * num_train))
        # 确保毒化数量不超过非目标类的总数
        if num_to_poison > len(non_target_train_indices):
            num_to_poison = len(non_target_train_indices)
            if is_main_process(rank): print(
                f"Warning: Poison rate {cfg.poison_rate} is too high. Poisoning all {num_to_poison} non-target samples.")

        g = torch.Generator()
        g.manual_seed(cfg.seed)  # 确保所有进程的随机选择一致
        perm = torch.randperm(len(non_target_train_indices), generator=g)
        selected_indices_in_subset = perm[:num_to_poison].tolist()

        # 将子集索引映射回原始数据集索引
        poison_indices_set = set([non_target_train_indices[i] for i in selected_indices_in_subset])

        if is_main_process(rank):
            print(
                f"Poisoning {len(poison_indices_set)} samples (target: {num_to_poison}) from {len(non_target_train_indices)} non-target samples.")

        # 创建用于训练的 PoisonedDataset (模式 'train' 会修改标签)
        train_dataset_poisoned = PoisonedDataset(train_dataset_clean, attacker, poison_indices_set, target_label,
                                                 mode='train')

        # --- 2. 创建 BA 验证集 (全干净) ---
        # 使用空的毒化索引集和 'test_clean' 模式
        test_dataset_clean_for_ba = PoisonedDataset(test_dataset_clean, attacker, set(), target_label,
                                                    mode='test_clean')

        # --- 3. 创建 ASR 验证集 (仅非目标类, 且全部毒化) ---
        non_target_test_indices = [
            i for i, (_, label) in enumerate(test_dataset_clean) if label != target_label
        ]
        # 创建一个只包含非目标类样本的子集
        test_subset_non_target = Subset(test_dataset_clean, non_target_test_indices)
        # 包装这个子集，并毒化 *子集* 中的所有样本
        asr_poison_indices_set = set(range(len(test_subset_non_target)))
        test_dataset_poison_for_asr = PoisonedDataset(test_subset_non_target, attacker, asr_poison_indices_set,
                                                      target_label, mode='test_attack')

        if is_main_process(rank):
            print(f"BA test set size: {len(test_dataset_clean_for_ba)}")
            print(f"ASR test set size (non-target): {len(test_dataset_poison_for_asr)}")

    except Exception as e:
        if is_main_process(rank): print(f"Error loading data or creating poison sets: {e}")
        import traceback
        traceback.print_exc()
        cleanup_ddp()
        return
    # ---------------------------------------------------

    # DDP Samplers and Loaders (使用新的数据集实例)
    train_sampler = DistributedSampler(train_dataset_poisoned, num_replicas=world_size, rank=rank, shuffle=True,
                                       seed=cfg.seed)
    test_ba_sampler = DistributedSampler(test_dataset_clean_for_ba, num_replicas=world_size, rank=rank, shuffle=False)
    test_asr_sampler = DistributedSampler(test_dataset_poison_for_asr, num_replicas=world_size, rank=rank,
                                          shuffle=False)

    train_loader = DataLoader(train_dataset_poisoned, batch_size=cfg.batch_size, sampler=train_sampler,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_batch_size = cfg.batch_size * 2
    test_ba_loader = DataLoader(test_dataset_clean_for_ba, batch_size=val_batch_size, sampler=test_ba_sampler,
                                num_workers=cfg.num_workers, pin_memory=True)
    test_asr_loader = DataLoader(test_dataset_poison_for_asr, batch_size=val_batch_size, sampler=test_asr_sampler,
                                 num_workers=cfg.num_workers, pin_memory=True)

    if is_main_process(rank): print("DataLoaders created.")
    barrier()

    # --- Load Victim Model ---
    if is_main_process(rank): print("Loading victim model...")
    victim_model_base = load_victim_model(cfg.victim_model.name, cfg.num_classes, cfg.victim_model.pretrained).to(
        device)
    victim_model = DDP(victim_model_base, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    barrier()

    # Optimizer, Scaler, Criterion, Scheduler
    optimizer = optim.SGD(victim_model.parameters(), lr=cfg.victim_model.lr,
                          momentum=cfg.victim_model.momentum, weight_decay=cfg.victim_model.weight_decay)
    scaler = GradScaler(enabled=cfg.use_amp)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.victim_model.epochs)

    # --- Resume from Checkpoint (Optional) ---
    start_epoch = 0
    best_ba = 0.0
    best_asr = 0.0
    victim_ckpt_path = cfg.victim_model.checkpoint_path
    vic_ckpt_dir = os.path.dirname(victim_ckpt_path)
    vic_base_name = os.path.splitext(os.path.basename(victim_ckpt_path))[0]
    last_vic_ckpt_path = os.path.join(vic_ckpt_dir, f"{vic_base_name}_last.pt")
    best_vic_ckpt_path = os.path.join(vic_ckpt_dir, f"{vic_base_name}_best.pt")

    if getattr(cfg, 'resume_victim', False) and os.path.exists(last_vic_ckpt_path):
        # ... (恢复逻辑保持不变) ...
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        if is_main_process(rank): print(f"Attempting to resume victim training from {last_vic_ckpt_path}")
        checkpoint = torch.load(last_vic_ckpt_path, map_location=map_location)
        try:
            missing, unexpected = victim_model.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if is_main_process(rank):
                print(f"Loaded victim model state from {last_vic_ckpt_path}")
                if missing: print(f"  Warning: Missing keys: {missing}")
                if unexpected: print(f"  Warning: Unexpected keys: {unexpected}")
        except Exception as e:
            if is_main_process(rank): print(f"Warning: Could not load victim model state: {e}")
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if is_main_process(rank): print("Loaded optimizer state.")
        except Exception as e:
            if is_main_process(rank): print(f"Warning: Could not load optimizer state: {e}. Re-initializing optimizer.")
            optimizer = optim.SGD(victim_model.parameters(), lr=cfg.victim_model.lr,
                                  momentum=cfg.victim_model.momentum, weight_decay=cfg.victim_model.weight_decay)
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if is_main_process(rank): print("Loaded scheduler state.")
        except Exception as e:
            if is_main_process(rank): print(f"Warning: Could not load scheduler state: {e}. Re-initializing scheduler.")
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.victim_model.epochs)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_ba = checkpoint.get('ba', 0.0)
        best_asr = checkpoint.get('asr', 0.0)
        try:
            if cfg.use_amp and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                if is_main_process(rank): print("Loaded scaler state.")
        except Exception as e:
            if is_main_process(rank): print(f"Warning: Could not load scaler state: {e}. Re-initializing scaler.")
            scaler = GradScaler(enabled=cfg.use_amp)
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
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        num_processed = 0

        pbar = tqdm(train_loader, desc=f"Victim Train Epoch {epoch + 1}/{cfg.victim_model.epochs} [Rank {rank}]",
                    disable=not is_main_process(rank))

        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.use_amp):
                outputs = victim_model(images)
                loss = criterion(outputs, labels)

            if torch.isnan(loss):
                if is_main_process(rank): print(
                    f"Warning: NaN loss detected at epoch {epoch + 1}, step {i}. Skipping batch.")
                continue

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
            epoch_loss_tensor = torch.tensor([epoch_loss], device=device)
            dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
            reduced_avg_loss = epoch_loss_tensor.item() / world_size
        else:
            reduced_avg_loss = 0.0

        scheduler.step()

        if is_main_process(rank):
            epoch_duration = time.time() - epoch_start_time
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch [{epoch + 1}/{cfg.victim_model.epochs}] Completed in {epoch_duration:.2f}s. LR: {current_lr:.6f}")
            print(f"  Avg Train Loss: {reduced_avg_loss:.4f}")

        # --- Validation (使用修改后的 ASR/BA 加载器) ---
        if (epoch + 1) % 5 == 0 or epoch == cfg.victim_model.epochs - 1:
            victim_model.eval()

            correct_clean_local = 0
            total_clean_local = 0
            val_loss_clean_local = 0.0
            if is_main_process(rank): print("  Calculating BA...")
            with torch.no_grad():
                val_ba_pbar = tqdm(test_ba_loader, desc=f"Victim BA Val [Rank {rank}]",
                                   disable=not is_main_process(rank))
                for images, labels in val_ba_pbar:
                    images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    with autocast(enabled=cfg.use_amp):
                        outputs = victim_model(images)
                        loss = criterion(outputs, labels)
                    val_loss_clean_local += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total_clean_local += labels.size(0)
                    correct_clean_local += (predicted == labels).sum().item()

            correct_poison_local = 0
            total_poison_local = 0
            val_loss_poison_local = 0.0
            # ASR 验证集只包含非目标类样本
            target_labels_template = torch.full((val_batch_size,), cfg.target_label, dtype=torch.long, device=device)
            if is_main_process(rank): print("  Calculating ASR (on non-target samples)...")
            with torch.no_grad():
                val_asr_pbar = tqdm(test_asr_loader, desc=f"Victim ASR Val [Rank {rank}]",
                                    disable=not is_main_process(rank))
                # 注意：test_asr_loader 现在只返回非目标类样本
                for images, original_labels_ignored in val_asr_pbar:
                    images = images.to(device, non_blocking=True)
                    current_batch_size = images.size(0)

                    if target_labels_template.size(0) != current_batch_size:
                        current_target_labels = target_labels_template[:current_batch_size]
                    else:
                        current_target_labels = target_labels_template

                    with autocast(enabled=cfg.use_amp):
                        outputs = victim_model(images)
                        loss = criterion(outputs, current_target_labels)
                    val_loss_poison_local += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total_poison_local += images.size(0)
                    # ASR 是指成功预测为目标标签
                    correct_poison_local += (predicted == cfg.target_label).sum().item()

            # --- Aggregate Validation Results Across GPUs ---
            val_results_local = torch.tensor(
                [correct_clean_local, total_clean_local, val_loss_clean_local,
                 correct_poison_local, total_poison_local, val_loss_poison_local],
                dtype=torch.float64, device=device
            )
            dist.all_reduce(val_results_local, op=dist.ReduceOp.SUM)

            reduced_correct_clean = val_results_local[0].item()
            reduced_total_clean = val_results_local[1].item()
            reduced_val_loss_clean = val_results_local[2].item()
            reduced_correct_poison = val_results_local[3].item()
            reduced_total_poison = val_results_local[4].item()
            reduced_val_loss_poison = val_results_local[5].item()

            final_val_ba = 100 * reduced_correct_clean / reduced_total_clean if reduced_total_clean > 0 else 0
            avg_val_loss_clean = reduced_val_loss_clean / reduced_total_clean if reduced_total_clean > 0 else 0

            # ASR 现在是基于正确的非目标类样本计算的
            final_val_asr = 100 * reduced_correct_poison / reduced_total_poison if reduced_total_poison > 0 else 0
            avg_val_loss_poison = reduced_val_loss_poison / reduced_total_poison if reduced_total_poison > 0 else 0

            avg_val_loss = (avg_val_loss_clean + avg_val_loss_poison) / 2.0 if (
                        reduced_total_clean > 0 and reduced_total_poison > 0) else avg_val_loss_clean + avg_val_loss_poison
            # -----------------------------------------------

            if is_main_process(rank):
                print(
                    f"Validation Epoch [{epoch + 1}]: BA: {final_val_ba:.2f}%, ASR: {final_val_asr:.2f}%, AvgValLoss: {avg_val_loss:.4f} (Clean: {avg_val_loss_clean:.4f}, Poison: {avg_val_loss_poison:.4f})")

                # --- Checkpoint Saving Logic ---
                # 保存标准：ASR 足够高 且 BA 也是历史最佳
                save_criterion_met = (final_val_ba > best_ba and final_val_asr > 80.0)

                if save_criterion_met:
                    best_ba = final_val_ba
                    best_asr = final_val_asr

                # 在最后一个 epoch 或满足标准时保存
                if save_criterion_met or epoch == cfg.victim_model.epochs - 1:
                    save_dict = {
                        'epoch': epoch,
                        'model_state_dict': victim_model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'ba': final_val_ba,
                        'asr': final_val_asr,
                        'avg_val_loss': avg_val_loss,
                        'scaler_state_dict': scaler.state_dict() if cfg.use_amp else None,
                        'config': vars(cfg)
                    }
                    os.makedirs(vic_ckpt_dir, exist_ok=True)

                    # 保存 last checkpoint
                    torch.save(save_dict, last_vic_ckpt_path)

                    if save_criterion_met:
                        torch.save(save_dict, best_vic_ckpt_path)
                        print(
                            f"Best victim model saved to {best_vic_ckpt_path} (BA: {best_ba:.2f}%, ASR: {best_asr:.2f}%)")
                    elif epoch == cfg.victim_model.epochs - 1:
                        print(f"Last epoch victim model saved to {last_vic_ckpt_path}")

            barrier()

    if is_main_process(rank): print("Victim training finished.")
    cleanup_ddp()


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:  # 只在主进程打印一次
            print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"Warning: Could not set multiprocessing start method to 'spawn': {e}")

    parser = argparse.ArgumentParser(description='Train NFSBA Victim Model with DDP')
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