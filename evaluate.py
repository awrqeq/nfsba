# /home/machao/pythonproject/nfsba/evaluate.py

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, Subset
import numpy as np
import random
import os
import argparse
from tqdm import tqdm
import time
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP  # 导入 DDP

# --- 项目导入 ---
from utils.config import load_config
from utils.distributed import setup_ddp, cleanup_ddp, is_main_process, reduce_tensor, barrier
# --- 修改：导入新的 PoisonedDataset ---
from data.datasets import get_dataset, PoisonedDataset
from models.victim import load_victim_model
from models.generator import MLPGenerator, CNNGenerator
from attacks.nfsba import NFSBA_Attack
from constants import CIFAR10_MEAN, CIFAR10_STD


# --- Set Seed Function ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(cfg, rank, world_size, is_ddp):
    """主评估函数，现在接收 DDP 参数"""
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if rank == 0: print(f"Evaluation using device: {device}")
    set_seed(cfg.seed + rank)

    # --- 1. 加载生成器 (用于创建 ASR 测试集) ---
    if rank == 0: print("Loading generator...")
    gen_ckpt_path_to_load = None
    generator_model = None

    try:
        gen_ckpt_path_config = cfg.generator.checkpoint_path
        ckpt_dir = os.path.dirname(gen_ckpt_path_config)
        base_ckpt_name = os.path.splitext(os.path.basename(gen_ckpt_path_config))[0]

        # 优先加载 _best.pt (因为它被 victim 训练使用)
        best_gen_ckpt_path = os.path.join(ckpt_dir, f"{base_ckpt_name}_best.pt")
        last_gen_ckpt_path = os.path.join(ckpt_dir, f"{base_ckpt_name}_last.pt")

        if os.path.exists(best_gen_ckpt_path):
            gen_ckpt_path_to_load = best_gen_ckpt_path
        elif os.path.exists(last_gen_ckpt_path):
            gen_ckpt_path_to_load = last_gen_ckpt_path
            if rank == 0: print(
                f"Warning: Best generator checkpoint not found. Using last generator checkpoint: {last_gen_ckpt_path}")
        else:
            if rank == 0: print(
                f"Error: No generator checkpoint (_best or _last) found at paths derived from {cfg.generator.checkpoint_path}. Cannot create ASR test set.")
            if is_ddp: cleanup_ddp()
            return

        # --- 根据 cfg.generator.type 实例化 ---
        generator_type = getattr(cfg.generator, 'type', 'MLP').upper()
        if rank == 0: print(f"Loading generator type: {generator_type}")

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

        # 加载权重
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank} if is_ddp else device
        checkpoint_gen = torch.load(gen_ckpt_path_to_load, map_location=map_location)
        state_dict_gen = checkpoint_gen['model_state_dict'] if isinstance(checkpoint_gen,
                                                                          dict) and 'model_state_dict' in checkpoint_gen else checkpoint_gen
        if "module." in list(state_dict_gen.keys())[0]:
            state_dict_gen = {k.replace("module.", ""): v for k, v in state_dict_gen.items()}
        generator_model.load_state_dict(state_dict_gen, strict=False)
        if rank == 0: print(f"Generator weights loaded from {gen_ckpt_path_to_load}")

    except Exception as e:
        if rank == 0: print(f"Error loading generator model: {e}")
        if is_ddp: cleanup_ddp()
        return
    generator_model.eval()

    # --- 2. 初始化攻击器 ---
    if rank == 0: print("Initializing attacker...")
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
            if rank == 0: print(f"Warning: Condition embeddings not found at {emb_path}")

    # --- 3. 加载训练好的受害者模型 ---
    if rank == 0: print("Loading victim model...")
    victim_model = load_victim_model(cfg.victim_model.name, cfg.num_classes, pretrained=False).to(device)

    vic_ckpt_path_config = cfg.victim_model.checkpoint_path
    vic_ckpt_dir = os.path.dirname(vic_ckpt_path_config)
    vic_base_name = os.path.splitext(os.path.basename(vic_ckpt_path_config))[0]

    best_vic_ckpt_path = os.path.join(vic_ckpt_dir, f"{vic_base_name}_best.pt")
    last_vic_ckpt_path = os.path.join(vic_ckpt_dir, f"{vic_base_name}_last.pt")

    victim_ckpt_path_to_load = None
    if os.path.exists(best_vic_ckpt_path):  # 优先加载 best
        victim_ckpt_path_to_load = best_vic_ckpt_path
    elif os.path.exists(last_vic_ckpt_path):  # 其次加载 last
        victim_ckpt_path_to_load = last_vic_ckpt_path
    elif os.path.exists(vic_ckpt_path_config):  # 最后尝试配置文件路径
        victim_ckpt_path_to_load = vic_ckpt_path_config
    else:
        if rank == 0: print(
            f"Error: No trained victim checkpoint found at paths derived from {vic_ckpt_path_config}. Did STEP 2 run successfully?")
        if is_ddp: cleanup_ddp()
        return

    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank} if is_ddp else device
    checkpoint_vic = torch.load(victim_ckpt_path_to_load, map_location=map_location)

    state_dict_vic = checkpoint_vic['model_state_dict'] if isinstance(checkpoint_vic,
                                                                      dict) and 'model_state_dict' in checkpoint_vic else checkpoint_vic
    if "module." in list(state_dict_vic.keys())[0]:
        state_dict_vic = {k.replace("module.", ""): v for k, v in state_dict_vic.items()}

    victim_model.load_state_dict(state_dict_vic, strict=False)
    if rank == 0: print(f"Victim model weights loaded from {victim_ckpt_path_to_load}")

    if is_ddp:
        victim_model = DDP(victim_model, device_ids=[rank], output_device=rank)

    victim_model.eval()

    # --- 4. 加载评估数据 (!!! 关键修改部分 !!!) ---
    if rank == 0: print("Loading evaluation data...")
    test_dataset_clean = get_dataset(cfg.dataset, cfg.data_path, train=False, img_size=cfg.img_size)

    # --- BA 评估集 (全干净) ---
    test_dataset_clean_for_ba = PoisonedDataset(test_dataset_clean, attacker, set(), cfg.target_label,
                                                mode='test_clean')

    # --- ASR 评估集 (仅非目标类, 且全部毒化) ---
    target_label = cfg.target_label
    non_target_test_indices = [
        i for i, (_, label) in enumerate(test_dataset_clean) if label != target_label
    ]
    test_subset_non_target = Subset(test_dataset_clean, non_target_test_indices)
    asr_poison_indices_set = set(range(len(test_subset_non_target)))  # 毒化子集中的所有
    test_dataset_poison_for_asr = PoisonedDataset(test_subset_non_target, attacker, asr_poison_indices_set,
                                                  target_label, mode='test_attack')

    if rank == 0:
        print(f"BA test set size: {len(test_dataset_clean_for_ba)}")
        print(f"ASR test set size (non-target): {len(test_dataset_poison_for_asr)}")

    # DDP Samplers 和 Loaders
    test_ba_sampler = DistributedSampler(test_dataset_clean_for_ba, num_replicas=world_size, rank=rank,
                                         shuffle=False) if is_ddp else None
    test_asr_sampler = DistributedSampler(test_dataset_poison_for_asr, num_replicas=world_size, rank=rank,
                                          shuffle=False) if is_ddp else None

    val_batch_size = cfg.batch_size * 2
    test_ba_loader = DataLoader(test_dataset_clean_for_ba, batch_size=val_batch_size,
                                sampler=test_ba_sampler, shuffle=False,  # 评估时不需要 shuffle
                                num_workers=cfg.num_workers, pin_memory=True)
    test_asr_loader = DataLoader(test_dataset_poison_for_asr, batch_size=val_batch_size,
                                 sampler=test_asr_sampler, shuffle=False,
                                 num_workers=cfg.num_workers, pin_memory=True)

    # --- 5. 执行评估 ---
    if rank == 0: print("Starting final evaluation...")

    correct_clean_local = 0
    total_clean_local = 0
    correct_poison_local = 0
    total_poison_local = 0

    with torch.no_grad():
        ba_pbar = tqdm(test_ba_loader, desc="Final BA Eval", disable=(rank != 0))
        for images, labels in ba_pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = victim_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_clean_local += labels.size(0)
            correct_clean_local += (predicted == labels).sum().item()

        asr_pbar = tqdm(test_asr_loader, desc="Final ASR Eval", disable=(rank != 0))
        target_labels_template = torch.full((val_batch_size,), cfg.target_label, dtype=torch.long, device=device)
        for images, _ in asr_pbar:  # 忽略 ASR loader 返回的原始标签
            images = images.to(device, non_blocking=True)
            current_batch_size = images.size(0)

            if target_labels_template.size(0) != current_batch_size:
                current_target_labels = target_labels_template[:current_batch_size]
            else:
                current_target_labels = target_labels_template

            outputs = victim_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_poison_local += images.size(0)
            # ASR 是指成功预测为目标标签
            correct_poison_local += (predicted == cfg.target_label).sum().item()

    # --- 聚合 DDP 结果 ---
    val_results_local = torch.tensor(
        [correct_clean_local, total_clean_local,
         correct_poison_local, total_poison_local],
        dtype=torch.float64, device=device
    )

    if is_ddp:
        dist.all_reduce(val_results_local, op=dist.ReduceOp.SUM)

    # --- 在 Rank 0 上计算并打印最终结果 ---
    if rank == 0:
        reduced_correct_clean = val_results_local[0].item()
        reduced_total_clean = val_results_local[1].item()
        reduced_correct_poison = val_results_local[2].item()
        reduced_total_poison = val_results_local[3].item()

        final_ba = 100 * reduced_correct_clean / reduced_total_clean if reduced_total_clean > 0 else 0
        final_asr = 100 * reduced_correct_poison / reduced_total_poison if reduced_total_poison > 0 else 0

        print("\n--- Final Evaluation Results ---")
        print(f"Loaded Victim Model: {victim_ckpt_path_to_load}")
        print(f"Loaded Generator:    {gen_ckpt_path_to_load}")
        print(f"Benign Accuracy (BA): {final_ba:.2f}% ({int(reduced_correct_clean)} / {int(reduced_total_clean)})")
        print(
            f"Attack Success Rate (ASR): {final_asr:.2f}% ({int(reduced_correct_poison)} / {int(reduced_total_poison)})")
        print("----------------------------------")

    if is_ddp:
        cleanup_ddp()


if __name__ == "__main__":
    try:
        mp.set_start_method('spawn', force=True)
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError as e:
        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            print(f"Warning: Could not set multiprocessing start method to 'spawn': {e}")

    parser = argparse.ArgumentParser(description='Evaluate NFSBA Victim Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')

    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        exit()
    except Exception as e:
        print(f"Error loading config file {args.config}: {e}")
        exit()

    # --- DDP 环境变量 (由 launch.sh 设置) ---
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    is_ddp = world_size > 1
    if is_ddp:
        try:
            setup_ddp(local_rank, world_size, cfg.master_port)
            if local_rank == 0: print(f"DDP Setup complete for {world_size} processes.")
        except Exception as e:
            print(f"Rank {local_rank}: DDP Setup failed: {e}")
            exit()
    else:
        print("Running in single-process mode (no DDP).")

    try:
        # 将 DDP 参数传递给 main
        main(cfg, local_rank, world_size, is_ddp)
    except Exception as e:
        print(f"Rank {local_rank}: An error occurred during evaluation: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if is_ddp and dist.is_initialized():
            cleanup_ddp()