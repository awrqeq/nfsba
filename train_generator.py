# /home/machao/pythonproject/nfsba/train_generator.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset, TensorDataset
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
from functools import partial # 用于 hook

# --- Project Imports ---
from utils.config import load_config
from utils.distributed import setup_ddp, cleanup_ddp, is_main_process, reduce_tensor, barrier
from utils.dct import block_dct, block_idct, get_ac_coeffs, set_ac_coeffs
from utils.stats import calculate_stat_loss, load_target_stats
from models.generator import MLPGenerator, CNNGenerator
from models.proxy import load_proxy_models
from attacks.nfsba import NFSBA_Attack
from data.datasets import get_dataset
# --- 修改: 移除 metrics 中的 calculate_lpips ---
# from utils.metrics import calculate_lpips
# --- 修改: 导入新的损失函数 ---
from utils.losses import (
    perturbation_loss,
    statistics_loss,
    feature_consistency_loss, # 新增
    frequency_smoothness_loss, # 新增
    calculate_generator_loss # 更新
)
from constants import CIFAR10_MEAN, CIFAR10_STD

# --- 定义反标准化函数 (保持不变) ---
def denormalize_batch(tensor, mean, std):
    mean = mean.to(tensor.device)
    std = std.to(tensor.device)
    tensor = tensor.clone()
    tensor.mul_(std).add_(mean)
    tensor = torch.clamp(tensor, 0, 1)
    return tensor

# --- Set Seed Function (保持不变) ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 新增: 提取中间层特征的辅助函数 ---
# 存储特征的全局变量或类属性
features_out = {}
def get_activation(name):
    def hook(model, input, output):
        # 对于avgpool层，输出可能是 (N, C, 1, 1)，需要展平
        features_out[name] = output.detach().view(output.size(0), -1)
    return hook

def get_intermediate_features(model, layer_name, input_tensor):
    """
    使用 forward hook 提取指定层的特征。
    注意: 这不是最高效的方式，但相对简单。对于特定模型可能有更优方法。
    """
    global features_out
    features_out = {} # 清空之前的特征

    # 尝试找到指定的层
    target_layer = None
    try:
        # 尝试直接按名称获取 (适用于 Sequential 或简单模型)
        module = model
        # 兼容 DDP/DataParallel 包装的模型
        if isinstance(model, (DDP, torch.nn.DataParallel)):
            module = model.module

        # 逐级查找 (处理嵌套模块)
        parts = layer_name.split('.')
        for part in parts:
             if hasattr(module, part):
                  module = getattr(module, part)
             elif isinstance(module, nn.Sequential) and part.isdigit(): # 处理 Sequential 中的数字索引
                  module = module[int(part)]
             else: # 回退到 _modules.get
                  module = module._modules.get(part)
                  if module is None:
                      raise AttributeError(f"Layer '{part}' not found in sequence '{layer_name}'")
        target_layer = module

    except AttributeError as e:
         print(f"Error finding layer '{layer_name}': {e}. Feature extraction might fail.")
         return None
    except Exception as e:
         print(f"Unexpected error finding layer '{layer_name}': {e}")
         return None

    if target_layer is None:
        print(f"Warning: Layer '{layer_name}' not found in the model.")
        return None

    handle = target_layer.register_forward_hook(get_activation(layer_name))
    with torch.no_grad(): # 确保特征提取不计算梯度
        _ = model(input_tensor) # 执行前向传播以触发 hook
    handle.remove() # 移除 hook

    return features_out.get(layer_name) # 返回提取到的特征

# --- 新增: 预计算目标类特征中心 ---
def precompute_target_feature_center(proxy_model, layer_name, target_label, dataset, num_samples, batch_size, device):
    """
    计算目标类别在指定代理模型层的平均特征向量。
    """
    print(f"Precomputing target feature center for label {target_label} using {num_samples} samples...")
    proxy_model.eval() # 确保模型在评估模式

    # 找到目标类的样本索引
    target_indices = [i for i, (_, label) in enumerate(dataset) if label == target_label]
    if len(target_indices) < num_samples:
        print(f"Warning: Only found {len(target_indices)} samples for target label {target_label}. Using all available.")
        num_samples = len(target_indices)
    elif len(target_indices) == 0:
         print(f"Error: No samples found for target label {target_label}. Cannot compute feature center.")
         return None

    # 随机选择 num_samples 个索引
    selected_indices = random.sample(target_indices, num_samples)
    target_subset = Subset(dataset, selected_indices)
    target_loader = DataLoader(target_subset, batch_size=batch_size, shuffle=False)

    all_features = []
    with torch.no_grad():
        for images, _ in tqdm(target_loader, desc="Extracting features"):
            images = images.to(device)
            features = get_intermediate_features(proxy_model, layer_name, images)
            if features is not None:
                all_features.append(features.cpu()) # 收集到 CPU 节省 GPU 内存

    if not all_features:
        print(f"Error: Failed to extract features from layer '{layer_name}'. Cannot compute center.")
        return None

    # 计算平均特征
    all_features_tensor = torch.cat(all_features, dim=0)
    target_center = torch.mean(all_features_tensor, dim=0, keepdim=True) # Shape (1, FeatureDim)
    print(f"Target feature center computed with shape: {target_center.shape}")
    return target_center.to(device) # 返回到目标设备

# --- main 函数 ---
def main(rank, world_size, cfg_path):
    # --- Basic Setup (保持不变) ---
    cfg = load_config(cfg_path)
    try:
        setup_ddp(rank, world_size, cfg.master_port)
    except Exception as e:
        print(f"Rank {rank}: DDP Setup failed: {e}")
        return
    device = torch.device(f"cuda:{rank}")
    set_seed(cfg.seed + rank)
    if is_main_process(rank): print(f"NFSBA Generator Training Started - Rank {rank}/{world_size} on GPU {torch.cuda.current_device()}")

    # --- Load Data (保持不变) ---
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

    # --- Load Models (保持不变, 但会加载代理模型用于特征提取) ---
    if is_main_process(rank): print("Loading models...")
    try:
        if cfg.generator.type == "MLP":
            # ... (MLP 初始化代码不变)
            if is_main_process(rank): print("Using MLPGenerator")
            ac_dim = cfg.dct_block_size ** 2 - 1
            generator_model = MLPGenerator(ac_dim=ac_dim,
                                           condition_dim=cfg.condition_dim,
                                           hidden_dims=cfg.generator.hidden_dims,
                                           initial_scale=getattr(cfg.generator, 'initial_scale', 0.1),
                                           learnable_scale=getattr(cfg.generator, 'learnable_scale', False)
                                          ).to(device)
        elif cfg.generator.type == "CNN":
            # ... (CNN 初始化代码不变)
            if is_main_process(rank): print("Using CNNGenerator")
            generator_model = CNNGenerator(condition_dim=cfg.condition_dim,
                                           initial_scale=getattr(cfg.generator, 'initial_scale', 0.1),
                                           learnable_scale=getattr(cfg.generator, 'learnable_scale', False)
                                          ).to(device)
        else:
            raise ValueError(f"Unknown generator type: {cfg.generator.type}")

        generator = DDP(generator_model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

        proxy_models = load_proxy_models(cfg.proxy_models.names, cfg.proxy_models.paths, device, cfg.num_classes)
        # --- 选择用于特征一致性损失的代理模型 ---
        proxy_for_features = None
        if hasattr(cfg, 'feature_consistency') and cfg.feature_consistency.proxy_idx < len(proxy_models):
            proxy_for_features = proxy_models[cfg.feature_consistency.proxy_idx]
            proxy_for_features.eval() # 确保在评估模式
            if is_main_process(rank):
                print(f"Using proxy model '{cfg.proxy_models.names[cfg.feature_consistency.proxy_idx]}' (index {cfg.feature_consistency.proxy_idx}) for feature consistency loss.")
        elif len(proxy_models) > 0:
             proxy_for_features = proxy_models[0] # 默认使用第一个
             proxy_for_features.eval()
             if is_main_process(rank):
                  print(f"Warning: feature_consistency.proxy_idx not specified or invalid. Defaulting to proxy model 0 ('{cfg.proxy_models.names[0]}').")
        else:
             if is_main_process(rank):
                  print("Warning: No proxy models loaded. Feature consistency loss will be disabled.")
        # ----------------------------------------

        if is_main_process(rank): print(f"Loaded {len(proxy_models)} proxy models.")

        # --- Attacker Util (保持不变) ---
        attacker_util = NFSBA_Attack(
            generator.module, # 传入未包装的模型
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

    # --- 预计算目标类特征中心 ---
    target_feature_center = None
    if proxy_for_features and hasattr(cfg, 'feature_consistency'):
        if is_main_process(rank): # 只在主进程计算，然后广播
            target_feature_center = precompute_target_feature_center(
                proxy_for_features,
                cfg.feature_consistency.feature_layer_name,
                cfg.target_label,
                full_train_dataset, # 使用完整数据集来找目标类样本
                cfg.feature_consistency.center_num_samples,
                val_batch_size, # 复用验证批大小
                device
            )
            # 将计算得到的中心广播给其他进程
            center_list = [target_feature_center] if target_feature_center is not None else [torch.zeros(1,1, device=device)] # 占位符
            dist.broadcast_object_list(center_list, src=0)
            if target_feature_center is None: # 如果主进程计算失败
                 print(f"Rank {rank}: Failed to receive target feature center from rank 0.")
        else:
             # 其他进程接收广播
             center_list = [torch.zeros(1,1, device=device)] # 准备接收的占位符
             dist.broadcast_object_list(center_list, src=0)
             target_feature_center = center_list[0]
             if target_feature_center.shape == (1,1): # 检查是否收到了占位符（即计算失败）
                  target_feature_center = None
                  print(f"Rank {rank}: Received None or placeholder for target feature center.")
             else:
                  target_feature_center = target_feature_center.to(device) # 确保在当前设备

        if target_feature_center is None and is_main_process(rank):
            print("Warning: Failed to compute target feature center. Feature consistency loss might be less effective or disabled if target center is required.")
    barrier()
    # ---------------------------

    # --- Load Stats (保持不变) ---
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

    # --- Optimizer, Scaler (保持不变) ---
    optimizer = optim.Adam(generator.parameters(), lr=cfg.generator.lr, betas=tuple(cfg.generator.betas))
    scaler = GradScaler(enabled=cfg.use_amp)
    if is_main_process(rank): print(f"Optimizer: Adam, LR: {cfg.generator.lr}, Betas: {cfg.generator.betas}")
    if is_main_process(rank): print(f"Automatic Mixed Precision (AMP): {'Enabled' if cfg.use_amp else 'Disabled'}")

    # --- Resume Logic (保持不变) ---
    start_epoch = 0
    best_val_loss = float('inf')
    ckpt_dir = os.path.dirname(cfg.generator.checkpoint_path)
    base_ckpt_name = os.path.splitext(os.path.basename(cfg.generator.checkpoint_path))[0]
    last_ckpt_path = os.path.join(ckpt_dir, f"{base_ckpt_name}_last.pt")
    best_ckpt_path = os.path.join(ckpt_dir, f"{base_ckpt_name}_best.pt")

    if os.path.exists(last_ckpt_path) and getattr(cfg.generator, 'resume_generator', False):
        try:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            checkpoint = torch.load(last_ckpt_path, map_location=map_location)
            # --- 兼容可能缺少 module. 前缀的情况 ---
            state_dict = checkpoint['model_state_dict']
            if not list(state_dict.keys())[0].startswith('module.') and world_size > 1:
                # 如果保存时未使用 DDP 但现在使用 DDP 加载，需要添加 'module.' 前缀
                state_dict = {'module.' + k: v for k, v in state_dict.items()}
            elif list(state_dict.keys())[0].startswith('module.') and world_size == 1:
                 # 如果保存时使用了 DDP 但现在单卡加载，需要移除 'module.' 前缀
                 state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

            # 根据当前是否 DDP 加载模型状态
            if world_size > 1:
                generator.load_state_dict(state_dict)
            else:
                generator.module.load_state_dict(state_dict) # 加载到 .module
            # ---------------------------------------
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            if cfg.use_amp and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            if is_main_process(rank): print(f"Resuming generator training from epoch {start_epoch}")
        except Exception as e:
            if is_main_process(rank): print(f"Warning: Could not load resume checkpoint '{last_ckpt_path}': {e}. Starting from scratch.")
            start_epoch = 0
            best_val_loss = float('inf')
    barrier()


    # --- 创建 DC 掩码 (保持不变) ---
    dc_mask = torch.ones(1, 1, cfg.dct_block_size, cfg.dct_block_size, device=device)
    dc_mask[..., 0, 0] = 0

    # --- Training Loop ---
    if is_main_process(rank): print("Starting generator training...")
    num_blocks_h = cfg.img_size // cfg.dct_block_size
    num_blocks_w = cfg.img_size // cfg.dct_block_size
    ac_dim = cfg.dct_block_size ** 2 - 1

    for epoch in range(start_epoch, cfg.generator.epochs):
        epoch_start_time = time.time()
        generator.train()
        train_sampler.set_epoch(epoch)

        # --- 更新: 修改 running_losses 字典 ---
        running_losses = {'total': 0.0, 'dist': 0.0, 'energy': 0.0, 'feat_cons': 0.0, 'smooth_freq': 0.0, 'perturb': 0.0}
        num_steps = 0

        # Determine loss weights for the current phase
        is_phase1 = epoch < int(cfg.generator.epochs * cfg.generator.phase1_epochs_ratio)
        phase_idx = 0 if is_phase1 else 1
        # --- 更新: 使用新的 lambda 权重 ---
        current_weights = {
            'dist': cfg.generator.lambda_stat_dist[phase_idx],
            'energy': cfg.generator.lambda_stat_energy[phase_idx],
            'feature_consistency': getattr(cfg.generator, 'lambda_feature_consistency', [0.0, 0.0])[phase_idx],
            'smooth_freq': getattr(cfg.generator, 'lambda_smooth_freq', [0.0, 0.0])[phase_idx],
            'perturb': cfg.generator.lambda_perturb[phase_idx]
            # 移除了 trigger 和 lpips
        }
        # ------------------------------------

        if is_main_process(rank) and (epoch == start_epoch or epoch == int(cfg.generator.epochs * cfg.generator.phase1_epochs_ratio)):
            weights_str = ', '.join([f"L{k}={v:.3f}" for k, v in current_weights.items()])
            print(f"Epoch {epoch+1} [Phase {phase_idx+1}] Weights: {weights_str}")

        pbar = tqdm(train_loader, desc=f"Gen Epoch {epoch + 1}/{cfg.generator.epochs}", disable=not is_main_process(rank))

        for i, (images_normalized, labels) in enumerate(pbar):
            images_normalized = images_normalized.to(device, non_blocking=True)
            targets = torch.full_like(labels, cfg.target_label).to(device, non_blocking=True)
            batch_size = images_normalized.shape[0]
            num_blocks_total_per_sample = images_normalized.shape[1] * num_blocks_h * num_blocks_w

            images_0_1 = denormalize_batch(images_normalized, CIFAR10_MEAN, CIFAR10_STD)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=cfg.use_amp):
                # --- Forward Pass & Perturbation Generation ---
                dct_blocks = block_dct(images_0_1, cfg.dct_block_size)
                N = dct_blocks.numel() // (cfg.dct_block_size ** 2)
                condition_vec = attacker_util.get_condition_vec(targets, num_blocks_total_per_sample)

                # --- 初始化扰动变量 ---
                scaled_perturbation = None # 用于 L_smooth_freq
                dct_blocks_poisoned = None # 计算其他损失

                if cfg.generator.type == "MLP":
                    ac_coeffs_flat = get_ac_coeffs(dct_blocks).view(N, ac_dim)
                    if ac_coeffs_flat.shape[0] != condition_vec.shape[0]:
                         raise RuntimeError(f"MLP: Dim 0 mismatch ac_coeffs ({ac_coeffs_flat.shape[0]}) vs condition_vec ({condition_vec.shape[0]})")
                    scaled_perturbation_flat = generator(ac_coeffs_flat.contiguous(), condition_vec.contiguous())
                    loss_perturb = perturbation_loss(scaled_perturbation_flat)
                    dct_blocks_poisoned = set_ac_coeffs(dct_blocks, scaled_perturbation_flat)
                    # --- 为 MLP 情况构造 scaled_perturbation (形状需匹配) ---
                    # 注意：这可能不是最精确的方式，但用于计算平滑度损失
                    perturb_shape = (N, 1, cfg.dct_block_size, cfg.dct_block_size)
                    scaled_perturbation = torch.zeros(perturb_shape, device=device)
                    scaled_perturbation_flat_view = scaled_perturbation.view(N, -1)
                    scaled_perturbation_flat_view[:, 1:] = scaled_perturbation_flat # 填充 AC 部分
                    # ----------------------------------------------------

                elif cfg.generator.type == "CNN":
                    dct_blocks_2d = dct_blocks.view(N, 1, cfg.dct_block_size, cfg.dct_block_size)
                    scaled_perturbation_2d = generator(dct_blocks_2d.contiguous(), condition_vec.contiguous())
                    scaled_perturbation = scaled_perturbation_2d # 直接使用CNN输出计算 L_smooth_freq
                    final_perturbation_2d = scaled_perturbation_2d * dc_mask
                    loss_perturb = F.mse_loss(final_perturbation_2d, torch.zeros_like(final_perturbation_2d))
                    final_perturbation_blocks = final_perturbation_2d.view(batch_size, images_normalized.shape[1], num_blocks_h, num_blocks_w, cfg.dct_block_size, cfg.dct_block_size)
                    dct_blocks_poisoned = dct_blocks + final_perturbation_blocks

                # --- 计算除特征一致性外的损失 ---
                temp_target_stats_dev = {
                    'dist_params': [p.to(device, non_blocking=True) for p in target_stats['dist_params']],
                    'energy_ratios': [r.to(device, non_blocking=True) for r in target_stats['energy_ratios']]
                }
                loss_stat_dist, loss_stat_energy = statistics_loss(
                    dct_blocks_poisoned, temp_target_stats_dev, fixed_beta=fixed_beta_stats
                )
                # --- 新增: 计算频域平滑度损失 ---
                loss_smooth_freq = frequency_smoothness_loss(
                    scaled_perturbation, # 使用生成器直接输出的扰动
                    block_size=cfg.dct_block_size,
                    # 可以考虑从 cfg 中读取 high_freq_threshold
                    high_freq_threshold=getattr(cfg.generator, 'smooth_freq_threshold', 30) # 默认 30
                )
                # ------------------------------

                # --- 计算特征一致性损失 ---
                loss_feat_cons = torch.tensor(0.0, device=device)
                features_poisoned = None # 初始化
                if proxy_for_features and target_feature_center is not None and current_weights['feature_consistency'] > 0:
                    # 将中毒的 DCT 块转换回图像空间并归一化
                    x_poisoned_0_1 = block_idct(dct_blocks_poisoned, cfg.dct_block_size)
                    x_poisoned_0_1 = torch.clamp(x_poisoned_0_1, 0.0, 1.0)
                    mean = CIFAR10_MEAN.to(device)
                    std = CIFAR10_STD.to(device)
                    x_poisoned_normalized = (x_poisoned_0_1 - mean) / std

                    # 提取特征
                    features_poisoned = get_intermediate_features(
                        proxy_for_features,
                        cfg.feature_consistency.feature_layer_name,
                        x_poisoned_normalized
                    )

                    # 计算损失
                    if features_poisoned is not None:
                        loss_feat_cons = feature_consistency_loss(
                            features_poisoned,
                            target_feature_center,
                            loss_type=getattr(cfg.feature_consistency, 'loss_type', 'l2') # 从配置读取损失类型
                        )
                    else:
                         if rank == 0: print(f"Warning: Failed to extract features for consistency loss in step {i}.")
                # ----------------------------

                # --- 移除 LPIPS 计算 ---

                # --- 更新: current_losses 字典 ---
                current_losses = {
                    'dist': loss_stat_dist,
                    'energy': loss_stat_energy,
                    'feature_consistency': loss_feat_cons,
                    'smooth_freq': loss_smooth_freq,
                    'perturb': loss_perturb
                    # 移除了 'trigger' 和 'lpips'
                }
                # ----------------------------------

                # Calculate total weighted loss using the updated function
                total_loss = calculate_generator_loss(current_losses, current_weights)

            # --- Backward Pass & Optimize (保持不变) ---
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # --- Log losses (更新) ---
            running_losses['total'] += total_loss.item()
            running_losses['dist'] += current_losses['dist'].item() if torch.is_tensor(current_losses['dist']) else current_losses['dist']
            running_losses['energy'] += current_losses['energy'].item() if torch.is_tensor(current_losses['energy']) else current_losses['energy']
            running_losses['feat_cons'] += current_losses['feature_consistency'].item() # 新增
            running_losses['smooth_freq'] += current_losses['smooth_freq'].item() # 新增
            running_losses['perturb'] += current_losses['perturb'].item() if torch.is_tensor(current_losses['perturb']) else current_losses['perturb']
            # 移除了 lpips
            num_steps += 1

            if is_main_process(rank) and i % 50 == 0:
                 # --- 更新: 修改进度条显示 ---
                 pbar.set_postfix({
                     'Loss': f"{total_loss.item():.4f}",
                     'Lfeat': f"{current_losses['feature_consistency'].item():.4f}", # 显示特征损失
                     'Lsmooth': f"{current_losses['smooth_freq'].item():.4f}" # 显示平滑度损失
                     # 移除了 Ltrig 和 LPIPS
                 })
                 # -------------------------
            # ---------------------------------

        # --- End of Epoch (日志部分更新) ---
        barrier()
        if num_steps > 0:
            avg_losses = {k: v / num_steps for k, v in running_losses.items()}
            loss_tensor = torch.tensor(list(avg_losses.values())).to(device)
            reduced_losses_tensor = reduce_tensor(loss_tensor, world_size)

            if is_main_process(rank):
                epoch_duration = time.time() - epoch_start_time
                reduced_losses_list = reduced_losses_tensor.cpu().numpy()
                loss_keys = list(avg_losses.keys()) # 使用更新后的 keys
                # --- 更新: 日志字符串格式 ---
                log_str = ', '.join([f"{key.replace('_',' ').capitalize()}={reduced_losses_list[idx]:.4f}" for idx, key in enumerate(loss_keys)])
                # --------------------------
                print(f"Epoch [{epoch+1}/{cfg.generator.epochs}] Completed in {epoch_duration:.2f}s.")
                print(f"  Avg Train Losses: {log_str}")
        else:
             if is_main_process(rank): print(f"Epoch [{epoch+1}/{cfg.generator.epochs}] had no training steps.")

        # --- Validation Step (修改以计算新损失) ---
        if (epoch + 1) % 5 == 0 or epoch == cfg.generator.epochs - 1:
            generator.eval()
            # --- 更新: val_losses 字典 ---
            val_losses = {'total': 0.0, 'dist': 0.0, 'energy': 0.0, 'feat_cons': 0.0, 'smooth_freq': 0.0, 'perturb': 0.0}
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
                        # --- Validation Forward Pass & Loss Calculation (类似训练) ---
                        dct_blocks = block_dct(images_0_1, cfg.dct_block_size)
                        condition_vec = attacker_util.get_condition_vec(targets, num_blocks_total_per_sample_val)

                        scaled_perturbation = None
                        dct_blocks_poisoned = None

                        if cfg.generator.type == "MLP":
                            ac_coeffs_flat = get_ac_coeffs(dct_blocks).view(N_val, ac_dim)
                            scaled_perturbation_flat = generator(ac_coeffs_flat.contiguous(), condition_vec.contiguous())
                            loss_perturb = perturbation_loss(scaled_perturbation_flat)
                            dct_blocks_poisoned = set_ac_coeffs(dct_blocks, scaled_perturbation_flat)
                            perturb_shape = (N_val, 1, cfg.dct_block_size, cfg.dct_block_size)
                            scaled_perturbation = torch.zeros(perturb_shape, device=device)
                            scaled_perturbation_flat_view = scaled_perturbation.view(N_val, -1)
                            scaled_perturbation_flat_view[:, 1:] = scaled_perturbation_flat

                        elif cfg.generator.type == "CNN":
                            dct_blocks_2d = dct_blocks.view(N_val, 1, cfg.dct_block_size, cfg.dct_block_size)
                            scaled_perturbation_2d = generator(dct_blocks_2d.contiguous(), condition_vec.contiguous())
                            scaled_perturbation = scaled_perturbation_2d
                            final_perturbation_2d = scaled_perturbation_2d * dc_mask
                            loss_perturb = F.mse_loss(final_perturbation_2d, torch.zeros_like(final_perturbation_2d))
                            final_perturbation_blocks = final_perturbation_2d.view(batch_size_val, images_normalized.shape[1], num_blocks_h, num_blocks_w, cfg.dct_block_size, cfg.dct_block_size)
                            dct_blocks_poisoned = dct_blocks + final_perturbation_blocks

                        # --- 计算验证损失 (不包括 LPIPS) ---
                        temp_target_stats_dev = {
                           'dist_params': [p.to(device, non_blocking=True) for p in target_stats['dist_params']],
                           'energy_ratios': [r.to(device, non_blocking=True) for r in target_stats['energy_ratios']]
                        }
                        loss_stat_dist, loss_stat_energy = statistics_loss(
                           dct_blocks_poisoned, temp_target_stats_dev, fixed_beta=fixed_beta_stats
                        )
                        loss_smooth_freq = frequency_smoothness_loss(
                           scaled_perturbation,
                           block_size=cfg.dct_block_size,
                           high_freq_threshold=getattr(cfg.generator, 'smooth_freq_threshold', 30)
                        )

                        loss_feat_cons = torch.tensor(0.0, device=device)
                        if proxy_for_features and target_feature_center is not None and current_weights['feature_consistency'] > 0:
                            x_poisoned_0_1 = block_idct(dct_blocks_poisoned, cfg.dct_block_size)
                            x_poisoned_0_1 = torch.clamp(x_poisoned_0_1, 0.0, 1.0)
                            mean = CIFAR10_MEAN.to(device)
                            std = CIFAR10_STD.to(device)
                            x_poisoned_normalized = (x_poisoned_0_1 - mean) / std
                            features_poisoned = get_intermediate_features(
                                proxy_for_features,
                                cfg.feature_consistency.feature_layer_name,
                                x_poisoned_normalized
                            )
                            if features_poisoned is not None:
                                loss_feat_cons = feature_consistency_loss(
                                    features_poisoned,
                                    target_feature_center,
                                    loss_type=getattr(cfg.feature_consistency, 'loss_type', 'l2')
                                )

                        # --- 更新: current_val_losses 字典 ---
                        current_val_losses = {
                            'dist': loss_stat_dist, 'energy': loss_stat_energy,
                            'feature_consistency': loss_feat_cons,
                            'smooth_freq': loss_smooth_freq,
                            'perturb': loss_perturb
                            # 移除了 'trigger' 和 'lpips'
                        }
                        # --------------------------------------
                        total_loss = calculate_generator_loss(current_val_losses, current_weights)

                    # Accumulate validation losses (更新)
                    val_losses['total'] += total_loss.item()
                    val_losses['dist'] += current_val_losses['dist'].item() if torch.is_tensor(current_val_losses['dist']) else current_val_losses['dist']
                    val_losses['energy'] += current_val_losses['energy'].item() if torch.is_tensor(current_val_losses['energy']) else current_val_losses['energy']
                    val_losses['feat_cons'] += current_val_losses['feature_consistency'].item()
                    val_losses['smooth_freq'] += current_val_losses['smooth_freq'].item()
                    val_losses['perturb'] += current_val_losses['perturb'].item() if torch.is_tensor(current_val_losses['perturb']) else current_val_losses['perturb']
                    # 移除了 lpips
                    val_steps += 1
                # ---------------------------------------------

            # --- Aggregate and Log Validation Losses (日志部分更新) ---
            barrier()
            if val_steps > 0:
                avg_val_losses = {k: v / val_steps for k, v in val_losses.items()}
                val_loss_tensor = torch.tensor(list(avg_val_losses.values())).to(device)
                reduced_val_losses_tensor = reduce_tensor(val_loss_tensor, world_size)

                if is_main_process(rank):
                    reduced_val_losses_list = reduced_val_losses_tensor.cpu().numpy()
                    loss_keys_val = list(avg_val_losses.keys()) # 使用更新后的 keys
                    # --- 更新: 日志字符串格式 ---
                    log_str_val = ', '.join([f"{key.replace('_',' ').capitalize()}={reduced_val_losses_list[idx]:.4f}" for idx, key in enumerate(loss_keys_val)])
                    # --------------------------
                    current_val_loss = reduced_val_losses_list[0] # Total loss

                    print(f"  Avg Val Losses: {log_str_val}")

                    is_best = current_val_loss < best_val_loss
                    if is_best:
                        best_val_loss = current_val_loss
                        print(f"  New best validation loss: {best_val_loss:.4f}")

                    # --- Save Checkpoint (保持不变) ---
                    save_dict = {
                        'epoch': epoch,
                        'model_state_dict': generator.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_val_loss': best_val_loss,
                        'scaler_state_dict': scaler.state_dict() if cfg.use_amp else None,
                        'config': vars(cfg)
                    }
                    os.makedirs(ckpt_dir, exist_ok=True)
                    torch.save(save_dict, last_ckpt_path)
                    if is_best:
                        torch.save(save_dict, best_ckpt_path)
                        print(f"  Best generator model saved to {best_ckpt_path}")
            else:
                if is_main_process(rank): print("  Validation loader was empty or validation step failed.")

        barrier()

    if is_main_process(rank): print("Generator training finished.")
    cleanup_ddp()

# --- __main__ 部分保持不变 ---
if __name__ == "__main__":
    try:
        # --- 尝试设置 spawn ---
        current_start_method = mp.get_start_method(allow_none=True)
        if current_start_method != 'spawn':
             mp.set_start_method('spawn', force=True)
             if is_main_process(int(os.environ.get("LOCAL_RANK", 0))): # 仅主进程打印
                  print(f"Multiprocessing start method set to 'spawn' (was '{current_start_method}').")
        elif is_main_process(int(os.environ.get("LOCAL_RANK", 0))):
             print("Multiprocessing start method already set to 'spawn'.")
        # --------------------
    except RuntimeError as e:
         # 在某些环境 (如特定版本的 Jupyter) 可能无法强制设置
         if is_main_process(int(os.environ.get("LOCAL_RANK", 0))):
              print(f"Warning: Could not set multiprocessing start method to 'spawn': {e}. Using default ('{mp.get_start_method()}').")
    except ValueError as e: # 例如，如果上下文已设置且 force=False
         if is_main_process(int(os.environ.get("LOCAL_RANK", 0))):
              print(f"Warning: Multiprocessing context already set. Using default ('{mp.get_start_method()}'). Error: {e}")


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