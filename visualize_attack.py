# /home/machao/pythonproject/nfsba/visualize_attack.py

import torch
import torchvision.transforms as transforms # 确保导入 transforms
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import argparse
import os
import random
import numpy as np

from utils.config import load_config
from data.datasets import get_dataset # 用于加载数据
# --- 同时导入两种 Generator ---
from models.generator import MLPGenerator, CNNGenerator
from attacks.nfsba import NFSBA_Attack # 攻击器类定义
from constants import CIFAR10_MEAN, CIFAR10_STD # 从 constants 导入

def denormalize(tensor, mean, std):
    """反标准化图像张量 (使用传入的 mean/std)"""
    # 确保 mean 和 std 在正确的设备上且形状匹配
    mean = mean.to(tensor.device)
    std = std.to(tensor.device)
    tensor = tensor.clone() # 避免修改原始张量
    # 调整形状以匹配 (C, H, W)
    if mean.dim() == 4: mean = mean.squeeze(0)
    if std.dim() == 4: std = std.squeeze(0)
    # 确保张量也是 3D (C, H, W)
    if tensor.dim() == 4: tensor = tensor.squeeze(0)

    tensor.mul_(std).add_(mean)
    tensor = torch.clamp(tensor, 0, 1) # 确保值在 [0, 1] 范围内
    return tensor

def visualize_and_save(clean_img, poisoned_img, save_path, title_prefix="", amplify_factor=10):
    """
    生成并保存包含原始图像、毒化图像、残差和放大残差的组合图像。
    Args:
        clean_img (Tensor): 单个干净图像张量 (C, H, W)，范围 [0, 1]。
        poisoned_img (Tensor): 单个毒化图像张量 (C, H, W)，范围 [0, 1]。
        save_path (str): 保存图像的路径。
        title_prefix (str): 图像标题的前缀。
        amplify_factor (int): 残差放大倍数。
    """
    clean_img_cpu = clean_img.cpu()
    poisoned_img_cpu = poisoned_img.cpu()

    residual = torch.abs(poisoned_img_cpu - clean_img_cpu)
    residual_amplified = torch.clamp(residual * amplify_factor, 0, 1)

    # 确保 permute 前维度正确
    if clean_img_cpu.dim() != 3 or poisoned_img_cpu.dim() != 3:
         print(f"Warning: Unexpected image dimensions for visualization. Clean: {clean_img_cpu.shape}, Poisoned: {poisoned_img_cpu.shape}")
         return # 或者进行适当的处理

    clean_display = clean_img_cpu.permute(1, 2, 0).numpy()
    poisoned_display = poisoned_img_cpu.permute(1, 2, 0).numpy()
    residual_display = residual.permute(1, 2, 0).numpy()
    residual_amplified_display = residual_amplified.permute(1, 2, 0).numpy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f'{title_prefix}Attack Visualization', fontsize=16)

    axes[0].imshow(clean_display)
    axes[0].set_title("Clean Image")
    axes[0].axis('off')

    axes[1].imshow(poisoned_display)
    axes[1].set_title("Poisoned Image")
    axes[1].axis('off')

    # 使用灰色 colormap 显示残差可能更清晰
    # mean(axis=2) 用于将彩色残差转为灰度，如果已经是灰度则不需要
    res_gray = residual_display.mean(axis=2) if residual_display.shape[2] > 1 else residual_display.squeeze()
    res_max = res_gray.max() if res_gray.max() > 0 else 1 # 避免除零
    im_res = axes[2].imshow(res_gray, cmap='gray', vmin=0, vmax=res_max)
    axes[2].set_title("Residual (Difference)")
    axes[2].axis('off')

    res_amp_gray = residual_amplified_display.mean(axis=2) if residual_amplified_display.shape[2] > 1 else residual_amplified_display.squeeze()
    im_amp = axes[3].imshow(res_amp_gray, cmap='gray', vmin=0, vmax=1)
    axes[3].set_title(f"Residual x{amplify_factor}")
    axes[3].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Visualization saved to: {save_path}")
    except Exception as e:
        print(f"Error saving visualization: {e}")
    plt.close(fig)

def main(cfg, image_index, save_dir):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # --- 加载生成器 ---
    print("Loading generator...")
    gen_ckpt_path_to_load = None
    if args.generator_path: # 优先使用命令行参数
        if os.path.exists(args.generator_path):
            gen_ckpt_path_to_load = args.generator_path
            print(f"Using generator path from command line argument: {gen_ckpt_path_to_load}")
        else:
            print(f"Warning: Generator path specified via command line not found: {args.generator_path}. Will try paths from config.")

    if gen_ckpt_path_to_load is None: # 如果命令行参数无效或未提供
        gen_ckpt_path_config = cfg.generator.checkpoint_path
        ckpt_dir = os.path.dirname(gen_ckpt_path_config)
        base_ckpt_name = os.path.splitext(os.path.basename(gen_ckpt_path_config))[0]

        # 按优先级查找路径
        potential_paths = [
            gen_ckpt_path_config.replace('.pt', '_quant_finetune.pt'),
            os.path.join(ckpt_dir, f"{base_ckpt_name}_best.pt"),
            gen_ckpt_path_config,
            os.path.join(ckpt_dir, f"{base_ckpt_name}_last.pt"),
            gen_ckpt_path_config.replace('.pt', '_ltrig_debug_v2.pt')
        ]
        for path in potential_paths:
            if os.path.exists(path):
                gen_ckpt_path_to_load = path
                print(f"Found generator checkpoint at: {gen_ckpt_path_to_load}")
                break
        if gen_ckpt_path_to_load is None:
            print(f"Error: Generator checkpoint not found at command line path or paths derived from config: {cfg.generator.checkpoint_path}. Please check config or --generator_path.")
            return

    # --- 根据配置文件类型实例化 Generator ---
    # 使用 getattr 获取类型，如果未定义，默认为 'MLP'
    generator_type = getattr(cfg.generator, 'type', 'MLP').upper()
    print(f"Generator type specified in config: {generator_type}")

    try:
        if generator_type == "MLP":
            # ---!!! 检查 hidden_dims 是否存在 !!!---
            if not hasattr(cfg.generator, 'hidden_dims') or cfg.generator.hidden_dims is None:
                raise AttributeError("Config 'generator.type' is MLP, but 'generator.hidden_dims' is missing or null in the YAML file.")
            ac_dim = cfg.dct_block_size ** 2 - 1
            generator_model = MLPGenerator(ac_dim=ac_dim,
                                           condition_dim=cfg.condition_dim,
                                           hidden_dims=cfg.generator.hidden_dims, # <-- 只有当 type 是 MLP 时才会执行到这里
                                           initial_scale=getattr(cfg.generator, 'initial_scale', 0.1),
                                           learnable_scale=getattr(cfg.generator, 'learnable_scale', False)
                                          ).to(device)
            print("Instantiated MLPGenerator.")
        elif generator_type == "CNN":
            # CNNGenerator 不需要 hidden_dims
            generator_model = CNNGenerator(condition_dim=cfg.condition_dim,
                                           initial_scale=getattr(cfg.generator, 'initial_scale', 0.1),
                                           learnable_scale=getattr(cfg.generator, 'learnable_scale', False)
                                          ).to(device)
            print("Instantiated CNNGenerator.")
        else:
            raise ValueError(f"Unknown generator type in config: {generator_type}")

        # --- 加载权重 ---
        print(f"Attempting to load weights from: {gen_ckpt_path_to_load}")
        checkpoint_gen = torch.load(gen_ckpt_path_to_load, map_location='cpu')
        state_dict_gen = checkpoint_gen['model_state_dict'] if isinstance(checkpoint_gen, dict) and 'model_state_dict' in checkpoint_gen else checkpoint_gen
        if list(state_dict_gen.keys())[0].startswith('module.'):
            state_dict_gen = {k.replace("module.", ""): v for k, v in state_dict_gen.items()}

        missing_keys, unexpected_keys = generator_model.load_state_dict(state_dict_gen, strict=False)
        generator_model.to(device)
        print(f"Generator weights loaded successfully from {gen_ckpt_path_to_load}")
        if missing_keys: print(f"  Warning: Missing keys: {missing_keys}")
        if unexpected_keys: print(f"  Warning: Unexpected keys: {unexpected_keys}")

    except AttributeError as e: # 捕获属性错误 (例如 hidden_dims 不存在)
        print(f"Error initializing generator: {e}")
        print("Please ensure your config file specifies the correct 'generator.type' ('MLP' or 'CNN')")
        print("and includes 'generator.hidden_dims' with appropriate values (e.g., [256, 128]) if using MLP.")
        return
    except Exception as e:
        print(f"Error loading generator model or weights from {gen_ckpt_path_to_load}: {e}")
        return
    generator_model.eval()

    # --- 初始化攻击器 ---
    print("Initializing attacker...")
    try:
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
                print("Loaded condition embeddings for attacker.")
            else:
                print(f"Warning: Condition embeddings not found at {emb_path}. Using random embeddings.")
    except Exception as e:
        print(f"Error initializing attacker: {e}")
        return

    # --- 加载数据集 ---
    print("Loading dataset...")
    try:
        # 定义一个只包含 ToTensor 的 transform
        transform_to_tensor_only = transforms.Compose([
             transforms.Resize((cfg.img_size, cfg.img_size)), # 确保尺寸统一
             transforms.ToTensor() # 转换到 [0, 1] 范围
        ])
        # 假设 get_dataset 已被修改以接受 transform_override
        test_dataset_clean = get_dataset(cfg.dataset, cfg.data_path, train=False, img_size=cfg.img_size,
                                         transform_override=transform_to_tensor_only)
        if test_dataset_clean is None:
             raise ValueError("Failed to load dataset. Check data path and dataset name.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # 这里可以添加对 get_dataset 的修改建议
        return

    # 确保索引有效
    if not (0 <= image_index < len(test_dataset_clean)):
        print(f"Error: image_index {image_index} is out of bounds (0 to {len(test_dataset_clean) - 1}).")
        image_index = random.randint(0, len(test_dataset_clean) - 1)
        print(f"Using random index instead: {image_index}")

    # 获取指定的干净图像 (范围 [0, 1])
    try:
        clean_img_tensor_0_1, original_label = test_dataset_clean[image_index]
        clean_img_tensor_0_1 = clean_img_tensor_0_1.unsqueeze(0).to(device) # 添加 batch 维度并移动到设备
    except Exception as e:
        print(f"Error getting image at index {image_index}: {e}")
        return

    # --- 生成毒化图像 ---
    print(f"Generating poisoned image for index {image_index} (Original Label: {original_label}), Target Label: {cfg.target_label}...")
    target_tensor = torch.tensor([cfg.target_label], dtype=torch.long).to(device)

    try:
        with torch.no_grad():
            # generate_poison_batch 期望输入是 [0, 1] 范围
            poisoned_img_tensor_0_1 = attacker.generate_poison_batch(
                clean_img_tensor_0_1, # 传入 [0, 1] 范围的图像
                target_tensor
            )
            # 结果也是 [0, 1] 范围
    except Exception as e:
        print(f"Error during poison generation: {e}")
        return

    # --- 准备可视化 ---
    # 从 batch 中取出单个图像，并移动到 CPU
    clean_vis = clean_img_tensor_0_1.squeeze(0).cpu()
    poisoned_vis = poisoned_img_tensor_0_1.squeeze(0).cpu()

    # --- 保存可视化结果 ---
    save_filename = f"attack_vis_{cfg.dataset}_idx{image_index}_target{cfg.target_label}_{generator_type}.png" # 加入数据集和类型信息
    full_save_path = os.path.join(save_dir, save_filename)
    visualize_and_save(clean_vis, poisoned_vis, full_save_path,
                       title_prefix=f"{generator_type} (Index {image_index}, Target {cfg.target_label}) ")


# --- 提示修改 get_dataset 函数 (如果需要) ---
# 在 data/datasets.py 中:
# def get_dataset(dataset_name, data_path, train=True, img_size=32, download=True, transform_override=None):
#     """Loads the specified dataset."""
#     if transform_override:
#         transform_to_use = transform_override
#     else:
#         # Use your existing get_transforms logic here
#         transform_train, transform_test = get_transforms(dataset_name, img_size)
#         transform_to_use = transform_train if train else transform_test
#
#     # ... rest of your dataset loading logic using transform_to_use ...
#     # e.g., dataset = datasets.CIFAR10(..., transform=transform_to_use)
#     # return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize NFSBA Attack')
    parser.add_argument('--config', type=str, required=True, help='Path to the main config file (e.g., configs/nfsba_cifar10.yaml)')
    parser.add_argument('--image_index', type=int, default=0, help='Index of the test image to visualize (default: 0)')
    parser.add_argument('--save_dir', type=str, default='./visualizations', help='Directory to save the output image (default: ./visualizations)')
    parser.add_argument('--generator_path', type=str, default=None, help='(Optional) Path to the generator checkpoint, overrides config file')

    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        exit()
    except Exception as e:
        print(f"Error loading config file {args.config}: {e}")
        exit()

    # 命令行参数覆盖配置文件的 generator_path (保持不变)
    if args.generator_path and os.path.exists(args.generator_path):
        cfg.generator.checkpoint_path = args.generator_path
        print(f"Note: Overriding generator path with command line argument: {args.generator_path}")
    elif args.generator_path: # 如果提供了路径但文件不存在
        print(f"Warning: Generator path specified via command line not found: {args.generator_path}. Using path from config file.")

    main(cfg, args.image_index, args.save_dir)