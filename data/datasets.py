# /home/machao/pythonproject/nfsba/data/datasets.py

import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
import os
import numpy as np

# --- 导入共享常量 ---
from constants import CIFAR10_MEAN, CIFAR10_STD


def get_transforms(dataset_name, img_size):
    """获取标准的数据转换（归一化）"""
    if dataset_name == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.Resize((img_size, img_size)),  # 确保尺寸
            transforms.RandomCrop(img_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN.squeeze(), CIFAR10_STD.squeeze())
        ])
        transform_test = transforms.Compose([
            transforms.Resize((img_size, img_size)),  # 确保尺寸
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN.squeeze(), CIFAR10_STD.squeeze())
        ])
    else:
        # 默认为 CIFAR10，您可以为其他数据集添加
        raise ValueError(f"Transforms for dataset {dataset_name} not defined.")
    return transform_train, transform_test


def get_dataset(dataset_name, data_path, train=True, img_size=32, download=True, transform_override=None):
    """加载指定的数据集"""

    if transform_override:
        transform_to_use = transform_override
    else:
        transform_train, transform_test = get_transforms(dataset_name, img_size)
        transform_to_use = transform_train if train else transform_test

    if dataset_name == 'CIFAR10':
        dataset_class = datasets.CIFAR10
        dataset_path = os.path.join(data_path, 'CIFAR10')
        print(f"Loading CIFAR10 {'train' if train else 'test'} set...")
    else:
        raise ValueError(f"Dataset {dataset_name} not supported by get_dataset.")

    try:
        dataset = dataset_class(root=dataset_path, train=train, download=download, transform=transform_to_use)
        if train:
            print(f"CIFAR10 train loaded. Found {len(dataset)} samples.")
        else:
            print(f"CIFAR10 test loaded. Found {len(dataset)} samples.")
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        if download:
            print("Please check network connection or data path permissions.")
        return None

    return dataset


# --- 新的、更健壮的 PoisonedDataset 类 ---
class PoisonedDataset(Dataset):
    """
    一个包装器数据集，用于按需应用后门触发器。
    它负责处理反归一化、调用攻击器、重新归一化以及标签修改。
    """

    def __init__(self, base_dataset, attacker, poison_indices_set, target_label, mode='train'):
        """
        Args:
            base_dataset (Dataset): 原始的、已归一化的数据集。
            attacker (NFSBA_Attack): 已初始化的攻击器实例。
            poison_indices_set (set): 一个包含应被毒化样本索引的集合。
            target_label (int): 攻击的目标标签。
            mode (str):
                'train': 应用触发器并**修改标签**为 target_label。
                'test_clean': 不应用触发器，返回原始图像和标签 (用于 BA)。
                'test_attack': 应用触发器并**返回原始标签** (用于 ASR)。
        """
        self.base_dataset = base_dataset
        self.attacker = attacker
        self.poison_indices_set = poison_indices_set
        self.target_label = target_label
        self.mode = mode

        # 假设基础数据集使用了 CIFAR10 统计数据
        # 我们需要反向操作它们
        self.mean = CIFAR10_MEAN.squeeze(0)  # (3, 1, 1)
        self.std = CIFAR10_STD.squeeze(0)  # (3, 1, 1)

        if self.attacker is not None:
            self.device = self.attacker.device  # 从攻击器获取设备
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):
        # 1. 从基础数据集获取（已归一化）的图像和原始标签
        img_normalized, label = self.base_dataset[index]

        # 检查是否应该毒化
        should_poison = (index in self.poison_indices_set) and (self.mode in ['train', 'test_attack'])

        if should_poison:
            # 2. 反归一化: (img_norm * std) + mean -> [0, 1]
            # 确保 mean/std 与 img_normalized 在同一设备或类型上（通常是 CPU 上的 tensor）
            img_0_1 = img_normalized.clone() * self.std + self.mean
            img_0_1 = torch.clamp(img_0_1, 0.0, 1.0)

            # 3. 应用触发器 (攻击器在 GPU 上运行)
            # 添加 batch 维度 (1, C, H, W) 并移到攻击器设备
            img_0_1_batch = img_0_1.unsqueeze(0).to(self.device)
            target_tensor_batch = torch.tensor([self.target_label], dtype=torch.long).to(self.device)

            with torch.no_grad():
                poisoned_img_0_1_batch = self.attacker.generate_poison_batch(img_0_1_batch, target_tensor_batch)

            # 移除 batch 维度并移回 CPU
            poisoned_img_0_1 = poisoned_img_0_1_batch.squeeze(0).cpu()

            # 4. 重新归一化: (img_poisoned - mean) / std
            poisoned_img_normalized = (poisoned_img_0_1 - self.mean) / self.std

            # 5. 根据模式返回
            if self.mode == 'train':
                # 训练模式：返回毒化图像和*目标标签*
                return poisoned_img_normalized, self.target_label
            else:  # 'test_attack'
                # ASR 测试模式：返回毒化图像和*原始标签*
                return poisoned_img_normalized, label

        else:
            # 'test_clean' 模式或未被选中的索引
            # 返回原始图像和原始标签
            return img_normalized, label