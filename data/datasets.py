import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np
import os
# --- 移除: from attacks.nfsba import NFSBA_Attack ---

def get_transforms(dataset_name, img_size):
    """Gets appropriate transforms for training and testing."""
    # --- CIFAR-10 Specific Transforms ---
    if dataset_name == 'CIFAR10':
        # ** Standard CIFAR-10 Augmentations and Normalization **
        transform_train = transforms.Compose([
            transforms.RandomCrop(img_size, padding=4), # Standard Aug
            transforms.RandomHorizontalFlip(), # Standard Aug
            transforms.ToTensor(), # To Tensor, scales to [0, 1]
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # ** MUST BE ENABLED **
        ])
        transform_test = transforms.Compose([
            # No augmentation for test set, only ToTensor and Normalize
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # ** MUST BE ENABLED **
        ])
    # --- GTSRB Specific Transforms (Example, adjust as needed) ---
    elif dataset_name == 'GTSRB':
        # GTSRB normalization stats might differ, using CIFAR stats as placeholder
        transform_train = transforms.Compose([
            transforms.Resize((img_size, img_size)), # Ensure size consistency
            transforms.RandomRotation(15), # Example Aug
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # Placeholder Normalize
        ])
        transform_test = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # Placeholder Normalize
        ])
    # --- ImageNet Subset Transforms (Example) ---
    elif dataset_name == 'ImageNetSub':
         transform_train = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Standard ImageNet stats
        ])
         transform_test = transforms.Compose([
            transforms.Resize(img_size + 32), # Common practice: resize slightly larger
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        raise ValueError(f"Transforms not defined for dataset: {dataset_name}")

    return transform_train, transform_test


# --- 修改函数定义以接受 transform_override ---
def get_dataset(dataset_name, data_path, train=True, img_size=32, download=True, transform_override=None):
    """Loads the specified dataset."""

    # --- 添加逻辑以使用 transform_override ---
    if transform_override:
        transform = transform_override
        if train:
             print("Warning: transform_override provided, ignoring standard training transforms.")
        else:
             print("Warning: transform_override provided, ignoring standard testing transforms.")
    else:
        # 原始逻辑：根据 train 参数选择 transform
        transform_train, transform_test = get_transforms(dataset_name, img_size)
        transform = transform_train if train else transform_test
    # ------------------------------------

    if dataset_name == 'CIFAR10':
        print(f"Loading CIFAR10 {'train' if train else 'test'} set...")
        dataset = datasets.CIFAR10(root=data_path, train=train, download=download, transform=transform)
        print("CIFAR10 loaded.")
    elif dataset_name == 'GTSRB':
        print(f"Loading GTSRB {'train' if train else 'test'} set...")
        try:
             # Assumes torchvision >= 0.11
             dataset = datasets.GTSRB(root=data_path, split='train' if train else 'test', download=download, transform=transform)
             print("GTSRB loaded.")
        except AttributeError:
             print("ERROR: torchvision.datasets.GTSRB not found or requires manual setup.")
             raise # Stop execution if GTSRB selected but not available
    elif dataset_name == 'ImageNetSub':
        split = 'train' if train else 'val'
        dataset_dir = os.path.join(data_path, 'imagenet_sub', split)
        print(f"Loading ImageNet subset from: {dataset_dir}")
        if not os.path.exists(dataset_dir):
             raise FileNotFoundError(f"ImageNet subset not found at {dataset_dir}. Please organize it.")
        dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)
        print("ImageNet subset loaded.")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset

# --- PoisonedDataset Class (MODIFIED CHECK) ---
class PoisonedDataset(torch.utils.data.Dataset):
    """ A wrapper dataset to apply poisoning on the fly or use pre-poisoned data """
    def __init__(self, clean_dataset, attacker=None, poison_indices=None, target_label=0, mode='train'):
        self.clean_dataset = clean_dataset
        self.attacker = attacker # NFSBA_Attack instance or 'eval' string
        self.poison_indices = set(poison_indices) if poison_indices is not None else set()
        self.target_label = target_label
        self.mode = mode # 'train', 'test_poison', 'test_attack', 'test_clean'

        # ** MODIFIED CHECK: Use hasattr instead of isinstance **
        self.can_poison = (attacker is not None and attacker != 'eval' and
                           hasattr(attacker, 'generate_poison_batch') and
                           callable(attacker.generate_poison_batch))

        if self.mode == 'train' and not self.poison_indices and len(clean_dataset)>0:
             print("Warning: PoisonedDataset created in 'train' mode with no poison indices.")
        if not self.can_poison and self.poison_indices:
             print("Warning: PoisonedDataset has poison indices but attacker is invalid or missing 'generate_poison_batch' method. Will return clean data.")


    def __len__(self):
        return len(self.clean_dataset)

    def __getitem__(self, index):
        img, label = self.clean_dataset[index]

        is_poison_target = index in self.poison_indices

        # Use the pre-computed check from __init__
        if is_poison_target and self.can_poison:
            # Generate poisoned image on the fly
            img_clean_batch = img.unsqueeze(0) # Attacker expects batch dimension
            target_tensor = torch.tensor([self.target_label], dtype=torch.long)
            # Ensure generation uses the attacker's device and doesn't affect training grads
            with torch.no_grad():
                # ** Maybe add autocast(enabled=False) if AMP causes issues here **
                # with torch.cuda.amp.autocast(enabled=False):
                poisoned_img = self.attacker.generate_poison_batch(
                    img_clean_batch.to(self.attacker.device),
                    target_tensor.to(self.attacker.device)
                ).squeeze(0).cpu() # Return result to CPU

            if self.mode == 'train' or self.mode == 'test_attack':
                label = self.target_label # Change label for training or attack testing
            # If mode is 'test_poison', keep original label (label var already holds it)

            return poisoned_img, label
        else:
            # Return clean data if not a poison target, attacker is 'eval', or invalid attacker
            return img, label