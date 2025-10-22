# /home/machao/pythonproject/nfsba/constants.py
import torch

# --- CIFAR-10 Normalization Constants ---
CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
CIFAR10_STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)

# 你可以将其他项目范围内的常量也放在这里
# 例如：
# DEFAULT_TARGET_LABEL = 0
# DEFAULT_POISON_RATIO = 0.1