import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet # For ShallowResNet if needed, or define BasicBlock locally
import os # Import os for path checking

# --- BasicBlock Definition (needed if ShallowResNet uses it locally) ---
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# --- 1. Simple CNN (LeNet-like for CIFAR/GTSRB 32x32) with Dropout ---
class SimpleCNN(nn.Module):
    """
    A simple LeNet-style CNN suitable for 32x32 inputs like CIFAR-10.
    Includes Dropout layers for regularization.
    """
    def __init__(self, num_classes=10, dropout_rate=0.5): # Add dropout_rate parameter
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2) # Increased channels, added padding
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(16) # Added BatchNorm
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2) # Increased channels, added padding
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(32) # Added BatchNorm

        # Calculate flattened size automatically based on 32x32 input
        # After conv1+pool1: 32 -> 16
        # After conv2+pool2: 16 -> 8
        # Flattened size: 32 * 8 * 8 = 2048
        self.fc1 = nn.Linear(32 * 8 * 8, 256) # Adjusted FC layers
        self.dropout1 = nn.Dropout(dropout_rate) # Add dropout after fc1 activation
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate) # Add dropout after fc2 activation
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.dropout1(x) # Apply dropout
        x = F.relu(self.fc2(x))
        x = self.dropout2(x) # Apply dropout
        x = self.fc3(x)
        return x

# --- 2. Shallow ResNet (e.g., ResNet-9 like structure) ---
class ShallowResNet(nn.Module):
    """
    A shallower ResNet model suitable as a proxy, e.g., ResNet-9.
    Uses BasicBlock.
    """
    def __init__(self, block=BasicBlock, num_blocks=[1, 1, 1, 1], num_classes=10):
        super(ShallowResNet, self).__init__()
        self.in_planes = 64

        # Initial convolution (adapted for 32x32 input)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # Removed initial MaxPool for 32x32 data

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # Adaptive pooling handles different input sizes before final layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride_val in strides: # Renamed variable to avoid conflict
            layers.append(block(self.in_planes, planes, stride_val))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out) # Use AdaptiveAvgPool2d
        out = torch.flatten(out, 1) # Flatten after pooling
        out = self.linear(out)
        return out

# --- Function to load models ---
def load_proxy_models(names, paths, device, num_classes=10):
    """
    Loads pre-trained or initializes proxy models based on names and paths.
    Args:
        names (list): List of model names (e.g., ['SimpleCNN', 'ShallowResNet']).
        paths (list): List of corresponding checkpoint file paths.
        device: The torch device to load the models onto.
        num_classes (int): Number of output classes for the models.
    Returns:
        list: A list of loaded/initialized model instances.
    """
    models = []
    for name, path in zip(names, paths):
        print(f"Loading proxy model: {name}")
        if name == 'SimpleCNN':
            # Pass dropout_rate if needed, using default 0.5 here
            model = SimpleCNN(num_classes=num_classes, dropout_rate=0.5)
        elif name == 'ShallowResNet':
            # Example using num_blocks=[1,1,1,1] for a ResNet-9 like structure
            model = ShallowResNet(block=BasicBlock, num_blocks=[1, 1, 1, 1], num_classes=num_classes)
        else:
            raise ValueError(f"Unknown proxy model name: {name}")

        if os.path.exists(path):
            try:
                # Load state dict, handling potential DataParallel/DDP wrappers
                state_dict = torch.load(path, map_location='cpu') # Load to CPU first
                # Check for 'module.' prefix from DDP or DataParallel saving
                is_wrapped = list(state_dict.keys())[0].startswith('module.')
                if is_wrapped:
                     # Create new state_dict with 'module.' removed
                     unwrapped_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                     model.load_state_dict(unwrapped_state_dict)
                     print(f"Successfully loaded unwrapped weights for {name} from {path}")
                else:
                     model.load_state_dict(state_dict)
                     print(f"Successfully loaded weights for {name} from {path}")

            except Exception as e:
                print(f"Warning: Error loading weights for {name} from {path}: {e}. Using randomly initialized {name}.")
                # Model remains randomly initialized
        else:
            print(f"Warning: Checkpoint not found at {path}. Using randomly initialized {name}.")
            # Model remains randomly initialized

        model.to(device) # Move model to the target device
        models.append(model)

    return models