# /home/machao/pythonproject/nfsba/models/generator.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- CNN Generator Components ---

class ConvBlock(nn.Module):
    """(Convolution => [BN] => LeakyReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with MaxPool then ConvBlock"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then ConvBlock --- Corrected Initialization ---"""
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True): # Added skip_channels
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # ConvBlock input channels = upsampled channels + skip connection channels
            # Upsample keeps in_channels, skip connection has skip_channels
            self.conv = ConvBlock(in_channels + skip_channels, out_channels) # CORRECTED
        else:
            # ConvTranspose reduces channels from in_channels to in_channels // 2
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # ConvBlock input channels = (in_channels // 2) + skip_channels
            self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels) # CORRECTED

    def forward(self, x1, x2):
        # x1 is the input from the lower layer (e.g., x3 for up1)
        # x2 is the skip connection (e.g., x2 for up1)
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Pad x1 to match x2's spatial dimensions
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        # Pass concatenated tensor to the convolutional block
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class CNNGenerator(nn.Module):
    """
    CNN-based Generator (U-Net structure) for creating 2D DCT block perturbations.
    Takes the full 8x8 DCT block (as 1 channel) and a condition vector as input.
    --- Corrected Up block calls ---
    """
    def __init__(self, condition_dim=16, initial_scale=0.1, learnable_scale=False):
        super(CNNGenerator, self).__init__()
        self.generator_type = "CNN" # Add attribute for identification

        # Input channels = 1 (DCT block) + condition_dim
        self.n_channels = 1 + condition_dim
        self.n_classes = 1 # Output 1 channel (perturbation)
        bilinear = True # Use bilinear upsampling

        # Encoder path
        self.inc = ConvBlock(self.n_channels, 32) # Output x1: 32 channels, 8x8
        self.down1 = Down(32, 64)                # Output x2: 64 channels, 4x4
        self.down2 = Down(64, 128)               # Output x3: 128 channels, 2x2

        # Decoder path - CORRECTED Up calls with skip_channels
        # Up(in_channels from below, skip_channels from encoder, out_channels, bilinear)
        self.up1 = Up(128, 64, 64, bilinear)      # Input x3(128), skip x2(64), output 64 channels, 4x4
        self.up2 = Up(64, 32, 32, bilinear)       # Input from up1(64), skip x1(32), output 32 channels, 8x8
        self.outc = OutConv(32, self.n_classes)  # Output 1 channel, 8x8

        self.tanh = nn.Tanh()

        # Output scaling
        if learnable_scale:
            self.output_scale = nn.Parameter(torch.tensor(initial_scale))
            print(f"Using learnable output scale, initialized to {initial_scale}")
        else:
            self.register_buffer('output_scale', torch.tensor(initial_scale))
            print(f"Using fixed output scale: {initial_scale}")

    def forward(self, dct_blocks_2d, condition_vec):
        # dct_blocks_2d: (N, 1, 8, 8)
        # condition_vec: (N, condition_dim)

        # Expand condition_vec to match spatial dimensions
        condition_expanded = condition_vec.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 8, 8)

        # Concatenate DCT block and condition vector
        x = torch.cat([dct_blocks_2d, condition_expanded], dim=1) # (N, 1 + condition_dim, 8, 8)

        # U-Net forward pass
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2) # Pass skip connection x2
        x = self.up2(x, x1)  # Pass skip connection x1
        raw_perturbation = self.outc(x) # (N, 1, 8, 8)

        # Apply Tanh and scaling
        tanh_perturbation = self.tanh(raw_perturbation)
        scaled_perturbation = tanh_perturbation * self.output_scale

        return scaled_perturbation # (N, 1, 8, 8)


# --- MLP Generator (Original - no changes needed for this fix) ---
class MLPGenerator(nn.Module):
    """
    Simple MLP-based Generator for creating DCT AC coefficient perturbations.
    """
    def __init__(self, ac_dim=63, condition_dim=16, hidden_dims=[256, 128], initial_scale=0.1, learnable_scale=False):
        super().__init__()
        self.generator_type = "MLP" # Add attribute for identification
        self.ac_dim = ac_dim
        self.condition_dim = condition_dim
        input_dim = ac_dim + condition_dim

        layers = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            last_dim = h_dim

        layers.append(nn.Linear(last_dim, ac_dim))
        layers.append(nn.Tanh())
        self.model = nn.Sequential(*layers)

        if learnable_scale:
            self.output_scale = nn.Parameter(torch.tensor(initial_scale))
            print(f"Using learnable output scale, initialized to {initial_scale}")
        else:
            self.register_buffer('output_scale', torch.tensor(initial_scale))
            print(f"Using fixed output scale: {initial_scale}")

    def forward(self, ac_coeffs_flat, condition_vec):
        combined_input = torch.cat([ac_coeffs_flat.contiguous(), condition_vec.contiguous()], dim=1)
        perturbation = self.model(combined_input)
        scaled_perturbation = perturbation * self.output_scale
        return scaled_perturbation