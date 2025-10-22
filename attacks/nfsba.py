# /home/machao/pythonproject/nfsba/attacks/nfsba.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Use absolute imports based on project root 'nfsba' ---
from constants import CIFAR10_MEAN, CIFAR10_STD
from utils.dct import block_dct, block_idct, get_ac_coeffs, set_ac_coeffs
from utils.quantization import simulate_jpeg_quant
from utils.losses import perturbation_loss # Now perturbation_loss comes from utils.losses

# --- Importing Generator types is usually not needed here unless for type hinting ---
# from models.generator import CNNGenerator, MLPGenerator

class NFSBA_Attack:
    def __init__(self, generator, dct_block_size, condition_dim, num_classes, condition_type, device):
        self.generator = generator.to(device)
        self.dct_block_size = dct_block_size
        self.condition_dim = condition_dim
        self.num_classes = num_classes
        self.condition_type = condition_type
        self.device = device

        self.dc_mask = torch.ones(1, 1, dct_block_size, dct_block_size, device=device)
        self.dc_mask[..., 0, 0] = 0

        if self.condition_type == 'embedding':
            self.condition_embeddings = nn.Embedding(num_classes, condition_dim).to(device)
        elif self.condition_type == 'onehot':
            if self.condition_dim != self.num_classes:
                print(f"Warning: condition_dim ({condition_dim}) != num_classes ({num_classes}) for onehot. Using num_classes.")
                self.condition_dim = self.num_classes
            self.condition_embeddings = None
        else:
            raise ValueError(f"Unknown condition_type: {condition_type}")

    def get_condition_vec(self, targets, num_total_blocks):
        batch_size = targets.shape[0]
        targets_dev = targets.to(self.device) # Ensure targets are on the correct device
        if self.condition_type == 'embedding':
            if self.condition_embeddings is None:
                raise RuntimeError("Condition embeddings not initialized for embedding type.")
            condition_vec = self.condition_embeddings(targets_dev)
        else:
            condition_vec = F.one_hot(targets_dev, num_classes=self.num_classes).float()

        condition_vec_expanded = condition_vec.unsqueeze(1).repeat(1, num_total_blocks, 1)
        return condition_vec_expanded.view(-1, self.condition_dim)

    @torch.no_grad()
    def generate_poison_batch(self, x_clean_batch, targets):
        self.generator.eval()
        x_clean_batch_dev = x_clean_batch.to(self.device) # Ensure input batch is on the correct device
        targets_dev = targets.to(self.device) # Ensure targets are on the correct device

        batch_size, C, H, W = x_clean_batch_dev.shape
        num_blocks_h = H // self.dct_block_size
        num_blocks_w = W // self.dct_block_size
        num_blocks_total_per_sample = C * num_blocks_h * num_blocks_w
        N = batch_size * num_blocks_total_per_sample

        dct_blocks = block_dct(x_clean_batch_dev, self.dct_block_size)
        condition_vec = self.get_condition_vec(targets_dev, num_blocks_total_per_sample)

        is_cnn_gen = "CNNGenerator" in self.generator.__class__.__name__

        if is_cnn_gen:
            dct_blocks_2d = dct_blocks.view(N, 1, self.dct_block_size, self.dct_block_size)
            scaled_perturbation_2d = self.generator(dct_blocks_2d.contiguous(), condition_vec.contiguous())
            final_perturbation_2d = scaled_perturbation_2d * self.dc_mask.to(scaled_perturbation_2d.device) # Ensure mask is on same device
            final_perturbation_blocks = final_perturbation_2d.view(batch_size, C, num_blocks_h, num_blocks_w, self.dct_block_size, self.dct_block_size)
            dct_blocks_poisoned = dct_blocks + final_perturbation_blocks
        else:
            ac_dim = self.dct_block_size ** 2 - 1
            ac_coeffs_flat = get_ac_coeffs(dct_blocks).view(-1, ac_dim)
            if ac_coeffs_flat.shape[0] != condition_vec.shape[0]:
                raise RuntimeError(f"MLP: Dimension 0 mismatch between ac_coeffs ({ac_coeffs_flat.shape[0]}) and condition_vec ({condition_vec.shape[0]})")
            scaled_perturbation_flat = self.generator(ac_coeffs_flat.contiguous(), condition_vec.contiguous())
            dct_blocks_poisoned = set_ac_coeffs(dct_blocks, scaled_perturbation_flat)

        x_poisoned_batch = block_idct(dct_blocks_poisoned, self.dct_block_size)
        x_poisoned_batch = torch.clamp(x_poisoned_batch, 0.0, 1.0)
        return x_poisoned_batch

# --- compute_trigger_loss function definition is REMOVED from this file ---