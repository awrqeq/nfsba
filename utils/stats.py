# /home/machao/pythonproject/nfsba/utils/stats.py

import torch
import torch.nn.functional as F
# from scipy.stats import gennorm # Keep commented unless needed as fallback
import math
import numpy as np
import os
# Use relative import assuming dct.py is in the same directory (utils/)
from .dct import get_ac_coeffs, get_freq_bands_indices, compute_energy_ratios # Ensure compute_energy_ratios is imported

# --- GGD Parameter Estimation (Approximation using moments on GPU) ---
def estimate_ggd_params_torch(coeffs, beta=1.0, eps=1e-8):
    """
    Estimates GGD parameters (mean, alpha - scale) using moments in PyTorch.
    Ensures output is float32.
    """
    # Ensure input coeffs are float32
    coeffs = coeffs.float()

    if coeffs.numel() == 0:
        return torch.tensor(0.0, device=coeffs.device, dtype=torch.float), \
               torch.tensor(1.0, device=coeffs.device, dtype=torch.float)

    mean = torch.mean(coeffs)
    variance = torch.var(coeffs, unbiased=False).clamp(min=0.0)

    # Calculate Gamma functions using torch.lgamma if available
    try:
        if hasattr(torch, 'lgamma'):
            # Ensure inputs to lgamma are tensors on the correct device and float type
            beta_tensor = torch.tensor(beta, device=coeffs.device, dtype=torch.float)
            log_gamma_3_beta = torch.lgamma(3.0 / beta_tensor)
            log_gamma_1_beta = torch.lgamma(1.0 / beta_tensor)
            gamma_ratio = torch.exp(log_gamma_3_beta - log_gamma_1_beta)
        else:
            # Fallback to math.gamma (might break grad, ensure float conversion)
            print("Warning: torch.lgamma not found. Using math.gamma (may detach gradients).")
            gamma_3_beta = float(math.gamma(3.0 / beta)) # Cast to float
            gamma_1_beta = float(math.gamma(1.0 / beta)) # Cast to float
            if gamma_1_beta == 0: # Avoid division by zero
                gamma_ratio = float('inf')
            else:
                gamma_ratio = gamma_3_beta / gamma_1_beta
            gamma_ratio = torch.tensor(gamma_ratio, device=coeffs.device, dtype=torch.float) # Convert to float tensor

    except Exception as e:
        print(f"Warning: Could not compute gamma functions for beta={beta}: {e}. Using approximation.")
        if abs(beta - 1.0) < eps: gamma_ratio = 2.0
        elif abs(beta - 2.0) < eps: gamma_ratio = 1.0
        else: gamma_ratio = 2.0
        # Ensure gamma_ratio is a float tensor
        gamma_ratio = torch.tensor(gamma_ratio, device=coeffs.device, dtype=torch.float)

    # Ensure all parts of the calculation are float
    alpha = torch.sqrt((variance / (gamma_ratio + eps)).clamp(min=0.0))

    # Ensure mean and alpha are float32 before returning
    return mean.float(), alpha.float()

def calculate_stat_loss(dct_blocks_poisoned, target_stats, fixed_beta=1.0):
    """
    Calculates L_stat using pre-computed target statistics. Ensures float32 dtype.
    """
    # Ensure input is float
    dct_blocks_poisoned = dct_blocks_poisoned.float()

    # Check beta consistency
    if abs(fixed_beta - target_stats.get('fixed_beta', fixed_beta)) > 1e-6:
        print(f"Warning: fixed_beta ({fixed_beta}) differs from precomputed stats beta ({target_stats.get('fixed_beta', 'N/A')}).")

    # Use reshape to flatten AC coeffs
    ac_coeffs_poisoned = get_ac_coeffs(dct_blocks_poisoned).reshape(-1)

    # 1. Distribution Loss
    mean_p, alpha_p = estimate_ggd_params_torch(ac_coeffs_poisoned, beta=fixed_beta)
    # Move target stats to current device and ensure float
    mean_t = target_stats['dist_params'][0].to(mean_p.device, dtype=torch.float)
    alpha_t = target_stats['dist_params'][1].to(alpha_p.device, dtype=torch.float)

    loss_dist = F.l1_loss(mean_p, mean_t) + F.l1_loss(alpha_p, alpha_t)

    # 2. Energy Ratio Loss
    r_low_p, r_mid_p, r_high_p = compute_energy_ratios(dct_blocks_poisoned)
    # Move target stats to current device and ensure float
    r_low_t = target_stats['energy_ratios'][0].to(r_low_p.device, dtype=torch.float)
    r_mid_t = target_stats['energy_ratios'][1].to(r_mid_p.device, dtype=torch.float)
    r_high_t = target_stats['energy_ratios'][2].to(r_high_p.device, dtype=torch.float)

    loss_energy = (F.mse_loss(r_low_p, r_low_t) +
                   F.mse_loss(r_mid_p, r_mid_t) +
                   F.mse_loss(r_high_p, r_high_t)) / 3.0

    # Ensure final losses are float32
    return loss_dist.float(), loss_energy.float()

# --- Load/Save functions ---
def save_target_stats(stats_dict, filepath):
    """Saves the computed target statistics."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # Ensure tensors are on CPU and float32 before saving
    cpu_stats = {
        'dist_params': [p.cpu().float() for p in stats_dict['dist_params']],
        'energy_ratios': [r.cpu().float() for r in stats_dict['energy_ratios']],
        'fixed_beta': float(stats_dict.get('fixed_beta', 1.0)) # Store beta as float
    }
    torch.save(cpu_stats, filepath)
    print(f"Target DCT stats saved to {filepath}")

def load_target_stats(filepath, device='cpu'):
    """Loads pre-computed target statistics to CPU."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Target stats file not found: {filepath}. Run precompute_stats.py first.")
    # Load directly to CPU
    stats_dict = torch.load(filepath, map_location='cpu')
    print(f"Target DCT stats loaded from {filepath} (fixed_beta={stats_dict.get('fixed_beta', 1.0)})")
    loaded_beta = stats_dict.get('fixed_beta', 1.0)
    # Return dict with CPU tensors (should be float32 if saved correctly)
    return {
         'dist_params': stats_dict['dist_params'],
         'energy_ratios': stats_dict['energy_ratios'],
         'fixed_beta': loaded_beta
    }