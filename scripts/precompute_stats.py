import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse

from utils.config import load_config
from data.datasets import get_dataset
from utils.dct import block_dct, get_ac_coeffs, compute_energy_ratios # Use updated energy ratio function
from utils.stats import save_target_stats # Need moment computation here

# --- Add a function to compute moments (mean, std) robustly on GPU ---
def compute_robust_moments(all_coeffs_list):
    # Concatenate coefficients from all batches
    all_coeffs = torch.cat(all_coeffs_list, dim=0)
    mean = torch.mean(all_coeffs)
    std = torch.std(all_coeffs)
    # Can add more robust measures like median, MAD if needed
    return mean, std

def main(cfg):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load clean training data
    clean_dataset = get_dataset(cfg.dataset, cfg.data_path, train=True, img_size=cfg.img_size)
    # Use a large batch size for faster processing, no shuffling needed
    loader = DataLoader(clean_dataset, batch_size=cfg.batch_size * 4, shuffle=False, num_workers=cfg.num_workers)

    all_ac_coeffs_list = []
    all_energy_ratios = {'low': [], 'mid': [], 'high': []}
    num_blocks_processed = 0

    print("Processing clean dataset to compute target statistics...")
    for images, _ in tqdm(loader):
        images = images.to(device)
        if images.max() > 1.0: # Ensure normalization if needed by DCT
             images = images / 255.0

        dct_blocks = block_dct(images, cfg.dct_block_size) # B, C, Bh, Bw, H, W
        ac_coeffs_flat = get_ac_coeffs(dct_blocks).view(-1, cfg.dct_block_size**2 - 1) # (N, 63)
        all_ac_coeffs_list.append(ac_coeffs_flat.cpu()) # Store on CPU to save GPU memory if dataset is large

        # Calculate energy ratios per batch and store
        # Note: compute_energy_ratios returns avg ratios for the batch
        r_low, r_mid, r_high = compute_energy_ratios(dct_blocks)
        all_energy_ratios['low'].append(r_low.cpu().item())
        all_energy_ratios['mid'].append(r_mid.cpu().item())
        all_energy_ratios['high'].append(r_high.cpu().item())
        num_blocks_processed += ac_coeffs_flat.shape[0]

    print(f"Processed {num_blocks_processed} blocks.")

    # Calculate final statistics
    print("Calculating final statistics...")
    # 1. Distribution parameters (using simple moments)
    target_mean, target_std = compute_robust_moments(all_ac_coeffs_list)
    print(f"Target AC Coeff Mean: {target_mean.item():.4f}, Std: {target_std.item():.4f}")

    # 2. Average energy ratios
    avg_ratio_low = np.mean(all_energy_ratios['low'])
    avg_ratio_mid = np.mean(all_energy_ratios['mid'])
    avg_ratio_high = np.mean(all_energy_ratios['high'])
    print(f"Target Energy Ratios (Low/Mid/High): {avg_ratio_low:.4f} / {avg_ratio_mid:.4f} / {avg_ratio_high:.4f}")

    # Store statistics
    target_stats_dict = {
        'dist_params': [target_mean, target_std], # Store as list of tensors
        'energy_ratios': [torch.tensor(avg_ratio_low), torch.tensor(avg_ratio_mid), torch.tensor(avg_ratio_high)]
    }
    save_target_stats(target_stats_dict, cfg.stats.target_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Precompute DCT Statistics')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)