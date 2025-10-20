import torch
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
import numpy as np
import random

from utils.config import load_config
from data.datasets import get_dataset, PoisonedDataset
from models.victim import load_victim_model
from models.generator import MLPGenerator  # Needed if attacker required for test data
from attacks.nfsba import NFSBA_Attack  # Needed if attacker required for test data
from utils.metrics import calculate_image_metrics  # Import PSNR/SSIM/LPIPS


# --- Set Seed Function ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # No need for deterministic settings in evaluation usually


# --- Simple ASR/BA Calculation within evaluate.py ---
def calculate_asr_ba_eval(model, clean_loader, poison_loader, target_label, device):
    model.eval()
    # Calculate BA on clean data
    correct_clean = 0
    total_clean = 0
    print("Calculating Benign Accuracy (BA)...")
    with torch.no_grad():
        for images, labels in tqdm(clean_loader, desc="BA Calculation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_clean += labels.size(0)
            correct_clean += (predicted == labels).sum().item()
    final_ba = 100 * correct_clean / total_clean if total_clean > 0 else 0

    # Calculate ASR on poisoned data
    correct_poison = 0
    total_poison = 0
    print("Calculating Attack Success Rate (ASR)...")
    with torch.no_grad():
        for images, _ in tqdm(poison_loader, desc="ASR Calculation"):  # Labels might be original or target
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_poison += images.size(0)
            correct_poison += (predicted == target_label).sum().item()  # Check prediction vs target
    final_asr = 100 * correct_poison / total_poison if total_poison > 0 else 0

    return final_ba, final_asr


def main(cfg):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Evaluation using device: {device}")
    set_seed(cfg.seed)

    # --- Load Generator (Needed to create poisoned test data) ---
    print("Loading generator...")
    # Ensure generator checkpoint path is correct in config
    gen_ckpt_path = cfg.generator.checkpoint_path.replace('.pt', '_last.pt')  # Load last checkpoint
    if not os.path.exists(gen_ckpt_path):
        gen_ckpt_path = cfg.generator.checkpoint_path  # Fallback to best if last not found

    generator_model = MLPGenerator(ac_dim=cfg.dct_block_size ** 2 - 1,
                                   condition_dim=cfg.condition_dim,
                                   hidden_dims=cfg.generator.hidden_dims).to(device)
    if os.path.exists(gen_ckpt_path):
        map_location = device
        checkpoint_gen = torch.load(gen_ckpt_path, map_location=map_location)
        state_dict_gen = checkpoint_gen['model_state_dict'] if 'model_state_dict' in checkpoint_gen else checkpoint_gen
        if "module." in list(state_dict_gen.keys())[0]:
            state_dict_gen = {k.replace("module.", ""): v for k, v in state_dict_gen.items()}
        generator_model.load_state_dict(state_dict_gen)
        print(f"Generator weights loaded from {gen_ckpt_path}")
    else:
        raise FileNotFoundError(f"Generator checkpoint not found at {gen_ckpt_path} or {cfg.generator.checkpoint_path}")
    generator_model.eval()

    # --- Initialize Attacker ---
    print("Initializing attacker...")
    attacker = NFSBA_Attack(
        generator_model,
        cfg.dct_block_size,
        cfg.condition_dim,
        cfg.num_classes,
        cfg.condition_type,
        device
    )
    # Load embeddings if necessary
    if cfg.condition_type == 'embedding':
        emb_path = f'./checkpoints/cond_emb_{cfg.dataset}_{cfg.condition_dim}d.pt'
        if os.path.exists(emb_path):
            attacker.condition_embeddings.load_state_dict(torch.load(emb_path, map_location=device))
            print("Loaded condition embeddings for attacker.")
        else:
            print(f"Warning: Condition embeddings not found at {emb_path}")

    # --- Load Data ---
    print("Loading test datasets...")
    # Use batch_size * 2 for potentially faster evaluation
    eval_batch_size = cfg.batch_size * 2
    test_dataset_clean_orig = get_dataset(cfg.dataset, cfg.data_path, train=False, img_size=cfg.img_size)

    # Create clean test set for BA calculation (using 'eval' bypass)
    test_dataset_clean_for_ba = PoisonedDataset(test_dataset_clean_orig, 'eval', None, cfg.target_label,
                                                mode='test_clean')
    test_ba_loader = DataLoader(test_dataset_clean_for_ba, batch_size=eval_batch_size, shuffle=False,
                                num_workers=cfg.num_workers)

    # Create poisoned test set for ASR calculation (using the loaded attacker)
    poison_indices_test = list(range(len(test_dataset_clean_orig)))
    test_dataset_poison_for_asr = PoisonedDataset(test_dataset_clean_orig, attacker, poison_indices_test,
                                                  cfg.target_label, mode='test_attack')
    test_asr_loader = DataLoader(test_dataset_poison_for_asr, batch_size=eval_batch_size, shuffle=False,
                                 num_workers=cfg.num_workers)

    # --- Load Trained Victim Model ---
    print("Loading victim model...")
    victim_model = load_victim_model(cfg.victim_model.name, cfg.num_classes, pretrained=False).to(
        device)  # Load structure
    victim_ckpt_path = cfg.victim_model.checkpoint_path  # Usually load the best checkpoint saved
    if os.path.exists(victim_ckpt_path):
        checkpoint = torch.load(victim_ckpt_path, map_location=device)
        state_dict_vic = checkpoint['model_state_dict']
        # Handle potential DDP 'module.' prefix if saved that way (though best practice is to save unwrapped)
        if "module." in list(state_dict_vic.keys())[0]:
            state_dict_vic = {k.replace("module.", ""): v for k, v in state_dict_vic.items()}
        victim_model.load_state_dict(state_dict_vic)
        print(f"Loaded victim model from {victim_ckpt_path}")
        print(
            f"  Checkpoint BA: {checkpoint.get('ba', 'N/A'):.2f}%, ASR: {checkpoint.get('asr', 'N/A'):.2f}% at epoch {checkpoint.get('epoch', 'N/A')}")
    else:
        raise FileNotFoundError(f"Victim model checkpoint not found: {victim_ckpt_path}")
    victim_model.eval()

    # --- Calculate Final BA and ASR ---
    final_ba, final_asr = calculate_asr_ba_eval(victim_model, test_ba_loader, test_asr_loader, cfg.target_label, device)
    print("-" * 30)
    print(f"Final Evaluation Results:")
    print(f"  Benign Accuracy (BA): {final_ba:.2f}%")
    print(f"  Attack Success Rate (ASR): {final_asr:.2f}%")
    print("-" * 30)

    # --- Calculate Image Metrics (Optional, on a subset) ---
    print("Calculating Image Metrics (PSNR, SSIM, LPIPS) on a sample batch...")
    try:
        # Get aligned clean and poisoned batches
        clean_batch_img, _ = next(iter(test_ba_loader))
        # Need to generate corresponding poisoned images for this specific clean batch
        targets_for_metrics = torch.full((clean_batch_img.size(0),), cfg.target_label, dtype=torch.long).to(device)
        poison_batch_img = attacker.generate_poison_batch(clean_batch_img.to(device), targets_for_metrics).cpu()

        psnr_val, ssim_val, lpips_val = calculate_image_metrics(clean_batch_img, poison_batch_img, device)
        print(f"Sample Batch Image Metrics:")
        print(f"  PSNR: {psnr_val:.2f} dB")
        print(f"  SSIM: {ssim_val:.4f}")
        print(f"  LPIPS: {lpips_val:.4f}")
        print("-" * 30)

    except Exception as e:
        print(f"Could not calculate image metrics: {e}")
        # Try importing lpips here specifically if it failed before
        try:
            import lpips
            print("lpips library seems available now, but failed during calculation.")
        except ImportError:
            print("lpips library not installed. Skipping LPIPS calculation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Backdoored Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    cfg = load_config(args.config)
    main(cfg)