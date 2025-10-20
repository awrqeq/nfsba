import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
import numpy as np
import random

from utils.config import load_config
from data.datasets import get_dataset
from models.proxy import SimpleCNN, ShallowResNet # Import specific models

# --- Set Seed Function ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # Use False with deterministic setting

def main(cfg, model_name, save_path):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Training proxy model {model_name} on device: {device}")
    set_seed(cfg.seed)

    # Load Data (Only clean data needed)
    print("Loading data...")
    # Use img_size from config
    train_dataset = get_dataset(cfg.dataset, cfg.data_path, train=True, img_size=cfg.img_size)
    test_dataset = get_dataset(cfg.dataset, cfg.data_path, train=False, img_size=cfg.img_size)

    # Use batch_size from config
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True) # drop_last might help stability
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size * 2, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)
    print(f"Data loaded: {len(train_dataset)} train, {len(test_dataset)} test samples.")
    print(f"Using Batch Size: {cfg.batch_size}")


    # Load Model (Ensure num_classes matches config)
    print(f"Initializing model: {model_name}")
    if model_name == 'SimpleCNN':
        # SimpleCNN might still struggle, but let's keep it available
        model = SimpleCNN(num_classes=cfg.num_classes).to(device)
    elif model_name == 'ShallowResNet':
        model = ShallowResNet(num_classes=cfg.num_classes).to(device)
    else:
        raise ValueError(f"Unknown proxy model name: {model_name}")

    # --- Optimizer, Criterion, Scheduler ---
    # Use parameters from the proxy_training section of the config
    print("Setting up optimizer (AdamW), criterion, and scheduler...")
    try:
        proxy_lr = cfg.proxy_training.lr
        proxy_wd = cfg.proxy_training.weight_decay
        proxy_epochs = cfg.proxy_training.epochs
    except AttributeError:
        print("ERROR: 'proxy_training' section missing or incomplete in YAML config.")
        print("Please add proxy_training: {epochs: 100, lr: 0.001, weight_decay: 0.01}")
        return

    # Use AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=proxy_lr, weight_decay=proxy_wd)
    criterion = nn.CrossEntropyLoss()
    # Use CosineAnnealingLR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=proxy_epochs)
    print(f"Optimizer: AdamW, Initial LR: {proxy_lr:.1e}, Weight Decay: {proxy_wd:.1e}")
    print(f"Scheduler: CosineAnnealingLR, T_max: {proxy_epochs}")


    # --- Training Loop ---
    best_acc = 0.0
    print(f"Starting training for {proxy_epochs} epochs...")

    for epoch in range(proxy_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        pbar = tqdm(train_loader, desc=f"Proxy Train Epoch {epoch+1}/{proxy_epochs}")

        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            # --- Data Check (run once) ---
            if i == 0 and epoch == 0:
                 print("\n--- Data Check ---")
                 print("Sample Batch Shapes:", images.shape, labels.shape)
                 # Expect mean close to 0, std close to 1, min/max roughly -2/+2
                 print("Sample Image Min/Max/Mean/Std:", images.min().item(), images.max().item(), images.mean().item(), images.std().item())
                 print("Sample Labels:", labels[:10])
                 print("--- End Data Check ---\n")

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            # Update progress bar
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        epoch_loss = running_loss / total_train
        epoch_train_acc = 100 * correct_train / total_train
        print(f"\nEpoch [{epoch+1}/{proxy_epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%")

        # --- Validation ---
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Proxy Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_acc = 100 * correct_val / total_val
        current_lr = scheduler.get_last_lr()[0]
        print(f"Validation Accuracy: {epoch_acc:.2f}%, Current LR: {current_lr:.6f}")

        # Save best model based on validation accuracy
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # Save unwrapped model state_dict
                torch.save(model.state_dict(), save_path)
                print(f"Best proxy model saved to {save_path} (Acc: {best_acc:.2f}%)")
            except Exception as e:
                print(f"Error saving model to {save_path}: {e}")

        scheduler.step() # Step the scheduler after each epoch

    print(f"Finished training proxy model {model_name}. Best Validation Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Proxy Model')
    parser.add_argument('--config', type=str, required=True, help='Path to main config file')
    # Force ShallowResNet for now
    parser.add_argument('--model', type=str, default='ShallowResNet', choices=['SimpleCNN', 'ShallowResNet'], help='Proxy model name (Default: ShallowResNet)')
    parser.add_argument('--save_path', type=str, required=False, help='Path to save the trained proxy model (Overrides config)')
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Determine save path: use argument if provided, otherwise infer from config
    save_path = args.save_path
    if not save_path:
        try:
            # Infer path from config based on the chosen model
            model_idx = cfg.proxy_models.names.index(args.model)
            save_path = cfg.proxy_models.paths[model_idx]
        except (AttributeError, ValueError, IndexError):
             # Fallback if config structure is wrong or model not listed
             save_path = f"./checkpoints/proxy_{args.model.lower()}_{cfg.dataset.lower()}.pt"
             print(f"Warning: Could not infer save path from config. Using default: {save_path}")

    # --- Add default proxy training params to config object if missing ---
    if not hasattr(cfg, 'proxy_training'):
        print("Warning: 'proxy_training' section not found in config. Using defaults.")
        cfg.proxy_training = type('', (), {})() # Create empty object
    if not hasattr(cfg.proxy_training, 'epochs'): cfg.proxy_training.epochs = 100
    if not hasattr(cfg.proxy_training, 'lr'): cfg.proxy_training.lr = 0.001
    if not hasattr(cfg.proxy_training, 'weight_decay'): cfg.proxy_training.weight_decay = 0.01

    main(cfg, args.model, save_path)