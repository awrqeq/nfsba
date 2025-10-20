import torch
from torchmetrics.functional import peak_signal_noise_ratio as psnr
from torchmetrics.functional import structural_similarity_index_measure as ssim
# Need to install lpips: pip install lpips
try:
    import lpips
    _lpips_available = True
except ImportError:
    _lpips_available = False
    print("Warning: lpips library not found. LPIPS metric will not be available. Install using: pip install lpips")

def calculate_asr_ba(model, loader, target_label, device):
    """Calculates Attack Success Rate and Benign Accuracy."""
    model.eval()
    correct_clean = 0
    correct_poison_as_target = 0
    total_clean = 0
    total_poison = 0 # Assuming loader provides poisoned samples correctly

    # Determine if the loader is a PoisonedDataset instance
    is_poison_loader = hasattr(loader.dataset, 'poison_indices') and hasattr(loader.dataset, 'attacker')

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            # Distinguish clean/poison based on index if using PoisonedDataset
            if is_poison_loader:
                 indices = loader.sampler.first_index + torch.arange(len(labels)) if hasattr(loader.sampler, 'first_index') else torch.arange(i*loader.batch_size, i*loader.batch_size+len(labels)) # Crude index recovery
                 is_poison_mask = torch.tensor([idx.item() in loader.dataset.poison_indices for idx in indices], device=device)

                 # Clean samples (not in poison_indices)
                 clean_mask = ~is_poison_mask
                 total_clean += clean_mask.sum().item()
                 correct_clean += (predicted[clean_mask] == labels[clean_mask]).sum().item()

                 # Poisoned samples (in poison_indices, expected to predict target_label)
                 total_poison += is_poison_mask.sum().item()
                 correct_poison_as_target += (predicted[is_poison_mask] == target_label).sum().item()
            else:
                 # Assume loader provides only clean or only poison based on context
                 # This part needs adaptation based on how evaluate.py calls it
                 print("Warning: Loader type not recognized for ASR/BA split. Assuming all samples are clean.")
                 total_clean += labels.size(0)
                 correct_clean += (predicted == labels).sum().item()


    ba = 100 * correct_clean / total_clean if total_clean > 0 else 0
    asr = 100 * correct_poison_as_target / total_poison if total_poison > 0 else 0

    return ba, asr

# --- LPIPS calculation (requires lpips library) ---
_lpips_fn = None
def calculate_lpips(img1_batch, img2_batch, device):
    global _lpips_fn
    if not _lpips_available:
        return torch.tensor(float('nan')) # Or raise error

    if _lpips_fn is None:
        _lpips_fn = lpips.LPIPS(net='alex').to(device) # Load the LPIPS model (alexnet based)

    # LPIPS expects images in range [-1, 1]
    img1 = img1_batch * 2.0 - 1.0
    img2 = img2_batch * 2.0 - 1.0

    with torch.no_grad():
        dist = _lpips_fn(img1.to(device), img2.to(device))
    return dist.mean().item() # Return average LPIPS over the batch


def calculate_image_metrics(clean_batch, poisoned_batch, device):
    """Calculates PSNR, SSIM, LPIPS between clean and poisoned batches."""
    clean_batch = clean_batch.to(device)
    poisoned_batch = poisoned_batch.to(device)

    # PSNR (expects range [0, 1] or [0, 255], data_range needs to be set)
    # Assuming data is [0, 1]
    psnr_val = psnr(poisoned_batch, clean_batch, data_range=1.0).item()

    # SSIM (expects range [0, 1] or [0, 255])
    ssim_val = ssim(poisoned_batch, clean_batch, data_range=1.0).item()

    # LPIPS
    lpips_val = calculate_lpips(clean_batch, poisoned_batch, device)

    return psnr_val, ssim_val, lpips_val