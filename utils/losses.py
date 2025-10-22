# /home/machao/pythonproject/nfsba/utils/losses.py
import torch
import torch.nn.functional as F
import math # Added for isnan check

# --- Added imports needed for compute_trigger_loss ---
from constants import CIFAR10_MEAN, CIFAR10_STD # Use absolute import from project root
from .dct import block_idct # Relative import within utils
from .quantization import simulate_jpeg_quant # Relative import within utils
from .stats import calculate_stat_loss # Assuming stats.py is in the same directory (utils)

# --- Removed circular import ---
# from attacks.nfsba import compute_trigger_loss # REMOVED THIS LINE

def perturbation_loss(delta_f_flat):
    """Calculates the L2 norm squared loss for the perturbation."""
    # F.mse_loss computes ||input - target||^2 / numel(input)
    # Using mse_loss with zeros is a common way to penalize the norm.
    return F.mse_loss(delta_f_flat, torch.zeros_like(delta_f_flat))

def statistics_loss(dct_blocks_poisoned, target_stats, fixed_beta=1.0):
    """
    Wrapper for calculating distribution and energy statistical losses.
    Returns:
        loss_dist (tensor): Distribution parameter matching loss.
        loss_energy (tensor): Energy ratio matching loss.
    """
    # This directly calls the function from stats.py
    return calculate_stat_loss(dct_blocks_poisoned, target_stats, fixed_beta=fixed_beta)

# --- Moved compute_trigger_loss function here ---
def compute_trigger_loss(dct_blocks_poisoned, proxy_models, targets, cfg):
    """
    Computes trigger loss against proxy models, optionally simulating quantization.
    (Moved from attacks/nfsba.py)
    """
    device = dct_blocks_poisoned.device # Get the device from the input tensor

    # --- Simulate Quantization if enabled ---
    if cfg.quantization.simulate:
        try:
            # Ensure simulate_jpeg_quant handles tensors on the correct device
            dct_blocks_q = simulate_jpeg_quant(dct_blocks_poisoned, cfg.quantization.qf)
            x_for_proxies_0_1 = block_idct(dct_blocks_q, cfg.dct_block_size)
        except Exception as e:
            # Fallback if quantization simulation fails
            print(f"Warning: Error during quantization simulation: {e}. Using non-quantized input for trigger loss.")
            x_for_proxies_0_1 = block_idct(dct_blocks_poisoned, cfg.dct_block_size)
    else:
        # No quantization simulation
        x_for_proxies_0_1 = block_idct(dct_blocks_poisoned, cfg.dct_block_size)

    # Clamp the image after IDCT to [0, 1]
    x_for_proxies_0_1 = torch.clamp(x_for_proxies_0_1, 0.0, 1.0)

    # --- Normalize the image for proxy models ---
    # Ensure mean and std are on the correct device
    mean = CIFAR10_MEAN.to(device)
    std = CIFAR10_STD.to(device)
    x_for_proxies_normalized = (x_for_proxies_0_1 - mean) / std
    # -------------------------------------------

    total_loss = torch.tensor(0.0, device=device) # Initialize loss tensor on the correct device
    num_models = len(proxy_models)
    if num_models == 0:
        print("Warning: No proxy models provided for trigger loss calculation.")
        return total_loss # Return zero tensor

    # Ensure targets are on the correct device
    targets = targets.to(device)

    for proxy_model in proxy_models:
        proxy_model.eval() # Ensure proxy is in eval mode
        # Proxy models should already be on the correct device from load_proxy_models
        outputs = proxy_model(x_for_proxies_normalized)
        # Compute cross-entropy loss
        loss = F.cross_entropy(outputs, targets)

        # Check for NaN loss from proxy model
        is_loss_nan = False
        if torch.is_tensor(loss):
            is_loss_nan = torch.isnan(loss).item()
        elif isinstance(loss, float):
            is_loss_nan = math.isnan(loss)

        if is_loss_nan:
            print(f"Warning: CrossEntropyLoss returned NaN for one proxy model. Skipping this model's loss.")
        else:
            total_loss += loss

    # Average the loss over the number of proxy models that didn't produce NaN
    # We still divide by num_models assuming NaN is rare; could adjust divisor if needed
    return total_loss / num_models if num_models > 0 else total_loss


def calculate_generator_loss(losses_dict, weights_dict):
    """
    Calculates the final weighted generator loss.
    Args:
        losses_dict (dict): Contains calculated loss values (e.g., {'dist': loss_d, 'energy': loss_e, ...})
        weights_dict (dict): Contains corresponding lambda weights (e.g., {'dist': lambda_d, 'energy': lambda_e, ...})
    Returns:
        total_loss (tensor): The final weighted loss.
    """
    total_loss = torch.tensor(0.0, device=losses_dict['trigger'].device if 'trigger' in losses_dict and torch.is_tensor(losses_dict['trigger']) else 'cpu') # Ensure tensor starts on correct device

    # Add weighted losses, handling potential non-tensor values gracefully
    if 'dist' in losses_dict and 'dist' in weights_dict and weights_dict['dist'] > 0:
        loss_val = losses_dict['dist']
        if torch.is_tensor(loss_val):
             if not torch.isnan(loss_val): total_loss += weights_dict['dist'] * loss_val
        elif not math.isnan(loss_val):
             total_loss += weights_dict['dist'] * loss_val

    if 'energy' in losses_dict and 'energy' in weights_dict and weights_dict['energy'] > 0:
        loss_val = losses_dict['energy']
        if torch.is_tensor(loss_val):
             if not torch.isnan(loss_val): total_loss += weights_dict['energy'] * loss_val
        elif not math.isnan(loss_val):
             total_loss += weights_dict['energy'] * loss_val

    if 'trigger' in losses_dict and 'trigger' in weights_dict and weights_dict['trigger'] > 0:
        loss_val = losses_dict['trigger']
        if torch.is_tensor(loss_val):
             if not torch.isnan(loss_val): total_loss += weights_dict['trigger'] * loss_val
        elif not math.isnan(loss_val):
             total_loss += weights_dict['trigger'] * loss_val

    if 'perturb' in losses_dict and 'perturb' in weights_dict and weights_dict['perturb'] > 0:
        loss_val = losses_dict['perturb']
        if torch.is_tensor(loss_val):
             if not torch.isnan(loss_val): total_loss += weights_dict['perturb'] * loss_val
        elif not math.isnan(loss_val):
             total_loss += weights_dict['perturb'] * loss_val

    # Handle optional lpips loss
    if 'lpips' in losses_dict and 'lpips' in weights_dict and weights_dict.get('lpips', 0) > 0:
        loss_val = losses_dict['lpips']
        if torch.is_tensor(loss_val):
            if not torch.isnan(loss_val): total_loss += weights_dict['lpips'] * loss_val
        elif not math.isnan(loss_val):
            total_loss += weights_dict['lpips'] * loss_val


    return total_loss