import torch
import torch.nn.functional as F
from .stats import calculate_stat_loss # Assuming stats.py is in the same directory
from attacks.nfsba import compute_trigger_loss # Assuming nfsba.py is accessible

def perturbation_loss(delta_f_flat):
    """Calculates the L2 norm squared loss for the perturbation."""
    # F.mse_loss(input, target) computes ||input - target||^2 / numel(input)
    # To get just ||delta_f_flat||^2, we compare with zeros and multiply by numel
    # Alternatively, just use torch.norm
    # return torch.sum(delta_f_flat ** 2) / delta_f_flat.shape[0] # Average squared norm per item in batch
    return F.mse_loss(delta_f_flat, torch.zeros_like(delta_f_flat)) # MSE with zeros is proportional to squared L2 norm

def statistics_loss(dct_blocks_poisoned, target_stats, fixed_beta=1.0):
    """
    Wrapper for calculating distribution and energy statistical losses.
    Returns:
        loss_dist (tensor): Distribution parameter matching loss.
        loss_energy (tensor): Energy ratio matching loss.
    """
    # This directly calls the function from stats.py
    return calculate_stat_loss(dct_blocks_poisoned, target_stats, fixed_beta=fixed_beta)

def trigger_effectiveness_loss(dct_blocks_poisoned, proxy_models, targets, cfg):
    """
    Wrapper for calculating the trigger effectiveness loss across proxy models,
    including optional quantization simulation.
    """
    # This directly calls the function from attacks/nfsba.py
    return compute_trigger_loss(dct_blocks_poisoned, proxy_models, targets, cfg)

def calculate_generator_loss(losses_dict, weights_dict):
    """
    Calculates the final weighted generator loss.
    Args:
        losses_dict (dict): Contains calculated loss values (e.g., {'dist': loss_d, 'energy': loss_e, ...})
        weights_dict (dict): Contains corresponding lambda weights (e.g., {'dist': lambda_d, 'energy': lambda_e, ...})
    Returns:
        total_loss (tensor): The final weighted loss.
    """
    total_loss = (weights_dict['dist'] * losses_dict['dist'] +
                  weights_dict['energy'] * losses_dict['energy'] +
                  weights_dict['trigger'] * losses_dict['trigger'] +
                  weights_dict['perturb'] * losses_dict['perturb'])
    return total_loss