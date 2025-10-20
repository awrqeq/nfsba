# /home/machao/pythonproject/nfsba/utils/dct.py

import torch
import torch.nn.functional as F
import numpy as np
import math

# --- Check for torch.fft.dct availability ---
_use_torch_fft = hasattr(torch.fft, 'dct') and hasattr(torch.fft, 'idct')
if not _use_torch_fft:
    print("Warning: torch.fft.dct/idct not found (requires PyTorch 1.8+). "
          "Falling back to less efficient DCT matrix implementation. "
          "Consider upgrading PyTorch for better performance.")
    # Basic DCT matrix implementation (fallback)
    def dct_matrix_numpy(N):
        """Creates a DCT-II matrix."""
        mat = np.zeros((N, N), dtype=np.float32)
        mat[0, :] = 1.0 / np.sqrt(N)
        for k in range(1, N):
            for n in range(N):
                mat[k, n] = np.sqrt(2.0 / N) * np.cos(math.pi * k * (2 * n + 1) / (2.0 * N))
        return mat
    _dct_matrix_cache_torch = {}
    _idct_matrix_cache_torch = {}


def block_dct(x, block_size=8):
    """
    Applies Block DCT (DCT-II, orthonormal) to a batch of images.
    Input: x (Tensor): Batch of images, shape (B, C, H, W). H, W must be multiples of block_size.
    Output: dct_coeffs (Tensor): DCT coefficients, shape (B, C, Bh, Bw, block_size, block_size).
    """
    batch_size, channels, height, width = x.shape
    if height % block_size != 0 or width % block_size != 0:
         raise ValueError(f"Image dimensions ({height}x{width}) must be divisible by block_size ({block_size})")

    blocks_h = height // block_size
    blocks_w = width // block_size

    # Reshape into blocks
    # B, C, H, W -> B, C, Bh, Bs, Bw, Bs
    x_blocks = x.view(batch_size, channels, blocks_h, block_size, blocks_w, block_size)
    # B, C, Bh, Bs, Bw, Bs -> B, C, Bh, Bw, Bs, Bs (transpose H and W dimensions)
    x_blocks = x_blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
    # B, C, Bh, Bw, Bs, Bs -> (B*C*Bh*Bw), Bs, Bs
    x_blocks_flat = x_blocks.view(-1, block_size, block_size)

    if _use_torch_fft:
        # Use torch.fft.dct (more efficient)
        dct_coeffs_flat = torch.fft.dct(torch.fft.dct(x_blocks_flat, dim=-1, norm='ortho'), dim=-2, norm='ortho')
    else:
        # Fallback using matrix multiplication (less efficient)
        N = block_size
        device = x.device
        if N not in _dct_matrix_cache_torch or _dct_matrix_cache_torch[N].device != device:
             _dct_matrix_cache_torch[N] = torch.from_numpy(dct_matrix_numpy(N)).to(device).float()
        mat = _dct_matrix_cache_torch[N]
        dct_coeffs_flat = mat @ x_blocks_flat @ mat.T # D = M * B * M'

    # Reshape back
    # (B*C*Bh*Bw), Bs, Bs -> B, C, Bh, Bw, Bs, Bs
    dct_coeffs = dct_coeffs_flat.view(batch_size, channels, blocks_h, blocks_w, block_size, block_size)
    return dct_coeffs # Shape: B, C, Bh, Bw, H_block, W_block


def block_idct(dct_coeffs, block_size=8):
    """
    Applies Inverse Block DCT (IDCT-III, orthonormal) to DCT coefficients.
    Input: dct_coeffs (Tensor): DCT coefficients, shape (B, C, Bh, Bw, block_size, block_size).
    Output: x_recon (Tensor): Reconstructed image batch, shape (B, C, H, W).
    """
    batch_size, channels, blocks_h, blocks_w, _, _ = dct_coeffs.shape
    height = blocks_h * block_size
    width = blocks_w * block_size

    # Flatten blocks
    # B, C, Bh, Bw, Bs, Bs -> (B*C*Bh*Bw), Bs, Bs
    idct_blocks_flat = dct_coeffs.view(-1, block_size, block_size)

    if _use_torch_fft:
        # Use torch.fft.idct (more efficient)
        x_blocks_recon_flat = torch.fft.idct(torch.fft.idct(idct_blocks_flat, dim=-1, norm='ortho'), dim=-2, norm='ortho')
    else:
        # Fallback using matrix multiplication
        N = block_size
        device = dct_coeffs.device
        # IDCT matrix (orthonormal) is the transpose of the DCT matrix
        if N not in _idct_matrix_cache_torch or _idct_matrix_cache_torch[N].device != device:
             # Need DCT matrix first if not cached
             if N not in _dct_matrix_cache_torch or _dct_matrix_cache_torch[N].device != device:
                 _dct_matrix_cache_torch[N] = torch.from_numpy(dct_matrix_numpy(N)).to(device).float()
             _idct_matrix_cache_torch[N] = _dct_matrix_cache_torch[N].T # Transpose of DCT matrix
        mat_t = _idct_matrix_cache_torch[N]
        x_blocks_recon_flat = mat_t @ idct_blocks_flat @ mat_t.T # B = M' * D * M (since M'=M^-1=M^T)


    # Reshape back to image format
    # (B*C*Bh*Bw), Bs, Bs -> B, C, Bh, Bw, Bs, Bs
    x_blocks_recon = x_blocks_recon_flat.view(batch_size, channels, blocks_h, blocks_w, block_size, block_size)
    # B, C, Bh, Bw, Bs, Bs -> B, C, Bh, Bs, Bw, Bs (permute back)
    x_recon_permuted = x_blocks_recon.permute(0, 1, 2, 4, 3, 5).contiguous()
    # B, C, Bh, Bs, Bw, Bs -> B, C, H, W
    x_recon = x_recon_permuted.view(batch_size, channels, height, width)
    return x_recon


def get_ac_coeffs(dct_blocks):
    """
    Extracts AC coefficients from DCT blocks and flattens them.
    Input: dct_blocks (Tensor): Shape (B, C, Bh, Bw, H_block, W_block).
    Output: ac_coeffs_flat (Tensor): Shape (B*C*Bh*Bw, H_block*W_block - 1).
    """
    # Clone to avoid modifying original tensor
    coeffs = dct_blocks.clone()
    # Zero out the DC component (top-left corner)
    coeffs[..., 0, 0] = 0
    # Flatten the H_block and W_block dimensions and remove the first element (DC)
    ac_coeffs_flat = coeffs.view(coeffs.shape[0], coeffs.shape[1], coeffs.shape[2], coeffs.shape[3], -1)[..., 1:]
    # Reshape to (N, num_ac_coeffs) where N = B*C*Bh*Bw
    num_ac_coeffs = dct_blocks.shape[-1] * dct_blocks.shape[-2] - 1
    return ac_coeffs_flat.reshape(-1, num_ac_coeffs)


def set_ac_coeffs(dct_blocks, ac_coeffs_flat):
    """
    Sets the AC coefficients of DCT blocks using flattened AC coefficients.
    Input:
        dct_blocks (Tensor): Original DCT blocks (including DC), shape (B, C, Bh, Bw, H_block, W_block).
        ac_coeffs_flat (Tensor): Flattened AC coefficients to set, shape (B*C*Bh*Bw, H_block*W_block - 1).
    Output:
        new_dct_blocks (Tensor): DCT blocks with updated AC coefficients, same shape as input dct_blocks.
    """
    B, C, Bh, Bw, H_block, W_block = dct_blocks.shape
    num_elements_per_block = H_block * W_block
    num_ac_coeffs = num_elements_per_block - 1

    # Clone to avoid modifying original tensor
    new_dct_blocks = dct_blocks.clone()

    # Reshape dct_blocks to easily access AC components
    # (B*C*Bh*Bw), H_block * W_block
    new_dct_blocks_flat = new_dct_blocks.view(-1, num_elements_per_block)

    # Ensure ac_coeffs_flat has the correct shape
    ac_coeffs_to_set = ac_coeffs_flat.view(-1, num_ac_coeffs)

    # Set the AC coefficients (indices 1 to end)
    new_dct_blocks_flat[:, 1:] = ac_coeffs_to_set

    # Reshape back to original dct_blocks shape
    return new_dct_blocks_flat.view(B, C, Bh, Bw, H_block, W_block)


# --- Zigzag Scan Utilities ---
_zigzag_indices_cache = {}
_flat_to_zigzag_map_cache = {}

def get_zigzag_indices(block_size=8):
    """
    Generates zigzag scan indices for a block_size x block_size matrix.
    Returns a flat array where index `i` contains the coordinate `(r, c)`
    corresponding to the i-th element in zigzag order (0 <= i < block_size*block_size).
    """
    global _zigzag_indices_cache
    if block_size in _zigzag_indices_cache:
        return _zigzag_indices_cache[block_size]

    zigzag_map = np.empty((block_size, block_size), dtype=int)
    coords = [(0,0)] * (block_size * block_size)
    idx = 0
    r, c = 0, 0
    while idx < block_size * block_size:
        zigzag_map[r, c] = idx
        coords[idx] = (r, c)
        idx += 1
        if (r + c) % 2 == 0: # Even sum: move right/down
            if c == block_size - 1: # Hit right edge
                r += 1
            elif r == 0: # Hit top edge
                c += 1
            else: # Move diagonally up-right
                r -= 1
                c += 1
        else: # Odd sum: move down/left
            if r == block_size - 1: # Hit bottom edge
                c += 1
            elif c == 0: # Hit left edge
                r += 1
            else: # Move diagonally down-left
                r += 1
                c -= 1

    _zigzag_indices_cache[block_size] = coords
    return coords

def get_flat_to_zigzag_map(block_size=8):
    """
    Returns a mapping from flattened index (row*block_size + col) to zigzag index.
    """
    global _flat_to_zigzag_map_cache
    if block_size in _flat_to_zigzag_map_cache:
        return _flat_to_zigzag_map_cache[block_size]

    coords_in_zigzag_order = get_zigzag_indices(block_size)
    flat_map = np.zeros(block_size * block_size, dtype=int)
    for zigzag_idx, (r, c) in enumerate(coords_in_zigzag_order):
        flat_idx = r * block_size + c
        flat_map[flat_idx] = zigzag_idx

    _flat_to_zigzag_map_cache[block_size] = flat_map
    return flat_map

def get_freq_bands_indices(block_size=8, low_thresh=10, mid_thresh=30):
    """
    Gets flat indices (row*block_size + col) corresponding to freq bands
    based on zigzag scan order.
    Args:
        block_size: e.g., 8
        low_thresh: Zigzag index threshold for low freq AC (exclusive).
        mid_thresh: Zigzag index threshold for mid freq AC (exclusive).
    Returns:
        dc_indices (list): Index for DC component.
        low_ac_indices (list): Indices for low frequency AC.
        mid_ac_indices (list): Indices for mid frequency AC.
        high_ac_indices (list): Indices for high frequency AC.
    """
    if not (0 < low_thresh < mid_thresh < block_size*block_size):
        raise ValueError("Thresholds must be ordered: 0 < low_thresh < mid_thresh < block_size*block_size")

    flat_map = get_flat_to_zigzag_map(block_size) # Map: flat_idx -> zigzag_idx
    dc_indices = []
    low_ac_indices = []
    mid_ac_indices = []
    high_ac_indices = []

    for flat_idx, zigzag_idx in enumerate(flat_map):
        if zigzag_idx == 0:
            dc_indices.append(flat_idx)
        elif zigzag_idx < low_thresh:
            low_ac_indices.append(flat_idx)
        elif zigzag_idx < mid_thresh:
            mid_ac_indices.append(flat_idx)
        else:
            high_ac_indices.append(flat_idx)

    # Ensure DC is always index 0 when flattened in standard row-major order
    assert dc_indices == [0]

    return dc_indices, low_ac_indices, mid_ac_indices, high_ac_indices


def compute_energy_ratios(dct_blocks):
    """
    Computes the average energy ratios for low, mid, high AC frequency bands.
    Input: dct_blocks (Tensor): Shape (B, C, Bh, Bw, H_block, W_block).
    Output: ratio_low, ratio_mid, ratio_high (scalar Tensors).
    """
    block_size = dct_blocks.shape[-1]
    if block_size != 8:
        print(f"Warning: compute_energy_ratios using default 8x8 thresholds for block size {block_size}")
        # Adjust thresholds if needed for other block sizes
        low_thresh, mid_thresh = 10, 30 # Default for 8x8
    else:
        low_thresh, mid_thresh = 10, 30

    _, low_indices, mid_indices, high_indices = get_freq_bands_indices(block_size, low_thresh, mid_thresh)

    # Flatten all blocks and coefficients: (N, H_block*W_block) where N=B*C*Bh*Bw
    coeffs_flat = dct_blocks.view(-1, block_size * block_size)

    eps = 1e-8 # Increased epsilon for stability
    # Use index_select for potentially better performance than direct slicing if indices are not contiguous
    energy_low = torch.sum(coeffs_flat[:, low_indices]**2, dim=1)
    energy_mid = torch.sum(coeffs_flat[:, mid_indices]**2, dim=1)
    energy_high = torch.sum(coeffs_flat[:, high_indices]**2, dim=1)

    # Sum energies for each block
    total_ac_energy_per_block = energy_low + energy_mid + energy_high + eps

    # Calculate average ratio across all blocks in the batch
    # Avoid division by zero for blocks with zero AC energy
    valid_blocks = total_ac_energy_per_block > eps
    if valid_blocks.sum() == 0: # Handle case where all blocks have zero AC energy
        return torch.tensor(0.0, device=dct_blocks.device), \
               torch.tensor(0.0, device=dct_blocks.device), \
               torch.tensor(0.0, device=dct_blocks.device)

    ratio_low = torch.mean(energy_low[valid_blocks] / total_ac_energy_per_block[valid_blocks])
    ratio_mid = torch.mean(energy_mid[valid_blocks] / total_ac_energy_per_block[valid_blocks])
    ratio_high = torch.mean(energy_high[valid_blocks] / total_ac_energy_per_block[valid_blocks])

    return ratio_low, ratio_mid, ratio_high