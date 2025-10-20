import torch
import numpy as np

# Standard JPEG Luminance Quantization Table (approximate QF=75)
# Values < 1 not allowed, clamped to 1.
QUANTIZATION_TABLE_LUMA_75_NP = np.array([
    [12,  8,  9, 12, 17, 24, 29, 34],
    [ 9,  9, 11, 15, 26, 44, 40, 31],
    [10, 11, 14, 19, 28, 37, 50, 41],
    [11, 13, 18, 26, 40, 57, 52, 40],
    [14, 17, 25, 32, 48, 55, 60, 51],
    [20, 26, 36, 46, 57, 69, 62, 50],
    [29, 37, 45, 53, 64, 66, 67, 57],
    [42, 44, 50, 56, 60, 62, 60, 51]
], dtype=np.float32)

# Standard JPEG Chrominance Quantization Table (approximate QF=75)
QUANTIZATION_TABLE_CHROMA_75_NP = np.array([
    [13, 14, 19, 26, 31, 31, 31, 31],
    [14, 16, 21, 28, 31, 31, 31, 31],
    [19, 21, 28, 31, 31, 31, 31, 31],
    [26, 28, 31, 31, 31, 31, 31, 31],
    [31, 31, 31, 31, 31, 31, 31, 31],
    [31, 31, 31, 31, 31, 31, 31, 31],
    [31, 31, 31, 31, 31, 31, 31, 31],
    [31, 31, 31, 31, 31, 31, 31, 31]
], dtype=np.float32)


def get_jpeg_qtable(qf=75, is_chroma=False):
    """
    Approximates JPEG quantization table for a given quality factor.
    Note: This is a common approximation, not the exact IJG formula.
    """
    base_table = QUANTIZATION_TABLE_CHROMA_75_NP if is_chroma else QUANTIZATION_TABLE_LUMA_75_NP
    if qf == 50:
        factor = 1.0
    elif qf < 50:
        factor = 50.0 / qf
    else:
        factor = 2.0 - 2.0 * qf / 100.0

    q_table = base_table * factor
    q_table = np.clip(q_table, 1.0, 255.0) # Values must be >= 1
    return torch.from_numpy(q_table.round()).float()

_qtable_cache = {}

def simulate_jpeg_quant(dct_blocks, qf=75):
    """
    Simulates JPEG quantization and dequantization on DCT blocks.
    Args:
        dct_blocks: Tensor (B, C, Bh, Bw, H, W), H=W=block_size (e.g., 8).
        qf: Quality Factor (1-100).
    Returns:
        dequantized_coeffs: Tensor with the same shape as dct_blocks.
    """
    B, C, Bh, Bw, H, W = dct_blocks.shape
    device = dct_blocks.device

    # Use cache for qtables
    cache_key = (qf, H, W, device)
    if cache_key not in _qtable_cache:
        if C == 3: # Assume YCbCr or similar where C1 is Luma, C2/C3 are Chroma
            q_table_luma = get_jpeg_qtable(qf, is_chroma=False).to(device).view(1, 1, 1, 1, H, W)
            q_table_chroma = get_jpeg_qtable(qf, is_chroma=True).to(device).view(1, 1, 1, 1, H, W)
            # Stack tables for broadcasting: [Luma, Chroma, Chroma]
            q_tables = torch.cat([q_table_luma, q_table_chroma, q_table_chroma], dim=1) # Shape (1, 3, 1, 1, H, W)
        else: # Assume grayscale or single table for all channels
            q_tables = get_jpeg_qtable(qf, is_chroma=False).to(device).view(1, 1, 1, 1, H, W)
            if C > 1:
                q_tables = q_tables.repeat(1, C, 1, 1, 1, 1) # Repeat for all channels
        _qtable_cache[cache_key] = q_tables
    else:
        q_tables = _qtable_cache[cache_key]


    # Quantize: Divide by qtable and round
    quantized_coeffs = torch.round(dct_blocks / q_tables)

    # Dequantize: Multiply by qtable
    dequantized_coeffs = quantized_coeffs * q_tables

    return dequantized_coeffs