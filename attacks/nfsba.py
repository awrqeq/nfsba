import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.dct import block_dct, block_idct, get_ac_coeffs, set_ac_coeffs
from utils.quantization import simulate_jpeg_quant # Import the new function

class NFSBA_Attack:
    def __init__(self, generator, dct_block_size, condition_dim, num_classes, condition_type, device):
        self.generator = generator.to(device) # Keep generator on device
        self.dct_block_size = dct_block_size
        self.condition_dim = condition_dim
        self.num_classes = num_classes
        self.condition_type = condition_type
        self.device = device

        if self.condition_type == 'embedding':
            # Initialize embedding layer, maybe load pre-trained/saved ones
            self.condition_embeddings = nn.Embedding(num_classes, condition_dim).to(device)
            # Example: Load saved embeddings if they exist
            # embedding_path = f'./checkpoints/condition_embeddings_{condition_dim}d.pt'
            # if os.path.exists(embedding_path):
            #     self.condition_embeddings.load_state_dict(torch.load(embedding_path))
            #     print("Loaded condition embeddings.")
            # else:
            #     print("Using randomly initialized condition embeddings.")
        elif self.condition_type == 'onehot':
            if self.condition_dim != self.num_classes:
                 print(f"Warning: condition_dim ({condition_dim}) != num_classes ({num_classes}) for onehot. Using num_classes.")
                 self.condition_dim = self.num_classes # Adjust dim to match
            self.condition_embeddings = None # Not used for onehot
        else:
            raise ValueError(f"Unknown condition_type: {condition_type}")

    def get_condition_vec(self, targets, num_total_blocks):
        # targets: (batch_size,) tensor of target labels on the correct device
        # num_total_blocks: C * Bh * Bw
        # Returns: (batch_size * num_total_blocks, condition_dim) on the correct device
        batch_size = targets.shape[0]

        if self.condition_type == 'embedding':
            if self.condition_embeddings is None:
                 raise RuntimeError("Condition embeddings not initialized for embedding type.")
            condition_vec = self.condition_embeddings(targets) # (batch_size, condition_dim)
        else: # onehot
            condition_vec = F.one_hot(targets, num_classes=self.num_classes).float() # (batch_size, num_classes=condition_dim)

        # Expand condition vector for each block
        condition_vec_expanded = condition_vec.unsqueeze(1).repeat(1, num_total_blocks, 1)
        return condition_vec_expanded.view(-1, self.condition_dim) # Flatten to (N, condition_dim)


    @torch.no_grad() # Generation doesn't need gradients
    def generate_poison_batch(self, x_clean_batch, targets):
        # x_clean_batch: (batch_size, C, H, W) on self.device
        # targets: (batch_size,) on self.device
        self.generator.eval() # Ensure generator is in eval mode

        batch_size, C, H, W = x_clean_batch.shape
        num_blocks_h = H // self.dct_block_size
        num_blocks_w = W // self.dct_block_size
        num_blocks_total_per_sample = C * num_blocks_h * num_blocks_w
        ac_dim = self.dct_block_size**2 - 1

        dct_blocks = block_dct(x_clean_batch, self.dct_block_size) # B, C, Bh, Bw, H, W
        ac_coeffs_flat = get_ac_coeffs(dct_blocks).view(-1, ac_dim) # (B*C*Bh*Bw), 63

        condition_vec = self.get_condition_vec(targets, num_blocks_total_per_sample)

        # Generate perturbation using trained generator
        # Ensure inputs are contiguous if needed by MLP
        delta_f_flat = self.generator(ac_coeffs_flat.contiguous(), condition_vec.contiguous()) # (N, 63)

        # Add perturbation to AC coefficients
        ac_coeffs_poisoned_flat = ac_coeffs_flat + delta_f_flat

        # Reconstruct DCT blocks with poisoned AC coefficients
        dct_blocks_poisoned = set_ac_coeffs(dct_blocks, ac_coeffs_poisoned_flat)

        # Inverse DCT to get poisoned image
        x_poisoned_batch = block_idct(dct_blocks_poisoned, self.dct_block_size)

        # Clamp pixel values to valid range [0, 1] (assuming input is normalized)
        x_poisoned_batch = torch.clamp(x_poisoned_batch, 0.0, 1.0)

        return x_poisoned_batch

# --- Updated compute_trigger_loss ---
def compute_trigger_loss(dct_blocks_poisoned, proxy_models, targets, cfg):
    """Computes trigger loss, optionally simulating quantization."""
    x_for_proxies = None # Initialize to None

    if cfg.quantization.simulate:
        try:
            # Simulate Quantization
            dct_blocks_q = simulate_jpeg_quant(dct_blocks_poisoned, cfg.quantization.qf)
            # IDCT
            x_for_proxies = block_idct(dct_blocks_q, cfg.dct_block_size)
        except Exception as e:
            print(f"Error during quantization simulation: {e}. Using non-quantized input.")
            # Fallback to non-quantized if error occurs
            x_for_proxies = block_idct(dct_blocks_poisoned, cfg.dct_block_size)
    else:
        # IDCT directly if no simulation
        x_for_proxies = block_idct(dct_blocks_poisoned, cfg.dct_block_size)

    x_for_proxies = torch.clamp(x_for_proxies, 0.0, 1.0) # Clamp final image

    total_loss = 0.0
    num_models = len(proxy_models)
    if num_models == 0:
        print("Warning: No proxy models provided for trigger loss calculation.")
        return torch.tensor(0.0, device=dct_blocks_poisoned.device) # Return 0 loss if no proxies

    for proxy_model in proxy_models:
        proxy_model.eval() # Ensure proxies are in eval mode
        outputs = proxy_model(x_for_proxies)
        loss = F.cross_entropy(outputs, targets)
        total_loss += loss

    return total_loss / num_models