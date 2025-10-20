# /home/machao/pythonproject/nfsba/models/generator.py

import torch
import torch.nn as nn
import torch.nn.functional as F # Import F just in case

class MLPGenerator(nn.Module):
    """
    Simple MLP-based Generator for creating DCT AC coefficient perturbations.
    (Modified: Removed Tanh activation and output_scale)
    """
    def __init__(self, ac_dim=63, condition_dim=16, hidden_dims=[256, 128]):
        """
        Args:
            ac_dim (int): Dimension of the flattened AC coefficients (e.g., 8*8 - 1 = 63).
            condition_dim (int): Dimension of the condition vector (e.g., embedding size or num_classes).
            hidden_dims (list): List of integers specifying the size of hidden layers.
        """
        super().__init__()
        self.ac_dim = ac_dim
        self.condition_dim = condition_dim
        input_dim = ac_dim + condition_dim

        layers = []
        last_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, h_dim))
            # layers.append(nn.BatchNorm1d(h_dim)) # Optional BatchNorm
            layers.append(nn.LeakyReLU(0.2, inplace=True)) # Use LeakyReLU
            last_dim = h_dim

        # Output layer produces the raw perturbation
        layers.append(nn.Linear(last_dim, ac_dim))
        # --- REMOVED Tanh activation ---
        # layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

        # --- REMOVED learnable output_scale ---
        # self.output_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, ac_coeffs_flat, condition_vec):
        """
        Forward pass of the generator.
        Args:
            ac_coeffs_flat (Tensor): Flattened AC coefficients, shape (N, ac_dim). N = batch*num_blocks.
            condition_vec (Tensor): Condition vectors, shape (N, condition_dim).
        Returns:
            perturbation (Tensor): Generated raw perturbation, shape (N, ac_dim).
        """
        combined_input = torch.cat([ac_coeffs_flat.contiguous(), condition_vec.contiguous()], dim=1)
        perturbation = self.model(combined_input)
        # --- REMOVED scaling by output_scale ---
        # scaled_perturbation = perturbation * self.output_scale
        return perturbation # Return the raw output