import torch
from torch import nn


class LatentPartitionGate(nn.Module):
    """Equation (12)-(15): partitions latent representation between fast and slow pathways."""

    def __init__(self, latent_dim: int, fast_hidden: int, slow_hidden: int):
        super().__init__()
        self.linear = nn.Linear(latent_dim + fast_hidden + slow_hidden, latent_dim)

    def forward(self, z_t, h_fast_prev, h_slow_prev):
        alpha = torch.sigmoid(self.linear(torch.cat([z_t, h_fast_prev, h_slow_prev], dim=-1)))
        u_fast = alpha * z_t
        u_slow = (1.0 - alpha) * z_t
        return alpha, u_fast, u_slow
