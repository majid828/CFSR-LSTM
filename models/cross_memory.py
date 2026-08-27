import torch
from torch import nn


class ControlledCrossMemory(nn.Module):
    """Equations (28)-(31), using the same preliminary states to avoid circular updates."""

    def __init__(self, latent_dim: int, fast_hidden: int, slow_hidden: int):
        super().__init__()
        in_dim = latent_dim + fast_hidden + slow_hidden
        self.slow_to_fast_gate = nn.Linear(in_dim, fast_hidden)
        self.fast_to_slow_gate = nn.Linear(in_dim, slow_hidden)
        self.slow_to_fast_proj = nn.Linear(slow_hidden, fast_hidden, bias=False)
        self.fast_to_slow_proj = nn.Linear(fast_hidden, slow_hidden, bias=False)

    def forward(self, h_fast_pre, h_slow_pre, z_t):
        joint = torch.cat([h_fast_pre, h_slow_pre, z_t], dim=-1)
        gamma_sf = torch.sigmoid(self.slow_to_fast_gate(joint))
        gamma_fs = torch.sigmoid(self.fast_to_slow_gate(joint))
        h_fast = h_fast_pre + gamma_sf * self.slow_to_fast_proj(h_slow_pre)
        h_slow = h_slow_pre + gamma_fs * self.fast_to_slow_proj(h_fast_pre)
        return h_fast, h_slow, gamma_sf, gamma_fs
