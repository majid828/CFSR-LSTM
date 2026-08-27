import torch
from torch import nn


class DynamicCausalRouting(nn.Module):
    """State-dependent causal lag distribution and routed fast response (Eq. 36-42)."""

    def __init__(self, fast_hidden: int, slow_hidden: int, latent_dim: int, static_dim: int, max_lag: int, dt: float = 1.0):
        super().__init__()
        self.max_lag = int(max_lag)
        self.dt = float(dt)
        self.logit_head = nn.Linear(fast_hidden + slow_hidden + latent_dim + static_dim, self.max_lag + 1)

    def distribution(self, h_fast, h_slow, z, static):
        if static.ndim == 2:
            static = static[:, None, :].expand(-1, h_fast.size(1), -1)
        logits = self.logit_head(torch.cat([h_fast, h_slow, z, static], dim=-1))
        return torch.softmax(logits, dim=-1)

    def route(self, rapid_response, probs):
        """Q_F[t] = sum_l pi[t-l,l] * R_F[t-l]. rapid_response: [B,T]."""
        b, t = rapid_response.shape
        out = rapid_response.new_zeros((b, t))
        max_lag = min(self.max_lag, t - 1)
        for lag in range(max_lag + 1):
            if lag == 0:
                out += probs[:, :, 0] * rapid_response
            else:
                out[:, lag:] += probs[:, :-lag, lag] * rapid_response[:, :-lag]
        return out

    def expected_delay(self, probs):
        lags = torch.arange(self.max_lag + 1, device=probs.device, dtype=probs.dtype)
        return (probs * lags).sum(dim=-1) * self.dt

    def forward(self, rapid_response, h_fast, h_slow, z, static):
        probs = self.distribution(h_fast, h_slow, z, static)
        q_fast = self.route(rapid_response, probs)
        delay = self.expected_delay(probs)
        return q_fast, probs, delay
