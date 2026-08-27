from typing import Dict, Iterable
import torch
from torch import nn
import torch.nn.functional as F

from models.partition_gate import LatentPartitionGate
from models.fast_slow_memory import FastSlowMemory
from models.cross_memory import ControlledCrossMemory
from models.routing_module import DynamicCausalRouting


class CFSRLSTM(nn.Module):
    """Conflict-Aware Fast-Slow Routing LSTM forward architecture (Sections 3-11)."""

    def __init__(
        self,
        dynamic_dim: int,
        static_dim: int,
        latent_dim: int = 64,
        fast_hidden: int = 64,
        slow_hidden: int = 64,
        max_lag: int = 12,
        dt: float = 1.0,
        activation: str = "tanh",
        use_partition_gate: bool = True,
        use_cross_memory: bool = True,
        use_routing: bool = True,
    ):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.static_dim = static_dim
        self.latent_dim = latent_dim
        self.fast_hidden = fast_hidden
        self.slow_hidden = slow_hidden
        self.use_partition_gate = use_partition_gate
        self.use_cross_memory = use_cross_memory
        self.use_routing = use_routing

        self.encoder = nn.Linear(dynamic_dim + static_dim, latent_dim)
        self.activation = nn.Tanh() if activation == "tanh" else nn.ReLU()
        self.partition_gate = LatentPartitionGate(latent_dim, fast_hidden, slow_hidden)
        self.memories = FastSlowMemory(latent_dim, fast_hidden, slow_hidden)
        self.cross_memory = ControlledCrossMemory(latent_dim, fast_hidden, slow_hidden)
        self.fast_response_head = nn.Linear(fast_hidden, 1)
        self.slow_output_head = nn.Linear(slow_hidden, 1)
        self.recession_head = nn.Linear(slow_hidden, 1)
        self.routing = DynamicCausalRouting(fast_hidden, slow_hidden, latent_dim, static_dim, max_lag, dt)

    def forward(self, x: torch.Tensor, static: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, t, _ = x.shape
        static_seq = static[:, None, :].expand(-1, t, -1)
        z = self.activation(self.encoder(torch.cat([x, static_seq], dim=-1)))

        h_f = x.new_zeros((b, self.fast_hidden)); c_f = x.new_zeros((b, self.fast_hidden))
        h_s = x.new_zeros((b, self.slow_hidden)); c_s = x.new_zeros((b, self.slow_hidden))
        hfs, hss, alphas, gsf, gfs, recession_k = [], [], [], [], [], []

        for i in range(t):
            recession_k.append(torch.sigmoid(self.recession_head(h_s)).squeeze(-1))
            if self.use_partition_gate:
                alpha, u_f, u_s = self.partition_gate(z[:, i], h_f, h_s)
            else:
                alpha = z[:, i].new_full(z[:, i].shape, 0.5)
                u_f = z[:, i]
                u_s = z[:, i]
            (hf_pre, c_f), (hs_pre, c_s) = self.memories(u_f, u_s, (h_f, c_f), (h_s, c_s))
            if self.use_cross_memory:
                h_f, h_s, gamma_sf, gamma_fs = self.cross_memory(hf_pre, hs_pre, z[:, i])
            else:
                h_f, h_s = hf_pre, hs_pre
                gamma_sf = hf_pre.new_zeros(hf_pre.shape)
                gamma_fs = hs_pre.new_zeros(hs_pre.shape)
            hfs.append(h_f); hss.append(h_s); alphas.append(alpha); gsf.append(gamma_sf); gfs.append(gamma_fs)

        h_fast = torch.stack(hfs, dim=1)
        h_slow = torch.stack(hss, dim=1)
        alpha = torch.stack(alphas, dim=1)
        rapid = F.softplus(self.fast_response_head(h_fast).squeeze(-1))
        q_slow = F.softplus(self.slow_output_head(h_slow).squeeze(-1))
        if self.use_routing:
            q_fast, routing_probs, expected_delay = self.routing(rapid, h_fast, h_slow, z, static)
        else:
            q_fast = rapid
            routing_probs = rapid.new_zeros((b, t, self.routing.max_lag + 1))
            routing_probs[..., 0] = 1.0
            expected_delay = rapid.new_zeros((b, t))
        q_total = q_fast + q_slow
        return {
            "q_total": q_total,
            "q_fast": q_fast,
            "q_slow": q_slow,
            "rapid_response": rapid,
            "routing_probs": routing_probs,
            "expected_delay": expected_delay,
            "partition_alpha": alpha,
            "h_fast": h_fast,
            "h_slow": h_slow,
            "gamma_slow_to_fast": torch.stack(gsf, dim=1),
            "gamma_fast_to_slow": torch.stack(gfs, dim=1),
            "recession_k": torch.stack(recession_k, dim=1),
            "z": z,
        }

    def parameter_groups(self) -> Dict[str, Iterable[nn.Parameter]]:
        """Paper partition: theta_sh, theta_F, theta_S, theta_R (Section 14)."""
        shared_modules = [self.encoder, self.partition_gate, self.cross_memory]
        fast_modules = [self.memories.fast_cell, self.fast_response_head]
        slow_modules = [self.memories.slow_cell, self.slow_output_head, self.recession_head]
        routing_modules = [self.routing]
        def params(modules):
            seen = set(); out = []
            for m in modules:
                for p in m.parameters():
                    if p.requires_grad and id(p) not in seen:
                        seen.add(id(p)); out.append(p)
            return out
        return {"shared": params(shared_modules), "fast": params(fast_modules), "slow": params(slow_modules), "routing": params(routing_modules)}
