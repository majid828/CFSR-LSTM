import torch
from torch import nn
import torch.nn.functional as F


class StandardLSTM(nn.Module):
    def __init__(self, dynamic_dim: int, static_dim: int, hidden_dim: int = 64, num_layers: int = 1):
        super().__init__()
        self.input_dim = dynamic_dim + static_dim
        self.lstm = nn.LSTM(self.input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, static):
        s = static[:, None, :].expand(-1, x.size(1), -1)
        h, _ = self.lstm(torch.cat([x, s], dim=-1))
        q = F.softplus(self.head(h).squeeze(-1))
        return {"q_total": q, "hidden": h}
