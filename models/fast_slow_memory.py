from torch import nn


class FastSlowMemory(nn.Module):
    """Two independent recurrent memories as in Sections 5 and 6."""

    def __init__(self, latent_dim: int, fast_hidden: int, slow_hidden: int):
        super().__init__()
        self.fast_cell = nn.LSTMCell(latent_dim, fast_hidden)
        self.slow_cell = nn.LSTMCell(latent_dim, slow_hidden)

    def forward(self, u_fast, u_slow, fast_state, slow_state):
        h_f, c_f = self.fast_cell(u_fast, fast_state)
        h_s, c_s = self.slow_cell(u_slow, slow_state)
        return (h_f, c_f), (h_s, c_s)
