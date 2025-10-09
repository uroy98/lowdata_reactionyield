# low-data hybrid model with uncertainty (ensemble neural net)
# Here’s a small placeholder (I will expand this later in November)

import torch
import torch.nn as nn

class HybridModel(nn.Module):
    """Numeric + fingerprint fusion model for low-data yield prediction."""
    def __init__(self, num_numeric, fp_dim=1024, hidden=256):
        super().__init__()
        self.numeric = nn.Sequential(
            nn.Linear(num_numeric, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.fp = nn.Sequential(
            nn.Linear(fp_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.out = nn.Linear(2 * hidden, 1)

    def forward(self, x_num, x_fp):
        n = self.numeric(x_num)
        f = self.fp(x_fp)
        x = torch.cat([n, f], dim=1)
        return self.out(x)
