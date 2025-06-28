from typing import Tuple
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, inp_dim: int, n_classes: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(inp_dim, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.mlp(x)
