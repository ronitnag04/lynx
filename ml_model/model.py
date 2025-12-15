import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Iterable

class LynxMLModel(nn.Module):
    def __init__(self, input_size: int, hidden_dims: Iterable[int], output_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.ReLU(),
            nn.Linear(hidden_dims[2], output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)