import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Iterable


INPUT_SIZE: int = 128
HIDDEN_DIMS: Iterable[int] = (64, 32, 16)
OUTPUT_SIZE: int = 1


class LynxMLModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_DIMS[0]),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMS[0], HIDDEN_DIMS[1]),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMS[1], HIDDEN_DIMS[2]),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIMS[2], OUTPUT_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)