from typing import Iterable
import torch
import torch.nn as nn


class SimpleClassifier(nn.Module):
def __init__(self, input_dim: int, hidden: int = 32):
super().__init__()
self.net = nn.Sequential(
nn.Linear(input_dim, hidden),
nn.ReLU(),
nn.Linear(hidden, 1)
)


def forward(self, x: torch.Tensor) -> torch.Tensor:
return self.net(x).squeeze(1) # logits




def count_parameters(model: nn.Module) -> int:
return sum(p.numel() for p in model.parameters() if p.requires_grad)