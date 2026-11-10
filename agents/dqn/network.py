"""Mạng Q: MLP nhận feature vector → Q-value cho 3 action tương đối.

Dueling head: Q(s,a) = V(s) + A(s,a) - mean_a A(s,a) — ổn định hơn khi nhiều
state mà action nào cũng gần như nhau (đi giữa bàn trống).
"""

from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, n_inputs: int, n_actions: int, hidden: int = 256) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden, 1)
        self.advantage = nn.Linear(hidden, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        adv = self.advantage(h)
        return self.value(h) + adv - adv.mean(dim=1, keepdim=True)
