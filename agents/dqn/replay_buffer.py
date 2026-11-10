"""Replay buffer vòng (numpy) — lưu transition, lấy mẫu ngẫu nhiên phá tương quan."""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, n_features: int, rng: np.random.Generator) -> None:
        self.capacity = int(capacity)
        self.rng = rng
        self.states = np.zeros((capacity, n_features), dtype=np.float32)
        self.next_states = np.zeros((capacity, n_features), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(self, s, a: int, r: float, s2, done: bool) -> None:
        i = self.pos
        self.states[i] = s
        self.actions[i] = a
        self.rewards[i] = r
        self.next_states[i] = s2
        self.dones[i] = float(done)
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = self.rng.integers(0, self.size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )
