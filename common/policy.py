"""Chọn action khi train / eval / chơi — luôn từ policy agent (Q-table).

Không gian hành động là 3 hướng tương đối (thẳng / trái / phải).
Không ghi đè bằng A* hay heuristic.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from common.snake_env import SnakeEnv


class ActionAgent(Protocol):
    def choose_action(
        self,
        state: np.ndarray,
        greedy: bool = False,
        forbidden: int | None = None,
    ) -> int: ...


def select_action(
    env: SnakeEnv,
    agent: ActionAgent,
    obs: np.ndarray,
    *,
    greedy: bool = False,
) -> int:
    """ε-greedy / greedy Q trên 3 hướng hợp lệ."""
    return agent.choose_action(obs, greedy=greedy)
