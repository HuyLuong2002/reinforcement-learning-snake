"""Chọn action khi train / eval / chơi — luôn từ policy agent (Q-table).

Không ghi đè bằng A* hay heuristic. 180° bị cấm vì env cũng biến thành đi thẳng,
nên agent chỉ học 3 hướng thật sự thực hiện được.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from common.snake_env import OPPOSITE, SnakeEnv


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
    """ε-greedy / greedy Q. Cấm 180° — cùng luật lúc train và lúc chơi."""
    assert env.state is not None
    forbidden = OPPOSITE[env.state.direction]
    return agent.choose_action(obs, greedy=greedy, forbidden=forbidden)
