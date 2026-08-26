"""Tham số môi trường Snake — dùng chung cho mọi agent (SARSA, Q-learning, ...)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SnakeEnvHyperparameters:
    """Tham số môi trường game."""

    grid_size: int = 10
    # Trần số bước/episode. Cần đủ lớn cho score cao:
    # ~15-20 bước/quả khi rắn dài → score 50 cần cỡ 800-1000 bước.
    max_steps: int = 2000
    # Cắt episode nếu đi bấy nhiêu bước mà không ăn được gì.
    # Ngăn rắn đi vòng vô tận làm train chậm, mà vẫn cho phép đường đi dài.
    max_steps_without_food: int = 100
    reward_food: float = 10.0  # thưởng khi ăn thức ăn
    reward_death: float = -20.0  # phạt cơ bản khi chết (env cộng thêm theo score)
    reward_step: float = -0.001  # phạt mỗi bước → khuyến khích đi hiệu quả
    reward_closer: float = 0.02  # thưởng khi tiến gần thức ăn (reward shaping)
    death_penalty_per_score: float = 0.5  # chết ở score 20 → phạt thêm -10
