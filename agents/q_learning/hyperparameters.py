"""Siêu tham số Q-learning — chỉnh tại đây trước khi train.

Cách dùng:
  1. Sửa giá trị mặc định bên dưới
  2. Chạy: python -m agents.q_learning.train
  3. Hoặc override tạm: python -m agents.q_learning.train --episodes 1000 --grid 10x10
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from common.env_hyperparameters import (
    DEFAULT_TRAIN_LEVEL,
    SnakeEnvHyperparameters,
    parse_play_grid,
)

AGENT_NAME = "q_learning"


@dataclass
class QLearningHyperparameters:
    """Tham số thuật toán Q-learning."""

    alpha: float = 0.1  # tốc độ học Q
    gamma: float = 0.99  # hệ số chiết khấu — coi trọng phần thưởng tương lai
    epsilon_start: float = 1.0  # explore 100% lúc đầu
    epsilon_min: float = 0.01  # explore tối thiểu
    # Nhân mỗi episode. 0.9998 → chạm đáy ~ep 23000 (cùng SARSA).
    epsilon_decay: float = 0.9998


@dataclass
class TrainingHyperparameters:
    """Tham số vòng lặp train và lưu model."""

    episodes: int = 30000  # ← số episode train (chỉnh ở đây)
    seed: int = 42
    log_every: int = 100  # in log ra console mỗi N episode
    eval_every: int = 1000  # đánh giá greedy Q mỗi N episode (không heuristic)
    eval_episodes: int = 100  # số game khi eval → lấy mean score
    save_best_checkpoint: bool = True  # lưu agent_best.pkl khi eval tốt hơn
    export_best_as_final: bool = True  # agent.pkl = best sau train
    # None = tự tạo output/q_learning/YYYY-MM-DD_HH-MM-SS mỗi lần train.
    output_dir: Path | None = None


def _default_env() -> SnakeEnvHyperparameters:
    """Mặc định train trên 15×20; max_steps scale theo diện tích."""
    return SnakeEnvHyperparameters().for_shape(*DEFAULT_TRAIN_LEVEL)


@dataclass
class Hyperparameters:
    """Gộp tham số env + Q-learning + training."""

    env: SnakeEnvHyperparameters = field(default_factory=_default_env)
    q_learning: QLearningHyperparameters = field(default_factory=QLearningHyperparameters)
    training: TrainingHyperparameters = field(default_factory=TrainingHyperparameters)

    def to_dict(self) -> dict[str, Any]:
        """Chuyển config sang dict — dùng khi ghi log/metadata."""
        data = asdict(self)
        data["training"]["output_dir"] = (
            str(self.training.output_dir) if self.training.output_dir else None
        )
        return data


DEFAULT_HYPERPARAMETERS = Hyperparameters()


def build_config(
    *,
    episodes: int | None = None,
    seed: int | None = None,
    output_dir: Path | str | None = None,
    grid: str | None = None,
) -> Hyperparameters:
    """Tạo config: lấy DEFAULT + override từ CLI nếu có."""
    env = DEFAULT_HYPERPARAMETERS.env
    training = replace(DEFAULT_HYPERPARAMETERS.training)
    updates: dict[str, Any] = {}
    if episodes is not None:
        updates["episodes"] = episodes
    if seed is not None:
        updates["seed"] = seed
    if output_dir is not None:
        updates["output_dir"] = Path(output_dir)
    if updates:
        training = replace(training, **updates)
    if grid is not None:
        width, height = parse_play_grid(grid)
        env = env.for_shape(width, height)
    return replace(DEFAULT_HYPERPARAMETERS, env=env, training=training)
