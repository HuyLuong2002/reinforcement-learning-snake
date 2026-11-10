"""Siêu tham số DQN — chỉnh tại đây trước khi train.

  python -m agents.dqn.train
  python -m agents.dqn.train --episodes 2000 --grid 10x10
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

AGENT_NAME = "dqn"


@dataclass
class DQNHyperparameters:
    lr: float = 5e-4  # Adam learning rate
    gamma: float = 0.99
    hidden: int = 256  # số neuron mỗi lớp ẩn
    batch_size: int = 128
    buffer_size: int = 200_000
    learning_starts: int = 5_000  # số bước random trước khi bắt đầu học
    train_freq: int = 4  # học 1 batch mỗi N bước env
    target_update_every: int = 2_000  # copy online → target mỗi N bước env
    double_dqn: bool = True  # chọn a' bằng online net, đánh giá bằng target net
    grad_clip: float = 10.0
    reward_scale: float = 0.1  # reward env (±10..±55) → khoảng ±1..±5 cho mạng
    epsilon_start: float = 1.0
    epsilon_min: float = 0.02
    # Epsilon giảm tuyến tính THEO BƯỚC (không theo episode như tabular).
    epsilon_decay_steps: int = 300_000


@dataclass
class TrainingHyperparameters:
    episodes: int = 5000
    seed: int = 42
    log_every: int = 50
    eval_every: int = 250
    eval_episodes: int = 30
    save_best_checkpoint: bool = True
    export_best_as_final: bool = True
    device: str = "auto"  # auto | cpu | cuda
    output_dir: Path | None = None


def _default_env() -> SnakeEnvHyperparameters:
    return SnakeEnvHyperparameters().for_shape(*DEFAULT_TRAIN_LEVEL)


@dataclass
class Hyperparameters:
    env: SnakeEnvHyperparameters = field(default_factory=_default_env)
    dqn: DQNHyperparameters = field(default_factory=DQNHyperparameters)
    training: TrainingHyperparameters = field(default_factory=TrainingHyperparameters)

    def to_dict(self) -> dict[str, Any]:
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
    device: str | None = None,
) -> Hyperparameters:
    env = DEFAULT_HYPERPARAMETERS.env
    updates: dict[str, Any] = {}
    if episodes is not None:
        updates["episodes"] = episodes
    if seed is not None:
        updates["seed"] = seed
    if output_dir is not None:
        updates["output_dir"] = Path(output_dir)
    if device is not None:
        updates["device"] = device
    training = replace(DEFAULT_HYPERPARAMETERS.training, **updates)
    if grid is not None:
        env = env.for_shape(*parse_play_grid(grid))
    return replace(DEFAULT_HYPERPARAMETERS, env=env, training=training)
