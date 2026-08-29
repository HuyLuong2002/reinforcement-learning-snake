"""Siêu tham số Q-learning — chỉnh tại đây trước khi train.

Cách dùng:
  1. Sửa giá trị mặc định bên dưới
  2. Chạy: python -m agents.q_learning.train
  3. Hoặc override tạm: python -m agents.q_learning.train --episodes 1000
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from common.env_hyperparameters import SnakeEnvHyperparameters

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_NAME = "q_learning"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / AGENT_NAME / "training"


@dataclass
class QLearningHyperparameters:
    """Tham số thuật toán Q-learning."""

    alpha: float = 0.1  # tốc độ học Q
    gamma: float = 0.99  # hệ số chiết khấu — coi trọng phần thưởng tương lai
    epsilon_start: float = 1.0  # explore 100% lúc đầu
    epsilon_min: float = 0.01  # explore tối thiểu
    # Nhân mỗi episode. 0.9997 → chạm đáy ở ~ep 15000, đủ lâu để khám phá
    # các thế cờ khi rắn đã dài (0.999 chạm đáy quá sớm, ~ep 4600).
    epsilon_decay: float = 0.9997


@dataclass
class TrainingHyperparameters:
    """Tham số vòng lặp train và lưu model."""

    episodes: int = 40000  # ← số episode train (chỉnh ở đây)
    seed: int = 42
    log_every: int = 100  # in log ra console mỗi N episode
    eval_every: int = 1000  # đánh giá greedy mỗi N episode
    eval_episodes: int = 100  # số game khi eval → lấy mean score
    save_best_checkpoint: bool = True  # lưu agent_best.pkl khi eval tốt hơn
    export_best_as_final: bool = True  # agent.pkl = best sau train
    output_dir: Path = DEFAULT_OUTPUT_DIR


@dataclass
class Hyperparameters:
    """Gộp tham số env + Q-learning + training."""

    env: SnakeEnvHyperparameters = field(default_factory=SnakeEnvHyperparameters)
    q_learning: QLearningHyperparameters = field(default_factory=QLearningHyperparameters)
    training: TrainingHyperparameters = field(default_factory=TrainingHyperparameters)

    def to_dict(self) -> dict[str, Any]:
        """Chuyển config sang dict — dùng khi ghi log/metadata."""
        data = asdict(self)
        data["training"]["output_dir"] = str(self.training.output_dir)
        return data


DEFAULT_HYPERPARAMETERS = Hyperparameters()


def build_config(
    *,
    episodes: int | None = None,
    seed: int | None = None,
    output_dir: Path | str | None = None,
) -> Hyperparameters:
    """Tạo config: lấy DEFAULT + override từ CLI nếu có."""
    training = DEFAULT_HYPERPARAMETERS.training
    updates: dict[str, Any] = {}
    if episodes is not None:
        updates["episodes"] = episodes
    if seed is not None:
        updates["seed"] = seed
    if output_dir is not None:
        updates["output_dir"] = Path(output_dir)
    if updates:
        training = replace(training, **updates)
    return replace(DEFAULT_HYPERPARAMETERS, training=training)
