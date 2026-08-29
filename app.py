"""Entry point — mở cửa sổ game, agent tự chơi sau khi bấm Bắt đầu.

Chạy:
  python app.py
  python app.py --agent sarsa
  python app.py --agent q_learning
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np

from agents.registry import get_agent_spec, load_agent
from common.env_hyperparameters import SnakeEnvHyperparameters
from common.game_window import GameWindow
from common.snake_env import SnakeEnv

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snake game — agent auto-play")
    parser.add_argument(
        "--agent",
        type=str,
        default="sarsa",
        choices=["sarsa", "q_learning"],
        help="Thuật toán RL (mặc định: sarsa)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_env(seed: int, env_cfg: SnakeEnvHyperparameters | None = None) -> SnakeEnv:
    cfg = env_cfg or SnakeEnvHyperparameters()
    rng = np.random.default_rng(seed)
    return SnakeEnv(
        grid_size=cfg.grid_size,
        max_steps=cfg.max_steps,
        max_steps_without_food=cfg.max_steps_without_food,
        reward_food=cfg.reward_food,
        reward_death=cfg.reward_death,
        reward_step=cfg.reward_step,
        reward_closer=cfg.reward_closer,
        death_penalty_per_score=cfg.death_penalty_per_score,
        rng=rng,
    )


def warn_missing_model(spec) -> None:
    paths = [spec.output_dir / "agent.pkl", spec.output_dir / "agent_best.pkl"]
    msg = (
        f"Chưa tìm thấy model {spec.label} tại:\n"
        + "\n".join(f"  - {p}" for p in paths)
        + f"\nChạy `{spec.train_command}` để train trước. Game sẽ dùng random policy."
    )
    warnings.warn(msg, UserWarning, stacklevel=2)
    print(f"WARNING: Model not found. Run `{spec.train_command}` first. Using random policy.")


def main() -> None:
    args = parse_args()
    spec = get_agent_spec(args.agent)

    if args.agent == "sarsa":
        from agents.sarsa.hyperparameters import DEFAULT_HYPERPARAMETERS

        env_cfg = DEFAULT_HYPERPARAMETERS.env
    elif args.agent == "q_learning":
        from agents.q_learning.hyperparameters import DEFAULT_HYPERPARAMETERS

        env_cfg = DEFAULT_HYPERPARAMETERS.env
    else:
        env_cfg = SnakeEnvHyperparameters()

    env = build_env(args.seed, env_cfg)
    agent, spec, model_missing = load_agent(args.agent)

    if model_missing:
        warn_missing_model(spec)

    model_hint = str((spec.output_dir / "agent.pkl").relative_to(PROJECT_ROOT))

    window = GameWindow(
        env=env,
        agent=agent,
        title=f"{spec.label} Snake",
        agent_label=spec.label,
        model_missing=model_missing,
        model_path_hint=model_hint,
        train_command_hint=spec.train_command,
    )
    window.run()


if __name__ == "__main__":
    main()
