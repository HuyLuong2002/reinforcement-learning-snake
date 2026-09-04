"""Entry point — mở cửa sổ game, agent tự chơi sau khi bấm Bắt đầu.

Chạy:
  python app.py
  python app.py --agent sarsa
  python app.py --agent q_learning
Trong menu: chọn SARSA hoặc Q-Learning, tick Load model, chọn lần train, chọn màn.
"""

from __future__ import annotations

import argparse

import numpy as np

from agents.registry import AGENT_REGISTRY, get_agent_spec
from common.env_hyperparameters import (
    PLAY_GRID_CHOICES,
    SnakeEnvHyperparameters,
    parse_play_grid,
)
from common.game_window import AgentMenuItem, GameWindow
from common.run_store import ModelRun, list_model_runs
from common.snake_env import SnakeEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snake game — agent auto-play")
    parser.add_argument(
        "--agent",
        type=str,
        default="sarsa",
        choices=["sarsa", "q_learning"],
        help="Thuật toán RL (mặc định: sarsa)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Cố định seed để chơi lại cùng ván. Mặc định: mỗi ván một seed ngẫu nhiên.",
    )
    parser.add_argument(
        "--grid",
        type=str,
        default=None,
        choices=list(PLAY_GRID_CHOICES),
        help="Bỏ menu, vào thẳng màn 10x10, 15x20 hoặc 30x30",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Chơi random, không load model (menu mặc định cũng không tick)",
    )
    return parser.parse_args()


def build_env(seed: int | None, env_cfg: SnakeEnvHyperparameters | None = None) -> SnakeEnv:
    cfg = env_cfg or SnakeEnvHyperparameters()
    rng = np.random.default_rng(seed)
    return cfg.create_env(rng)


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

    env = build_env(args.seed, env_cfg.for_shape(env_cfg.width, env_cfg.height))
    model_hint = f"output/{args.agent}/YYYY-MM-DD_HH-MM-SS/agent.pkl"
    available_agents = [
        AgentMenuItem(item.name, item.label, item.train_command)
        for item in AGENT_REGISTRY.values()
    ]

    def make_env(width: int, height: int) -> SnakeEnv:
        return build_env(args.seed, env_cfg.for_shape(width, height))

    def list_runs(agent_name: str) -> list[ModelRun]:
        return list_model_runs(agent_name)

    def load_run(agent_name: str, run: ModelRun):
        agent_spec = get_agent_spec(agent_name)
        print(f"Loaded {agent_spec.label}: {run.model_path}")
        return agent_spec.load_fn(run.model_path)

    window = GameWindow(
        env=env,
        agent=None,
        title=f"{spec.label} Snake",
        agent_label=spec.label,
        model_missing=False,
        model_path_hint=model_hint,
        train_command_hint=spec.train_command,
        make_env=make_env,
        replay_seed=args.seed,
        list_runs_fn=list_runs,
        load_run_fn=load_run,
        use_agent=False,
        agent_name=args.agent,
        available_agents=available_agents,
    )
    if args.grid is not None:
        width, height = parse_play_grid(args.grid)
        if not args.random:
            window.set_use_agent(True)
        window.apply_level(width, height)
    window.run()


if __name__ == "__main__":
    main()
