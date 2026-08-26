"""Chạy agent SARSA nhiều ván và phân tích nguyên nhân thua.

Dùng để debug sau train — xem agent chết vì gì (death vs max_steps),
còn bao nhiêu ô trống, và vài bước cuối trước khi kết thúc.

Chạy:
  python -m agents.sarsa.analyze_games --games 10
  python -m agents.sarsa.analyze_games --games 50 --model output/sarsa/training/agent.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from agents.sarsa.agent import SarsaAgent
from agents.sarsa.hyperparameters import DEFAULT_HYPERPARAMETERS
from common.snake_env import SnakeEnv

ACTION_NAMES = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}


def build_env(seed: int) -> SnakeEnv:
    """Tạo env với cùng tham số trong hyperparameters.py."""
    cfg = DEFAULT_HYPERPARAMETERS.env
    return SnakeEnv(
        grid_size=cfg.grid_size,
        max_steps=cfg.max_steps,
        max_steps_without_food=cfg.max_steps_without_food,
        reward_food=cfg.reward_food,
        reward_death=cfg.reward_death,
        reward_step=cfg.reward_step,
        reward_closer=cfg.reward_closer,
        death_penalty_per_score=cfg.death_penalty_per_score,
        rng=np.random.default_rng(seed),
    )


def run_episode(agent: SarsaAgent, seed: int, trace_last: int = 6) -> dict:
    """
    Chạy 1 ván greedy, ghi lại N bước cuối.

    Trả về score, end_reason, free_cells lúc chết, và last_moves để phân tích.
    """
    env = build_env(seed)
    obs, _ = env.reset(seed=seed)
    history: list[dict] = []
    total_reward = 0.0

    while True:
        action = agent.greedy_policy_action(obs)
        assert env.state is not None
        head = env.state.snake[0]
        food = env.state.food
        dist = abs(head[0] - food[0]) + abs(head[1] - food[1])
        free = env.grid_size**2 - len(env.state.snake)

        history.append(
            {
                "action": ACTION_NAMES[action],
                "score": env.state.score,
                "dist_food": dist,
                "free_cells": free,
                "head": head,
                # BFS mỗi hướng thẳng/trái/phải: 0=đâm ngay, 1=kẹt, 2=chật, 3=thoáng
                "safety": obs[2:5].tolist(),
            }
        )

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            return {
                "seed": seed,
                "score": info["score"],
                "steps": info["steps"],
                "end": info["end_reason"],
                "snake_len": len(env.state.snake) if env.state else 0,
                "free_cells": free,
                "reward": total_reward,
                "last_moves": history[-trace_last:],
            }

    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phân tích nhiều ván greedy của SARSA")
    parser.add_argument("--games", type=int, default=10, help="Số ván chạy")
    parser.add_argument(
        "--model",
        type=str,
        default="output/sarsa/training/agent.pkl",
        help="Đường dẫn file model .pkl",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    agent = SarsaAgent.load(model_path)
    cfg = DEFAULT_HYPERPARAMETERS.env

    print(f"Model: {model_path}")
    print(f"max_score={SnakeEnv.max_score(cfg.grid_size)} max_steps={cfg.max_steps} grid={cfg.grid_size}")
    print(f"=== {args.games} games (greedy) ===\n")

    results = [run_episode(agent, seed=i) for i in range(args.games)]

    for i, r in enumerate(results, 1):
        print(
            f"Game {i:2d}: score={r['score']:2d} steps={r['steps']:3d} "
            f"end={r['end']:9s} free_cells={r['free_cells']:2d} reward={r['reward']:7.1f}"
        )

    scores = [r["score"] for r in results]
    ends: dict[str, int] = {}
    for r in results:
        ends[r["end"]] = ends.get(r["end"], 0) + 1

    print("\n--- Summary ---")
    print(f"mean_score={np.mean(scores):.1f}  max={max(scores)}  min={min(scores)}  std={np.std(scores):.1f}")
    print(f"end_reasons={ends}")

    # Phân tích riêng từng loại kết thúc
    deaths = [r for r in results if r["end"] == "death"]
    truncs = [r for r in results if r["end"] == "max_steps"]
    starved = [r for r in results if r["end"] == "no_food"]
    if deaths:
        print(
            f"death: count={len(deaths)} avg_score={np.mean([r['score'] for r in deaths]):.1f} "
            f"avg_steps={np.mean([r['steps'] for r in deaths]):.1f} "
            f"avg_free_cells={np.mean([r['free_cells'] for r in deaths]):.1f}"
        )
    if truncs:
        print(f"max_steps: count={len(truncs)} avg_score={np.mean([r['score'] for r in truncs]):.1f}")
    if starved:
        print(f"no_food: count={len(starved)} avg_score={np.mean([r['score'] for r in starved]):.1f}")

    # In chi tiết 3 ván score cao nhất — xem pattern bước cuối
    interesting = sorted(results, key=lambda x: x["score"], reverse=True)[:3]
    print("\n--- Top 3 games (last moves before end) ---")
    for r in interesting:
        print(f"\nGame seed={r['seed']} score={r['score']} end={r['end']} steps={r['steps']}")
        for m in r["last_moves"]:
            print(
                f"  {m['action']:5s} score={m['score']} dist={m['dist_food']} free={m['free_cells']} "
                f"safety={m['safety']} head={m['head']}"
            )


if __name__ == "__main__":
    main()
