"""Baseline random — dùng so sánh với agent sau khi train."""

from __future__ import annotations

import numpy as np


def choose_random_action(n_actions: int, rng: np.random.Generator) -> int:
    return int(rng.integers(0, n_actions))


def run_random_episode(env, rng: np.random.Generator) -> dict:
    """Chạy 1 episode random — in ra cuối train để so sánh baseline."""
    obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
    total_reward = 0.0
    steps = 0
    score = 0

    while True:
        action = choose_random_action(env.n_actions, rng)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        score = info.get("score", score)
        if terminated or truncated:
            break

    return {
        "total_reward": total_reward,
        "steps": steps,
        "score": score,
        "terminated": terminated,
    }
