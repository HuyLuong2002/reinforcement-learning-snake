"""Vẽ biểu đồ đánh giá quá trình train."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _moving_average(values: list[float], window: int) -> np.ndarray:
    if not values:
        return np.array([])
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def save_training_plots(
    episodes: list[int],
    scores: list[int],
    rewards: list[float],
    epsilons: list[float],
    eval_episodes: list[int],
    eval_scores: list[float],
    output_dir: Path,
    ma_window: int = 50,
    agent_label: str = "Agent",
) -> list[Path]:
    """Lưu các biểu đồ training vào output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards, alpha=0.25, color="#4a90d9", label="Reward/episode")
    ma_rewards = _moving_average(rewards, ma_window)
    if len(ma_rewards) > 0:
        if len(rewards) >= ma_window:
            ma_x = episodes[ma_window - 1 :]
        else:
            ma_x = episodes[: len(ma_rewards)]
        ax.plot(ma_x, ma_rewards, color="#1f5fbf", linewidth=2, label=f"MA({ma_window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total reward")
    ax.set_title("Learning curve — Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = output_dir / "learning_curve_reward.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, scores, alpha=0.25, color="#5ad282", label="Score/episode")
    ma_scores = _moving_average([float(s) for s in scores], ma_window)
    if len(ma_scores) > 0:
        if len(scores) >= ma_window:
            ma_x = episodes[ma_window - 1 :]
        else:
            ma_x = episodes[: len(ma_scores)]
        ax.plot(ma_x, ma_scores, color="#2d8a50", linewidth=2, label=f"MA({ma_window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Score (food eaten)")
    ax.set_title("Learning curve — Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = output_dir / "learning_curve_score.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved.append(path)

    if eval_episodes:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(eval_episodes, eval_scores, marker="o", color="#e67e22", linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Mean eval score")
        ax.set_title("Evaluation score (greedy policy)")
        ax.grid(True, alpha=0.3)
        path = output_dir / "eval_score.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        saved.append(path)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(episodes, epsilons, color="#9b59b6", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title("Exploration rate (epsilon decay)")
    ax.grid(True, alpha=0.3)
    path = output_dir / "epsilon_decay.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved.append(path)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(episodes, rewards, alpha=0.2, color="#4a90d9")
    if len(ma_rewards := _moving_average(rewards, ma_window)) > 0:
        ma_x = episodes[ma_window - 1 :] if len(rewards) >= ma_window else episodes[: len(ma_rewards)]
        axes[0, 0].plot(ma_x, ma_rewards, color="#1f5fbf", linewidth=2)
    axes[0, 0].set_title("Reward")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(episodes, scores, alpha=0.2, color="#5ad282")
    if len(ma_scores := _moving_average([float(s) for s in scores], ma_window)) > 0:
        ma_x = episodes[ma_window - 1 :] if len(scores) >= ma_window else episodes[: len(ma_scores)]
        axes[0, 1].plot(ma_x, ma_scores, color="#2d8a50", linewidth=2)
    axes[0, 1].set_title("Score")
    axes[0, 1].grid(True, alpha=0.3)

    if eval_episodes:
        axes[1, 0].plot(eval_episodes, eval_scores, marker="o", color="#e67e22")
    axes[1, 0].set_title("Eval score")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(episodes, epsilons, color="#9b59b6")
    axes[1, 1].set_title("Epsilon")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"{agent_label} Snake — Training dashboard", fontsize=14)
    fig.tight_layout()
    path = output_dir / "training_dashboard.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    saved.append(path)

    return saved
