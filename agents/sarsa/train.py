"""Train agent SARSA tabular trên Snake.

Output (trong output/sarsa/training/):
  - training_log.csv
  - agent_best.pkl
  - agent.pkl
  - agent_last.pkl
  - *.png
  - agent_metadata.json
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from agents.sarsa.agent import SarsaAgent
from agents.sarsa.hyperparameters import DEFAULT_HYPERPARAMETERS, Hyperparameters, build_config
from common.random_policy import run_random_episode
from common.snake_env import SnakeEnv
from common.training_plots import save_training_plots


def build_env(config: Hyperparameters, rng: np.random.Generator) -> SnakeEnv:
    env_cfg = config.env
    return SnakeEnv(
        grid_size=env_cfg.grid_size,
        max_steps=env_cfg.max_steps,
        max_steps_without_food=env_cfg.max_steps_without_food,
        reward_food=env_cfg.reward_food,
        reward_death=env_cfg.reward_death,
        reward_step=env_cfg.reward_step,
        reward_closer=env_cfg.reward_closer,
        death_penalty_per_score=env_cfg.death_penalty_per_score,
        rng=rng,
    )


def build_agent(config: Hyperparameters, env: SnakeEnv, rng: np.random.Generator) -> SarsaAgent:
    sarsa_cfg = config.sarsa
    return SarsaAgent(
        n_state_dims=env.n_state_dims(),
        n_actions=env.n_actions,
        alpha=sarsa_cfg.alpha,
        gamma=sarsa_cfg.gamma,
        epsilon=sarsa_cfg.epsilon_start,
        epsilon_min=sarsa_cfg.epsilon_min,
        epsilon_decay=sarsa_cfg.epsilon_decay,
        rng=rng,
    )


def evaluate_agent(env: SnakeEnv, agent: SarsaAgent, n_episodes: int, rng: np.random.Generator) -> dict:
    scores: list[int] = []
    rewards: list[float] = []
    lengths: list[int] = []

    for _ in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        total_reward = 0.0
        steps = 0
        score = 0

        while True:
            action = agent.choose_action(obs, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            score = info.get("score", score)
            if terminated or truncated:
                break

        scores.append(score)
        rewards.append(total_reward)
        lengths.append(steps)

    return {
        "mean_score": float(np.mean(scores)),
        "mean_reward": float(np.mean(rewards)),
        "mean_length": float(np.mean(lengths)),
    }


def train_sarsa(config: Hyperparameters = DEFAULT_HYPERPARAMETERS) -> tuple[SarsaAgent, SnakeEnv]:
    train_cfg = config.training
    output_dir = Path(train_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Training SARSA: episodes={train_cfg.episodes}, seed={train_cfg.seed}, "
        f"eval_every={train_cfg.eval_every}, output={output_dir}"
    )

    rng = np.random.default_rng(train_cfg.seed)
    env = build_env(config, rng)
    agent = build_agent(config, env, rng)

    log_path = output_dir / "training_log.csv"
    best_score = -1.0
    best_episode = 0
    best_path = output_dir / "agent_best.pkl"
    final_path = output_dir / "agent.pkl"
    last_path = output_dir / "agent_last.pkl"

    history_episodes: list[int] = []
    history_scores: list[int] = []
    history_rewards: list[float] = []
    history_epsilons: list[float] = []
    eval_episodes: list[int] = []
    eval_scores: list[float] = []

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["episode", "score", "reward", "steps", "epsilon", "eval_mean_score"],
        )
        writer.writeheader()

        for episode in range(1, train_cfg.episodes + 1):
            obs, _ = env.reset()
            action = agent.choose_action(obs)
            total_reward = 0.0
            steps = 0
            score = 0

            while True:
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
                score = info.get("score", score)

                next_action = agent.choose_action(next_obs) if not done else 0
                agent.update(obs, action, reward, next_obs, next_action, done)
                obs, action = next_obs, next_action

                if done:
                    break

            agent.decay_epsilon()
            eval_mean_score = ""

            history_episodes.append(episode)
            history_scores.append(score)
            history_rewards.append(total_reward)
            history_epsilons.append(agent.epsilon)

            if episode % train_cfg.eval_every == 0:
                metrics = evaluate_agent(env, agent, train_cfg.eval_episodes, rng)
                eval_mean_score = metrics["mean_score"]
                eval_episodes.append(episode)
                eval_scores.append(metrics["mean_score"])
                if metrics["mean_score"] > best_score:
                    best_score = metrics["mean_score"]
                    best_episode = episode
                    if train_cfg.save_best_checkpoint:
                        agent.save(best_path)
                        print(
                            f"  >> New best checkpoint: eval_score={best_score:.2f} "
                            f"@ episode {best_episode} -> {best_path.name}"
                        )

            writer.writerow(
                {
                    "episode": episode,
                    "score": score,
                    "reward": total_reward,
                    "steps": steps,
                    "epsilon": agent.epsilon,
                    "eval_mean_score": eval_mean_score,
                }
            )

            if episode % train_cfg.log_every == 0:
                print(
                    f"Episode {episode}/{train_cfg.episodes} | "
                    f"score={score} reward={total_reward:.2f} eps={agent.epsilon:.3f}"
                    + (f" eval_score={eval_mean_score}" if eval_mean_score != "" else ""),
                    flush=True,
                )
                # Xả bộ đệm xuống đĩa, nếu không training_log.csv trông như bị
                # đứng hàng nghìn episode dù train vẫn đang chạy bình thường.
                f.flush()

    random_baseline = run_random_episode(env, rng)
    print(f"Random baseline (1 ep): score={random_baseline['score']} reward={random_baseline['total_reward']:.2f}")

    plot_paths = save_training_plots(
        episodes=history_episodes,
        scores=history_scores,
        rewards=history_rewards,
        epsilons=history_epsilons,
        eval_episodes=eval_episodes,
        eval_scores=eval_scores,
        output_dir=output_dir,
        ma_window=max(10, train_cfg.log_every // 2),
        agent_label="SARSA",
    )
    print("Saved training plots:")
    for p in plot_paths:
        print(f"  - {p}")

    agent.save(last_path)

    if train_cfg.export_best_as_final and best_path.exists():
        best_agent = SarsaAgent.load(best_path)
        best_agent.save(final_path)
        print(
            f"Exported best checkpoint (episode {best_episode}, eval_score={best_score:.2f}) "
            f"-> {final_path.name}"
        )
    else:
        agent.save(final_path)
        print(f"Saved final model -> {final_path.name}")

    agent.save_metadata(
        output_dir / "agent_metadata.json",
        extra={
            "agent": "sarsa",
            "best_eval_score": best_score,
            "best_episode": best_episode,
            "final_episode": train_cfg.episodes,
            "best_checkpoint": str(best_path.name) if best_path.exists() else None,
            "final_checkpoint": str(final_path.name),
            "last_checkpoint": str(last_path.name),
        },
    )
    return agent, env


def parse_args() -> argparse.Namespace:
    defaults = DEFAULT_HYPERPARAMETERS.training
    parser = argparse.ArgumentParser(
        description="Train SARSA Snake agent (mặc định lấy từ agents/sarsa/hyperparameters.py)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help=f"Số episode train (mặc định: {defaults.episodes})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=f"Random seed (mặc định: {defaults.seed})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Thư mục output (mặc định: {defaults.output_dir})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_config(
        episodes=args.episodes,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    agent, env = train_sarsa(config)
    out = config.training.output_dir
    print("Training complete.")
    print(f"  Best model:  {out / 'agent.pkl'}")
    print(f"  Best ckpt:   {out / 'agent_best.pkl'}")
    print(f"  Last ckpt:   {out / 'agent_last.pkl'}")
    print(f"  Charts:      {out / 'training_dashboard.png'}")


if __name__ == "__main__":
    main()
