"""Train agent DQN trên Snake (state = feature vector, cách B).

Mỗi lần chạy tạo folder mới: output/dqn/YYYY-MM-DD_HH-MM-SS/
  - training_log.csv   (thêm cột loss, q_mean, total_steps)
  - agent.pkl / agent_best.pkl / agent_last.pkl   (torch.save)
  - *.png, agent_metadata.json
"""

from __future__ import annotations

import argparse
import csv
import time

import numpy as np
import torch

from agents.dqn.agent import DQNAgent
from agents.dqn.hyperparameters import AGENT_NAME, DEFAULT_HYPERPARAMETERS, Hyperparameters, build_config
from common.env_hyperparameters import PLAY_GRID_CHOICES, grid_label
from common.run_store import new_run_dir
from common.snake_env import N_FEATURES, SnakeEnv
from common.training_plots import save_training_plots


def build_agent(config: Hyperparameters, env: SnakeEnv, rng: np.random.Generator) -> DQNAgent:
    c = config.dqn
    return DQNAgent(
        n_features=N_FEATURES,
        n_actions=env.n_actions,
        lr=c.lr,
        gamma=c.gamma,
        hidden=c.hidden,
        batch_size=c.batch_size,
        buffer_size=c.buffer_size,
        learning_starts=c.learning_starts,
        train_freq=c.train_freq,
        target_update_every=c.target_update_every,
        double_dqn=c.double_dqn,
        grad_clip=c.grad_clip,
        reward_scale=c.reward_scale,
        epsilon_start=c.epsilon_start,
        epsilon_min=c.epsilon_min,
        epsilon_decay_steps=c.epsilon_decay_steps,
        device=config.training.device,
        rng=rng,
    )


def evaluate_agent(env: SnakeEnv, agent: DQNAgent, n_episodes: int, rng: np.random.Generator) -> dict:
    scores: list[int] = []
    for _ in range(n_episodes):
        env.reset(seed=int(rng.integers(0, 1_000_000)))
        score = 0
        while True:
            action = agent.choose_action(env.encode_features(), greedy=True)
            _, _, terminated, truncated, info = env.step(action)
            score = info.get("score", score)
            if terminated or truncated:
                break
        scores.append(score)
    return {"mean_score": float(np.mean(scores)), "max_score": int(np.max(scores))}


def train_dqn(config: Hyperparameters):
    tc = config.training
    rng = np.random.default_rng(tc.seed)
    torch.manual_seed(tc.seed)
    env = config.env.create_env(rng)
    eval_env = config.env.create_env(np.random.default_rng(tc.seed + 1))
    agent = build_agent(config, env, rng)

    output_dir = tc.output_dir or new_run_dir(AGENT_NAME)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "agent_best.pkl"
    last_path = output_dir / "agent_last.pkl"
    final_path = output_dir / "agent.pkl"
    print(f"Output: {output_dir} | device={agent.device} | grid={grid_label(env.width, env.height)}")

    hist_ep, hist_score, hist_reward, hist_eps = [], [], [], []
    eval_eps, eval_scores = [], []
    best_score, best_episode = -1.0, 0
    t0 = time.time()

    fields = ["episode", "score", "reward", "steps", "epsilon", "loss", "q_mean", "total_steps", "eval_mean_score"]
    with (output_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for episode in range(1, tc.episodes + 1):
            env.reset(seed=int(rng.integers(0, 1_000_000)))
            state = env.encode_features()
            total_reward, steps, score = 0.0, 0, 0
            while True:
                action = agent.choose_action(state, greedy=False)
                _, reward, terminated, truncated, info = env.step(action)
                next_state = env.encode_features()
                agent.remember(state, action, reward, next_state, terminated)
                agent.step()
                state = next_state
                total_reward += reward
                steps += 1
                score = info.get("score", score)
                if terminated or truncated:
                    break

            hist_ep.append(episode)
            hist_score.append(score)
            hist_reward.append(total_reward)
            hist_eps.append(agent.epsilon)

            eval_mean = ""
            if episode % tc.eval_every == 0:
                ev = evaluate_agent(eval_env, agent, tc.eval_episodes, rng)
                eval_mean = ev["mean_score"]
                eval_eps.append(episode)
                eval_scores.append(eval_mean)
                print(f"  [eval] ep {episode}: mean={eval_mean:.2f} max={ev['max_score']}", flush=True)
                if eval_mean > best_score:
                    best_score, best_episode = eval_mean, episode
                    if tc.save_best_checkpoint:
                        agent.save(best_path)
                        print(f"  >> New best {best_score:.2f} -> {best_path.name}", flush=True)

            loss, q_mean = agent.pop_stats()
            writer.writerow({
                "episode": episode, "score": score, "reward": total_reward, "steps": steps,
                "epsilon": agent.epsilon, "loss": "" if loss is None else loss,
                "q_mean": "" if q_mean is None else q_mean, "total_steps": agent.total_steps,
                "eval_mean_score": eval_mean,
            })
            if episode % tc.log_every == 0:
                recent = np.mean(hist_score[-tc.log_every:])
                print(
                    f"Episode {episode}/{tc.episodes} | avg_score={recent:.2f} eps={agent.epsilon:.3f} "
                    f"steps={agent.total_steps} loss={loss if loss is None else round(loss, 4)} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
                f.flush()

    save_training_plots(
        episodes=hist_ep, scores=hist_score, rewards=hist_reward, epsilons=hist_eps,
        eval_episodes=eval_eps, eval_scores=eval_scores, output_dir=output_dir,
        ma_window=max(10, tc.log_every), agent_label="DQN",
    )
    agent.save(last_path)
    if tc.export_best_as_final and best_path.exists():
        DQNAgent.load(best_path).save(final_path)
    else:
        agent.save(final_path)

    agent.save_metadata(
        output_dir / "agent_metadata.json",
        extra={
            "agent": "dqn",
            "policy": "off-policy (Double DQN)" if agent.double_dqn else "off-policy (DQN)",
            "grid": grid_label(env.width, env.height),
            "grid_width": env.width,
            "grid_height": env.height,
            "max_score": SnakeEnv.max_score(env.width, env.height),
            "n_features": N_FEATURES,
            "n_actions": agent.n_actions,
            "seed": tc.seed,
            "episodes": tc.episodes,
            "best_eval_score": best_score,
            "best_episode": best_episode,
            "train_seconds": round(time.time() - t0, 1),
            "hyperparameters": config.to_dict(),
        },
    )
    return agent, env, output_dir


def parse_args() -> argparse.Namespace:
    d = DEFAULT_HYPERPARAMETERS.training
    p = argparse.ArgumentParser(description="Train DQN Snake (agents/dqn/hyperparameters.py)")
    p.add_argument("--episodes", type=int, default=None, help=f"mặc định {d.episodes}")
    p.add_argument("--seed", type=int, default=None, help=f"mặc định {d.seed}")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--grid", type=str, default=None, choices=list(PLAY_GRID_CHOICES))
    p.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"])
    return p.parse_args()


def main() -> None:
    a = parse_args()
    config = build_config(episodes=a.episodes, seed=a.seed, output_dir=a.output_dir, grid=a.grid, device=a.device)
    _, _, out = train_dqn(config)
    print(f"Training complete. Model: {out / 'agent.pkl'}")


if __name__ == "__main__":
    main()
