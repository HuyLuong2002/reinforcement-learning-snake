"""Chạy agent SARSA nhiều ván và phân tích nguyên nhân thua.

Đo hiệu quả đường đi (so với A*), kiểu chết (tường / tự cắn),
và việc chọn hướng nguy hiểm khi vẫn còn hướng OPEN.

Chạy:
  python -m agents.sarsa.analyze_games --games 1000 --grid both
  python -m agents.sarsa.analyze_games --games 50 --model output/sarsa/2026-08-27_16-33-57/agent.pkl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from agents.sarsa.agent import SarsaAgent
from agents.sarsa.hyperparameters import DEFAULT_HYPERPARAMETERS
from common.env_hyperparameters import PLAYABLE_LEVELS, grid_label, parse_play_grid
from common.policy import select_action
from common.run_store import latest_model_path
from common.snake_env import (
    DIRECTION_VECTORS,
    N_RELATIVE_ACTIONS,
    SAFETY_COLLISION,
    SAFETY_OPEN,
    SAFETY_TIGHT,
    SAFETY_TRAP,
    SnakeEnv,
    relative_to_absolute,
)

SAFETY_NAMES = {0: "collision", 1: "trap", 2: "tight", 3: "open"}


def build_env(seed: int, width: int, height: int) -> SnakeEnv:
    """Tạo env với cùng tham số trong hyperparameters.py, scale theo lưới."""
    cfg = DEFAULT_HYPERPARAMETERS.env.for_shape(width, height)
    return cfg.create_env(np.random.default_rng(seed))


def _to_heading(direction: int, action: int) -> int:
    return relative_to_absolute(direction, action)


def _safety_of(obs: np.ndarray, action: int) -> int:
    return int(obs[2 + int(action)])


def _next_cell(head: tuple[int, int], heading: int) -> tuple[int, int]:
    dx, dy = DIRECTION_VECTORS[heading]
    return head[0] + dx, head[1] + dy


def run_episode(agent: SarsaAgent, seed: int, width: int, height: int) -> dict:
    """Chạy 1 ván greedy, gom thống kê đường đi / an toàn / kiểu chết."""
    env = build_env(seed, width, height)
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0

    food_legs: list[dict] = []
    leg_astar: int | None = env.astar_distance(env.state.snake[0], env.state.food)
    leg_steps = 0
    leg_on_astar = 0
    leg_detours = 0
    leg_start_len = len(env.state.snake)

    chose_collision_when_open = 0
    chose_tight_when_open = 0
    chose_trap_when_open = 0
    steps_short = 0  # fill_bucket == 0
    detours_short = 0
    on_astar_short = 0
    unseen_state_steps = 0

    last_safety = 3
    last_best_safety = 3
    last_q = [0.0, 0.0, 0.0]
    last_action = 0
    last_fill = 0
    last_was_unseen = False

    while True:
        assert env.state is not None
        direction = env.state.direction
        head = env.state.snake[0]
        fill = env._length_bucket()
        suggested = env.astar_next_action()
        action = select_action(env, agent, obs, greedy=True)
        heading = _to_heading(direction, action)
        q_vals = [agent.get_q(obs, a) for a in range(N_RELATIVE_ACTIONS)]
        unseen = all(q == 0.0 for q in q_vals)
        if unseen:
            unseen_state_steps += 1

        chosen_safety = _safety_of(obs, action)
        best_safety = max(int(obs[2]), int(obs[3]), int(obs[4]))
        if best_safety == SAFETY_OPEN and chosen_safety == SAFETY_COLLISION:
            chose_collision_when_open += 1
        if best_safety == SAFETY_OPEN and chosen_safety == SAFETY_TIGHT:
            chose_tight_when_open += 1
        if best_safety == SAFETY_OPEN and chosen_safety == SAFETY_TRAP:
            chose_trap_when_open += 1

        last_safety = chosen_safety
        last_best_safety = best_safety
        last_q = q_vals
        last_action = action
        last_fill = fill
        last_was_unseen = unseen

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        ate = info["score"] > len(food_legs) and info["end_reason"] != "death"
        on_astar = ate or (
            suggested is not None
            and heading == suggested
        )
        path_known = suggested is not None

        if fill == 0:
            steps_short += 1
            if on_astar:
                on_astar_short += 1
            elif path_known:
                detours_short += 1

        leg_steps += 1
        if on_astar:
            leg_on_astar += 1
        elif path_known:
            leg_detours += 1

        if ate and not terminated:
            food_legs.append(
                {
                    "score": info["score"],
                    "astar": leg_astar,
                    "steps": leg_steps,
                    "on_astar": leg_on_astar,
                    "detours": leg_detours,
                    "start_len": leg_start_len,
                    "extra": None
                    if leg_astar is None
                    else max(0, leg_steps - leg_astar),
                }
            )
            assert env.state is not None
            leg_astar = env.astar_distance(env.state.snake[0], env.state.food)
            leg_steps = 0
            leg_on_astar = 0
            leg_detours = 0
            leg_start_len = len(env.state.snake)

        if terminated or truncated:
            death_kind = ""
            if info["end_reason"] == "death":
                nx, ny = _next_cell(head, _to_heading(direction, last_action))
                if not (0 <= nx < width and 0 <= ny < height):
                    death_kind = "wall"
                else:
                    death_kind = "self"

            extras = [
                leg["extra"]
                for leg in food_legs
                if leg["extra"] is not None and leg["start_len"] <= 8
            ]
            extras_all = [leg["extra"] for leg in food_legs if leg["extra"] is not None]
            return {
                "seed": seed,
                "grid": grid_label(width, height),
                "score": info["score"],
                "steps": info["steps"],
                "end": info["end_reason"],
                "death_kind": death_kind,
                "snake_len": len(env.state.snake) if env.state else 0,
                "reward": total_reward,
                "fill_at_end": last_fill,
                "last_safety": last_safety,
                "last_best_safety": last_best_safety,
                "last_unseen": last_was_unseen,
                "last_q": last_q,
                "chose_collision_when_open": chose_collision_when_open,
                "chose_tight_when_open": chose_tight_when_open,
                "chose_trap_when_open": chose_trap_when_open,
                "steps_short": steps_short,
                "on_astar_short": on_astar_short,
                "detours_short": detours_short,
                "unseen_state_steps": unseen_state_steps,
                "n_food": len(food_legs),
                "mean_extra_short": float(np.mean(extras)) if extras else None,
                "mean_extra_all": float(np.mean(extras_all)) if extras_all else None,
                "mean_astar_follow_short": (
                    on_astar_short / max(1, on_astar_short + detours_short)
                ),
            }

    raise RuntimeError("unreachable")


def summarize(results: list[dict], grid: str) -> dict:
    n = len(results)
    scores = [r["score"] for r in results]
    ends = Counter(r["end"] for r in results)
    deaths = [r for r in results if r["end"] == "death"]
    death_kinds = Counter(r["death_kind"] for r in deaths)
    last_safety = Counter(SAFETY_NAMES.get(r["last_safety"], str(r["last_safety"])) for r in deaths)
    last_best = Counter(
        SAFETY_NAMES.get(r["last_best_safety"], str(r["last_best_safety"])) for r in deaths
    )

    collision_open = sum(r["chose_collision_when_open"] for r in results)
    tight_open = sum(r["chose_tight_when_open"] for r in results)
    trap_open = sum(r["chose_trap_when_open"] for r in results)
    unseen_deaths = sum(1 for r in deaths if r["last_unseen"])
    suicide_open = sum(
        1
        for r in deaths
        if r["last_best_safety"] == SAFETY_OPEN and r["last_safety"] == SAFETY_COLLISION
    )

    extras_short = [r["mean_extra_short"] for r in results if r["mean_extra_short"] is not None]
    extras_all = [r["mean_extra_all"] for r in results if r["mean_extra_all"] is not None]
    follow = [r["mean_astar_follow_short"] for r in results if r["steps_short"] > 0]
    unseen_frac = [
        r["unseen_state_steps"] / r["steps"] for r in results if r["steps"] > 0
    ]

    score_bins = {"0-9": 0, "10-19": 0, "20-29": 0, "30-39": 0, "40-49": 0, "50+": 0}
    for s in scores:
        if s < 10:
            score_bins["0-9"] += 1
        elif s < 20:
            score_bins["10-19"] += 1
        elif s < 30:
            score_bins["20-29"] += 1
        elif s < 40:
            score_bins["30-39"] += 1
        elif s < 50:
            score_bins["40-49"] += 1
        else:
            score_bins["50+"] += 1

    fill_at_death = Counter(r["fill_at_end"] for r in deaths)

    return {
        "grid": grid,
        "games": n,
        "mean_score": float(np.mean(scores)),
        "median_score": float(np.median(scores)),
        "max_score": int(max(scores)),
        "min_score": int(min(scores)),
        "std_score": float(np.std(scores)),
        "end_reasons": dict(ends),
        "death_kinds": dict(death_kinds),
        "death_count": len(deaths),
        "self_bite_pct": (death_kinds.get("self", 0) / n) * 100,
        "wall_pct": (death_kinds.get("wall", 0) / n) * 100,
        "death_last_safety": dict(last_safety),
        "death_best_available_safety": dict(last_best),
        "suicide_despite_open": suicide_open,
        "unseen_state_deaths": unseen_deaths,
        "mean_unseen_step_frac": float(np.mean(unseen_frac)) if unseen_frac else 0.0,
        "chose_collision_when_open_total": collision_open,
        "chose_tight_when_open_total": tight_open,
        "chose_trap_when_open_total": trap_open,
        "mean_extra_steps_when_short": float(np.mean(extras_short)) if extras_short else None,
        "mean_extra_steps_all_food": float(np.mean(extras_all)) if extras_all else None,
        "mean_astar_follow_when_short": float(np.mean(follow)) if follow else None,
        "score_bins": score_bins,
        "fill_bucket_at_death": {str(k): v for k, v in sorted(fill_at_death.items())},
    }


def print_summary(s: dict) -> None:
    print(f"\n========== GRID {s['grid']}x{s['grid']} — {s['games']} games ==========")
    print(
        f"score: mean={s['mean_score']:.2f}  median={s['median_score']:.1f}  "
        f"min={s['min_score']}  max={s['max_score']}  std={s['std_score']:.2f}"
    )
    print(f"end_reasons={s['end_reasons']}")
    print(
        f"death_kinds={s['death_kinds']}  self_bite={s['self_bite_pct']:.1f}%  "
        f"wall={s['wall_pct']:.1f}%"
    )
    print(f"death last chosen safety={s['death_last_safety']}")
    print(f"death best available safety={s['death_best_available_safety']}")
    print(
        f"suicide despite OPEN available: {s['suicide_despite_open']}  "
        f"unseen-state deaths: {s['unseen_state_deaths']}"
    )
    print(f"mean unseen-state step fraction: {s['mean_unseen_step_frac']:.3f}")
    print(
        f"chose collision/tight/trap when OPEN existed (step counts): "
        f"{s['chose_collision_when_open_total']}/"
        f"{s['chose_tight_when_open_total']}/"
        f"{s['chose_trap_when_open_total']}"
    )
    print(
        f"A* follow when snake short (fill<20%): "
        f"{s['mean_astar_follow_when_short']}"
    )
    print(
        f"extra steps vs A* (short snake / all food): "
        f"{s['mean_extra_steps_when_short']} / {s['mean_extra_steps_all_food']}"
    )
    print(f"score bins={s['score_bins']}")
    print(f"fill_bucket at death={s['fill_bucket_at_death']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phân tích nhiều ván greedy của SARSA")
    parser.add_argument("--games", type=int, default=10, help="Số ván mỗi lưới")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="File model .pkl (mặc định: lần train mới nhất)",
    )
    parser.add_argument(
        "--grid",
        type=str,
        default="15x20",
        help="10, 15x20, 30, hoặc both",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Ghi summary JSON ra file",
    )
    args = parser.parse_args()

    if args.model:
        model_path = Path(args.model)
    else:
        latest = latest_model_path("sarsa")
        if latest is None:
            raise SystemExit(
                "Chưa có model. Chạy `python -m agents.sarsa.train` trước."
            )
        model_path = latest
    agent = SarsaAgent.load(model_path)
    if args.grid == "both":
        levels = list(PLAYABLE_LEVELS)
    else:
        levels = [parse_play_grid(args.grid)]

    print(f"Model: {model_path}")
    summaries: list[dict] = []
    for width, height in levels:
        cfg = DEFAULT_HYPERPARAMETERS.env.for_shape(width, height)
        label = grid_label(width, height)
        print(
            f"\n>>> {args.games} games on {label} "
            f"(max_score={SnakeEnv.max_score(width, height)} max_steps={cfg.max_steps})",
            flush=True,
        )
        results = []
        for i in range(args.games):
            results.append(run_episode(agent, seed=i, width=width, height=height))
            if (i + 1) % 100 == 0 or i + 1 == args.games:
                print(f"  ... {i + 1}/{args.games}", flush=True)
        summary = summarize(results, label)
        print_summary(summary)
        summaries.append(summary)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
