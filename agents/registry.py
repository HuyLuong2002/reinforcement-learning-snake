"""Đăng ký và load agent theo tên."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class PlayableAgent(Protocol):
    def greedy_policy_action(self, state: np.ndarray) -> int: ...

    @classmethod
    def load(cls, path: Path) -> Any: ...


@dataclass(frozen=True)
class AgentSpec:
    name: str
    label: str
    output_dir: Path
    train_command: str
    load_fn: type[PlayableAgent]


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sarsa_output_dir() -> Path:
    return PROJECT_ROOT / "output" / "sarsa" / "training"


def _q_learning_output_dir() -> Path:
    return PROJECT_ROOT / "output" / "q_learning" / "training"


def _load_sarsa(path: Path):
    from agents.sarsa.agent import SarsaAgent

    return SarsaAgent.load(path)


def _load_q_learning(path: Path):
    try:
        from agents.q_learning.agent import QLearningAgent
    except ImportError as exc:
        raise FileNotFoundError(
            "Chưa có code Q-learning trong agents/q_learning/. "
            "Xem agents/q_learning/README.md"
        ) from exc
    return QLearningAgent.load(path)


AGENT_REGISTRY: dict[str, AgentSpec] = {
    "sarsa": AgentSpec(
        name="sarsa",
        label="SARSA",
        output_dir=_sarsa_output_dir(),
        train_command="python -m agents.sarsa.train",
        load_fn=_load_sarsa,
    ),
    "q_learning": AgentSpec(
        name="q_learning",
        label="Q-Learning",
        output_dir=_q_learning_output_dir(),
        train_command="python -m agents.q_learning.train",
        load_fn=_load_q_learning,
    ),
}


def get_agent_spec(agent_name: str) -> AgentSpec:
    key = agent_name.lower()
    if key not in AGENT_REGISTRY:
        available = ", ".join(sorted(AGENT_REGISTRY))
        raise ValueError(f"Agent '{agent_name}' không hợp lệ. Chọn: {available}")
    return AGENT_REGISTRY[key]


def model_search_paths(spec: AgentSpec) -> list[Path]:
    """Thứ tự ưu tiên load model."""
    out = spec.output_dir
    paths = [
        out / "agent.pkl",
        out / "agent_best.pkl",
        out / "agent_last.pkl",
    ]
    if spec.name == "sarsa":
        legacy = PROJECT_ROOT / "output" / "training"
        paths.extend(
            [
                legacy / "sarsa_agent.pkl",
                legacy / "sarsa_agent_best.pkl",
                legacy / "sarsa_agent_last.pkl",
            ]
        )
    return paths


def load_agent(agent_name: str) -> tuple[PlayableAgent | None, AgentSpec, bool]:
    """
    Trả về (agent, spec, model_missing).
    model_missing=True nếu không tìm thấy file model.
    """
    spec = get_agent_spec(agent_name)
    for path in model_search_paths(spec):
        if path.exists():
            print(f"Loaded model: {path}")
            return spec.load_fn(path), spec, False
    return None, spec, True
