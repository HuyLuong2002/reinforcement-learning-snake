"""Đăng ký và load agent theo tên."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from common.run_store import agent_output_root, latest_model_path, list_model_runs


class PlayableAgent(Protocol):
    def greedy_policy_action(self, state: np.ndarray) -> int: ...

    def choose_action(
        self,
        state: np.ndarray,
        greedy: bool = False,
        forbidden: int | None = None,
    ) -> int: ...

    @classmethod
    def load(cls, path: Path) -> Any: ...


@dataclass(frozen=True)
class AgentSpec:
    name: str
    label: str
    output_dir: Path
    train_command: str
    load_fn: type[PlayableAgent]


def _sarsa_output_dir() -> Path:
    return agent_output_root("sarsa")


def _q_learning_output_dir() -> Path:
    return agent_output_root("q_learning")


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
    """Mọi file model có thể load, lần train mới nhất trước."""
    return [run.model_path for run in list_model_runs(spec.name)]


def load_agent(agent_name: str) -> tuple[PlayableAgent | None, AgentSpec, bool]:
    """
    Load model mới nhất. Trả về (agent, spec, model_missing).
    """
    spec = get_agent_spec(agent_name)
    path = latest_model_path(agent_name)
    if path is None:
        return None, spec, True
    print(f"Loaded model: {path}")
    return spec.load_fn(path), spec, False
