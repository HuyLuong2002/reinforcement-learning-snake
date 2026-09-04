"""Agent SARSA dạng bảng Q (tabular).

SARSA (on-policy):
  Q(s,a) ← Q(s,a) + α [ r + γ Q(s',a') - Q(s,a) ]

Khác Q-learning (off-policy) ở chỗ dùng a' thực sự sẽ chọn (epsilon-greedy),
không dùng max_a' Q(s',a').

Khi chơi / eval: greedy argmax Q trên 3 hướng tương đối. Không heuristic, không A* chọn hộ.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np


class SarsaAgent:
    def __init__(
        self,
        n_state_dims: Sequence[int],
        n_actions: int,
        alpha: float = 0.1,       # learning rate
        gamma: float = 0.99,      # discount factor
        epsilon: float = 1.0,     # xác suất explore ban đầu
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,  # nhân mỗi episode
        rng: np.random.Generator | None = None,
    ) -> None:
        self.n_state_dims = tuple(int(n) for n in n_state_dims)
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = rng or np.random.default_rng()

        # Q-table: shape = (dim0, dim1, ..., n_actions)
        self.q_table = np.zeros(self.n_state_dims + (n_actions,), dtype=np.float64)

    def _as_state(self, state: np.ndarray | Sequence[int]) -> tuple[int, ...]:
        """Chuẩn hóa observation → tuple index an toàn cho Q-table."""
        if isinstance(state, np.ndarray):
            values = state.tolist()
        else:
            values = list(state)
        return tuple(
            int(np.clip(values[i], 0, self.n_state_dims[i] - 1))
            for i in range(len(self.n_state_dims))
        )

    def get_q(self, state: np.ndarray | Sequence[int], action: int) -> float:
        """Lấy giá trị Q(s, a) — tiện debug / phân tích."""
        state_key = self._as_state(state)
        action = int(np.clip(action, 0, self.n_actions - 1))
        return float(self.q_table[state_key][action])

    def choose_action(
        self,
        state: np.ndarray | Sequence[int],
        greedy: bool = False,
        forbidden: int | None = None,
    ) -> int:
        """
        Chọn action:
          - greedy=False (train): epsilon-greedy — random với xác suất epsilon
          - greedy=True  (eval/play): luôn chọn action có Q cao nhất
          - forbidden: giữ tương thích API; action space đã chỉ còn 3 hướng hợp lệ
        """
        state_key = self._as_state(state)
        q = self.q_table[state_key]
        legal = (
            [a for a in range(self.n_actions) if a != forbidden]
            if forbidden is not None
            else list(range(self.n_actions))
        )
        if not greedy and self.rng.random() < self.epsilon:
            return int(legal[int(self.rng.integers(0, len(legal)))])
        return int(max(legal, key=lambda a: q[a]))

    def update(
        self,
        state: np.ndarray | Sequence[int],
        action: int,
        reward: float,
        next_state: np.ndarray | Sequence[int],
        next_action: int,
        done: bool,
    ) -> float:
        """
        Cập nhật SARSA một bước.

        next_action phải là action agent THỰC SỰ sẽ làm ở state tiếp theo
        (on-policy — khác Q-learning dùng max Q).
        """
        state_key = self._as_state(state)
        next_state_key = self._as_state(next_state)
        action = int(np.clip(action, 0, self.n_actions - 1))
        next_action = int(np.clip(next_action, 0, self.n_actions - 1))

        current_q = self.q_table[state_key][action]
        # done=True → không cộng Q(s',a') vì episode kết thúc
        target = reward if done else reward + self.gamma * self.q_table[next_state_key][next_action]
        td_error = target - current_q
        self.q_table[state_key][action] = current_q + self.alpha * td_error
        return float(td_error)

    def decay_epsilon(self) -> float:
        """Giảm epsilon sau mỗi episode train."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def greedy_policy_action(
        self,
        state: np.ndarray | Sequence[int],
        forbidden: int | None = None,
    ) -> int:
        """Policy khi chơi game / eval — không explore."""
        return self.choose_action(state, greedy=True, forbidden=forbidden)

    def save(self, path: Path) -> None:
        """Lưu Q-table + hyperparameters ra file .pkl."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "agent_type": "sarsa",
            "n_state_dims": self.n_state_dims,
            "n_actions": self.n_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "q_table": self.q_table,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path) -> SarsaAgent:
        """Load agent từ file .pkl đã train."""
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        agent = cls(
            n_state_dims=tuple(payload["n_state_dims"]),
            n_actions=payload["n_actions"],
            alpha=payload["alpha"],
            gamma=payload["gamma"],
            epsilon=payload["epsilon"],
            epsilon_min=payload["epsilon_min"],
            epsilon_decay=payload["epsilon_decay"],
        )
        agent.q_table = payload["q_table"]
        return agent

    def export_q_summary(self) -> dict:
        """Thống kê Q-table — ghi vào agent_metadata.json."""
        return {
            "q_mean": float(np.mean(self.q_table)),
            "q_std": float(np.std(self.q_table)),
            "q_max": float(np.max(self.q_table)),
            "q_min": float(np.min(self.q_table)),
            "nonzero_states": int(np.count_nonzero(np.any(self.q_table != 0, axis=-1))),
        }

    def save_metadata(self, path: Path, extra: dict | None = None) -> None:
        """Ghi thống kê Q-table (+ thông tin train) ra JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.export_q_summary()
        if extra:
            data.update(extra)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
