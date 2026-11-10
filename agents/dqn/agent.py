"""Agent Deep Q-Network (DQN).

Thay Q-table bằng mạng neural Q_θ(s) → [Q(s,thẳng), Q(s,trái), Q(s,phải)].
State là feature vector liên tục từ SnakeEnv.encode_features() (cách B),
không phải 7 số rời rạc của tabular.

Hai thành phần giữ cho việc học ổn định:
  - Replay buffer: học trên mini-batch ngẫu nhiên thay vì transition liên tiếp.
  - Target network Q_θ⁻: bản sao đóng băng, sync định kỳ, dùng để tính target
        y = r + γ · Q_θ⁻(s', argmax_a' Q_θ(s', a'))   (Double DQN)

Khi chơi / eval: greedy argmax Q_θ. Không heuristic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from agents.dqn.network import QNetwork
from agents.dqn.replay_buffer import ReplayBuffer


def _pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class DQNAgent:
    # common.policy.select_action đọc cờ này để đưa env.encode_features() vào.
    uses_features = True

    def __init__(
        self,
        n_features: int,
        n_actions: int,
        *,
        lr: float = 5e-4,
        gamma: float = 0.99,
        hidden: int = 256,
        batch_size: int = 128,
        buffer_size: int = 200_000,
        learning_starts: int = 5_000,
        train_freq: int = 4,
        target_update_every: int = 2_000,
        double_dqn: bool = True,
        grad_clip: float = 10.0,
        reward_scale: float = 0.1,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.02,
        epsilon_decay_steps: int = 300_000,
        device: str = "auto",
        rng: np.random.Generator | None = None,
        with_buffer: bool = True,
    ) -> None:
        self.n_features = int(n_features)
        self.n_actions = int(n_actions)
        self.gamma = gamma
        self.hidden = hidden
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.target_update_every = target_update_every
        self.double_dqn = double_dqn
        self.grad_clip = grad_clip
        self.reward_scale = reward_scale
        self.epsilon_start = epsilon_start
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps
        self.lr = lr
        self.rng = rng or np.random.default_rng()
        self.device = _pick_device(device)

        self.online = QNetwork(n_features, n_actions, hidden).to(self.device)
        self.target = QNetwork(n_features, n_actions, hidden).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)

        self.buffer = (
            ReplayBuffer(buffer_size, n_features, self.rng) if with_buffer else None
        )
        self.total_steps = 0
        self.recent_losses: list[float] = []
        self.recent_q: list[float] = []

    # ---------------------------------------------------------------- policy
    @torch.no_grad()
    def q_values(self, state: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(state, dtype=np.float32), device=self.device)
        return self.online(x.unsqueeze(0)).squeeze(0).cpu().numpy()

    def choose_action(
        self,
        state: np.ndarray,
        greedy: bool = False,
        forbidden: int | None = None,
    ) -> int:
        legal = [a for a in range(self.n_actions) if a != forbidden]
        if not greedy and self.rng.random() < self.epsilon:
            return int(legal[int(self.rng.integers(0, len(legal)))])
        q = self.q_values(state)
        return int(max(legal, key=lambda a: q[a]))

    def greedy_policy_action(self, state: np.ndarray, forbidden: int | None = None) -> int:
        return self.choose_action(state, greedy=True, forbidden=forbidden)

    # -------------------------------------------------------------- learning
    def remember(self, s, a: int, r: float, s2, terminated: bool) -> None:
        """Lưu transition. Chỉ `terminated` (chết/thắng) cắt bootstrap;
        truncated (hết bước) vẫn bootstrap vì state sau không thật sự kết thúc."""
        assert self.buffer is not None
        self.buffer.add(s, a, r * self.reward_scale, s2, terminated)

    def step(self) -> float | None:
        """Gọi sau mỗi bước env: cập nhật epsilon, học, sync target. Trả loss nếu có học."""
        self.total_steps += 1
        frac = min(1.0, self.total_steps / max(1, self.epsilon_decay_steps))
        self.epsilon = self.epsilon_start + frac * (self.epsilon_min - self.epsilon_start)

        loss = None
        if (
            self.buffer is not None
            and len(self.buffer) >= max(self.learning_starts, self.batch_size)
            and self.total_steps % self.train_freq == 0
        ):
            loss = self._train_batch()
        if self.total_steps % self.target_update_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        return loss

    def _train_batch(self) -> float:
        assert self.buffer is not None
        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        dev = self.device
        s = torch.as_tensor(s, device=dev)
        a = torch.as_tensor(a, device=dev)
        r = torch.as_tensor(r, device=dev)
        s2 = torch.as_tensor(s2, device=dev)
        d = torch.as_tensor(d, device=dev)

        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double_dqn:
                next_a = self.online(s2).argmax(dim=1, keepdim=True)
                next_q = self.target(s2).gather(1, next_a).squeeze(1)
            else:
                next_q = self.target(s2).max(dim=1).values
            target = r + self.gamma * (1.0 - d) * next_q

        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optimizer.step()

        value = float(loss.item())
        self.recent_losses.append(value)
        self.recent_q.append(float(q.mean().item()))
        return value

    def pop_stats(self) -> tuple[float | None, float | None]:
        """Loss / Q trung bình từ lần gọi trước — dùng cho log."""
        if not self.recent_losses:
            return None, None
        loss = float(np.mean(self.recent_losses))
        q = float(np.mean(self.recent_q))
        self.recent_losses.clear()
        self.recent_q.clear()
        return loss, q

    def decay_epsilon(self) -> float:
        """Giữ API giống tabular — epsilon DQN giảm theo bước trong step()."""
        return self.epsilon

    # ------------------------------------------------------------------ I/O
    def _config(self) -> dict:
        return {
            "n_features": self.n_features,
            "n_actions": self.n_actions,
            "lr": self.lr,
            "gamma": self.gamma,
            "hidden": self.hidden,
            "batch_size": self.batch_size,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "target_update_every": self.target_update_every,
            "double_dqn": self.double_dqn,
            "grad_clip": self.grad_clip,
            "reward_scale": self.reward_scale,
            "epsilon_start": self.epsilon_start,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay_steps": self.epsilon_decay_steps,
        }

    def save(self, path: Path) -> None:
        """Lưu trọng số mạng (torch.save). Giữ đuôi .pkl để app/run_store tìm thấy."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "agent_type": "dqn",
                "config": self._config(),
                "epsilon": self.epsilon,
                "total_steps": self.total_steps,
                "online": self.online.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, device: str = "auto") -> DQNAgent:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        agent = cls(**payload["config"], device=device, with_buffer=False)
        agent.online.load_state_dict(payload["online"])
        agent.target.load_state_dict(payload["online"])
        agent.online.eval()
        agent.epsilon = payload.get("epsilon", agent.epsilon_min)
        agent.total_steps = payload.get("total_steps", 0)
        return agent

    def save_metadata(self, path: Path, extra: dict | None = None) -> None:
        n_params = sum(p.numel() for p in self.online.parameters())
        data = {"n_parameters": int(n_params), "total_steps": self.total_steps}
        if extra:
            data.update(extra)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
