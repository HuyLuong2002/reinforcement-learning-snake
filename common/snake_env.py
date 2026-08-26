"""Môi trường Snake — tuân chuẩn Gymnasium.

State tabular 7 chiều:
  [food_dx, food_dy, safety_straight, safety_left, safety_right, direction, length_bucket]

Điểm mấu chốt là 3 chiều `safety`: với mỗi hướng đi (thẳng / trái / phải), env chạy
BFS thử sau khi di chuyển để biết hướng đó dẫn vào ngõ cụt hay còn đường thoát.
Nhờ vậy agent phân biệt được "rẽ trái thì kẹt, rẽ phải thì sống" — thứ mà một
chỉ số reachable chung cho cả bàn cờ không diễn tả được.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

DIRECTION_VECTORS: dict[int, tuple[int, int]] = {
    UP: (0, -1),
    RIGHT: (1, 0),
    DOWN: (0, 1),
    LEFT: (-1, 0),
}

LEFT_TURN = {UP: LEFT, RIGHT: UP, DOWN: RIGHT, LEFT: DOWN}
RIGHT_TURN = {UP: RIGHT, RIGHT: DOWN, DOWN: LEFT, LEFT: UP}
OPPOSITE = {UP: DOWN, RIGHT: LEFT, DOWN: UP, LEFT: RIGHT}

INITIAL_SNAKE_LENGTH = 3

# Mức an toàn của một hướng đi (dùng cho 3 chiều safety trong state).
#
# Tiêu chí chính là "sau khi đi, đầu rắn còn tới được đuôi mình không".
# Nếu còn tới được đuôi thì rắn luôn có thể bám theo đuôi mà sống tiếp, nên đó
# là ranh giới an toàn tự nhiên — và không phụ thuộc vào độ dài rắn.
SAFETY_COLLISION = 0  # đâm tường/thân ngay lập tức
SAFETY_TRAP = 1       # số ô tới được < độ dài rắn → không đủ chỗ chứa thân, chết chắc
SAFETY_TIGHT = 2      # đủ chỗ nhưng mất dấu đuôi → rủi ro cao
SAFETY_OPEN = 3       # còn tới được đuôi → an toàn


@dataclass
class SnakeState:
    snake: list[tuple[int, int]]
    food: tuple[int, int]
    direction: int
    score: int
    steps: int
    steps_since_food: int = 0


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        grid_size: int = 10,
        max_steps: int = 2000,
        max_steps_without_food: int = 100,
        reward_food: float = 10.0,
        reward_death: float = -10.0,
        reward_step: float = -0.01,
        reward_closer: float = 0.02,
        death_penalty_per_score: float = 0.5,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.max_steps_without_food = max_steps_without_food
        self.reward_food = reward_food
        self.reward_death = reward_death
        self.reward_step = reward_step
        self.reward_closer = reward_closer
        self.death_penalty_per_score = death_penalty_per_score
        self.rng = rng or np.random.default_rng()

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([3, 3, 4, 4, 4, 4, 5])

        self.state: SnakeState | None = None

    @property
    def n_actions(self) -> int:
        return int(self.action_space.n)

    def n_state_dims(self) -> tuple[int, ...]:
        return tuple(int(n) for n in self.observation_space.nvec)

    @staticmethod
    def max_score(grid_size: int, initial_length: int = INITIAL_SNAKE_LENGTH) -> int:
        return grid_size * grid_size - initial_length

    def _manhattan(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _length_bucket(self) -> int:
        assert self.state is not None
        return min(self.state.score // 5, 4)

    def _body_without_tail(self) -> list[tuple[int, int]]:
        """Ô thân chặn đầu rắn. Bỏ đuôi vì đuôi sẽ rời đi ở bước tiếp theo."""
        assert self.state is not None
        if len(self.state.snake) <= 1:
            return list(self.state.snake)
        return self.state.snake[:-1]

    def _cell_mask(self, cells) -> bytearray:
        """Đánh dấu ô bị chặn trên lưới phẳng (index = y * grid_size + x)."""
        g = self.grid_size
        mask = bytearray(g * g)
        for x, y in cells:
            mask[y * g + x] = 1
        return mask

    def _flood_fill(
        self,
        start: tuple[int, int],
        blocked: bytearray,
        target: tuple[int, int],
        enough: int,
    ) -> tuple[int, bool]:
        """
        BFS từ `start`, trả về (số ô tới được, có chạm `target` không).

        `target` là ô đuôi rắn — chạm được đuôi nghĩa là rắn còn đường sống.
        Dừng sớm ngay khi đã chạm đuôi và đếm đủ `enough` ô, vì lúc đó kết luận
        "an toàn" đã chắc chắn, đếm tiếp cũng không đổi kết quả.

        Dùng chỉ số phẳng và bytearray thay cho set tuple vì hàm này chạy
        3 lần mỗi bước game (một lần cho mỗi hướng đi).
        """
        g = self.grid_size
        last = g - 1
        start_idx = start[1] * g + start[0]
        target_idx = target[1] * g + target[0]

        visited = bytearray(g * g)
        visited[start_idx] = 1
        queue = [start_idx]
        count = 1
        found_target = start_idx == target_idx
        i = 0

        while i < len(queue):
            idx = queue[i]
            i += 1
            y, x = divmod(idx, g)

            if x > 0:
                n = idx - 1
                if not visited[n] and not blocked[n]:
                    visited[n] = 1
                    count += 1
                    if n == target_idx:
                        found_target = True
                    if found_target and count >= enough:
                        return count, True
                    queue.append(n)
            if x < last:
                n = idx + 1
                if not visited[n] and not blocked[n]:
                    visited[n] = 1
                    count += 1
                    if n == target_idx:
                        found_target = True
                    if found_target and count >= enough:
                        return count, True
                    queue.append(n)
            if y > 0:
                n = idx - g
                if not visited[n] and not blocked[n]:
                    visited[n] = 1
                    count += 1
                    if n == target_idx:
                        found_target = True
                    if found_target and count >= enough:
                        return count, True
                    queue.append(n)
            if y < last:
                n = idx + g
                if not visited[n] and not blocked[n]:
                    visited[n] = 1
                    count += 1
                    if n == target_idx:
                        found_target = True
                    if found_target and count >= enough:
                        return count, True
                    queue.append(n)

        return count, found_target

    def _safety_level(
        self,
        direction: int,
        mask_body: bytearray,
        mask_after_move: bytearray,
    ) -> int:
        """
        Đánh giá một hướng đi: đâm ngay / kẹt / chật / thoáng.

        mask_body      = thân trừ đuôi (snake[:-1]) — dùng khi rắn ăn (đuôi ở lại)
        mask_after_move = snake[:-2]                — dùng khi rắn không ăn (đuôi rời đi)
        """
        assert self.state is not None
        g = self.grid_size
        hx, hy = self.state.snake[0]
        dx, dy = DIRECTION_VECTORS[direction]
        nx, ny = hx + dx, hy + dy

        if not (0 <= nx < g and 0 <= ny < g):
            return SAFETY_COLLISION
        if mask_body[ny * g + nx]:
            return SAFETY_COLLISION

        # Ăn thì đuôi ở lại (thân dài thêm), không ăn thì đuôi rời đi.
        eats = (nx, ny) == self.state.food
        if eats:
            blocked = mask_body
            new_snake_tail = self.state.snake[-1]
            new_length = len(self.state.snake) + 1
        else:
            blocked = mask_after_move
            new_snake_tail = self.state.snake[-2] if len(self.state.snake) >= 2 else (nx, ny)
            new_length = len(self.state.snake)

        reachable, tail_reachable = self._flood_fill(
            (nx, ny), blocked, new_snake_tail, enough=new_length
        )

        if reachable < new_length:
            return SAFETY_TRAP
        if not tail_reachable:
            return SAFETY_TIGHT
        return SAFETY_OPEN

    def _is_danger(self, direction: int) -> bool:
        """Hướng này có đâm tường/thân ngay không — dùng để debug."""
        assert self.state is not None
        head_x, head_y = self.state.snake[0]
        dx, dy = DIRECTION_VECTORS[direction]
        next_x, next_y = head_x + dx, head_y + dy

        if not (0 <= next_x < self.grid_size and 0 <= next_y < self.grid_size):
            return True
        return (next_x, next_y) in self._body_without_tail()

    def _spawn_food(self) -> tuple[int, int]:
        assert self.state is not None
        occupied = set(self.state.snake)
        empty = [
            (x, y)
            for x in range(self.grid_size)
            for y in range(self.grid_size)
            if (x, y) not in occupied
        ]
        if not empty:
            return self.state.food
        idx = int(self.rng.integers(0, len(empty)))
        return empty[idx]

    def encode_observation(self) -> np.ndarray:
        assert self.state is not None
        snake = self.state.snake
        head_x, head_y = snake[0]
        food_x, food_y = self.state.food

        # Hai mask này dùng chung cho cả 3 hướng nên chỉ dựng một lần mỗi bước.
        mask_body = self._cell_mask(snake[:-1])
        mask_after_move = self._cell_mask(snake[:-2]) if len(snake) > 2 else bytearray(self.grid_size**2)

        direction = self.state.direction
        safety_straight = self._safety_level(direction, mask_body, mask_after_move)
        safety_left = self._safety_level(LEFT_TURN[direction], mask_body, mask_after_move)
        safety_right = self._safety_level(RIGHT_TURN[direction], mask_body, mask_after_move)

        return np.array(
            [
                int(np.sign(food_x - head_x)) + 1,
                int(np.sign(food_y - head_y)) + 1,
                safety_straight,
                safety_left,
                safety_right,
                direction,
                self._length_bucket(),
            ],
            dtype=np.int64,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        center = self.grid_size // 2
        snake = [(center, center), (center - 1, center), (center - 2, center)]
        self.state = SnakeState(
            snake=snake,
            food=(center, center + 2),
            direction=RIGHT,
            score=0,
            steps=0,
            steps_since_food=0,
        )
        self.state.food = self._spawn_food()
        return self.encode_observation(), {"score": 0}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        action = int(action)
        if action == OPPOSITE[self.state.direction]:
            action = self.state.direction

        old_dist = self._manhattan(self.state.snake[0], self.state.food)

        self.state.direction = action
        head_x, head_y = self.state.snake[0]
        dx, dy = DIRECTION_VECTORS[action]
        new_head = (head_x + dx, head_y + dy)

        terminated = False
        truncated = False
        reward = self.reward_step
        death_penalty = self.reward_death - self.death_penalty_per_score * self.state.score

        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            reward = death_penalty
            terminated = True
        elif new_head in self._body_without_tail():
            reward = death_penalty
            terminated = True
        else:
            self.state.snake.insert(0, new_head)
            if new_head == self.state.food:
                self.state.score += 1
                self.state.steps_since_food = 0
                reward = self.reward_food
                self.state.food = self._spawn_food()
            else:
                self.state.snake.pop()
                self.state.steps_since_food += 1

            new_dist = self._manhattan(self.state.snake[0], self.state.food)
            reward += self.reward_closer * (old_dist - new_dist)

        self.state.steps += 1
        end_reason = ""
        if terminated:
            end_reason = "death"
        elif self.state.steps >= self.max_steps:
            truncated = True
            end_reason = "max_steps"
        elif (
            self.max_steps_without_food > 0
            and self.state.steps_since_food >= self.max_steps_without_food
        ):
            # Rắn đi lòng vòng không ăn được gì — cắt sớm để khỏi kéo dài train.
            truncated = True
            end_reason = "no_food"

        info = {
            "score": self.state.score,
            "steps": self.state.steps,
            "end_reason": end_reason,
            "truncated": truncated,
            "terminated": terminated,
        }
        return self.encode_observation(), reward, terminated, truncated, info

    def render(self) -> str | None:
        if self.state is None:
            return None

        grid = [["." for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        fx, fy = self.state.food
        grid[fy][fx] = "F"

        for i, (x, y) in enumerate(self.state.snake):
            grid[y][x] = "H" if i == 0 else "s"

        lines = ["+" + "-" * self.grid_size + "+"]
        for row in grid:
            lines.append("|" + "".join(row) + "|")
        lines.append("+" + "-" * self.grid_size + "+")
        lines.append(f"score={self.state.score} steps={self.state.steps}")
        return "\n".join(lines)
