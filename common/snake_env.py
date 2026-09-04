"""Môi trường Snake — tuân chuẩn Gymnasium.

State tabular 7 chiều:
  [food_dx, food_dy, safety_straight, safety_left, safety_right, direction, fill_bucket]

Action: Discrete(3) tương đối theo hướng hiện tại:
  0 = đi thẳng, 1 = rẽ trái, 2 = rẽ phải. Không có lùi 180°.

fill_bucket = mật độ bàn 0–4 (~20% ô/bậc), chung cho 10×10, 15×20 và 30×30.

Điểm mấu chốt là 3 chiều `safety`: với mỗi hướng đi (thẳng / trái / phải), env chạy
BFS thử sau khi di chuyển để biết hướng đó dẫn vào ngõ cụt hay còn đường thoát.
Nhờ vậy agent phân biệt được "rẽ trái thì kẹt, rẽ phải thì sống" — thứ mà một
chỉ số reachable chung cho cả bàn cờ không diễn tả được.
"""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush

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

# Action tương đối — agent chỉ chọn 3 hướng hợp lệ, không lùi 180°.
REL_STRAIGHT = 0
REL_LEFT = 1
REL_RIGHT = 2
N_RELATIVE_ACTIONS = 3
RELATIVE_ACTION_NAMES = {
    REL_STRAIGHT: "straight",
    REL_LEFT: "left",
    REL_RIGHT: "right",
}


def relative_to_absolute(direction: int, relative: int) -> int:
    """Đổi action tương đối (thẳng/trái/phải) thành hướng tuyệt đối trên bàn."""
    if relative == REL_STRAIGHT:
        return direction
    if relative == REL_LEFT:
        return LEFT_TURN[direction]
    if relative == REL_RIGHT:
        return RIGHT_TURN[direction]
    raise ValueError(f"action tương đối không hợp lệ: {relative}")

INITIAL_SNAKE_LENGTH = 3

# Mức an toàn của một hướng đi (dùng cho 3 chiều safety trong state).
#
# Tiêu chí chính là "sau khi đi, đầu rắn còn tới được đuôi mình không".
# Nếu còn tới được đuôi thì rắn luôn có thể bám theo đuôi mà sống tiếp, nên đó
# là ranh giới an toàn tự nhiên — và không phụ thuộc vào độ dài rắn.
SAFETY_COLLISION = 0  # đâm tường/thân ngay lập tức
SAFETY_TRAP = 1  # số ô tới được < độ dài rắn → không đủ chỗ chứa thân, chết chắc
SAFETY_TIGHT = 2  # đủ chỗ nhưng mất dấu đuôi → rủi ro cao
SAFETY_OPEN = 3  # còn tới được đuôi → an toàn


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
        reward_death: float = -40.0,
        reward_self_bite: float = -55.0,
        reward_step: float = -0.05,
        reward_keep_tail: float = 0.08,
        reward_unsafe: float = 0.8,
        death_penalty_per_score: float = 1.0,
        width: int | None = None,
        height: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        super().__init__()
        self.width = int(width if width is not None else grid_size)
        self.height = int(height if height is not None else grid_size)
        self.max_steps = max_steps
        self.max_steps_without_food = max_steps_without_food
        self.reward_food = reward_food
        self.reward_death = reward_death
        self.reward_self_bite = reward_self_bite
        self.reward_step = reward_step
        self.reward_keep_tail = reward_keep_tail
        self.reward_unsafe = reward_unsafe
        self.death_penalty_per_score = death_penalty_per_score
        self.rng = rng or np.random.default_rng()
        self._obs_safety = (SAFETY_OPEN, SAFETY_OPEN, SAFETY_OPEN)

        self.action_space = spaces.Discrete(N_RELATIVE_ACTIONS)
        self.observation_space = spaces.MultiDiscrete([3, 3, 4, 4, 4, 4, 5])

        self.state: SnakeState | None = None

    @property
    def n_actions(self) -> int:
        return int(self.action_space.n)

    def n_state_dims(self) -> tuple[int, ...]:
        return tuple(int(n) for n in self.observation_space.nvec)

    @property
    def n_cells(self) -> int:
        return self.width * self.height

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    @staticmethod
    def max_score(
        width: int,
        height: int | None = None,
        initial_length: int = INITIAL_SNAKE_LENGTH,
    ) -> int:
        if height is None:
            height = width
        return width * height - initial_length

    def _manhattan(self, a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _path_blocked_cells(self) -> set[tuple[int, int]]:
        """
        Ô thân chặn A* — không tính đầu (điểm bắt đầu) và đuôi (sẽ rời nếu không ăn).
        """
        assert self.state is not None
        snake = self.state.snake
        if len(snake) <= 2:
            return set()
        return set(snake[1:-1])

    def astar_path(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        blocked: set[tuple[int, int]] | None = None,
    ) -> list[tuple[int, int]] | None:
        """Đường A* 4 hướng từ start tới goal; None nếu không tới được."""
        if start == goal:
            return [start]
        w, h = self.width, self.height
        if blocked is None:
            blocked = self._path_blocked_cells()

        def heur(p: tuple[int, int]) -> int:
            return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

        heap: list[tuple[int, int, tuple[int, int]]] = [(heur(start), 0, start)]
        best = {start: 0}
        closed: set[tuple[int, int]] = set()
        came_from: dict[tuple[int, int], tuple[int, int]] = {}

        while heap:
            _, cost, node = heappop(heap)
            if node in closed:
                continue
            if node == goal:
                path = [node]
                while path[-1] != start:
                    path.append(came_from[path[-1]])
                path.reverse()
                return path
            closed.add(node)
            x, y = node
            for dx, dy in DIRECTION_VECTORS.values():
                nxt = (x + dx, y + dy)
                nx, ny = nxt
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if nxt in blocked and nxt != goal:
                    continue
                ncost = cost + 1
                if ncost >= best.get(nxt, 10**9):
                    continue
                best[nxt] = ncost
                came_from[nxt] = node
                heappush(heap, (ncost + heur(nxt), ncost, nxt))
        return None

    def astar_distance(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        blocked: set[tuple[int, int]] | None = None,
    ) -> int | None:
        """Độ dài đường A* ngắn nhất; None nếu thức ăn không tới được."""
        path = self.astar_path(start, goal, blocked)
        return None if path is None else len(path) - 1

    def astar_next_action(self) -> int | None:
        """Hướng bước đầu trên A* — chỉ dùng để đo, không chọn action lúc chơi."""
        assert self.state is not None
        path = self.astar_path(self.state.snake[0], self.state.food)
        if path is None or len(path) < 2:
            return None
        nxt = path[1]
        hx, hy = self.state.snake[0]
        for candidate, (dx, dy) in DIRECTION_VECTORS.items():
            if (hx + dx, hy + dy) == nxt:
                return candidate
        return None

    def _safety_for_action(self, action: int) -> int:
        """Safety của action tương đối: 0 thẳng / 1 trái / 2 phải."""
        return int(self._obs_safety[int(action)])

    def _tail_reachable_on(self, snake: list[tuple[int, int]]) -> bool:
        blocked = (
            self._cell_mask(snake[1:-1])
            if len(snake) > 2
            else bytearray(self.n_cells)
        )
        _, ok = self._flood_fill(
            snake[0], blocked, snake[-1], enough=max(len(snake), 1)
        )
        return ok

    def _tail_reachable_now(self) -> bool:
        """Đầu còn BFS tới đuôi không — dùng cho thưởng giữ hành lang."""
        assert self.state is not None
        return self._tail_reachable_on(self.state.snake)

    def _length_bucket(self) -> int:
        """Mật độ bàn 0–4 (mỗi bậc ~20% ô), không phụ thuộc grid_size hay score tuyệt đối."""
        assert self.state is not None
        fill = len(self.state.snake) / float(self.n_cells)
        return min(4, int(fill * 5))

    def _body_without_tail(self) -> list[tuple[int, int]]:
        """Ô thân chặn đầu rắn. Bỏ đuôi vì đuôi sẽ rời đi ở bước tiếp theo."""
        assert self.state is not None
        if len(self.state.snake) <= 1:
            return list(self.state.snake)
        return self.state.snake[:-1]

    def _cell_mask(self, cells) -> bytearray:
        """Đánh dấu ô bị chặn trên lưới phẳng (index = y * width + x)."""
        w = self.width
        mask = bytearray(self.n_cells)
        for x, y in cells:
            mask[y * w + x] = 1
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
        w, h = self.width, self.height
        last_x = w - 1
        last_y = h - 1
        start_idx = start[1] * w + start[0]
        target_idx = target[1] * w + target[0]

        visited = bytearray(self.n_cells)
        visited[start_idx] = 1
        queue = [start_idx]
        count = 1
        found_target = start_idx == target_idx
        i = 0

        while i < len(queue):
            idx = queue[i]
            i += 1
            y, x = divmod(idx, w)

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
            if x < last_x:
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
                n = idx - w
                if not visited[n] and not blocked[n]:
                    visited[n] = 1
                    count += 1
                    if n == target_idx:
                        found_target = True
                    if found_target and count >= enough:
                        return count, True
                    queue.append(n)
            if y < last_y:
                n = idx + w
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
        w, h = self.width, self.height
        hx, hy = self.state.snake[0]
        dx, dy = DIRECTION_VECTORS[direction]
        nx, ny = hx + dx, hy + dy

        if not (0 <= nx < w and 0 <= ny < h):
            return SAFETY_COLLISION
        if mask_body[ny * w + nx]:
            return SAFETY_COLLISION

        # Ăn thì đuôi ở lại (thân dài thêm), không ăn thì đuôi rời đi.
        eats = (nx, ny) == self.state.food
        if eats:
            blocked = mask_body
            new_snake_tail = self.state.snake[-1]
            new_length = len(self.state.snake) + 1
        else:
            blocked = mask_after_move
            new_snake_tail = (
                self.state.snake[-2] if len(self.state.snake) >= 2 else (nx, ny)
            )
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

        if not self._in_bounds(next_x, next_y):
            return True
        return (next_x, next_y) in self._body_without_tail()

    def _spawn_food(self) -> tuple[int, int]:
        assert self.state is not None
        occupied = set(self.state.snake)
        empty = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
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
        mask_after_move = (
            self._cell_mask(snake[:-2])
            if len(snake) > 2
            else bytearray(self.n_cells)
        )

        direction = self.state.direction
        safety_straight = self._safety_level(direction, mask_body, mask_after_move)
        safety_left = self._safety_level(
            LEFT_TURN[direction], mask_body, mask_after_move
        )
        safety_right = self._safety_level(
            RIGHT_TURN[direction], mask_body, mask_after_move
        )
        self._obs_safety = (safety_straight, safety_left, safety_right)

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

        cx = max(2, self.width // 2)
        cy = self.height // 2
        snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]
        self.state = SnakeState(
            snake=snake,
            food=(cx, cy + 1 if cy + 1 < self.height else cy - 1),
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

        action = int(np.clip(int(action), REL_STRAIGHT, REL_RIGHT))
        heading = relative_to_absolute(self.state.direction, action)

        chosen_safety = self._safety_for_action(action)
        best_safety = max(self._obs_safety)

        self.state.direction = heading
        head_x, head_y = self.state.snake[0]
        dx, dy = DIRECTION_VECTORS[heading]
        new_head = (head_x + dx, head_y + dy)

        terminated = False
        truncated = False
        won = False
        reward = self.reward_step
        extra_death = self.death_penalty_per_score * self.state.score
        if best_safety > chosen_safety:
            reward -= self.reward_unsafe * (best_safety - chosen_safety)

        if not self._in_bounds(new_head[0], new_head[1]):
            reward = self.reward_death - extra_death
            terminated = True
        elif new_head in self._body_without_tail():
            reward = self.reward_self_bite - extra_death
            terminated = True
        else:
            self.state.snake.insert(0, new_head)
            if new_head == self.state.food:
                self.state.score += 1
                self.state.steps_since_food = 0
                reward = self.reward_food
                if len(self.state.snake) >= self.n_cells:
                    won = True
                    terminated = True
                else:
                    self.state.food = self._spawn_food()
            else:
                self.state.snake.pop()
                self.state.steps_since_food += 1
                if self._length_bucket() >= 1 and self._tail_reachable_now():
                    reward += self.reward_keep_tail

        self.state.steps += 1
        end_reason = ""
        if won:
            end_reason = "win"
        elif terminated:
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

        grid = [["." for _ in range(self.width)] for _ in range(self.height)]
        fx, fy = self.state.food
        grid[fy][fx] = "F"

        for i, (x, y) in enumerate(self.state.snake):
            grid[y][x] = "H" if i == 0 else "s"

        lines = ["+" + "-" * self.width + "+"]
        for row in grid:
            lines.append("|" + "".join(row) + "|")
        lines.append("+" + "-" * self.width + "+")
        lines.append(f"score={self.state.score} steps={self.state.steps}")
        return "\n".join(lines)
