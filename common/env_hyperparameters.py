"""Tham số môi trường Snake — dùng chung cho mọi agent (SARSA, Q-learning, ...)."""

from __future__ import annotations

from dataclasses import dataclass, replace

# Train mặc định 15×20. Chơi / CLI: 10×10, 15×20, 30×30 (cùng Q-table, state tương đối).
PLAYABLE_GRIDS: tuple[int, ...] = (10, 30)
PLAYABLE_LEVELS: tuple[tuple[int, int], ...] = ((10, 10), (15, 20), (30, 30))
PLAY_GRID_CHOICES: tuple[str, ...] = ("10", "30", "10x10", "15x20", "30x30")
DEFAULT_TRAIN_LEVEL: tuple[int, int] = (15, 20)
BASE_GRID: int = 10
BASE_MAX_STEPS: int = 2000
BASE_STEPS_WITHOUT_FOOD: int = 100


def grid_label(width: int, height: int) -> str:
    return f"{width}x{height}"


def parse_play_grid(value: str) -> tuple[int, int]:
    """Đổi --grid 10 / 15x20 thành (width, height)."""
    aliases: dict[str, tuple[int, int]] = {
        "10": (10, 10),
        "30": (30, 30),
        "10x10": (10, 10),
        "15x20": (15, 20),
        "30x30": (30, 30),
    }
    key = value.lower().replace(" ", "")
    if key not in aliases:
        raise ValueError(f"Màn không hỗ trợ: {value}")
    return aliases[key]


@dataclass
class SnakeEnvHyperparameters:
    """Tham số môi trường game."""

    grid_size: int = 10  # chỉ dùng khi bàn vuông (CLI --grid 10)
    width: int = 10
    height: int = 10
    # Trần số bước/episode. Cần đủ lớn cho score cao:
    # ~15-20 bước/quả khi rắn dài → score 50 cần cỡ 800-1000 bước.
    max_steps: int = 2000
    # Cắt episode nếu đi bấy nhiêu bước mà không ăn được gì.
    # Ngăn rắn đi vòng vô tận làm train chậm, mà vẫn cho phép đường đi dài.
    max_steps_without_food: int = 100
    reward_food: float = 10.0  # thưởng khi ăn thức ăn
    reward_death: float = -40.0  # phạt đụng tường (cộng thêm theo score)
    reward_self_bite: float = -55.0  # phạt tự cắn — nặng hơn tường
    reward_step: float = -0.05  # phạt mỗi bước → đi dài / đi vòng bị lỗ
    # Sống sót (BFS): agent vẫn tự chọn action. A* không còn trong reward
    # (chỉ dùng lúc analyze để đo đường đi).
    reward_keep_tail: float = 0.08  # thưởng còn BFS tới đuôi khi fill ≳ 20%
    # Phạt mỗi bậc safety thấp hơn hướng tốt nhất (OPEN=3 … COLLISION=0).
    reward_unsafe: float = 0.8
    death_penalty_per_score: float = 1.0  # chết ở score 20 → phạt thêm -20

    def for_grid(self, grid_size: int) -> SnakeEnvHyperparameters:
        """Clone config lưới vuông (train 10 hoặc 30)."""
        return self.for_shape(grid_size, grid_size)

    def for_shape(self, width: int, height: int) -> SnakeEnvHyperparameters:
        """Clone config theo kích thước bàn. max_steps scale theo diện tích."""
        if width < 4 or height < 4:
            raise ValueError(f"lưới {width}x{height} quá nhỏ (rắn ban đầu dài 3).")
        area_ratio = (width * height) / (BASE_GRID * BASE_GRID)
        span = max(width, height)
        return replace(
            self,
            grid_size=width if width == height else 0,
            width=width,
            height=height,
            max_steps=max(200, int(BASE_MAX_STEPS * area_ratio)),
            max_steps_without_food=max(
                40, int(BASE_STEPS_WITHOUT_FOOD * span / BASE_GRID)
            ),
        )

    def create_env(self, rng=None):
        """Tạo SnakeEnv đúng width × height (không suy từ grid_size)."""
        from common.snake_env import SnakeEnv

        return SnakeEnv(
            width=self.width,
            height=self.height,
            max_steps=self.max_steps,
            max_steps_without_food=self.max_steps_without_food,
            reward_food=self.reward_food,
            reward_death=self.reward_death,
            reward_self_bite=self.reward_self_bite,
            reward_step=self.reward_step,
            reward_keep_tail=self.reward_keep_tail,
            reward_unsafe=self.reward_unsafe,
            death_penalty_per_score=self.death_penalty_per_score,
            rng=rng,
        )
