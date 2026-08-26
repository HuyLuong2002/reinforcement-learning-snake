"""Cửa sổ pygame — xem agent chơi Snake."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable, Protocol

import numpy as np
import pygame

from common.snake_env import SnakeEnv

if TYPE_CHECKING:
    pass


class PlayableAgent(Protocol):
    def greedy_policy_action(self, state: np.ndarray) -> int: ...


CELL_SIZE = 40
HUD_HEIGHT = 72
FPS = 8

COLOR_BG = (18, 18, 28)
COLOR_GRID = (45, 45, 60)
COLOR_SNAKE_HEAD = (90, 210, 130)
COLOR_SNAKE_BODY = (55, 170, 100)
COLOR_FOOD = (230, 85, 85)
COLOR_TEXT = (230, 230, 240)
COLOR_MUTED = (140, 140, 155)
COLOR_BTN = (60, 120, 200)
COLOR_BTN_HOVER = (80, 150, 230)
COLOR_BTN_TEXT = (255, 255, 255)
COLOR_OVERLAY = (0, 0, 0, 160)
COLOR_WARN_BG = (120, 70, 20)
COLOR_WARN_BORDER = (255, 180, 60)
COLOR_WARN_TEXT = (255, 220, 120)


class GamePhase(Enum):
    MENU = auto()
    PLAYING = auto()
    GAME_OVER = auto()


@dataclass
class GameUIState:
    phase: GamePhase = GamePhase.MENU
    score: int = 0
    steps: int = 0
    last_reward: float = 0.0
    end_reason: str = ""


class GameWindow:
    def __init__(
        self,
        env: SnakeEnv,
        agent: PlayableAgent | None = None,
        policy_fn: Callable[[np.ndarray], int] | None = None,
        title: str = "Snake RL",
        agent_label: str = "Agent",
        model_missing: bool = False,
        model_path_hint: str = "output/sarsa/training/agent.pkl",
        train_command_hint: str = "python -m agents.sarsa.train",
    ) -> None:
        self.env = env
        self.agent = agent
        self.policy_fn = policy_fn
        self.agent_label = agent_label
        self.model_missing = model_missing
        self.model_path_hint = model_path_hint
        self.train_command_hint = train_command_hint
        self.ui = GameUIState()
        self.obs: np.ndarray | None = None

        board_px = env.grid_size * CELL_SIZE
        self.width = board_px
        self.height = board_px + HUD_HEIGHT

        pygame.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.font_lg = pygame.font.SysFont("consolas", 28, bold=True)

        btn_w, btn_h = 160, 44
        self.start_btn = pygame.Rect(
            (self.width - btn_w) // 2,
            self.height // 2 - btn_h // 2,
            btn_w,
            btn_h,
        )

    def reset_game(self) -> None:
        self.obs, info = self.env.reset()
        self.ui.score = info.get("score", 0)
        self.ui.steps = 0
        self.ui.last_reward = 0.0
        self.ui.end_reason = ""
        self.ui.phase = GamePhase.PLAYING

    def choose_action(self) -> int:
        assert self.obs is not None
        if self.agent is not None:
            return self.agent.greedy_policy_action(self.obs)
        if self.policy_fn is not None:
            return self.policy_fn(self.obs)
        return int(np.random.randint(0, self.env.n_actions))

    def step_game(self) -> None:
        if self.obs is None or self.ui.phase != GamePhase.PLAYING:
            return

        action = self.choose_action()
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.ui.last_reward = reward
        self.ui.score = info.get("score", self.ui.score)
        self.ui.steps = info.get("steps", self.ui.steps)

        if terminated or truncated:
            self.ui.end_reason = info.get("end_reason", "death" if terminated else "max_steps")
            self.ui.phase = GamePhase.GAME_OVER

    def cell_rect(self, col: int, row: int) -> pygame.Rect:
        x, y = col * CELL_SIZE, row * CELL_SIZE
        padding = 2
        return pygame.Rect(x + padding, y + padding, CELL_SIZE - 2 * padding, CELL_SIZE - 2 * padding)

    def draw_grid(self) -> None:
        size = self.env.grid_size
        for i in range(size + 1):
            x = i * CELL_SIZE
            y = i * CELL_SIZE
            pygame.draw.line(self.screen, COLOR_GRID, (x, 0), (x, size * CELL_SIZE))
            pygame.draw.line(self.screen, COLOR_GRID, (0, y), (size * CELL_SIZE, y))

    def draw_snake(self) -> None:
        if self.env.state is None:
            return
        for i, (col, row) in enumerate(self.env.state.snake):
            color = COLOR_SNAKE_HEAD if i == 0 else COLOR_SNAKE_BODY
            pygame.draw.rect(self.screen, color, self.cell_rect(col, row), border_radius=4)

    def draw_food(self) -> None:
        if self.env.state is None:
            return
        col, row = self.env.state.food
        pygame.draw.rect(self.screen, COLOR_FOOD, self.cell_rect(col, row), border_radius=6)

    def draw_warning_banner(self) -> None:
        if not self.model_missing:
            return

        banner_h = 36
        banner = pygame.Rect(0, 0, self.width, banner_h)
        pygame.draw.rect(self.screen, COLOR_WARN_BG, banner)
        pygame.draw.line(self.screen, COLOR_WARN_BORDER, (0, banner_h), (self.width, banner_h), 2)

        text = self.font.render(
            f"Chua co model — chay: {self.train_command_hint}  ({self.model_path_hint})",
            True,
            COLOR_WARN_TEXT,
        )
        self.screen.blit(text, (8, 9))

    def draw_hud(self) -> None:
        hud_y = self.env.grid_size * CELL_SIZE + 10
        banner_offset = 36 if self.model_missing else 0
        hud_y += banner_offset
        agent_label = f"{self.agent_label} (greedy)" if self.agent is not None else "Random (chua co model)"
        status = {
            GamePhase.MENU: "San sang",
            GamePhase.PLAYING: "Dang choi",
            GamePhase.GAME_OVER: "Ket thuc",
        }[self.ui.phase]

        max_sc = SnakeEnv.max_score(self.env.grid_size)
        line1 = (
            f"Score: {self.ui.score}/{max_sc}  |  Steps: {self.ui.steps}/{self.env.max_steps}  "
            f"|  Agent: {agent_label}  |  {status}"
        )
        line2 = "Space/Click Bat dau  |  Esc thoat"
        if self.ui.phase == GamePhase.GAME_OVER:
            reason = {
                "death": "va tuong/than",
                "max_steps": f"het buoc ({self.env.max_steps})",
                "no_food": f"khong an duoc trong {self.env.max_steps_without_food} buoc",
            }.get(self.ui.end_reason, "")
            line2 = f"Game over! Score: {self.ui.score} ({reason})  —  Space/Click choi lai  |  Esc thoat"

        for i, (text, color) in enumerate([(line1, COLOR_TEXT), (line2, COLOR_MUTED)]):
            surface = self.font.render(text, True, color)
            self.screen.blit(surface, (8, hud_y + i * 22))

    def draw_button(self, rect: pygame.Rect, label: str, hovered: bool) -> None:
        color = COLOR_BTN_HOVER if hovered else COLOR_BTN
        pygame.draw.rect(self.screen, color, rect, border_radius=8)
        text = self.font_lg.render(label, True, COLOR_BTN_TEXT)
        text_rect = text.get_rect(center=rect.center)
        self.screen.blit(text, text_rect)

    def draw_overlay_menu(self, mouse_pos: tuple[int, int]) -> None:
        overlay = pygame.Surface((self.width, self.height - HUD_HEIGHT), pygame.SRCALPHA)
        overlay.fill(COLOR_OVERLAY)
        self.screen.blit(overlay, (0, 0))

        title = self.font_lg.render(f"{self.agent_label} Snake", True, COLOR_TEXT)
        title_rect = title.get_rect(center=(self.width // 2, self.height // 2 - 70))
        self.screen.blit(title, title_rect)

        hovered = self.start_btn.collidepoint(mouse_pos)
        label = "Bat dau" if self.ui.phase == GamePhase.MENU else "Choi lai"
        self.draw_button(self.start_btn, label, hovered)

        if self.agent is None:
            warn_lines = [
                f"CHUA CO MODEL {self.agent_label.upper()}",
                "Agent dang dung random policy",
                f"Chay: {self.train_command_hint}",
            ]
            for i, line in enumerate(warn_lines):
                color = COLOR_WARN_TEXT if i == 0 else COLOR_MUTED
                font = self.font_lg if i == 0 else self.font
                hint = font.render(line, True, color)
                hint_rect = hint.get_rect(center=(self.width // 2, self.start_btn.bottom + 28 + i * 24))
                self.screen.blit(hint, hint_rect)

    def draw(self) -> None:
        self.screen.fill(COLOR_BG)
        self.draw_warning_banner()
        self.draw_grid()
        if self.env.state is not None and self.ui.phase != GamePhase.MENU:
            self.draw_food()
            self.draw_snake()
        self.draw_hud()

        if self.ui.phase in (GamePhase.MENU, GamePhase.GAME_OVER):
            mouse_pos = pygame.mouse.get_pos()
            self.draw_overlay_menu(mouse_pos)

    def start_or_restart(self) -> None:
        self.reset_game()

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    if self.ui.phase in (GamePhase.MENU, GamePhase.GAME_OVER):
                        self.start_or_restart()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.ui.phase in (GamePhase.MENU, GamePhase.GAME_OVER):
                    if self.start_btn.collidepoint(event.pos):
                        self.start_or_restart()

        return True

    def tick(self) -> None:
        if self.ui.phase == GamePhase.PLAYING:
            self.step_game()

    def run(self) -> None:
        running = True
        while running:
            running = self.handle_events()
            self.tick()
            self.draw()
            pygame.display.flip()
            self.clock.tick(FPS)

        pygame.quit()
