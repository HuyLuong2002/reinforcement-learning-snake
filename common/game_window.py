"""Cửa sổ pygame — xem agent chơi Snake."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Protocol

import numpy as np
import pygame

from common.env_hyperparameters import PLAYABLE_LEVELS
from common.policy import select_action
from common.random_policy import choose_random_action
from common.run_store import ModelRun
from common.snake_env import SnakeEnv

MIN_CELL_SIZE = 12
MAX_CELL_SIZE = 48
MAX_BOARD_PX = 840
MIN_WINDOW_WIDTH = 880
HUD_HEIGHT = 108
LEVEL_HOTKEYS = (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4)

COLOR_BG = (16, 16, 24)
COLOR_CELL_A = (28, 28, 42)
COLOR_CELL_B = (22, 22, 34)
COLOR_BOARD_BORDER = (72, 76, 98)
COLOR_SNAKE_HEAD = (110, 220, 140)
COLOR_SNAKE_BODY = (58, 168, 108)
COLOR_FOOD = (232, 88, 88)
COLOR_TEXT = (232, 232, 242)
COLOR_MUTED = (148, 148, 164)
COLOR_BTN = (60, 120, 200)
COLOR_BTN_HOVER = (80, 150, 230)
COLOR_BTN_TEXT = (255, 255, 255)
COLOR_OVERLAY = (0, 0, 0, 168)
COLOR_WARN_BG = (120, 70, 20)
COLOR_WARN_BORDER = (255, 180, 60)
COLOR_WARN_TEXT = (255, 220, 120)
COLOR_CHECK_BOX = (210, 214, 228)
COLOR_CHECK_ON = (80, 170, 120)
COLOR_DROPDOWN_BG = (32, 36, 52)
COLOR_DROPDOWN_SEL = (60, 120, 200)
DROPDOWN_VISIBLE = 6
SELECT_H = 40


class PlayableAgent(Protocol):
    def greedy_policy_action(self, state: np.ndarray) -> int: ...

    def choose_action(
        self,
        state: np.ndarray,
        greedy: bool = False,
        forbidden: int | None = None,
    ) -> int: ...


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


def cell_size_for_board(
    width: int,
    height: int,
    screen_w: int,
    screen_h: int,
    banner_h: int = 0,
) -> int:
    """Chọn cạnh ô sao cho bàn 10×10 to rõ, 15×20 / 30×30 vẫn vừa màn hình."""
    if width <= 0 or height <= 0:
        raise ValueError(f"board={width}x{height}")
    avail_h = int(screen_h * 0.86) - HUD_HEIGHT - banner_h - 48
    avail_w = int(screen_w * 0.90) - 48
    by_w = max(avail_w, MIN_CELL_SIZE) // width
    by_h = max(avail_h, MIN_CELL_SIZE) // height
    by_cap = MAX_BOARD_PX // max(width, height)
    return max(MIN_CELL_SIZE, min(MAX_CELL_SIZE, by_w, by_h, by_cap))


def fps_for_board(width: int, height: int) -> int:
    """Bàn lớn nhiều bước hơn — tăng FPS để xem không bị chậm."""
    return 28 if max(width, height) >= 20 else 10


class GameWindow:
    def __init__(
        self,
        env: SnakeEnv,
        agent: PlayableAgent | None = None,
        policy_fn: Callable[[np.ndarray], int] | None = None,
        title: str = "Snake RL",
        agent_label: str = "Agent",
        model_missing: bool = False,
        model_path_hint: str = "output/sarsa/<thoi-gian>/agent.pkl",
        train_command_hint: str = "python -m agents.sarsa.train",
        make_env: Callable[[int, int], SnakeEnv] | None = None,
        replay_seed: int | None = None,
        list_runs_fn: Callable[[], list[ModelRun]] | None = None,
        load_run_fn: Callable[[ModelRun], PlayableAgent | None] | None = None,
        use_agent: bool = False,
    ) -> None:
        self.env = env
        self.make_env = make_env
        self.replay_seed = replay_seed
        self._episode_rng = np.random.default_rng()
        self.agent = agent
        self.policy_fn = policy_fn
        self.list_runs_fn = list_runs_fn
        self.load_run_fn = load_run_fn
        self.model_runs: list[ModelRun] = []
        self.selected_run_i = 0
        self.dropdown_open = False
        self.dropdown_scroll = 0
        self.use_agent = use_agent
        self.agent_label = agent_label
        self.model_missing = model_missing
        self.model_path_hint = model_path_hint
        self.train_command_hint = train_command_hint
        self.ui = GameUIState()
        self.obs: np.ndarray | None = None
        self.level_btns: dict[tuple[int, int], pygame.Rect] = {}
        self.use_agent_hit = pygame.Rect(0, 0, 300, 40)
        self.run_select_hit = pygame.Rect(0, 0, 300, SELECT_H)
        self._dropdown_panel = pygame.Rect(0, 0, 0, 0)
        self._option_hits: list[tuple[int, pygame.Rect]] = []
        self.start_btn = pygame.Rect(0, 0, 180, 48)
        self.back_btn = pygame.Rect(0, 0, 180, 40)
        self.board_origin = (0, 0)
        self.cell_size = MAX_CELL_SIZE
        self.fps = fps_for_board(env.width, env.height)
        self._desktop: tuple[int, int] = (1280, 720)

        pygame.init()
        pygame.display.set_caption(title)
        # Lấy kích thước màn hình TRƯỚC set_mode — sau đó Info() trả về cửa sổ hiện tại.
        self._desktop = self._read_display_size()
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.font_md = pygame.font.SysFont("consolas", 20, bold=True)
        self.font_lg = pygame.font.SysFont("consolas", 28, bold=True)
        if self.use_agent:
            self._ensure_model_loaded()
        self._resize_window()

    def _show_missing_banner(self) -> bool:
        return self.use_agent and (self.model_missing or self.agent is None)

    def using_trained_agent(self) -> bool:
        return self.use_agent and self.agent is not None

    def _refresh_runs(self) -> None:
        if self.list_runs_fn is None:
            self.model_runs = []
            return
        self.model_runs = self.list_runs_fn()
        if self.selected_run_i >= len(self.model_runs):
            self.selected_run_i = 0
        max_scroll = max(0, len(self.model_runs) - DROPDOWN_VISIBLE)
        self.dropdown_scroll = min(self.dropdown_scroll, max_scroll)

    def _load_selected_run(self) -> None:
        if not self.model_runs or self.load_run_fn is None:
            self.agent = None
            self.model_missing = True
            return
        run = self.model_runs[self.selected_run_i]
        self.agent = self.load_run_fn(run)
        self.model_missing = self.agent is None
        self.model_path_hint = str(run.model_path)

    def _ensure_model_loaded(self) -> None:
        self._refresh_runs()
        self._load_selected_run()

    def select_run(self, index: int) -> None:
        if not self.model_runs:
            return
        self.selected_run_i = index % len(self.model_runs)
        self._load_selected_run()
        self.dropdown_open = False

    def set_use_agent(self, enabled: bool) -> None:
        self.use_agent = enabled
        self.dropdown_open = False
        if enabled:
            self._ensure_model_loaded()
        self._resize_window()

    def toggle_use_agent(self) -> None:
        self.set_use_agent(not self.use_agent)

    def _read_display_size(self) -> tuple[int, int]:
        info = pygame.display.Info()
        w, h = int(info.current_w), int(info.current_h)
        if w <= 0 or h <= 0:
            return 1280, 720
        return w, h

    def _resize_window(self) -> None:
        banner_h = 36 if self._show_missing_banner() else 0
        screen_w, screen_h = self._desktop
        self.cell_size = cell_size_for_board(
            self.env.width, self.env.height, screen_w, screen_h, banner_h
        )
        self.fps = fps_for_board(self.env.width, self.env.height)
        board_w = self.env.width * self.cell_size
        board_h = self.env.height * self.cell_size
        self.width = max(board_w, min(MIN_WINDOW_WIDTH, screen_w))
        self.height = banner_h + board_h + HUD_HEIGHT
        self.board_origin = ((self.width - board_w) // 2, banner_h)
        self.screen = pygame.display.set_mode((self.width, self.height))
        self._layout_buttons()

    def _layout_buttons(self) -> None:
        btn_w, btn_h = 300, 46
        chk_h = 40
        cx = self.width // 2
        ox, oy = self.board_origin
        board_h = self.env.height * self.cell_size
        n_levels = len(PLAYABLE_LEVELS)
        select_block = (SELECT_H + 10) if self.use_agent else 0
        stack_h = chk_h + 16 + select_block + n_levels * btn_h + max(0, n_levels - 1) * 12
        menu_y = oy + max(80, (board_h - stack_h) // 2)
        self.use_agent_hit = pygame.Rect(cx - btn_w // 2, menu_y, btn_w, chk_h)
        self.run_select_hit = pygame.Rect(
            cx - btn_w // 2, menu_y + chk_h + 10, btn_w, SELECT_H
        )
        self.level_btns = {}
        levels_y = menu_y + chk_h + 16 + select_block
        for i, shape in enumerate(PLAYABLE_LEVELS):
            self.level_btns[shape] = pygame.Rect(
                cx - btn_w // 2,
                levels_y + i * (btn_h + 12),
                btn_w,
                btn_h,
            )
        self.start_btn = pygame.Rect(cx - 90, oy + board_h // 2 - 28, 180, 48)
        self.back_btn = pygame.Rect(cx - 90, self.start_btn.bottom + 12, 180, 40)

    def apply_level(self, width: int, height: int | None = None) -> None:
        if self.make_env is None:
            raise RuntimeError("Chua truyen make_env — khong doi duoc man choi.")
        if height is None:
            height = width
        self.env = self.make_env(width, height)
        self._resize_window()
        self.reset_game()

    def reset_game(self) -> None:
        if self.replay_seed is not None:
            seed = self.replay_seed
        else:
            seed = int(self._episode_rng.integers(0, 2**31 - 1))
        self.obs, info = self.env.reset(seed=seed)
        self.ui.score = info.get("score", 0)
        self.ui.steps = 0
        self.ui.last_reward = 0.0
        self.ui.end_reason = ""
        self.ui.phase = GamePhase.PLAYING

    def choose_action(self) -> int:
        assert self.obs is not None
        if self.using_trained_agent():
            assert self.agent is not None
            return select_action(self.env, self.agent, self.obs, greedy=True)
        return choose_random_action(self.env.n_actions, self._episode_rng)

    def step_game(self) -> None:
        if self.obs is None or self.ui.phase != GamePhase.PLAYING:
            return

        action = self.choose_action()
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.ui.last_reward = reward
        self.ui.score = info.get("score", self.ui.score)
        self.ui.steps = info.get("steps", self.ui.steps)

        if terminated or truncated:
            self.ui.end_reason = info.get(
                "end_reason", "death" if terminated else "max_steps"
            )
            self.ui.phase = GamePhase.GAME_OVER

    def cell_rect(self, col: int, row: int) -> pygame.Rect:
        ox, oy = self.board_origin
        cs = self.cell_size
        x, y = ox + col * cs, oy + row * cs
        padding = 0 if cs < 16 else (1 if cs < 28 else max(2, cs // 16))
        return pygame.Rect(
            x + padding, y + padding, cs - 2 * padding, cs - 2 * padding
        )

    def draw_board(self) -> None:
        cols, rows = self.env.width, self.env.height
        ox, oy = self.board_origin
        cs = self.cell_size
        board = pygame.Rect(ox, oy, cols * cs, rows * cs)
        pygame.draw.rect(self.screen, COLOR_CELL_B, board)
        for row in range(rows):
            for col in range(cols):
                if (col + row) % 2 == 0:
                    cell = pygame.Rect(ox + col * cs, oy + row * cs, cs, cs)
                    pygame.draw.rect(self.screen, COLOR_CELL_A, cell)
        pygame.draw.rect(self.screen, COLOR_BOARD_BORDER, board, width=2)

    def draw_snake(self) -> None:
        if self.env.state is None:
            return
        radius = max(1, self.cell_size // 6)
        for i, (col, row) in enumerate(self.env.state.snake):
            color = COLOR_SNAKE_HEAD if i == 0 else COLOR_SNAKE_BODY
            pygame.draw.rect(
                self.screen, color, self.cell_rect(col, row), border_radius=radius
            )

    def draw_food(self) -> None:
        if self.env.state is None:
            return
        col, row = self.env.state.food
        rect = self.cell_rect(col, row)
        pygame.draw.circle(
            self.screen,
            COLOR_FOOD,
            rect.center,
            max(3, min(rect.width, rect.height) // 2),
        )

    def draw_warning_banner(self) -> None:
        if not self._show_missing_banner():
            return

        banner_h = 36
        banner = pygame.Rect(0, 0, self.width, banner_h)
        pygame.draw.rect(self.screen, COLOR_WARN_BG, banner)
        pygame.draw.line(
            self.screen, COLOR_WARN_BORDER, (0, banner_h), (self.width, banner_h), 2
        )

        text = self.font.render(
            f"Chua co model — chay: {self.train_command_hint}  ({self.model_path_hint})",
            True,
            COLOR_WARN_TEXT,
        )
        self.screen.blit(text, (8, 9))

    def draw_hud(self) -> None:
        ox, oy = self.board_origin
        hud_y = oy + self.env.height * self.cell_size + 14
        agent_label = (
            f"{self.agent_label} (greedy Q)"
            if self.using_trained_agent()
            else "Random"
        )
        status = {
            GamePhase.MENU: "San sang",
            GamePhase.PLAYING: "Dang choi",
            GamePhase.GAME_OVER: "Ket thuc",
        }[self.ui.phase]

        max_sc = SnakeEnv.max_score(self.env.width, self.env.height)
        shortcuts = " / ".join(
            f"{i + 1}={w}x{h}" for i, (w, h) in enumerate(PLAYABLE_LEVELS)
        )
        line1 = (
            f"Man {self.env.width}x{self.env.height}  |  "
            f"Score: {self.ui.score}/{max_sc}  |  Steps: {self.ui.steps}/{self.env.max_steps}  "
            f"|  {status}"
        )
        line2 = f"Menu: bam {shortcuts}  |  A = load model  |  chon lan train  |  Esc thoat"
        if self.ui.phase == GamePhase.PLAYING:
            line2 = f"Agent: {agent_label}  |  Esc thoat"
        if self.ui.phase == GamePhase.GAME_OVER:
            reason = {
                "death": "va tuong/than",
                "max_steps": f"het buoc ({self.env.max_steps})",
                "no_food": f"khong an duoc trong {self.env.max_steps_without_food} buoc",
                "win": "thang (lap ban)",
            }.get(self.ui.end_reason, "")
            line2 = (
                f"Game over! Score: {self.ui.score} ({reason})  —  "
                f"Choi lai / Chon man  |  Esc thoat"
            )

        for i, (text, color) in enumerate([(line1, COLOR_TEXT), (line2, COLOR_MUTED)]):
            surface = self.font.render(text, True, color)
            self.screen.blit(surface, (16, hud_y + i * 24))

    def draw_button(
        self, rect: pygame.Rect, label: str, hovered: bool, large: bool = True
    ) -> None:
        color = COLOR_BTN_HOVER if hovered else COLOR_BTN
        pygame.draw.rect(self.screen, color, rect, border_radius=8)
        font = self.font_lg if large else self.font_md
        text = font.render(label, True, COLOR_BTN_TEXT)
        text_rect = text.get_rect(center=rect.center)
        self.screen.blit(text, text_rect)

    def draw_agent_checkbox(self, mouse_pos: tuple[int, int]) -> None:
        hit = self.use_agent_hit
        hovered = hit.collidepoint(mouse_pos)
        pygame.draw.rect(self.screen, COLOR_BTN_HOVER if hovered else COLOR_BTN, hit, border_radius=8)
        box = pygame.Rect(hit.x + 12, hit.y + (hit.h - 20) // 2, 20, 20)
        pygame.draw.rect(self.screen, COLOR_CHECK_BOX, box, border_radius=4)
        if self.use_agent:
            inner = box.inflate(-8, -8)
            pygame.draw.rect(self.screen, COLOR_CHECK_ON, inner, border_radius=2)
        label = f"Load model {self.agent_label}"
        text = self.font_md.render(label, True, COLOR_BTN_TEXT)
        self.screen.blit(text, (box.right + 12, hit.y + (hit.h - text.get_height()) // 2))

    def draw_run_select(self, mouse_pos: tuple[int, int]) -> None:
        if not self.use_agent:
            self._option_hits = []
            self._dropdown_panel = pygame.Rect(0, 0, 0, 0)
            return

        hit = self.run_select_hit
        hovered = hit.collidepoint(mouse_pos)
        pygame.draw.rect(
            self.screen, COLOR_BTN_HOVER if hovered else COLOR_BTN, hit, border_radius=8
        )
        if self.model_runs:
            label = self.model_runs[self.selected_run_i].label()
        else:
            label = "Chua co lan train"
        caret = "^" if self.dropdown_open else "v"
        text = self.font.render(label, True, COLOR_BTN_TEXT)
        caret_s = self.font_md.render(caret, True, COLOR_BTN_TEXT)
        self.screen.blit(
            text, (hit.x + 12, hit.y + (hit.h - text.get_height()) // 2)
        )
        self.screen.blit(
            caret_s,
            (hit.right - 22, hit.y + (hit.h - caret_s.get_height()) // 2),
        )

    def draw_run_dropdown(self, mouse_pos: tuple[int, int]) -> None:
        if not self.use_agent or not self.dropdown_open or not self.model_runs:
            self._option_hits = []
            self._dropdown_panel = pygame.Rect(0, 0, 0, 0)
            return

        hit = self.run_select_hit
        visible = min(DROPDOWN_VISIBLE, len(self.model_runs))
        opt_h = 36
        panel = pygame.Rect(hit.x, hit.bottom + 4, hit.w, visible * opt_h)
        pygame.draw.rect(self.screen, COLOR_DROPDOWN_BG, panel, border_radius=6)
        pygame.draw.rect(self.screen, COLOR_BOARD_BORDER, panel, width=1, border_radius=6)
        start = self.dropdown_scroll
        end = min(start + visible, len(self.model_runs))
        self._option_hits = []
        for row, run_i in enumerate(range(start, end)):
            rect = pygame.Rect(panel.x, panel.y + row * opt_h, panel.w, opt_h)
            if run_i == self.selected_run_i or rect.collidepoint(mouse_pos):
                pygame.draw.rect(self.screen, COLOR_DROPDOWN_SEL, rect)
            opt = self.font.render(self.model_runs[run_i].label(), True, COLOR_BTN_TEXT)
            self.screen.blit(
                opt, (rect.x + 12, rect.y + (rect.h - opt.get_height()) // 2)
            )
            self._option_hits.append((run_i, rect))
        self._dropdown_panel = panel

    def draw_overlay_menu(self, mouse_pos: tuple[int, int]) -> None:
        ox, oy = self.board_origin
        overlay = pygame.Surface(
            (self.width, oy + self.env.height * self.cell_size), pygame.SRCALPHA
        )
        overlay.fill(COLOR_OVERLAY)
        self.screen.blit(overlay, (0, 0))

        title = self.font_lg.render(f"{self.agent_label} Snake", True, COLOR_TEXT)
        title_rect = title.get_rect(center=(self.width // 2, oy + 36))
        self.screen.blit(title, title_rect)

        if self.ui.phase == GamePhase.MENU:
            self.draw_agent_checkbox(mouse_pos)
            self.draw_run_select(mouse_pos)
            for (w, h), rect in self.level_btns.items():
                win_score = SnakeEnv.max_score(w, h)
                self.draw_button(
                    rect,
                    f"{w}x{h}  (thang {win_score})",
                    rect.collidepoint(mouse_pos),
                    large=False,
                )
            self.draw_run_dropdown(mouse_pos)
        else:
            self.draw_button(
                self.start_btn,
                "Choi lai",
                self.start_btn.collidepoint(mouse_pos),
            )
            self.draw_button(
                self.back_btn,
                "Chon man",
                self.back_btn.collidepoint(mouse_pos),
            )

        if self.use_agent and self.agent is None:
            warn = self.font.render("Chua co model — random policy", True, COLOR_WARN_TEXT)
            self.screen.blit(
                warn, warn.get_rect(center=(self.width // 2, self.height - HUD_HEIGHT - 18))
            )

    def draw(self) -> None:
        self.screen.fill(COLOR_BG)
        self.draw_warning_banner()
        self.draw_board()
        if self.env.state is not None and self.ui.phase != GamePhase.MENU:
            self.draw_food()
            self.draw_snake()
        self.draw_hud()

        if self.ui.phase in (GamePhase.MENU, GamePhase.GAME_OVER):
            mouse_pos = pygame.mouse.get_pos()
            self.draw_overlay_menu(mouse_pos)

    def start_or_restart(self) -> None:
        self.reset_game()

    def back_to_menu(self) -> None:
        self.ui.phase = GamePhase.MENU
        self.dropdown_open = False
        self.obs = None
        self.ui.score = 0
        self.ui.steps = 0
        self.ui.end_reason = ""

    def handle_events(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            if event.type == pygame.MOUSEWHEEL and self.ui.phase == GamePhase.MENU:
                if self.use_agent and self.model_runs:
                    mouse = pygame.mouse.get_pos()
                    if self.dropdown_open and self._dropdown_panel.collidepoint(mouse):
                        max_scroll = max(0, len(self.model_runs) - DROPDOWN_VISIBLE)
                        self.dropdown_scroll = max(
                            0, min(max_scroll, self.dropdown_scroll - event.y)
                        )
                    elif self.run_select_hit.collidepoint(mouse):
                        self.select_run(self.selected_run_i - event.y)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                if self.ui.phase == GamePhase.MENU:
                    if event.key == pygame.K_a:
                        self.toggle_use_agent()
                    elif self.use_agent and event.key in (
                        pygame.K_LEFT,
                        pygame.K_RIGHT,
                        pygame.K_UP,
                        pygame.K_DOWN,
                    ):
                        delta = (
                            1
                            if event.key in (pygame.K_RIGHT, pygame.K_DOWN)
                            else -1
                        )
                        self.select_run(self.selected_run_i + delta)
                    else:
                        for key, (w, h) in zip(LEVEL_HOTKEYS, PLAYABLE_LEVELS):
                            if event.key == key:
                                self.apply_level(w, h)
                                break
                elif self.ui.phase == GamePhase.GAME_OVER:
                    if event.key in (pygame.K_SPACE, pygame.K_RETURN):
                        self.start_or_restart()
                    elif event.key == pygame.K_m:
                        self.back_to_menu()

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.ui.phase == GamePhase.MENU:
                    if self.dropdown_open:
                        picked = False
                        for run_i, rect in self._option_hits:
                            if rect.collidepoint(event.pos):
                                self.select_run(run_i)
                                picked = True
                                break
                        if not picked:
                            self.dropdown_open = False
                            if self.run_select_hit.collidepoint(event.pos):
                                pass
                            elif self.use_agent_hit.collidepoint(event.pos):
                                self.toggle_use_agent()
                            elif not self._dropdown_panel.collidepoint(event.pos):
                                for (w, h), rect in self.level_btns.items():
                                    if rect.collidepoint(event.pos):
                                        self.apply_level(w, h)
                                        break
                    elif self.use_agent_hit.collidepoint(event.pos):
                        self.toggle_use_agent()
                    elif self.use_agent and self.run_select_hit.collidepoint(event.pos):
                        self._refresh_runs()
                        self.dropdown_open = True
                        self.dropdown_scroll = max(
                            0, self.selected_run_i - DROPDOWN_VISIBLE + 1
                        )
                    else:
                        for (w, h), rect in self.level_btns.items():
                            if rect.collidepoint(event.pos):
                                self.apply_level(w, h)
                                break
                elif self.ui.phase == GamePhase.GAME_OVER:
                    if self.start_btn.collidepoint(event.pos):
                        self.start_or_restart()
                    elif self.back_btn.collidepoint(event.pos):
                        self.back_to_menu()

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
            self.clock.tick(self.fps)

        pygame.quit()
