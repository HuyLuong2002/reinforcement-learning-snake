"""Thư mục output mỗi lần train — tên folder theo thời gian.

Ví dụ: output/sarsa/2026-08-27_16-33-57/
Lần train sau không ghi đè lần trước. Game chọn folder để load.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_STAMP_FMT = "%Y-%m-%d_%H-%M-%S"
MODEL_FILES = ("agent.pkl", "agent_best.pkl", "agent_last.pkl")


def agent_output_root(agent_name: str) -> Path:
    return PROJECT_ROOT / "output" / agent_name


def new_run_dir(agent_name: str) -> Path:
    """Tạo folder output mới theo thời điểm bắt đầu train."""
    root = agent_output_root(agent_name)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime(RUN_STAMP_FMT)
    path = root / stamp
    suffix = 2
    while path.exists():
        path = root / f"{stamp}_{suffix}"
        suffix += 1
    path.mkdir(parents=True)
    return path


def model_file_in(run_dir: Path) -> Path | None:
    """File model ưu tiên trong một lần train: agent.pkl → best → last."""
    for name in MODEL_FILES:
        path = run_dir / name
        if path.is_file():
            return path
    return None


def _read_metadata(run_dir: Path) -> dict:
    meta = run_dir / "agent_metadata.json"
    if not meta.is_file():
        return {}
    try:
        data = json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _eval_score(data: dict) -> float | None:
    value = data.get("best_eval_score")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _grid_label(data: dict) -> str | None:
    grid = data.get("grid")
    if isinstance(grid, str) and grid:
        return grid
    width, height = data.get("grid_width"), data.get("grid_height")
    if isinstance(width, int) and isinstance(height, int):
        return f"{width}x{height}"
    return None


@dataclass(frozen=True)
class ModelRun:
    name: str
    dir: Path
    model_path: Path
    eval_score: float | None = None
    grid: str | None = None

    def label(self) -> str:
        """Nhãn trên select: ngày giờ + lưới + điểm eval nếu có."""
        parts = self.name.split("_")
        pretty = self.name
        if len(parts) >= 2 and len(parts[0]) == 10:
            pretty = f"{parts[0]} {parts[1][:5].replace('-', ':')}"
        if self.grid:
            pretty = f"{pretty}  {self.grid}"
        if self.eval_score is None:
            return pretty
        return f"{pretty}  ({self.eval_score:.1f})"


def list_model_runs(agent_name: str) -> list[ModelRun]:
    """Mọi lần train có file model, mới nhất trước."""
    root = agent_output_root(agent_name)
    if not root.is_dir():
        return []
    runs: list[ModelRun] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        model = model_file_in(child)
        if model is None:
            continue
        data = _read_metadata(child)
        runs.append(
            ModelRun(
                name=child.name,
                dir=child,
                model_path=model,
                eval_score=_eval_score(data),
                grid=_grid_label(data),
            )
        )
    runs.sort(key=lambda r: r.dir.stat().st_mtime, reverse=True)
    return runs


def latest_model_path(agent_name: str) -> Path | None:
    runs = list_model_runs(agent_name)
    return runs[0].model_path if runs else None
