"""What every grade module shares: the JSON type, the image budget, the frame tool's names,
the `grade_work/` paths and span arithmetic."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from scout.run import RunDir

JSON = dict[str, object]
IMAGE_CAP = 20
TOOL = "frames"
SERVER = "scout"


def work(run: RunDir) -> Path:
    return run.path / "grade_work"


def frame_log(run: RunDir) -> Path:
    return work(run) / "frames.jsonl"


def fmt(v: float) -> str:
    return f"{v:.2f}" if abs(v) < 1 else f"{v:.1f}"


def overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 <= b1 + 1e-6 and b0 <= a1 + 1e-6


def union_s(a: float, b: float, spans: Iterable[tuple[float, float]]) -> float:
    """Seconds of [a, b] covered by the union of `spans`."""
    total, at = 0.0, a
    for s0, s1 in sorted((max(a, s0), min(b, s1)) for s0, s1 in spans):
        if s1 > at and s1 > s0:
            total += s1 - max(at, s0)
            at = s1
    return total
