"""The match clock and who is who: period and scoreboard clock at a video time, the phase,
the target's own active time, a team's relation to the target and which corner camera shows
the target's own end."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from scout.camera import Camera
from scout.games.rebuilt2026 import FIELD_LENGTH_M, FIELD_WIDTH_M
from scout.schemas.match import MatchInfo

Corner = Literal["corner_left", "corner_right"]


def clock(match: MatchInfo, t: float) -> tuple[str, str]:
    """Period and scoreboard clock at video time `t`, as facts.json writes them."""
    if t < match.auto_start_t:
        return "pre_match", "0:20"
    if t < match.auto_end_t:
        fit, length, period = match.clock_fits["auto"], 20.0, "auto"
    elif t < match.teleop_start_t:
        return "pause", "2:20"
    elif t <= match.teleop_end_t:
        fit, length, period = match.clock_fits["teleop"], 140.0, "teleop"
    else:
        return "post_match", "0:00"
    s = max(0, math.ceil(length - (t - fit.start_t) / fit.slope - 1e-6))
    return period, f"{s // 60}:{s % 60:02d}"


def phase(match: MatchInfo, t: float) -> str:
    for p in match.phases:
        if p.t_start <= t < p.t_end:
            return p.name
    return "-"


def own_active(match: MatchInfo) -> list[tuple[float, float]]:
    """Teleop spans with the target's hub active."""
    return [
        (p.t_start, p.t_end)
        for p in match.phases
        if p.name != "auto" and p.active_hubs is not None and match.target_alliance in p.active_hubs
    ]


def relation(match: MatchInfo, team: int) -> Literal["partner", "opponent", "unknown"]:
    slot = next((s for s in match.roster if s.team == team), None)
    if slot is None:
        return "unknown"
    return "partner" if slot.alliance == match.target_alliance else "opponent"


def home_view(cam: Camera, match: MatchInfo) -> Corner:
    """The corner panel showing the target's own end: the side of the main view its own
    zone projects to."""
    x = 2.0 if match.target_alliance == "blue" else FIELD_LENGTH_M - 2.0
    u = float(cam.project(np.array([[x, FIELD_WIDTH_M / 2, 0.0]]))[0, 0])
    main = match.views["main"]
    return "corner_left" if u < main.x + main.w / 2 else "corner_right"
