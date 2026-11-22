"""`match.json`, written by the match stage: broadcast layout, roster, target team, and
the match timeline in video seconds."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from scout.games.first_overlay_2026 import LayoutName
from scout.games.rebuilt2026 import PhaseName
from scout.schemas.common import Alliance, PixelRect, StageFile, ViewName


class TeamSlot(BaseModel):
    """One of the six teams on the scoreboard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    team: int
    alliance: Alliance
    station: Literal[1, 2, 3]


class PhaseSpan(BaseModel):
    """One match phase in video seconds. `active_hubs` is None in a shift when the
    auto winner is unknown."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: PhaseName
    t_start: float
    t_end: float
    active_hubs: list[Alliance] | None


class ClockFit(BaseModel):
    """Line `t = start_t + slope * elapsed` fitted to the clock's tick moments in one period.

    A tick to displayed value v happens `elapsed = period_length - v` seconds into the
    period (the clock shows remaining time rounded up). `ticks` were used, `dropped` were
    rejected as misreads; residuals are over the used ticks. `slope` is the period's mean
    clock rate in video seconds per clock second.

    The line is only the summary: on some broadcasts the clock's rate wanders by a few
    percent, so the match times and phases come from a local line through the nearest
    ticks instead. `timeline_max_abs_s` is the largest tick residual from that timeline
    (None in files written before it existed).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    start_t: float
    slope: float
    ticks: int
    dropped: int
    rms_s: float
    max_abs_s: float
    timeline_max_abs_s: float | None = None


class MatchInfo(StageFile):
    """Overlay layout, roster and timeline for the match in the video."""

    schema_version: Literal[1] = 1
    layout: LayoutName | None = None
    """Overlay layout the stage found (`scout.games.first_overlay_2026`). None in files
    written before it was recorded."""
    scoreboard: PixelRect
    """The scoreboard's rectangle. On the official layout it covers the top of `main`."""
    views: dict[ViewName, PixelRect]
    """Panel rectangles of the camera views in full-frame pixels. They may overlap
    `scoreboard`: the overlay is drawn on top of the camera picture."""
    roster: list[TeamSlot]
    target_team: int
    target_alliance: Alliance
    auto_start_t: float
    auto_end_t: float
    teleop_start_t: float
    teleop_end_t: float
    auto_winner: Alliance | None
    """Alliance whose hub the FMS made inactive in shifts 1 and 3 (more fuel in auto, or the
    FMS tie-break), read from the overlay's active-hub arrows and checked against the fuel
    counters. None when the arrows are unclear or disagree with the counters."""
    phases: list[PhaseSpan]
    clock_fits: dict[Literal["auto", "teleop"], ClockFit]
    """How the auto and teleop times were measured, with fit residuals."""
    clock_rate: float | None = None
    """Video seconds per game-clock second over the whole match (one slope fitted to both
    periods). The FMS clock runs in real time, so a speed measured per video second times
    this is per real second; official broadcasts measure 1.01 to 1.02. None in files
    written before it was recorded."""
