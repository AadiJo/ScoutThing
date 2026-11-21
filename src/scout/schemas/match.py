"""`match.json`, written by the match stage: broadcast layout, roster, target team, and
the match timeline in video seconds."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from scout.games.rebuilt2026 import PhaseName
from scout.schemas.common import Alliance, PixelRect, StageFile, ViewName


class TeamSlot(BaseModel):
    """One of the six teams on the scoreboard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    team: int
    alliance: Alliance
    station: Literal[1, 2, 3]


class PhaseSpan(BaseModel):
    """One match phase in video seconds. `active_hubs` is None when unknown (auto tie)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: PhaseName
    t_start: float
    t_end: float
    active_hubs: list[Alliance] | None


class MatchInfo(StageFile):
    """Overlay layout, roster and timeline for the match in the video."""

    schema_version: Literal[1] = 1
    scoreboard: PixelRect
    views: dict[ViewName, PixelRect]
    """Panel rectangles of the camera views in full-frame pixels."""
    roster: list[TeamSlot]
    target_team: int
    target_alliance: Alliance
    auto_start_t: float
    auto_end_t: float
    teleop_start_t: float
    teleop_end_t: float
    auto_winner: Alliance | None
    """Alliance that scored more fuel in auto; None on a tie or when unread."""
    phases: list[PhaseSpan]
