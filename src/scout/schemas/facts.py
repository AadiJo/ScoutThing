"""`facts.json`, written by the facts stage: timed events for the target team, computed
by plain code. The grader cites these events by `id`."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from scout.schemas.common import Alliance, StageFile

FactKind = Literal["zone_change", "crossing", "stall", "near_opponent", "phase"]


class Fact(BaseModel):
    """One event. Instant events have `t_end == t_start`. `data` holds kind-specific values
    (for example `{"from": "neutral", "to": "own"}` for a zone change)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    kind: FactKind
    frame_start: int
    frame_end: int
    t_start: float
    t_end: float
    data: dict[str, str | float | int | bool | None]


class Facts(StageFile):
    """All events for one team, sorted by `t_start`."""

    schema_version: Literal[1] = 1
    team: int
    alliance: Alliance
    facts: list[Fact]
