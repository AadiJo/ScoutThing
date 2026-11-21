"""`identity.json` and `reads.parquet`, written by the identify stage.

`identity.json` labels time segments of each track with a team, or null when unsure
(a gap beats a wrong label). `reads.parquet` holds every raw number read behind them.
"""

from __future__ import annotations

from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from scout.schemas.common import VIEW_NAMES, StageFile, TableSpec


class LabelSegment(BaseModel):
    """Track `track_id` is `team` (or unknown) from `t_start` to `t_end` inclusive."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    track_id: int
    frame_start: int
    frame_end: int
    t_start: float
    t_end: float
    team: int | None
    confidence: float = Field(ge=0.0, le=1.0)
    source: str
    """What decided the label, for example "digits", "embedder", "elimination", "answer"."""


class Identity(StageFile):
    """Label segments for every track."""

    schema_version: Literal[1] = 1
    segments: list[LabelSegment]


READS = TableSpec(
    "reads",
    {
        "frame": pl.Int64(),
        "t": pl.Float64(),
        "view": pl.Enum(VIEW_NAMES),
        "track_id": pl.Int32(),
        "x1": pl.Float32(),
        "y1": pl.Float32(),
        "x2": pl.Float32(),
        "y2": pl.Float32(),
        "text": pl.String(),
        "team": pl.Int32(),
        "conf": pl.Float32(),
        "reader": pl.String(),
    },
)
"""One row per number read. `track_id` and `team` are null when unmatched."""

read_reads = READS.read
write_reads = READS.write
