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


READER_NAME = "s1_digits"
"""`reader` value for the PARSeq bumper number reader (asset "digits")."""

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
        "text_p": pl.Float32(),
        "roster": pl.List(pl.Int32()),
        "probs": pl.List(pl.Float32()),
        "roster_mass": pl.Float32(),
        "conf": pl.Float32(),
        "best": pl.Int32(),
        "static": pl.Boolean(),
        "abstain": pl.Boolean(),
        "team": pl.Int32(),
        "reader": pl.String(),
    },
)
"""One row per detector box in detections.parquet (same frame, view and box), read
against the match roster.

- `track_id`: null until the labelling half joins reads to tracks.
- `text`, `text_p`: the free (unconstrained) read and its sequence probability; `text` may
  be any digit string, or empty.
- `roster`: the teams scored, in match.json roster order; `probs[k]` is the probability of
  `roster[k]`, a temperature-1 softmax over the roster teams' sequence log-likelihoods.
- `roster_mass`: the summed sequence probability of the roster numbers, near 0 when the
  crop reads as no roster number. `log(probs[k] * roster_mass)` is team k's raw
  log-likelihood, so any other temperature or rule can be applied downstream.
- `conf`, `best`: the top probability and its team, always set.
- `static`: the box sits at a fixed spot in many frames (a display, banner or timer).
- `abstain`, `team`: the stage's own conservative verdict; `team` is `best` when the reader
  names it, null when it abstains.
"""

read_reads = READS.read
write_reads = READS.write
