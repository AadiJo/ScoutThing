"""`detections.parquet`, written by the detect stage: every raw box per sampled frame per
view, unfiltered, in full-frame pixels. One row per box.

The detect stage samples every `round(fps / SAMPLE_HZ)`-th frame from frame 0, so the
sampled frames are `range(0, video.frame_count, round(video.fps / SAMPLE_HZ))` with
`video` from `video.json`. A sampled frame with no boxes has no rows.

- `kind` is what the box frames: `bumper` (a bumper band, from a bumper detector) or `robot`
  (the whole robot, from a whole-robot detector). A robot seen by both detectors has a box
  of each kind; nothing here pairs them.
- `color` is the bumper class for `bumper` boxes and always `unknown` for `robot` boxes.
- `detector` is the asset name of the model that produced the box.
"""

from __future__ import annotations

from typing import Literal, get_args

import polars as pl

from scout.schemas.common import ALLIANCES, VIEW_NAMES, TableSpec

SAMPLE_HZ = 10.0
"""Frames per video second that the detect stage runs on."""

Kind = Literal["bumper", "robot"]

VIEW = pl.Enum(VIEW_NAMES)
COLOR = pl.Enum([*ALLIANCES, "unknown"])
KIND = pl.Enum(get_args(Kind))

DETECTIONS = TableSpec(
    "detections",
    {
        "frame": pl.Int64(),
        "t": pl.Float64(),
        "view": VIEW,
        "x1": pl.Float32(),
        "y1": pl.Float32(),
        "x2": pl.Float32(),
        "y2": pl.Float32(),
        "conf": pl.Float32(),
        "color": COLOR,
        "kind": KIND,
        "detector": pl.String(),
    },
)

read_detections = DETECTIONS.read
write_detections = DETECTIONS.write
