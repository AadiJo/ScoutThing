"""`detections.parquet`, written by the detect stage: every raw bumper box per sampled
frame per view, unfiltered, in full-frame pixels. One row per box.

The detect stage samples every `round(fps / SAMPLE_HZ)`-th frame from frame 0, so the
sampled frames are `range(0, video.frame_count, round(video.fps / SAMPLE_HZ))` with
`video` from `video.json`. A sampled frame with no boxes has no rows.
"""

from __future__ import annotations

import polars as pl

from scout.schemas.common import ALLIANCES, VIEW_NAMES, TableSpec

SAMPLE_HZ = 10.0
"""Frames per video second that the detect stage runs on."""

VIEW = pl.Enum(VIEW_NAMES)
COLOR = pl.Enum([*ALLIANCES, "unknown"])

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
        "detector": pl.String(),
    },
)

read_detections = DETECTIONS.read
write_detections = DETECTIONS.write
