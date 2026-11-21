"""`detections.parquet`, written by the detect stage: every raw bumper box per sampled
frame per view, unfiltered, in full-frame pixels. One row per box."""

from __future__ import annotations

import polars as pl

from scout.schemas.common import ALLIANCES, VIEW_NAMES, TableSpec

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
