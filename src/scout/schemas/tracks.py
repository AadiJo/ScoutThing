"""`tracks.parquet`, written by the track stage: six robot tracks in field metres, one
row per track per sampled frame. `x`, `y` are the robot centre on the carpet."""

from __future__ import annotations

import polars as pl

from scout.schemas.common import ALLIANCES, TableSpec

ALLIANCE = pl.Enum(ALLIANCES)
STATE = pl.Enum(["measured", "interpolated"])

TRACKS = TableSpec(
    "tracks",
    {
        "track_id": pl.Int32(),
        "alliance": ALLIANCE,
        "frame": pl.Int64(),
        "t": pl.Float64(),
        "x": pl.Float64(),
        "y": pl.Float64(),
        "state": STATE,
    },
)

read_tracks = TRACKS.read
write_tracks = TRACKS.write
