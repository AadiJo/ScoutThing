"""Where each robot was: the one place readers get robot positions and their team labels.

    pos = load_positions(run)                  # main view + corner cameras
    main = load_positions(run, corners=False)  # main view only

`pos.tracks` has the `tracks.parquet` columns and `pos.segments` the `identity.json` label
segments. With corners, the follow stage's corner tracks and labels are appended: robots
placed through a corner camera only in frames where the main view had no sample of them, so
a main-view sample is never replaced or moved, and `corner_carried` labels on main-view
stretches identify left unlabelled where a labelled corner track rode on them (main-view
data only: `corners=False`). Corner tracks' `view` is the corner view, so a reader can still
ask for main-view positions only (`view == "main"`), and their track ids start at
`CORNER_TRACK_ID_BASE`. A corner position is coarser than a main-view one (about 0.2 to
0.35 m, smoothed) and has no main-view box.

Every stage after the follow stage, the bench and the annotator read positions here; the
track, identify and corners stages run before it and read `tracks.parquet` themselves.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from scout.run import RunDir
from scout.schemas.identity import CornerIdentity, Identity, LabelSegment
from scout.schemas.tracks import read_tracks


@dataclass(frozen=True)
class Positions:
    """Track samples (`tracks.parquet` columns) and the label segments that name them."""

    tracks: pl.DataFrame
    segments: list[LabelSegment]


def load_positions(run: RunDir, corners: bool = True) -> Positions:
    """The main view's tracks and labels, plus the corner tracks when `corners` and the
    follow stage has written them (a run folder without them gives the main view only)."""
    tracks = read_tracks(run.tracks)
    segments = list(Identity.load(run.identity_json).segments)
    if corners and run.corner_tracks.exists() and run.corner_identity_json.exists():
        tracks = pl.concat([tracks, read_tracks(run.corner_tracks)])
        segments += CornerIdentity.load(run.corner_identity_json).segments
    return Positions(tracks, segments)


def with_teams(tracks: pl.DataFrame, segments: list[LabelSegment]) -> pl.DataFrame:
    """`tracks` with a `team` column (Int64): the team of the labelled segment covering each
    sample, null where none does or where segments of two teams do."""
    named = [s for s in segments if s.team is not None]
    seg = pl.DataFrame(
        {
            "track_id": [s.track_id for s in named],
            "f0": [s.frame_start for s in named],
            "f1": [s.frame_end for s in named],
            "team": [s.team for s in named],
        },
        schema={"track_id": pl.Int32, "f0": pl.Int64, "f1": pl.Int64, "team": pl.Int64},
    )
    labels = (
        tracks.select("track_id", "frame")
        .join(seg, on="track_id")
        .filter(pl.col("frame").is_between(pl.col("f0"), pl.col("f1")))
        .group_by("track_id", "frame")
        .agg(pl.col("team").unique())
        .select(
            "track_id",
            "frame",
            pl.when(pl.col("team").list.len() == 1)
            .then(pl.col("team").list.first())
            .otherwise(None)
            .alias("team"),
        )
    )
    return tracks.join(labels, on=["track_id", "frame"], how="left")
