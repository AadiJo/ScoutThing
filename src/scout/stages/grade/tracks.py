"""Tracked samples: the target's and every other robot's, from tracks.parquet and
identity.json (`load_tracks`), and how long the target needs to get home from where it was
(`home_allow`)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from numpy.typing import NDArray

from scout.games.rebuilt2026 import ALLIANCE_ZONE_DEPTH_M, own_wall_m
from scout.run import RunDir
from scout.schemas.common import Alliance
from scout.schemas.identity import Identity
from scout.schemas.match import MatchInfo
from scout.schemas.tracks import read_tracks

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
TRIP_ALLOW_S = 3.0
HOME_CROSS_S = 1.2
HOME_MPS = 2.0
"""Seconds a heaped robot out of its own zone is allowed to get home before the rest of its
trip is charged: a normal crossing plus its distance outside its zone at HOME_MPS (robots
drive home at 2 to 2.5 m/s), and never less than TRIP_ALLOW_S."""
TRACKED_WITHIN_S = 0.15
"""A sample this close in time counts as a robot's position at a tile."""


@dataclass(frozen=True)
class Target:
    """The target team's samples (what facts.json calls its path)."""

    t: F64
    xy: F64
    box: F64
    """(n, 4) main-view box in full-frame pixels; NaN when interpolated or from a corner view."""

    def near(self, a: float, b: float) -> F64:
        """Positions (k, 2) tracked between video times a and b."""
        return self.xy[(self.t >= a) & (self.t <= b)]


@dataclass(frozen=True)
class Others:
    """Every other track's samples with its team label (-1 when unlabelled)."""

    t: F64
    track: I64
    team: I64
    xy: F64
    box: F64
    """(n, 4) main-view box; NaN when interpolated or from a corner view."""


def load_tracks(run: RunDir, match: MatchInfo) -> tuple[Target, Others]:
    """The target: samples inside identity segments labelled with it, minus frames two
    tracks both claim. Others: every other sample, with its label."""
    ident = Identity.load(run.identity_json)
    tracks = read_tracks(run.tracks)
    target = match.target_team
    named = [s for s in ident.segments if s.team is not None]
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
        .unique(["track_id", "frame"], keep="first")
        .select("track_id", "frame", "team")
    )
    rows = tracks.join(labels, on=["track_id", "frame"], how="left")

    def boxes(df: pl.DataFrame) -> F64:
        main = ((df["state"] == "measured") & (df["view"] == "main")).fill_null(False)
        box = df.select("x1", "y1", "x2", "y2").to_numpy().astype(np.float64)
        box[~main.to_numpy()] = np.nan
        return box

    mine = rows.filter(pl.col("team") == target)
    mine = mine.unique(["track_id", "frame"]).filter(pl.col("frame").is_unique()).sort("frame")
    rest = rows.filter(pl.col("team").is_null() | (pl.col("team") != target)).sort("frame")
    return (
        Target(
            t=mine["t"].to_numpy().astype(np.float64),
            xy=mine.select("x", "y").to_numpy().astype(np.float64),
            box=boxes(mine),
        ),
        Others(
            t=rest["t"].to_numpy().astype(np.float64),
            track=rest["track_id"].to_numpy().astype(np.int64),
            team=rest["team"].fill_null(-1).to_numpy().astype(np.int64),
            xy=rest.select("x", "y").to_numpy().astype(np.float64),
            box=boxes(rest),
        ),
    )


def home_allow(target: Target, alliance: Alliance, t: float) -> float:
    """Seconds a heaped robot at video time `t` needs to get home (see TRIP_ALLOW_S)."""
    near = target.near(t - 1.0, t + 1.0)
    if not len(near):
        return TRIP_ALLOW_S
    out_m = max(0.0, own_wall_m(float(np.median(near[:, 0])), alliance) - ALLIANCE_ZONE_DEPTH_M)
    return max(TRIP_ALLOW_S, HOME_CROSS_S + out_m / HOME_MPS)
