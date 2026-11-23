"""`identity.json` and `reads.parquet`, written by the identify stage.

`identity.json` labels time segments of each track with a team, or null when unsure
(a gap beats a wrong label). `reads.parquet` holds every raw number read behind them.
"""

from __future__ import annotations

from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

from scout.schemas.common import VIEW_NAMES, StageFile, TableSpec
from scout.schemas.detections import COLOR

LabelSource = Literal[
    "digits",
    "digits_extended",
    "votes",
    "appearance",
    "appearance_continuity",
    "no_reads",
    "below_rule",
    "conflict",
]
"""What decided a segment.

- `digits`: between the first and last of at least three agreeing named bumper reads on the
  track, with no named read of another team in between.
- `digits_extended`: the same label carried past those reads toward the track's ends, until
  the unnamed reads turn against the team or a time limit.
- `votes`: a stretch the rules above left unlabelled, between the first and last of many
  confident reads of one team on bumper boxes of that team's alliance colour, whose free read
  need not spell the number (a bumper whose digits the free read garbles, such as 2468's blue
  set read as "9409" or "268"), where that team holds nearly all confident reads of the
  stretch. Off for the whole match for a team once one of its vote labels clashes with
  another track's label of it (`TeamEvidence.votes_off`).
- `appearance`: a stretch with no number-read label, named because its robot looks like that
  team's number-read stretches in this match (appearance embedder) by a clear margin over
  every team not labelled elsewhere at the time. See `AppearanceEvidence`.
- `appearance_continuity`: the same with a looser appearance bar, allowed because the stretch
  also continues a track labelled that team (same track, or the only track that ended or
  began where and when this one starts or ends).
- `no_reads` (team null): no bumper read joins this stretch of track.
- `below_rule` (team null): reads join it, but not enough agreeing named reads.
- `conflict` (team null): the track holds named reads of two teams around this stretch, or
  another track claimed the same team at the same time.
"""


class AppearanceEvidence(BaseModel):
    """Why an `appearance*` segment carries its team: its strength, to measure it apart from
    number reads."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cosine: float
    """Mean over the stretch's boxes of the mean of the 10 highest cosines to the team's
    number-read boxes in this match."""
    margin: float
    """`cosine` minus the runner-up candidate's (9.0 when there was no runner-up)."""
    runner_up: int | None
    """The next best candidate team, None when there was none."""
    n_boxes: int = Field(ge=0)
    """Embedded boxes on the stretch."""
    candidates: list[int]
    """Teams not labelled on another track during the stretch."""


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
    """Median probability of `team` over the reads joined to this segment (0 when it has none,
    or when `team` is null). Low on a `digits_extended` segment means weak support. On an
    `appearance*` segment, `appearance.cosine` clipped to [0, 1]."""
    source: LabelSource
    n_reads: int = Field(ge=0)
    """Agreeing named reads behind the label, or votes on a `votes` segment (0 when `team` is
    null or not from reads)."""
    appearance: AppearanceEvidence | None = None
    """Set on `appearance*` segments only."""


class ClaimConflict(BaseModel):
    """Two tracks labelled the same team at the same time, before resolution. A team is in one
    place at a time, so each of these means some label was wrong."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    team: int
    track_ids: tuple[int, int]
    t_start: float
    t_end: float
    kept: int | None
    """The track that kept the label over the overlap (read-backed against extended), or None
    when both lost it."""


class TeamEvidence(BaseModel):
    """How much number evidence one roster team had, and why it got no label if it got none."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    team: int
    named_reads: int = Field(ge=0)
    """Named (non-abstaining) reads in any view, joined to a track or not."""
    named_on_track: int = Field(ge=0)
    """Named main-view reads joined to a track."""
    vote_reads: int = Field(ge=0)
    """Main-view reads joined to a track that count as votes for the team (see `votes`)."""
    corner_votes: int = Field(ge=0)
    """Corner-view reads that count as votes; corners are not tracked, so they label nothing,
    but many of them with no label means the robot was readable only there."""
    votes_off: bool
    """True when a `votes` label of this team clashed with another track's label of it (so
    another robot also reads as it), and none of its reads counted as votes in this match."""
    labelled_s: float = Field(ge=0.0)
    """Track time labelled with the team, from any source."""
    reason: str | None
    """Why the team has no label, when `labelled_s` is 0; None otherwise."""


class Identity(StageFile):
    """Label segments covering every track sample, the same-team conflicts resolved, and the
    number evidence per roster team."""

    schema_version: Literal[1] = 1
    segments: list[LabelSegment]
    conflicts: list[ClaimConflict]
    teams: list[TeamEvidence] = Field(default_factory=list[TeamEvidence])
    """One entry per roster team, in roster order (empty in files written before it existed)."""


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
        "color": COLOR,
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

- `track_id`: the track whose measured box covers at least 80% of this main-view box in
  the same frame (the best one if several); null for corner views and unmatched boxes.
- `color`: the detector's bumper colour class for the box (`red`, `blue` or `unknown`).
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
