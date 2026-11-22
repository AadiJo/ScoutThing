"""`bench.json`, written by `scout bench` into a run folder: the run's tracks and labels
scored against a truth file (`scout.schemas.truth`). `bench.md` beside it is the same
numbers for people.

Time is counted on the tracks' 10 Hz grid: each grid frame where a robot has truth (the
nearest truth sample within half a grid step) weighs one grid step of seconds. Shares are
None when their denominator is zero. Every per-robot score is reported twice: for grid
frames where the truth is `visible` (the headline) and where it is `estimated`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from scout.schemas.common import StageFile
from scout.schemas.truth import TruthSource


class WrongStretch(BaseModel):
    """Consecutive grid frames where track `track_id` was labelled with the robot's team
    but lay outside the gate of the robot's truth position."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    track_id: int
    frame_start: int
    frame_end: int
    t_start: float
    t_end: float
    seconds: float
    median_distance_m: float
    on_team: int | None
    """The truth robot the track was actually within the gate of, if any had truth there."""


class GateScore(BaseModel):
    """Scores at one distance gate. Shares of truth time unless noted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gate_m: float
    label_recall: float | None
    """Some track within the gate is labelled with this team."""
    tracked_share: float | None
    """Some track, labelled or not, is within the gate."""
    no_track_share: float | None
    """Not recalled: no track within the gate."""
    unlabelled_track_share: float | None
    """Not recalled: a track within the gate, and some track there is unlabelled."""
    other_label_share: float | None
    """Not recalled: tracks within the gate, all labelled with other teams."""
    labelled_s: float
    """Track time labelled with this team while this robot has truth of this state."""
    wrong_label_s: float
    wrong_label_share: float | None
    """`wrong_label_s / labelled_s`: labelled time on another robot or on nothing."""
    wrong_stretches: list[WrongStretch]


class PositionScore(BaseModel):
    """Distance between truth and the matched track sample (one-to-one assignment per grid
    frame, within `gate_m`)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gate_m: float
    matched_s: float
    median_m: float | None
    p90_m: float | None
    mean_los_offset_m: float | None
    """Mean of (track - truth) along the ground direction from the main camera to the truth
    point. Positive: the track sits farther from the camera than the robot."""


class StateScore(BaseModel):
    """Scores over the grid frames where this robot's truth has one state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    truth_s: float
    gates: list[GateScore]
    position: PositionScore
    identity_switches: int
    """Times the matched track changed to a track matched earlier to a different robot."""
    track_changes: int
    """Times the matched track changed at all (fragmentation, including switches)."""


class RobotBench(BaseModel):
    """One robot in the truth."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    team: int
    is_target: bool
    truth_samples: int
    skipped_samples: int
    """Truth samples not in the main view or whose pixel does not reach the field plane."""
    visible: StateScore
    estimated: StateScore
    labelled_without_truth_s: float
    """Track time labelled with this team at grid frames where this robot has no truth."""


class Targets(BaseModel):
    """The agreed targets for the target team, judged at `gate_m`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gate_m: float
    max_wrong_label_share: float
    min_label_recall: float
    max_median_error_m: float
    team: int
    in_truth: bool
    wrong_label_ok: bool | None
    recall_ok: bool | None
    median_ok: bool | None


class Bench(StageFile):
    """Bench result for one run against one truth file."""

    schema_version: Literal[1] = 1
    video_sha256: str
    truth_path: str
    truth_source: TruthSource
    click_height_m: float
    grid_hz: float
    created_at: datetime
    targets: Targets
    robots: list[RobotBench]
    """Target team first, then the truth file's order."""
