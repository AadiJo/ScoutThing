"""`facts.json`, written by the facts stage: timed, measured events for the target team,
computed by plain code. The grader reads every fact and cites them by `id`.

Conventions:
- times are video seconds, positions WPILib field metres, speeds metres per second;
- `clock` is the scoreboard clock at `t_start` as a person would read it ("2:09"), with
  `period` saying which clock it is (auto counts down from 0:20, teleop from 2:20);
- the target "is tracked" at a moment when a track labelled with the target team has a
  sample there; "untracked" says nothing about what the robot did;
- other robots are only those the tracker followed. Detection misses many robots, so a
  robot that is not in the tracks may still have been there. Fields about other robots
  are named `*_tracked` to keep that in view, and no fact states that a robot was absent.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from scout.games.rebuilt2026 import PhaseName, Zone
from scout.schemas.common import Alliance, StageFile

Period = Literal["pre_match", "auto", "pause", "teleop", "post_match"]
Lane = Literal["bump", "trench"]
Barrier = Literal["blue", "red"]
"""Which hub's line of bumps and trenches: the one between that alliance's zone and the
neutral zone."""
BumperColor = Literal["red", "blue", "unknown"]
Relation = Literal["partner", "opponent", "unknown"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Seen(_Model):
    """The target's tracked position at one moment."""

    t: float
    x: float
    y: float
    zone: Zone


class Coverage(_Model):
    """How much of a span the target was tracked, where it was, and how many other robots
    the tracker followed meanwhile.

    `own_s + neutral_s + opponent_s + untracked_s == duration_s`. `measured_share` is the
    share of the span backed by a detector box (the rest of the tracked time is the
    tracker's interpolation across gaps up to 1 s). `others_tracked_*` count other tracks
    live per sampled frame (any robot but the target; six robots play, so at most five are
    others)."""

    duration_s: float
    tracked_s: float
    tracked_share: float
    measured_share: float
    own_s: float
    neutral_s: float
    opponent_s: float
    untracked_s: float
    gaps: int
    """Gap facts (untracked at least GAP_S) that overlap this span."""
    longest_gap_s: float
    path_length_m: float
    """Distance driven while tracked, not counting jumps across gaps."""
    others_tracked_mean: float
    others_tracked_min: int
    others_tracked_max: int


class _Fact(_Model):
    """Fields every fact has. Instant facts have `t_end == t_start`."""

    id: str
    t_start: float
    t_end: float
    frame_start: int
    frame_end: int
    period: Period
    clock: str
    phase: PhaseName | None
    """Match phase at `t_start`; None in the auto-teleop pause and outside the match."""


# --- phase ----------------------------------------------------------------------


class PhaseData(_Model):
    own_hub_active: bool | None
    """Whether the target alliance's hub scores in this phase; None when unknown."""
    coverage: Coverage


class PhaseFact(_Fact):
    """One match phase: coverage, time per zone and how many other robots were tracked."""

    kind: Literal["phase"] = "phase"
    data: PhaseData


# --- coverage gaps --------------------------------------------------------------

GapCause = Literal["match_start", "match_end", "track_ended", "label_ended", "label_conflict"]


class GapData(_Model):
    duration_s: float
    before: Seen | None
    """Last tracked position before the gap; None when the gap starts at match start."""
    after: Seen | None
    """First tracked position after the gap; None when it runs to match end."""
    straight_line_m: float | None
    """Distance between `before` and `after`."""
    cause: GapCause
    """Why the target's path stops: the match starts or ends, the target's track ended
    (the robot was lost), the track went on but its label stopped being the target, or two
    tracks were both labelled the target at once (that time is dropped, never averaged)."""


class GapFact(_Fact):
    """The target was untracked for at least GAP_S. Nothing is known about it here."""

    kind: Literal["gap"] = "gap"
    data: GapData


# --- zones ----------------------------------------------------------------------


class ZoneVisitData(_Model):
    zone: Zone
    duration_s: float
    untracked_s: float
    """Time inside the visit with no position (short gaps; longer gaps end the visit)."""
    came_from: Zone | Literal["untracked", "match_start"]
    went_to: Zone | Literal["untracked", "match_end"]
    path_length_m: float
    mean_speed_mps: float | None


class ZoneVisitFact(_Fact):
    """One stay in one zone, with jitter across the zone line merged (ZONE_HYST_M)."""

    kind: Literal["zone_visit"] = "zone_visit"
    data: ZoneVisitData


# --- crossings ------------------------------------------------------------------

CrossingOutcome = Literal["crossed", "returned", "unknown"]


class CrossingData(_Model):
    barrier: Barrier
    lane: Lane
    y_m: float
    """Median y while on the barrier; the lane follows from it."""
    from_zone: Zone | None
    """Zone of the last box before the barrier (None: the path starts on it)."""
    to_zone: Zone | None
    """Zone of the first box after the barrier (None: the path ends on it)."""
    outcome: CrossingOutcome
    """`crossed`: the boxes before and after are on opposite sides (certain even across a
    gap). `returned`: both sides were seen without a gap and are the same side. `unknown`:
    one side is missing, or the same side is seen only across a gap (the robot may have
    crossed and come back unseen)."""
    entry_unseen_s: float | None
    """Seconds between the box before and the first box on the barrier; above GAP_S the
    approach was not seen."""
    exit_unseen_s: float | None
    """Seconds between the last box on the barrier and the box after."""
    time_on_barrier_s: float | None
    """From entering to leaving the band, with both edges seen; None when either side is
    beyond a gap."""
    observed_s: float
    """Time the facts below cover: the whole pass when both edges were seen, otherwise
    only the time between the first and last box on the barrier."""
    stopped_s: float
    """Time on the barrier with speed below STOP_MPS."""
    min_speed_mps: float | None
    mean_speed_mps: float | None
    measured_s: float
    """Part of `observed_s` backed by detector boxes; the rest is interpolated."""


class CrossingFact(_Fact):
    """The target was seen on a bump or trench band (centre within BAND_MARGIN_M of the
    bumps' depth): what it crossed, how long it took, how long it was stopped there."""

    kind: Literal["crossing"] = "crossing"
    data: CrossingData


class CrossingInGapData(_Model):
    barrier: Barrier
    from_zone: Zone
    to_zone: Zone
    before: Seen
    after: Seen
    unseen_s: float
    """Time between the last box on one side and the first box on the other."""


class CrossingInGapFact(_Fact):
    """The target was last seen on one side of a barrier and next seen on the other, with
    no box on the barrier between. It crossed somewhere in that time; how is unknown."""

    kind: Literal["crossing_in_gap"] = "crossing_in_gap"
    data: CrossingInGapData


# --- stops ----------------------------------------------------------------------


class StopData(_Model):
    duration_s: float
    x: float
    y: float
    zone: Zone
    on_barrier: Barrier | None
    ends: Literal["moved", "untracked", "match_end"]
    nearest_tracked_m: float | None
    """Closest other tracked robot during the stop; None when none was tracked. A robot
    the tracker missed may have been closer."""
    others_tracked_max: int


class StopFact(_Fact):
    """The target's speed stayed below STOP_MPS for at least STOP_MIN_S."""

    kind: Literal["stop"] = "stop"
    data: StopData


# --- contacts -------------------------------------------------------------------


class OtherRobot(_Model):
    """Another track that came within CONTACT_M of the target."""

    track_id: int
    team: int | None
    """Team label of that track at that time; None when the track is unlabelled."""
    relation: Relation
    bumper: BumperColor
    """Bumper colour seen. Not the alliance: at offseason events a team can play for one
    alliance in the other's bumpers."""
    min_distance_m: float
    close_s: float
    """Time this robot spent within CONTACT_M of the target."""


class SpeedWindow(_Model):
    """Target speed and progress over one window."""

    duration_s: float
    tracked_s: float
    mean_speed_mps: float | None
    net_progress_m: float | None
    """Straight-line distance from the window's first to last tracked position."""


ContactEnd = Literal[
    "target_moved_away", "other_moved_away", "other_untracked", "target_untracked", "match_end"
]


class ContactData(_Model):
    others: list[OtherRobot]
    duration_s: float
    min_distance_m: float
    before: SpeedWindow
    during: SpeedWindow
    after: SpeedWindow
    longest_still_s: float
    """Longest run during the contact with target speed below STOP_MPS."""
    ended: ContactEnd
    others_tracked_min: int
    others_tracked_max: int
    """Other robots tracked anywhere on the field during the contact (of five)."""


class ContactFact(_Fact):
    """Another tracked robot's centre was within CONTACT_M of the target's: touching or
    close enough to block. Contacts with robots the tracker missed do not appear at all."""

    kind: Literal["contact"] = "contact"
    data: ContactData


# --- hub turning on -------------------------------------------------------------


class HubOnData(_Model):
    phase: PhaseName
    at: Seen | None
    """Target position when the hub turned on; None when untracked at that moment."""
    last_seen: Seen | None
    """When untracked at that moment: the last position before it."""
    next_seen: Seen | None
    """When untracked at that moment: the first position after it."""
    into_own_zone_m: float | None
    """Signed distance of the robot centre past its own zone line (negative: outside)."""
    speed_mps: float | None
    in_own_zone: bool | None
    own_zone_since_s: float | None
    """When in its own zone: how long it had been there (the visit's start, which may be
    bounded by a gap; see `since_after_gap`)."""
    since_after_gap: bool
    reached_own_zone_after_s: float | None
    """When not in its own zone: seconds until it was first tracked there; None if never
    in this phase."""
    reached_after_gap: bool
    """True when that arrival was first seen after a gap, so it arrived at most that late."""


class HubOnFact(_Fact):
    """The target alliance's hub turned on: where the target was."""

    kind: Literal["hub_on"] = "hub_on"
    data: HubOnData


# --- auto -----------------------------------------------------------------------


class Waypoint(_Model):
    t: float
    x: float
    y: float
    zone: Zone
    after_gap: bool
    """True when the target was untracked for at least GAP_S just before this point."""


class AutoRouteData(_Model):
    start: Seen | None
    """First tracked position in auto."""
    start_untracked_s: float
    """Seconds from auto start to `start`."""
    waypoints: list[Waypoint]
    """The tracked route simplified to turns (ROUTE_TOLERANCE_M)."""
    end: Seen | None
    """Last tracked position in auto."""
    end_untracked_s: float
    """Seconds from `end` to auto end."""
    crossings: list[str]
    """Ids of crossing and crossing_in_gap facts in auto."""
    max_past_centre_line_m: float | None
    """Furthest the robot centre was seen past the centre line toward the opponent
    (negative: it stayed that far short). Only tracked time counts."""
    coverage: Coverage


class AutoRouteFact(_Fact):
    """Where the target started, went and ended in auto. Reported, not scored."""

    kind: Literal["auto_route"] = "auto_route"
    data: AutoRouteData


Fact = Annotated[
    PhaseFact
    | GapFact
    | ZoneVisitFact
    | CrossingFact
    | CrossingInGapFact
    | StopFact
    | ContactFact
    | HubOnFact
    | AutoRouteFact,
    Field(discriminator="kind"),
]
FactKind = Literal[
    "phase",
    "gap",
    "zone_visit",
    "crossing",
    "crossing_in_gap",
    "stop",
    "contact",
    "hub_on",
    "auto_route",
]


class Facts(StageFile):
    """All facts for one team, sorted by `t_start`, plus match-wide coverage."""

    schema_version: Literal[2] = 2
    team: int
    alliance: Alliance
    thresholds: dict[str, float]
    """The named constants behind every fact, by name (units in the name)."""
    notes: list[str]
    """What the facts can and cannot show; the grader should read these first."""
    coverage: Coverage
    """The whole match, auto start to teleop end."""
    team_tracked_share: dict[int, float]
    """Each roster team: share of the match's sampled frames in which a track labelled as
    that team exists. A low share means that robot is mostly invisible to these facts."""
    facts: list[Fact]
