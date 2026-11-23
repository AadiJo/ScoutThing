"""`facts.json`, written by the facts stage: timed, measured events for the target team,
computed by plain code. The grader reads every fact and cites them by `id`.

Conventions:
- times and durations are video seconds, positions WPILib field metres, speeds metres per
  real second (video speeds times `match.clock_rate`);
- `clock` is the scoreboard clock at `t_start` as a person would read it ("2:09"), with
  `period` saying which clock it is (auto counts down from 0:20, teleop from 2:20);
- the target "is tracked" at a moment when a track labelled with the target team has a
  sample there; "untracked" or "unseen" says nothing about what the robot did;
- anything concluded about unseen time is named `inferred*`, `*_unseen_s` or `inference`
  and is never mixed into a measured field without a field saying how much;
- other robots are only those the tracker followed. Detection misses many robots, so a
  robot that is not in the tracks may still have been there. Fields about other robots
  are named `*_tracked` to keep that in view, and no fact states that a robot was absent.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from scout.games.rebuilt2026 import Lane, PhaseName, Zone
from scout.schemas.common import Alliance, StageFile

Period = Literal["pre_match", "auto", "pause", "teleop", "post_match"]
Barrier = Literal["blue", "red"]
"""Which hub's line of bumps and trenches: the one between that alliance's zone and the
neutral zone."""
Place = Zone | Literal["on_barrier"]
"""A zone, or `on_barrier`: on the barrier band (see `Facts.notes`) for a stay of at least
PERCH_MIN_S. A centre inside the own zone is never on a barrier."""
BumperColor = Literal["red", "blue", "unknown"]
Relation = Literal["partner", "opponent", "unknown"]
Reach = Literal["centre", "footprint", "corner"]
"""How close the robot centre came to a barrier's bumps or trench (depth, along x): `centre`
within BAND_MARGIN_M (the centre was on it, to position accuracy); `footprint` within half a
robot (PERCH_M), so part of the robot was over it whatever its heading; `corner` within half
the robot's diagonal (CORNER_M), so only a corner could have been, and only if the robot was
turned; such a pass counts only when the robot was driving toward the bumps (HEADING_MPS).
Tracks carry no heading, and positions are about 0.2 m accurate."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Seen(_Model):
    """The target's tracked position at one moment."""

    t: float
    x: float
    y: float
    zone: Zone


class Where(_Model):
    """A place in plain measures, from the robot centre: say where it was from these, not
    from the zone name alone."""

    x: float
    y: float
    zone: Zone
    """Zone of the centre itself (no hysteresis)."""
    into_own_zone_m: float
    """Signed distance of the centre past its own zone line (negative: outside it)."""
    own_hub_m: float
    """Centre to the nearest point of its own hub's base."""
    own_wall_m: float
    """Centre to its own alliance (driver-station) wall."""
    guardrail_m: float
    """Centre to the nearer side guardrail."""


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
GapInference = Literal[
    "stayed_in_place", "in_own_zone_beyond_barrier", "entered_own_zone", "own_zone_both_sides"
]


class NearInGap(_Model):
    """Another tracked robot that came within CONTACT_M of where the target was last or next
    seen, while the target was unseen."""

    track_id: int
    team: int | None
    relation: Relation
    min_distance_m: float
    """Closest distance to the target's last-seen or next-seen position, not to the target."""
    t: float
    """When that closest distance was measured."""


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
    inference: GapInference | None
    """What the positions either side suggest about the unseen time; an inference, never a
    measurement. `stayed_in_place`: at most BRIDGE_MAX_S long and `before` and `after` at most
    BRIDGE_M apart, so it very probably stayed there. `in_own_zone_beyond_barrier`: last seen
    on or at its own barrier driving toward its own wall and next seen at the same barrier
    driving away from it, so it very probably spent the gap in its own zone.
    `entered_own_zone`: at least OWN_INFER_MIN_S long, last seen on or at its own barrier
    driving toward its own wall and next seen in its own zone, so it very probably crossed
    home just after it was last seen. `own_zone_both_sides`: at most OWN_BOTH_MAX_S long, last
    and next seen in its own zone, so it was probably home throughout (it may have left and
    come back unseen). None: unknown."""
    others_near: list[NearInGap]
    """Other tracked robots near the target's last-seen or next-seen position during the gap
    (unlabelled tracks closer than SAME_ROBOT_M are left out: they are probably the target
    itself, unlabelled). A pointer to frames worth opening, not a contact."""


class GapFact(_Fact):
    """The target was untracked for at least GAP_S. Nothing is measured here."""

    kind: Literal["gap"] = "gap"
    data: GapData


# --- zones ----------------------------------------------------------------------


class ZoneVisitData(_Model):
    zone: Place
    barrier: Barrier | None
    """Set when `zone` is `on_barrier`: which barrier."""
    lane: Lane | None
    """Set when `zone` is `on_barrier`: bump or trench."""
    duration_s: float
    untracked_s: float
    """Time inside the visit with no position (short gaps; longer gaps end the visit)."""
    came_from: Place | Literal["untracked", "match_start"]
    went_to: Place | Literal["untracked", "match_end"]
    path_length_m: float
    mean_speed_mps: float | None
    excess_s: float | None
    """Set when `zone` is `on_barrier`: `duration_s` beyond NORMAL_CROSSING_S, the slowest
    normal pass across the band (0 when within it)."""
    own_active_s: float
    """Part of the visit in teleop time with the target's own hub active."""


class ZoneVisitFact(_Fact):
    """One stay in one zone, or on a barrier, with jitter across the zone line merged
    (ZONE_HYST_M). Barrier stays shorter than PERCH_MIN_S count toward the zone the centre
    was in (a crossing covers them)."""

    kind: Literal["zone_visit"] = "zone_visit"
    data: ZoneVisitData


# --- crossings ------------------------------------------------------------------

CrossingOutcome = Literal["crossed", "unknown"]


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
    gap). `unknown`: one side is missing, or the same side is seen only across a gap (the
    robot may have crossed and come back unseen). Passes that came back out the same side
    are `crossing_turned_back` facts."""
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
    excess_s: float
    """`time_on_barrier_s` (or `observed_s` when an edge was unseen, so a lower bound) beyond
    NORMAL_CROSSING_S, the slowest normal pass across the band; 0 when within it."""
    own_active_s: float
    """Part of the pass in teleop time with the target's own hub active."""


class CrossingFact(_Fact):
    """The target was seen on a barrier band (see `Facts.notes`): what it crossed, how long
    it took, how long it was stopped there."""

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


ArrivalBy = Literal["seen", "crossing", "inference", "after_gap"]


class Arrival(_Model):
    """The earliest evidence that the target got somewhere (home, or the far side of a
    barrier). `seen`: tracked arriving. When it was first tracked there after a gap: `crossing`
    (a `crossing` fact last saw it on the band crossing that way) or `inference` (the gap's
    `inference` puts it there): `t` is then an estimate, the rest of the way at the speed it
    was going that way when last seen; else `after_gap`: `t` is the first sample after
    the gap, so it got there at most then (an upper bound)."""

    t: float
    by: ArrivalBy
    fact: str | None
    """The crossing or gap fact behind a `crossing` or `inference` estimate."""
    first_tracked_t: float
    """First tracked time there (equal to `t` when `seen`)."""


CostEnd = Literal[
    "reached_far_side", "tried_again", "hub_went_inactive", "track_lost", "match_end", "came_back"
]


class CrossingTurnedBackData(_Model):
    barrier: Barrier
    lane: Lane
    y_m: float
    """Median y on the barrier during the attempt."""
    side: Zone
    """The zone it came from and went back to."""
    far_side: Zone
    """The zone across the barrier from `side`: where it was heading."""
    toward_home: bool
    """True when `far_side` is nearer its own wall (into its own zone, or out of the
    opponent's zone): then the cost runs until it got there."""
    reach: Reach
    """How close the centre came to the barrier (see `Reach`)."""
    nearest_edge_m: float
    """Closest the centre came to the bumps' or trench's depth along x (0: over it)."""
    reached_far_edge: bool
    """True when the centre got past the far edge of the band but never more than half a
    robot clear of the bumps' depth (so part of it stayed over them), then came back."""
    time_on_barrier_s: float
    """From entering the band to leaving it back to `side`."""
    stopped_s: float
    """Part of that with speed below STOP_MPS, including time inferred across bridged gaps
    (see `unseen_s`)."""
    unseen_s: float
    """Untracked time on the band that was bridged (robot at nearly the same place either
    side)."""
    min_speed_mps: float | None
    slowest_mps: float | None
    """Lowest speed on the band, from entering to leaving it, read on the filled path (a
    bridged gap counts as the straight line across it, so a stall hidden in one reads slow);
    None when no speed could be read. A drive past reads at driving pace; a stall or a
    reversal reads slow."""
    arrived: Arrival | None
    """When it got to `far_side` (off the barrier) after turning back; None if never seen
    there before the match ended."""
    seconds_to_far_side: float | None
    """From entering the band to `arrived.t` (an upper bound when `arrived.by` is
    `after_gap`)."""
    across_id: str | None
    """The `crossing` or `crossing_in_gap` fact of this barrier by which it got to
    `far_side` within the cost window; None if none."""
    cost_s: float
    """The fact's span: from entering the band to `cost_end`."""
    cost_end: CostEnd
    """What ended the span. Toward home: the earliest of `reached_far_side` (`arrived.t`),
    `tried_again`
    (its next attempt on this barrier from `side` began), `hub_went_inactive` (the end of the
    first own-active stretch ending after it began), `track_lost` (the last sample before an
    untracked stretch longer than BRIDGE_MAX_S: how long it really took is unknown, at least
    this) and `match_end`. Away from home: `came_back`
    (the span is the attempt itself: it was back on the side nearer home)."""
    own_active_s: float
    """Part of `cost_s` in teleop time with the target's own hub active."""
    furthest_from_barrier_m: float | None
    """After turning back and within the span: the furthest the centre got from the bumps'
    depth along x, on `side` (a reroute or going elsewhere)."""
    contact_ids: list[str]
    """Contact facts overlapping the span."""
    nearest_tracked_m: float | None
    """Closest other tracked robot while on the barrier; a robot the tracker missed may
    have been closer."""


class CrossingTurnedBackFact(_Fact):
    """The target went onto (or with `reach` `corner`, possibly against) a barrier and came
    back out the side it came from. One attempt per fact: attempts are never chained. Spans
    from entering the band to `cost_end`. The shape only: whether it was blocked, stuck,
    staging or changing its mind is not known from positions."""

    kind: Literal["crossing_turned_back"] = "crossing_turned_back"
    data: CrossingTurnedBackData


# --- stops ----------------------------------------------------------------------

StopStart = Literal[
    "seen_slowing", "began_while_unseen", "first_seen_already_stopped", "match_start"
]
StopEnd = Literal["seen_moving_off", "ended_while_unseen", "last_seen_still_stopped", "match_end"]


class StopData(_Model):
    duration_s: float
    seen_s: float
    """Part of the stop with a tracked position."""
    unseen_s: float
    """Part of the stop inferred across bridged gaps: the robot was untracked but at most
    BRIDGE_M from where it was before. 0 when the whole stop was seen."""
    gaps_bridged: int
    moved_across_gaps_m: float | None
    """Largest straight-line move across a bridged gap; None when no gap was bridged."""
    where: Where
    """Median tracked position of the stop."""
    own_active_s: float
    """Part of the stop in teleop time with the target's own hub active."""
    barrier: Barrier
    """The nearer barrier (along x)."""
    barrier_edge_m: float
    """Centre to that barrier's bumps' depth along x (0: over it)."""
    lane: Lane | None
    """Lane of that barrier at the stop's y; None beside a hub face."""
    on_barrier: Barrier | None
    """Set when the centre was on that barrier's band (see `Facts.notes`); never when the
    centre was in its own zone, whatever part of the robot reached the barrier."""
    start: StopStart
    """`seen_slowing`: seen arriving. `began_while_unseen`: the stop starts inside a bridged
    gap, so when it began is not known within that gap. `first_seen_already_stopped`: first
    seen after an unbridged gap already stopped; it may have stopped earlier."""
    end: StopEnd
    """`seen_moving_off`: seen leaving. `ended_while_unseen`: ends inside a bridged gap.
    `last_seen_still_stopped`: tracking lost it while stopped; it may have stayed."""
    nearest_tracked_m: float | None
    """Closest other tracked robot during the seen part of the stop; None when none was
    tracked. A robot the tracker missed may have been closer."""
    others_tracked_max: int


class StopFact(_Fact):
    """The target's speed stayed below STOP_MPS for at least STOP_MIN_S, counting bridged
    gaps as part of the stop (see `unseen_s`)."""

    kind: Literal["stop"] = "stop"
    data: StopData


# --- trips out ------------------------------------------------------------------


class TripData(_Model):
    left: Seen
    """Last tracked position in its own zone before leaving."""
    left_unseen_s: float
    """Untracked time between `left` and the first position outside. 0: seen leaving, and the
    trip starts at `left`. Above 0: when it left is not known (a lost track or a lost label
    looks the same), and the trip starts at the first position seen outside."""
    own_active_left_s: float
    """Own-active time remaining at the trip's start."""
    back: Seen | None
    """First tracked position back in its own zone, if before its own-active time ended."""
    home: Arrival | None
    """When it got home (see `Arrival`; earlier than `back` when a crossing or a gap's
    inference shows it crossing home before it was tracked there). The trip ends here."""
    at_active_end: Seen | None
    """When not back before its own-active time ended: the tracked position nearest that
    moment (within 1 s), else None."""
    duration_s: float
    """From the trip's start to `home.t`, or to the end of its own-active time."""
    unseen_s: float
    """Untracked time inside `duration_s`."""
    furthest: Seen
    """Tracked position furthest from its own wall."""
    past_centre_line_m: float
    """How far `furthest` is past the centre line toward the opponent (negative: short)."""
    nearest_home_after_furthest: Seen | None
    """After `furthest`, the tracked position outside its zone nearest its own wall: how
    far back it came, useful when it did not get home before its active time ended."""
    stop_ids: list[str]
    """Stop facts outside its own zone during the trip."""
    longest_below_slow_s: float
    """Longest stretch outside its own zone with speed below SLOW_MPS (tracked or bridged)."""
    path_length_m: float


class TripFact(_Fact):
    """The target left its own alliance zone while its own hub was active (transition,
    its shifts, endgame; not auto). The shape only: positions cannot see fuel, so whether
    the trip was a reload, a feed or wasted is not known."""

    kind: Literal["left_own_zone_while_active"] = "left_own_zone_while_active"
    data: TripData


# --- contacts -------------------------------------------------------------------


class OtherRobot(_Model):
    """Another track that came within CONTACT_M of the target."""

    track_id: int
    team: int | None
    """Team label of that track at that time; None when the track is unlabelled."""
    alliance: Alliance | None
    """The labelled team's alliance from the roster: the authority. None when unlabelled."""
    relation: Relation
    """From the roster alliance of the labelled team, never from bumper colour."""
    bumper_seen: BumperColor
    """Majority bumper colour the detector saw on this track. An observation that can be
    wrong, and not the alliance: some teams run the other alliance's colour or reddish
    bumpers on blue."""
    min_distance_m: float
    close_s: float
    """Tracked time this robot spent within CONTACT_M of the target."""
    near_in_gap_s: float
    """Time while the target was unseen (in a bridged gap) that this robot was within
    CONTACT_M of the target's last-seen or next-seen position."""
    first_close_t: float
    last_close_t: float
    target_closing_mps: float | None
    """In the CONTEXT_S before `first_close_t`: the target's speed toward this robot's
    position (negative: away). None when either was untracked then."""
    other_closing_mps: float | None
    """Same window: this robot's speed toward the target's position."""
    other_speed_before_mps: float | None
    other_speed_during_mps: float | None
    """This robot's mean speed in the window before and while close."""


class SpeedWindow(_Model):
    """Target speed and progress over one window."""

    duration_s: float
    tracked_s: float
    mean_speed_mps: float | None
    net_progress_m: float | None
    """Straight-line distance from the window's first to last tracked position."""


ContactEnd = Literal[
    "target_moved_away",
    "other_moved_away",
    "unknown_other_track_lost",
    "unknown_target_track_lost",
    "match_end",
]


class ContactData(_Model):
    others: list[OtherRobot]
    duration_s: float
    seen_s: float
    """Part of `duration_s` with the target tracked."""
    unseen_s: float
    """Part of `duration_s` with the target untracked: the contact was seen on both sides of
    a gap, or another robot stayed beside where the target sat still across it."""
    min_distance_m: float
    target_zone: Place
    """Where the target was when the contact began."""
    where: Where
    """Median tracked position of the target during the contact."""
    before: SpeedWindow
    during: SpeedWindow
    after: SpeedWindow
    still_s: float
    """Time during the contact, in STILL_WINDOW_S windows, that the target moved slower
    than STILL_MPS (nearly still). Includes `still_unseen_s`."""
    still_unseen_s: float
    """Part of `still_s` where the target was untracked and the straight line between its
    positions either side of the gap was that slow: inferred, a robot can leave and come back
    unseen."""
    longest_still_s: float
    """Longest unbroken run of such windows (seen or inferred): how long it was held, or held
    itself, nearly still beside this robot. Positions cannot say which."""
    end: ContactEnd
    """How the closeness ended. `target_moved_away` / `other_moved_away`: both tracked and
    that robot moved further in the next second. `unknown_*_track_lost`: tracking lost that
    robot; it may still have been there, so nothing is known about how the contact ended."""
    others_tracked_min: int
    others_tracked_max: int
    """Other robots tracked anywhere on the field during the contact (of five)."""


class ContactFact(_Fact):
    """Another tracked robot's centre was within CONTACT_M of the target's: touching or
    close enough to block. Says who moved toward whom and whose speed dropped; it does not
    say who was defending. Contacts with robots the tracker missed do not appear at all."""

    kind: Literal["contact"] = "contact"
    data: ContactData


# --- hub turning on -------------------------------------------------------------


class HubOnData(_Model):
    phase: PhaseName
    previous_phase: PhaseName
    own_hub_active_before: bool | None
    """False: the hub really turned on here. None: unknown whether it was already on."""
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
    """When in its own zone: how long it had been there (bounded by a gap; see
    `since_after_gap`)."""
    since_after_gap: bool
    reached_own_zone_after_s: float | None
    """When not tracked in its own zone: seconds from the activation until it got there
    (`reached.t`, 0 when a crossing or inference puts it home before the activation); None if
    never in this phase."""
    reached: Arrival | None
    """How that arrival is known (`after_gap`: it arrived at most that late)."""


class HubOnFact(_Fact):
    """The target alliance's hub turned on (it was inactive in the phase before): where the
    target was. Phases where the hub stayed on have no hub_on fact."""

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
    first_tracked: Seen | None
    """First tracked position in auto: not necessarily where it started."""
    first_tracked_after_s: float
    """Seconds from auto start to `first_tracked`."""
    waypoints: list[Waypoint]
    """The tracked route simplified to turns (ROUTE_TOLERANCE_M)."""
    last_tracked: Seen | None
    """Last tracked position in auto: not necessarily where it ended."""
    untracked_after_last_s: float
    """Seconds from `last_tracked` to auto end."""
    gap_ids: list[str]
    """Gap facts overlapping auto; each says what its two sides suggest (`inference`)."""
    crossings: list[str]
    """Ids of crossing, crossing_in_gap and crossing_turned_back facts in auto."""
    max_past_centre_line_m: float | None
    """Furthest the robot centre was seen past the centre line toward the opponent
    (negative: it stayed that far short). Only tracked time counts."""
    coverage: Coverage


class AutoRouteFact(_Fact):
    """Where the target was tracked in auto. Reported, not scored."""

    kind: Literal["auto_route"] = "auto_route"
    data: AutoRouteData


Fact = Annotated[
    PhaseFact
    | GapFact
    | ZoneVisitFact
    | CrossingFact
    | CrossingInGapFact
    | CrossingTurnedBackFact
    | StopFact
    | TripFact
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
    "crossing_turned_back",
    "stop",
    "left_own_zone_while_active",
    "contact",
    "hub_on",
    "auto_route",
]


class Facts(StageFile):
    """All facts for one team, sorted by `t_start`, plus match-wide coverage."""

    schema_version: Literal[6] = 6
    team: int
    alliance: Alliance
    clock_rate: float
    """Video seconds per match second used to turn video speeds into real speeds."""
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
