"""facts: reads tracks.parquet, identity.json, match.json.

Writes facts.json (see `scout.schemas.facts`): timed, measured events for the target
team's robot, and `facts_check.jpg`, a match timeline with the phases, the target's place,
its speed and every fact, for a person to check against the video.

Steps:

1. The target's path is every track sample inside an identity segment labelled with the
   target team, in time order, within the match (auto start to teleop end). Frames where
   two tracks both claim the target are dropped and reported as a gap, never averaged.
2. Samples further apart than GAP_S are not linked: nothing is measured across them, and
   the hole is a `gap` fact. Speed is the straight-line distance over SPEED_WINDOW_S of
   consecutive samples, so it is never taken across a missing sample, times the match's
   clock rate (video runs slow on broadcasts).
3. A gap of at most BRIDGE_MAX_S whose two sides are at most BRIDGE_M apart is "bridged":
   the robot very probably stayed put. The filled path draws a straight line across bridged
   gaps (and holes under GAP_S); only stillness (stops, barrier stays, contacts) is read
   from it, and every fact that uses it says how much of its time was unseen.
4. Zones use the robot centre with ZONE_HYST_M of hysteresis at each zone line. Stays are
   `zone_visit` facts; a stay of PERCH_MIN_S or more with part of the robot over a bump or
   trench is its own `on_barrier` visit. Bump and trench passes are judged from measured
   (boxed) samples only: seen on the barrier is a `crossing`, last seen one side and next
   seen the other is a `crossing_in_gap`, and a pass that came back out the side it came
   from is folded with the crossing that next got it across into `crossing_turned_back`.
5. Other robots are every other track in the same frame. They are only the robots the
   tracker followed, so facts give the closest tracked robot and the count tracked, and
   never claim a robot was absent.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np
import polars as pl
from numpy.typing import NDArray

from scout.games.rebuilt2026 import (
    ALLIANCE_ZONE_DEPTH_M,
    BARRIER_X,
    CENTER_LINE_X_M,
    FIELD_LENGTH_M,
    HUBS,
    ROBOT_FOOTPRINT_M,
    ROBOT_HALF_DIAGONAL_M,
    Lane,
    PhaseName,
    Zone,
    barrier_at,
    barrier_edge_m,
    guardrail_m,
    lane_at,
    own_wall_m,
    rect_distance_m,
    zone_of,
)
from scout.run import RunDir, Stage
from scout.schemas.common import ALLIANCES, Alliance
from scout.schemas.facts import (
    AutoRouteData,
    AutoRouteFact,
    Barrier,
    BumperColor,
    ContactData,
    ContactEnd,
    ContactFact,
    CostEnd,
    Coverage,
    CrossingData,
    CrossingFact,
    CrossingInGapData,
    CrossingInGapFact,
    CrossingTurnedBackData,
    CrossingTurnedBackFact,
    Fact,
    Facts,
    GapCause,
    GapData,
    GapFact,
    GapInference,
    HubOnData,
    HubOnFact,
    NearInGap,
    OtherRobot,
    Period,
    PhaseData,
    PhaseFact,
    Place,
    Reach,
    Relation,
    Seen,
    SpeedWindow,
    StopData,
    StopEnd,
    StopFact,
    StopStart,
    TripData,
    TripFact,
    Waypoint,
    Where,
    ZoneVisitData,
    ZoneVisitFact,
)
from scout.schemas.identity import Identity
from scout.schemas.match import MatchInfo
from scout.schemas.tracks import read_tracks

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
B = NDArray[np.bool_]

GAP_S = 1.0
"""Samples further apart than this are not linked; the tracker already interpolates
holes up to 1 s inside one track, so this extends the same rule across track changes."""
SPEED_WINDOW_S = 1.0
"""Speed is distance over this much time: 0.2 m position noise is then at most about
0.3 m/s, and a parked robot reads under 0.05 m/s on this clip."""
STOP_MPS = 0.3
"""Below this the robot is called stopped: above parked noise (<0.05 m/s), below slow
collecting (0.5 m/s and up on this clip)."""
STOP_MIN_S = 2.0
"""Shortest stop worth a fact: a shot or a turn on the spot takes about this long."""
ZONE_HYST_M = 0.3
"""A zone change needs the centre this far past the line: 0.2 m accuracy plus margin, so
jitter on a line is one visit."""
ZONE_DWELL_S = 1.0
"""Or this long on the other side of the line, so a robot parked just inside a line (the
audit's shooter sat 0.09 m inside its zone for 6 s) is on the side it is on."""
BAND_MARGIN_M = 0.2
"""The centre is on a trench (or the opponent's barrier) within this of its depth: position
accuracy. Trenches are flat floor under an arm, so only the centre's place matters there."""
CORNER_M = ROBOT_HALF_DIAGONAL_M
"""On the neutral face of its own bumps (the climb home) the band reaches this far: a corner
can touch the ramp from here if the robot is turned. The audit's nose-on-ramp stall read
0.66 m (partial boxes, robot diagonal in frame); 0.475 m catches 3 of the 4 frame-checked
stalls, this all 4 plus 4 frame-checked nose-in-and-back touches the human did not list."""
BAND_HYST_M = 0.2
"""Going back to the side it came from needs the centre this far clear of the band (position
accuracy), so a robot sitting on the band's edge is one attempt, not a flicker of them."""
PERCH_M = ROBOT_FOOTPRINT_M / 2
"""Centre within this of the bumps' depth puts part of the robot over them whatever its
heading (`reach` `footprint`)."""
PERCH_MIN_S = 2.5
"""A barrier stay this long is its own visit: a clean pass across the band takes 0.5 to 1 s,
a slow one 2 s."""
NORMAL_CROSSING_S = 1.2
"""Slowest normal pass across the band: 90% of 71 clean seen dev crossings took at most
1.06 s (own bumps median 0.77 s), plus margin. Time beyond it is `excess_s`; each 0.1 s
changes every excess by 0.1 s."""
BRIDGE_MAX_S = 6.0
"""Longest gap that can be bridged: the audit's pinned and stalled robots went unseen for
1.4 to 5 s, and leaving and coming back to the same spot takes longer than this."""
BRIDGE_M = 0.5
"""Most a robot may move across a bridged gap: two 0.2 m position errors plus margin."""
NEVER_CLEARED_M = PERCH_M
"""A crossing whose far side stayed within this of the bumps' far end before it came back
never got off the barrier (same as PERCH_M)."""
STILL_MPS = 0.4
"""Nearly still in a contact: a pinned robot pushing back read 0.1 to 0.3 m/s; slow
collecting is 0.5 m/s and up."""
STILL_WINDOW_S = 1.0
"""Window for that speed, so 0.2 m position noise reads at most 0.3 m/s."""
CONTACT_M = 1.6
"""Centre distance for a contact: touching robots are 0.95 m (side by side) to 1.34 m
(corner to corner) apart, plus 0.2 m accuracy on each."""
CONTACT_BRIDGE_S = 0.5
"""Contact samples closer in time than this are one contact (bumpers bounce apart)."""
CONTACT_GAP_MAX_S = 6.0
"""A contact seen on both sides of a target gap up to this long is one episode (the audit's
pins hid in gaps of 4.5 to 4.7 s)."""
CONTACT_EDGE_S = 1.0
"""For that, the contact must be seen within this of the gap's start and end."""
CONTACT_STAY_M = 2.5
"""Two contacts are one when their closest robot stayed within this between them: the
audit's pinner read 1.7 to 2.2 m from the pinned target (box placement bias)."""
SAME_ROBOT_M = 0.7
"""An unlabelled track this close to the target's spot is probably the target itself."""
NEAR_MIN_S = 0.5
"""A robot beside a still target's spot for this long in a gap makes an inferred contact."""
CONTEXT_S = 2.0
"""Length of the before and after windows around a contact."""
MOTION_TOL_S = 0.5
"""Who-moved-toward-whom uses positions this close in time to the window's ends (other
robots' tracks are patchy)."""
ROUTE_TOLERANCE_M = 0.75
"""The auto route keeps a point where the path bends more than this from a straight line."""
TRIP_MIN_S = 2.0
"""Shortest trip out of the own zone worth a fact: shorter is jitter on the line."""
SLOW_MPS = 0.8
"""Slow stretch on a trip: below driving pace (1.5 m/s and up), above slow collecting."""
HEADING_MPS = 0.5
"""A gap edge counts as driving toward or away from the own wall above this speed; a pass
that only a corner could have touched counts only when driving toward the bumps this fast."""
NEAR_BARRIER_M = 1.5
"""A gap edge this close to the own barrier's depth counts as at the barrier."""
OWN_INFER_MIN_S = 2.0
"""Shortest gap read as a trip into the own zone: a turn in front of the barrier is ~1 s."""

THRESHOLDS: dict[str, float] = {
    "GAP_S": GAP_S,
    "SPEED_WINDOW_S": SPEED_WINDOW_S,
    "STOP_MPS": STOP_MPS,
    "STOP_MIN_S": STOP_MIN_S,
    "ZONE_HYST_M": ZONE_HYST_M,
    "ZONE_DWELL_S": ZONE_DWELL_S,
    "BAND_MARGIN_M": BAND_MARGIN_M,
    "CORNER_M": CORNER_M,
    "BAND_HYST_M": BAND_HYST_M,
    "PERCH_M": PERCH_M,
    "PERCH_MIN_S": PERCH_MIN_S,
    "NORMAL_CROSSING_S": NORMAL_CROSSING_S,
    "BRIDGE_MAX_S": BRIDGE_MAX_S,
    "BRIDGE_M": BRIDGE_M,
    "NEVER_CLEARED_M": NEVER_CLEARED_M,
    "STILL_MPS": STILL_MPS,
    "STILL_WINDOW_S": STILL_WINDOW_S,
    "CONTACT_M": CONTACT_M,
    "CONTACT_BRIDGE_S": CONTACT_BRIDGE_S,
    "CONTACT_GAP_MAX_S": CONTACT_GAP_MAX_S,
    "CONTACT_EDGE_S": CONTACT_EDGE_S,
    "CONTACT_STAY_M": CONTACT_STAY_M,
    "SAME_ROBOT_M": SAME_ROBOT_M,
    "NEAR_MIN_S": NEAR_MIN_S,
    "CONTEXT_S": CONTEXT_S,
    "MOTION_TOL_S": MOTION_TOL_S,
    "ROUTE_TOLERANCE_M": ROUTE_TOLERANCE_M,
    "TRIP_MIN_S": TRIP_MIN_S,
    "SLOW_MPS": SLOW_MPS,
    "HEADING_MPS": HEADING_MPS,
    "NEAR_BARRIER_M": NEAR_BARRIER_M,
    "OWN_INFER_MIN_S": OWN_INFER_MIN_S,
}

CHECK_IMAGE = "facts_check.jpg"
SHIFT_PHASES: tuple[PhaseName, ...] = ("shift1", "shift2", "shift3", "shift4", "endgame")


def _r(v: float, nd: int = 2) -> float:
    return round(float(v), nd)


def _ro(v: float | None, nd: int = 2) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v), nd)


# --- inputs ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Path:
    """The target's samples in time order. `link[i]` says samples i and i+1 are at most
    GAP_S apart; `bridged[i]` says they are not but the gap is bridged (see step 3);
    `speed` is NaN where the window would cross a missing sample."""

    frame: I64
    t: F64
    x: F64
    y: F64
    measured: B
    track: I64
    link: B
    bridged: B
    speed: F64
    zone: list[Zone]


@dataclass(frozen=True)
class Filled:
    """The path with bridged gaps and holes under GAP_S filled by straight lines, one
    sample per `step` frames inside each filled stretch. `src[k]` is the path index of
    sample k, -1 where filled. Read only for stillness."""

    frame: I64
    t: F64
    x: F64
    y: F64
    src: I64
    speed: F64
    of_path: I64
    """Path index -> filled index."""


@dataclass(frozen=True)
class Others:
    """Every other track's samples, by frame and by track, with labels."""

    by_frame: dict[int, list[tuple[int, float, float]]]
    """frame -> (track_id, x, y) of other tracks live in that frame."""
    by_track: dict[int, tuple[I64, F64, F64]]
    """track_id -> (frames, x, y)."""
    team: dict[tuple[int, int], int]
    """(track_id, frame) -> labelled team, for labelled samples."""
    color: dict[int, BumperColor]


@dataclass(frozen=True)
class Ctx:
    """Everything the fact builders share."""

    match: MatchInfo
    alliance: Alliance
    fps: float
    step: int
    dt: float
    rate: float
    """Video seconds per match second; speeds are multiplied by it."""
    t0: float
    t1: float
    path: Path
    filled: Filled
    others: Others
    grid: I64
    """Sampled frames in the match."""
    others_count: I64
    """Other tracks live in each `grid` frame."""
    conflicts: list[tuple[int, int]]
    """Frame ranges where two tracks were both labelled the target; dropped."""
    track_frames: dict[int, I64]
    """Every track's frames, for telling a lost track from a dropped label."""
    team_share: dict[int, float]
    """Share of sampled match frames with a track labelled as each roster team."""
    active: list[tuple[float, float]]
    """Teleop spans with the own hub active, merged (auto is the robot's code)."""
    place: list[Place] = field(default_factory=list[Place])
    """Per path sample: its zone, or `on_barrier` inside a barrier stay of PERCH_MIN_S."""
    perch: dict[int, tuple[Barrier, Lane]] = field(default_factory=dict[int, tuple[Barrier, Lane]])
    """Path index -> barrier and lane, for samples whose place is `on_barrier`."""

    def frame(self, t: float) -> int:
        return round(t * self.fps)

    def seen(self, i: int) -> Seen:
        p = self.path
        return Seen(t=_r(p.t[i]), x=_r(p.x[i]), y=_r(p.y[i]), zone=p.zone[i])

    def raw_zone(self, x: float) -> Zone:
        return zone_of(x, self.alliance)

    def toward_own(self, dx: float) -> float:
        """A field-x displacement as a displacement toward the target's own wall."""
        return -dx if self.alliance == "blue" else dx

    def band_edges(self, barrier: Barrier, y: float) -> tuple[float, float] | None:
        """Field-x extent of `barrier`'s band at y; None beside a hub face. The band is the
        bumps' depth widened by BAND_MARGIN_M, by CORNER_M on the neutral face of the own
        bumps, and cut at the own zone line: a centre in its own zone is in its own zone."""
        lane = lane_at(y)
        if lane is None:
            return None
        x0, x1 = BARRIER_X[barrier]
        if barrier != self.alliance:
            return x0 - BAND_MARGIN_M, x1 + BAND_MARGIN_M
        reach = CORNER_M if lane == "bump" else BAND_MARGIN_M
        line = ALLIANCE_ZONE_DEPTH_M
        if barrier == "blue":
            return line, x1 + reach
        return x0 - reach, FIELD_LENGTH_M - line

    def band(self, x: float, y: float) -> tuple[Barrier, Lane] | None:
        """The barrier band (see `band_edges`) and lane under (x, y), if any."""
        for barrier in ALLIANCES:
            edges = self.band_edges(barrier, y)
            if edges is not None and edges[0] <= x <= edges[1]:
                lane = lane_at(y)
                assert lane is not None
                return barrier, lane
        return None

    def where(self, x: float, y: float) -> Where:
        return Where(
            x=_r(x),
            y=_r(y),
            zone=self.raw_zone(x),
            into_own_zone_m=_r(ALLIANCE_ZONE_DEPTH_M - own_wall_m(x, self.alliance)),
            own_hub_m=_r(rect_distance_m(HUBS[self.alliance], x, y)),
            own_wall_m=_r(own_wall_m(x, self.alliance)),
            guardrail_m=_r(guardrail_m(y)),
        )

    def active_s(self, a: float, b: float) -> float:
        """Seconds of [a, b] in teleop time with the own hub active."""
        return sum(max(0.0, min(b, e) - max(a, s)) for s, e in self.active)


def _load(run: RunDir) -> Ctx:
    """Read the inputs and build the target's path and the other tracks."""
    match = MatchInfo.load(run.match_json)
    ident = Identity.load(run.identity_json)
    tracks = read_tracks(run.tracks)
    t0, t1 = match.auto_start_t, match.teleop_end_t
    rate = match.clock_rate or 1.0
    some = tracks.filter(pl.col("t") > 0).row(0, named=True)
    fps = float(some["frame"]) / float(some["t"])
    frames: I64 = np.unique(tracks["frame"].to_numpy().astype(np.int64))
    step = int(np.median(np.diff(frames)))
    target = match.target_team

    seg = pl.DataFrame(
        {
            "track_id": [s.track_id for s in ident.segments if s.team is not None],
            "f0": [s.frame_start for s in ident.segments if s.team is not None],
            "f1": [s.frame_end for s in ident.segments if s.team is not None],
            "team": [s.team for s in ident.segments if s.team is not None],
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
    rows = tracks.filter(pl.col("t").is_between(t0, t1)).join(
        labels, on=["track_id", "frame"], how="left"
    )
    claims = [s for s in ident.segments if s.team == target]
    conflicts = [
        (max(a.frame_start, b.frame_start), min(a.frame_end, b.frame_end))
        for k, a in enumerate(claims)
        for b in claims[k + 1 :]
        if a.track_id != b.track_id
        and max(a.frame_start, b.frame_start) <= min(a.frame_end, b.frame_end)
    ]
    mine = rows.filter(pl.col("team") == target)
    for f0, f1 in conflicts:
        mine = mine.filter(~pl.col("frame").is_between(f0, f1))
    mine = mine.sort("frame")
    others = rows.filter(pl.col("team").is_null() | (pl.col("team") != target))

    frame = mine["frame"].to_numpy().astype(np.int64)
    t = mine["t"].to_numpy().astype(np.float64)
    x = mine["x"].to_numpy().astype(np.float64)
    y = mine["y"].to_numpy().astype(np.float64)
    link = np.diff(t) <= GAP_S + 1e-6
    for f0, f1 in conflicts:
        link &= ~((frame[:-1] < f0) & (frame[1:] > f1))
    moved = np.hypot(np.diff(x), np.diff(y))
    bridged = ~link & (np.diff(t) <= BRIDGE_MAX_S + 1e-6) & (moved <= BRIDGE_M)
    speed = _speed(frame, t, x, y, step, fps) * rate
    path = Path(
        frame=frame,
        t=t,
        x=x,
        y=y,
        measured=(mine["state"] == "measured").to_numpy(),
        track=mine["track_id"].to_numpy().astype(np.int64),
        link=link,
        bridged=bridged,
        speed=speed,
        zone=_zones(t, x, link, match.target_alliance),
    )

    by_frame: dict[int, list[tuple[int, float, float]]] = {}
    for tid, f, ox, oy in others.select("track_id", "frame", "x", "y").iter_rows():
        by_frame.setdefault(int(f), []).append((int(tid), float(ox), float(oy)))
    by_track: dict[int, tuple[I64, F64, F64]] = {}
    for (tid,), g in others.sort("frame").group_by("track_id"):
        assert isinstance(tid, int)
        by_track[tid] = (
            g["frame"].to_numpy().astype(np.int64),
            g["x"].to_numpy().astype(np.float64),
            g["y"].to_numpy().astype(np.float64),
        )
    team = {
        (int(tid), int(f)): int(tm)
        for tid, f, tm in others.filter(pl.col("team").is_not_null())
        .select("track_id", "frame", "team")
        .iter_rows()
    }
    color: dict[int, BumperColor] = {}
    for tid, c in tracks.select("track_id", "color").unique("track_id").iter_rows():
        color[int(tid)] = "red" if c == "red" else "blue" if c == "blue" else "unknown"
    track_frames = {
        int(tid): g["frame"].to_numpy().astype(np.int64)
        for (tid,), g in tracks.select("track_id", "frame").group_by("track_id")
        if isinstance(tid, int)
    }

    grid = np.arange(math.ceil(t0 * fps / step) * step, math.floor(t1 * fps) + 1, step)
    counts = np.array([len(by_frame.get(int(f), [])) for f in grid], np.int64)
    labelled = dict(
        rows.filter(pl.col("team").is_not_null())
        .group_by("team")
        .agg(pl.col("frame").n_unique())
        .iter_rows()
    )
    team_share = {
        s.team: _r(min(1.0, int(labelled.get(s.team, 0)) / max(1, len(grid))), 3)
        for s in match.roster
    }
    ctx = Ctx(
        match=match,
        alliance=match.target_alliance,
        fps=fps,
        step=step,
        dt=step / fps,
        rate=rate,
        t0=t0,
        t1=t1,
        path=path,
        filled=_fill(path, step, fps, rate),
        others=Others(by_frame=by_frame, by_track=by_track, team=team, color=color),
        grid=grid.astype(np.int64),
        others_count=counts,
        conflicts=conflicts,
        track_frames=track_frames,
        team_share=team_share,
        active=_own_active_spans(match),
    )
    place, perch = _places(ctx)
    ctx.place.extend(place)
    ctx.perch.update(perch)
    return ctx


def _speed(frame: I64, t: F64, x: F64, y: F64, step: int, fps: float) -> F64:
    """Speed per video second at each sample over SPEED_WINDOW_S centred on it; NaN when
    any sample in the window is missing."""
    h = max(1, round(SPEED_WINDOW_S * fps / step / 2))
    out = np.full(len(t), np.nan)
    for i in range(h, len(t) - h):
        a, b = i - h, i + h
        if frame[b] - frame[a] == 2 * h * step:
            out[i] = math.hypot(x[b] - x[a], y[b] - y[a]) / (t[b] - t[a])
    return out


def _fill(p: Path, step: int, fps: float, rate: float) -> Filled:
    """The filled path (see `Filled`)."""
    fr: list[int] = []
    xs: list[float] = []
    ys: list[float] = []
    src: list[int] = []
    n = len(p.t)
    for i in range(n):
        fr.append(int(p.frame[i]))
        xs.append(float(p.x[i]))
        ys.append(float(p.y[i]))
        src.append(i)
        if i + 1 < n and (p.link[i] or p.bridged[i]):
            f0, f1 = int(p.frame[i]), int(p.frame[i + 1])
            k = round((f1 - f0) / step)
            for j in range(1, k):
                a = j / k
                fr.append(f0 + j * step)
                xs.append(float(p.x[i] + a * (p.x[i + 1] - p.x[i])))
                ys.append(float(p.y[i] + a * (p.y[i + 1] - p.y[i])))
                src.append(-1)
    frame = np.array(fr, np.int64)
    t = frame / fps
    x, y = np.array(xs), np.array(ys)
    s = np.array(src, np.int64)
    return Filled(
        frame=frame,
        t=t,
        x=x,
        y=y,
        src=s,
        speed=_speed(frame, t, x, y, step, fps) * rate,
        of_path=np.flatnonzero(s >= 0).astype(np.int64),
    )


def _zones(t: F64, x: F64, link: B, alliance: Alliance) -> list[Zone]:
    """Zone per sample with ZONE_HYST_M hysteresis, restarting after every gap: the zone
    changes when the centre is clear of the line by the margin, or has stayed on the other
    side of it for ZONE_DWELL_S (then from where that stay began)."""
    out: list[Zone] = []
    cur: Zone = "neutral"
    since: int | None = None
    for i, xi in enumerate(x.tolist()):
        raw = zone_of(xi, alliance)
        lo, hi = zone_of(xi - ZONE_HYST_M, alliance), zone_of(xi + ZONE_HYST_M, alliance)
        if i == 0 or not link[i - 1]:
            cur, since = raw, None
        elif lo == hi:
            cur, since = lo, None
        elif raw == cur:
            since = None
        elif since is None or zone_of(float(x[since]), alliance) != raw:
            since = i
        out.append(cur)
        if since is not None and t[i] - t[since] >= ZONE_DWELL_S - 1e-6:
            fill: list[Zone] = [raw] * (i - since + 1)
            out[since:] = fill
            cur, since = raw, None
    return out


def _places(ctx: Ctx) -> tuple[list[Place], dict[int, tuple[Barrier, Lane]]]:
    """Each sample's zone, replaced by `on_barrier` inside stays of PERCH_MIN_S or more on
    one barrier's band (filled path, so a stay survives a bridged gap)."""
    p, f = ctx.path, ctx.filled
    on = [ctx.band(float(x), float(y)) for x, y in zip(f.x, f.y, strict=True)]
    place: list[Place] = list(p.zone)
    perch: dict[int, tuple[Barrier, Lane]] = {}
    mask = np.array([o is not None for o in on], np.bool_)

    def same(k: int) -> bool:
        a, b = on[k], on[k + 1]
        return _f_contiguous(ctx, k) and a is not None and b is not None and a[0] == b[0]

    for a, b in _runs(mask, same):
        if f.t[b] - f.t[a] + ctx.dt < PERCH_MIN_S - 1e-6:
            continue
        lanes: Counter[Lane] = Counter(o[1] for o in on[a : b + 1] if o is not None)
        first = on[a]
        assert first is not None
        lane = max(lanes, key=lambda k: lanes[k])
        for k in range(a, b + 1):
            if f.src[k] >= 0:
                place[int(f.src[k])] = "on_barrier"
                perch[int(f.src[k])] = (first[0], lane)
    return place, perch


# --- helpers --------------------------------------------------------------------------


def _clock(ctx: Ctx, t: float) -> tuple[Period, str]:
    """Scoreboard clock at video time `t`: remaining time rounded up, as displayed."""
    m = ctx.match
    if t < m.auto_start_t:
        return "pre_match", "0:20"
    if t < m.auto_end_t:
        fit = m.clock_fits["auto"]
        left, period = 20.0 - (t - fit.start_t) / fit.slope, "auto"
    elif t < m.teleop_start_t:
        return "pause", "2:20"
    elif t <= m.teleop_end_t:
        fit = m.clock_fits["teleop"]
        left, period = 140.0 - (t - fit.start_t) / fit.slope, "teleop"
    else:
        return "post_match", "0:00"
    s = max(0, math.ceil(left - 1e-6))
    return period, f"{s // 60}:{s % 60:02d}"


def _phase_at(ctx: Ctx, t: float) -> PhaseName | None:
    for p in ctx.match.phases:
        if p.t_start <= t < p.t_end or (p.name == "endgame" and t == p.t_end):
            return p.name
    return None


def _common(ctx: Ctx, fid: str, t_start: float, t_end: float) -> dict[str, object]:
    """The fields every fact shares, from its time span."""
    period, clock = _clock(ctx, t_start)
    return {
        "id": fid,
        "t_start": _r(t_start),
        "t_end": _r(t_end),
        "frame_start": ctx.frame(t_start),
        "frame_end": ctx.frame(t_end),
        "period": period,
        "clock": clock,
        "phase": _phase_at(ctx, t_start),
    }


def _in_span(ctx: Ctx, a: float, b: float) -> I64:
    """Path indices with a <= t <= b."""
    t = ctx.path.t
    return np.flatnonzero((t >= a - 1e-6) & (t <= b + 1e-6))


def _f_span(ctx: Ctx, a: float, b: float) -> I64:
    """Filled indices with a <= t <= b."""
    t = ctx.filled.t
    return np.flatnonzero((t >= a - 1e-6) & (t <= b + 1e-6))


def _path_length(ctx: Ctx, idx: I64) -> float:
    """Distance along linked consecutive samples among `idx` (a contiguous index range)."""
    if len(idx) < 2:
        return 0.0
    p = ctx.path
    a, b = idx[:-1], idx[1:]
    ok = (b == a + 1) & p.link[a]
    return float(np.hypot(p.x[b] - p.x[a], p.y[b] - p.y[a])[ok].sum())


def _mean(v: F64) -> float | None:
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else None


def _nearest(ctx: Ctx, i: int) -> list[tuple[float, int]]:
    """(distance, track_id) of every other tracked robot in sample i's frame."""
    p = ctx.path
    return sorted(
        (math.hypot(ox - p.x[i], oy - p.y[i]), tid)
        for tid, ox, oy in ctx.others.by_frame.get(int(p.frame[i]), [])
    )


def _others_range(ctx: Ctx, a: float, b: float) -> tuple[float, int, int]:
    """Mean, min and max count of other tracked robots per sampled frame in [a, b]."""
    tg = ctx.grid / ctx.fps
    c = ctx.others_count[(tg >= a - 1e-6) & (tg <= b + 1e-6)]
    if len(c) == 0:
        return 0.0, 0, 0
    return float(c.mean()), int(c.min()), int(c.max())


def _runs(mask: B, joined: Callable[[int], bool]) -> list[tuple[int, int]]:
    """(first, last) index runs where `mask` holds and `joined(i)` links i to i+1."""
    out: list[tuple[int, int]] = []
    for i in np.flatnonzero(mask).tolist():
        if out and out[-1][1] == i - 1 and joined(i - 1):
            out[-1] = (out[-1][0], i)
        else:
            out.append((i, i))
    return out


def _f_contiguous(ctx: Ctx, k: int) -> bool:
    """Filled sample k+1 directly follows sample k."""
    return bool(ctx.filled.frame[k + 1] - ctx.filled.frame[k] == ctx.step)


def _f_still_s(ctx: Ctx, idx: I64, below: float) -> tuple[float, float]:
    """(total, longest run) seconds among filled indices `idx` with speed below `below`."""
    if not len(idx):
        return 0.0, 0.0
    sp = np.nan_to_num(ctx.filled.speed[idx], nan=np.inf) < below
    runs = _runs(sp, lambda k: bool(idx[k + 1] == idx[k] + 1) and _f_contiguous(ctx, int(idx[k])))
    return float(sp.sum()) * ctx.dt, max(((b - a + 1) * ctx.dt for a, b in runs), default=0.0)


def _team_of(ctx: Ctx, team: int | None) -> tuple[Alliance | None, Relation]:
    """Roster alliance and relation of a labelled team: the roster is the authority."""
    slot = next((s for s in ctx.match.roster if s.team == team), None)
    if slot is None:
        return None, "unknown"
    return slot.alliance, "partner" if slot.alliance == ctx.alliance else "opponent"


def _other_near(ctx: Ctx, tid: int, frame: int, tol: int) -> tuple[float, float] | None:
    """Track `tid`'s position at the sample nearest `frame`, within `tol` frames."""
    track = ctx.others.by_track.get(tid)
    if track is None or not len(track[0]):
        return None
    k = int(np.searchsorted(track[0], frame))
    best = min(
        (j for j in (k - 1, k) if 0 <= j < len(track[0])), key=lambda j: abs(track[0][j] - frame)
    )
    if abs(int(track[0][best]) - frame) > tol:
        return None
    return float(track[1][best]), float(track[2][best])


def _target_near(ctx: Ctx, t: float, tol_s: float) -> tuple[float, float] | None:
    """Target position at the path sample nearest `t`, within `tol_s`."""
    p = ctx.path
    if not len(p.t):
        return None
    k = int(np.clip(np.searchsorted(p.t, t), 0, len(p.t) - 1))
    best = min((j for j in (k - 1, k) if 0 <= j < len(p.t)), key=lambda j: abs(p.t[j] - t))
    if abs(p.t[best] - t) > tol_s:
        return None
    return float(p.x[best]), float(p.y[best])


def _near_in_gap(
    ctx: Ctx, j0: int | None, j1: int | None
) -> dict[int, list[tuple[int, float, bool]]]:
    """Other tracks within CONTACT_M of the target's last-seen (j0) or next-seen (j1)
    position while it was unseen between them: track -> [(frame, distance, True)], the
    shape of contact hits. Unlabelled tracks closer than SAME_ROBOT_M are left out (they are
    probably the target)."""
    p = ctx.path
    ends = [(float(p.x[j]), float(p.y[j])) for j in (j0, j1) if j is not None]
    fa = int(p.frame[j0]) if j0 is not None else ctx.frame(ctx.t0)
    fb = int(p.frame[j1]) if j1 is not None else ctx.frame(ctx.t1)
    out: dict[int, list[tuple[int, float, bool]]] = {}
    for f in ctx.grid[(ctx.grid > fa) & (ctx.grid < fb)].tolist():
        for tid, ox, oy in ctx.others.by_frame.get(f, []):
            d = min(math.hypot(ox - ex, oy - ey) for ex, ey in ends)
            labelled = (tid, f) in ctx.others.team
            if d <= CONTACT_M and (d >= SAME_ROBOT_M or labelled):
                out.setdefault(tid, []).append((f, d, True))
    return out


def _label(ctx: Ctx, tid: int, frames: Sequence[int]) -> int | None:
    """Most common team label of track `tid` over `frames` (None when unlabelled)."""
    teams = Counter(ctx.others.team.get((tid, f)) for f in frames)
    return teams.most_common(1)[0][0] if teams else None


# --- gaps -----------------------------------------------------------------------------


@dataclass(frozen=True)
class GapSpan:
    """A hole of more than GAP_S: path indices either side (None at match start/end)."""

    before: int | None
    after: int | None
    a: float
    b: float


def _gap_spans(ctx: Ctx) -> list[GapSpan]:
    p = ctx.path
    out: list[GapSpan] = []
    if len(p.t) == 0:
        return [GapSpan(None, None, ctx.t0, ctx.t1)]
    if p.t[0] - ctx.t0 > GAP_S:
        out.append(GapSpan(None, 0, ctx.t0, float(p.t[0])))
    for i in np.flatnonzero(~p.link).tolist():
        out.append(GapSpan(i, i + 1, float(p.t[i]), float(p.t[i + 1])))
    if ctx.t1 - p.t[-1] > GAP_S:
        out.append(GapSpan(len(p.t) - 1, None, float(p.t[-1]), ctx.t1))
    return out


def _heading(ctx: Ctx, i: int, back: bool) -> float | None:
    """Speed toward the own wall at gap edge i, from up to 0.5 s of linked samples on its
    tracked side (`back`: before i); None when under 0.2 s is available."""
    p = ctx.path
    k = i
    while True:
        nk = k - 1 if back else k + 1
        if not 0 <= nk < len(p.t) or not p.link[min(k, nk)] or abs(p.t[nk] - p.t[i]) > 0.5:
            break
        k = nk
    span = abs(float(p.t[i] - p.t[k]))
    if span < 0.2:
        return None
    a, b = (k, i) if back else (i, k)
    return ctx.toward_own(float(p.x[b] - p.x[a])) / float(p.t[b] - p.t[a]) * ctx.rate


def _inference(ctx: Ctx, g: GapSpan) -> GapInference | None:
    """What the two sides of a gap suggest (see `GapData.inference`)."""
    p = ctx.path
    if g.before is None or g.after is None:
        return None
    if p.bridged[g.before]:
        return "stayed_in_place"
    if g.b - g.a < OWN_INFER_MIN_S:
        return None

    def at_own(i: int) -> bool:
        on = barrier_at(float(p.x[i]), float(p.y[i]), NEAR_BARRIER_M)
        return on is not None and on[0] == ctx.alliance

    vin, vout = _heading(ctx, g.before, True), _heading(ctx, g.after, False)
    if not at_own(g.before) or vin is None or vin < HEADING_MPS:
        return None
    if at_own(g.after) and vout is not None and vout <= -HEADING_MPS:
        return "in_own_zone_beyond_barrier"
    if ctx.raw_zone(float(p.x[g.after])) == "own":
        return "entered_own_zone"
    return None


def _gap_facts(ctx: Ctx, spans: list[GapSpan]) -> list[Fact]:
    p = ctx.path
    out: list[Fact] = []
    for n, g in enumerate(spans, 1):
        cause: GapCause
        if g.before is None:
            cause = "match_start"
        elif g.after is None:
            cause = "match_end"
        else:
            f0, f1 = int(p.frame[g.before]), int(p.frame[g.after])
            mine = ctx.track_frames[int(p.track[g.before])]
            inside = mine[(mine > f0) & (mine < f1)]
            if any(c0 <= f1 and c1 >= f0 for c0, c1 in ctx.conflicts):
                cause = "label_conflict"
            elif len(inside):
                cause = "label_ended"
            else:
                cause = "track_ended"
        before = ctx.seen(g.before) if g.before is not None else None
        after = ctx.seen(g.after) if g.after is not None else None
        line = math.hypot(after.x - before.x, after.y - before.y) if before and after else None
        near: list[NearInGap] = []
        found = _near_in_gap(ctx, g.before, g.after)
        for tid, hits in sorted(found.items(), key=lambda kv: min(d for _, d, _ in kv[1])):
            fb, d, _ = min(hits, key=lambda h: h[1])
            at = fb / ctx.fps
            team = _label(ctx, tid, [fb])
            near.append(
                NearInGap(
                    track_id=tid,
                    team=team,
                    relation=_team_of(ctx, team)[1],
                    min_distance_m=_r(d),
                    t=_r(at),
                )
            )
        data = GapData(
            duration_s=_r(g.b - g.a),
            before=before,
            after=after,
            straight_line_m=_ro(line),
            cause=cause,
            inference=_inference(ctx, g),
            others_near=near,
        )
        out.append(GapFact.model_validate({**_common(ctx, f"gap-{n:02d}", g.a, g.b), "data": data}))
    return out


def _coverage(ctx: Ctx, a: float, b: float, gaps: list[GapSpan]) -> Coverage:
    """Coverage of [a, b): one sample stands for `dt` seconds of tracked time."""
    p = ctx.path
    idx = np.flatnonzero((p.t >= a - 1e-6) & (p.t < b - 1e-6))
    dur = max(b - a, 1e-9)
    zones = Counter(p.zone[i] for i in idx.tolist())
    tracked = min(dur, len(idx) * ctx.dt)
    mean, lo, hi = _others_range(ctx, a, b)
    overlap = [min(g.b, b) - max(g.a, a) for g in gaps if g.a < b and g.b > a]
    return Coverage(
        duration_s=_r(dur),
        tracked_s=_r(tracked),
        tracked_share=_r(tracked / dur, 3),
        measured_share=_r(min(dur, int(p.measured[idx].sum()) * ctx.dt) / dur, 3),
        own_s=_r(zones["own"] * ctx.dt),
        neutral_s=_r(zones["neutral"] * ctx.dt),
        opponent_s=_r(zones["opponent"] * ctx.dt),
        untracked_s=_r(dur - tracked),
        gaps=len(overlap),
        longest_gap_s=_r(max(overlap, default=0.0)),
        path_length_m=_r(_path_length(ctx, idx), 1),
        others_tracked_mean=_r(mean),
        others_tracked_min=lo,
        others_tracked_max=hi,
    )


def _phase_facts(ctx: Ctx, gaps: list[GapSpan]) -> list[Fact]:
    out: list[Fact] = []
    for ph in ctx.match.phases:
        active = None if ph.active_hubs is None else ctx.alliance in ph.active_hubs
        data = PhaseData(own_hub_active=active, coverage=_coverage(ctx, ph.t_start, ph.t_end, gaps))
        out.append(
            PhaseFact.model_validate(
                {**_common(ctx, f"phase-{ph.name}", ph.t_start, ph.t_end), "data": data}
            )
        )
    return out


# --- zones ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Visit:
    place: Place
    first: int
    last: int


def _visits(ctx: Ctx) -> list[Visit]:
    """Runs of one place, split at gaps."""
    p = ctx.path
    out: list[Visit] = []
    for i in range(len(p.t)):
        if out and out[-1].place == ctx.place[i] and p.link[i - 1]:
            out[-1] = Visit(ctx.place[i], out[-1].first, i)
        else:
            out.append(Visit(ctx.place[i], i, i))
    return out


def _visit_facts(ctx: Ctx, visits: list[Visit]) -> list[Fact]:
    p = ctx.path
    out: list[Fact] = []
    for n, v in enumerate(visits, 1):
        k = n - 1
        prev = visits[k - 1] if k else None
        nxt = visits[k + 1] if k + 1 < len(visits) else None
        came = (
            prev.place
            if prev and p.link[v.first - 1]
            else "match_start"
            if v.first == 0 and p.t[0] - ctx.t0 <= GAP_S
            else "untracked"
        )
        went = (
            nxt.place
            if nxt and p.link[v.last]
            else "match_end"
            if nxt is None and ctx.t1 - p.t[-1] <= GAP_S
            else "untracked"
        )
        idx = np.arange(v.first, v.last + 1)
        dur = float(p.t[v.last] - p.t[v.first])
        holes = np.diff(p.t[idx]) - ctx.dt
        length = _path_length(ctx, idx)
        perch = ctx.perch.get(v.first) if v.place == "on_barrier" else None
        data = ZoneVisitData(
            zone=v.place,
            barrier=perch[0] if perch else None,
            lane=perch[1] if perch else None,
            duration_s=_r(dur),
            untracked_s=_r(float(holes[holes > 1e-6].sum())),
            came_from=came,
            went_to=went,
            path_length_m=_r(length, 1),
            mean_speed_mps=_r(length / dur * ctx.rate) if dur > 0 else None,
            excess_s=_r(max(0.0, dur - NORMAL_CROSSING_S)) if perch else None,
            own_active_s=_r(ctx.active_s(float(p.t[v.first]), float(p.t[v.last]))),
        )
        out.append(
            ZoneVisitFact.model_validate(
                {
                    **_common(ctx, f"zone-{n:02d}", float(p.t[v.first]), float(p.t[v.last])),
                    "data": data,
                }
            )
        )
    return out


# --- crossings ------------------------------------------------------------------------


@dataclass(frozen=True)
class Pass:
    """One run of boxes on a barrier band, before it becomes a fact."""

    id: str
    t_in: float
    t_out: float
    first: int
    last: int
    outcome: str
    """`crossed`, `returned` or `unknown` (see `CrossingData.outcome`)."""
    barrier: Barrier
    lane: Lane
    y_m: float
    from_zone: Zone | None
    to_zone: Zone | None
    nearest_edge_m: float
    """Closest the centre came to the bumps' or trench's depth along x."""
    data: dict[str, object]
    """The other `CrossingData` fields."""

    def crossing(self) -> CrossingData:
        return CrossingData.model_validate(
            {
                **self.data,
                "barrier": self.barrier,
                "lane": self.lane,
                "y_m": self.y_m,
                "from_zone": self.from_zone,
                "to_zone": self.to_zone,
                "outcome": self.outcome,
            }
        )


@dataclass(frozen=True)
class GapPass:
    id: str
    t_in: float
    t_out: float
    data: CrossingInGapData


def _edge_time(ctx: Ctx, out_i: int, in_i: int, edge: float) -> float:
    """Time the centre crossed field x `edge` between samples out_i and in_i (linear)."""
    p = ctx.path
    dx = p.x[in_i] - p.x[out_i]
    f = 0.0 if dx == 0 else float(np.clip((edge - p.x[out_i]) / dx, 0.0, 1.0))
    return float(p.t[out_i] + f * (p.t[in_i] - p.t[out_i]))


def _band_sides(ctx: Ctx, barrier: Barrier, meas: I64) -> I64:
    """Per measured sample: 0 on `barrier`'s band, else -1 or +1 for the low-x or high-x
    side. Leaving back to the side it came from needs the centre BAND_HYST_M clear."""
    p = ctx.path
    x0, x1 = BARRIER_X[barrier]
    mid = (x0 + x1) / 2
    out = np.zeros(len(meas), np.int64)
    entry = 0
    for k, i in enumerate(meas.tolist()):
        x, s = float(p.x[i]), -1 if p.x[i] < mid else 1
        e = ctx.band_edges(barrier, float(p.y[i]))
        linked = k > 0 and p.t[i] - p.t[meas[k - 1]] <= GAP_S + 1e-6
        on_before = linked and out[k - 1] == 0
        inside = e is not None and e[0] <= x <= e[1]
        if not inside and on_before and e is not None and s == entry:
            inside = e[0] - BAND_HYST_M <= x <= e[1] + BAND_HYST_M
        if inside and not on_before:
            entry = int(out[k - 1]) if linked else 0
        out[k] = 0 if inside else s
    return out


def _edge_x(ctx: Ctx, barrier: Barrier, i: int, side: int) -> float:
    """Field x of `barrier`'s band edge on `side` at path sample i's y."""
    e = ctx.band_edges(barrier, float(ctx.path.y[i]))
    x0, x1 = BARRIER_X[barrier]
    lo, hi = e if e is not None else (x0 - BAND_MARGIN_M, x1 + BAND_MARGIN_M)
    return lo if side < 0 else hi


def _passes(ctx: Ctx) -> tuple[list[Pass], list[GapPass]]:
    """Bump and trench passes judged from measured samples only. A pass is a run of boxes
    on a barrier band (`Ctx.band_edges`); its sides are the nearest boxes before and after
    it, even across a gap. Two boxes on opposite sides with none on the band between are a
    crossing in a gap. Returns (passes, crossings in gaps), each in time order with ids."""
    p = ctx.path
    meas = np.flatnonzero(p.measured)
    seen: list[Pass] = []
    gap: list[tuple[float, float, CrossingInGapData]] = []
    for barrier in ALLIANCES:
        side = _band_sides(ctx, barrier, meas)
        j = 0
        while j < len(meas):
            if side[j] != 0:
                if j + 1 < len(meas) and side[j + 1] == -side[j]:
                    a, b = int(meas[j]), int(meas[j + 1])
                    before, after = ctx.seen(a), ctx.seen(b)
                    gap.append(
                        (
                            before.t,
                            after.t,
                            CrossingInGapData(
                                barrier=barrier,
                                from_zone=ctx.raw_zone(before.x),
                                to_zone=ctx.raw_zone(after.x),
                                before=before,
                                after=after,
                                unseen_s=_r(after.t - before.t),
                            ),
                        )
                    )
                j += 1
                continue
            s = j
            while (
                j + 1 < len(meas)
                and side[j + 1] == 0
                and p.t[meas[j + 1]] - p.t[meas[j]] <= GAP_S + 1e-6
            ):
                j += 1
            e = j
            j += 1
            first, last = int(meas[s]), int(meas[e])
            prev = int(meas[s - 1]) if s > 0 and side[s - 1] != 0 else None
            nxt = int(meas[e + 1]) if e + 1 < len(meas) and side[e + 1] != 0 else None
            entry_unseen = None if prev is None else float(p.t[first] - p.t[prev])
            exit_unseen = None if nxt is None else float(p.t[nxt] - p.t[last])
            entry_ok = entry_unseen is not None and entry_unseen <= GAP_S + 1e-6
            exit_ok = exit_unseen is not None and exit_unseen <= GAP_S + 1e-6
            t_in = (
                _edge_time(ctx, prev, first, _edge_x(ctx, barrier, prev, int(side[s - 1])))
                if entry_ok and prev is not None
                else float(p.t[first])
            )
            t_out = (
                _edge_time(ctx, last, nxt, _edge_x(ctx, barrier, nxt, int(side[e + 1])))
                if exit_ok and nxt is not None
                else float(p.t[last])
            )
            outcome = "unknown"
            if prev is not None and nxt is not None:
                if side[s - 1] != side[e + 1]:
                    outcome = "crossed"
                elif entry_ok and exit_ok:
                    outcome = "returned"
            on = _in_span(ctx, t_in, t_out)
            sp = p.speed[on]
            y_med = float(np.median(p.y[meas[s : e + 1]]))
            span = max(t_out - t_in, ctx.dt)
            measured_s = min(span, int(p.measured[on].sum()) * ctx.dt)
            whole = _r(span) if entry_ok and exit_ok else None
            data: dict[str, object] = {
                "entry_unseen_s": _ro(entry_unseen),
                "exit_unseen_s": _ro(exit_unseen),
                "time_on_barrier_s": whole,
                "observed_s": _r(span),
                "stopped_s": _r(int((sp < STOP_MPS).sum()) * ctx.dt),
                "min_speed_mps": _ro(float(np.nanmin(sp)) if np.isfinite(sp).any() else None),
                "mean_speed_mps": _ro(_mean(sp)),
                "measured_s": _r(measured_s),
                "excess_s": _r(max(0.0, span - NORMAL_CROSSING_S)),
                "own_active_s": _r(ctx.active_s(t_in, t_out)),
            }
            nearest = min(barrier_edge_m(float(p.x[i]), barrier) for i in range(first, last + 1))
            if nearest > PERCH_M:
                v = _heading(ctx, first, True)
                if v is None or v < HEADING_MPS:
                    continue
            seen.append(
                Pass(
                    id="",
                    t_in=t_in,
                    t_out=t_out,
                    first=first,
                    last=last,
                    outcome=outcome,
                    barrier=barrier,
                    lane=lane_at(y_med) or "bump",
                    y_m=_r(y_med),
                    from_zone=None if prev is None else ctx.raw_zone(float(p.x[prev])),
                    to_zone=None if nxt is None else ctx.raw_zone(float(p.x[nxt])),
                    nearest_edge_m=nearest,
                    data=data,
                )
            )
    seen.sort(key=lambda v: v.t_in)
    gap.sort(key=lambda v: v[0])
    passes = [dataclasses.replace(ps, id=f"cross-{n:02d}") for n, ps in enumerate(seen, 1)]
    in_gap = [GapPass(f"crossgap-{n:02d}", a, b, d) for n, (a, b, d) in enumerate(gap, 1)]
    return passes, in_gap


def _never_cleared(ctx: Ctx, i0: int, i1: int, barrier: Barrier) -> bool:
    """Every sample from i0 to i1 stayed within NEVER_CLEARED_M of `barrier`'s bumps, with
    every gap between bridged: the robot never got off the barrier."""
    p = ctx.path
    for i in range(i0, i1 + 1):
        on = barrier_at(float(p.x[i]), float(p.y[i]), NEVER_CLEARED_M)
        if on is None or on[0] != barrier:
            return False
        if i < i1 and not p.link[i] and not p.bridged[i]:
            return False
    return True


@dataclass(frozen=True)
class Attempt:
    """One time the target went onto a barrier and came back out the side it came from."""

    passes: tuple[Pass, ...]
    side: Zone
    t_in: float
    t_out: float
    far: bool


def _attempts(ctx: Ctx, passes: list[Pass], in_gap: list[GapPass]) -> list[Attempt]:
    """`returned` passes, and a `crossed` pass whose robot never got off the barrier before
    crossing back (one attempt of two passes). Never chained: each is its own attempt."""
    out: list[Attempt] = []
    used: set[str] = set()
    for k, ps in enumerate(passes):
        side = ps.from_zone
        if ps.id in used or side is None:
            continue
        if ps.outcome == "returned":
            out.append(Attempt((ps,), side, ps.t_in, ps.t_out, far=False))
            continue
        if ps.outcome != "crossed":
            continue
        nxt = next((q for q in passes[k + 1 :] if q.barrier == ps.barrier), None)
        if (
            nxt is not None
            and nxt.outcome == "crossed"
            and nxt.to_zone == side
            and not any(
                g.data.barrier == ps.barrier and ps.t_out <= g.t_in < nxt.t_in for g in in_gap
            )
            and _never_cleared(ctx, ps.last, nxt.first, ps.barrier)
        ):
            out.append(Attempt((ps, nxt), side, ps.t_in, nxt.t_out, far=True))
            used.add(nxt.id)
    return sorted(out, key=lambda a: a.t_in)


def _reach(lane: Lane, d: float) -> Reach:
    if lane == "trench" or d <= BAND_MARGIN_M:
        return "centre"
    return "footprint" if d <= PERCH_M else "corner"


def _cost_end(
    ctx: Ctx, a: Attempt, far: Zone, nxt: Attempt | None
) -> tuple[float, CostEnd, int | None]:
    """End of a toward-home attempt's cost window (see `CrossingTurnedBackData.cost_end`),
    and the path index where it was first tracked on the far side (None if never)."""
    p = ctx.path
    arrive = next(
        (
            k
            for k in np.flatnonzero(p.t > a.t_out + 1e-6).tolist()
            if ctx.raw_zone(float(p.x[k])) == far and ctx.band(float(p.x[k]), float(p.y[k])) is None
        ),
        None,
    )
    ends: list[tuple[float, CostEnd]] = [(ctx.t1, "match_end")]
    if arrive is not None:
        ends.append((float(p.t[arrive]), "reached_far_side"))
    if nxt is not None:
        ends.append((nxt.t_in, "tried_again"))
    inactive = next((e for _, e in ctx.active if e > a.t_in + 1e-6), None)
    if inactive is not None:
        ends.append((inactive, "hub_went_inactive"))
    stop = arrive if arrive is not None else len(p.t) - 1
    lost = next(
        (
            k
            for k in range(int(np.searchsorted(p.t, a.t_out)), stop)
            if p.t[k + 1] - p.t[k] > BRIDGE_MAX_S + 1e-6
        ),
        None,
    )
    if lost is not None:
        ends.append((max(a.t_out, float(p.t[lost])), "track_lost"))
    t, why = min(ends, key=lambda v: v[0])
    return t, why, arrive


def _crossing_facts(
    ctx: Ctx, contacts: Sequence[ContactFact]
) -> tuple[list[Fact], list[Fact], list[Fact]]:
    """(crossings, crossings in gaps, turned-back crossings), each in time order. Passes
    folded into a turned-back fact are not emitted as crossings."""
    p = ctx.path
    passes, in_gap = _passes(ctx)
    attempts = _attempts(ctx, passes, in_gap)
    folded = {ps.id for a in attempts for ps in a.passes}
    crossings: list[Fact] = [
        CrossingFact.model_validate(
            {**_common(ctx, ps.id, ps.t_in, ps.t_out), "data": ps.crossing()}
        )
        for ps in passes
        if ps.id not in folded
    ]
    gaps: list[Fact] = [
        CrossingInGapFact.model_validate({**_common(ctx, g.id, g.t_in, g.t_out), "data": g.data})
        for g in in_gap
    ]
    turned: list[Fact] = []
    for n, a in enumerate(attempts, 1):
        first = a.passes[0]
        barrier = first.barrier
        far: Zone = (
            ("own" if barrier == ctx.alliance else "opponent") if a.side == "neutral" else "neutral"
        )
        home = far == "own" or a.side == "opponent"
        nxt = next(
            (
                q
                for q in attempts
                if q.t_in > a.t_in and q.side == a.side and q.passes[0].barrier == barrier
            ),
            None,
        )
        arrive: int | None = None
        if home:
            t_end, why, arrive = _cost_end(ctx, a, far, nxt)
        else:
            t_end, why = a.t_out, "came_back"
        across = next(
            (
                q
                for q in sorted([*passes, *in_gap], key=lambda q: q.t_in)
                if a.t_out - 1e-6 <= q.t_in
                and q.t_out <= t_end + 1e-6
                and (
                    q.id not in folded and q.barrier == barrier and q.outcome == "crossed"
                    if isinstance(q, Pass)
                    else q.data.barrier == barrier
                )
                and (q.from_zone if isinstance(q, Pass) else q.data.from_zone) == a.side
            ),
            None,
        )
        after = _in_span(ctx, a.t_out, t_end)
        away = [
            barrier_edge_m(float(p.x[i]), barrier)
            for i in after.tolist()
            if ctx.raw_zone(float(p.x[i])) == a.side
        ]
        on = _in_span(ctx, a.t_in, a.t_out)
        fidx = _f_span(ctx, a.t_in, a.t_out)
        stopped, _ = _f_still_s(ctx, fidx, STOP_MPS)
        fsp = ctx.filled.speed[fidx]
        sp = p.speed[on]
        near = [d for i in on.tolist() for d, _ in _nearest(ctx, i)[:1]]
        d_min = min(ps.nearest_edge_m for ps in a.passes)
        reached = float(p.t[arrive]) if arrive is not None else None
        data = CrossingTurnedBackData(
            barrier=barrier,
            lane=first.lane,
            y_m=first.y_m,
            side=a.side,
            far_side=far,
            toward_home=home,
            reach=_reach(first.lane, d_min),
            nearest_edge_m=_r(d_min),
            reached_far_edge=a.far,
            time_on_barrier_s=_r(a.t_out - a.t_in),
            stopped_s=_r(stopped),
            unseen_s=_r(int((ctx.filled.src[fidx] < 0).sum()) * ctx.dt),
            min_speed_mps=_ro(float(np.nanmin(sp)) if np.isfinite(sp).any() else None),
            slowest_mps=_ro(float(np.nanmin(fsp)) if np.isfinite(fsp).any() else None),
            far_side_t=_ro(reached),
            far_side_after_gap=arrive is not None and arrive > 0 and not bool(p.link[arrive - 1]),
            seconds_to_far_side=None if reached is None else _r(reached - a.t_in),
            across_id=across.id if across else None,
            cost_s=_r(t_end - a.t_in),
            cost_end=why,
            own_active_s=_r(ctx.active_s(a.t_in, t_end)),
            furthest_from_barrier_m=_ro(max(away) if away else None),
            contact_ids=[c.id for c in contacts if c.t_start <= t_end and c.t_end >= a.t_in],
            nearest_tracked_m=_ro(min(near) if near else None),
        )
        turned.append(
            CrossingTurnedBackFact.model_validate(
                {**_common(ctx, f"turnback-{n:02d}", a.t_in, t_end), "data": data}
            )
        )
    return crossings, gaps, turned


# --- stops ----------------------------------------------------------------------------


def _stop_facts(ctx: Ctx) -> list[StopFact]:
    """Stops on the filled path, so a stop survives a bridged gap and says how much of it
    was unseen."""
    p, f = ctx.path, ctx.filled
    still = np.nan_to_num(f.speed, nan=np.inf) < STOP_MPS
    out: list[StopFact] = []
    for a, b in _runs(still, lambda k: _f_contiguous(ctx, k)):
        dur = float(f.t[b] - f.t[a]) + ctx.dt
        src = f.src[a : b + 1]
        seen = src[src >= 0]
        if dur < STOP_MIN_S - 1e-6 or not len(seen):
            continue
        bridged = [
            i
            for i in np.flatnonzero(p.bridged).tolist()
            if p.t[i] < f.t[b] - 1e-6 and p.t[i + 1] > f.t[a] + 1e-6
        ]
        moved = [math.hypot(p.x[i + 1] - p.x[i], p.y[i + 1] - p.y[i]) for i in bridged]
        start: StopStart = (
            "began_while_unseen"
            if f.src[a] < 0
            else "seen_slowing"
            if a > 0 and _f_contiguous(ctx, a - 1) and np.isfinite(f.speed[a - 1])
            else "match_start"
            if f.t[a] - ctx.t0 <= GAP_S
            else "first_seen_already_stopped"
        )
        end: StopEnd = (
            "ended_while_unseen"
            if f.src[b] < 0
            else "seen_moving_off"
            if b + 1 < len(f.t) and _f_contiguous(ctx, b) and np.isfinite(f.speed[b + 1])
            else "match_end"
            if ctx.t1 - f.t[b] <= GAP_S
            else "last_seen_still_stopped"
        )
        near = [d for i in seen.tolist() for d, _ in _nearest(ctx, i)[:1]]
        xm, ym = float(np.median(p.x[seen])), float(np.median(p.y[seen]))
        barrier = min(ALLIANCES, key=lambda b: barrier_edge_m(xm, b))
        on = ctx.band(xm, ym)
        unseen = int((src < 0).sum()) * ctx.dt
        data = StopData(
            duration_s=_r(dur),
            seen_s=_r(dur - unseen),
            unseen_s=_r(unseen),
            gaps_bridged=len(bridged),
            moved_across_gaps_m=_ro(max(moved) if moved else None),
            where=ctx.where(xm, ym),
            own_active_s=_r(ctx.active_s(float(f.t[a]), float(f.t[b]) + ctx.dt)),
            barrier=barrier,
            barrier_edge_m=_r(barrier_edge_m(xm, barrier)),
            lane=lane_at(ym),
            on_barrier=on[0] if on else None,
            start=start,
            end=end,
            nearest_tracked_m=_ro(min(near) if near else None),
            others_tracked_max=_others_range(ctx, float(f.t[a]), float(f.t[b]))[2],
        )
        n = len(out) + 1
        out.append(
            StopFact.model_validate(
                {**_common(ctx, f"stop-{n:02d}", float(f.t[a]), float(f.t[b])), "data": data}
            )
        )
    return out


# --- trips out while the own hub is active --------------------------------------------


def _own_active_spans(match: MatchInfo) -> list[tuple[float, float]]:
    """Teleop spans with the target's hub active (auto is the robot's code), merged."""
    out: list[tuple[float, float]] = []
    for ph in match.phases:
        if ph.name == "auto" or ph.active_hubs is None:
            continue
        if match.target_alliance not in ph.active_hubs:
            continue
        if out and abs(out[-1][1] - ph.t_start) < 1e-3:
            out[-1] = (out[-1][0], ph.t_end)
        else:
            out.append((ph.t_start, ph.t_end))
    return out


def _trip_facts(ctx: Ctx, stops: Sequence[StopFact]) -> list[Fact]:
    """Each time the target left its own zone during own-active time, until it was back
    or that time ended. A robot perched on its barrier is not home."""
    p = ctx.path
    home = [z == "own" and pl_ != "on_barrier" for z, pl_ in zip(p.zone, ctx.place, strict=True)]
    out: list[Fact] = []
    for sa, sb in ctx.active:
        i = 0
        while i + 1 < len(p.t):
            if not (home[i] and not home[i + 1] and sa <= p.t[i] < sb):
                i += 1
                continue
            back_i = next((k for k in range(i + 1, len(p.t)) if home[k]), None)
            back = back_i is not None and p.t[back_i] <= sb
            t_end = float(p.t[back_i]) if back and back_i is not None else sb
            t_left = float(p.t[i])
            nxt = back_i if back_i is not None else len(p.t)
            if t_end - t_left < TRIP_MIN_S - 1e-6:
                i = nxt
                continue
            outside = np.arange(i + 1, nxt)
            outside = outside[p.t[outside] <= t_end]
            if not len(outside):
                i = nxt
                continue
            depth = [ctx.toward_own(-float(p.x[k])) for k in outside.tolist()]
            far_k = int(np.argmax(depth))
            far = int(outside[far_k])
            back_k = (
                far_k + 1 + int(np.argmin(depth[far_k + 1 :])) if far_k + 1 < len(depth) else None
            )
            past = (
                float(p.x[far]) - CENTER_LINE_X_M
                if ctx.alliance == "blue"
                else CENTER_LINE_X_M - float(p.x[far])
            )
            out_t = p.t[outside]
            fidx = _f_span(ctx, float(out_t[0]), float(out_t[-1]))
            fx = ctx.filled.x
            fidx = fidx[[ctx.raw_zone(float(fx[k])) != "own" for k in fidx.tolist()]]
            _, slow = _f_still_s(ctx, fidx, SLOW_MPS)
            tracked = len(_in_span(ctx, t_left, t_end)) * ctx.dt
            at_end = None
            if not back:
                k = int(np.argmin(np.abs(p.t - sb)))
                at_end = ctx.seen(k) if abs(p.t[k] - sb) <= 1.0 else None
            data = TripData(
                left=ctx.seen(i),
                left_unseen_s=_r(0.0 if p.link[i] else p.t[i + 1] - p.t[i]),
                own_active_left_s=_r(sb - t_left),
                back=ctx.seen(back_i) if back and back_i is not None else None,
                at_active_end=at_end,
                duration_s=_r(t_end - t_left),
                unseen_s=_r(max(0.0, t_end - t_left + ctx.dt - tracked)),
                furthest=ctx.seen(far),
                past_centre_line_m=_r(past),
                nearest_home_after_furthest=(
                    None if back_k is None else ctx.seen(int(outside[back_k]))
                ),
                stop_ids=[
                    s.id
                    for s in stops
                    if s.t_start < t_end and s.t_end > t_left and s.data.where.zone != "own"
                ],
                longest_below_slow_s=_r(slow),
                path_length_m=_r(_path_length(ctx, _in_span(ctx, t_left, t_end)), 1),
            )
            n = len(out) + 1
            out.append(
                TripFact.model_validate(
                    {**_common(ctx, f"trip-{n:02d}", t_left, t_end), "data": data}
                )
            )
            i = nxt
    return out


# --- contacts -------------------------------------------------------------------------


def _window(ctx: Ctx, a: float, b: float) -> SpeedWindow:
    p = ctx.path
    idx = _in_span(ctx, a, b)
    prog = (
        math.hypot(p.x[idx[-1]] - p.x[idx[0]], p.y[idx[-1]] - p.y[idx[0]])
        if len(idx) >= 2
        else None
    )
    return SpeedWindow(
        duration_s=_r(b - a),
        tracked_s=_r(min(b - a + ctx.dt, len(idx) * ctx.dt)),
        mean_speed_mps=_ro(_mean(p.speed[idx])),
        net_progress_m=_ro(prog),
    )


def _position(ctx: Ctx, t: float) -> tuple[float, float, bool] | None:
    """Target (x, y, seen) at time t: the path sample within half a step, else the straight
    line across the gap around t (seen False); None outside the path."""
    p = ctx.path
    k = int(np.searchsorted(p.t, t))
    for j in (k - 1, k):
        if 0 <= j < len(p.t) and abs(p.t[j] - t) <= ctx.dt / 2 + 1e-6:
            return float(p.x[j]), float(p.y[j]), True
    if k == 0 or k >= len(p.t):
        return None
    f = (t - p.t[k - 1]) / (p.t[k] - p.t[k - 1])
    xy = (p.x[k - 1] + f * (p.x[k] - p.x[k - 1]), p.y[k - 1] + f * (p.y[k] - p.y[k - 1]))
    return float(xy[0]), float(xy[1]), bool(p.link[k - 1])


def _still(ctx: Ctx, a: float, b: float) -> tuple[float, float, float]:
    """(still, still unseen, longest still run) seconds in [a, b]: STILL_WINDOW_S windows in
    which the target moved slower than STILL_MPS, positions inside gaps taken on the straight
    line across them (such windows count as unseen)."""
    still = unseen = run = longest = 0.0
    t = a
    while t < b - 1e-6:
        e = min(b, t + STILL_WINDOW_S)
        u, v = _position(ctx, t), _position(ctx, e)
        slow = (
            u is not None
            and v is not None
            and e - t >= ctx.dt
            and math.hypot(v[0] - u[0], v[1] - u[1]) / (e - t) * ctx.rate < STILL_MPS
        )
        if slow and u is not None and v is not None:
            still += e - t
            unseen += 0.0 if u[2] and v[2] else e - t
            run += e - t
            longest = max(longest, run)
        else:
            run = 0.0
        t = e
    return still, unseen, longest


def _contact_end(ctx: Ctx, last: int, tid: int) -> ContactEnd:
    """How a contact ended, judged from the sample after its last close sample and the
    positions one second later."""
    p = ctx.path
    if ctx.t1 - p.t[last] <= CONTACT_BRIDGE_S:
        return "match_end"
    nxt = last + 1
    if nxt >= len(p.t) or p.t[nxt] - p.t[last] > CONTACT_BRIDGE_S:
        return "unknown_target_track_lost"
    if _other_near(ctx, tid, int(p.frame[nxt]), 0) is None:
        return "unknown_other_track_lost"
    later = _in_span(ctx, float(p.t[last]) + 1.0, float(p.t[last]) + 1.0 + ctx.dt)
    start = _other_near(ctx, tid, int(p.frame[last]), 0)
    end = _other_near(ctx, tid, int(p.frame[later[0]]), 0) if len(later) else None
    if start is None or end is None or not len(later):
        return "unknown_other_track_lost"
    j = int(later[0])
    me = math.hypot(p.x[j] - p.x[last], p.y[j] - p.y[last])
    them = math.hypot(end[0] - start[0], end[1] - start[1])
    return "target_moved_away" if me >= them else "other_moved_away"


def _net_speed(ctx: Ctx, tid: int, a: float, b: float) -> float | None:
    """Track `tid`'s straight-line distance over [a, b] per second, if tracked at both."""
    tol = round(MOTION_TOL_S * ctx.fps)
    pa, pb = _other_near(ctx, tid, ctx.frame(a), tol), _other_near(ctx, tid, ctx.frame(b), tol)
    if pa is None or pb is None or b - a < ctx.dt:
        return None
    return math.hypot(pb[0] - pa[0], pb[1] - pa[1]) / (b - a) * ctx.rate


def _closing(ctx: Ctx, tid: int, t: float) -> tuple[float | None, float | None]:
    """(target, other) speed toward the other's position at `t` over the CONTEXT_S before."""
    a = t - CONTEXT_S
    tol = round(MOTION_TOL_S * ctx.fps)
    t0, t1 = _target_near(ctx, a, MOTION_TOL_S), _target_near(ctx, t, MOTION_TOL_S)
    o0, o1 = _other_near(ctx, tid, ctx.frame(a), tol), _other_near(ctx, tid, ctx.frame(t), tol)
    if t0 is None or t1 is None or o0 is None or o1 is None:
        return None, None

    def d(u: tuple[float, float], v: tuple[float, float]) -> float:
        return math.hypot(u[0] - v[0], u[1] - v[1])

    k = ctx.rate / CONTEXT_S
    return (d(t0, o1) - d(t1, o1)) * k, (d(o0, t1) - d(o1, t1)) * k


@dataclass
class Piece:
    """Part of a contact: close tracked samples, or a bridged gap with robots beside the
    target's spot. `hits` is track -> [(frame, distance, in_gap)]."""

    a: float
    b: float
    hits: dict[int, list[tuple[int, float, bool]]]


def _contact_facts(ctx: Ctx) -> list[ContactFact]:
    """Contacts from close tracked samples, joined across target gaps three ways: the same
    robot close on both sides of a gap, a contact's robots near the target's spot inside
    the gap next to it, and robots beside a still target's spot in a bridged gap."""
    p = ctx.path
    near = [_nearest(ctx, i) for i in range(len(p.t))]
    close = np.array([bool(n) and n[0][0] <= CONTACT_M for n in near], np.bool_)
    gaps = [j for j in np.flatnonzero(~p.link).tolist() if p.t[j + 1] - p.t[j] <= CONTACT_GAP_MAX_S]

    def key(tid: int, hits: list[tuple[int, float, bool]]) -> tuple[str, int]:
        """A robot across track changes: its team label, else its track."""
        team = _label(ctx, tid, [f for f, _, _ in hits])
        return ("team", team) if team is not None else ("track", tid)

    def keys(q: Piece, seen_only: bool = False) -> set[tuple[str, int]]:
        return {key(t, h) for t, h in q.hits.items() if not seen_only or any(not g for *_, g in h)}

    def add(q: Piece, found: dict[int, list[tuple[int, float, bool]]], only: set[tuple[str, int]]):
        for tid, h in found.items():
            if key(tid, h) in only:
                q.hits.setdefault(tid, []).extend(h)
                q.a = min(q.a, min(f for f, _, _ in h) / ctx.fps)
                q.b = max(q.b, max(f for f, _, _ in h) / ctx.fps)

    pieces: list[Piece] = []
    for a, b in _runs(close, lambda i: bool(p.t[i + 1] - p.t[i] <= CONTACT_BRIDGE_S + 1e-6)):
        seen_hits: dict[int, list[tuple[int, float, bool]]] = {}
        for i in range(a, b + 1):
            for d, tid in near[i]:
                if d <= CONTACT_M:
                    seen_hits.setdefault(tid, []).append((int(p.frame[i]), d, False))
        q = Piece(float(p.t[a]), float(p.t[b]), seen_hits)
        mine = keys(q)
        for j in gaps:
            if 0 <= p.t[a] - p.t[j + 1] <= CONTACT_EDGE_S or 0 <= p.t[j] - p.t[b] <= CONTACT_EDGE_S:
                add(q, _near_in_gap(ctx, j, j + 1), mine)
        pieces.append(q)
    for j in np.flatnonzero(p.bridged).tolist():
        found = _near_in_gap(ctx, j, j + 1)
        if any(len(h) * ctx.dt >= NEAR_MIN_S - 1e-6 for h in found.values()):
            pieces.append(Piece(float(p.t[j]), float(p.t[j + 1]), found))
    pieces.sort(key=lambda q: q.a)

    def across_gap(cur: Piece, nxt: Piece) -> bool:
        """The same robot is close on both sides of a target gap of at most
        CONTACT_GAP_MAX_S, seen within CONTACT_EDGE_S of each end."""
        return bool(keys(cur, True) & keys(nxt, True)) and any(
            p.t[j] - cur.b <= CONTACT_EDGE_S and nxt.a - p.t[j + 1] <= CONTACT_EDGE_S
            for j in gaps
            if cur.b - 1e-6 <= p.t[j] and p.t[j + 1] <= nxt.a + 1e-6
        )

    def closest(q: Piece) -> int:
        seen = {t: h for t, h in q.hits.items() if any(not g for *_, g in h)} or q.hits
        return min(seen, key=lambda tid: min(d for _, d, _ in seen[tid]))

    def stayed(cur: Piece, nxt: Piece) -> bool:
        """Both pieces' closest seen robot is the same track, tracked within CONTACT_STAY_M
        of the target at every target sample between them."""
        tid = closest(cur)
        if tid != closest(nxt) or nxt.a - cur.b > CONTACT_GAP_MAX_S:
            return False
        for i in _in_span(ctx, cur.b, nxt.a).tolist():
            o = _other_near(ctx, tid, int(p.frame[i]), 0)
            if o is None or math.hypot(o[0] - p.x[i], o[1] - p.y[i]) > CONTACT_STAY_M:
                return False
        return True

    merged: list[Piece] = []
    for q in pieces:
        cur = merged[-1] if merged else None
        if cur is not None and (
            q.a - cur.b <= CONTACT_BRIDGE_S + 1e-6 or across_gap(cur, q) or stayed(cur, q)
        ):
            cur.b = max(cur.b, q.b)
            for tid, h in q.hits.items():
                cur.hits.setdefault(tid, []).extend(h)
        else:
            merged.append(Piece(q.a, q.b, {k: list(v) for k, v in q.hits.items()}))

    out: list[ContactFact] = []
    for n, c in enumerate(merged, 1):
        ta, tb = c.a, c.b
        idx = _in_span(ctx, ta, tb)
        robots: list[OtherRobot] = []
        for tid, hits in sorted(c.hits.items(), key=lambda kv: min(d for _, d, _ in kv[1])):
            frames = [f for f, _, _ in hits]
            team = _label(ctx, tid, frames)
            alliance, rel = _team_of(ctx, team)
            first_t, last_t = min(frames) / ctx.fps, max(frames) / ctx.fps
            tc, oc = _closing(ctx, tid, first_t)
            robots.append(
                OtherRobot(
                    track_id=tid,
                    team=team,
                    alliance=alliance,
                    relation=rel,
                    bumper_seen=ctx.others.color.get(tid, "unknown"),
                    min_distance_m=_r(min(d for _, d, _ in hits)),
                    close_s=_r(sum(not g for _, _, g in hits) * ctx.dt),
                    near_in_gap_s=_r(sum(g for _, _, g in hits) * ctx.dt),
                    first_close_t=_r(first_t),
                    last_close_t=_r(last_t),
                    target_closing_mps=_ro(tc),
                    other_closing_mps=_ro(oc),
                    other_speed_before_mps=_ro(_net_speed(ctx, tid, first_t - CONTEXT_S, first_t)),
                    other_speed_during_mps=_ro(_net_speed(ctx, tid, first_t, last_t)),
                )
            )
        still, still_unseen, longest = _still(ctx, ta, tb + ctx.dt)
        _, lo, hi = _others_range(ctx, ta, tb)
        last = int(idx[-1])
        last_near = min(
            ((d, tid) for d, tid in near[last] if d <= CONTACT_M),
            default=(0.0, max(robots, key=lambda r: r.last_close_t).track_id),
        )
        span = tb - ta + ctx.dt
        seen = min(span, len(idx) * ctx.dt)
        data = ContactData(
            others=robots,
            duration_s=_r(span),
            seen_s=_r(seen),
            unseen_s=_r(span - seen),
            min_distance_m=_r(min(r.min_distance_m for r in robots)),
            target_zone=ctx.place[int(idx[0])],
            where=ctx.where(float(np.median(p.x[idx])), float(np.median(p.y[idx]))),
            before=_window(ctx, ta - CONTEXT_S, ta - ctx.dt),
            during=_window(ctx, ta, tb),
            after=_window(ctx, tb + ctx.dt, tb + CONTEXT_S),
            still_s=_r(still),
            still_unseen_s=_r(still_unseen),
            longest_still_s=_r(longest),
            end=_contact_end(ctx, last, last_near[1]),
            others_tracked_min=lo,
            others_tracked_max=hi,
        )
        out.append(
            ContactFact.model_validate({**_common(ctx, f"contact-{n:02d}", ta, tb), "data": data})
        )
    return out


# --- hub turning on -------------------------------------------------------------------


def _hub_on_facts(ctx: Ctx) -> list[Fact]:
    """One fact per real turn-on of the own hub: a shift or endgame with it active after a
    phase with it inactive (or unknown)."""
    p = ctx.path
    out: list[Fact] = []
    phases = ctx.match.phases
    for k, ph in enumerate(phases):
        if ph.name not in SHIFT_PHASES or ph.active_hubs is None or k == 0:
            continue
        if ctx.alliance not in ph.active_hubs:
            continue
        prev = phases[k - 1]
        before = None if prev.active_hubs is None else ctx.alliance in prev.active_hubs
        if before:
            continue
        t = ph.t_start
        j = int(np.searchsorted(p.t, t))
        cand = [i for i in (j - 1, j) if 0 <= i < len(p.t) and abs(p.t[i] - t) <= ctx.dt * 0.6]
        at = cand[0] if cand else None
        last_i = j - 1 if j - 1 >= 0 else None
        next_i = j if j < len(p.t) else None
        in_own = None if at is None else p.zone[at] == "own"
        since = since_gap = None
        if at is not None and in_own:
            s = at
            while s > 0 and p.link[s - 1] and p.zone[s - 1] == "own":
                s -= 1
            since = t - float(p.t[s])
            since_gap = s == 0 or not p.link[s - 1]
        reached = None
        reached_gap = False
        if not in_own:
            after = [i for i in range(j, len(p.t)) if p.t[i] < ph.t_end and p.zone[i] == "own"]
            if after:
                reached = float(p.t[after[0]]) - t
                start = last_i if last_i is not None else after[0]
                reached_gap = bool((~p.link[start : after[0]]).any()) or at is None
        from_wall = (
            None
            if at is None
            else (p.x[at] if ctx.alliance == "blue" else FIELD_LENGTH_M - p.x[at])
        )
        data = HubOnData(
            phase=ph.name,
            previous_phase=prev.name,
            own_hub_active_before=before,
            at=ctx.seen(at) if at is not None else None,
            last_seen=ctx.seen(last_i) if at is None and last_i is not None else None,
            next_seen=ctx.seen(next_i) if at is None and next_i is not None else None,
            into_own_zone_m=_ro(None if from_wall is None else ALLIANCE_ZONE_DEPTH_M - from_wall),
            speed_mps=_ro(None if at is None else float(p.speed[at])),
            in_own_zone=in_own,
            own_zone_since_s=_ro(since),
            since_after_gap=bool(since_gap),
            reached_own_zone_after_s=_ro(reached),
            reached_after_gap=reached_gap,
        )
        out.append(
            HubOnFact.model_validate({**_common(ctx, f"hubon-{ph.name}", t, t), "data": data})
        )
    return out


# --- auto -----------------------------------------------------------------------------


def _simplify(xy: F64, tol: float) -> list[int]:
    """Douglas-Peucker: indices of the points that keep the polyline within `tol`."""
    if len(xy) <= 2:
        return list(range(len(xy)))
    a, b = xy[0], xy[-1]
    ab = b - a
    norm = float(np.hypot(*ab))
    rel = xy - a
    d = (
        np.abs(ab[0] * rel[:, 1] - ab[1] * rel[:, 0]) / norm
        if norm > 1e-9
        else np.hypot(rel[:, 0], rel[:, 1])
    )
    k = int(np.argmax(d))
    if d[k] <= tol:
        return [0, len(xy) - 1]
    left = _simplify(xy[: k + 1], tol)
    right = _simplify(xy[k:], tol)
    return left + [k + i for i in right[1:]]


def _auto_fact(
    ctx: Ctx, gaps: list[GapSpan], gap_facts: Sequence[Fact], crossings: Sequence[Fact]
) -> Fact:
    p = ctx.path
    m = ctx.match
    idx = _in_span(ctx, m.auto_start_t, m.auto_end_t)
    waypoints: list[Waypoint] = []
    runs = _runs(np.isin(np.arange(len(p.t)), idx), lambda i: bool(p.link[i]))
    for a, b in runs:
        keep = _simplify(np.stack([p.x[a : b + 1], p.y[a : b + 1]], 1), ROUTE_TOLERANCE_M)
        for j, k in enumerate(keep):
            i = a + k
            after_gap = j == 0 and (p.t[i] - m.auto_start_t > GAP_S if a == idx[0] else True)
            s = ctx.seen(i)
            waypoints.append(Waypoint(t=s.t, x=s.x, y=s.y, zone=s.zone, after_gap=after_gap))
    start = ctx.seen(int(idx[0])) if len(idx) else None
    end = ctx.seen(int(idx[-1])) if len(idx) else None
    past = None
    if len(idx):
        xs = p.x[idx]
        past = float(
            np.max(CENTER_LINE_X_M - xs if ctx.alliance == "red" else xs - CENTER_LINE_X_M)
        )
    data = AutoRouteData(
        first_tracked=start,
        first_tracked_after_s=_r((start.t if start else m.auto_end_t) - m.auto_start_t),
        waypoints=waypoints,
        last_tracked=end,
        untracked_after_last_s=_r(m.auto_end_t - (end.t if end else m.auto_start_t)),
        gap_ids=[f.id for f in gap_facts if f.t_start < m.auto_end_t and f.t_end > m.auto_start_t],
        crossings=[f.id for f in crossings if m.auto_start_t <= f.t_start <= m.auto_end_t],
        max_past_centre_line_m=_ro(past),
        coverage=_coverage(ctx, m.auto_start_t, m.auto_end_t, gaps),
    )
    return AutoRouteFact.model_validate(
        {**_common(ctx, "auto-route", m.auto_start_t, m.auto_end_t), "data": data}
    )


def _notes(ctx: Ctx, coverage: Coverage, sources: set[str]) -> list[str]:
    """What the grader must keep in mind, with this match's numbers filled in."""
    shares = ", ".join(
        f"{s.team} ({'partner' if s.alliance == ctx.alliance else 'opponent'}) "
        f"{ctx.team_share[s.team]:.0%}"
        for s in ctx.match.roster
        if s.team != ctx.match.target_team
    )
    return [
        "Untracked time is unknown, not idle. Each phase fact gives the share of the "
        "phase with a tracked position; gap facts give each hole and why it opened. The "
        "same zone before and after a gap does not mean the robot stayed there, unless the "
        "gap's `inference` says so; inferences are probable, not measured.",
        f"A gap of at most {BRIDGE_MAX_S:g} s with the robot at most {BRIDGE_M:g} m from where "
        "it was is bridged: stops, barrier stays and contacts run across it, and each such "
        "fact gives its `unseen_s`. A stop with `unseen_s` above 0 was partly inferred.",
        "Other robots: only robots the tracker followed appear. On average "
        f"{coverage.others_tracked_mean:.1f} of the 5 other robots were tracked "
        f"(min {coverage.others_tracked_min}, max {coverage.others_tracked_max}). Share of "
        f"the match each was tracked and labelled: {shares}. A robot absent from these "
        "facts may have been right beside the target. Contact facts list only tracked "
        "robots; defense by an untracked robot leaves no contact fact, and may show only as "
        "a stop, a turned-back crossing or a gap. Check frames before scoring defense.",
        "A labelled robot's alliance and relation come from the roster and are right even "
        "when `bumper_seen` disagrees; `bumper_seen` is the detector's colour and can be "
        "wrong. Unlabelled robots may be partners or opponents. Two robots' centres closer "
        f"than about {SAME_ROBOT_M:g} m are one robot seen twice, not contact. Contact facts "
        "say who was moving toward whom and whose speed dropped, never who was defending.",
        "The barrier band, used for crossings, barrier stays and stops: the centre within "
        f"{BAND_MARGIN_M:g} m of a trench's or the opponent barrier's depth; on the neutral "
        f"face of its own bumps (the climb home) within {CORNER_M:.2f} m, half the robot's "
        "diagonal, where a turned robot's corner can be on the ramp. A centre inside its own "
        "zone is never on a barrier, whatever part of the robot reaches over it. Tracks carry "
        "no heading: `reach` says how close the centre got, and `corner` means only a corner "
        "could have touched.",
        "`crossing` facts were seen on the band; `crossing_in_gap` facts only know the robot "
        "was on one side and later on the other; each `crossing_turned_back` is one attempt "
        "that came back out the side it came from (attempts are never chained), with "
        "`cost_s` to where it was going, bounded as `cost_end` says, and `own_active_s` of it "
        "in own-active time. `excess_s` is time beyond a normal pass. Trenches are taken as "
        "the strip from each outer bump to the guardrail at the bumps' depth.",
        f"Positions are robot centres, about 0.2 m accurate. Zone lines use the centre with "
        f"{ZONE_HYST_M} m hysteresis, or {ZONE_DWELL_S:g} s on the other side; `where` "
        "gives distances to the own hub, own wall and guardrail: describe places from those.",
        f"Speeds are per real second (video speeds x clock rate {ctx.rate:.4f}); times and "
        "durations are video seconds.",
        f"Target identity labels come from identify sources: {', '.join(sorted(sources))}.",
    ]


def build(run: RunDir) -> tuple[Facts, Ctx]:
    """Compute every fact for the target team; `facts` saves the result."""
    ctx = _load(run)
    gaps = _gap_spans(ctx)
    gap_facts = _gap_facts(ctx, gaps)
    contacts = _contact_facts(ctx)
    stops = _stop_facts(ctx)
    crossings, in_gap, turned = _crossing_facts(ctx, contacts)
    facts: list[Fact] = [
        *_phase_facts(ctx, gaps),
        *gap_facts,
        *_visit_facts(ctx, _visits(ctx)),
        *crossings,
        *in_gap,
        *turned,
        *stops,
        *_trip_facts(ctx, stops),
        *contacts,
        *_hub_on_facts(ctx),
        _auto_fact(ctx, gaps, gap_facts, [*crossings, *in_gap, *turned]),
    ]
    facts.sort(key=lambda f: (f.t_start, f.kind != "phase", f.id))
    coverage = _coverage(ctx, ctx.t0, ctx.t1, gaps)
    sources = {
        s.source
        for s in Identity.load(run.identity_json).segments
        if s.team == ctx.match.target_team
    }
    out = Facts(
        team=ctx.match.target_team,
        alliance=ctx.alliance,
        clock_rate=ctx.rate,
        thresholds=THRESHOLDS,
        notes=_notes(ctx, coverage, sources),
        coverage=coverage,
        team_tracked_share=ctx.team_share,
        facts=facts,
    )
    return out, ctx


# --- check image ----------------------------------------------------------------------

W = 2000
LEFT = 130
RIGHT = 20
PLACE_BGR: dict[Place, tuple[int, int, int]] = {
    "own": (60, 60, 220),
    "neutral": (150, 150, 150),
    "opponent": (220, 130, 40),
    "on_barrier": (0, 220, 255),
}
ALLIANCE_BGR: dict[Alliance, tuple[int, int, int]] = {"red": (60, 60, 220), "blue": (220, 130, 40)}
KIND_ROWS: tuple[str, ...] = (
    "gap",
    "zone_visit",
    "crossing",
    "crossing_in_gap",
    "crossing_turned_back",
    "stop",
    "left_own_zone_while_active",
    "contact",
    "hub_on",
)
KIND_LABEL: dict[str, str] = {
    "crossing_turned_back": "turned_back",
    "left_own_zone_while_active": "trip_out",
}


def _unseen_part(f: Fact) -> float:
    """Unseen seconds of a fact that has them, for drawing it hollow."""
    if isinstance(f, StopFact | ContactFact | CrossingTurnedBackFact):
        return f.data.unseen_s
    return 0.0


def _write_check(run: RunDir, facts: Facts, ctx: Ctx) -> None:
    """Timeline: phases and active hubs, place band (gaps empty), speed, one row per fact
    kind with each fact's span and id. Facts with unseen time are drawn hollow; gaps with
    an inference are olive."""
    t0, t1 = ctx.t0 - 2, ctx.t1 + 2
    plot_w = W - LEFT - RIGHT

    def tx(t: float) -> int:
        return round(LEFT + (t - t0) / (t1 - t0) * plot_w)

    row_h = 46
    top = 40
    h = top + 60 + 40 + 200 + len(KIND_ROWS) * row_h + 60
    img = np.full((h, W, 3), 24, np.uint8)
    white = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        img,
        f"facts for {facts.team} ({facts.alliance}): {len(facts.facts)} facts, "
        f"tracked {facts.coverage.tracked_share:.0%} of the match; hollow = partly unseen",
        (LEFT, 26),
        font,
        0.6,
        white,
        1,
    )
    for tt in range(math.ceil(t0 / 10) * 10, int(t1) + 1, 10):
        cv2.line(img, (tx(tt), top), (tx(tt), h - 30), (55, 55, 55), 1)
        cv2.putText(img, str(tt), (tx(tt) - 10, h - 12), font, 0.4, (200, 200, 200), 1)

    y = top
    cv2.putText(img, "phase / hub", (8, y + 22), font, 0.45, white, 1)
    cv2.putText(img, "hub rows: red, blue active", (8, y + 44), font, 0.32, white, 1)
    for ph in ctx.match.phases:
        a, b = tx(ph.t_start), tx(ph.t_end)
        cv2.rectangle(img, (a, y), (b, y + 28), (70, 70, 70), 1)
        cv2.putText(img, ph.name, (a + 4, y + 18), font, 0.42, white, 1)
        hubs = ph.active_hubs or []
        for k, al in enumerate(ALLIANCES):
            if al in hubs:
                cv2.rectangle(
                    img, (a + 1, y + 30 + k * 12), (b - 1, y + 40 + k * 12), ALLIANCE_BGR[al], -1
                )
    y += 60
    cv2.putText(img, "place", (8, y + 22), font, 0.45, white, 1)
    p = ctx.path
    for i in range(len(p.t)):
        cv2.rectangle(
            img,
            (tx(float(p.t[i])), y),
            (tx(float(p.t[i]) + ctx.dt), y + 30),
            PLACE_BGR[ctx.place[i]],
            -1,
        )
    cv2.putText(
        img,
        "red=own grey=neutral blue=opponent yellow=on barrier, empty=untracked",
        (LEFT, y + 46),
        font,
        0.4,
        (200, 200, 200),
        1,
    )
    y += 60
    sh = 180
    cv2.putText(img, "speed m/s", (8, y + 20), font, 0.45, white, 1)
    for v in range(0, 5):
        yy = y + sh - round(v / 4 * sh)
        cv2.line(img, (LEFT, yy), (W - RIGHT, yy), (60, 60, 60) if v else (120, 120, 120), 1)
        cv2.putText(img, str(v), (LEFT - 20, yy + 4), font, 0.4, (200, 200, 200), 1)
    yy = y + sh - round(STOP_MPS / 4 * sh)
    cv2.line(img, (LEFT, yy), (W - RIGHT, yy), (0, 140, 255), 1)
    ok = np.isfinite(p.speed)
    for a, b in _runs(ok, lambda i: bool(p.frame[i + 1] - p.frame[i] == ctx.step)):
        pts = np.array(
            [
                [tx(float(p.t[i])), y + sh - round(min(float(p.speed[i]), 4.0) / 4 * sh)]
                for i in range(a, b + 1)
            ],
            np.int32,
        )
        cv2.polylines(img, [pts], False, white, 1, cv2.LINE_AA)
    y += sh + 20
    colours = {
        "gap": (90, 90, 90),
        "zone_visit": (150, 150, 150),
        "crossing": (0, 200, 255),
        "crossing_in_gap": (0, 120, 160),
        "crossing_turned_back": (0, 90, 255),
        "stop": (200, 200, 0),
        "left_own_zone_while_active": (255, 0, 200),
        "contact": (60, 60, 255),
        "hub_on": (0, 255, 0),
    }
    for r, kind in enumerate(KIND_ROWS):
        ry = y + r * row_h
        cv2.putText(img, KIND_LABEL.get(kind, kind), (8, ry + 18), font, 0.45, white, 1)
        cv2.line(img, (LEFT, ry + row_h - 2), (W - RIGHT, ry + row_h - 2), (45, 45, 45), 1)
        mine = [g for g in facts.facts if g.kind == kind]
        for n, g in enumerate(mine):
            a, b = tx(g.t_start), max(tx(g.t_end), tx(g.t_start) + 2)
            col = colours[kind]
            if isinstance(g, ZoneVisitFact):
                col = PLACE_BGR[g.data.zone]
            if isinstance(g, GapFact) and g.data.inference is not None:
                col = (160, 160, 60)
            cv2.rectangle(img, (a, ry + 4), (b, ry + 16), col, 1 if _unseen_part(g) > 0 else -1)
            label = g.id.split("-", 1)[1]
            cv2.putText(img, label, (a, ry + 30 + (n % 2) * 12), font, 0.35, col, 1)
    cv2.imwrite(str(run.path / CHECK_IMAGE), img, [cv2.IMWRITE_JPEG_QUALITY, 92])


def facts(run: RunDir) -> None:
    start = time.perf_counter()
    out, ctx = build(run)
    out.save(run.facts_json)
    _write_check(run, out, ctx)
    kinds = Counter(f.kind for f in out.facts)
    print(
        f"facts: {len(out.facts)} facts: " + ", ".join(f"{k} {n}" for k, n in sorted(kinds.items()))
    )
    for f in out.facts:
        if isinstance(f, PhaseFact):
            c = f.data.coverage
            print(
                f"  {f.id:<18} tracked {c.tracked_share:5.1%} "
                f"(measured {c.measured_share:5.1%}), own {c.own_s:5.1f} s, "
                f"neutral {c.neutral_s:5.1f} s, untracked {c.untracked_s:5.1f} s, "
                f"others tracked {c.others_tracked_mean:.1f}"
            )
    print(f"  wall {time.perf_counter() - start:.2f} s")


STAGE = Stage(
    "facts",
    inputs=lambda r: (r.tracks, r.identity_json, r.match_json),
    outputs=lambda r: (r.facts_json,),
    fn=facts,
)
