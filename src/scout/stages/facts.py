"""facts: reads tracks.parquet, identity.json, match.json.

Writes facts.json (see `scout.schemas.facts`): timed, measured events for the target
team's robot, and `facts_check.jpg`, a match timeline with the phases, the target's zone,
its speed and every fact, for a person to check against the video.

Steps:

1. The target's path is every track sample inside an identity segment labelled with the
   target team, in time order, within the match (auto start to teleop end). Frames where
   two tracks both claim the target are dropped and reported as a gap, never averaged.
2. Samples further apart than GAP_S are not linked: nothing is measured across them, and
   the hole is a `gap` fact. Speed is the straight-line distance over SPEED_WINDOW_S of
   consecutive samples, so it is never taken across a missing sample.
3. Zones use the robot centre with ZONE_HYST_M of hysteresis at each zone line. Stays are
   `zone_visit` facts. Bump and trench passes are judged from measured (boxed) samples
   only: seen on the barrier is a `crossing`, last seen one side and next seen the other
   is a `crossing_in_gap`.
4. Other robots are every other track in the same frame. They are only the robots the
   tracker followed, so facts give the closest tracked robot and the count tracked, and
   never claim a robot was absent.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np
import polars as pl
from numpy.typing import NDArray

from scout.games.rebuilt2026 import (
    ALLIANCE_ZONE_DEPTH_M,
    BARRIER_X,
    CENTER_LINE_X_M,
    FIELD_LENGTH_M,
    TRENCHES,
    PhaseName,
    Zone,
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
    Coverage,
    CrossingData,
    CrossingFact,
    CrossingInGapData,
    CrossingInGapFact,
    CrossingOutcome,
    Fact,
    Facts,
    GapCause,
    GapData,
    GapFact,
    HubOnData,
    HubOnFact,
    OtherRobot,
    Period,
    PhaseData,
    PhaseFact,
    Seen,
    SpeedWindow,
    StopData,
    StopFact,
    Waypoint,
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
BAND_MARGIN_M = 0.2
"""The barrier band is the bumps' depth plus this on each side (position accuracy), so a
robot riding a ramp's edge counts as on it while one beside the hub face does not."""
CONTACT_M = 1.6
"""Centre distance for a contact: touching robots are 0.95 m (side by side) to 1.34 m
(corner to corner) apart, plus 0.2 m accuracy on each."""
CONTACT_BRIDGE_S = 0.5
"""Contact samples closer in time than this are one contact (bumpers bounce apart)."""
CONTEXT_S = 2.0
"""Length of the before and after windows around a contact."""
ROUTE_TOLERANCE_M = 0.75
"""The auto route keeps a point where the path bends more than this from a straight line."""

THRESHOLDS: dict[str, float] = {
    "GAP_S": GAP_S,
    "SPEED_WINDOW_S": SPEED_WINDOW_S,
    "STOP_MPS": STOP_MPS,
    "STOP_MIN_S": STOP_MIN_S,
    "ZONE_HYST_M": ZONE_HYST_M,
    "BAND_MARGIN_M": BAND_MARGIN_M,
    "CONTACT_M": CONTACT_M,
    "CONTACT_BRIDGE_S": CONTACT_BRIDGE_S,
    "CONTEXT_S": CONTEXT_S,
    "ROUTE_TOLERANCE_M": ROUTE_TOLERANCE_M,
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
    GAP_S apart; `speed` is NaN where the window would cross a missing sample."""

    frame: I64
    t: F64
    x: F64
    y: F64
    measured: B
    track: I64
    link: B
    speed: F64
    zone: list[Zone]


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
    t0: float
    t1: float
    path: Path
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

    def frame(self, t: float) -> int:
        return round(t * self.fps)

    def seen(self, i: int) -> Seen:
        p = self.path
        return Seen(t=_r(p.t[i]), x=_r(p.x[i]), y=_r(p.y[i]), zone=p.zone[i])

    def raw_zone(self, x: float) -> Zone:
        return zone_of(x, self.alliance)


def _load(run: RunDir) -> Ctx:
    """Read the inputs and build the target's path and the other tracks."""
    match = MatchInfo.load(run.match_json)
    ident = Identity.load(run.identity_json)
    tracks = read_tracks(run.tracks)
    t0, t1 = match.auto_start_t, match.teleop_end_t
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
    speed = _speed(frame, t, x, y, step, fps)
    path = Path(
        frame=frame,
        t=t,
        x=x,
        y=y,
        measured=(mine["state"] == "measured").to_numpy(),
        track=mine["track_id"].to_numpy().astype(np.int64),
        link=link,
        speed=speed,
        zone=_zones(x, link, match.target_alliance),
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
    return Ctx(
        match=match,
        alliance=match.target_alliance,
        fps=fps,
        step=step,
        dt=step / fps,
        t0=t0,
        t1=t1,
        path=path,
        others=Others(by_frame=by_frame, by_track=by_track, team=team, color=color),
        grid=grid.astype(np.int64),
        others_count=counts,
        conflicts=conflicts,
        track_frames=track_frames,
        team_share=team_share,
    )


def _speed(frame: I64, t: F64, x: F64, y: F64, step: int, fps: float) -> F64:
    """Speed at each sample over SPEED_WINDOW_S centred on it; NaN when any sample in the
    window is missing."""
    h = max(1, round(SPEED_WINDOW_S * fps / step / 2))
    out = np.full(len(t), np.nan)
    for i in range(h, len(t) - h):
        a, b = i - h, i + h
        if frame[b] - frame[a] == 2 * h * step:
            out[i] = math.hypot(x[b] - x[a], y[b] - y[a]) / (t[b] - t[a])
    return out


def _zones(x: F64, link: B, alliance: Alliance) -> list[Zone]:
    """Zone per sample with ZONE_HYST_M hysteresis, restarting after every gap: the zone
    changes only when the centre is clear of the line on both sides of the margin."""
    out: list[Zone] = []
    cur: Zone = "neutral"
    for i, xi in enumerate(x.tolist()):
        lo, hi = zone_of(xi - ZONE_HYST_M, alliance), zone_of(xi + ZONE_HYST_M, alliance)
        if i == 0 or not link[i - 1]:
            cur = zone_of(xi, alliance)
        elif lo == hi:
            cur = lo
        out.append(cur)
    return out


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


def _barrier_of(x: float) -> Barrier | None:
    for b, (x0, x1) in BARRIER_X.items():
        if x0 - BAND_MARGIN_M <= x <= x1 + BAND_MARGIN_M:
            return b
    return None


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


# --- facts ----------------------------------------------------------------------------


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
        data = GapData(
            duration_s=_r(g.b - g.a),
            before=before,
            after=after,
            straight_line_m=_ro(line),
            cause=cause,
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


@dataclass(frozen=True)
class Visit:
    zone: Zone
    first: int
    last: int


def _visits(ctx: Ctx) -> list[Visit]:
    """Runs of one zone, split at gaps."""
    p = ctx.path
    out: list[Visit] = []
    for i in range(len(p.t)):
        if out and out[-1].zone == p.zone[i] and p.link[i - 1]:
            out[-1] = Visit(p.zone[i], out[-1].first, i)
        else:
            out.append(Visit(p.zone[i], i, i))
    return out


def _visit_facts(ctx: Ctx, visits: list[Visit]) -> list[Fact]:
    p = ctx.path
    out: list[Fact] = []
    for n, v in enumerate(visits, 1):
        k = n - 1
        prev = visits[k - 1] if k else None
        nxt = visits[k + 1] if k + 1 < len(visits) else None
        came = (
            prev.zone
            if prev and p.link[v.first - 1]
            else "match_start"
            if v.first == 0 and p.t[0] - ctx.t0 <= GAP_S
            else "untracked"
        )
        went = (
            nxt.zone
            if nxt and p.link[v.last]
            else "match_end"
            if nxt is None and ctx.t1 - p.t[-1] <= GAP_S
            else "untracked"
        )
        idx = np.arange(v.first, v.last + 1)
        dur = float(p.t[v.last] - p.t[v.first])
        holes = np.diff(p.t[idx]) - ctx.dt
        length = _path_length(ctx, idx)
        data = ZoneVisitData(
            zone=v.zone,
            duration_s=_r(dur),
            untracked_s=_r(float(holes[holes > 1e-6].sum())),
            came_from=came,
            went_to=went,
            path_length_m=_r(length, 1),
            mean_speed_mps=_r(length / dur) if dur > 0 else None,
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


def _edge_time(ctx: Ctx, out_i: int, in_i: int, edge: float) -> float:
    """Time the centre crossed field x `edge` between samples out_i and in_i (linear)."""
    p = ctx.path
    dx = p.x[in_i] - p.x[out_i]
    f = 0.0 if dx == 0 else float(np.clip((edge - p.x[out_i]) / dx, 0.0, 1.0))
    return float(p.t[out_i] + f * (p.t[in_i] - p.t[out_i]))


def _crossing_facts(ctx: Ctx) -> tuple[list[Fact], list[Fact]]:
    """Bump and trench passes judged from measured samples only. A pass is a run of boxes
    on a barrier band; its sides are the nearest boxes before and after it, even across a
    gap. Two boxes on opposite sides with none on the band between are a crossing in a
    gap. Returns (crossings, crossings in gaps), each in time order."""
    p = ctx.path
    meas = np.flatnonzero(p.measured)
    seen: list[tuple[float, float, CrossingData]] = []
    gap: list[tuple[float, float, CrossingInGapData]] = []
    for barrier, (x0, x1) in BARRIER_X.items():
        lo, hi = x0 - BAND_MARGIN_M, x1 + BAND_MARGIN_M
        side = np.where(p.x[meas] < lo, -1, np.where(p.x[meas] > hi, 1, 0))
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
                _edge_time(ctx, prev, first, lo if side[s - 1] < 0 else hi)
                if entry_ok and prev is not None
                else float(p.t[first])
            )
            t_out = (
                _edge_time(ctx, last, nxt, lo if side[e + 1] < 0 else hi)
                if exit_ok and nxt is not None
                else float(p.t[last])
            )
            outcome: CrossingOutcome = "unknown"
            if prev is not None and nxt is not None:
                if side[s - 1] != side[e + 1]:
                    outcome = "crossed"
                elif entry_ok and exit_ok:
                    outcome = "returned"
            on = _in_span(ctx, t_in, t_out)
            sp = p.speed[on]
            y_med = float(np.median(p.y[meas[s : e + 1]]))
            trench = any(r.y0 <= y_med <= r.y1 and r.x0 == x0 for r in TRENCHES)
            span = max(t_out - t_in, ctx.dt)
            measured_s = min(span, int(p.measured[on].sum()) * ctx.dt)
            seen.append(
                (
                    t_in,
                    t_out,
                    CrossingData(
                        barrier=barrier,
                        lane="trench" if trench else "bump",
                        y_m=_r(y_med),
                        from_zone=None if prev is None else ctx.raw_zone(float(p.x[prev])),
                        to_zone=None if nxt is None else ctx.raw_zone(float(p.x[nxt])),
                        outcome=outcome,
                        entry_unseen_s=_ro(entry_unseen),
                        exit_unseen_s=_ro(exit_unseen),
                        time_on_barrier_s=_r(span) if entry_ok and exit_ok else None,
                        observed_s=_r(span),
                        stopped_s=_r(int((sp < STOP_MPS).sum()) * ctx.dt),
                        min_speed_mps=_ro(float(np.nanmin(sp)) if np.isfinite(sp).any() else None),
                        mean_speed_mps=_ro(_mean(sp)),
                        measured_s=_r(measured_s),
                    ),
                )
            )
    seen.sort(key=lambda v: v[0])
    gap.sort(key=lambda v: v[0])
    crossings: list[Fact] = [
        CrossingFact.model_validate({**_common(ctx, f"cross-{n:02d}", a, b), "data": d})
        for n, (a, b, d) in enumerate(seen, 1)
    ]
    in_gap: list[Fact] = [
        CrossingInGapFact.model_validate({**_common(ctx, f"crossgap-{n:02d}", a, b), "data": d})
        for n, (a, b, d) in enumerate(gap, 1)
    ]
    return crossings, in_gap


def _runs(mask: B, joined: Callable[[int], bool]) -> list[tuple[int, int]]:
    """(first, last) index runs where `mask` holds and `joined(i)` links i to i+1."""
    out: list[tuple[int, int]] = []
    for i in np.flatnonzero(mask).tolist():
        if out and out[-1][1] == i - 1 and joined(i - 1):
            out[-1] = (out[-1][0], i)
        else:
            out.append((i, i))
    return out


def _contiguous(ctx: Ctx, i: int) -> bool:
    """Sample i+1 directly follows sample i (no missing sample)."""
    return bool(ctx.path.frame[i + 1] - ctx.path.frame[i] == ctx.step)


def _stop_facts(ctx: Ctx) -> list[Fact]:
    p = ctx.path
    still = np.nan_to_num(p.speed, nan=np.inf) < STOP_MPS
    out: list[Fact] = []
    for a, b in _runs(still, lambda i: _contiguous(ctx, i)):
        dur = float(p.t[b] - p.t[a]) + ctx.dt
        if dur < STOP_MIN_S - 1e-6:
            continue
        nxt = b + 1
        ends = (
            "moved"
            if nxt < len(p.t) and _contiguous(ctx, b) and np.isfinite(p.speed[nxt])
            else "match_end"
            if ctx.t1 - p.t[b] <= GAP_S
            else "untracked"
        )
        idx = range(a, b + 1)
        near = [d for i in idx for d, _ in _nearest(ctx, i)[:1]]
        xm, ym = float(np.median(p.x[a : b + 1])), float(np.median(p.y[a : b + 1]))
        zones = Counter(p.zone[a : b + 1])
        data = StopData(
            duration_s=_r(dur),
            x=_r(xm),
            y=_r(ym),
            zone=max(zones, key=lambda z: zones[z]),
            on_barrier=_barrier_of(xm),
            ends=ends,
            nearest_tracked_m=_ro(min(near) if near else None),
            others_tracked_max=_others_range(ctx, float(p.t[a]), float(p.t[b]))[2],
        )
        n = len(out) + 1
        out.append(
            StopFact.model_validate(
                {**_common(ctx, f"stop-{n:02d}", float(p.t[a]), float(p.t[b])), "data": data}
            )
        )
    return out


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


def _other_at(ctx: Ctx, tid: int, frame: int) -> tuple[float, float] | None:
    track = ctx.others.by_track.get(tid)
    if track is None:
        return None
    k = int(np.searchsorted(track[0], frame))
    if k < len(track[0]) and track[0][k] == frame:
        return float(track[1][k]), float(track[2][k])
    return None


def _contact_end(ctx: Ctx, last: int, tid: int) -> ContactEnd:
    """How a contact ended, judged from the sample after its last close sample and the
    positions one second later."""
    p = ctx.path
    if ctx.t1 - p.t[last] <= CONTACT_BRIDGE_S:
        return "match_end"
    nxt = last + 1
    if nxt >= len(p.t) or p.t[nxt] - p.t[last] > CONTACT_BRIDGE_S:
        return "target_untracked"
    if _other_at(ctx, tid, int(p.frame[nxt])) is None:
        return "other_untracked"
    later = _in_span(ctx, float(p.t[last]) + 1.0, float(p.t[last]) + 1.0 + ctx.dt)
    start = _other_at(ctx, tid, int(p.frame[last]))
    end = _other_at(ctx, tid, int(p.frame[later[0]])) if len(later) else None
    if start is None or end is None or not len(later):
        return "other_untracked"
    j = int(later[0])
    me = math.hypot(p.x[j] - p.x[last], p.y[j] - p.y[last])
    them = math.hypot(end[0] - start[0], end[1] - start[1])
    return "target_moved_away" if me >= them else "other_moved_away"


def _relation(ctx: Ctx, team: int | None) -> tuple[int | None, str]:
    if team is None:
        return None, "unknown"
    slot = next((s for s in ctx.match.roster if s.team == team), None)
    if slot is None:
        return team, "unknown"
    return team, "partner" if slot.alliance == ctx.alliance else "opponent"


def _contact_facts(ctx: Ctx) -> list[Fact]:
    p = ctx.path
    near = [_nearest(ctx, i) for i in range(len(p.t))]
    close = np.array([bool(n) and n[0][0] <= CONTACT_M for n in near], np.bool_)
    runs = _runs(close, lambda i: bool(p.t[i + 1] - p.t[i] <= CONTACT_BRIDGE_S + 1e-6))
    merged: list[tuple[int, int]] = []
    for a, b in runs:
        if merged and p.t[a] - p.t[merged[-1][1]] <= CONTACT_BRIDGE_S:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    out: list[Fact] = []
    for n, (a, b) in enumerate(merged, 1):
        ta, tb = float(p.t[a]), float(p.t[b])
        per: dict[int, list[tuple[int, float]]] = {}
        for i in range(a, b + 1):
            for d, tid in near[i]:
                if d <= CONTACT_M:
                    per.setdefault(tid, []).append((int(p.frame[i]), d))
        robots: list[OtherRobot] = []
        for tid, hits in sorted(per.items(), key=lambda kv: min(d for _, d in kv[1])):
            teams = Counter(ctx.others.team.get((tid, f)) for f, _ in hits)
            team, rel = _relation(ctx, teams.most_common(1)[0][0])
            robots.append(
                OtherRobot.model_validate(
                    {
                        "track_id": tid,
                        "team": team,
                        "relation": rel,
                        "bumper": ctx.others.color.get(tid, "unknown"),
                        "min_distance_m": _r(min(d for _, d in hits)),
                        "close_s": _r(len(hits) * ctx.dt),
                    }
                )
            )
        sp = p.speed[a : b + 1]
        still = _runs(np.nan_to_num(sp, nan=np.inf) < STOP_MPS, lambda i: True)
        _, lo, hi = _others_range(ctx, ta, tb)
        last_near = min(
            ((d, tid) for d, tid in near[b] if d <= CONTACT_M), default=(0.0, robots[0].track_id)
        )
        data = ContactData(
            others=robots,
            duration_s=_r(tb - ta + ctx.dt),
            min_distance_m=_r(min(r.min_distance_m for r in robots)),
            before=_window(ctx, ta - CONTEXT_S, ta - ctx.dt),
            during=_window(ctx, ta, tb),
            after=_window(ctx, tb + ctx.dt, tb + CONTEXT_S),
            longest_still_s=_r(max(((e - s + 1) * ctx.dt for s, e in still), default=0.0)),
            ended=_contact_end(ctx, b, last_near[1]),
            others_tracked_min=lo,
            others_tracked_max=hi,
        )
        out.append(
            ContactFact.model_validate({**_common(ctx, f"contact-{n:02d}", ta, tb), "data": data})
        )
    return out


def _hub_on_facts(ctx: Ctx, visits: list[Visit]) -> list[Fact]:
    p = ctx.path
    out: list[Fact] = []
    visit_of = {i: v for v in visits for i in range(v.first, v.last + 1)}
    for ph in ctx.match.phases:
        if ph.name not in SHIFT_PHASES or ph.active_hubs is None:
            continue
        if ctx.alliance not in ph.active_hubs:
            continue
        t = ph.t_start
        k = int(np.searchsorted(p.t, t))
        cand = [i for i in (k - 1, k) if 0 <= i < len(p.t) and abs(p.t[i] - t) <= ctx.dt * 0.6]
        at = cand[0] if cand else None
        last_i = k - 1 if k - 1 >= 0 else None
        next_i = k if k < len(p.t) else None
        in_own = None if at is None else p.zone[at] == "own"
        since = since_gap = None
        if at is not None and in_own:
            v = visit_of[at]
            since = t - float(p.t[v.first])
            since_gap = v.first == 0 or not p.link[v.first - 1]
        reached = None
        reached_gap = False
        if not in_own:
            after = [i for i in range(k, len(p.t)) if p.t[i] < ph.t_end and p.zone[i] == "own"]
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


def _auto_fact(ctx: Ctx, gaps: list[GapSpan], crossings: Sequence[Fact]) -> Fact:
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
        start=start,
        start_untracked_s=_r((start.t if start else m.auto_end_t) - m.auto_start_t),
        waypoints=waypoints,
        end=end,
        end_untracked_s=_r(m.auto_end_t - (end.t if end else m.auto_start_t)),
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
        "same zone before and after a gap does not mean the robot stayed there.",
        "Other robots: only robots the tracker followed appear. On average "
        f"{coverage.others_tracked_mean:.1f} of the 5 other robots were tracked "
        f"(min {coverage.others_tracked_min}, max {coverage.others_tracked_max}). Share of "
        f"the match each was tracked and labelled: {shares}. A robot absent from these "
        "facts may have been right beside the target. Contact facts list only tracked "
        "robots; defense by an untracked robot leaves no contact fact, and may show only as "
        "a stop, a slow crossing or a gap. Check frames before scoring defense.",
        "Unlabelled other robots may be partners or opponents. Bumper colour is not alliance "
        "at offseason events. Two robots' centres closer than about 0.7 m are one robot seen "
        "twice, not contact.",
        "`crossing` facts were seen on a bump or trench; `crossing_in_gap` facts only know "
        "the robot was on one side and later on the other. Trenches are taken as the strip "
        "from each outer bump to the guardrail at the bumps' depth.",
        f"Positions are robot centres, about 0.2 m accurate. Zone lines use the centre with "
        f"{ZONE_HYST_M} m hysteresis; a robot straddling its zone line may count as either.",
        f"Target identity labels come from identify sources: {', '.join(sorted(sources))}.",
    ]


def build(run: RunDir) -> tuple[Facts, Ctx]:
    """Compute every fact for the target team; `facts` saves the result."""
    ctx = _load(run)
    gaps = _gap_spans(ctx)
    visits = _visits(ctx)
    crossings, in_gap = _crossing_facts(ctx)
    facts: list[Fact] = [
        *_phase_facts(ctx, gaps),
        *_gap_facts(ctx, gaps),
        *_visit_facts(ctx, visits),
        *crossings,
        *in_gap,
        *_stop_facts(ctx),
        *_contact_facts(ctx),
        *_hub_on_facts(ctx, visits),
        _auto_fact(ctx, gaps, [*crossings, *in_gap]),
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
ZONE_BGR: dict[Zone, tuple[int, int, int]] = {
    "own": (60, 60, 220),
    "neutral": (150, 150, 150),
    "opponent": (220, 130, 40),
}
ALLIANCE_BGR: dict[Alliance, tuple[int, int, int]] = {"red": (60, 60, 220), "blue": (220, 130, 40)}
KIND_ROWS: tuple[str, ...] = (
    "gap",
    "zone_visit",
    "crossing",
    "crossing_in_gap",
    "stop",
    "contact",
    "hub_on",
)


def _write_check(run: RunDir, facts: Facts, ctx: Ctx) -> None:
    """Timeline: phases and active hubs, zone band (gaps empty), speed, one row per fact
    kind with each fact's span and id."""
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
        f"tracked {facts.coverage.tracked_share:.0%} of the match",
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
    cv2.putText(img, "zone", (8, y + 22), font, 0.45, white, 1)
    p = ctx.path
    for i in range(len(p.t)):
        cv2.rectangle(
            img,
            (tx(float(p.t[i])), y),
            (tx(float(p.t[i]) + ctx.dt), y + 30),
            ZONE_BGR[p.zone[i]],
            -1,
        )
    cv2.putText(
        img,
        "red=own grey=neutral blue=opponent, empty=untracked",
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
    for a, b in _runs(ok, lambda i: _contiguous(ctx, i)):
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
        "stop": (200, 200, 0),
        "contact": (60, 60, 255),
        "hub_on": (0, 255, 0),
    }
    for r, kind in enumerate(KIND_ROWS):
        ry = y + r * row_h
        cv2.putText(img, kind, (8, ry + 18), font, 0.45, white, 1)
        cv2.line(img, (LEFT, ry + row_h - 2), (W - RIGHT, ry + row_h - 2), (45, 45, 45), 1)
        mine = [f for f in facts.facts if f.kind == kind]
        for n, f in enumerate(mine):
            a, b = tx(f.t_start), max(tx(f.t_end), tx(f.t_start) + 2)
            col = colours[kind]
            if kind == "zone_visit" and isinstance(f, ZoneVisitFact):
                col = ZONE_BGR[f.data.zone]
            cv2.rectangle(img, (a, ry + 4), (b, ry + 16), col, -1)
            label = f.id.split("-", 1)[1]
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
