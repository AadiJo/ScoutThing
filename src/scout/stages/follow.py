"""follow: reads tracks.parquet, identity.json, reads.parquet, detections.parquet,
corner_calibration.json, match.json.

Writes corner_tracks.parquet (`tracks.parquet` columns, see `scout.schemas.tracks`) and
corner_identity.json (`CornerIdentity`): robots followed through the corner cameras in frames
where the main view has no sample of them, with a team only on evidence, and corner labels
carried onto main-view tracks identify left unlabelled. Read them with the
main view's through `scout.positions.load_positions`; this stage never changes the track or
identify stages' files, and a main-view sample is never replaced or moved.

Each corner camera looks across one alliance zone at its hub from a field corner, so it sees
some robots the main view cannot: against the hub's side, in the near corner. Each calibrated
corner view (corners stage) on its own:

1. Place boxes. A corner bumper box (not `static`, detector confidence CONF_MIN, clear of the
   panel edge) is cast to the field at bumper height and pushed `face_m` away from the camera
   (its centre is on the near bumper face). It is kept only within the view's `trusted_m`
   distances from the camera, WALL_M inside the guardrail, off the hubs, and clear of the
   bumps: the camera is under a metre up, so a robot 0.165 m up a bump casts a metre or more
   too far, and is left out rather than misplaced. Boxes of one frame within MERGE_M (or
   overlapping by MERGE_IOU) are one robot.
2. Tracklets. Frame by frame, robots join live tracklets by distance within what a robot can
   drive (MAX_SPEED_MPS plus NOISE_M). Purity first, as in the track stage: a link is cut when
   a rival robot or tracklet is within AMBIGUOUS_M of as good, or the bumper colours disagree;
   a tracklet coasts at most COAST_S; tracklets shorter than MIN_TRACKLET_S are dropped.
   Positions are smoothed with a local linear fit over SMOOTH_S.
3. The main view. A tracklet sample with a main-view sample within SAME_ROBOT_M (or one of its
   bumper colour within DUP_M) is a robot the main view has: evidence, never written. A sample
   with exactly one main-view sample within SAME_ROBOT_M and no other within CLEAR_M is an
   overlap sample of that main track.
4. Labels, from evidence on a tracklet's samples:
   - `corner_digits`: a named number read (`reads.parquet` `team`) on the sample's boxes;
   - `corner_overlap`: an overlap sample on a main sample labelled with a team (identify's
     number-read and appearance labels alike);
   - `corner_handoff`: at the tracklet's first (last) sample, a team's labelled main-view
     samples ended (began) within HANDOFF_S, within HANDOFF_M plus driving at MAX_SPEED_MPS,
     with no robot of another team in the main view within CLEAR_M of that point in that
     time, and no other tracklet's end there.
   Consecutive evidence of one team is a run, as in the identify stage. A run with MIN_OVERLAP
   overlap samples, MIN_READS reads or a hand-off labels the tracklet from its first to its
   last evidence, the first run from the tracklet's start and the last to its end; between
   runs of two teams the robot may have changed, so nothing is labelled there. A label is
   dropped when the main view has its team more than SAME_ROBOT_M away in any of its frames,
   when another tracklet carries the team at the same time (unless the two stay within DUP_M
   throughout: one robot's front and side bumpers boxed apart), or when a written stretch
   begins or ends more than HANDOFF_JUMP_M (plus driving at JUMP_SPEED_MPS) from the team's
   main-view sample next to it: the corner stretch does not continue the main one.
5. Written: a tracklet's samples the main view does not have (step 3) and, when labelled,
   where it has no sample of the team; per frame one sample per robot (`_dedupe`); stretches
   shorter than MIN_WRITTEN_S dropped. A labelled stretch is shifted to meet the team's
   main-view samples at its seams (`_anchor`), so a hand-off does not jump by the corner's
   bias. Unlabelled tracklets are written too, as other robots the main view did not follow,
   except fixtures: evidence-less, never seen, within STILL_SPAN_M all along.
6. Carried labels (`_carry`). Where a labelled corner sample is an overlap sample (step 3) of
   a main-view sample identify left unlabelled (not `conflict`), and the main view has no
   sample of the team in that frame, the main track is that team: runs of at least MIN_CARRY
   such samples become `corner_carried` segments on the main track. A run is not carried when
   another main track has the team in an overlapping segment, when the track has another
   team's label within CARRY_GUARD_S of it, or when another carried run gives the track
   another team, or the team another track, at the same time. Main-view positions stay as
   they are; only the label is added.

The two corner cameras' trusted ranges do not overlap on CapCity, so each view is followed on
its own. On the sixteen CapCity matches, hiding labelled main-view stretches the corner sees
and following them here recovered about 90% of the hidden time at 0.11 to 0.14 m median error,
with no recovered sample carrying another team; hiding labels instead (the stretch made a
new unlabelled track), step 6 carried the team back onto about 74% of it, with no carried
sample of another team.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment  # pyright: ignore[reportMissingTypeStubs]

from scout.camera import Camera
from scout.games.rebuilt2026 import (
    BUMP_HEIGHT_M,
    BUMPER_CENTER_HEIGHT_M,
    BUMPS,
    FIELD_LENGTH_M,
    FIELD_WIDTH_M,
    HUBS,
)
from scout.positions import with_teams
from scout.run import RunDir, Stage
from scout.schemas.calibration import CornerCalibration, CornerCalibrations
from scout.schemas.common import PixelRect, ViewName
from scout.schemas.detections import SAMPLE_HZ, read_detections
from scout.schemas.identity import (
    CornerIdentity,
    Identity,
    LabelSegment,
    LabelSource,
    read_reads,
)
from scout.schemas.match import MatchInfo
from scout.schemas.tracks import (
    COLOR,
    CORNER_TRACK_ID_BASE,
    KIND,
    STATE,
    TRACKS,
    VIEW,
    read_tracks,
    write_tracks,
)

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]

CORNER_VIEWS: tuple[ViewName, ...] = ("corner_left", "corner_right")

# Boxes.
CONF_MIN = 0.25  # detector confidence, as the track stage
EDGE_PX = 2.0  # a box this close to the panel edge is cut off; its centre is not the robot's
WALL_M = 0.25
"""A robot centre is half a robot (0.475 m) inside the guardrail; one cast closer than this,
less the corner error, is a fixture on the wall (CapCity: the outposts' blue frames)."""
BUMP_MARGIN_M = 0.5  # a centre this close to a bump can have a wheel on it (half a robot)
MERGE_M = 0.6  # one robot's boxes (front and side bumpers) cast within this of each other
MERGE_IOU = 0.5

# Tracklets.
MAX_SPEED_MPS = 5.5  # as the track stage
NOISE_M = 0.5  # corner placement error between consecutive samples, with margin
AMBIGUOUS_M = 0.4  # a rival this much worse or better cuts the link
COAST_S = 0.35  # a tracklet may miss this long (three samples at 10 Hz)
MIN_TRACKLET_S = 1.0
SMOOTH_S = 0.3  # Gaussian sigma of the local linear fit

# The main view.
SAME_ROBOT_M = 1.0  # corner and main placements of one robot: 0.35 m + 0.2 m p90, margin
CLEAR_M = 1.8  # no second robot this close for a sample to name one main track

# Labels.
# A number read at the reader's 0.99 bar is rarely wrong, but one box can be misread: two.
MIN_READS = 2
MIN_OVERLAP = 3  # 0.3 s riding on one labelled main track with no other robot near
HANDOFF_S = 1.0  # a hand-off is a tracklet starting within this of the team's last main sample
HANDOFF_M = 1.0  # ...and this close to it (corner and main error), plus driving
HANDOFF_JUMP_M = 0.6  # corner 0.4 m (90th percentile) plus main 0.2 m
JUMP_SPEED_MPS = 3.0  # driving allowed across a seam: above the target's 99th percentile speed
ANCHOR_STEPS = 2  # a seam's main sample at most this many samples (0.2 s) away
MIN_CARRY = 5  # 0.5 s of a corner label riding on one unlabelled main track
CARRY_GUARD_S = 2.0  # no label of another team on that track this near (a changed robot)
DUP_M = 1.3  # two corner boxes of one robot (front and side faces) cast up to this far apart
MIN_WRITTEN_S = 0.5
STILL_SPAN_M = 0.5
"""An unlabelled tracklet the main view never saw that stays within this is not written: on
CapCity these are fixtures (fuel frames by the walls) boxed as bumpers for whole minutes."""


@dataclass
class Robots:
    """One view's robots: boxes of one frame merged, in frame order."""

    frame: I64
    t: F64
    xy: F64
    box: F64
    conf: F64
    color: list[str]
    teams: list[list[int]]
    """Named number reads on the robot's boxes."""


@dataclass
class Tracklet:
    idx: list[int] = field(default_factory=list[int])


# --- boxes ----------------------------------------------------------------------------------


def _on_bump(xy: F64) -> NDArray[np.bool_]:
    m = BUMP_MARGIN_M
    out = np.zeros(len(xy), np.bool_)
    for r in BUMPS:
        out |= (
            (xy[:, 0] >= r.x0 - m)
            & (xy[:, 0] <= r.x1 + m)
            & (xy[:, 1] >= r.y0 - m)
            & (xy[:, 1] <= r.y1 + m)
        )
    return out


def _place(cam: Camera, cal: CornerCalibration, uv: F64) -> tuple[F64, NDArray[np.bool_]]:
    """Robot centres (n, 2) of corner box centres `uv`, and which may be used (step 1)."""
    c = cam.center[:2]

    def cast(z: float) -> tuple[F64, F64]:
        p = cam.to_field(uv, z)
        d = p - c
        dist = np.linalg.norm(d, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            centre = p + cal.face_m * d / dist[:, None]
        return centre, np.linalg.norm(centre - c, axis=1)

    xy, dist = cast(BUMPER_CENTER_HEIGHT_M)
    up, _ = cast(BUMPER_CENTER_HEIGHT_M + BUMP_HEIGHT_M)
    near, far = cal.trusted_m
    m = -WALL_M
    with np.errstate(invalid="ignore"):
        ok = (
            np.isfinite(xy).all(1)
            & (dist >= near)
            & (dist <= far)
            & (xy[:, 0] >= -m)
            & (xy[:, 0] <= FIELD_LENGTH_M + m)
            & (xy[:, 1] >= -m)
            & (xy[:, 1] <= FIELD_WIDTH_M + m)
            & ~_on_bump(xy)
            & ~(np.isfinite(up).all(1) & _on_bump(np.nan_to_num(up)))
        )
    for h in HUBS.values():
        ok &= ~((xy[:, 0] >= h.x0) & (xy[:, 0] <= h.x1) & (xy[:, 1] >= h.y0) & (xy[:, 1] <= h.y1))
    return xy, ok


def _iou(a: F64, b: F64) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return float(inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter))


def _robots(boxes: pl.DataFrame, cam: Camera, cal: CornerCalibration, rect: PixelRect) -> Robots:
    """Step 1 for one view: placed boxes, merged per frame."""
    e = EDGE_PX
    b = boxes.filter(
        ~pl.col("static")
        & (pl.col("det_conf") >= CONF_MIN)
        & (pl.col("x1") > rect.x + e)
        & (pl.col("x2") < rect.x + rect.w - e)
        & (pl.col("y1") > rect.y + e)
        & (pl.col("y2") < rect.y + rect.h - e)
    ).sort("frame")
    box = b.select("x1", "y1", "x2", "y2").to_numpy().astype(np.float64)
    uv = np.c_[(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2]
    xy, ok = _place(cam, cal, uv)
    frame = b["frame"].to_numpy().astype(np.int64)[ok]
    t = b["t"].to_numpy().astype(np.float64)[ok]
    conf = b["det_conf"].to_numpy().astype(np.float64)[ok]
    color = [str(c) for c in b["color"].to_list()]
    color = [c for c, k in zip(color, ok.tolist(), strict=True) if k]
    team = [v for v, k in zip(b["team"].to_list(), ok.tolist(), strict=True) if k]
    xy, box = xy[ok], box[ok]
    out = Robots(
        np.zeros(0, np.int64), np.zeros(0), np.zeros((0, 2)), np.zeros((0, 4)), np.zeros(0), [], []
    )
    fr: list[int] = []
    ts: list[float] = []
    xys: list[F64] = []
    bxs: list[F64] = []
    cfs: list[float] = []
    for lo, hi in _runs_of(frame):
        for g in _groups(xy[lo:hi], box[lo:hi]):
            g = [lo + i for i in g]
            w = conf[g]
            fr.append(int(frame[lo]))
            ts.append(float(t[lo]))
            xys.append((xy[g] * w[:, None]).sum(0) / w.sum())
            bb = box[g]
            bxs.append(np.r_[bb[:, :2].min(0), bb[:, 2:].max(0)])
            cfs.append(float(w.max()))
            votes: dict[str, float] = {}
            for k in g:
                votes[color[k]] = votes.get(color[k], 0.0) + float(conf[k])
            out.color.append(max(votes, key=lambda c: votes[c]))
            out.teams.append([int(v) for k in g if (v := team[k]) is not None])
    if fr:
        out.frame = np.array(fr, np.int64)
        out.t = np.array(ts)
        out.xy = np.array(xys)
        out.box = np.array(bxs)
        out.conf = np.array(cfs)
    return out


def _groups(xy: F64, box: F64) -> list[list[int]]:
    """Indices of one frame's boxes grouped into robots: cast within MERGE_M of each other,
    or overlapping by MERGE_IOU, chained."""
    parent = list(range(len(xy)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(xy)):
        for j in range(i + 1, len(xy)):
            close = float(np.linalg.norm(xy[i] - xy[j])) < MERGE_M
            if close or _iou(box[i], box[j]) > MERGE_IOU:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(xy)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _runs_of(frame: I64) -> list[tuple[int, int]]:
    """(start, end) index ranges of equal values in a sorted array."""
    if not len(frame):
        return []
    cuts = np.flatnonzero(np.diff(frame)) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, len(frame)]
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


# --- tracklets ------------------------------------------------------------------------------


def _link(r: Robots) -> list[Tracklet]:
    """Step 2: tracklets of one view's robots, cut wherever a link is in doubt."""
    pieces: list[Tracklet] = []
    live: list[int] = []
    for lo, hi in _runs_of(r.frame):
        sel = np.arange(lo, hi)
        now = float(r.t[lo])
        live = [p for p in live if now - r.t[pieces[p].idx[-1]] <= COAST_S + 1e-6]
        assigned: dict[int, int] = {}
        ended: set[int] = set()
        if live:
            last = np.array([pieces[p].idx[-1] for p in live])
            diff = r.xy[last][:, None, :] - r.xy[sel][None, :, :]
            dist = np.hypot(diff[..., 0], diff[..., 1])
            gate = MAX_SPEED_MPS * (now - r.t[last]) + NOISE_M
            clash = np.array(
                [[_clash(r.color[a], r.color[c]) for c in sel.tolist()] for a in last.tolist()],
                np.bool_,
            ).reshape(len(last), len(sel))
            feasible = (dist <= gate[:, None]) & ~clash
            rows, cols = linear_sum_assignment(np.where(feasible, dist, 1e6))
            for a, c in zip(rows.tolist(), cols.tolist(), strict=True):
                if not feasible[a, c]:
                    continue
                close = dist[a, c] + AMBIGUOUS_M
                near_gate = dist <= gate[:, None]
                rivals = int(np.sum(near_gate[a] & (dist[a] < close))) + int(
                    np.sum(near_gate[:, c] & (dist[:, c] < close))
                )
                if rivals > 2:
                    ended.add(live[a])
                else:
                    assigned[c] = live[a]
        live = [p for p in live if p not in ended and p not in assigned.values()] + [
            p for p in assigned.values()
        ]
        for c, j in enumerate(sel.tolist()):
            if c in assigned:
                pieces[assigned[c]].idx.append(j)
            else:
                pieces.append(Tracklet([j]))
                live.append(len(pieces) - 1)
    return [p for p in pieces if r.t[p.idx[-1]] - r.t[p.idx[0]] >= MIN_TRACKLET_S - 1e-6]


def _clash(a: str, b: str) -> bool:
    return a != b and a != "unknown" and b != "unknown"


def _smooth(t: F64, xy: F64) -> F64:
    """Gaussian-weighted local linear fit of positions over time, evaluated at `t`."""
    out = np.empty_like(xy)
    for i, ti in enumerate(t):
        lo, hi = np.searchsorted(t, [ti - 3 * SMOOTH_S, ti + 3 * SMOOTH_S])
        dt = t[lo:hi] - ti
        w = np.sqrt(np.exp(-0.5 * (dt / SMOOTH_S) ** 2))
        a = np.stack([np.ones_like(dt), dt], 1) * w[:, None]
        coef, *_ = np.linalg.lstsq(a, xy[lo:hi] * w[:, None], rcond=None)
        out[i] = coef[0]
    return out


# --- the main view and labels ------------------------------------------------------------------


@dataclass(frozen=True)
class Main:
    """Main-view samples by frame: positions (n, 2), bumper colours (the track's), teams (-1:
    unlabelled, -2: unlabelled because identify saw two teams there) and track ids."""

    by_frame: dict[int, tuple[F64, NDArray[np.str_], I64, I64]]
    team_frames: dict[int, dict[int, F64]]
    """team -> frame -> position of its labelled main-view sample (a single one)."""

    def at(self, frame: int) -> tuple[F64, NDArray[np.str_], I64, I64]:
        none = np.zeros(0, np.int64)
        return self.by_frame.get(frame, (np.zeros((0, 2)), np.zeros(0, np.str_), none, none))


def _main(tracks: pl.DataFrame, segments: list[LabelSegment]) -> Main:
    df = with_teams(tracks, segments).with_columns(pl.col("team").fill_null(-1)).sort("frame")
    frame = df["frame"].to_numpy().astype(np.int64)
    xy = df.select("x", "y").to_numpy().astype(np.float64)
    color = df["color"].cast(pl.String).to_numpy().astype(np.str_)
    team = df["team"].to_numpy().astype(np.int64)
    tid = df["track_id"].to_numpy().astype(np.int64)
    for sg in segments:
        if sg.source == "conflict":
            team[(tid == sg.track_id) & (frame >= sg.frame_start) & (frame <= sg.frame_end)] = -2
    by_frame = {
        int(frame[lo]): (xy[lo:hi], color[lo:hi], team[lo:hi], tid[lo:hi])
        for lo, hi in _runs_of(frame)
    }
    team_frames: dict[int, dict[int, F64]] = {}
    for f, (p, _, tm, _) in by_frame.items():
        for k, v in enumerate(tm.tolist()):
            if v >= 0 and int((tm == v).sum()) == 1:
                team_frames.setdefault(v, {})[f] = p[k]
    return Main(by_frame, team_frames)


Evidence = tuple[int, int, LabelSource]
"""(sample index, team, kind) of one piece of label evidence on a tracklet."""


@dataclass
class Label:
    """Samples `k0`..`k1` (inclusive) of a tracklet are `team`, or were until `dropped`."""

    k0: int
    k1: int
    team: int
    source: LabelSource
    n_reads: int
    dropped: str | None = None


@dataclass
class Follow:
    """One tracklet with its smoothed positions, what the main view had at each sample, its
    label evidence and labels, and which samples are written."""

    view: ViewName
    frame: I64
    t: F64
    xy: F64
    box: F64
    conf: F64
    color: str
    seen: NDArray[np.bool_]
    """The main view has a robot within SAME_ROBOT_M, or one of the same bumper colour within
    DUP_M (the corner places a robot's side face up to a metre off)."""
    evidence: list[Evidence]
    labels: list[Label] = field(default_factory=list[Label])
    written: NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, np.bool_))

    def team_at(self) -> I64:
        """Team per sample from the labels still standing, -1 where none."""
        out = np.full(len(self.frame), -1, np.int64)
        for lb in self.labels:
            if lb.dropped is None:
                out[lb.k0 : lb.k1 + 1] = lb.team
        return out


def _follow(view: ViewName, r: Robots, p: Tracklet, main: Main) -> Follow:
    idx = np.array(p.idx)
    t = r.t[idx]
    xy = _smooth(t, r.xy[idx])
    frame = r.frame[idx]
    color = Counter(r.color[j] for j in idx.tolist()).most_common(1)[0][0]
    seen = np.zeros(len(idx), np.bool_)
    evidence: list[Evidence] = []
    for k, f in enumerate(frame.tolist()):
        for v in r.teams[int(idx[k])]:
            evidence.append((k, v, "corner_digits"))
        mxy, mcolor, mteam, _ = main.at(f)
        if not len(mxy):
            continue
        d = np.linalg.norm(mxy - xy[k], axis=1)
        seen[k] = bool(((d <= SAME_ROBOT_M) | ((d <= DUP_M) & (mcolor == color))).any())
        if int((d <= SAME_ROBOT_M).sum()) == 1 and int((d <= CLEAR_M).sum()) == 1:
            tm = int(mteam[int(np.argmin(d))])
            if tm >= 0:
                evidence.append((k, tm, "corner_overlap"))
    return Follow(
        view=view,
        frame=frame,
        t=t,
        xy=xy,
        box=r.box[idx],
        conf=r.conf[idx],
        color=color,
        seen=seen,
        evidence=evidence,
    )


def _handoffs(follows: list[Follow], main: Main, step: int, fps: float) -> None:
    """Add hand-off evidence at each tracklet's ends (step 4)."""
    win = round(HANDOFF_S * fps / step)
    ends = [(int(f.frame[k]), f.xy[k]) for f in follows for k in (0, -1)]
    for fl in follows:
        for k, sign in ((0, -1), (len(fl.frame) - 1, 1)):
            f0, p0 = int(fl.frame[k]), fl.xy[k]
            # Labelled main samples on the far side of this end within HANDOFF_S, of a team
            # the main view does not have at this end's frame.
            cands: set[int] = set()
            for team, frames in main.team_frames.items():
                if f0 in frames:
                    continue
                for n in range(1, win + 1):
                    q = frames.get(f0 + sign * n * step)
                    if q is not None:
                        reach = HANDOFF_M + MAX_SPEED_MPS * n * step / fps
                        if float(np.linalg.norm(q - p0)) <= reach:
                            cands.add(team)
                        break
            if len(cands) != 1:
                continue
            team = next(iter(cands))
            # No other robot near: main samples of other teams, other tracklets' ends.
            crowded = any(
                bool(((np.linalg.norm(mxy - p0, axis=1) <= CLEAR_M) & (mteam != team)).any())
                for n in range(-win, win + 1)
                for mxy, _, mteam, _ in (main.at(f0 + n * step),)
                if len(mxy)
            ) or any(
                abs(fj - f0) <= win * step and 0 < float(np.linalg.norm(pj - p0)) <= CLEAR_M
                for fj, pj in ends
                if not (fj == f0 and pj is p0)
            )
            if not crowded:
                fl.evidence.append((k, team, "corner_handoff"))


def _label(fl: Follow) -> None:
    """Labels from the tracklet's evidence (step 4): consecutive evidence of one team is a
    run; a run with enough of it labels from its first to its last evidence, the first run
    from the tracklet's start and the last run to its end. Between runs of two teams the
    robot may have changed, so nothing is labelled there."""
    ev = sorted(fl.evidence, key=lambda e: e[0])
    runs: list[list[Evidence]] = []
    for e in ev:
        if runs and runs[-1][-1][1] == e[1]:
            runs[-1].append(e)
        else:
            runs.append([e])
    for i, run in enumerate(runs):
        kinds = Counter(kind for _, _, kind in run)
        source: LabelSource | None = (
            "corner_overlap"
            if kinds["corner_overlap"] >= MIN_OVERLAP
            else "corner_digits"
            if kinds["corner_digits"] >= MIN_READS
            else "corner_handoff"
            if kinds["corner_handoff"]
            else None
        )
        if source is None:
            continue
        k0 = 0 if i == 0 else run[0][0]
        k1 = len(fl.frame) - 1 if i == len(runs) - 1 else run[-1][0]
        fl.labels.append(Label(k0, k1, run[0][1], source, kinds["corner_digits"]))


def _check(follows: list[Follow], main: Main, step: int, fps: float) -> None:
    """Drop labels the main view or another tracklet contradicts, choose the written
    samples, and drop labels whose written stretches jump from the main view (step 4, 5)."""
    for fl in follows:
        for lb in fl.labels:
            frames = main.team_frames.get(lb.team, {})
            for k in range(lb.k0, lb.k1 + 1):
                q = frames.get(int(fl.frame[k]))
                if q is not None and float(np.linalg.norm(q - fl.xy[k])) > SAME_ROBOT_M:
                    lb.dropped = "main_elsewhere"
                    break
    spans = [(fl, lb) for fl in follows for lb in fl.labels]
    for fa, a in spans:
        for fb, b in spans:
            if fa is fb or a.team != b.team:
                continue
            meet = fa.frame[a.k0] <= fb.frame[b.k1] and fb.frame[b.k0] <= fa.frame[a.k1]
            if meet and not _same_robot(fa, a, fb, b):
                a.dropped = b.dropped = "two_tracklets"
    for fl in follows:
        team = fl.team_at()
        fl.written = ~fl.seen & np.array(
            [
                tm < 0 or f not in main.team_frames.get(tm, {})
                for f, tm in zip(fl.frame.tolist(), team.tolist(), strict=True)
            ],
            np.bool_,
        )
    for fl in follows:
        span = float(np.linalg.norm(fl.xy.max(0) - fl.xy.min(0)))
        if not fl.evidence and not fl.seen.any() and span < STILL_SPAN_M:
            fl.written[:] = False
    _dedupe(follows)
    for fl in follows:
        _drop_short(fl, step, fps)
        for lb in fl.labels:
            if lb.dropped is not None:
                continue
            own = main.team_frames.get(lb.team, {})
            inside = np.zeros(len(fl.frame), np.bool_)
            inside[lb.k0 : lb.k1 + 1] = True
            for lo, hi in _true_runs(fl.written & inside):
                for k, sign in ((lo, -1), (hi - 1, 1)):
                    f0 = int(fl.frame[k])
                    for n in range(1, round(HANDOFF_S * fps / step) + 1):
                        q = own.get(f0 + sign * n * step)
                        if q is None:
                            continue
                        allow = HANDOFF_JUMP_M + JUMP_SPEED_MPS * n * step / fps
                        if float(np.linalg.norm(q - fl.xy[k])) > allow:
                            lb.dropped = "jump"
                        break
            if lb.dropped is None:
                for lo, hi in _true_runs(fl.written & inside):
                    _anchor(fl, lo, hi, own, step)


def _anchor(fl: Follow, lo: int, hi: int, own: dict[int, F64], step: int) -> None:
    """Shift written samples lo..hi-1 so they meet the team's main-view samples next to them
    (within ANCHOR_STEPS samples): the offset at each seam, interpolated in time between two
    seams, held from one."""
    offs: list[tuple[float, F64]] = []
    for k, sign in ((lo, -1), (hi - 1, 1)):
        for n in range(1, ANCHOR_STEPS + 1):
            q = own.get(int(fl.frame[k]) + sign * n * step)
            if q is not None:
                offs.append((float(fl.t[k]), q - fl.xy[k]))
                break
    if not offs:
        return
    t = fl.t[lo:hi]
    if len(offs) == 2 and offs[1][0] > offs[0][0]:
        w = ((t - offs[0][0]) / (offs[1][0] - offs[0][0]))[:, None]
        fl.xy[lo:hi] += (1 - w) * offs[0][1] + w * offs[1][1]
    else:
        fl.xy[lo:hi] += offs[0][1]


def _same_robot(fa: Follow, a: Label, fb: Follow, b: Label) -> bool:
    """Two labels of one team on two tracklets of one view at the same time are one robot
    (two of its bumper faces boxed apart) when they stay within DUP_M in every frame both
    cover, and are seen together at least once."""
    if fa.view != fb.view:
        return False
    fr_a = fa.frame[a.k0 : a.k1 + 1]
    fr_b = fb.frame[b.k0 : b.k1 + 1]
    common, ia, ib = np.intersect1d(fr_a, fr_b, return_indices=True)
    if not len(common):
        return False
    d = np.linalg.norm(fa.xy[a.k0 + ia] - fb.xy[b.k0 + ib], axis=1)
    return bool((d <= DUP_M).all())


def _dedupe(follows: list[Follow]) -> None:
    """One written sample per robot (step 5): of two written samples of one view and frame
    within DUP_M, of one team or one unlabelled of the same colour, only the labelled one,
    else the one on the longer tracklet, is kept."""
    by_frame: dict[tuple[ViewName, int], list[tuple[Follow, int, int]]] = {}
    for fl in follows:
        team = fl.team_at()
        for k in np.flatnonzero(fl.written).tolist():
            by_frame.setdefault((fl.view, int(fl.frame[k])), []).append((fl, k, int(team[k])))
    for items in by_frame.values():
        if len(items) < 2:
            continue
        order = sorted(items, key=lambda it: (it[2] < 0, -len(it[0].frame)))
        kept: list[tuple[Follow, int, int]] = []
        for fl, k, tm in order:
            dup = any(
                float(np.linalg.norm(fl.xy[k] - g.xy[j])) <= DUP_M
                and (tm == tg or (tm < 0 and fl.color == g.color))
                for g, j, tg in kept
            )
            if dup:
                fl.written[k] = False
            else:
                kept.append((fl, k, tm))


def _true_runs(m: NDArray[np.bool_]) -> list[tuple[int, int]]:
    d = np.diff(np.r_[0, m.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist(), strict=True))


def _drop_short(fl: Follow, step: int, fps: float) -> None:
    """Unwrite stretches of written samples (no more than COAST_S apart) shorter than
    MIN_WRITTEN_S."""
    w = fl.written.copy()
    idx = np.flatnonzero(w)
    if not len(idx):
        return
    breaks = np.flatnonzero(np.diff(fl.frame[idx]) > COAST_S * fps + step) + 1
    for g in np.split(idx, breaks):
        if fl.t[g[-1]] - fl.t[g[0]] < MIN_WRITTEN_S - 1e-6:
            w[g] = False
    fl.written = w


# --- the stage ------------------------------------------------------------------------------


def follow_views(
    tracks: pl.DataFrame,
    segments: list[LabelSegment],
    boxes: pl.DataFrame,
    cal: CornerCalibrations,
    match: MatchInfo,
) -> tuple[list[Follow], list[LabelSegment]]:
    """Every corner tracklet with its labels and written samples, and the labels carried onto
    main-view tracks (step 6). `boxes` are the corner rows of reads.parquet with the
    detector's confidence as `det_conf`."""
    main = _main(tracks, segments)
    frames = np.unique(tracks["frame"].to_numpy())
    step = int(np.median(np.diff(frames)))
    some = tracks.filter(pl.col("t") > 0).row(0, named=True)
    fps = float(some["frame"]) / float(some["t"])
    out: list[Follow] = []
    for view in CORNER_VIEWS:
        c = cal.views.get(view)
        if c is None or view not in match.views:
            continue
        cam = Camera.from_calibration(c.camera)
        r = _robots(boxes.filter(pl.col("view") == view), cam, c, match.views[view])
        out += [_follow(view, r, p, main) for p in _link(r)]
    _handoffs(out, main, step, fps)
    for fl in out:
        _label(fl)
    _check(out, main, step, fps)
    return out, _carry(out, main, segments, step, fps)


def _carry(
    follows: list[Follow], main: Main, segments: list[LabelSegment], step: int, fps: float
) -> list[LabelSegment]:
    """Step 6: label segments for main-view tracks from the corner labels riding on them."""
    hits: dict[tuple[int, int], list[int]] = {}
    for fl in follows:
        team = fl.team_at()
        for k in np.flatnonzero((team >= 0) & fl.seen).tolist():
            f, tm = int(fl.frame[k]), int(team[k])
            if f in main.team_frames.get(tm, {}):
                continue
            mxy, _, mteam, mtid = main.at(f)
            d = np.linalg.norm(mxy - fl.xy[k], axis=1)
            i = int(np.argmin(d))
            alone = int((d <= SAME_ROBOT_M).sum()) == 1 and int((d <= CLEAR_M).sum()) == 1
            if alone and int(mteam[i]) == -1:
                hits.setdefault((int(mtid[i]), tm), []).append(f)
    guard = round(CARRY_GUARD_S * fps)
    runs: list[tuple[int, int, int, int]] = []
    for (tid, tm), frames in hits.items():
        fr = np.unique(np.array(frames, np.int64))
        for g in np.split(fr, np.flatnonzero(np.diff(fr) > step) + 1):
            if len(g) < MIN_CARRY:
                continue
            f0, f1 = int(g[0]), int(g[-1])
            clash = any(
                sg.team is not None
                and (
                    (
                        (sg.team == tm and sg.track_id != tid and sg.frame_start <= f1)
                        and f0 <= sg.frame_end
                    )
                    or (
                        (sg.team != tm and sg.track_id == tid and sg.frame_start <= f1 + guard)
                        and f0 - guard <= sg.frame_end
                    )
                )
                for sg in segments
            )
            if not clash:
                runs.append((tid, tm, f0, f1))
    out: list[LabelSegment] = []
    for tid, tm, f0, f1 in runs:
        rival = any((t2 == tm) != (i2 == tid) and f0 <= b1 and a1 <= f1 for i2, t2, a1, b1 in runs)
        if not rival:
            out.append(
                LabelSegment(
                    track_id=tid,
                    frame_start=f0,
                    frame_end=f1,
                    t_start=f0 / fps,
                    t_end=f1 / fps,
                    team=tm,
                    confidence=1.0,
                    source="corner_carried",
                    n_reads=0,
                )
            )
    return out


def corner_boxes(run: RunDir) -> pl.DataFrame:
    """Corner rows of reads.parquet with the detector's confidence (`det_conf`)."""
    keys = ["frame", "view", "x1", "y1", "x2", "y2"]
    det = (
        read_detections(run.detections)
        .filter(pl.col("view") != "main")
        .select(*keys, pl.col("conf").alias("det_conf"))
        .unique(keys, keep="first")
    )
    return read_reads(run.reads).filter(pl.col("view") != "main").join(det, on=keys, how="inner")


def tables(
    follows: list[Follow], carried: list[LabelSegment]
) -> tuple[pl.DataFrame, CornerIdentity]:
    """corner_tracks rows and label segments of the written samples (one corner track per
    tracklet with written samples, one segment per run of written samples with one label or
    none), then the carried main-view segments."""
    rows: list[pl.DataFrame] = []
    segs: list[LabelSegment] = []
    for fl in follows:
        m = fl.written
        if not m.any():
            continue
        tid = CORNER_TRACK_ID_BASE + len(rows)
        k = int(m.sum())
        rows.append(
            pl.DataFrame(
                {
                    "track_id": pl.Series([tid] * k, dtype=pl.Int32),
                    "color": pl.Series([fl.color] * k, dtype=COLOR),
                    "frame": pl.Series(fl.frame[m], dtype=pl.Int64),
                    "t": pl.Series(fl.t[m], dtype=pl.Float64),
                    "x": pl.Series(fl.xy[m, 0], dtype=pl.Float64),
                    "y": pl.Series(fl.xy[m, 1], dtype=pl.Float64),
                    "state": pl.Series(["measured"] * k, dtype=STATE),
                    "view": pl.Series([fl.view] * k, dtype=VIEW),
                    "x1": pl.Series(fl.box[m, 0], dtype=pl.Float32),
                    "y1": pl.Series(fl.box[m, 1], dtype=pl.Float32),
                    "x2": pl.Series(fl.box[m, 2], dtype=pl.Float32),
                    "y2": pl.Series(fl.box[m, 3], dtype=pl.Float32),
                    "conf": pl.Series(fl.conf[m], dtype=pl.Float32),
                    "kind": pl.Series(["bumper"] * k, dtype=KIND),
                    "partial": pl.Series([False] * k, dtype=pl.Boolean),
                }
            )
        )
        team = fl.team_at()
        label = {k: lb for lb in fl.labels if lb.dropped is None for k in range(lb.k0, lb.k1 + 1)}
        idx = np.flatnonzero(m)
        cuts = np.flatnonzero(np.diff(team[idx])) + 1
        for g in np.split(idx, cuts):
            lb = label.get(int(g[0]))
            dropped = any(
                x.dropped is not None and x.k0 <= g[-1] and g[0] <= x.k1 for x in fl.labels
            )
            segs.append(
                LabelSegment(
                    track_id=tid,
                    frame_start=int(fl.frame[g[0]]),
                    frame_end=int(fl.frame[g[-1]]),
                    t_start=float(fl.t[g[0]]),
                    t_end=float(fl.t[g[-1]]),
                    team=lb.team if lb else None,
                    confidence=1.0 if lb else 0.0,
                    source=lb.source if lb else "conflict" if dropped else "no_reads",
                    n_reads=lb.n_reads if lb else 0,
                )
            )
    df = pl.concat(rows) if rows else pl.DataFrame(schema=dict(TRACKS.columns))
    return df, CornerIdentity(segments=segs + carried)


def follow(run: RunDir) -> None:
    match = MatchInfo.load(run.match_json)
    cal = CornerCalibrations.load(run.corner_calibration_json)
    follows, carried = follow_views(
        read_tracks(run.tracks),
        list(Identity.load(run.identity_json).segments),
        corner_boxes(run),
        cal,
        match,
    )
    df, ident = tables(follows, carried)
    write_tracks(run.corner_tracks, df)
    ident.save(run.corner_identity_json)
    dt = 1.0 / SAMPLE_HZ
    for view in CORNER_VIEWS:
        if view not in cal.views:
            print(f"follow: {view}: not calibrated")
            continue
        fs = [f for f in follows if f.view == view]
        seconds: Counter[str] = Counter()
        for fl in fs:
            src = {
                k: lb.source
                for lb in fl.labels
                if lb.dropped is None
                for k in range(lb.k0, lb.k1 + 1)
            }
            for k in np.flatnonzero(fl.written).tolist():
                seconds[src.get(k, "unlabelled")] += 1
        dropped = Counter(lb.dropped for fl in fs for lb in fl.labels if lb.dropped)
        print(
            f"follow: {view}: {len(fs)} tracklets; written "
            + ", ".join(f"{k} {v * dt:.1f} s" for k, v in seconds.most_common())
            + f"; labels dropped {dict(dropped)}"
        )
    print(f"follow: carried onto main tracks {sum(s.t_end - s.t_start for s in carried):.1f} s")


STAGE = Stage(
    "follow",
    inputs=lambda r: (
        r.tracks,
        r.identity_json,
        r.reads,
        r.detections,
        r.corner_calibration_json,
        r.match_json,
    ),
    outputs=lambda r: (r.corner_tracks, r.corner_identity_json),
    fn=follow,
)
