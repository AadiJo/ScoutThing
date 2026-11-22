"""track: reads detections.parquet, calibration.json, match.json, video.json.

Writes tracks.parquet: robot tracks in field metres (see `scout.schemas.tracks`) and
`tracks_check.jpg`, a top-down plot of every track, a timeline of when each track exists,
and the number of live tracks over time.

Purity comes before completeness: a track that might have swapped robots is cut, and the
pieces are left for the identify stage to label. Steps:

1. Place boxes. Only the main view is calibrated, so only its boxes are used. A bumper
   box's centre is cast onto the plane at bumper centre height: a bumper wraps the robot,
   so the centre of its box sits over the robot's centre (the bottom edge is the near
   bumper, half a robot toward the camera). A whole-robot box's centre is cast onto the
   plane at `ROBOT_CENTER_HEIGHT_M`. Boxes that land off the field (wall banners, displays,
   hub tops all project far off it), inside a hub, or whose pixel size does not fit a robot
   at that spot are dropped. Nothing else is called static: parked robots are real.
2. Merge duplicates. A bumper box mostly inside a larger box of the other colour is
   dropped. Same-colour bumper boxes in one frame closer than `MERGE_M`, or boxes of any
   colour that nearly coincide, are one robot: their union box is placed again. Whole-robot
   boxes are merged among themselves the same way, then each is dropped when a bumper box
   explains it (lies mostly inside it, or lands within `MERGE_M`): bumper boxes place better
   and carry a colour. Only whole-robot boxes with no bumper box on them become
   observations, with an unknown colour.
3. Tracklets. Frame by frame, boxes are assigned to live tracklets by distance in metres
   within what a robot can drive. A link is cut instead when a rival box or tracklet is
   nearly as good, or when another robot vanished next to the box in the last second, or
   when it passes between whole-robot-only and bumper boxes with another robot near.
   Tracklets bridge at most one missed sample. Short or weak tracklets are dropped.
4. Joins. With the whole match in hand, a tracklet end is joined to a later start across
   gaps up to `JOIN_MAX_S` when both ends, carried forward at their own velocities, meet,
   the colours agree, and no rival comes close for either end.
5. Duplicates. Where two tracks sit closer than `MERGE_M` for `DUP_MIN_S`, the weaker
   loses those samples.
6. Samples. Positions are smoothed with a local linear fit in time, gaps up to
   `INTERP_MAX_S` are filled by linear interpolation, longer gaps are left empty.
"""

from __future__ import annotations

import resource
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2
import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment  # pyright: ignore[reportMissingTypeStubs]

from scout.camera import Camera
from scout.games.rebuilt2026 import (
    BUMPER_CENTER_HEIGHT_M,
    BUMPER_Z0_M,
    BUMPER_Z1_M,
    BUMPS,
    FIELD_LENGTH_M,
    FIELD_WIDTH_M,
    HUBS,
    ROBOT_FOOTPRINT_M,
    Rect,
)
from scout.run import RunDir, Stage
from scout.schemas.detections import SAMPLE_HZ, read_detections
from scout.schemas.match import MatchInfo
from scout.schemas.tracks import COLOR, KIND, STATE, VIEW, write_tracks
from scout.schemas.video import VideoInfo

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]

CONF_MIN = 0.25
"""Boxes below this are dropped; below it most in-field boxes were the wrong size."""
FIELD_MARGIN_M = 0.3
"""How far off the carpet a placed box may land: calibration error near the field ends."""
SIZE_RATIO_W = (0.2, 1.6)
"""Allowed box width over the width a 0.95 m robot's bumper band projects to at that spot;
below 0.2 is a sliver of something, above 1.6 is bigger than any robot."""
SIZE_RATIO_H = (0.3, 2.0)
"""The same for box height (tall boxes come from robots seen close and from above)."""
ROBOT_CENTER_HEIGHT_M = 0.35
"""Plane a whole-robot box's centre is cast onto. Against San Diego hand-tracked truth
(6738 boxes on one robot each) it lands at median 0.20 m, p90 0.41 m, +0.05 m along the line
of sight (0.30 m: +0.21; 0.40 m: -0.11); on the CapCity hand marks median 0.09-0.10 m. The
box's bottom centre on the carpet lands 0.25-0.5 m short. On the same robot it sits a median
0.2 m (p90 0.4 m) from the bumper rule's point."""
ROBOT_SIZE_W = (0.45, 1.4)
"""Allowed whole-robot box width over a robot's bumper band width at that spot. Narrower
boxes are people, poles and slivers of half-hidden robots: on San Diego 97% of boxes on
hand-tracked robots pass, and 101 of 121 boxes on no hand-tracked robot fail."""
ROBOT_SIZE_H = (0.8, 3.0)
"""The same for height over the bumper band's height: the box holds the whole robot (1%-99%
of boxes on robots: 0.97-2.5)."""
ROBOT_COVER = 0.6
"""A bumper box with this share of its area inside a whole-robot box explains it."""
MERGE_M = 0.75
"""Same-colour boxes closer than this in one frame are one robot: two robots' centres
cannot be closer than a robot width (0.95 m) less calibration noise."""
MERGE_IOU = 0.7
"""Boxes of any colour overlapping this much are one object classed twice; at 0.5 two
robots overlapping in the image were fused."""
INSIDE_FRAC = 0.6
"""A box with this share of its area inside a larger box of the other colour is dropped:
a robot with magenta bumpers gets a red box plus a small blue box at its edge."""
MAX_SPEED_MPS = 5.5
"""Fastest a robot drives."""
NOISE_M = 0.35
"""Position noise allowed on top of driving: box jitter plus calibration error."""
COLOR_PENALTY_M = 0.3
"""Cost added for assigning a box of the other bumper colour to a tracklet."""
AMBIGUOUS_M = 0.3
"""An assignment whose runner-up costs less than this more is too close to call: cut."""
COAST_S = 0.25
"""A tracklet with no box for longer than this ends; longer gaps are left to joins, which
see both sides. At 0.5 s a robot hidden by a referee was continued onto its neighbour."""
CONTACT_M = 1.5
"""Centre distance at which a robot that vanishes may have been taken for its neighbour:
touching robots are about 0.95 m apart, plus calibration noise and some driving."""
CONTACT_S = 1.0
"""How long a vanished robot stays a possible stand-in for a neighbour's box."""
SWITCH_S = 2.0
"""A tracklet passing between whole-robot-only and bumper boxes is cut if another robot
was within CONTACT_M of the box this recently. A whole-robot box has no colour and can sit
between two touching robots: in Playoff 12 at 69 s a tracklet followed 2468 on whole-robot
boxes and then took 2687's bumper box, 1.3 s after 2687's own tracklet was last seen. Cutting
at every switch instead doubled the track count and cost label recall (Qualification 16
target untracked 30 s -> 39 s), because one robot's pieces then block each other's joins."""
MIN_TRACKLET_OBS = 3
"""Tracklets with fewer boxes are dropped: 0.3 s is too short to be checked."""
MIN_TRACKLET_CONF = 0.35
"""Tracklets whose median confidence is below this are dropped (the reference floor)."""
JOIN_MAX_S = 2.0
"""Longest gap a join may bridge; beyond it too many robots could have come and gone."""
END_WINDOW_S = 0.5
"""Tracklet end and start position and velocity come from this much time at each end."""
ACCEL_MPS2 = 6.0
"""Hardest a robot changes velocity; bounds how far a join may bend the path in a gap."""
JOIN_MARGIN_M = 0.5
"""A join must beat every rival for either end by this much meeting distance."""
DUP_MIN_S = 0.5
"""Two tracks must sit on one spot this long before one is called a duplicate."""
SMOOTH_S = 0.25
"""Gaussian time scale of the local linear smoother: removes 10 Hz box jitter but keeps
a turn (a robot at 3 m/s moves 0.75 m in this time)."""
INTERP_MAX_S = 1.0
"""Gaps up to this are interpolated; longer gaps are left empty."""

CHECK_IMAGE = "tracks_check.jpg"
COLOR_CODES = ("red", "blue", "unknown")
UNKNOWN = 2


@dataclass
class Obs:
    """Placed boxes after merging, one row per robot per frame, sorted by frame."""

    frame: I64
    t: F64
    xy: F64
    box: NDArray[np.float32]
    conf: NDArray[np.float32]
    color: I64
    whole: NDArray[np.bool_]
    """Backed by a whole-robot box only (no bumper box on that robot in that frame)."""


@dataclass
class Piece:
    """A tracklet or joined track: indices into `Obs`, in time order."""

    idx: list[int] = field(default_factory=list[int])


def _expected_size(cam: Camera, xy: F64) -> tuple[F64, F64]:
    """Pixel width and height of a robot's bumper band at each field point, the mean over
    robots square to the field and turned 45 degrees."""
    h = ROBOT_FOOTPRINT_M / 2
    ws: list[F64] = []
    hs: list[F64] = []
    for th in (0.0, np.pi / 4):
        c, s = np.cos(th), np.sin(th)
        sq = np.array([[-h, -h], [h, -h], [h, h], [-h, h]]) @ np.array([[c, s], [-s, c]])
        corners = [
            np.stack([xy[:, 0:1] + sq[:, 0], xy[:, 1:2] + sq[:, 1], np.full((len(xy), 4), z)], -1)
            for z in (BUMPER_Z0_M, BUMPER_Z1_M)
        ]
        uv = cam.project(np.concatenate(corners, 1))
        ws.append(np.max(uv[..., 0], 1) - np.min(uv[..., 0], 1))
        hs.append(np.max(uv[..., 1], 1) - np.min(uv[..., 1], 1))
    return np.mean(ws, 0), np.mean(hs, 0)


def _in_rect(xy: F64, r: Rect, margin: float = 0.0) -> NDArray[np.bool_]:
    return (
        (xy[:, 0] >= r.x0 - margin)
        & (xy[:, 0] <= r.x1 + margin)
        & (xy[:, 1] >= r.y0 - margin)
        & (xy[:, 1] <= r.y1 + margin)
    )


def _place(
    cam: Camera, box: NDArray[np.float32], whole: NDArray[np.bool_]
) -> tuple[F64, NDArray[np.bool_]]:
    """Field centres of boxes (x1 y1 x2 y2 rows; `whole` marks whole-robot boxes) and which
    of them can be a robot."""
    b = box.astype(np.float64)
    z = np.where(whole, ROBOT_CENTER_HEIGHT_M, BUMPER_CENTER_HEIGHT_M)
    xy = cam.to_field(np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1), z)
    field_rect = Rect(0.0, 0.0, FIELD_LENGTH_M, FIELD_WIDTH_M)
    ok = np.isfinite(xy).all(1) & _in_rect(np.nan_to_num(xy, nan=-99.0), field_rect, FIELD_MARGIN_M)
    for hub in HUBS.values():
        ok &= ~_in_rect(np.nan_to_num(xy, nan=-99.0), hub)
    ew, eh = _expected_size(cam, np.where(ok[:, None], xy, FIELD_LENGTH_M / 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        rw = (b[:, 2] - b[:, 0]) / ew
        rh = (b[:, 3] - b[:, 1]) / eh
    lo_w, hi_w = (
        np.where(whole, ROBOT_SIZE_W[0], SIZE_RATIO_W[0]),
        np.where(whole, ROBOT_SIZE_W[1], SIZE_RATIO_W[1]),
    )
    lo_h, hi_h = (
        np.where(whole, ROBOT_SIZE_H[0], SIZE_RATIO_H[0]),
        np.where(whole, ROBOT_SIZE_H[1], SIZE_RATIO_H[1]),
    )
    ok &= (rw >= lo_w) & (rw <= hi_w) & (rh >= lo_h) & (rh <= hi_h)
    return xy, ok


def _area(a: NDArray[np.float32]) -> float:
    return float((a[2] - a[0]) * (a[3] - a[1]))


def _inter(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return float(w * h) if w > 0 and h > 0 else 0.0


def _iou(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    inter = _inter(a, b)
    return inter / (_area(a) + _area(b) - inter)


def _inside_other(box: NDArray[np.float32], color: I64) -> NDArray[np.bool_]:
    """Boxes in one frame that lie mostly (INSIDE_FRAC) inside a larger box of another
    known colour."""
    n = len(box)
    out = np.zeros(n, np.bool_)
    for i in range(n):
        for j in range(n):
            if i == j or color[i] == color[j] or UNKNOWN in (color[i], color[j]):
                continue
            if _area(box[j]) > _area(box[i]) and _inter(box[i], box[j]) >= INSIDE_FRAC * _area(
                box[i]
            ):
                out[i] = True
    return out


def _merge_frame(xy: F64, box: NDArray[np.float32], color: I64) -> list[list[int]]:
    """Groups of box indices in one frame that are the same robot (union-find over pairs
    closer than MERGE_M with the same colour, or overlapping more than MERGE_IOU)."""
    n = len(xy)
    parent = list(range(n))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            near = float(np.hypot(*(xy[i] - xy[j]))) < MERGE_M and color[i] == color[j]
            if near or _iou(box[i], box[j]) > MERGE_IOU:
                parent[root(j)] = root(i)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    return list(groups.values())


def _union(b: NDArray[np.float32]) -> NDArray[np.float32]:
    return np.array([b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()], np.float32)


def _explained(
    robot_box: NDArray[np.float32], robot_xy: F64, box: NDArray[np.float32], xy: F64
) -> bool:
    """Whether any of a frame's kept bumper boxes (`box`, placed at `xy`) is on the same
    robot as a merged whole-robot box: mostly inside it, or placed within MERGE_M of any of
    its member boxes' placements."""
    for b, p in zip(box, xy, strict=True):
        if _inter(b, robot_box) >= ROBOT_COVER * _area(b):
            return True
        if np.min(np.hypot(*(robot_xy - p).T)) < MERGE_M:
            return True
    return False


def _observations(run: RunDir, cam: Camera) -> tuple[Obs, dict[str, int]]:
    """Main-view boxes placed on the field, filtered and merged per frame."""
    det = read_detections(run.detections)
    counts = {"boxes": len(det)}
    det = det.filter((pl.col("view") == "main") & (pl.col("conf") >= CONF_MIN))
    counts["main_conf"] = len(det)
    box = det.select("x1", "y1", "x2", "y2").to_numpy().astype(np.float32)
    whole = (det["kind"] == "robot").to_numpy()
    xy, ok = _place(cam, box, whole)
    counts["placed"] = int(ok.sum())
    counts["robot_placed"] = int((ok & whole).sum())
    frame = det["frame"].to_numpy()[ok]
    t = det["t"].to_numpy()[ok]
    xy, box, whole = xy[ok], box[ok], whole[ok]
    conf = det["conf"].to_numpy()[ok].astype(np.float32)
    color = (
        det["color"]
        .cast(pl.String)
        .replace_strict({c: i for i, c in enumerate(COLOR_CODES)}, return_dtype=pl.Int64)
        .to_numpy()[ok]
    )

    out_f: list[int] = []
    out_t: list[float] = []
    out_box: list[NDArray[np.float32]] = []
    out_conf: list[float] = []
    out_col: list[int] = []
    out_whole: list[bool] = []

    def emit(ii: I64, is_whole: bool) -> None:
        best = ii[int(np.argmax(conf[ii]))]
        out_box.append(_union(box[ii]))
        out_f.append(int(frame[best]))
        out_t.append(float(t[best]))
        out_conf.append(float(conf[ii].max()))
        out_col.append(UNKNOWN if is_whole else int(color[best]))
        out_whole.append(is_whole)

    order = np.argsort(frame, kind="stable")
    bounds = np.flatnonzero(np.diff(frame[order])) + 1
    for sel in np.split(order, bounds):
        if len(sel) == 0:
            continue
        bum, rob = sel[~whole[sel]], sel[whole[sel]]
        inside = _inside_other(box[bum], color[bum])
        counts["inside"] = counts.get("inside", 0) + int(inside.sum())
        bum = bum[~inside]
        for g in _merge_frame(xy[bum], box[bum], color[bum]):
            emit(bum[g], False)
        for g in _merge_frame(xy[rob], box[rob], color[rob]):
            ii = rob[g]
            if _explained(_union(box[ii]), xy[ii], box[bum], xy[bum]):
                counts["robot_explained"] = counts.get("robot_explained", 0) + 1
            else:
                emit(ii, True)
    mbox = np.array(out_box, np.float32).reshape(-1, 4)
    mwhole = np.array(out_whole, np.bool_)
    mxy, _ = _place(cam, mbox, mwhole)
    obs = Obs(
        frame=np.array(out_f, np.int64),
        t=np.array(out_t, np.float64),
        xy=mxy,
        box=mbox,
        conf=np.array(out_conf, np.float32),
        color=np.array(out_col, np.int64),
        whole=mwhole,
    )
    counts["merged"] = len(obs.t)
    counts["robot_only"] = int(mwhole.sum())
    return obs, counts


def _tracklets(obs: Obs) -> tuple[list[Piece], dict[str, int]]:
    """Frame-by-frame assignment into tracklets that end at every close call.

    A link from a tracklet to a box is cut when a rival box or tracklet is nearly as good
    (`AMBIGUOUS_M`), or when another robot vanished within `CONTACT_M` of the box in the
    last `CONTACT_S` (it may be the robot now seen). A link between a whole-robot-only
    observation and a bumper observation is also cut when another robot was seen within
    `CONTACT_M` of the box in the last `SWITCH_S`, whether it is still tracked or not.
    Returns the tracklets and how many links each rule cut."""
    pieces: list[Piece] = []
    cut: list[bool] = []
    live: list[int] = []
    recent: list[int] = []
    cuts = {"rival": 0, "contact": 0, "kind": 0}
    for sel in _frames(obs):
        now = float(obs.t[sel[0]])
        live = [p for p in live if now - obs.t[pieces[p].idx[-1]] <= COAST_S]
        recent = [p for p in recent if now - obs.t[pieces[p].idx[-1]] <= SWITCH_S]
        assigned: dict[int, int] = {}
        ended: set[int] = set()
        if live:
            last = np.array([pieces[p].idx[-1] for p in live])
            diff = obs.xy[last][:, None, :] - obs.xy[sel][None, :, :]
            dist = np.hypot(diff[..., 0], diff[..., 1])
            gate = MAX_SPEED_MPS * (now - obs.t[last]) + NOISE_M
            feasible = dist <= gate[:, None]
            cost = dist + COLOR_PENALTY_M * _mismatch(obs.color[last], obs.color[sel])
            rows, cols = linear_sum_assignment(np.where(feasible, cost, 1e6))
            for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
                if feasible[r, c]:
                    assigned[c] = live[r]
            taken = set(assigned.values())
            for c, p in list(assigned.items()):
                r = live.index(p)
                close = cost[r, c] + AMBIGUOUS_M
                rivals = np.sum(feasible[r] & (cost[r] < close)) + np.sum(
                    feasible[:, c] & (cost[:, c] < close)
                )
                j = int(sel[c])
                contact = _near_other(obs, pieces, recent, p, j, CONTACT_S, taken, cut)
                kind = obs.whole[pieces[p].idx[-1]] != obs.whole[j] and _near_other(
                    obs, pieces, recent, p, j, SWITCH_S
                )
                if rivals > 2 or contact or kind:
                    why = "rival" if rivals > 2 else "contact" if contact else "kind"
                    cuts[why] += 1
                    del assigned[c]
                    ended.add(p)
                    cut[p] = True
        live = [p for p in live if p not in ended]
        for c, j in enumerate(sel.tolist()):
            if c in assigned:
                pieces[assigned[c]].idx.append(j)
            else:
                pieces.append(Piece([j]))
                cut.append(False)
                live.append(len(pieces) - 1)
                recent.append(len(pieces) - 1)
    return pieces, cuts


def _near_other(
    obs: Obs,
    pieces: list[Piece],
    recent: list[int],
    p: int,
    j: int,
    window_s: float,
    skip: set[int] | None = None,
    cut: list[bool] | None = None,
) -> bool:
    """True if an established tracklet other than `p` (MIN_TRACKLET_OBS boxes) was last
    seen within CONTACT_M of box `j`, at most `window_s` before it. One- or two-box blips
    are not robots that can stand in. With `cut` (the contact rule), tracklets in `skip`
    (seen this frame) or themselves cut are ignored, and only those last seen after `p`
    started count, since a tracklet that ended before `p` began may be `p`'s own robot."""
    start = obs.t[pieces[p].idx[0]]
    for q in recent:
        if q == p or len(pieces[q].idx) < MIN_TRACKLET_OBS:
            continue
        k = pieces[q].idx[-1]
        if obs.t[j] - obs.t[k] > window_s:
            continue
        if cut is not None and (cut[q] or q in (skip or set()) or obs.t[k] <= start):
            continue
        if np.hypot(*(obs.xy[k] - obs.xy[j])) < CONTACT_M:
            return True
    return False


def _frames(obs: Obs) -> list[I64]:
    """Observation indices grouped by frame, in time order."""
    order = np.argsort(obs.frame, kind="stable")
    return [s for s in np.split(order, np.flatnonzero(np.diff(obs.frame[order])) + 1) if len(s)]


def _mismatch(a: I64, b: I64) -> NDArray[np.bool_]:
    """Pairwise: both colours known and different."""
    return (a[:, None] != b[None, :]) & (a[:, None] != UNKNOWN) & (b[None, :] != UNKNOWN)


def _color(obs: Obs, piece: Piece) -> int:
    """Confidence-weighted majority bumper colour of a piece, ignoring unknown boxes."""
    votes = np.bincount(obs.color[piece.idx], weights=obs.conf[piece.idx], minlength=3)
    return UNKNOWN if votes[:2].sum() == 0 else int(np.argmax(votes[:2]))


def _end_state(obs: Obs, piece: Piece, at_start: bool) -> tuple[F64, F64]:
    """Position and velocity over the first or last END_WINDOW_S of a piece, from a line
    fitted in time. Velocity is zero when the window holds too little to fit."""
    idx = np.asarray(piece.idx)
    t = obs.t[idx]
    near = t <= t[0] + END_WINDOW_S if at_start else t >= t[-1] - END_WINDOW_S
    tt, xy = t[near], obs.xy[idx[near]]
    t_end = tt[0] if at_start else tt[-1]
    if len(tt) < MIN_TRACKLET_OBS or tt[-1] - tt[0] < 0.2:
        return np.median(xy, 0), np.zeros(2)
    a = np.stack([np.ones_like(tt), tt - t_end], 1)
    coef, *_ = np.linalg.lstsq(a, xy, rcond=None)
    v = coef[1]
    speed = float(np.hypot(*v))
    return coef[0], v * min(1.0, MAX_SPEED_MPS / speed) if speed > 0 else v


def _join(obs: Obs, pieces: list[Piece]) -> tuple[list[Piece], int]:
    """Join tracklet ends to later starts across gaps up to JOIN_MAX_S.

    Each side is carried to the middle of the gap at its own end velocity; the two must
    meet within what NOISE_M and ACCEL_MPS2 allow, have compatible colours, and be
    reachable at MAX_SPEED_MPS. A pair is joined only when it beats by JOIN_MARGIN_M every
    rival for either end: every other reachable, colour-compatible pairing, whether or not it
    meets the bend limit. Without that, a robot's own continuation that bent a little too
    hard was no rival, and a looser pairing with a neighbour won unopposed (Qualification 16,
    46-48 s, 6800 joined onto 2468). Returns the joined tracks and the join count."""
    n = len(pieces)
    if n == 0:
        return [], 0
    t0 = np.array([obs.t[p.idx[0]] for p in pieces])
    t1 = np.array([obs.t[p.idx[-1]] for p in pieces])
    col = np.array([_color(obs, p) for p in pieces])
    s_xy, s_v = (np.array(v) for v in zip(*(_end_state(obs, p, True) for p in pieces), strict=True))
    e_xy, e_v = (
        np.array(v) for v in zip(*(_end_state(obs, p, False) for p in pieces), strict=True)
    )
    gap = t0[None, :] - t1[:, None]
    g = np.clip(gap, 0.0, None)[..., None]
    meet = (e_xy[:, None, :] + e_v[:, None, :] * g / 2) - (
        s_xy[None, :, :] - s_v[None, :, :] * g / 2
    )
    resid = np.hypot(meet[..., 0], meet[..., 1])
    straight = e_xy[:, None, :] - s_xy[None, :, :]
    dist = np.hypot(straight[..., 0], straight[..., 1])
    reach = (gap > 0) & (gap <= JOIN_MAX_S) & ~_mismatch(col, col)
    reach &= dist <= MAX_SPEED_MPS * gap + NOISE_M
    ok = reach & (resid <= NOISE_M + ACCEL_MPS2 * gap**2 / 4)
    cost = np.where(ok, resid, np.inf)
    rival = np.where(reach, resid, np.inf)
    succ: dict[int, int] = {}
    for e, s_ in np.argwhere(ok).tolist():
        rivals_e = np.delete(rival[e], s_)
        rivals_s = np.delete(rival[:, s_], e)
        margin = cost[e, s_] + JOIN_MARGIN_M
        if (rivals_e >= margin).all() and (rivals_s >= margin).all():
            succ[e] = s_
    preds = set(succ.values())
    tracks: list[Piece] = []
    for i in range(n):
        if i in preds:
            continue
        chain = Piece(list(pieces[i].idx))
        j = i
        while j in succ:
            j = succ[j]
            chain.idx.extend(pieces[j].idx)
        tracks.append(chain)
    return tracks, len(succ)


def _drop_duplicates(obs: Obs, tracks: list[Piece]) -> tuple[list[Piece], int]:
    """Where two tracks share at least DUP_MIN_S of frames closer than MERGE_M, they are
    one robot seen twice (or two robots too entangled to place apart): the track with less
    confidence there loses those frames. Returns the tracks (empty ones removed) and the
    number of samples dropped."""
    by_frame = [dict(zip(obs.frame[p.idx].tolist(), p.idx, strict=True)) for p in tracks]
    drop: list[set[int]] = [set() for _ in tracks]
    min_shared = round(DUP_MIN_S * SAMPLE_HZ)
    for a in range(len(tracks)):
        for b in range(a + 1, len(tracks)):
            shared = by_frame[a].keys() & by_frame[b].keys()
            if len(shared) < min_shared:
                continue
            ia = np.array([by_frame[a][f] for f in shared])
            ib = np.array([by_frame[b][f] for f in shared])
            close = np.hypot(*(obs.xy[ia] - obs.xy[ib]).T) < MERGE_M
            if close.sum() < min_shared:
                continue
            weak = a if obs.conf[ia[close]].sum() < obs.conf[ib[close]].sum() else b
            drop[weak].update((ia if weak == a else ib)[close].tolist())
    out = [Piece([i for i in p.idx if i not in d]) for p, d in zip(tracks, drop, strict=True)]
    return [p for p in out if len(p.idx) >= MIN_TRACKLET_OBS], sum(len(d) for d in drop)


def _smooth(t: F64, xy: F64) -> F64:
    """Gaussian-weighted local linear fit of positions over time, evaluated at `t`."""
    out = np.empty_like(xy)
    for i, ti in enumerate(t):
        lo, hi = np.searchsorted(t, [ti - 3 * SMOOTH_S, ti + 3 * SMOOTH_S])
        dt = t[lo:hi] - ti
        w = np.exp(-0.5 * (dt / SMOOTH_S) ** 2)
        a = np.stack([np.ones_like(dt), dt], 1) * np.sqrt(w)[:, None]
        coef, *_ = np.linalg.lstsq(a, xy[lo:hi] * np.sqrt(w)[:, None], rcond=None)
        out[i] = coef[0]
    return out


def _rows(obs: Obs, tracks: Sequence[Piece], fps: float, step: int) -> pl.DataFrame:
    """One row per track per sampled frame where it exists: smoothed measured samples
    plus linearly interpolated samples in gaps up to INTERP_MAX_S."""
    cols: dict[str, list[NDArray[np.generic]]] = {
        k: [] for k in ("track_id", "color", "frame", "x", "y", "measured", "src")
    }
    for tid, p in enumerate(tracks):
        idx = np.asarray(p.idx)
        fr = obs.frame[idx]
        xy = _smooth(obs.t[idx], obs.xy[idx])
        frames_out = [fr]
        xs, ys, src = [xy[:, 0]], [xy[:, 1]], [idx]
        for k in np.flatnonzero((np.diff(fr) > step) & (np.diff(fr) / fps <= INTERP_MAX_S + 1e-9)):
            fill = np.arange(fr[k] + step, fr[k + 1], step)
            a = (fill - fr[k]) / (fr[k + 1] - fr[k])
            frames_out.append(fill)
            xs.append(xy[k, 0] + a * (xy[k + 1, 0] - xy[k, 0]))
            ys.append(xy[k, 1] + a * (xy[k + 1, 1] - xy[k, 1]))
            src.append(np.full(len(fill), -1))
        f = np.concatenate(frames_out)
        cols["track_id"].append(np.full(len(f), tid))
        cols["color"].append(np.full(len(f), _color(obs, p)))
        cols["frame"].append(f)
        cols["x"].append(np.concatenate(xs))
        cols["y"].append(np.concatenate(ys))
        cols["src"].append(np.concatenate(src))
    src = np.concatenate(cols["src"]).astype(np.int64) if tracks else np.zeros(0, np.int64)
    m = src >= 0
    safe = np.where(m, src, 0)

    def boxcol(values: NDArray[np.float32]) -> pl.Series:
        return pl.Series(np.where(m, values[safe], np.nan).astype(np.float32)).fill_nan(None)

    frame = np.concatenate(cols["frame"]).astype(np.int64) if tracks else np.zeros(0, np.int64)
    color = np.concatenate(cols["color"]).astype(np.int64) if tracks else np.zeros(0, np.int64)
    box = obs.box if len(obs.box) else np.zeros((1, 4), np.float32)
    conf = obs.conf if len(obs.conf) else np.zeros(1, np.float32)
    whole = obs.whole if len(obs.whole) else np.zeros(1, np.bool_)
    kind = [("robot" if whole[i] else "bumper") if i >= 0 else None for i in src.tolist()]
    df = pl.DataFrame(
        {
            "track_id": pl.Series(
                np.concatenate(cols["track_id"]) if tracks else [], dtype=pl.Int32
            ),
            "color": pl.Series([COLOR_CODES[c] for c in color.tolist()], dtype=COLOR),
            "frame": pl.Series(frame, dtype=pl.Int64),
            "t": pl.Series(frame / fps, dtype=pl.Float64),
            "x": pl.Series(np.concatenate(cols["x"]) if tracks else [], dtype=pl.Float64),
            "y": pl.Series(np.concatenate(cols["y"]) if tracks else [], dtype=pl.Float64),
            "state": pl.Series(
                ["measured" if v else "interpolated" for v in m.tolist()], dtype=STATE
            ),
            "view": pl.Series(["main" if v else None for v in m.tolist()], dtype=VIEW),
            "x1": boxcol(box[:, 0]),
            "y1": boxcol(box[:, 1]),
            "x2": boxcol(box[:, 2]),
            "y2": boxcol(box[:, 3]),
            "conf": boxcol(conf),
            "kind": pl.Series(kind, dtype=KIND),
        }
    )
    return df.sort("track_id", "frame")


# --- check image ----------------------------------------------------------------

PX_PER_M = 70
PAD = 30
BGR = {"red": (60, 60, 230), "blue": (230, 140, 40), "unknown": (160, 160, 160)}


def _palette(n: int) -> list[tuple[int, int, int]]:
    """n distinct bright BGR colours (golden-angle hues)."""
    out: list[tuple[int, int, int]] = []
    for i in range(n):
        hsv = np.array([[[int(i * 137.5) % 180, 220, 255]]], np.uint8)
        b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
        out.append((int(b), int(g), int(r)))
    return out


def _field_panel(df: pl.DataFrame, n_tracks: int, step: int) -> NDArray[np.uint8]:
    """Top-down field with hubs, bumps and every track in its own colour (red wall
    right, blue wall left, as in WPILib x). Gaps left empty are not drawn across."""
    w = int(FIELD_LENGTH_M * PX_PER_M) + 2 * PAD
    h = int(FIELD_WIDTH_M * PX_PER_M) + 2 * PAD
    img = np.full((h, w, 3), 24, np.uint8)

    def px(x: F64, y: F64) -> NDArray[np.int32]:
        return np.stack([PAD + x * PX_PER_M, h - PAD - y * PX_PER_M], -1).round().astype(np.int32)

    def rect(r: Rect, colour: tuple[int, int, int], fill: bool) -> None:
        a = px(np.array(r.x0), np.array(r.y0))
        b = px(np.array(r.x1), np.array(r.y1))
        cv2.rectangle(img, a.tolist(), b.tolist(), colour, -1 if fill else 2)

    rect(Rect(0, 0, FIELD_LENGTH_M, FIELD_WIDTH_M), (200, 200, 200), False)
    for b in BUMPS:
        rect(b, (40, 90, 110), True)
    for name, hub in HUBS.items():
        rect(hub, BGR[name], True)
    colours = _palette(n_tracks)
    for (tid,), g in df.group_by("track_id"):
        assert isinstance(tid, int)
        fr = g["frame"].to_numpy()
        pts = px(g["x"].to_numpy(), g["y"].to_numpy())
        for seg in np.split(np.arange(len(fr)), np.flatnonzero(np.diff(fr) > step) + 1):
            cv2.polylines(img, [pts[seg]], False, colours[tid], 1, cv2.LINE_AA)
    cv2.putText(
        img, f"{n_tracks} tracks, top-down (blue wall left)", (PAD, 22), 0, 0.6, (255, 255, 255), 1
    )
    return img


def _timeline_panel(
    df: pl.DataFrame, n_tracks: int, t0: float, t1: float, width: int, match: MatchInfo
) -> NDArray[np.uint8]:
    """One row per track showing when it exists, in its bumper colour; interpolated spans
    white; then the number of live tracks per sampled frame with a line at six."""
    row = 4 if n_tracks > 120 else 7
    left, top = 60, 30
    plot_w = width - left - 20
    chart_h = 160
    h = top + n_tracks * row + 40 + chart_h + 40
    img = np.full((h, width, 3), 24, np.uint8)

    def tx(t: F64) -> NDArray[np.int32]:
        return (left + (t - t0) / (t1 - t0) * plot_w).round().astype(np.int32)

    for tt in np.arange(np.ceil(t0 / 10) * 10, t1, 10):
        x = int(tx(np.array(tt)))
        cv2.line(img, (x, top), (x, h - 40), (60, 60, 60), 1)
        cv2.putText(img, f"{tt:.0f}", (x - 10, h - 20), 0, 0.4, (200, 200, 200), 1)
    for tt in (match.auto_start_t, match.teleop_start_t, match.teleop_end_t):
        x = int(tx(np.array(tt)))
        cv2.line(img, (x, top), (x, h - 40), (0, 200, 255), 1)
    for (tid,), g in df.group_by("track_id"):
        assert isinstance(tid, int)
        y = top + tid * row
        xs = tx(g["t"].to_numpy())
        colour = BGR[str(g["color"][0])]
        interp = (g["state"] == "interpolated").to_numpy()
        for x, it in zip(xs.tolist(), interp.tolist(), strict=True):
            cv2.line(img, (x, y), (x + 1, y), (255, 255, 255) if it else colour, row - 1)
    cv2.putText(
        img,
        "tracks over time (rows; white = interpolated; orange = auto, teleop, end)",
        (left, 20),
        0,
        0.5,
        (255, 255, 255),
        1,
    )
    base = top + n_tracks * row + 40 + chart_h
    scale = chart_h / 10
    for k in range(0, 11, 2):
        yk = int(base - k * scale)
        cv2.line(img, (left, yk), (left + plot_w, yk), (80, 80, 80) if k != 6 else (0, 0, 255), 1)
        cv2.putText(img, str(k), (left - 25, yk + 4), 0, 0.4, (200, 200, 200), 1)
    ts = df.group_by("frame").agg(pl.col("t").first(), pl.len()).sort("frame")
    pts = np.stack(
        [tx(ts["t"].to_numpy()), (base - ts["len"].to_numpy() * scale).round().astype(np.int32)], 1
    )
    cv2.polylines(img, [pts.astype(np.int32)], False, (255, 255, 255), 1)
    cv2.putText(
        img,
        "live tracks per frame (red line = 6)",
        (left, base - chart_h - 8),
        0,
        0.5,
        (255, 255, 255),
        1,
    )
    return img


def _write_check(
    run: RunDir, df: pl.DataFrame, n_tracks: int, match: MatchInfo, info: VideoInfo, step: int
) -> None:
    top = _field_panel(df, n_tracks, step)
    bottom = _timeline_panel(df, n_tracks, 0.0, info.duration_s, top.shape[1], match)
    cv2.imwrite(
        str(run.path / CHECK_IMAGE), np.vstack([top, bottom]), [cv2.IMWRITE_JPEG_QUALITY, 90]
    )


def track(run: RunDir) -> None:
    start = time.perf_counter()
    info = VideoInfo.load(run.video_json)
    match = MatchInfo.load(run.match_json)
    cam = Camera.load(run.calibration_json, "main")
    step = max(1, round(info.fps / SAMPLE_HZ))
    obs, counts = _observations(run, cam)
    raw, cuts = _tracklets(obs)
    pieces = [
        p
        for p in raw
        if len(p.idx) >= MIN_TRACKLET_OBS and np.median(obs.conf[p.idx]) >= MIN_TRACKLET_CONF
    ]
    tracks, joins = _join(obs, pieces)
    tracks, dup_dropped = _drop_duplicates(obs, tracks)
    tracks.sort(key=lambda p: (obs.t[p.idx[0]], obs.xy[p.idx[0], 0]))
    df = _rows(obs, tracks, info.fps, step)
    write_tracks(run.tracks, df)
    _write_check(run, df, len(tracks), match, info, step)

    life = np.array([obs.t[p.idx[-1]] - obs.t[p.idx[0]] for p in tracks])
    live = df.group_by("frame").len()["len"].to_numpy()
    print(
        f"track: {counts['boxes']} boxes, {counts['main_conf']} main >= {CONF_MIN}, "
        f"{counts['placed']} placed ({counts['robot_placed']} whole-robot), "
        f"{counts.get('inside', 0)} inside another colour, {counts['merged']} after merging "
        f"({counts.get('robot_explained', 0)} whole-robot explained by bumpers, "
        f"{counts['robot_only']} whole-robot only)"
    )
    print(
        f"  {len(raw)} tracklets (cut: {cuts['rival']} rival, {cuts['contact']} contact, "
        f"{cuts['kind']} box kind), "
        f"{len(pieces)} kept, {joins} joins, {dup_dropped} duplicate samples dropped, "
        f"{len(tracks)} tracks"
    )
    if len(life):
        print(
            f"  track life s: median {np.median(life):.1f}, p90 {np.percentile(life, 90):.1f}, "
            f"max {life.max():.1f}; live tracks per frame: max {live.max()}, "
            f"frames over 6: {(live > 6).sum()} of {len(live)}"
        )
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"  {len(df)} rows; wall {time.perf_counter() - start:.2f} s, peak RSS {rss:.2f} GiB")


STAGE = Stage(
    "track",
    inputs=lambda r: (r.detections, r.calibration_json, r.match_json, r.video_json),
    outputs=lambda r: (r.tracks,),
    fn=track,
)
