"""corners: video, video.json, match.json, tracks, identity.json, reads -> corner_calibration.json.

The corner panels (CapCity: two 652x344 panels below the main view) show fisheye cameras at
two field corners, outside the guardrail about a metre up, looking across an alliance zone
at its hub. The calibrate stage cannot calibrate them: at that size the field's tags are a
few pixels wide, it decodes at most one per panel, and tags alone leave the camera free to
slide toward the hub while the focal length changes. Robots are the targets here, with the
field's tags holding the camera's height and aim. For each corner panel:

1. Pairs: a corner bumper box with a team read (`reads.parquet`: `team` set, not `static`,
   box clear of the panel edge) and that team's main-view position at the same frame (a
   measured sample of the one track labelled the team from number reads; partial samples
   too, as leaving them out cost a third of the pairs and no accuracy). The
   box centre is on the robot's near bumper face at bumper height; the robot centre is
   `face_m` behind it, away from the camera.
2. Split by time: pairs in even SPLIT_S blocks of the match fit the camera, pairs in odd
   blocks only test it.
3. Start: for each field corner, a camera outside it looking at that end's hub, with focal
   length, k1 and pose fitted to the fit pairs and any decoded tag (principal point at the
   panel centre). The corner whose fit has the lowest robust cost is the start. Every fit
   holds the camera within CORNER_REACH_M of that corner and CAMERA_Z_M up (`_bounds`); a
   camera left on that bound is rejected.
4. Tags: decoded as in the calibrate stage, and undecoded ones located near the camera's
   prediction in each quarter median by a ring template (`_locate`), a whole structure (the
   visible faces of a hub, a wall pair) under one shift. Robots and tags are then fitted
   together with a fisheye lens (focal length, principal point and k1, under weak priors);
   a located tag the fit leaves more than MAX_TAG_PX off was matched wrongly and is dropped.
   Locate again from the better camera, refit.
5. Test: each held-out box centre is cast to the field at bumper height and pushed `face_m`
   away from the camera; its distance to the main-view position is the error in metres. The
   main-view position (pulled `face_m` toward the camera) projected into the panel against
   the box centre is the error in pixels. Both by distance from the camera (`RANGES_M`).
   The trusted distances are the contiguous ranges with at least MIN_RANGE_PAIRS pairs and a
   median within TRUST_MEDIAN_M (`_trusted`).

A view is written only if it had enough pairs, a camera off its bounds, and a trusted span of at
least MIN_TRUSTED_SPAN_M; otherwise it is left out and the reason printed. The stage never
fails for a corner view. `corners_check.jpg` shows the field lines and tag outlines drawn
through each corner camera on the corner panels' median, and the held-out pairs (white: box
centre, red: main-view position projected; green line joining them).

The test shares the main view's own errors with the fit (both are main-view positions);
it measures how well a corner box places a robot where the main view places it.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
import polars as pl
from numpy.typing import NDArray
from pupil_apriltags import Detector  # pyright: ignore[reportMissingTypeStubs]
from scipy.optimize import least_squares  # pyright: ignore[reportMissingTypeStubs]

from scout.camera import Camera
from scout.field_tags import (
    Hits,
    Tag,
    clusters,
    decode_view,
    detector,
    faces,
    merge,
    tag_corners,
    window_medians,
)
from scout.games.rebuilt2026 import (
    ALLIANCE_ZONE_DEPTH_M,
    BUMPER_CENTER_HEIGHT_M,
    BUMPS,
    CENTER_LINE_X_M,
    FIELD_LENGTH_M,
    FIELD_WIDTH_M,
    HUB_CENTERS,
    HUBS,
    Rect,
)
from scout.run import RunDir, Stage
from scout.schemas.calibration import (
    CameraCalibration,
    CornerCalibration,
    CornerCalibrations,
    RangeError,
)
from scout.schemas.common import PixelRect, ViewName
from scout.schemas.identity import Identity, read_reads
from scout.schemas.match import MatchInfo
from scout.schemas.tracks import read_tracks
from scout.schemas.video import VideoInfo

F64 = NDArray[np.float64]
U8 = NDArray[np.uint8]

CORNER_VIEWS: tuple[ViewName, ...] = ("corner_left", "corner_right")

# Pairs.
LABEL_SOURCES = frozenset({"digits", "digits_extended", "votes"})  # number reads, not looks
EDGE_PX = 2.0  # a box this close to the panel edge is cut off, so its centre is not the robot's
SPLIT_S = 15.0  # fit on even blocks, test on odd: a robot moves well clear in 15 s
MIN_FIT_PAIRS = 60
MIN_TEST_PAIRS = 40

# Fit. Robot points scatter about 10 px (box centres of robots at every angle, and the main
# view's own error), tags about a pixel. The priors only keep the lens from wandering where
# neither pins it. A k2 term made no held-out difference on CapCity and let k1 swing from
# -0.8 to 1.6 between matches of one camera; with k1 alone it stays in -0.1 to 1.0.
ROBOT_PX = 8.0
TAG_PX = 1.0
CENTRE_PRIOR_PX = 60.0
K1_PRIOR = 0.5
FACE_BOUNDS_M = (0.0, 0.7)  # 0: the box centre is the robot centre; 0.7: past the corner
# The camera sits at a field corner: within CORNER_REACH_M of it along x and y and
# CAMERA_Z_M up; the fits are bounded there. Unbounded, a floor-only fit can drift to a far,
# long lens that maps the same floor (a CapCity start went 16 m off the field, f 1974 px).
CAMERA_Z_M = (0.3, 4.0)
CORNER_REACH_M = 3.0
FIELD_CORNERS = (
    (0.0, 0.0),
    (0.0, FIELD_WIDTH_M),
    (FIELD_LENGTH_M, 0.0),
    (FIELD_LENGTH_M, FIELD_WIDTH_M),
)
START_OUT_M = 0.6  # start this far outside both field edges at the corner
START_Z_M = 1.2
START_F = 0.4  # start focal length, in panel widths
START_FACE_M = 0.4

# Locating tags (`_locate`). The white border runs from the black square (0.1651 m) out to
# 0.2064 m; the template is that ring against the square inside it and a band outside it.
LOCATE_RADIUS_PX = 30.0  # the start is 15-20 px off on CapCity
OUTER = 0.2064 / 0.1651
SURROUND = 1.25  # outside band, as a multiple of the outer square
UP = 4  # template matching on a 4x upscale for sub-pixel centres
STRUCTURE_M = 1.0  # tags this close form one structure (both visible faces of a hub)
MIN_SIDE_PX = 3.0
FLAT_VAR = 25.0  # grey-level variance under which a patch carries no evidence
UNIQUE = 0.8  # no other shift may score this close to the best one
MIN_NCC = 0.3  # a tag counts at the structure's shift only with this correlation
MAX_TAG_PX = 6.0
LOCATE_ROUNDS = 2

# Test and trust.
# A held-out box the camera cannot cast to the field (its ray misses the floor) counts as
# this far off, so it lifts the percentiles as a failure should.
CAST_FAILED_M = 100.0
RANGES_M = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 13.0, 20.0)
MIN_RANGE_PAIRS = 20
TRUST_MEDIAN_M = 0.3
MIN_TRUSTED_SPAN_M = 2.0

CHECK_IMAGE = "corners_check.jpg"
RAIL_Z_M = 0.487  # guardrail top tube (field_wireframe asset)


@dataclass(frozen=True)
class Pairs:
    """Robot pairs of one corner view: box centres (N, 2) full-frame px, main-view robot
    centres (N, 2) field metres, and whether each is held out for the test."""

    uv: F64
    xy: F64
    test: NDArray[np.bool_]


@dataclass(frozen=True)
class Lens:
    """Fisheye camera parameters as one vector: rvec (3), camera centre (3, field metres),
    f, cx, cy, k1, face."""

    x: F64

    @property
    def k(self) -> F64:
        f, cx, cy = self.x[6:9]
        return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])

    @property
    def dist(self) -> F64:
        return np.array([self.x[9], 0.0, 0.0, 0.0])

    @property
    def face(self) -> float:
        return float(self.x[10])

    @property
    def centre(self) -> F64:
        return self.x[3:6].copy()

    @property
    def tvec(self) -> F64:
        r, _ = cv2.Rodrigues(self.x[:3])
        return -np.asarray(r, np.float64) @ self.x[3:6]

    def project(self, xyz: F64) -> F64:
        uv, _ = cv2.fisheye.projectPoints(
            xyz.reshape(-1, 1, 3), self.x[:3], self.tvec, self.k, self.dist
        )
        return np.asarray(uv, np.float64).reshape(-1, 2)

    def bumper_points(self, xy: F64) -> F64:
        """Field points (N, 3) of the box centres of robots centred at `xy`: `face` toward
        the camera, at bumper height."""
        d = self.centre[:2] - xy
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return np.c_[xy + self.face * d, np.full(len(xy), BUMPER_CENTER_HEIGHT_M)]

    def schema(self, tags: dict[int, Tag]) -> CameraCalibration:
        err = self._tag_err(tags)
        ids = sorted(tags)
        pts = np.array([tags[t].obj.mean(0)[:2] for t in ids]).reshape(-1, 2)
        spread = float(np.max(np.linalg.norm(pts[:, None] - pts[None], axis=2))) if ids else 0.0
        return CameraCalibration(
            k=[[float(v) for v in row] for row in self.k],
            dist=[float(v) for v in self.dist],
            rvec=[float(v) for v in self.x[:3]],
            tvec=[float(v) for v in self.tvec],
            rms_px=round(float(np.sqrt(np.mean(err**2))), 3) if len(err) else 0.0,
            max_px=round(float(err.max()), 3) if len(err) else 0.0,
            n_points=len(err),
            model="c_kb3",
            tags=[t for t in ids if not tags[t].located],
            located=[t for t in ids if tags[t].located],
            spread_m=round(spread, 3),
            holdout_m={},
            camera_xyz=[round(float(v), 3) for v in self.centre],
        )

    def camera(self) -> Camera:
        return Camera.from_calibration(self.schema({}))

    def _tag_err(self, tags: dict[int, Tag]) -> F64:
        if not tags:
            return np.zeros(0)
        obj = np.concatenate([t.obj for t in tags.values()])
        img = np.concatenate([t.img for t in tags.values()])
        return np.linalg.norm(self.project(obj) - img, axis=1)


# --- pairs ---------------------------------------------------------------------


def _pairs(run: RunDir, view: ViewName, rect: PixelRect) -> Pairs:
    """Corner boxes read as a team, joined with that team's main-view position (step 1)."""
    ident = Identity.load(run.identity_json)
    segs = pl.DataFrame(
        [
            {"track_id": s.track_id, "lo": s.frame_start, "hi": s.frame_end, "team": s.team}
            for s in ident.segments
            if s.team is not None and s.source in LABEL_SOURCES
        ],
        schema={"track_id": pl.Int32(), "lo": pl.Int64(), "hi": pl.Int64(), "team": pl.Int32()},
    )
    main = (
        read_tracks(run.tracks)
        .filter(pl.col("state") == "measured")
        .join(segs, on="track_id")
        .filter(pl.col("frame").is_between(pl.col("lo"), pl.col("hi")))
        .group_by("frame", "team")
        .agg(pl.col("x").first(), pl.col("y").first(), pl.len().alias("n"))
        .filter(pl.col("n") == 1)
    )
    e = EDGE_PX
    boxes = (
        read_reads(run.reads)
        .filter((pl.col("view") == view) & pl.col("team").is_not_null() & ~pl.col("static"))
        .filter(
            (pl.col("x1") > rect.x + e)
            & (pl.col("x2") < rect.x + rect.w - e)
            & (pl.col("y1") > rect.y + e)
            & (pl.col("y2") < rect.y + rect.h - e)
        )
        .group_by("frame", "team")
        .agg(pl.col("t").first(), pl.col("x1", "y1", "x2", "y2").first(), pl.len().alias("n"))
        .filter(pl.col("n") == 1)  # one team read on two boxes: neither is sure
    )
    p = boxes.join(main, on=["frame", "team"]).sort("frame", "team")
    uv = np.c_[
        (p["x1"].to_numpy() + p["x2"].to_numpy()) / 2, (p["y1"].to_numpy() + p["y2"].to_numpy()) / 2
    ].astype(np.float64)
    xy = p.select("x", "y").to_numpy().astype(np.float64)
    test = (np.floor(p["t"].to_numpy() / SPLIT_S) % 2) == 1
    return Pairs(uv, xy, test)


# --- fitting -------------------------------------------------------------------


def _look(c: F64, target: F64) -> F64:
    """Rodrigues vector of a level camera at `c` looking at `target`."""
    f = target - c
    f /= np.linalg.norm(f)
    right = np.cross(f, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    r = np.stack([right, np.cross(f, right), f])
    rv, _ = cv2.Rodrigues(r)
    return np.asarray(rv, np.float64).reshape(3)


def _bounds(corner: tuple[float, float]) -> tuple[F64, F64]:
    """Parameter bounds: the camera centre near `corner`, the face offset in FACE_BOUNDS_M."""
    r = CORNER_REACH_M
    lo = np.full(11, -np.inf)
    hi = np.full(11, np.inf)
    lo[3:6] = (corner[0] - r, corner[1] - r, CAMERA_Z_M[0])
    hi[3:6] = (corner[0] + r, corner[1] + r, CAMERA_Z_M[1])
    lo[10], hi[10] = FACE_BOUNDS_M
    return lo, hi


def _at_bound(lens: Lens, corner: tuple[float, float]) -> bool:
    """The fitted camera centre sits on its bound: the fit wanted it elsewhere."""
    lo, hi = _bounds(corner)
    c = lens.centre
    return bool(np.any(np.isclose(c, lo[3:6], atol=0.01) | np.isclose(c, hi[3:6], atol=0.01)))


Corner = tuple[float, float]


def _start(pairs: Pairs, decoded: dict[int, Tag], rect: PixelRect) -> tuple[Lens, Corner]:
    """Step 3: the best of four corner starts, fitted to the fit pairs and decoded tags with
    the principal point held at the panel centre and k2 zero; with the corner it is at."""
    fit = ~pairs.test
    obj = np.concatenate([t.obj for t in decoded.values()]) if decoded else np.zeros((0, 3))
    img = np.concatenate([t.img for t in decoded.values()]) if decoded else np.zeros((0, 2))
    pc = (rect.x + rect.w / 2, rect.y + rect.h / 2)
    best: tuple[float, Lens, Corner] | None = None
    for corner in FIELD_CORNERS:
        out_x = -START_OUT_M if corner[0] == 0.0 else START_OUT_M
        out_y = -START_OUT_M if corner[1] == 0.0 else START_OUT_M
        c = np.array([corner[0] + out_x, corner[1] + out_y, START_Z_M])
        hub = np.array([*HUB_CENTERS["blue" if corner[0] == 0.0 else "red"], 0.5])

        def full(s: F64) -> Lens:
            return Lens(np.r_[s[:7], pc, s[7], START_FACE_M])

        def resid(s: F64) -> F64:
            lens = full(s)
            e = lens.project(lens.bumper_points(pairs.xy[fit])) - pairs.uv[fit]
            et = (lens.project(obj) - img) if len(obj) else np.zeros((0, 2))
            return np.r_[e.reshape(-1) / ROBOT_PX, et.reshape(-1) / TAG_PX]

        lo, hi = _bounds(corner)
        sol = least_squares(
            resid,
            np.r_[_look(c, hub), c, START_F * rect.w, 0.0],
            loss="soft_l1",
            bounds=(np.r_[lo[:6], 1.0, -np.inf], np.r_[hi[:6], np.inf, np.inf]),
        )
        if best is None or float(sol.cost) < best[0]:
            best = (float(sol.cost), full(np.asarray(sol.x, np.float64)), corner)
    assert best is not None
    return best[1], best[2]


def _fit(pairs: Pairs, tags: dict[int, Tag], start: Lens, corner: Corner, rect: PixelRect) -> Lens:
    """Robots (fit pairs) and tags together, fisheye lens under the priors (step 4)."""
    fit = ~pairs.test
    pc = np.array([rect.x + rect.w / 2, rect.y + rect.h / 2])
    obj = np.concatenate([t.obj for t in tags.values()]) if tags else np.zeros((0, 3))
    img = np.concatenate([t.img for t in tags.values()]) if tags else np.zeros((0, 2))

    def resid(x: F64) -> F64:
        lens = Lens(x)
        e = lens.project(lens.bumper_points(pairs.xy[fit])) - pairs.uv[fit]
        et = (lens.project(obj) - img) if len(obj) else np.zeros((0, 2))
        prior = np.r_[(x[7:9] - pc) / CENTRE_PRIOR_PX, x[9] / K1_PRIOR]
        return np.r_[e.reshape(-1) / ROBOT_PX, et.reshape(-1) / TAG_PX, prior]

    lo, hi = _bounds(corner)
    x0 = np.clip(start.x, lo + 1e-6, hi - 1e-6)
    sol = least_squares(resid, x0, loss="soft_l1", bounds=(lo, hi))
    return Lens(np.asarray(sol.x, np.float64))


def _fit_tags(
    pairs: Pairs,
    decoded: dict[int, Tag],
    located: dict[int, Tag],
    start: Lens,
    corner: Corner,
    rect: PixelRect,
) -> tuple[Lens, dict[int, Tag]]:
    """`_fit` with every tag, dropping the located tag worst off the fit while it is past
    MAX_TAG_PX (matched to the wrong frame). Decoded tags always stay."""
    located = dict(located)
    while True:
        tags = decoded | located
        lens = _fit(pairs, tags, start, corner, rect)
        err = {
            t: float(np.max(np.linalg.norm(lens.project(tags[t].obj) - tags[t].img, axis=1)))
            for t in located
        }
        worst = max(err, key=lambda t: err[t], default=None)
        if worst is None or err[worst] <= MAX_TAG_PX:
            return lens, tags
        del located[worst]


# --- locating tags ---------------------------------------------------------------


def _template(cam: Camera, c3: F64) -> tuple[F64, F64, F64] | None:
    """Projected black square, white border and surround (each (4, 2) px) of one tag."""
    ctr = c3.mean(0)
    quads = [cam.project(ctr + (c3 - ctr) * s) for s in (1.0, OUTER, OUTER * SURROUND)]
    if not all(np.isfinite(q).all() for q in quads):
        return None
    return quads[0], quads[1], quads[2]


def _ncc_map(med: U8, quads: tuple[F64, F64, F64], radius: float) -> F64 | None:
    """Normalised correlation of the panel with one tag's template (white border against
    the darker square and surround) for every shift within `radius` px of the prediction,
    on a UP-times grid: shape (2R+1, 2R+1) with R = radius * UP, rows = y shift."""
    inner, outer, sur = quads
    pc = inner.mean(0)
    half = int(np.ceil(np.abs(sur - pc).max())) + 2
    r = int(np.ceil(radius))
    x0, y0 = int(pc[0]) - r - half, int(pc[1]) - r - half
    x1, y1 = int(pc[0]) + r + half + 2, int(pc[1]) + r + half + 2
    if x0 < 0 or y0 < 0 or y1 > med.shape[0] or x1 > med.shape[1]:
        return None
    crop = med[y0:y1, x0:x1].astype(np.float32)
    img = np.asarray(
        cv2.resize(crop, None, fx=UP, fy=UP, interpolation=cv2.INTER_CUBIC), np.float32
    )
    size = 2 * half * UP + 1
    org = np.array([half * UP, half * UP], np.float64)
    masks: list[U8] = []
    for q in (inner, outer, sur):
        m = np.zeros((size, size), np.uint8)
        cv2.fillConvexPoly(m, np.round((q - pc) * UP + org).astype(np.int32), 1)
        masks.append(m)
    ring = (masks[1] > 0) & (masks[0] == 0)
    inside = cv2.erode(masks[0], np.ones((3, 3), np.uint8)) > 0
    band = (masks[2] > 0) & (masks[1] == 0)
    if ring.sum() < 4 or inside.sum() < 4:
        return None
    m_all = (ring | inside | band).astype(np.float32)
    t = ring.astype(np.float32)
    n = float(m_all.sum())
    mt = float(t.sum()) / n

    def corr(a: NDArray[np.float32], k: NDArray[np.float32]) -> NDArray[np.float32]:
        out = cv2.filter2D(a, -1, k, borderType=cv2.BORDER_REPLICATE)
        return np.asarray(out, np.float32)

    s1 = corr(img, m_all) / n
    s2 = corr(img * img, m_all) / n
    st = corr(img, t) / n
    var = s2 - s1 * s1
    ncc = (st - s1 * mt) / np.sqrt(np.maximum(var, 1e-3) * mt * (1 - mt))
    ncc = np.where(var > FLAT_VAR, ncc, 0.0)
    cx, cy = (pc[0] - x0) * UP, (pc[1] - y0) * UP
    steps = np.arange(-r * UP, r * UP + 1)
    ys = np.clip(np.round(cy + steps).astype(int), 0, ncc.shape[0] - 1)
    xs = np.clip(np.round(cx + steps).astype(int), 0, ncc.shape[1] - 1)
    return np.asarray(ncc[np.ix_(ys, xs)], np.float64)


def _locate(
    medians: list[U8], lens: Lens, corners: dict[int, F64], rect: PixelRect, skip: set[int]
) -> Hits:
    """Centres of undecoded camera-facing tags in each window median. A structure (tags
    within STRUCTURE_M) matches under the one shift within LOCATE_RADIUS_PX where its mean
    correlation is highest, if no shift farther than half a tag spacing scores within UNIQUE
    of it; each of its tags with correlation MIN_NCC there is then placed at its own best
    shift within 2 px."""
    cam = lens.camera()
    ids: list[int] = []
    for t, c3 in corners.items():
        uv = cam.project(c3)
        if t in skip or not np.isfinite(uv).all() or not faces(c3, cam):
            continue
        ctr = uv.mean(0)
        side = float(np.mean(np.linalg.norm(uv - np.roll(uv, 1, 0), axis=1)))
        inside = (
            rect.x + 4 <= ctr[0] < rect.x + rect.w - 4
            and rect.y + 4 <= ctr[1] < rect.y + rect.h - 4
        )
        if side >= MIN_SIDE_PX and inside:
            ids.append(t)
    hits: Hits = {}
    for group in clusters(ids, corners, STRUCTURE_M, same_facing=False):
        quads = {t: q for t in group if (q := _template(cam, corners[t])) is not None}
        if not quads:
            continue
        centres = np.array([q[0].mean(0) for q in quads.values()])
        gaps = [float(np.linalg.norm(a - b)) for i, a in enumerate(centres) for b in centres[:i]]
        apart = max(3.0, 0.5 * min(gaps)) * UP if gaps else 4.0 * UP
        for w, med in enumerate(medians):
            maps = {
                t: m
                for t, q in quads.items()
                if (m := _ncc_map(med, q, LOCATE_RADIUS_PX)) is not None
            }
            if not maps:
                continue
            score = cast(F64, sum(maps.values())) / len(maps)
            i, j = np.unravel_index(int(np.argmax(score)), score.shape)
            yy, xx = np.mgrid[: score.shape[0], : score.shape[1]]
            far = np.hypot(yy - i, xx - j) > apart
            best = float(score[i, j])
            if best <= 0 or (far.any() and float(score[far].max()) > UNIQUE * best):
                continue
            n = score.shape[0] // 2
            for t, m in maps.items():
                if m[i, j] < MIN_NCC:
                    continue
                r = 2 * UP
                a0, b0 = max(0, i - r), max(0, j - r)
                sub = m[a0 : i + r + 1, b0 : j + r + 1]
                a, b = np.unravel_index(int(np.argmax(sub)), sub.shape)
                shift = np.array([b0 + b - n, a0 + a - n], np.float64) / UP
                hits.setdefault(t, []).append((w, (quads[t][0].mean(0) + shift)[None]))
    return hits


# --- testing -------------------------------------------------------------------


def _errors(lens: Lens, pairs: Pairs) -> tuple[F64, F64, F64]:
    """Held-out error per test pair: metres, pixels, and distance from the camera."""
    m = pairs.test
    cam = lens.camera()
    c = lens.centre[:2]
    cast_xy = cam.to_field(pairs.uv[m], BUMPER_CENTER_HEIGHT_M)
    away = cast_xy - c
    away /= np.linalg.norm(away, axis=1, keepdims=True)
    metres = np.linalg.norm(cast_xy + lens.face * away - pairs.xy[m], axis=1)
    pixels = np.linalg.norm(lens.project(lens.bumper_points(pairs.xy[m])) - pairs.uv[m], axis=1)
    dist = np.linalg.norm(pairs.xy[m] - c, axis=1)
    return metres, pixels, dist


def _ranges(metres: F64, pixels: F64, dist: F64) -> list[RangeError]:
    out: list[RangeError] = []
    for near, far in itertools.pairwise(RANGES_M):
        b = (dist >= near) & (dist < far)
        m, p = metres[b], pixels[b]
        m = np.where(np.isfinite(m), m, CAST_FAILED_M)

        def q(a: F64, pct: float) -> float | None:
            return round(float(np.percentile(a, pct)), 3) if len(a) else None

        out.append(
            RangeError(
                near_m=near,
                far_m=far,
                n=int(b.sum()),
                median_m=q(m, 50),
                p90_m=q(m, 90),
                median_px=q(p, 50),
                p90_px=q(p, 90),
            )
        )
    return out


def _trusted(ranges: list[RangeError]) -> tuple[float, float]:
    """The longest run of contiguous ranges, each with MIN_RANGE_PAIRS pairs and a median
    within TRUST_MEDIAN_M (near, far metres; (0, 0) when none)."""
    best = (0.0, 0.0)
    start: float | None = None
    for r in ranges:
        ok = r.n >= MIN_RANGE_PAIRS and r.median_m is not None and r.median_m <= TRUST_MEDIAN_M
        if ok:
            start = r.near_m if start is None else start
            if r.far_m - start > best[1] - best[0]:
                best = (start, r.far_m)
        else:
            start = None
    return best


# --- the view --------------------------------------------------------------------


@dataclass(frozen=True)
class ViewResult:
    lens: Lens | None
    tags: dict[int, Tag]
    pairs: Pairs | None
    result: CornerCalibration | None
    reason: str | None


def _calibrate_view(
    run: RunDir,
    view: ViewName,
    rect: PixelRect,
    medians: list[U8],
    corners: dict[int, F64],
    det: Detector,
) -> ViewResult:
    pairs = _pairs(run, view, rect)
    n_fit, n_test = int((~pairs.test).sum()), int(pairs.test.sum())
    if n_fit < MIN_FIT_PAIRS or n_test < MIN_TEST_PAIRS:
        need = f"{MIN_FIT_PAIRS} and {MIN_TEST_PAIRS}"
        why = f"only {n_fit} fit and {n_test} test robot pairs (need {need})"
        return ViewResult(None, {}, pairs, None, why)
    hits = decode_view(det, medians, rect)
    decoded, _ = merge(hits, corners, located=False)
    centres = {t: c.mean(0)[None] for t, c in corners.items()}
    lens, corner = _start(pairs, decoded, rect)
    located: dict[int, Tag] = {}
    tags: dict[int, Tag] = dict(decoded)
    for _ in range(LOCATE_ROUNDS):
        found, _ = merge(_locate(medians, lens, corners, rect, set(hits)), centres, located=True)
        lens, tags = _fit_tags(pairs, decoded, located | found, lens, corner, rect)
        located = {t: v for t, v in tags.items() if v.located}
    metres, pixels, dist = _errors(lens, pairs)
    ranges = _ranges(metres, pixels, dist)
    trusted = _trusted(ranges)
    if _at_bound(lens, corner):
        c = lens.centre
        why = f"camera at ({c[0]:.1f}, {c[1]:.1f}, {c[2]:.1f}) m is not at a field corner"
        return ViewResult(lens, tags, pairs, None, why)
    if trusted[1] - trusted[0] < MIN_TRUSTED_SPAN_M:
        why = (
            f"no {MIN_TRUSTED_SPAN_M:g} m of distances with a held-out median within "
            f"{TRUST_MEDIAN_M:g} m"
        )
        return ViewResult(lens, tags, pairs, None, why)
    result = CornerCalibration(
        camera=lens.schema(tags),
        face_m=round(lens.face, 3),
        fit_pairs=n_fit,
        test_pairs=n_test,
        split_s=SPLIT_S,
        test=ranges,
        trusted_m=trusted,
    )
    return ViewResult(lens, tags, pairs, result, None)


def _report(view: ViewName, res: ViewResult) -> str:
    head = f"{view}: " + ("ok" if res.result else f"not calibrated ({res.reason})")
    if res.lens is None or res.pairs is None:
        return head
    lens, c = res.lens, res.lens.centre
    located = sorted(t for t, v in res.tags.items() if v.located)
    decoded = sorted(t for t, v in res.tags.items() if not v.located)
    lines = [
        head,
        f"  camera ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}) m, f {lens.x[6]:.0f}, "
        f"c ({lens.x[7]:.0f}, {lens.x[8]:.0f}), "
        f"k1 {lens.x[9]:.3f}, face {lens.face:.2f} m; tags decoded {decoded} located {located}",
        f"  pairs fit {int((~res.pairs.test).sum())} test {int(res.pairs.test.sum())}",
    ]
    metres, pixels, dist = _errors(lens, res.pairs)
    for r in _ranges(metres, pixels, dist):
        if r.n:
            lines.append(
                f"    {r.near_m:4.0f}-{r.far_m:<4.0f} m  n {r.n:4d}  median {r.median_m:.2f} m "
                f"p90 {r.p90_m:.2f} m  median {r.median_px:.1f} px p90 {r.p90_px:.1f} px"
            )
    if res.result:
        lines.append(
            f"  trusted {res.result.trusted_m[0]:g}-{res.result.trusted_m[1]:g} m from the camera"
        )
    return "\n".join(lines)


# --- the check image ---------------------------------------------------------------

Colour = tuple[int, int, int]
Point3 = tuple[float, float, float]


def _rect_edges(r: Rect, z: float) -> list[tuple[Point3, Point3]]:
    c = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]
    return [((*c[i], z), (*c[(i + 1) % 4], z)) for i in range(4)]


def _check_lines() -> list[tuple[Colour, Point3, Point3]]:
    """(colour, start, end) of the field lines drawn in the check image."""
    field = Rect(0, 0, FIELD_LENGTH_M, FIELD_WIDTH_M)
    out = [((230, 230, 0), a, b) for a, b in _rect_edges(field, 0.0)]
    out += [((230, 230, 0), a, b) for a, b in _rect_edges(field, RAIL_Z_M)]
    mid, blue_x, red_x = (
        CENTER_LINE_X_M,
        ALLIANCE_ZONE_DEPTH_M,
        FIELD_LENGTH_M - ALLIANCE_ZONE_DEPTH_M,
    )
    out.append(((220, 60, 220), (mid, 0.0, 0.0), (mid, FIELD_WIDTH_M, 0.0)))
    out.append(((230, 120, 30), (blue_x, 0.0, 0.0), (blue_x, FIELD_WIDTH_M, 0.0)))
    out.append(((40, 40, 230), (red_x, 0.0, 0.0), (red_x, FIELD_WIDTH_M, 0.0)))
    out += [((0, 230, 230), a, b) for r in BUMPS for a, b in _rect_edges(r, 0.0)]
    out += [((0, 140, 255), a, b) for r in HUBS.values() for a, b in _rect_edges(r, 0.0)]
    return out


def _draw(img: U8, rect: PixelRect, res: ViewResult, corners: dict[int, F64]) -> None:
    """Field lines and tag outlines (green fitted, orange not) through the view's camera,
    measured tag points (white) and the held-out pairs, clipped to the panel."""
    canvas = img[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w]
    off = np.array([rect.x, rect.y], np.float64)
    if res.lens is None:
        return
    cam = res.lens.camera()

    def poly(uv: F64, colour: Colour, closed: bool) -> None:
        good = np.isfinite(uv).all(1)
        for idx in np.split(np.arange(len(uv)), np.flatnonzero(np.diff(good.astype(np.int8))) + 1):
            if len(idx) >= 2 and good[idx[0]]:
                pts = np.round((uv[idx] - off) * 16).astype(np.int32)
                cv2.polylines(
                    canvas, [pts], closed and len(idx) == len(uv), colour, 1, cv2.LINE_AA, 4
                )

    for colour, a, b in _check_lines():
        n = max(2, int(np.linalg.norm(np.subtract(b, a)) / 0.03))
        poly(cam.project(np.linspace(a, b, n)), colour, False)
    for tid, c3 in corners.items():
        uv = cam.project(c3)
        ctr = uv.mean(0) - off
        if (
            not np.isfinite(uv).all()
            or not (0 <= ctr[0] < rect.w and 0 <= ctr[1] < rect.h)
            or not faces(c3, cam)
        ):
            continue
        colour = (60, 220, 60) if tid in res.tags else (0, 140, 255)
        poly(uv, colour, True)
        cv2.putText(
            canvas,
            str(tid),
            (int(ctr[0]) + 5, int(ctr[1]) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            colour,
            1,
            cv2.LINE_AA,
        )
    for tag in res.tags.values():
        for p in tag.img - off:
            cv2.circle(canvas, (round(p[0]), round(p[1])), 1, (255, 255, 255), -1)
    if res.pairs is not None:
        m = res.pairs.test
        proj = res.lens.project(res.lens.bumper_points(res.pairs.xy[m]))
        for a, b in zip(res.pairs.uv[m][::3] - off, proj[::3] - off, strict=True):
            if np.isfinite(b).all():
                pa, pb = (round(a[0]), round(a[1])), (round(b[0]), round(b[1]))
                cv2.line(canvas, pa, pb, (60, 200, 60), 1, cv2.LINE_AA)
                cv2.circle(canvas, pa, 2, (255, 255, 255), -1)
                cv2.circle(canvas, pb, 2, (40, 40, 230), -1)


def calibrate_corners(run: RunDir) -> None:
    info = VideoInfo.load(run.video_json)
    match = MatchInfo.load(run.match_json)
    views: dict[ViewName, PixelRect] = {v: match.views[v] for v in CORNER_VIEWS if v in match.views}
    out: dict[ViewName, CornerCalibration] = {}
    rejected: dict[ViewName, str] = {}
    results: dict[ViewName, ViewResult] = {}
    if views:
        quarters, median = window_medians(
            run.video, info, views.values(), match.auto_start_t, match.teleop_end_t
        )
        field = tag_corners()
        det = detector()  # one for the stage, as in calibrate
        for view, rect in views.items():
            res = _calibrate_view(run, view, rect, quarters, field, det)
            print("corners: " + _report(view, res))
            results[view] = res
            if res.result is not None:
                out[view] = res.result
            else:
                rejected[view] = res.reason or "unknown"
        img = cast(U8, cv2.cvtColor(median, cv2.COLOR_GRAY2BGR))
        for view, res in results.items():
            _draw(img, views[view], res, field)
        y0 = min(r.y for r in views.values())
        y1 = max(r.y + r.h for r in views.values())
        band = img[y0:y1]
        cv2.imwrite(
            str(run.path / CHECK_IMAGE),
            cv2.resize(band, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC),
            [cv2.IMWRITE_JPEG_QUALITY, 90],
        )
    else:
        print("corners: no corner views in match.json")
        cv2.imwrite(str(run.path / CHECK_IMAGE), np.zeros((8, 8, 3), np.uint8))
    CornerCalibrations(views=out, rejected=rejected).save(run.corner_calibration_json)


STAGE = Stage(
    "corners",
    inputs=lambda r: (r.video, r.video_json, r.match_json, r.tracks, r.identity_json, r.reads),
    outputs=lambda r: (r.corner_calibration_json, r.path / CHECK_IMAGE),
    fn=calibrate_corners,
)
