"""calibrate: video, video.json, match.json -> calibration.json and calibration_check.jpg.

The field's AprilTags are the targets, helped in the main view by straight field lines.
For each camera panel in `match.json`:

1. Decode the match (auto start to teleop end) once at 0.5 Hz, grey, and take the median
   of each quarter. Robots and game pieces move, so each median shows the bare field.
   The optical centre is expected at the frame centre for a panel that is the top of a
   full-frame picture (official broadcasts) and at the panel centre otherwise (`View`).
2. Decode tag36h11 in each quarter median at 2x, 3x and 4x, without decimation. A tag
   is trusted only if it decodes in at least 2 of the 4 quarters (so it was in view for
   a good share of the match) with corners that agree; its corners are the median of
   every hit. Tags seen in fewer quarters are reported as suspect and not used.
3. Tags only: fit a ladder of lens models, most constrained first (`LensModel`). A freer
   model needs enough tags, a field spread of at least 2 m, a plausible pose and a lens
   that reaches every pixel of the panel, and it is kept only if it is clearly better
   (`_freer_wins`).
4. Locate undecoded tags. Most tags are too small to decode (about 1.5 px per data bit)
   but their white frames are plain. For each structure of camera-facing tags (a pair
   on a wall or hub face, or a single tag) the fit predicts the tag centres; the frames
   found near them must match the whole structure under one shift, and exactly one shift
   may fit, so a neighbouring tag cannot be taken for the predicted one. A located tag
   gives one point, its centre, and must be found in 2 of the 4 quarters. A located tag
   the refit cannot bring within `MAX_RESID_PX` was matched to the wrong frame and is
   dropped. Refit, locate again with the better prediction, refit.
5. Main view, tags and lines (`_line_path`): far, high cameras decode a few tags in two
   rows, which cannot separate focal length from distortion. The centre tape and the two
   guardrails (`FIELD_LINES`) are straight in 3D and cross the whole picture, so their
   image curves pin the distortion where tags are missing. Starting cameras are the
   tags-only fit and a grid of (k1, k2) with focal length and pose solved from the tags;
   the one whose projected lines lie best on bright ridges of the median starts an
   alternation of finding ridge points near each projected line (in a shrinking search
   band) and refitting tags and lines together. Tags are located as in step 4.
6. Validate by leaving each group of neighbouring tags out (tags on one structure are
   grouped, or the held-out tag would be pinned by its neighbour), refitting (with the
   lines, if any), casting its points onto the plane at their own height, and measuring
   the error in metres. Of the main-view fits that pass every gate, the one with the
   lowest held-out error is kept.

The main view must pass every gate or the stage raises `NeedsInput`. A corner view that
cannot be calibrated is left out. `calibration_check.jpg` shows the field lines and tag
outlines drawn through each calibration on the full-match median image, and the ridge
points the main fit used (red).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray
from pupil_apriltags import Detection, Detector  # pyright: ignore[reportMissingTypeStubs]
from pydantic import BaseModel, ConfigDict
from scipy.optimize import least_squares  # pyright: ignore[reportMissingTypeStubs]

from scout.assets import asset
from scout.camera import Camera, reach_radius
from scout.games.rebuilt2026 import (
    ALLIANCE_ZONE_DEPTH_M,
    BUMPS,
    CENTER_LINE_X_M,
    FIELD_LENGTH_M,
    FIELD_WIDTH_M,
    HUBS,
    Rect,
)
from scout.run import NeedsInput, RunDir, Stage
from scout.schemas.calibration import Calibration, CameraCalibration, LensModel, LineFit
from scout.schemas.common import VIEW_NAMES, PixelRect, ViewName
from scout.schemas.video import VideoInfo
from scout.video import THREADS, frames

F64 = NDArray[np.float64]
U8 = NDArray[np.uint8]

SAMPLE_HZ = 0.5
WINDOWS = 4
MIN_WINDOWS = 2
SCALES = (2, 3, 4)
MIN_MARGIN = 15.0
AGREE_PX = 2.0
TAG_SIZE_M = 0.1651  # black square side (6.5 in); fits rule out the 0.2064 m outer size
GROUP_M = 1.0
STRUCTURE_M = 0.6
# A wider search (60 px) found Hopper's far tags but also took neighbouring frames for
# them (tag 5 on tag 8's frame, 15 on 16's), so the radius stays at 30 px.
LOCATE_RADIUS_PX = 30.0
LOCATE_ROUNDS = 2
MIN_SPREAD_M = 2.0
FREER_GAIN = 0.8
HOLDOUT_CAP_M = 1.0

LADDER: tuple[LensModel, ...] = ("pp_k1", "cy_k1", "c_k1", "c_k2")
FREE: dict[LensModel, tuple[str, ...]] = {
    "pp_k1": ("f", "k1"),
    "cy_k1": ("f", "cy", "k1"),
    "c_k1": ("f", "cx", "cy", "k1"),
    "c_k2": ("f", "cx", "cy", "k1", "k2"),
    "pp_k2": ("f", "k1", "k2"),
    "c_k3": ("f", "cx", "cy", "k1", "k2", "k3"),
}
DIST_TERMS = ("k1", "k2", "p1", "p2", "k3")
MIN_TAGS: dict[LensModel, int] = {
    "pp_k1": 4,
    "cy_k1": 5,
    "c_k1": 6,
    "c_k2": 8,
    "pp_k2": 4,
    "c_k3": 4,
}
COVER_MARGIN = 1.02  # the lens must reach 2% past the panel's farthest corner

# Field lines (main view). Guardrail top: the field_wireframe asset's perimeter tubes at
# z 0.487 m, 0.058-0.061 m outside the field edge, beside wall tops at 0.483 m, 0.007 m
# outside. Which of these shows as the brightest ridge is not known, so the rail's height
# is fitted within RAIL_HEIGHT_FREE_M (a line parallel to x seen from one point fixes
# only the plane through the camera, so one offset is all the image can tell).
RAIL_Z_M = 0.487
RAIL_OUT_M = 0.035
RAIL_HEIGHT_FREE_M = 0.15
LINE_END_M = 0.15  # stay this far from the corners, where rails bend and posts stand
LINE_MARGIN_PX = 12  # ridge points stay this far inside the panel and below the scoreboard
LINE_STEP_PX = 12.0  # spacing of ridge points along a line's image curve
RIDGE_PX = 3.0  # a ridge point is brighter than both sides this far off it
MIN_RIDGE = 4.0  # grey levels a ridge must stand above its sides
MIN_LINE_POINTS = 10
LINE_LOSS_PX = 1.5  # soft-L1 knee: a ridge point is good to about a pixel
LINE_CAP_PX = 10.0  # a ridge point's residual is capped here in the reported RMS
LINE_INLIER_PX = 3.0
# Search bands (px) and models of the alternation; the principal point is held for the
# wide bands, where a ridge point may still be the wrong one.
ICP_ROUNDS: tuple[tuple[float, LensModel], ...] = (
    (24.0, "pp_k2"),
    (12.0, "pp_k2"),
    (6.0, "c_k2"),
    (4.0, "c_k2"),
)
# After each locate round. k3 is free here only to predict the far ends well enough for
# their tags to fall inside LOCATE_RADIUS_PX (Hopper's tags 15 and 16 are 35-50 px off a
# k2 lens); every located tag must then fit the kept model, or the gates fail.
LOCATE_ICP_ROUNDS: tuple[tuple[float, LensModel], ...] = ((6.0, "c_k3"), (4.0, "c_k3"))
GRID_K1 = tuple(round(-0.6 + 0.05 * i, 2) for i in range(13))
GRID_K2 = (0.0, 0.1, 0.2, 0.3)
GRID_MAX_RMS_PX = 4.0  # a grid start must still fit the decoded tags roughly
SCORE_PX = {False: 3.0, True: 8.0}  # start scoring band: fixed line, free-height rail

# Trust gates. A view failing any of these is not written.
# Tag reprojection. Good fits reach 0.4-1.7 px on the test videos; past 2 px the lens model
# cannot follow the picture or a located tag sits on the wrong frame, and one point 6 px
# off is a wrong match or an end of the field the lens does not reach.
MAX_RMS_PX = 2.0
MAX_RESID_PX = 6.0
# Held-out error, each tag capped at HOLDOUT_CAP_M (a tag past 1 m, or one the refit cannot
# reach at all, has simply failed). The worst group is usually a corner pair whose removal
# leaves that corner to extrapolation, so there is no gate on the worst tag: the median
# says whether the fit is good, the capped mean bounds how many tags may fail.
MAX_HOLDOUT_MEDIAN_M = 0.25
MAX_HOLDOUT_MEAN_M = 0.3
CAMERA_Z_M = (0.3, 15.0)
MAX_ROLL_DEG = 45.0
# Field lines, when used: the ridge points must lie on the fitted lines. A wrong edge, or a
# lens that cannot bend the lines as the picture does, shows as a high RMS or few inliers.
MAX_LINE_RMS_PX = 1.5
MIN_LINE_INLIERS = 0.8

CHECK_IMAGE = "calibration_check.jpg"


class _Layout(BaseModel):
    """The part of match.json this stage reads: the camera panels and the match window."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    scoreboard: PixelRect
    views: dict[ViewName, PixelRect]
    auto_start_t: float
    teleop_end_t: float


@dataclass(frozen=True)
class View:
    """A camera panel and where its optical centre is expected, in full-frame pixels."""

    rect: PixelRect
    centre: tuple[float, float]
    frame: tuple[int, int]
    box: tuple[float, float, float, float]
    """(x0, y0, x1, y1) where ridge points may lie: inside the panel by LINE_MARGIN_PX and
    below a scoreboard drawn over it."""

    @classmethod
    def of(cls, rect: PixelRect, info: VideoInfo, scoreboard: PixelRect) -> View:
        """A panel that starts at the top of the frame and spans its whole width is the top
        of a full-frame camera picture with the scoreboard and corner panels drawn over it
        (official broadcasts: the bow of the guardrails puts the optical centre within
        about 40 px of the frame centre on all three), so its optical centre is the frame
        centre. Any other panel is taken as centred on its own centre (a crop whose offset
        is unknown; on CapCity the free fits put it 95 px below)."""
        full = rect.y == 0 and rect.x == 0 and rect.w == info.width
        mid = (rect.x + rect.w / 2, rect.y + rect.h / 2)
        centre = (info.width / 2, info.height / 2) if full else mid
        top = rect.y
        if scoreboard.y <= rect.y < scoreboard.y + scoreboard.h:
            top = scoreboard.y + scoreboard.h
        m = LINE_MARGIN_PX
        box = (rect.x + m, top + m, rect.x + rect.w - m, rect.y + rect.h - m)
        return cls(rect, centre, (info.width, info.height), box)


@dataclass(frozen=True)
class Tag:
    """One tag seen in a view: image points (N, 2) in full-frame pixels and the field
    points (N, 3) they show. Decoded tags give 4 corners, located tags their centre."""

    id: int
    img: F64
    obj: F64
    located: bool


Hits = dict[int, list[tuple[int, F64]]]
"""Per tag id, (window index, image points) of every sighting."""


@dataclass(frozen=True)
class Fit:
    """One solved camera for a set of tags."""

    model: LensModel
    k: F64
    dist: F64
    rvec: F64
    tvec: F64
    tags: tuple[int, ...]
    rms: float
    max: float
    lines: tuple[LinePoints, ...] = ()
    dz: tuple[float, ...] = ()
    """Fitted height offset of each of `lines` (0 for a line whose height is known)."""
    line_err: tuple[F64, ...] = ()
    """Signed distance in px of each line's ridge points from its projection."""

    def camera(self) -> Camera:
        return Camera.from_calibration(_to_schema(self, {}, {}, 0.0))

    def offset(self, name: str) -> float:
        """Fitted height offset of the field line `name` (0 when it was not fitted)."""
        return next(
            (d for p, d in zip(self.lines, self.dz, strict=True) if p.line.name == name), 0.0
        )

    def line_stats(self) -> tuple[int, float, float]:
        """(points, capped RMS px, inlier share) over every line."""
        if not self.line_err:
            return 0, 0.0, 1.0
        e = np.abs(np.concatenate(self.line_err))
        rms = float(np.sqrt(np.mean(np.minimum(e, LINE_CAP_PX) ** 2)))
        return len(e), rms, float(np.mean(e < LINE_INLIER_PX))


Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class FieldLine:
    """A straight field feature that shows as a bright ridge in the median image. With
    `free_height` its height is fitted within RAIL_HEIGHT_FREE_M."""

    name: str
    a: Point3
    b: Point3
    free_height: bool

    def points(self, s: F64, dz: float) -> F64:
        """Field points (N, 3) at distances `s` metres from `a` toward `b`, raised `dz`."""
        a, b = np.array(self.a), np.array(self.b)
        u = (b - a) / np.linalg.norm(b - a)
        return a + s[:, None] * u + np.array([0.0, 0.0, dz])

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.subtract(self.b, self.a)))


_RAIL_X = (LINE_END_M, FIELD_LENGTH_M - LINE_END_M)
FIELD_LINES = (
    # The white centre tape, 2 in wide (field_wireframe tape at x 8.245 and 8.296 m).
    FieldLine(
        "centre",
        (CENTER_LINE_X_M, LINE_END_M, 0.0),
        (CENTER_LINE_X_M, FIELD_WIDTH_M - LINE_END_M, 0.0),
        False,
    ),
    FieldLine(
        "rail_y0",
        (_RAIL_X[0], -RAIL_OUT_M, RAIL_Z_M),
        (_RAIL_X[1], -RAIL_OUT_M, RAIL_Z_M),
        True,
    ),
    FieldLine(
        "rail_y8",
        (_RAIL_X[0], FIELD_WIDTH_M + RAIL_OUT_M, RAIL_Z_M),
        (_RAIL_X[1], FIELD_WIDTH_M + RAIL_OUT_M, RAIL_Z_M),
        True,
    ),
)


@dataclass(frozen=True)
class LinePoints:
    """Ridge points found along one field line: distance along it (metres from `a`) and
    the image point (full-frame px) of each."""

    line: FieldLine
    s: F64
    img: F64


@dataclass(frozen=True)
class ViewResult:
    """Outcome for one view: the best fit found (if any) and why it is not trusted."""

    fit: Fit | None
    holdout: dict[int, float]
    spread: float
    problems: list[str]
    tags: dict[int, Tag]
    suspect: list[int]


# --- field geometry ------------------------------------------------------------


def tag_corners() -> dict[int, F64]:
    """Every field tag's black-square corners (4, 3) in field metres. Order: bottom-left,
    bottom-right, top-right, top-left seen from the front, matching pupil_apriltags."""
    data = json.loads(asset("field_tags").read_text())
    s = TAG_SIZE_M / 2
    local = np.array([[0.0, -s, -s], [0.0, s, -s], [0.0, s, s], [0.0, -s, s]])
    out: dict[int, F64] = {}
    for t in data["tags"]:
        p = t["pose"]["translation"]
        q = t["pose"]["rotation"]["quaternion"]
        r = _quat_matrix(float(q["W"]), float(q["X"]), float(q["Y"]), float(q["Z"]))
        out[int(t["ID"])] = local @ r.T + np.array([p["x"], p["y"], p["z"]], np.float64)
    return out


def _quat_matrix(w: float, x: float, y: float, z: float) -> F64:
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _normal(c: F64) -> F64:
    """Outward normal of a tag from its corners (front face)."""
    n = np.cross(c[1] - c[0], c[3] - c[0])
    return n / np.linalg.norm(n)


def _faces(c: F64, cam: Camera) -> bool:
    return bool(_normal(c) @ (cam.center - c.mean(0)) > 0)


def _clusters(
    ids: list[int], corners: dict[int, F64], dist_m: float, same_facing: bool
) -> list[list[int]]:
    """Tags whose centres lie within `dist_m` of each other, transitively; with
    `same_facing`, only tags facing the same way join."""
    groups: list[list[int]] = []
    for t in sorted(ids):
        c, n = corners[t].mean(0), _normal(corners[t])
        near = [
            g
            for g in groups
            if any(
                np.linalg.norm(corners[u].mean(0) - c) < dist_m
                and (not same_facing or _normal(corners[u]) @ n > 0.95)
                for u in g
            )
        ]
        groups = [g for g in groups if g not in near] + [sorted([t, *(u for g in near for u in g)])]
    return sorted(groups)


def _spread(ids: list[int], corners: dict[int, F64]) -> float:
    pts = np.array([corners[t].mean(0)[:2] for t in ids])
    if len(pts) < 2:
        return 0.0
    return float(np.max(np.linalg.norm(pts[:, None] - pts[None], axis=2)))


# --- finding tags --------------------------------------------------------------


def _detector() -> Detector:
    return Detector(
        families="tag36h11",
        nthreads=THREADS,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
    )


def _decode_view(det: Detector, medians: list[U8], rect: PixelRect) -> Hits:
    """Every tag decoded in each window median of the panel at each of SCALES."""
    hits: Hits = {}
    off = np.array([rect.x, rect.y], np.float64)
    for w, med in enumerate(medians):
        crop = med[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w]
        for s in SCALES:
            up = np.asarray(
                cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC), np.uint8
            )
            # The bindings annotate detect() as one Detection and decision_margin as None.
            for d in cast(list[Detection], det.detect(up)):
                if d.hamming <= 1 and cast(float, d.decision_margin) > MIN_MARGIN:
                    corners = np.asarray(d.corners, np.float64) / s + off
                    hits.setdefault(int(d.tag_id), []).append((w, corners))
    return hits


def _merge(hits: Hits, obj: dict[int, F64], located: bool) -> tuple[dict[int, Tag], list[int]]:
    """Trusted tags (seen in MIN_WINDOWS windows with agreeing points) and suspect ids."""
    tags: dict[int, Tag] = {}
    suspect: list[int] = []
    for tid, hs in hits.items():
        stack = np.stack([p for _, p in hs])
        med = np.median(stack, 0)
        good = np.max(np.linalg.norm(stack - med, axis=2), axis=1) <= AGREE_PX
        windows = len({w for (w, _), g in zip(hs, good, strict=True) if g})
        if windows >= MIN_WINDOWS and tid in obj:
            tags[tid] = Tag(tid, np.median(stack[good], 0), obj[tid], located)
        else:
            suspect.append(tid)
    return tags, sorted(suspect)


def _frames(img: U8, x0: int, y0: int, x1: int, y1: int, min_area: float) -> F64:
    """Centres (M, 2) of white square frames (a bright ring around a dark square) in a
    box of `img`, full-frame pixels. Found on a 4x upscale under a few local thresholds
    (contrast varies with the panel behind the tag); the centre is the centroid of the
    dark interior."""
    s = 4
    up = cv2.resize(img[y0:y1, x0:x1], None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    out: list[tuple[float, float]] = []
    for block, offset in ((31, -15), (51, -15), (31, -8)):
        bw = cv2.adaptiveThreshold(
            up, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, offset
        )
        contours, hier = cv2.findContours(bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            continue
        for i, c in enumerate(contours):
            parent, child = int(hier[0][i][3]), int(hier[0][i][2])
            if parent != -1 or child == -1:
                continue
            area = cv2.contourArea(c)
            hole = contours[child]
            hole_area = cv2.contourArea(hole)
            solidity = area / max(cv2.contourArea(cv2.convexHull(c)), 1.0)
            if area < min_area * s * s or not 0.12 < hole_area / area < 0.7 or solidity < 0.9:
                continue
            m = cv2.moments(hole)
            q = (
                (m["m10"] / m["m00"] / s + x0, m["m01"] / m["m00"] / s + y0)
                if m["m00"] > 0
                else None
            )
            if q is not None and all(math.dist(q, o) > 1.5 for o in out):
                out.append(q)
    return np.array(out, np.float64).reshape(-1, 2)


def _locate(
    medians: list[U8], rect: PixelRect, fit: Fit, corners: dict[int, F64], skip: set[int]
) -> Hits:
    """Centres of undecoded camera-facing tags, found by their white frames and matched
    per structure by `_match_structure`."""
    cam = fit.camera()
    pred: dict[int, F64] = {}
    side: dict[int, float] = {}
    for tid, c3 in corners.items():
        uv = cam.project(c3)
        ctr = uv.mean(0)
        inside = (
            rect.x + 4 <= ctr[0] < rect.x + rect.w - 4
            and rect.y + 4 <= ctr[1] < rect.y + rect.h - 4
        )
        if tid in skip or not _faces(c3, cam) or not np.all(np.isfinite(uv)) or not inside:
            continue
        pred[tid] = ctr
        side[tid] = float(np.mean(np.linalg.norm(uv - np.roll(uv, 1, 0), axis=1)))
    hits: Hits = {}
    for group in _clusters(list(pred), corners, STRUCTURE_M, same_facing=True):
        p = np.stack([pred[t] for t in group])
        px = min(side[t] for t in group)
        if px < 4:
            continue
        r = LOCATE_RADIUS_PX
        lo = np.maximum(p.min(0) - r - 2 * px, [rect.x, rect.y]).astype(int)
        hi = np.minimum(p.max(0) + r + 2 * px, [rect.x + rect.w, rect.y + rect.h]).astype(int)
        for w, med in enumerate(medians):
            found = _frames(
                med, int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1]), min_area=0.6 * px * px
            )
            for t, q in zip(group, _match_structure(p, found, r), strict=True):
                if q is not None:
                    hits.setdefault(t, []).append((w, q[None]))
    return hits


def _match_structure(pred: F64, found: F64, radius: float) -> list[F64 | None]:
    """The found centre for each predicted tag centre of one structure, or None.

    The whole structure matches when exactly one shift shorter than `radius` puts every
    predicted centre within `tol` (0.4 of the tag spacing) of a found one. Failing that,
    a tag of a multi-tag structure (one tag hidden, say) matches alone when exactly one
    found centre lies within `tol` of it and none within `tol` of another tag: with an
    error under half the spacing, a neighbour cannot be mistaken for it."""
    gaps = [float(np.linalg.norm(pred[i] - pred[j])) for i in range(len(pred)) for j in range(i)]
    tol = max(3.0, 0.4 * min(gaps)) if gaps else radius
    shifts: list[F64] = []
    for q in found:
        for p in pred:
            d = q - p
            if np.linalg.norm(d) >= radius:
                continue
            dist = np.linalg.norm(found[None] - (pred + d)[:, None], axis=2)
            if np.all(dist.min(1) < tol) and not any(np.linalg.norm(d - e) < tol for e in shifts):
                shifts.append(d)
    if len(shifts) == 1:
        dist = np.linalg.norm(found[None] - (pred + shifts[0])[:, None], axis=2)
        return [found[i] for i in dist.argmin(1)]
    out: list[F64 | None] = [None] * len(pred)
    if len(pred) < 2 or shifts:
        return out
    near = np.linalg.norm(found[None] - pred[:, None], axis=2) < tol
    for i in range(len(pred)):
        mine = np.flatnonzero(near[i])
        if len(mine) == 1 and near[:, mine[0]].sum() == 1:
            out[i] = found[mine[0]]
    return out


# --- solving -------------------------------------------------------------------


def _points(tags: dict[int, Tag], ids: tuple[int, ...]) -> tuple[F64, F64]:
    return np.concatenate([tags[t].obj for t in ids]), np.concatenate([tags[t].img for t in ids])


def _project(obj: F64, k: F64, dist: F64, rvec: F64, tvec: F64) -> F64:
    uv, _ = cv2.projectPoints(obj, rvec, tvec, k, dist)
    return np.asarray(uv, np.float64).reshape(-1, 2)


def _finish(
    model: LensModel,
    k: F64,
    dist: F64,
    rvec: F64,
    tvec: F64,
    ids: tuple[int, ...],
    obj: F64,
    img: F64,
    lines: tuple[LinePoints, ...] = (),
    dz: tuple[float, ...] = (),
) -> Fit:
    err = np.linalg.norm(_project(obj, k, dist, rvec, tvec) - img, axis=1)
    rms, worst = float(np.sqrt(np.mean(err**2))), float(err.max())
    line_err = tuple(_line_resid(lines, dz, k, dist, rvec, tvec))
    return Fit(model, k, dist, rvec, tvec, ids, rms, worst, lines, dz, line_err)


def _line_resid(
    lines: tuple[LinePoints, ...], dz: tuple[float, ...], k: F64, dist: F64, rvec: F64, tvec: F64
) -> list[F64]:
    """Per line, the signed distance (px) of each ridge point from the line's projection,
    along the projected curve's normal. First order in the point's slide along the line,
    so the stale distance along it does not matter."""
    out: list[F64] = []
    for lp, d in zip(lines, dz, strict=True):
        p = lp.line.points(lp.s, d)
        q0 = _project(p, k, dist, rvec, tvec)
        t = _project(lp.line.points(lp.s + 0.02, d), k, dist, rvec, tvec) - q0
        t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
        out.append(np.sum((lp.img - q0) * np.stack([-t[:, 1], t[:, 0]], 1), axis=1))
    return out


def _seeds(obj: F64, img: F64, view: View, ids: tuple[int, ...]) -> list[Fit]:
    """Starting cameras from OpenCV's calibrateCamera (principal point at the view's
    optical centre guess, square pixels, k1 only) over a few focal lengths, best first."""
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
        | cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_FIX_ASPECT_RATIO
    )
    out: list[Fit] = []
    cx, cy = view.centre
    w = view.rect.w
    for f0 in (0.3 * w, 0.5 * w, 0.9 * w):
        for k10 in (0.0, -0.2):
            k0 = np.array([[f0, 0, cx], [0, f0, cy], [0, 0, 1]])
            try:
                _, k, d, rv, tv = cv2.calibrateCamera(
                    [obj.astype(np.float32)],
                    [img.astype(np.float32)],
                    view.frame,
                    k0,
                    np.array([k10, 0, 0, 0, 0], np.float64),
                    flags=flags,
                )
            except cv2.error:
                continue
            dist = np.zeros(5)
            dist[0] = float(np.asarray(d).reshape(-1)[0])
            rvec = np.asarray(rv[0], np.float64).reshape(3)
            tvec = np.asarray(tv[0], np.float64).reshape(3)
            out.append(_finish("pp_k1", np.asarray(k, np.float64), dist, rvec, tvec, ids, obj, img))
    return sorted(out, key=lambda f: f.rms)


def _refine(
    model: LensModel,
    start: Fit,
    obj: F64,
    img: F64,
    ids: tuple[int, ...],
    lines: tuple[LinePoints, ...] = (),
) -> Fit:
    """Least-squares refit of pose and `model`'s free lens parameters from `start`.
    Distortion terms the model does not free are zero; the rest start from `start`, as does
    the principal point when the model does not free it. With `lines`, their ridge points
    are fitted too (soft-L1, so a stray ridge point cannot pull the fit) along with the
    height of each free-height line."""
    names = FREE[model]
    base = {"f": start.k[0, 0], "cx": start.k[0, 2], "cy": start.k[1, 2]}
    base |= {n: float(start.dist[i]) if n in names else 0.0 for i, n in enumerate(DIST_TERMS)}
    free_dz = [i for i, lp in enumerate(lines) if lp.line.free_height]
    n_lens = 6 + len(names)

    def unpack(x: F64) -> tuple[F64, F64, F64, F64, tuple[float, ...]]:
        v = base | dict(zip(names, (float(a) for a in x[6:n_lens]), strict=True))
        k = np.array([[v["f"], 0, v["cx"]], [0, v["f"], v["cy"]], [0, 0, 1]], np.float64)
        dist = np.array([v[n] for n in DIST_TERMS], np.float64)
        dz = [0.0] * len(lines)
        for i, d in zip(free_dz, x[n_lens:], strict=True):
            dz[i] = float(d)
        return k, dist, x[:3].copy(), x[3:6].copy(), tuple(dz)

    def resid(x: F64) -> F64:
        k, dist, rv, tv, dz = unpack(x)
        tag = (_project(obj, k, dist, rv, tv) - img).reshape(-1)
        return np.concatenate([tag, *_line_resid(lines, dz, k, dist, rv, tv)])

    x0 = np.concatenate([start.rvec, start.tvec, [base[n] for n in names]])
    if not lines:
        sol = least_squares(resid, x0, method="lm", x_scale="jac")
        return _finish(model, *unpack(np.asarray(sol.x, np.float64))[:4], ids, obj, img)
    h = RAIL_HEIGHT_FREE_M * 0.999
    dz0 = [float(np.clip(start.offset(lines[i].line.name), -h, h)) for i in free_dz]
    lo = np.r_[np.full(n_lens, -np.inf), np.full(len(free_dz), -RAIL_HEIGHT_FREE_M)]
    sol = least_squares(
        resid,
        np.r_[x0, dz0],
        loss="soft_l1",
        f_scale=LINE_LOSS_PX,
        x_scale="jac",
        bounds=(lo, -lo),
    )
    k, dist, rv, tv, dz = unpack(np.asarray(sol.x, np.float64))
    return _finish(model, k, dist, rv, tv, ids, obj, img, lines, dz)


def _plausible(fit: Fit, corners: dict[int, F64], obj: F64) -> str | None:
    """None if the pose is believable for a broadcast camera, else the reason."""
    cam = fit.camera()
    c = cam.center
    if not CAMERA_Z_M[0] <= c[2] <= CAMERA_Z_M[1]:
        return f"camera height {c[2]:.2f} m is outside {CAMERA_Z_M[0]}-{CAMERA_Z_M[1]} m"
    up = cam.r @ np.array([0.0, 0.0, 1.0])
    if up[1] >= 0:
        return "camera is upside down"
    roll = math.degrees(math.atan2(up[0], -up[1]))
    if abs(roll) > MAX_ROLL_DEG:
        return f"camera roll {roll:.0f} deg exceeds {MAX_ROLL_DEG:g} deg"
    if np.any((obj @ cam.r.T + cam.t)[:, 2] <= 0.3):
        return "some tag points are behind the camera"
    on_field = 0 < c[0] < FIELD_LENGTH_M and 0 < c[1] < FIELD_WIDTH_M
    centre = np.array([FIELD_LENGTH_M / 2, FIELD_WIDTH_M / 2, 0.0])
    if not on_field and (cam.r @ centre + cam.t)[2] <= 0:
        return "the field centre is behind the camera"
    away = [t for t in fit.tags if not _faces(corners[t], cam)]
    if 2 * len(away) > len(fit.tags):
        return f"tags {away} would be seen from behind"
    return None


def _covers(fit: Fit, view: View) -> str | None:
    """None if the lens reaches every pixel of the panel, else the reason. A real lens maps
    every pixel to a ray; a polynomial that folds inside the panel is a fit that bent too
    far. (Points beyond the panel may lie past the fold; `Camera` maps those to NaN.)"""
    r = view.rect
    corners = np.array(
        [[r.x, r.y], [r.x + r.w, r.y], [r.x, r.y + r.h], [r.x + r.w, r.y + r.h]], np.float64
    )
    need = float(np.max(np.linalg.norm(corners - fit.k[:2, 2], axis=1))) / fit.k[0, 0]
    reach = reach_radius(fit.dist)
    if reach < need * COVER_MARGIN:
        return f"lens folds at radius {reach:.2f}, inside the panel (radius {need:.2f})"
    return None


def _holdout(fit: Fit, tags: dict[int, Tag], corners: dict[int, F64]) -> dict[int, float]:
    """Leave-one-group-out field error per tag in metres (inf when the refit cannot
    cast the held-out points, e.g. its lens folds before reaching them). The refit keeps
    the fit's field lines."""
    out: dict[int, float] = {}
    for group in _clusters(list(fit.tags), corners, GROUP_M, same_facing=False):
        rest = tuple(t for t in fit.tags if t not in group)
        if len(rest) < 3:
            out.update(dict.fromkeys(group, math.inf))
            continue
        obj, img = _points(tags, rest)
        cam = _refine(fit.model, fit, obj, img, rest, fit.lines).camera()
        for t in group:
            xy = cam.to_field(tags[t].img, tags[t].obj[:, 2])
            err = float(np.mean(np.linalg.norm(xy - tags[t].obj[:, :2], axis=1)))
            out[t] = err if np.isfinite(err) else math.inf
    return dict(sorted(out.items()))


def _mean(h: dict[int, float]) -> float:
    return float(np.mean([min(v, HOLDOUT_CAP_M) for v in h.values()])) if h else math.inf


def _freer_wins(cand: Fit, cand_h: dict[int, float], best: Fit, best_h: dict[int, float]) -> bool:
    """A freer lens replaces the current one if its held-out error is clearly lower, or if
    its reprojection RMS is clearly lower and its held-out error is no worse. Held-out
    error alone cannot decide: it is dominated by corner tags whose removal leaves the
    corner to extrapolation, which every model does badly."""
    clearly_held = _mean(cand_h) < FREER_GAIN * _mean(best_h)
    clearly_rms = cand.rms < FREER_GAIN * best.rms and _mean(cand_h) <= _mean(best_h)
    return clearly_held or clearly_rms


def _solve(
    tags: dict[int, Tag], corners: dict[int, F64], view: View, start: Fit | None = None
) -> tuple[Fit | None, dict[int, float], list[str]]:
    """Climb the tags-only lens ladder from each seed in turn: `start` (an earlier fit
    with its principal point moved back to the view's centre), then calibrateCamera solves.
    Each rung starts from the one below. The first rung that is plausible and covers the
    panel is the base; a freer rung replaces it only if `_freer_wins`. Returns the chosen
    fit, its hold-out errors and the reasons it should not be trusted (empty when it passes
    every gate)."""
    ids = tuple(sorted(tags))
    spread = _spread(list(ids), corners)
    if len(ids) < MIN_TAGS["pp_k1"]:
        return None, {}, [f"only {len(ids)} usable tags (need {MIN_TAGS['pp_k1']})"]
    obj, img = _points(tags, ids)
    seeds = _seeds(obj, img, view, ids)
    if start is not None:
        k = start.k.copy()
        k[0, 2], k[1, 2] = view.centre
        dist = np.array([start.dist[0], 0, 0, 0, 0], np.float64)
        seeds.insert(0, Fit("pp_k1", k, dist, start.rvec, start.tvec, ids, 0.0, 0.0))
    best: Fit | None = None
    holdout: dict[int, float] = {}
    why = "no seed solve converged"
    for seed in seeds:
        prev = seed
        for i, model in enumerate(LADDER):
            if len(ids) < MIN_TAGS[model] or (i > 0 and spread < MIN_SPREAD_M):
                break
            cand = prev = _refine(model, prev, obj, img, ids)
            reason = _plausible(cand, corners, obj) or _covers(cand, view)
            if reason:
                why = f"{model}: {reason}"
                continue
            h = _holdout(cand, tags, corners)
            if best is None or _freer_wins(cand, h, best, holdout):
                best, holdout = cand, h
        if best is not None:
            break
    if best is None:
        return None, {}, [why]
    return best, holdout, _problems(best, holdout, spread)


def _problems(fit: Fit, holdout: dict[int, float], spread: float) -> list[str]:
    """Every trust gate `fit` fails."""
    problems: list[str] = []
    if spread < MIN_SPREAD_M:
        problems.append(f"tags span only {spread:.1f} m of field (need {MIN_SPREAD_M:g})")
    if fit.rms > MAX_RMS_PX:
        problems.append(f"reprojection RMS {fit.rms:.2f} px > {MAX_RMS_PX:g}")
    if fit.max > MAX_RESID_PX:
        problems.append(f"worst point residual {fit.max:.1f} px > {MAX_RESID_PX:g}")
    errs = list(holdout.values()) or [math.inf]
    if float(np.median(errs)) > MAX_HOLDOUT_MEDIAN_M:
        problems.append(f"median held-out error {np.median(errs):.2f} m > {MAX_HOLDOUT_MEDIAN_M:g}")
    if _mean(holdout) > MAX_HOLDOUT_MEAN_M:
        problems.append(
            f"capped mean held-out error {_mean(holdout):.2f} m > {MAX_HOLDOUT_MEAN_M:g}"
        )
    if fit.lines:
        _, rms, inliers = fit.line_stats()
        if rms > MAX_LINE_RMS_PX:
            problems.append(f"field line RMS {rms:.2f} px > {MAX_LINE_RMS_PX:g}")
        if inliers < MIN_LINE_INLIERS:
            problems.append(f"only {inliers:.0%} of ridge points on the lines")
    return problems


def _relocate(
    medians: list[U8],
    view: View,
    fit: Fit,
    corners: dict[int, F64],
    decoded: dict[int, Tag],
    skip: set[int],
) -> tuple[dict[int, Tag], Fit]:
    """`decoded` plus the tags located with `fit`, refitted with `fit`'s model and lines.
    The located tag worst off the refit is dropped while it is past MAX_RESID_PX: it was
    matched to the wrong frame."""
    centres = {t: c.mean(0)[None] for t, c in corners.items()}
    located, _ = _merge(_locate(medians, view.rect, fit, corners, skip), centres, located=True)
    while True:
        tags = decoded | located
        ids = tuple(sorted(tags))
        obj, img = _points(tags, ids)
        new = _refine(fit.model, fit, obj, img, ids, fit.lines)
        err = {
            t: float(
                np.max(np.linalg.norm(new.camera().project(tags[t].obj) - tags[t].img, axis=1))
            )
            for t in located
        }
        worst = max(err, key=lambda t: err[t], default=None)
        if worst is None or err[worst] <= MAX_RESID_PX:
            return tags, new
        del located[worst]


def _calibrate_view(
    det: Detector, medians: list[U8], view: View, corners: dict[int, F64]
) -> tuple[ViewResult, dict[int, Tag], set[int]]:
    """The tags-only result, plus the decoded tags and every decoded id (for `_line_path`)."""
    rect = view.rect
    decoded_hits = _decode_view(det, medians, rect)
    decoded, suspect = _merge(decoded_hits, corners, located=False)
    tags = decoded
    fit, holdout, problems = _solve(tags, corners, view)
    for _ in range(LOCATE_ROUNDS):
        if fit is None:
            break
        tags, moved = _relocate(medians, view, fit, corners, decoded, set(decoded_hits))
        if len(tags) == len(decoded):
            break
        fit, holdout, problems = _solve(tags, corners, view, moved)
    spread = _spread(list(fit.tags), corners) if fit else 0.0
    return ViewResult(fit, holdout, spread, problems, tags, suspect), decoded, set(decoded_hits)


# --- field lines --------------------------------------------------------------------


def _ridge_image(median: U8) -> NDArray[np.float32]:
    return np.asarray(cv2.GaussianBlur(median.astype(np.float32), (0, 0), 1.0), np.float32)


def _sample(img: NDArray[np.float32], pts: F64) -> F64:
    """Bilinear samples of `img` at pixel points (..., 2); 0 outside."""
    x = pts[..., 0].astype(np.float32).reshape(1, -1)
    y = pts[..., 1].astype(np.float32).reshape(1, -1)
    out = cv2.remap(img, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return np.asarray(out, np.float64).reshape(pts.shape[:-1])


def _curve(
    cam: Camera, line: FieldLine, dz: float, box: tuple[float, float, float, float]
) -> tuple[F64, F64, F64]:
    """Distances along `line` (m), image points and unit image normals of its projection
    inside `box`, one every LINE_STEP_PX along the curve."""
    s = np.linspace(0.0, line.length, int(line.length / 0.01) + 1)
    uv = cam.project(line.points(s, dz))
    x0, y0, x1, y1 = box
    ok = np.isfinite(uv).all(1)
    ok[ok] &= (uv[ok, 0] > x0) & (uv[ok, 0] < x1) & (uv[ok, 1] > y0) & (uv[ok, 1] < y1)
    if ok.sum() < 3:
        return np.zeros(0), np.zeros((0, 2)), np.zeros((0, 2))
    s, uv = s[ok], uv[ok]
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(uv, axis=0), axis=1))]
    i = np.clip(np.searchsorted(arc, np.arange(0.0, arc[-1], LINE_STEP_PX)), 1, len(uv) - 2)
    t = uv[i + 1] - uv[i - 1]
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-12
    return s[i], uv[i], np.stack([-t[:, 1], t[:, 0]], 1)


def _ridge(ridge: NDArray[np.float32], uv: F64, nrm: F64, radius: float) -> tuple[F64, F64]:
    """Offsets (half-pixel steps within `radius`) along each normal and the ridge strength
    there: brightness above the brighter of the two points RIDGE_PX to either side."""
    d = np.arange(-radius, radius + 0.25, 0.5)
    pts = uv[:, None] + d[None, :, None] * nrm[:, None]
    side = RIDGE_PX * nrm[:, None]
    return d, _sample(ridge, pts) - np.maximum(
        _sample(ridge, pts - side), _sample(ridge, pts + side)
    )


def _line_points(
    fit: Fit, ridge: NDArray[np.float32], box: tuple[float, float, float, float], radius: float
) -> tuple[LinePoints, ...]:
    """For each field line, the strongest ridge within `radius` px of its projection at
    each sample, kept where it is clearly a ridge (MIN_RIDGE and half the line's median
    strength); lines with fewer than MIN_LINE_POINTS are left out."""
    cam = fit.camera()
    out: list[LinePoints] = []
    for line in FIELD_LINES:
        s, uv, nrm = _curve(cam, line, fit.offset(line.name), box)
        if len(s) < MIN_LINE_POINTS:
            continue
        d, r = _ridge(ridge, uv, nrm, radius)
        j = r.argmax(1)
        peak = r[np.arange(len(r)), j]
        good = peak > max(MIN_RIDGE, 0.5 * float(np.median(peak)))
        if good.sum() >= MIN_LINE_POINTS:
            out.append(LinePoints(line, s[good], uv[good] + d[j[good], None] * nrm[good]))
    return tuple(out)


def _line_score(
    fit: Fit, ridge: NDArray[np.float32], box: tuple[float, float, float, float]
) -> float:
    """Sum over field lines of the mean ridge strength found within SCORE_PX of the
    projection: how well a starting camera lies on the lines before any refit."""
    cam = fit.camera()
    total = 0.0
    for line in FIELD_LINES:
        s, uv, nrm = _curve(cam, line, 0.0, box)
        if len(s) >= MIN_LINE_POINTS:
            _, r = _ridge(ridge, uv, nrm, SCORE_PX[line.free_height])
            total += float(np.mean(np.clip(r.max(1), 0.0, None)))
    return total


def _grid_starts(tags: dict[int, Tag], corners: dict[int, F64], view: View) -> list[Fit]:
    """`pp_k2` cameras for a grid of (k1, k2), focal length and pose solved from `tags`,
    that are plausible, cover the panel and fit the tags within GRID_MAX_RMS_PX. Tags in
    two rows leave focal length and distortion tied; the grid spans that tie."""
    ids = tuple(sorted(tags))
    obj, img = _points(tags, ids)
    seeds = _seeds(obj, img, view, ids)
    if not seeds:
        return []
    seed = seeds[0]
    cx, cy = view.centre
    out: list[Fit] = []
    for k1 in GRID_K1:
        for k2 in GRID_K2:
            dist = np.array([k1, k2, 0.0, 0.0, 0.0])

            def resid(x: F64, dist: F64 = dist) -> F64:
                k = np.array([[x[6], 0, cx], [0, x[6], cy], [0, 0, 1]])
                return (_project(obj, k, dist, x[:3], x[3:6]) - img).reshape(-1)

            x0 = np.r_[seed.rvec, seed.tvec, seed.k[0, 0]]
            x = np.asarray(least_squares(resid, x0, method="lm").x, np.float64)
            k = np.array([[x[6], 0, cx], [0, x[6], cy], [0, 0, 1]])
            fit = _finish("pp_k2", k, dist, x[:3], x[3:6], ids, obj, img)
            bad = _plausible(fit, corners, obj) or _covers(fit, view)
            if bad is None and fit.rms <= GRID_MAX_RMS_PX:
                out.append(fit)
    return out


def _icp(
    fit: Fit,
    tags: dict[int, Tag],
    ridge: NDArray[np.float32],
    box: tuple[float, float, float, float],
    rounds: tuple[tuple[float, LensModel], ...],
) -> Fit | None:
    """Alternate finding ridge points near the projected lines and refitting tags and lines,
    over `rounds` of (search band px, model). None when fewer than 2 lines are found."""
    ids = tuple(sorted(tags))
    obj, img = _points(tags, ids)
    for radius, model in rounds:
        lines = _line_points(fit, ridge, box, radius)
        if len(lines) < 2:
            return None
        fit = _refine(model, fit, obj, img, ids, lines)
    return fit


def _line_path(
    medians: list[U8],
    ridge: NDArray[np.float32],
    view: View,
    corners: dict[int, F64],
    decoded: dict[int, Tag],
    skip: set[int],
    tag_fit: Fit | None,
) -> ViewResult | None:
    """Tags and field lines together (module docstring, step 5). None when no starting
    camera exists or the lines are not found."""
    if len(decoded) < MIN_TAGS["pp_k2"]:
        return None
    starts = _grid_starts(decoded, corners, view) + ([tag_fit] if tag_fit else [])
    if not starts:
        return None
    start = max(starts, key=lambda f: _line_score(f, ridge, view.box))
    fit = _icp(start, decoded, ridge, view.box, ICP_ROUNDS)
    tags = decoded
    for _ in range(LOCATE_ROUNDS):
        if fit is None:
            return None
        tags, fit = _relocate(medians, view, fit, corners, decoded, skip)
        fit = _icp(fit, tags, ridge, view.box, LOCATE_ICP_ROUNDS)
    if fit is None:
        return None
    # Principal point held at the view's expected centre (pp_k2), then free (c_k2), then
    # k3 too (c_k3, which on the Championship lenses folds inside the panel and is refused).
    ids = tuple(sorted(tags))
    obj, img = _points(tags, ids)
    k = fit.k.copy()
    k[0, 2], k[1, 2] = view.centre
    held = _refine("pp_k2", replace(fit, k=k), obj, img, ids, fit.lines)
    ladder = (held, _refine("c_k2", fit, obj, img, ids, fit.lines), fit)
    spread = _spread(list(ids), corners)
    reasons = {c.model: _plausible(c, corners, obj) or _covers(c, view) for c in ladder}
    good = [(c, _holdout(c, tags, corners)) for c in ladder if reasons[c.model] is None]
    if not good:
        why = "; ".join(f"{m} with field lines: {r}" for m, r in reasons.items())
        return ViewResult(None, {}, spread, [why], tags, [])
    best, holdout = good[0]
    for cand, h in good[1:]:
        if _freer_wins(cand, h, best, holdout):
            best, holdout = cand, h
    return ViewResult(best, holdout, spread, _problems(best, holdout, spread), tags, [])


def _to_schema(
    fit: Fit, tags: dict[int, Tag], holdout: dict[int, float], spread: float
) -> CameraCalibration:
    r, _ = cv2.Rodrigues(fit.rvec)
    centre = -np.asarray(r, np.float64).T @ fit.tvec
    return CameraCalibration(
        k=[[float(v) for v in row] for row in fit.k],
        dist=[float(v) for v in fit.dist],
        rvec=[float(v) for v in fit.rvec],
        tvec=[float(v) for v in fit.tvec],
        rms_px=round(fit.rms, 3),
        max_px=round(fit.max, 3),
        n_points=sum(len(tags[t].img) for t in fit.tags if t in tags),
        model=fit.model,
        tags=[t for t in fit.tags if t in tags and not tags[t].located],
        located=[t for t in fit.tags if t in tags and tags[t].located],
        lines={
            lp.line.name: LineFit(
                n_points=len(e),
                rms_px=round(float(np.sqrt(np.mean(np.minimum(np.abs(e), LINE_CAP_PX) ** 2))), 3),
                inliers=round(float(np.mean(np.abs(e) < LINE_INLIER_PX)), 3),
                height_m=round(lp.line.a[2] + d, 3),
            )
            for lp, d, e in zip(fit.lines, fit.dz, fit.line_err, strict=True)
        },
        spread_m=round(spread, 3),
        holdout_m={t: round(v, 3) if math.isfinite(v) else None for t, v in holdout.items()},
        camera_xyz=[round(float(v), 3) for v in centre],
    )


# --- inputs and the check image --------------------------------------------------


def _medians(run: RunDir, info: VideoInfo, layout: _Layout) -> tuple[list[U8], U8]:
    """Grey full-frame medians of each quarter of the match (auto start to teleop end)
    and of the whole match. Outside the match an official broadcast shows intro cards,
    results and other cameras, so a median over the whole file is not the field. Only
    the rows holding camera panels are decoded; the rest stay black."""
    views = layout.views
    y0 = min(r.y for r in views.values())
    y1 = max(r.y + r.h for r in views.values())
    band = PixelRect(x=0, y=y0, w=info.width, h=y1 - y0)
    grey = [
        cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY)
        for f in frames(
            run.video,
            info,
            sample_hz=SAMPLE_HZ,
            crop=band,
            start=max(0.0, layout.auto_start_t),
            end=min(info.duration_s, layout.teleop_end_t),
        )
    ]
    stack = np.stack(grey)
    n = len(stack)

    def full(img: NDArray[np.floating]) -> U8:
        out = np.zeros((info.height, info.width), np.uint8)
        out[y0:y1] = img.astype(np.uint8)
        return out

    quarters = [
        full(np.median(stack[i * n // WINDOWS : (i + 1) * n // WINDOWS], 0)) for i in range(WINDOWS)
    ]
    return quarters, full(np.median(stack, 0))


Colour = tuple[int, int, int]
GREEN: Colour = (60, 220, 60)
ORANGE: Colour = (0, 140, 255)
RED: Colour = (40, 40, 230)
BLUE: Colour = (230, 120, 30)
CYAN: Colour = (230, 230, 0)
YELLOW: Colour = (0, 230, 230)
MAGENTA: Colour = (220, 60, 220)
WHITE: Colour = (255, 255, 255)


def _rect_edges(r: Rect) -> list[tuple[Point3, Point3]]:
    c = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]
    return [((*c[i], 0.0), (*c[(i + 1) % 4], 0.0)) for i in range(4)]


def _field_lines() -> list[tuple[Colour, Point3, Point3]]:
    """(colour, start, end) of every carpet-level line drawn in the check image."""
    lines = [(CYAN, a, b) for a, b in _rect_edges(Rect(0, 0, FIELD_LENGTH_M, FIELD_WIDTH_M))]
    mid = FIELD_LENGTH_M / 2
    lines.append((MAGENTA, (mid, 0.0, 0.0), (mid, FIELD_WIDTH_M, 0.0)))
    blue_x, red_x = ALLIANCE_ZONE_DEPTH_M, FIELD_LENGTH_M - ALLIANCE_ZONE_DEPTH_M
    lines.append((BLUE, (blue_x, 0.0, 0.0), (blue_x, FIELD_WIDTH_M, 0.0)))
    lines.append((RED, (red_x, 0.0, 0.0), (red_x, FIELD_WIDTH_M, 0.0)))
    lines += [(YELLOW, a, b) for r in BUMPS for a, b in _rect_edges(r)]
    lines += [(ORANGE, a, b) for r in HUBS.values() for a, b in _rect_edges(r)]
    return lines


def _draw_view(img: U8, rect: PixelRect, res: ViewResult, corners: dict[int, F64]) -> None:
    """Draw one view's field lines, predicted tag outlines (green used, orange not used),
    the measured tag points (white) and the fitted ridge points (red) onto `img`, clipped
    to the panel."""
    canvas = img[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w]
    off = np.array([rect.x, rect.y], np.float64)

    def poly(uv: F64, colour: Colour, closed: bool) -> None:
        good = np.isfinite(uv).all(1)
        for idx in np.split(np.arange(len(uv)), np.flatnonzero(np.diff(good.astype(np.int8))) + 1):
            if len(idx) >= 2 and good[idx[0]]:
                pts = np.round((uv[idx] - off) * 16).astype(np.int32)
                cv2.polylines(
                    canvas, [pts], closed and len(idx) == len(uv), colour, 1, cv2.LINE_AA, 4
                )

    for tag in res.tags.values():
        for p in tag.img - off:
            centre = (round(p[0]), round(p[1]))
            if tag.located:
                cv2.drawMarker(canvas, centre, WHITE, cv2.MARKER_CROSS, 5, 1)
            else:
                cv2.circle(canvas, centre, 1, WHITE, -1)
    if res.fit is None:
        return
    for lp in res.fit.lines:
        for p in lp.img - off:
            cv2.circle(canvas, (round(p[0]), round(p[1])), 1, RED, -1)
    cam = res.fit.camera()
    for colour, a, b in _field_lines():
        n = max(2, int(np.linalg.norm(np.subtract(b, a)) / 0.03))
        poly(cam.project(np.linspace(a, b, n)), colour, False)
    for tid, c3 in corners.items():
        uv = cam.project(c3)
        ctr = uv.mean(0) - off
        visible = np.all(np.isfinite(uv)) and 0 <= ctr[0] < rect.w and 0 <= ctr[1] < rect.h
        if not visible or not _faces(c3, cam):
            continue
        colour = GREEN if tid in res.fit.tags else ORANGE
        poly(uv, colour, True)
        org = (int(ctr[0]) + 6, int(ctr[1]) - 8)
        cv2.putText(canvas, str(tid), org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)


def _write_check(
    run: RunDir,
    median: U8,
    views: dict[ViewName, PixelRect],
    results: dict[ViewName, ViewResult],
    corners: dict[int, F64],
) -> None:
    img = cast(U8, cv2.cvtColor(median, cv2.COLOR_GRAY2BGR))
    for name, res in results.items():
        _draw_view(img, views[name], res, corners)
    cv2.imwrite(str(run.path / CHECK_IMAGE), img, [cv2.IMWRITE_JPEG_QUALITY, 92])


def _report(name: ViewName, res: ViewResult) -> str:
    if res.fit is None:
        return f"{name}: no fit ({'; '.join(res.problems)}); decoded {sorted(res.tags)}"
    f = res.fit
    c = f.camera().center
    located = sorted(t for t in f.tags if res.tags[t].located)
    held = " ".join(f"{t}:{v:.2f}" for t, v in res.holdout.items())
    n, rms, inl = f.line_stats()
    lines = (
        f"lines {[lp.line.name for lp in f.lines]} ({n} points, rms {rms:.2f} px, {inl:.0%} "
        f"within {LINE_INLIER_PX:g} px)\n"
        if f.lines
        else "no field lines\n"
    )
    status = "ok" if not res.problems else "REJECTED (" + "; ".join(res.problems) + ")"
    return (
        f"{name}: {status}\n  model {f.model}, tags {list(f.tags)} (located {located}), "
        f"spread {res.spread:.1f} m, rms {f.rms:.2f} px, max {f.max:.2f} px, f {f.k[0, 0]:.0f}, "
        f"c ({f.k[0, 2]:.0f}, {f.k[1, 2]:.0f}), k1 {f.dist[0]:.4f}, k2 {f.dist[1]:.4f}\n  {lines}"
        f"  camera ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}) m; held-out error m {held}; "
        f"suspect tags {res.suspect}"
    )


def _pick(tags_only: ViewResult, lined: ViewResult) -> ViewResult:
    """Of the two main-view results, one that passes every gate, the lower capped mean
    held-out error first; with neither passing, the one with fewer problems, lines first."""
    ok = [r for r in (lined, tags_only) if r.fit is not None and not r.problems]
    if ok:
        return min(ok, key=lambda r: _mean(r.holdout))
    fitted = [r for r in (lined, tags_only) if r.fit is not None]
    return min(fitted, key=lambda r: len(r.problems)) if fitted else lined


def calibrate(run: RunDir) -> None:
    info = VideoInfo.load(run.video_json)
    layout = _Layout.model_validate_json(run.match_json.read_bytes())
    corners = tag_corners()
    quarters, median = _medians(run, info, layout)
    det = _detector()
    results: dict[ViewName, ViewResult] = {}
    for name in VIEW_NAMES:
        if name not in layout.views:
            continue
        view = View.of(layout.views[name], info, layout.scoreboard)
        res, decoded, skip = _calibrate_view(det, quarters, view, corners)
        if name == "main":
            print("calibrate: tags only: " + _report(name, res))
            lined = _line_path(
                quarters, _ridge_image(median), view, corners, decoded, skip, res.fit
            )
            if lined is not None:
                print("calibrate: tags and lines: " + _report(name, lined))
                res = _pick(res, lined)
        else:
            print("calibrate: " + _report(name, res))
        results[name] = res
    _write_check(run, median, layout.views, results, corners)

    main = results.get("main")
    if main is None or main.fit is None or main.problems:
        accepted = run.answer("calibrate.accept_main") == "yes"
        if main is None or main.fit is None or not accepted:
            reason = "no main view" if main is None else "; ".join(main.problems)
            raise NeedsInput(
                f"main view calibration is not trustworthy: {reason}",
                run.path / CHECK_IMAGE,
                f"check {CHECK_IMAGE}; only if the drawn lines sit on the field lines, run "
                f"`scout answer {run.path} calibrate.accept_main=yes`",
            )
    views: dict[ViewName, CameraCalibration] = {
        name: _to_schema(res.fit, res.tags, res.holdout, res.spread)
        for name, res in results.items()
        if res.fit is not None and (not res.problems or name == "main")
    }
    Calibration(views=views).save(run.calibration_json)


STAGE = Stage(
    "calibrate",
    inputs=lambda r: (r.video, r.video_json, r.match_json),
    outputs=lambda r: (r.calibration_json,),
    fn=calibrate,
)
