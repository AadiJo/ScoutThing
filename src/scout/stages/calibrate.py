"""calibrate: video, video.json, match.json -> calibration.json and calibration_check.jpg.

The field's AprilTags are the targets. For each camera panel in `match.json`:

1. Decode the video once at 0.5 Hz, grey, and take the median of each quarter of the
   match. Robots and game pieces move, so each median shows the bare field.
2. Decode tag36h11 in each quarter median at 2x, 3x and 4x, without decimation. A tag
   is trusted only if it decodes in at least 2 of the 4 quarters (so it was in view for
   a good share of the match) with corners that agree; its corners are the median of
   every hit. Tags seen in fewer quarters are reported as suspect and not used.
3. Fit a ladder of lens models, most constrained first (`LensModel`). A freer model
   needs enough tags, a field spread of at least 2 m, a plausible pose and a lens that
   stays monotonic out to the field perimeter, and it is kept only if it is clearly
   better (`_freer_wins`).
4. Locate undecoded tags. Most tags are too small to decode (about 1.5 px per data bit)
   but their white frames are plain. For each structure of camera-facing tags (a pair
   on a wall or hub face, or a single tag) the fit predicts the tag centres; the frames
   found near them must match the whole structure under one shift, and exactly one shift
   may fit, so a neighbouring tag cannot be taken for the predicted one. A located tag
   gives one point, its centre, and must be found in 2 of the 4 quarters. Refit, locate
   again with the better prediction, refit.
5. Validate by leaving each group of neighbouring tags out (tags on one structure are
   grouped, or the held-out tag would be pinned by its neighbour), casting its points
   onto the plane at their own height, and measuring the error in metres.

The main view must pass every gate or the stage raises `NeedsInput`. A corner view that
cannot be calibrated is left out. `calibration_check.jpg` shows the field lines and tag
outlines drawn through each calibration on the full-match median image.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray
from pupil_apriltags import Detection, Detector  # pyright: ignore[reportMissingTypeStubs]
from pydantic import BaseModel, ConfigDict
from scipy.optimize import least_squares  # pyright: ignore[reportMissingTypeStubs]

from scout.assets import asset
from scout.camera import Camera, fold_radius
from scout.games.rebuilt2026 import (
    ALLIANCE_ZONE_DEPTH_M,
    BUMPS,
    FIELD_LENGTH_M,
    FIELD_WIDTH_M,
    HUBS,
    Rect,
)
from scout.run import NeedsInput, RunDir, Stage
from scout.schemas.calibration import Calibration, CameraCalibration, LensModel
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
}
DIST_TERMS = ("k1", "k2", "p1", "p2", "k3")
MIN_TAGS: dict[LensModel, int] = {"pp_k1": 4, "cy_k1": 5, "c_k1": 6, "c_k2": 8}

# Trust gates. A view failing any of these is not written.
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

CHECK_IMAGE = "calibration_check.jpg"


class _Layout(BaseModel):
    """The part of match.json this stage reads: the camera panel rectangles."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    views: dict[ViewName, PixelRect]


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

    def camera(self) -> Camera:
        return Camera.from_calibration(_to_schema(self, {}, {}, 0.0))


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


def _perimeter(step: float = 0.1) -> F64:
    """Points along the field perimeter at carpet height, (N, 3)."""
    xs = np.arange(0.0, FIELD_LENGTH_M + 1e-9, step)
    ys = np.arange(0.0, FIELD_WIDTH_M + 1e-9, step)
    xy = np.concatenate(
        [
            np.stack([xs, np.zeros_like(xs)], 1),
            np.stack([xs, np.full_like(xs, FIELD_WIDTH_M)], 1),
            np.stack([np.zeros_like(ys), ys], 1),
            np.stack([np.full_like(ys, FIELD_LENGTH_M), ys], 1),
        ]
    )
    return np.concatenate([xy, np.zeros((len(xy), 1))], 1)


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
) -> Fit:
    err = np.linalg.norm(_project(obj, k, dist, rvec, tvec) - img, axis=1)
    return Fit(model, k, dist, rvec, tvec, ids, float(np.sqrt(np.mean(err**2))), float(err.max()))


def _seeds(obj: F64, img: F64, rect: PixelRect, ids: tuple[int, ...]) -> list[Fit]:
    """Starting cameras from OpenCV's calibrateCamera (principal point at the panel centre,
    square pixels, k1 only) over a few focal lengths, best first."""
    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
        | cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_FIX_ASPECT_RATIO
    )
    out: list[Fit] = []
    for f0 in (0.3 * rect.w, 0.5 * rect.w, 0.9 * rect.w):
        for k10 in (0.0, -0.2):
            k0 = np.array([[f0, 0, rect.x + rect.w / 2], [0, f0, rect.y + rect.h / 2], [0, 0, 1]])
            try:
                _, k, d, rv, tv = cv2.calibrateCamera(
                    [obj.astype(np.float32)],
                    [img.astype(np.float32)],
                    (rect.x + rect.w, rect.y + rect.h),
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


def _refine(model: LensModel, start: Fit, obj: F64, img: F64, ids: tuple[int, ...]) -> Fit:
    """Least-squares refit of pose and `model`'s free lens parameters from `start`.
    Distortion terms the model does not free are zero; the rest start from `start`."""
    names = FREE[model]
    base = {"f": start.k[0, 0], "cx": start.k[0, 2], "cy": start.k[1, 2]}
    base |= {n: float(start.dist[i]) if n in names else 0.0 for i, n in enumerate(DIST_TERMS)}

    def unpack(x: F64) -> tuple[F64, F64, F64, F64]:
        v = base | dict(zip(names, (float(a) for a in x[6:]), strict=True))
        k = np.array([[v["f"], 0, v["cx"]], [0, v["f"], v["cy"]], [0, 0, 1]], np.float64)
        dist = np.array([v[n] for n in DIST_TERMS], np.float64)
        return k, dist, x[:3].copy(), x[3:6].copy()

    def resid(x: F64) -> F64:
        return (_project(obj, *unpack(x)) - img).reshape(-1)

    x0 = np.concatenate([start.rvec, start.tvec, [base[n] for n in names]])
    sol = least_squares(resid, x0, method="lm", x_scale="jac")
    return _finish(model, *unpack(np.asarray(sol.x, np.float64)), ids, obj, img)


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


def _monotonic(fit: Fit) -> str | None:
    """None if the lens stays monotonic, with a 5% margin, out past the whole field
    perimeter at carpet height and at 1.2 m (robot and tag height), else the reason.
    Panel pixels beyond that (walls, ceiling) may lie past the fold; `Camera` maps them
    to NaN rather than to a wrong place."""
    cam = fit.camera()
    rim = _perimeter()
    pts = np.concatenate([rim, rim + np.array([0.0, 0.0, 1.2])]) @ cam.r.T + cam.t
    front = pts[:, 2] > 0.3
    r_max = float(np.max(np.hypot(pts[front, 0], pts[front, 1]) / pts[front, 2]))
    fold = fold_radius(fit.dist)
    if r_max * 1.05 >= fold:
        return f"lens folds at radius {fold:.2f}, inside the field perimeter (radius {r_max:.2f})"
    return None


def _holdout(fit: Fit, tags: dict[int, Tag], corners: dict[int, F64]) -> dict[int, float]:
    """Leave-one-group-out field error per tag in metres (inf when the refit cannot
    cast the held-out points, e.g. its lens folds before reaching them)."""
    out: dict[int, float] = {}
    for group in _clusters(list(fit.tags), corners, GROUP_M, same_facing=False):
        rest = tuple(t for t in fit.tags if t not in group)
        if len(rest) < 3:
            out.update(dict.fromkeys(group, math.inf))
            continue
        obj, img = _points(tags, rest)
        cam = _refine(fit.model, fit, obj, img, rest).camera()
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
    tags: dict[int, Tag], corners: dict[int, F64], rect: PixelRect, start: Fit | None = None
) -> tuple[Fit | None, dict[int, float], list[str]]:
    """Climb the lens ladder from each seed in turn: `start` (an earlier fit with its
    principal point moved back to the panel centre), then calibrateCamera solves. Each
    rung starts from the one below. The first rung that is plausible and monotonic is the
    base; a freer rung replaces it only with a clearly lower held-out error. Returns the
    chosen fit, its hold-out errors and the reasons it should not be trusted (empty when
    it passes every gate)."""
    ids = tuple(sorted(tags))
    spread = _spread(list(ids), corners)
    if len(ids) < MIN_TAGS["pp_k1"]:
        return None, {}, [f"only {len(ids)} usable tags (need {MIN_TAGS['pp_k1']})"]
    obj, img = _points(tags, ids)
    seeds = _seeds(obj, img, rect, ids)
    if start is not None:
        k = start.k.copy()
        k[0, 2], k[1, 2] = rect.x + rect.w / 2, rect.y + rect.h / 2
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
            reason = _plausible(cand, corners, obj) or _monotonic(cand)
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
    problems: list[str] = []
    if spread < MIN_SPREAD_M:
        problems.append(f"tags span only {spread:.1f} m of field (need {MIN_SPREAD_M:g})")
    if best.rms > MAX_RMS_PX:
        problems.append(f"reprojection RMS {best.rms:.2f} px > {MAX_RMS_PX:g}")
    if best.max > MAX_RESID_PX:
        problems.append(f"worst point residual {best.max:.1f} px > {MAX_RESID_PX:g}")
    errs = list(holdout.values()) or [math.inf]
    if float(np.median(errs)) > MAX_HOLDOUT_MEDIAN_M:
        problems.append(f"median held-out error {np.median(errs):.2f} m > {MAX_HOLDOUT_MEDIAN_M:g}")
    if _mean(holdout) > MAX_HOLDOUT_MEAN_M:
        problems.append(
            f"capped mean held-out error {_mean(holdout):.2f} m > {MAX_HOLDOUT_MEAN_M:g}"
        )
    return best, holdout, problems


def _calibrate_view(
    det: Detector, medians: list[U8], rect: PixelRect, corners: dict[int, F64]
) -> ViewResult:
    decoded_hits = _decode_view(det, medians, rect)
    decoded, suspect = _merge(decoded_hits, corners, located=False)
    tags = decoded
    fit, holdout, problems = _solve(tags, corners, rect)
    centres = {t: c.mean(0)[None] for t, c in corners.items()}
    for _ in range(LOCATE_ROUNDS):
        if fit is None:
            break
        located, _ = _merge(
            _locate(medians, rect, fit, corners, set(decoded_hits)), centres, located=True
        )
        if not located:
            break
        tags = decoded | located
        fit, holdout, problems = _solve(tags, corners, rect, fit)
    spread = _spread(list(fit.tags), corners) if fit else 0.0
    return ViewResult(fit, holdout, spread, problems, tags, suspect)


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
        spread_m=round(spread, 3),
        holdout_m={t: round(v, 3) if math.isfinite(v) else None for t, v in holdout.items()},
        camera_xyz=[round(float(v), 3) for v in centre],
    )


# --- inputs and the check image --------------------------------------------------


def _medians(run: RunDir, info: VideoInfo, views: dict[ViewName, PixelRect]) -> tuple[list[U8], U8]:
    """Grey full-frame medians of each quarter of the video and of the whole video.
    Only the rows holding camera panels are decoded; the rest stay black."""
    y0 = min(r.y for r in views.values())
    y1 = max(r.y + r.h for r in views.values())
    band = PixelRect(x=0, y=y0, w=info.width, h=y1 - y0)
    margin = min(3.0, info.duration_s / 10)
    grey = [
        cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY)
        for f in frames(
            run.video,
            info,
            sample_hz=SAMPLE_HZ,
            crop=band,
            start=margin,
            end=info.duration_s - margin,
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
Point3 = tuple[float, float, float]
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
    """Draw one view's field lines, predicted tag outlines (green used, orange not used)
    and the measured tag points (white) onto `img`, clipped to the panel."""
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
    status = "ok" if not res.problems else "REJECTED (" + "; ".join(res.problems) + ")"
    return (
        f"{name}: {status}\n  model {f.model}, tags {list(f.tags)} (located {located}), "
        f"spread {res.spread:.1f} m, rms {f.rms:.2f} px, max {f.max:.2f} px, f {f.k[0, 0]:.0f}, "
        f"c ({f.k[0, 2]:.0f}, {f.k[1, 2]:.0f}), k1 {f.dist[0]:.4f}, k2 {f.dist[1]:.4f}\n"
        f"  camera ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f}) m; held-out error m {held}; "
        f"suspect tags {res.suspect}"
    )


def calibrate(run: RunDir) -> None:
    info = VideoInfo.load(run.video_json)
    layout = _Layout.model_validate_json(run.match_json.read_bytes())
    corners = tag_corners()
    quarters, median = _medians(run, info, layout.views)
    det = _detector()
    results: dict[ViewName, ViewResult] = {}
    for name in VIEW_NAMES:
        if name in layout.views:
            results[name] = _calibrate_view(det, quarters, layout.views[name], corners)
            print("calibrate: " + _report(name, results[name]))
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
