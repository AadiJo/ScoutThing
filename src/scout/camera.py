"""Move between full-frame pixels and WPILib field metres through one calibrated view.

    cam = Camera.load(run.calibration_json, "main")
    left = Camera.load(run.calibration_json, "corner_left")  # from corner_calibration.json
    uv = cam.project(xyz)              # (..., 3) field metres -> (..., 2) full-frame pixels
    xy = cam.to_field(uv, z=0.08)      # (..., 2) pixels -> (..., 2) field (x, y) on plane z

Conventions:
- Field points are WPILib field metres: blue alliance wall at x=0, x along the length,
  y across, z up from the carpet.
- Pixels are full-frame pixels of the source video (not relative to a view's panel),
  x right, y down, with the same origin as `cv2.projectPoints` (pixel centres at integers).
- The model is OpenCV's pinhole camera: camera frame x right, y down, z forward;
  `pc = R @ p + t` with `R = Rodrigues(rvec)`, normalised `(x, y) = pc[:2] / pc[2]`.
- Distortion is one of two families, chosen by the calibration's lens model:
  - polynomial (OpenCV's five-term model) `dist = (k1, k2, p1, p2, k3)`: with r^2 =
    x^2 + y^2 and g = 1 + k1 r^2 + k2 r^4 + k3 r^6,
    `xd = x g + 2 p1 x y + p2 (r^2 + 2 x^2)`, `yd = y g + p1 (r^2 + 2 y^2) + 2 p2 x y`;
    `project` equals `cv2.projectPoints`.
  - fisheye (Kannala-Brandt, OpenCV's fisheye model) `dist = (k1, k2, k3, k4)`: with
    theta = atan(r), `(xd, yd) = (x, y) theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6
    + k4 theta^8) / r`; `project` equals `cv2.fisheye.projectPoints`.
  Then `u = fx xd + cx`, `v = fy yd + cy`. `to_field` inverts the lens to 1e-10 in
  normalised units (radial bisection, then Newton on the full polynomial model), not with
  OpenCV's few fixed-point steps.
- The lens is used only inside its monotonic region, out to `r_fold`, the undistorted
  radius where the radial mapping stops increasing (tangential terms are small and
  ignored there). Past it a point outside the field of view would fold back into the
  image, so such points project to NaN, as do points less than 0.3 m in front of the
  camera. A pixel past the fold, or whose ray is parallel to the plane or meets it behind
  the camera, maps to NaN.

All methods are vectorised over leading dimensions and return float64 arrays.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from scout.schemas.calibration import (
    CORNER_CALIBRATION_FILE,
    FISHEYE_MODELS,
    Calibration,
    CameraCalibration,
    CornerCalibrations,
)
from scout.schemas.common import ViewName

F64 = NDArray[np.float64]
NEAR_M = 0.3
_R_SEARCH_MAX = 20.0  # normalised radius searched for a fold (about 87 degrees off axis)


def dist5(dist: ArrayLike) -> tuple[float, float, float, float, float]:
    """(k1, k2, p1, p2, k3) from an OpenCV coefficient list of up to 5 terms."""
    src = np.asarray(dist, np.float64).reshape(-1)
    if len(src) > 5 and np.any(src[5:] != 0.0):
        raise ValueError(f"only the five-term distortion model is supported, got {src}")
    d = np.zeros(5)
    d[: min(5, len(src))] = src[:5]
    return float(d[0]), float(d[1]), float(d[2]), float(d[3]), float(d[4])


def lens_terms(dist: ArrayLike, fisheye: bool) -> tuple[float, ...]:
    """The coefficients of a lens family: (k1, k2, p1, p2, k3) polynomial or
    (k1, k2, k3, k4) fisheye, from a list of up to that many terms."""
    if not fisheye:
        return dist5(dist)
    src = np.asarray(dist, np.float64).reshape(-1)
    if len(src) > 4 and np.any(src[4:] != 0.0):
        raise ValueError(f"the fisheye model has four coefficients, got {src}")
    d = np.zeros(4)
    d[: min(4, len(src))] = src[:4]
    return tuple(float(v) for v in d)


def radial(dist: ArrayLike, fisheye: bool, r: F64) -> F64:
    """Distorted normalised radius of undistorted radius `r` (tangential terms ignored)."""
    if fisheye:
        k1, k2, k3, k4 = lens_terms(dist, True)
        th = np.arctan(r)
        t2 = th * th
        return th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    k1, k2, _, _, k3 = dist5(dist)
    r2 = r * r
    return r * (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3)


def fold_radius(dist: ArrayLike, fisheye: bool = False) -> float:
    """Undistorted normalised radius where the radial mapping stops increasing, or inf if
    it never does within about 87 degrees off axis. Used by `Camera` and the calibrate
    stage."""
    if fisheye:
        k1, k2, k3, k4 = lens_terms(dist, True)
        th = np.linspace(0.0, math.atan(_R_SEARCH_MAX), 200_001)
        t2 = th * th
        slope = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
        bad = np.flatnonzero(slope <= 0)
        return float("inf") if len(bad) == 0 else math.tan(float(th[bad[0]]))
    k1, k2, _, _, k3 = dist5(dist)
    r = np.linspace(0.0, _R_SEARCH_MAX, 200_001)
    r2 = r * r
    gain = 1 + 3 * k1 * r2 + 5 * k2 * r2**2 + 7 * k3 * r2**3
    bad = np.flatnonzero(gain <= 0)
    return float("inf") if len(bad) == 0 else float(r[bad[0]])


def reach_radius(dist: ArrayLike, fisheye: bool = False) -> float:
    """Largest distorted normalised radius the lens reaches before it folds (inf if it never
    folds; a fisheye that never folds reaches its value about 87 degrees off axis). A pixel
    farther than this from the principal point maps to no ray."""
    r = fold_radius(dist, fisheye)
    if not np.isfinite(r):
        if not fisheye:
            return float("inf")
        r = _R_SEARCH_MAX
    return float(radial(dist, fisheye, np.array(r)))


@dataclass(frozen=True)
class Camera:
    """One calibrated view. Build it with `Camera.load` or `Camera.from_calibration`."""

    k: F64
    """3x3 camera matrix in full-frame pixels."""
    dist: tuple[float, ...]
    """(k1, k2, p1, p2, k3), or (k1, k2, k3, k4) when `fisheye`."""
    fisheye: bool
    """Kannala-Brandt fisheye lens rather than the polynomial one."""
    r: F64
    """3x3 rotation, field to camera."""
    t: F64
    """(3,) translation in metres, field to camera."""
    r_fold: float
    """Undistorted normalised radius the lens is valid out to (inf: everywhere)."""

    @classmethod
    def from_calibration(cls, cal: CameraCalibration) -> Camera:
        fisheye = cal.model in FISHEYE_MODELS
        return cls(
            k=np.asarray(cal.k, np.float64).reshape(3, 3),
            dist=lens_terms(cal.dist, fisheye),
            fisheye=fisheye,
            r=_rodrigues(np.asarray(cal.rvec, np.float64).reshape(3)),
            t=np.asarray(cal.tvec, np.float64).reshape(3),
            r_fold=fold_radius(cal.dist, fisheye),
        )

    @classmethod
    def load(cls, calibration_json: Path, view: ViewName = "main") -> Camera:
        """The camera of `view` in a calibration.json, or, for a corner view it lacks, in the
        corner_calibration.json beside it (corners stage; check that file's `trusted_m`
        before placing a robot). KeyError if neither has the view."""
        views = Calibration.load(calibration_json).views
        if view in views:
            return cls.from_calibration(views[view])
        corner = calibration_json.with_name(CORNER_CALIBRATION_FILE)
        if corner.exists():
            found = CornerCalibrations.load(corner).views.get(view)
            if found is not None:
                return cls.from_calibration(found.camera)
        raise KeyError(view)

    @property
    def center(self) -> F64:
        """Camera centre in field metres, (3,)."""
        return -self.r.T @ self.t

    def project(self, xyz: ArrayLike) -> F64:
        """Field points (..., 3) to full-frame pixels (..., 2). NaN where the point is
        behind or within 0.3 m of the camera, or past the lens fold. Pixels outside the
        frame are returned as they are; clip them yourself."""
        p = np.asarray(xyz, np.float64)
        pc = p @ self.r.T + self.t
        z = pc[..., 2]
        front = z > NEAR_M
        zs = np.where(front, z, 1.0)
        x, y = pc[..., 0] / zs, pc[..., 1] / zs
        ok = front & (np.hypot(x, y) < self.r_fold)
        xd, yd = self._distort(x, y)
        u = self.k[0, 0] * xd + self.k[0, 1] * yd + self.k[0, 2]
        v = self.k[1, 1] * yd + self.k[1, 2]
        return np.where(ok[..., None], np.stack([u, v], axis=-1), np.nan)

    def rays(self, uv: ArrayLike) -> F64:
        """Full-frame pixels (..., 2) to unit ray directions in field axes (..., 3), NaN
        for pixels past the lens fold."""
        q = np.asarray(uv, np.float64)
        yd = (q[..., 1] - self.k[1, 2]) / self.k[1, 1]
        xd = (q[..., 0] - self.k[0, 2] - self.k[0, 1] * yd) / self.k[0, 0]
        x, y = self._undistort(xd, yd)
        d = np.stack([x, y, np.ones_like(x)], axis=-1) @ self.r
        return d / np.linalg.norm(d, axis=-1, keepdims=True)

    def to_field(self, uv: ArrayLike, z: ArrayLike = 0.0) -> F64:
        """Full-frame pixels (..., 2) to the field (x, y) (..., 2) where each pixel's ray
        meets the horizontal plane at height `z` metres (a scalar or broadcastable to
        `uv[..., 0]`). NaN when the ray is past the lens fold, parallel to the plane, or
        meets it behind the camera."""
        d = self.rays(uv)
        c = self.center
        dz = d[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (np.asarray(z, np.float64) - c[2]) / dz
        ok = np.isfinite(s) & (s > 0) & (np.abs(dz) > 1e-9)
        return np.where(ok[..., None], c[:2] + s[..., None] * d[..., :2], np.nan)

    def _gain(self, r2: F64) -> F64:
        k1, k2, _, _, k3 = self.dist
        return 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3

    def _distort(self, x: F64, y: F64) -> tuple[F64, F64]:
        if self.fisheye:
            r = np.hypot(x, y)
            safe = np.where(r > 0, r, 1.0)
            s = np.where(r > 0, radial(self.dist, True, r) / safe, 1.0)
            return x * s, y * s
        _, _, p1, p2, _ = self.dist
        r2 = x * x + y * y
        g = self._gain(r2)
        return x * g + 2 * p1 * x * y + p2 * (r2 + 2 * x * x), y * g + p1 * (
            r2 + 2 * y * y
        ) + 2 * p2 * x * y

    def _undistort(self, xd: F64, yd: F64) -> tuple[F64, F64]:
        """Invert `_distort`: exact radial inverse by bisection on [0, r_fold), then (for
        the polynomial model) Newton steps on the full model. NaN past the fold's reach or
        where Newton fails."""
        hi_r = min(self.r_fold, _R_SEARCH_MAX)
        reach = float(radial(self.dist, self.fisheye, np.array(hi_r)))
        rd = np.hypot(xd, yd)
        lo, hi = np.zeros_like(rd), np.full_like(rd, hi_r)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            below = radial(self.dist, self.fisheye, mid) < rd
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        scale = np.where(rd > 0, 0.5 * (lo + hi) / np.where(rd > 0, rd, 1.0), 1.0)
        x, y = xd * scale, yd * scale
        if self.fisheye:
            good = (rd < reach) & (np.hypot(x, y) < self.r_fold)
            return np.where(good, x, np.nan), np.where(good, y, np.nan)
        k1, k2, p1, p2, k3 = self.dist
        for _ in range(20):
            ex, ey = self._distort(x, y)
            ex, ey = ex - xd, ey - yd
            r2 = x * x + y * y
            g = self._gain(r2)
            gs = k1 + 2 * k2 * r2 + 3 * k3 * r2 * r2
            a = g + 2 * x * x * gs + 2 * p1 * y + 6 * p2 * x
            b = 2 * x * y * gs + 2 * p1 * x + 2 * p2 * y
            d = g + 2 * y * y * gs + 6 * p1 * y + 2 * p2 * x
            det = a * d - b * b
            x, y = x - (d * ex - b * ey) / det, y - (a * ey - b * ex) / det
        ex, ey = self._distort(x, y)
        good = (rd < reach) & (np.hypot(ex - xd, ey - yd) < 1e-10) & (np.hypot(x, y) < self.r_fold)
        return np.where(good, x, np.nan), np.where(good, y, np.nan)


def _rodrigues(rvec: F64) -> F64:
    """Rotation matrix of a Rodrigues vector (same as `cv2.Rodrigues`)."""
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3)
    a = rvec / theta
    kx = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(theta) * kx + (1 - np.cos(theta)) * (kx @ kx)
