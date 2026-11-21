"""`calibration.json`, written by the calibrate stage: one pinhole camera per view.

Intrinsics are in full-frame pixels, so `cv2.projectPoints(field_xyz, rvec, tvec, K, dist)`
gives full-frame pixel coordinates directly. The pose maps WPILib field metres to the
camera frame (OpenCV convention: x right, y down, z forward).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from scout.schemas.common import StageFile, ViewName


class CameraCalibration(BaseModel):
    """Intrinsics, distortion, pose and fit quality of one camera view."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    k: list[list[float]]
    """3x3 camera matrix."""
    dist: list[float]
    """OpenCV distortion coefficients (k1, k2, p1, p2[, k3...])."""
    rvec: list[float]
    """Rodrigues rotation, field to camera."""
    tvec: list[float]
    """Translation in metres, field to camera."""
    rms_px: float
    """Reprojection RMS over the points used in the fit."""
    max_px: float
    n_points: int


class Calibration(StageFile):
    """Calibrated views. A view missing from `views` could not be calibrated."""

    schema_version: Literal[1] = 1
    views: dict[ViewName, CameraCalibration]
