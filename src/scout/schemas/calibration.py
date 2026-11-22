"""`calibration.json`, written by the calibrate stage: one pinhole camera per view.

Intrinsics are in full-frame pixels, so `cv2.projectPoints(field_xyz, rvec, tvec, K, dist)`
gives full-frame pixel coordinates directly. The pose maps WPILib field metres to the
camera frame (OpenCV convention: x right, y down, z forward). Distortion is radial
(k1, k2, and k3 for `c_k3`). Use `scout.camera.Camera` rather than OpenCV to go back from
pixels to the field: it inverts the distortion exactly and refuses rays past the lens fold.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from scout.schemas.common import StageFile, ViewName

LensModel = Literal["pp_k1", "cy_k1", "c_k1", "c_k2", "pp_k2", "c_k3"]
"""Free lens parameters. `pp_k1`: focal length and k1, principal point at the view's
expected optical centre (the frame centre for a panel that is the top of a full-frame
picture, else the panel centre). `cy_k1`: also the principal point's row. `c_k1`: the
whole principal point. `c_k2`: also k2. `pp_k2`: focal length, k1 and k2 with the principal
point fixed as in `pp_k1`. `c_k3`: `c_k2` and k3, fitted only with field lines (on the
Championship lenses it folds inside the panel and is refused). Pixels are square and there
is no tangential term in any model: on the CapCity clips tangential terms fitted the tags
better but bent the untagged ends of the field."""


class LineFit(BaseModel):
    """How one straight field line (`FIELD_LINES` in the calibrate stage) was fitted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    n_points: int
    """Ridge points found along the line's image curve and fitted."""
    rms_px: float
    """Distance RMS of those points from the line's projection, each capped at 10 px."""
    inliers: float
    """Share of the points within 3 px of the projection."""
    height_m: float
    """Fitted height of the line; a guardrail's height is solved within a few centimetres
    because which of its parallel top edges shows brightest is not known."""


class CameraCalibration(BaseModel):
    """Intrinsics, distortion, pose and fit quality of one camera view."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    k: list[list[float]]
    """3x3 camera matrix."""
    dist: list[float]
    """OpenCV distortion coefficients (k1, k2, p1, p2, k3); p1 = p2 = 0, and k3 = 0 except
    for `c_k3`."""
    rvec: list[float]
    """Rodrigues rotation, field to camera."""
    tvec: list[float]
    """Translation in metres, field to camera."""
    rms_px: float
    """Reprojection RMS over the tag points used in the fit."""
    max_px: float
    n_points: int
    model: LensModel
    tags: list[int]
    """AprilTag ids decoded in the view whose four corners were fitted."""
    located: list[int]
    """AprilTag ids too small to decode, found by their white frame where the fit predicted
    them; only their centres were fitted."""
    lines: dict[str, LineFit] = Field(default_factory=dict)
    """Straight field lines fitted along with the tags (empty for a tags-only fit, and in
    files written before lines were used)."""
    spread_m: float
    """Largest field-plane distance between two fitted tags."""
    holdout_m: dict[int, float | None]
    """Held-out error per tag: the view is refitted without the tag's group (tags within
    1 m of it), the tag's points are cast onto the plane at their own height, and this is
    the mean distance in metres from their true positions. None when the refitted lens
    cannot reach the tag at all (it folds first): that corner depends on the tag itself."""
    camera_xyz: list[float]
    """Camera centre in field metres."""


class Calibration(StageFile):
    """Calibrated views. A view missing from `views` could not be calibrated."""

    schema_version: Literal[1] = 1
    views: dict[ViewName, CameraCalibration]
