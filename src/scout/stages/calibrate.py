"""calibrate: reads video, video.json, match.json.

Writes calibration.json: per camera view intrinsics, distortion, pose and residuals.
"""

from __future__ import annotations

from scout.run import RunDir, Stage


def calibrate(run: RunDir) -> None:
    raise NotImplementedError("calibrate")


STAGE = Stage(
    "calibrate",
    inputs=lambda r: (r.video, r.video_json, r.match_json),
    outputs=lambda r: (r.calibration_json,),
    fn=calibrate,
)
