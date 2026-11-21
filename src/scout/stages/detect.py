"""detect: reads video, video.json, match.json.

Writes detections.parquet: raw bumper boxes per sampled frame per view in full-frame
pixels, unfiltered.
"""

from __future__ import annotations

from scout.run import RunDir, Stage


def detect(run: RunDir) -> None:
    raise NotImplementedError("detect")


STAGE = Stage(
    "detect",
    inputs=lambda r: (r.video, r.video_json, r.match_json),
    outputs=lambda r: (r.detections,),
    fn=detect,
)
