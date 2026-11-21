"""track: reads detections.parquet, calibration.json, match.json.

Writes tracks.parquet: six robot tracks in field metres, one row per track per sampled frame.
"""

from __future__ import annotations

from scout.run import RunDir, Stage


def track(run: RunDir) -> None:
    raise NotImplementedError("track")


STAGE = Stage(
    "track",
    inputs=lambda r: (r.detections, r.calibration_json, r.match_json),
    outputs=lambda r: (r.tracks,),
    fn=track,
)
