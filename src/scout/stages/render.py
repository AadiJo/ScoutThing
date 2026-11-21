"""render: reads video, tracks.parquet, identity.json, calibration.json, match.json.

Writes overlay.mp4: the match with robot boxes beside a top-down field map.
"""

from __future__ import annotations

from scout.run import RunDir, Stage


def render(run: RunDir) -> None:
    raise NotImplementedError("render")


STAGE = Stage(
    "render",
    inputs=lambda r: (r.video, r.tracks, r.identity_json, r.calibration_json, r.match_json),
    outputs=lambda r: (r.overlay,),
    fn=render,
)
