"""identify: reads video, tracks.parquet, detections.parquet, match.json.

Writes identity.json (team label segments per track) and reads.parquet (raw number reads).
"""

from __future__ import annotations

from scout.run import RunDir, Stage


def identify(run: RunDir) -> None:
    raise NotImplementedError("identify")


STAGE = Stage(
    "identify",
    inputs=lambda r: (r.video, r.tracks, r.detections, r.match_json),
    outputs=lambda r: (r.identity_json, r.reads),
    fn=identify,
)
