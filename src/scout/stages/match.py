"""match: reads video, video.json, settings.json.

Writes match.json: overlay layout and panel rectangles, the six-team roster, target team and
alliance, auto/teleop times in video seconds, and match phases with active hubs.
"""

from __future__ import annotations

from scout.run import RunDir, Stage


def match(run: RunDir) -> None:
    raise NotImplementedError("match")


STAGE = Stage(
    "match",
    inputs=lambda r: (r.video, r.video_json, r.settings_json),
    outputs=lambda r: (r.match_json,),
    fn=match,
)
