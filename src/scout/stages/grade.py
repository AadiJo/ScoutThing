"""grade: reads facts.json, video, match.json.

Writes grade.json: scores with cited notes, from Claude Code run headless.
"""

from __future__ import annotations

from scout.run import RunDir, Stage


def grade(run: RunDir) -> None:
    raise NotImplementedError("grade")


STAGE = Stage(
    "grade",
    inputs=lambda r: (r.facts_json, r.video, r.match_json),
    outputs=lambda r: (r.grade_json,),
    fn=grade,
)
