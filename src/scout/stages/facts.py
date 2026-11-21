"""facts: reads tracks.parquet, identity.json, match.json.

Writes facts.json: timed events for the target team.
"""

from __future__ import annotations

from scout.run import RunDir, Stage


def facts(run: RunDir) -> None:
    raise NotImplementedError("facts")


STAGE = Stage(
    "facts",
    inputs=lambda r: (r.tracks, r.identity_json, r.match_json),
    outputs=lambda r: (r.facts_json,),
    fn=facts,
)
