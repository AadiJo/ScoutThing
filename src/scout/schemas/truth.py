"""`truth/<video-stem>.json` at the repo root: robots followed by hand (or imported) through
one match video, kept in git and scored by `scout bench`.

Truth stays in image space on purpose: `u`, `v` are full-frame pixels of the source video
and `frame`, `t` are the source frame index and video seconds exactly as `scout.video`
defines them. `scout bench` casts each sample through the run's calibration onto the plane
`click_height_m` above the carpet, so a recalibration never invalidates truth.

- `state` `visible`: the annotator could see the robot. `estimated`: it was hidden and the
  annotator kept the cursor where they believed it was.
- `view` is the panel the pixel lies in; only `main` is calibrated, so bench skips others.
- Samples of one robot are in strictly increasing `frame` order; each team appears once.

`Truth.save` writes one sample per line, so a densely followed match (about 60 samples per
second per robot) stays small enough for git and diffs line by line.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from scout.schemas.common import StageFile, ViewName, atomic_write

TruthState = Literal["visible", "estimated"]


class TruthSample(BaseModel):
    """Where the annotator held the cursor on one source frame."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    frame: int
    t: float
    view: ViewName
    u: float
    v: float
    state: TruthState


class TruthRobot(BaseModel):
    """One robot's samples, in increasing frame order."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    team: int
    samples: list[TruthSample]

    @model_validator(mode="after")
    def _frames_increase(self) -> Self:
        frames = [s.frame for s in self.samples]
        bad = next((i for i in range(1, len(frames)) if frames[i] <= frames[i - 1]), None)
        if bad is not None:
            raise ValueError(
                f"team {self.team}: sample frames must strictly increase "
                f"(frame {frames[bad]} after {frames[bad - 1]})"
            )
        return self


class Truth(StageFile):
    """Truth for one match video. `video_sha256` is the `sha256` in that video's video.json."""

    schema_version: Literal[1] = 1
    video_sha256: str
    video_name: str
    source: Literal["hand", "imported"]
    click_height_m: float
    """Height above the carpet of the point the annotator follows."""
    notes: str
    robots: list[TruthRobot]

    @model_validator(mode="after")
    def _teams_unique(self) -> Self:
        teams = [r.team for r in self.robots]
        if len(set(teams)) != len(teams):
            raise ValueError(f"each team may appear once in robots, got {teams}")
        return self

    def robot(self, team: int) -> TruthRobot | None:
        return next((r for r in self.robots if r.team == team), None)

    def save(self, path: Path) -> None:
        """Pretty JSON with one compact sample per line."""
        head = self.model_dump(mode="json", exclude={"robots"})
        lines = ["{"] + [f"  {json.dumps(k)}: {json.dumps(v)}," for k, v in head.items()]
        robots: list[str] = []
        for r in self.robots:
            samples = ",\n".join(
                "      " + json.dumps(s.model_dump(mode="json"), separators=(",", ":"))
                for s in r.samples
            )
            body = f"\n{samples}\n    " if r.samples else ""
            robots.append(f'    {{"team": {r.team}, "samples": [{body}]}}')
        lines.append('  "robots": [' + ("\n" + ",\n".join(robots) + "\n  " if robots else "") + "]")
        lines.append("}")
        atomic_write(path, ("\n".join(lines) + "\n").encode())
