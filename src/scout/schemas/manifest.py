"""`settings.json`, `answers.json` and `manifest.json`: the run folder's own files.

`settings.json` holds the command-line choices stages need (the runner rewrites it only
when they change, so its mtime marks stages that depend on it stale). `answers.json`
holds answers to NeedsInput questions. `manifest.json` records what the last runs did.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, RootModel

from scout.schemas.common import StageFile


class Settings(StageFile):
    """Run choices from the command line."""

    schema_version: Literal[1] = 1
    team: int


class Answers(RootModel[dict[str, str]]):
    """Flat `key -> value` answers. Keys are namespaced by stage, e.g. `calibrate.main_ok`."""


StageStatus = Literal["ok", "skipped", "failed", "needs_input", "not_implemented"]


class StageRecord(BaseModel):
    """Outcome of the latest attempt at one stage. `skipped` means outputs were fresh."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: StageStatus
    wall_s: float
    finished_at: datetime
    message: str | None = None


class Manifest(StageFile):
    """Provenance of a run folder. `stages` keeps the latest record per stage, in run order."""

    schema_version: Literal[1] = 1
    video_path: str
    video_sha256: str | None
    """From video.json; None until probe has run."""
    settings: Settings
    assets: dict[str, str | None]
    """Asset name to sha256 of the file on disk, None when missing."""
    stages: dict[str, StageRecord]
    updated_at: datetime
