"""Run folders and the pipeline runner.

A run folder (`runs/<video-stem>/`) holds every stage's output. Stages are files in,
files out: each reads earlier stages' files through `RunDir` and writes its own.

A stage module declares itself as

    STAGE = Stage("track", inputs=lambda r: (r.detections, r.calibration_json, r.match_json),
                  outputs=lambda r: (r.tracks,), fn=track)

where `fn(run: RunDir) -> None` writes every output. The runner skips a stage whose
outputs all exist and are newer than its inputs, deletes a stage's outputs before running
it, times it, records it in `manifest.json`, and stops at the first stage that does not
finish. A stage that cannot proceed safely raises `NeedsInput`.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from scout.assets import ROOT, asset_hashes
from scout.schemas.calibration import CORNER_CALIBRATION_FILE, Calibration, CornerCalibrations
from scout.schemas.common import SchemaError, atomic_write
from scout.schemas.detections import read_detections
from scout.schemas.facts import Facts
from scout.schemas.grade import Grade
from scout.schemas.identity import CORNER_IDENTITY_FILE, CornerIdentity, Identity, read_reads
from scout.schemas.manifest import Answers, Manifest, Settings, StageRecord, StageStatus
from scout.schemas.match import MatchInfo
from scout.schemas.tracks import CORNER_TRACKS_FILE, read_tracks
from scout.schemas.video import VideoInfo

RUNS_DIR = ROOT / "runs"

EXIT_CODES: dict[StageStatus, int] = {
    "ok": 0,
    "skipped": 0,
    "failed": 1,
    "needs_input": 2,
    "not_implemented": 3,
}


class NeedsInput(Exception):  # noqa: N818  (name fixed by the stage contract)
    """Raised by a stage that cannot proceed safely without a human answer.

    `image_path` shows the human what the question is about (write it into the run
    folder). `how_to_answer` is the exact command, e.g.
    `scout answer runs/q16 calibrate.main_ok=yes`. The runner prints all three and exits
    with code 2. On the rerun the stage finds the answer with `run.answer(key)`.
    """

    def __init__(self, message: str, image_path: Path | None, how_to_answer: str) -> None:
        super().__init__(message)
        self.message = message
        self.image_path = image_path
        self.how_to_answer = how_to_answer


class RunDir:
    """Paths of every file in one run folder, plus answers and settings access."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.settings_json = path / "settings.json"
        self.answers_json = path / "answers.json"
        self.manifest_json = path / "manifest.json"
        self.video_json = path / "video.json"
        self.match_json = path / "match.json"
        self.calibration_json = path / "calibration.json"
        self.corner_calibration_json = path / CORNER_CALIBRATION_FILE
        self.detections = path / "detections.parquet"
        self.tracks = path / "tracks.parquet"
        self.identity_json = path / "identity.json"
        self.reads = path / "reads.parquet"
        self.corner_tracks = path / CORNER_TRACKS_FILE
        self.corner_identity_json = path / CORNER_IDENTITY_FILE
        self.facts_json = path / "facts.json"
        self.overlay = path / "overlay.mp4"
        self.grade_json = path / "grade.json"

    @property
    def video(self) -> Path:
        """The source video: a `source.<ext>` symlink created by `prepare_run`."""
        found = sorted(self.path.glob("source.*"))
        if not found:
            raise FileNotFoundError(f"{self.path} has no source video link")
        return found[0]

    def settings(self) -> Settings:
        return Settings.load(self.settings_json)

    def answers(self) -> dict[str, str]:
        if not self.answers_json.exists():
            return {}
        return Answers.model_validate_json(self.answers_json.read_bytes()).root

    def answer(self, key: str) -> str | None:
        """The saved answer for `key`, or None. Check this before raising NeedsInput."""
        return self.answers().get(key)

    def write_answers(self, new: dict[str, str]) -> None:
        """Merge `new` into answers.json."""
        merged = Answers({**self.answers(), **new})
        atomic_write(self.answers_json, merged.model_dump_json(indent=2).encode() + b"\n")


@dataclass(frozen=True)
class Stage:
    """One pipeline stage. `inputs` and `outputs` map a run folder to file paths."""

    name: str
    inputs: Callable[[RunDir], Sequence[Path]]
    outputs: Callable[[RunDir], Sequence[Path]]
    fn: Callable[[RunDir], None]


def prepare_run(video: Path, team: int, grader_model: str = "opus") -> RunDir:
    """Create or reuse `runs/<video-stem>/`: link the source video and save settings.

    settings.json is rewritten only when it changes, so stages that list it as an input
    rerun exactly when a setting changed.
    """
    video = video.resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    run = RunDir(RUNS_DIR / video.stem)
    run.path.mkdir(parents=True, exist_ok=True)
    link = run.path / f"source{video.suffix}"
    existing = sorted(run.path.glob("source.*"))
    if existing and existing[0].resolve() != video:
        raise ValueError(f"{run.path} belongs to {existing[0].resolve()}, not {video}")
    if not existing:
        link.symlink_to(video)
    settings = Settings(team=team, grader_model=grader_model)
    if not run.settings_json.exists() or run.settings() != settings:
        settings.save(run.settings_json)
    return run


StageState = Literal["fresh", "stale", "missing"]


def stage_state(run: RunDir, stage: Stage) -> StageState:
    """`fresh`: all outputs exist and none is older than any input. `missing`: some output
    does not exist. `stale`: outputs exist but an input is newer or missing."""
    outs = stage.outputs(run)
    if not all(p.exists() for p in outs):
        return "missing"
    ins = stage.inputs(run)
    if not all(p.exists() for p in ins):
        return "stale"
    newest_in = max((p.stat().st_mtime_ns for p in ins), default=0)
    oldest_out = min(p.stat().st_mtime_ns for p in outs)
    return "fresh" if oldest_out >= newest_in else "stale"


def run_pipeline(run: RunDir, stages: Sequence[Stage], force: Collection[str] = ()) -> int:
    """Run `stages` in order, skipping fresh ones not named in `force`. Stops at the first
    stage that does not finish, then validates every stage file. Returns the exit code."""
    unknown = set(force) - {s.name for s in stages}
    if unknown:
        raise ValueError(f"unknown stage(s) {sorted(unknown)}")
    code = 0
    for stage in stages:
        if stage.name not in force and stage_state(run, stage) == "fresh":
            record = StageRecord(status="skipped", wall_s=0.0, finished_at=_now())
        else:
            record = _attempt(run, stage)
        _record(run, stage.name, record)
        _print_record(stage.name, record)
        code = EXIT_CODES[record.status]
        if code != 0:
            break
    print("\nvalidate:")
    invalid = print_validation(run)
    return code if code != 0 else (1 if invalid else 0)


def run_stage(run: RunDir, stage: Stage) -> int:
    """Run one stage regardless of freshness and record it. Returns the exit code."""
    record = _attempt(run, stage)
    _record(run, stage.name, record)
    _print_record(stage.name, record)
    return EXIT_CODES[record.status]


def _attempt(run: RunDir, stage: Stage) -> StageRecord:
    missing = [str(p) for p in stage.inputs(run) if not p.exists()]
    if missing:
        return _finish(stage, "failed", 0.0, f"missing inputs: {', '.join(missing)}")
    for p in stage.outputs(run):
        p.unlink(missing_ok=True)
    start = time.perf_counter()
    try:
        stage.fn(run)
    except NeedsInput as e:
        image = f"\n  image: {e.image_path}" if e.image_path else ""
        print(f"\nscout: stage {stage.name} needs input\n  {e.message}{image}")
        print(f"  answer with: {e.how_to_answer}\n")
        return _finish(stage, "needs_input", time.perf_counter() - start, e.message)
    except NotImplementedError as e:
        msg = f"stage {stage.name} is not implemented yet ({e})"
        print(f"\nscout: {msg}\n")
        return _finish(stage, "not_implemented", time.perf_counter() - start, msg)
    except Exception as e:
        traceback.print_exc()
        return _finish(stage, "failed", time.perf_counter() - start, f"{type(e).__name__}: {e}")
    wall = time.perf_counter() - start
    unwritten = [str(p) for p in stage.outputs(run) if not p.exists()]
    if unwritten:
        return _finish(stage, "failed", wall, f"did not write {', '.join(unwritten)}")
    return _finish(stage, "ok", wall, None)


def _finish(stage: Stage, status: StageStatus, wall: float, message: str | None) -> StageRecord:
    return StageRecord(status=status, wall_s=round(wall, 3), finished_at=_now(), message=message)


def _now() -> datetime:
    return datetime.now(UTC)


def _print_record(name: str, record: StageRecord) -> None:
    detail = f"  {record.message}" if record.status == "failed" and record.message else ""
    print(f"{name:<10} {record.status:<16} {record.wall_s:8.2f} s{detail}")


def _record(run: RunDir, name: str, record: StageRecord) -> None:
    """Update manifest.json with one stage record and fresh provenance."""
    stages: dict[str, StageRecord] = {}
    if run.manifest_json.exists():
        try:
            stages = dict(Manifest.load(run.manifest_json).stages)
        except ValidationError:
            stages = {}
    stages[name] = record
    sha = VideoInfo.load(run.video_json).sha256 if run.video_json.exists() else None
    Manifest(
        video_path=str(run.video.resolve()),
        video_sha256=sha,
        settings=run.settings(),
        assets=asset_hashes(),
        stages=stages,
        updated_at=_now(),
    ).save(run.manifest_json)


def schema_checks(run: RunDir) -> list[tuple[Path, Callable[[Path], object]]]:
    """Every schema-checked file in a run folder with the loader that validates it."""
    return [
        (run.settings_json, Settings.load),
        (run.answers_json, lambda p: Answers.model_validate_json(p.read_bytes())),
        (run.manifest_json, Manifest.load),
        (run.video_json, VideoInfo.load),
        (run.match_json, MatchInfo.load),
        (run.calibration_json, Calibration.load),
        (run.corner_calibration_json, CornerCalibrations.load),
        (run.detections, read_detections),
        (run.tracks, read_tracks),
        (run.identity_json, Identity.load),
        (run.reads, read_reads),
        (run.corner_tracks, read_tracks),
        (run.corner_identity_json, CornerIdentity.load),
        (run.facts_json, Facts.load),
        (run.grade_json, Grade.load),
    ]


def validate(run: RunDir) -> list[tuple[Path, str]]:
    """Check every file in `schema_checks`: "valid", "absent", or the error text."""
    results: list[tuple[Path, str]] = []
    for path, load in schema_checks(run):
        if not path.exists():
            results.append((path, "absent"))
            continue
        try:
            load(path)
            results.append((path, "valid"))
        except (ValidationError, SchemaError, ValueError, OSError) as e:
            results.append((path, f"{type(e).__name__}: {e}"))
    return results


def print_validation(run: RunDir) -> bool:
    """Print `validate(run)`; return True if any file is invalid."""
    invalid = False
    for path, result in validate(run):
        invalid |= result not in ("valid", "absent")
        print(f"  {path.name:<20} {result}")
    return invalid
