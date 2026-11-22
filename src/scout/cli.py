"""The `scout` command line. Exit codes: 0 done, 1 failed or invalid, 2 needs input,
3 stopped at a stage that is not implemented yet."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from scout.annotate.server import serve
from scout.assets import ROOT, fetch_assets
from scout.bench import BenchRefusedError, run_bench
from scout.pipeline import BY_NAME, STAGES
from scout.run import RunDir, prepare_run, print_validation, run_pipeline, run_stage, stage_state
from scout.schemas.manifest import Manifest
from scout.schemas.video import VideoInfo

app = typer.Typer(no_args_is_help=True, add_completion=False)

TRUTH_DIR = ROOT / "truth"

RunPath = Annotated[Path, typer.Argument(help="Run folder, e.g. runs/Qualification_16.")]


def _run_dir(path: Path) -> RunDir:
    if not path.is_dir():
        raise typer.BadParameter(f"{path} is not a run folder")
    return RunDir(path)


@app.command()
def run(
    video: Annotated[Path, typer.Argument(help="Match video.")],
    team: Annotated[int, typer.Option(help="Target team number.")],
    force: Annotated[
        list[str] | None, typer.Option(help="Rerun this stage even if fresh. Repeatable.")
    ] = None,
    grader_model: Annotated[
        str, typer.Option(help="Model for the grade stage's Claude Code agent.")
    ] = "opus",
) -> None:
    """Run the whole pipeline into runs/<video-stem>/."""
    unknown = sorted(set(force or ()) - set(BY_NAME))
    if unknown:
        raise typer.BadParameter(f"unknown stage(s) {unknown}; one of {', '.join(BY_NAME)}")
    rd = prepare_run(video, team, grader_model)
    print(f"run folder: {rd.path}")
    raise typer.Exit(run_pipeline(rd, STAGES, force or ()))


@app.command()
def stage(
    name: Annotated[str, typer.Argument(help=f"One of: {', '.join(BY_NAME)}.")],
    run_dir: RunPath,
) -> None:
    """Run one stage on an existing run folder, even if its outputs are fresh."""
    if name not in BY_NAME:
        raise typer.BadParameter(f"unknown stage {name!r}; one of {', '.join(BY_NAME)}")
    raise typer.Exit(run_stage(_run_dir(run_dir), BY_NAME[name]))


@app.command()
def status(run_dir: RunPath) -> None:
    """Show each stage's freshness and its last recorded result."""
    rd = _run_dir(run_dir)
    records = Manifest.load(rd.manifest_json).stages if rd.manifest_json.exists() else {}
    print(f"{'stage':<10} {'outputs':<8} {'last':<16} {'wall':>9}  message")
    for s in STAGES:
        rec = records.get(s.name)
        last = rec.status if rec else "-"
        wall = f"{rec.wall_s:.2f} s" if rec else "-"
        msg = (rec.message or "") if rec else ""
        print(f"{s.name:<10} {stage_state(rd, s):<8} {last:<16} {wall:>9}  {msg}")


@app.command()
def answer(
    run_dir: RunPath,
    pairs: Annotated[list[str], typer.Argument(help="key=value answers.")],
) -> None:
    """Save answers to questions a stage asked (NeedsInput) into answers.json."""
    new: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise typer.BadParameter(f"expected key=value, got {pair!r}")
        new[key] = value
    _run_dir(run_dir).write_answers(new)
    for key, value in new.items():
        print(f"saved {key}={value}")


@app.command(name="validate")
def validate_cmd(run_dir: RunPath) -> None:
    """Check every stage file in a run folder against its schema."""
    raise typer.Exit(1 if print_validation(_run_dir(run_dir)) else 0)


@app.command()
def annotate(
    run_dir: RunPath,
    truth: Annotated[
        Path | None,
        typer.Option(help="Truth file to edit. Default: truth/<video-stem>.json in the repo."),
    ] = None,
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8765,
    host: Annotated[str, typer.Option(help="Address to bind; 0.0.0.0 serves the LAN too.")] = (
        "0.0.0.0"
    ),
) -> None:
    """Serve a page for following robots by hand and saving the truth file."""
    rd = _run_dir(run_dir)
    missing = [
        p.name for p in (rd.video_json, rd.match_json, rd.calibration_json) if not p.exists()
    ]
    if missing:
        raise typer.BadParameter(f"{run_dir} is missing {', '.join(missing)}; run scout first")
    stem = Path(VideoInfo.load(rd.video_json).source_path).stem
    serve(rd, truth or TRUTH_DIR / f"{stem}.json", host, port)


@app.command()
def bench(
    run_dir: RunPath,
    truth: Annotated[Path, typer.Argument(help="Truth file, e.g. truth/Qualification_16.json.")],
) -> None:
    """Score a run's tracks and labels against a truth file; writes bench.json and bench.md."""
    try:
        report = run_bench(_run_dir(run_dir), truth)
    except BenchRefusedError as e:
        print(f"scout: {e}")
        raise typer.Exit(1) from e
    print(report)


@app.command(name="fetch-assets")
def fetch_assets_cmd(
    from_local: Annotated[
        Path | None,
        typer.Option(help="Copy from a local clone of the pinned commit instead of downloading."),
    ] = None,
) -> None:
    """Download (or copy) the pinned assets into assets/ and verify their sha256."""
    fetch_assets(from_local)
