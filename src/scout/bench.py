"""`scout bench <run-dir> <truth.json>`: score a run's tracks and team labels against truth.

    report = run_bench(RunDir(Path("runs/Qualification_16")), Path("truth/Qualification_16.json"))

writes `bench.json` (`scout.schemas.bench.Bench`) and `bench.md` into the run folder and
returns the markdown. It refuses truth made on a different video.

How it scores, on the tracks' grid (every `round(fps / SAMPLE_HZ)`-th frame):
- Each robot's truth sample nearest a grid frame, within half a grid step, is cast through
  the main camera onto the plane `click_height_m` above the carpet. Samples outside the
  main view, or that miss the plane, are skipped and counted.
- A track sample's label is the team of the identity segment covering it (none when no
  segment, or segments of two teams, cover it).
- At each gate: label recall (some track within the gate labelled with the team), tracked
  share (any track within the gate), the not-recalled time split by cause, and wrong-label
  time (track samples labelled with the team that lie outside the gate of its truth).
- Position error and identity switches use a one-to-one assignment of truth robots to
  tracks per grid frame (Hungarian on distance, within POSITION_GATE_M).
- Visible and estimated truth are scored separately; only visible is the headline.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment  # pyright: ignore[reportMissingTypeStubs]

from scout.camera import Camera
from scout.run import RunDir
from scout.schemas.bench import (
    Bench,
    GateScore,
    PositionScore,
    RobotBench,
    StateScore,
    Targets,
    WrongStretch,
)
from scout.schemas.detections import SAMPLE_HZ
from scout.schemas.identity import Identity
from scout.schemas.match import MatchInfo
from scout.schemas.tracks import read_tracks
from scout.schemas.truth import Truth, TruthSample
from scout.schemas.video import VideoInfo

F64 = NDArray[np.float64]

GATES_M = (0.5, 1.0)
"""Distance gates for recall, tracked share and wrong labels."""
POSITION_GATE_M = 1.0
"""Truth-to-track assignment gate for position error and identity switches."""
TARGET_GATE_M = 1.0
"""Gate the targets are judged at: robot centres cannot be closer than about 0.9 m, so a
track within 1 m of the truth is on that robot."""
TARGET_MAX_WRONG_SHARE = 0.01
TARGET_MIN_RECALL = 0.70
TARGET_MAX_MEDIAN_M = 0.4
NO_TEAM = 0
"""Label value of an unlabelled (or ambiguously labelled) track sample."""


class BenchRefusedError(Exception):
    """The run and truth cannot be compared (another video, or the run lacks tracks)."""


@dataclass
class _Acc:
    """Counts of grid frames for one robot, one truth state and one gate."""

    truth: int = 0
    hit: int = 0
    near: int = 0
    no_track: int = 0
    unlabelled: int = 0
    other: int = 0
    labelled: int = 0
    wrong: list[tuple[int, int, float, int | None]] = field(
        default_factory=lambda: list[tuple[int, int, float, int | None]]()
    )
    """(grid frame, track id, distance, team the track was actually on)."""


@dataclass
class _Match:
    frame: int
    track_id: int
    dist: float
    los: float
    estimated: bool


@dataclass
class _RobotTruth:
    """One robot's truth on the grid: frame -> (x, y, estimated)."""

    team: int
    at: dict[int, tuple[float, float, bool]]
    samples: int
    skipped: int


def run_bench(run: RunDir, truth_path: Path) -> str:
    """Score `run` against the truth file and write bench.json and bench.md."""
    truth = Truth.load(truth_path)
    info = VideoInfo.load(run.video_json)
    if truth.video_sha256 != info.sha256:
        raise BenchRefusedError(
            f"{truth_path} was made on video sha256 {truth.video_sha256[:12]}... "
            f"({truth.video_name}), but {run.path} holds {info.sha256[:12]}..."
        )
    for p in (run.tracks, run.identity_json, run.calibration_json, run.match_json):
        if not p.exists():
            raise BenchRefusedError(f"{run.path} has no {p.name}; run the pipeline first")
    match = MatchInfo.load(run.match_json)
    cam = Camera.load(run.calibration_json, "main")
    step = max(1, round(info.fps / SAMPLE_HZ))
    dt = step / info.fps

    robots = [_grid_truth(r.team, r.samples, cam, truth.click_height_m, step) for r in truth.robots]
    tracks = _labelled_tracks(run, step)
    by_frame = tracks.partition_by("frame", as_dict=True)

    acc: dict[tuple[int, bool, float], _Acc] = defaultdict(_Acc)
    unjudged: dict[int, int] = defaultdict(int)
    matches: dict[int, list[_Match]] = defaultdict(list)
    cam_xy = cam.center[:2]
    frames = sorted({f for r in robots for f in r.at} | set(tracks["frame"].unique().to_list()))
    for g in frames:
        here = [(r.team, r.at[g]) for r in robots if g in r.at]
        trk = by_frame.get((g,))
        tid = trk["track_id"].to_numpy() if trk is not None else np.zeros(0, np.int64)
        txy = trk.select("x", "y").to_numpy() if trk is not None else np.zeros((0, 2))
        team = trk["team"].to_numpy() if trk is not None else np.zeros(0, np.int64)
        for r in robots:
            if g not in r.at:
                unjudged[r.team] += int(np.sum(team == r.team))
        if not here:
            continue
        pts = np.array([[x, y] for _, (x, y, _) in here], np.float64)
        dist = np.hypot(pts[:, None, 0] - txy[None, :, 0], pts[:, None, 1] - txy[None, :, 1])
        for i, (t, (_, _, est)) in enumerate(here):
            for gate in GATES_M:
                a = acc[(t, est, gate)]
                near = dist[i] <= gate
                a.truth += 1
                a.near += int(near.any())
                if np.any(near & (team == t)):
                    a.hit += 1
                elif not near.any():
                    a.no_track += 1
                elif np.any(near & (team == NO_TEAM)):
                    a.unlabelled += 1
                else:
                    a.other += 1
                for k in np.flatnonzero(team == t):
                    a.labelled += 1
                    if dist[i, k] > gate:
                        on = [here[j][0] for j in range(len(here)) if j != i and dist[j, k] <= gate]
                        a.wrong.append((g, int(tid[k]), float(dist[i, k]), on[0] if on else None))
        if len(tid):
            cost = np.where(dist <= POSITION_GATE_M, dist, 1e6)
            rows, cols = linear_sum_assignment(cost)
            for i, k in zip(rows.tolist(), cols.tolist(), strict=True):
                if dist[i, k] > POSITION_GATE_M:
                    continue
                t, (x, y, est) = here[i]
                ray = np.array([x, y]) - cam_xy
                los = float(np.dot(txy[k] - [x, y], ray / np.linalg.norm(ray)))
                matches[t].append(_Match(g, int(tid[k]), float(dist[i, k]), los, est))

    first: dict[int, dict[int, int]] = defaultdict(dict)
    for t, ms in matches.items():
        for m in ms:
            first[m.track_id].setdefault(t, m.frame)
    target = match.target_team
    scored = [
        RobotBench(
            team=r.team,
            is_target=r.team == target,
            truth_samples=r.samples,
            skipped_samples=r.skipped,
            visible=_state(acc, matches, first, r.team, False, dt, info.fps),
            estimated=_state(acc, matches, first, r.team, True, dt, info.fps),
            labelled_without_truth_s=unjudged[r.team] * dt,
        )
        for r in robots
    ]
    scored.sort(key=lambda r: not r.is_target)
    bench = Bench(
        video_sha256=info.sha256,
        truth_path=str(truth_path),
        truth_source=truth.source,
        click_height_m=truth.click_height_m,
        grid_hz=info.fps / step,
        created_at=datetime.now(UTC),
        targets=_targets(scored, target),
        robots=scored,
    )
    bench.save(run.path / "bench.json")
    report = render_md(bench, run)
    (run.path / "bench.md").write_text(report)
    return report


def _grid_truth(
    team: int, rows: list[TruthSample], cam: Camera, height: float, step: int
) -> _RobotTruth:
    """Cast a robot's truth samples to the field and keep the nearest one per grid frame."""
    main = [s for s in rows if s.view == "main"]
    uv = np.array([[s.u, s.v] for s in main], np.float64).reshape(-1, 2)
    xy = cam.to_field(uv, z=height)
    best: dict[int, tuple[int, tuple[float, float, bool]]] = {}
    ok = 0
    for s, (x, y) in zip(main, xy.tolist(), strict=True):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        ok += 1
        g = round(s.frame / step) * step
        off = abs(s.frame - g)
        if 2 * off <= step and (g not in best or off < best[g][0]):
            best[g] = (off, (x, y, s.state == "estimated"))
    return _RobotTruth(team, {g: v for g, (_, v) in best.items()}, len(rows), len(rows) - ok)


def _labelled_tracks(run: RunDir, step: int) -> pl.DataFrame:
    """Track samples on the grid with the team of the identity segment covering each."""
    df = read_tracks(run.tracks).filter(pl.col("frame") % step == 0)
    df = df.select("track_id", "frame", "x", "y").with_row_index("row")
    teams = np.full(len(df), NO_TEAM, np.int64)
    tid, fr = df["track_id"].to_numpy(), df["frame"].to_numpy()
    for s in Identity.load(run.identity_json).segments:
        if s.team is None:
            continue
        on = (tid == s.track_id) & (fr >= s.frame_start) & (fr <= s.frame_end)
        clash = on & (teams != NO_TEAM) & (teams != s.team)
        teams[on] = s.team
        teams[clash] = -1
    teams[teams == -1] = NO_TEAM
    return df.with_columns(pl.Series("team", teams))


def _state(
    acc: dict[tuple[int, bool, float], _Acc],
    matches: dict[int, list[_Match]],
    first: dict[int, dict[int, int]],
    team: int,
    est: bool,
    dt: float,
    fps: float,
) -> StateScore:
    """One robot's scores over grid frames whose truth is `est`."""
    gates = [_gate(acc[(team, est, gate)], gate, dt, fps) for gate in GATES_M]
    ms = [m for m in matches[team] if m.estimated == est]
    d = np.array([m.dist for m in ms])
    switches = changes = 0
    for prev, m in pairwise(ms):
        if m.track_id == prev.track_id:
            continue
        changes += 1
        others = first[m.track_id]
        switches += any(t != team and f < m.frame for t, f in others.items())
    return StateScore(
        truth_s=acc[(team, est, GATES_M[0])].truth * dt,
        gates=gates,
        position=PositionScore(
            gate_m=POSITION_GATE_M,
            matched_s=len(ms) * dt,
            median_m=float(np.median(d)) if len(d) else None,
            p90_m=float(np.percentile(d, 90)) if len(d) else None,
            mean_los_offset_m=float(np.mean([m.los for m in ms])) if ms else None,
        ),
        identity_switches=switches,
        track_changes=changes,
    )


def _share(n: int, d: int) -> float | None:
    return n / d if d else None


def _gate(a: _Acc, gate: float, dt: float, fps: float) -> GateScore:
    stretches: list[WrongStretch] = []
    by_track: dict[int, list[tuple[int, float, int | None]]] = defaultdict(list)
    for g, tid, dist, on in sorted(a.wrong):
        by_track[tid].append((g, dist, on))
    step = round(dt * fps)
    for tid, evs in by_track.items():
        start = 0
        for i in range(1, len(evs) + 1):
            if i == len(evs) or evs[i][0] - evs[i - 1][0] > step:
                run = evs[start:i]
                ons = [o for _, _, o in run if o is not None]
                stretches.append(
                    WrongStretch(
                        track_id=tid,
                        frame_start=run[0][0],
                        frame_end=run[-1][0],
                        t_start=run[0][0] / fps,
                        t_end=run[-1][0] / fps,
                        seconds=len(run) * dt,
                        median_distance_m=float(np.median([d for _, d, _ in run])),
                        on_team=max(set(ons), key=ons.count) if ons else None,
                    )
                )
                start = i
    stretches.sort(key=lambda s: s.frame_start)
    return GateScore(
        gate_m=gate,
        label_recall=_share(a.hit, a.truth),
        tracked_share=_share(a.near, a.truth),
        no_track_share=_share(a.no_track, a.truth),
        unlabelled_track_share=_share(a.unlabelled, a.truth),
        other_label_share=_share(a.other, a.truth),
        labelled_s=a.labelled * dt,
        wrong_label_s=len(a.wrong) * dt,
        wrong_label_share=_share(len(a.wrong), a.labelled),
        wrong_stretches=stretches,
    )


def _targets(robots: list[RobotBench], team: int) -> Targets:
    r = next((r for r in robots if r.team == team), None)
    g = next((g for g in r.visible.gates if g.gate_m == TARGET_GATE_M), None) if r else None
    med = r.visible.position.median_m if r else None
    return Targets(
        gate_m=TARGET_GATE_M,
        max_wrong_label_share=TARGET_MAX_WRONG_SHARE,
        min_label_recall=TARGET_MIN_RECALL,
        max_median_error_m=TARGET_MAX_MEDIAN_M,
        team=team,
        in_truth=r is not None,
        wrong_label_ok=(
            g.wrong_label_share < TARGET_MAX_WRONG_SHARE
            if g and g.wrong_label_share is not None
            else None
        ),
        recall_ok=g.label_recall >= TARGET_MIN_RECALL if g and g.label_recall is not None else None,
        median_ok=med < TARGET_MAX_MEDIAN_M if med is not None else None,
    )


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def _m(x: float | None, signed: bool = False) -> str:
    return "-" if x is None else (f"{x:+.2f} m" if signed else f"{x:.2f} m")


def _verdict(ok: bool | None) -> str:
    return "no truth" if ok is None else ("met" if ok else "MISSED")


def render_md(b: Bench, run: RunDir) -> str:
    """The short readable report written as bench.md."""
    t = b.targets
    lines = [
        f"# bench: {run.path.name} vs {Path(b.truth_path).name}",
        "",
        f"Truth: {b.truth_source}, {len(b.robots)} robots, followed point "
        f"{b.click_height_m:.3f} m above the carpet. Scored on the {b.grid_hz:g} Hz track grid; "
        "headline numbers use visible truth only.",
        "",
    ]
    if t.in_truth:
        r = b.robots[0]
        g = next(g for g in r.visible.gates if g.gate_m == t.gate_m)
        lines += [
            f"Targets for {t.team} at the {t.gate_m:g} m gate: "
            f"wrong labels {_pct(g.wrong_label_share)} (< {_pct(t.max_wrong_label_share)}: "
            f"{_verdict(t.wrong_label_ok)}), label recall {_pct(g.label_recall)} "
            f"(>= {_pct(t.min_label_recall)}: {_verdict(t.recall_ok)}), median error "
            f"{_m(r.visible.position.median_m)} (< {t.max_median_error_m:g} m: "
            f"{_verdict(t.median_ok)}).",
            "",
        ]
    else:
        lines += [f"The target team {t.team} is not in the truth.", ""]
    for r in b.robots:
        lines += _robot_md(r)
    return "\n".join(lines) + "\n"


def _robot_md(r: RobotBench) -> list[str]:
    v, e = r.visible, r.estimated
    head = [f"## {r.team}{' (target)' if r.is_target else ''}", ""]
    head.append(
        f"{v.truth_s:.1f} s visible truth, {e.truth_s:.1f} s estimated; "
        f"{r.skipped_samples} of {r.truth_samples} samples skipped (not main view or off the "
        f"plane); {r.labelled_without_truth_s:.1f} s labelled {r.team} where it has no truth."
    )
    head += ["", "| visible truth | " + " | ".join(f"{g.gate_m:g} m gate" for g in v.gates) + " |"]
    head.append("|---|" + "---|" * len(v.gates))
    rows: list[tuple[str, list[str]]] = [
        ("label recall", [_pct(g.label_recall) for g in v.gates]),
        (
            "wrong-label share",
            [
                f"{_pct(g.wrong_label_share)} ({g.wrong_label_s:.1f} of {g.labelled_s:.1f} s)"
                for g in v.gates
            ],
        ),
        ("tracked share", [_pct(g.tracked_share) for g in v.gates]),
        ("missed: no track near", [_pct(g.no_track_share) for g in v.gates]),
        ("missed: near track unlabelled", [_pct(g.unlabelled_track_share) for g in v.gates]),
        ("missed: near track labelled other", [_pct(g.other_label_share) for g in v.gates]),
    ]
    head += [f"| {name} | " + " | ".join(cells) + " |" for name, cells in rows]
    p = v.position
    head += [
        "",
        f"Position ({p.matched_s:.1f} s matched within {p.gate_m:g} m): median {_m(p.median_m)}, "
        f"p90 {_m(p.p90_m)}, mean line-of-sight offset {_m(p.mean_los_offset_m, True)} "
        "(+ = track farther from the camera). "
        f"Identity switches {v.identity_switches}, track changes {v.track_changes}.",
    ]
    if e.truth_s:
        ge = e.gates[-1]
        head.append(
            f"Estimated truth at {ge.gate_m:g} m: recall {_pct(ge.label_recall)}, tracked "
            f"{_pct(ge.tracked_share)}, wrong labels {_pct(ge.wrong_label_share)}, median error "
            f"{_m(e.position.median_m)}."
        )
    gate = v.gates[-1]
    if gate.wrong_stretches:
        head += ["", f"Wrong-label stretches at {gate.gate_m:g} m (visible truth):", ""]
        for s in gate.wrong_stretches:
            on = f"on {s.on_team}" if s.on_team else "on no robot in the truth"
            head.append(
                f"- {s.t_start:.1f}-{s.t_end:.1f} s (frames {s.frame_start}-{s.frame_end}), "
                f"track {s.track_id}, {s.median_distance_m:.2f} m from the truth, {on}"
            )
    return [*head, ""]
