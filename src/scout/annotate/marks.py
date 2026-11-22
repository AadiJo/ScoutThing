"""AI annotation: images to read bumper-centre pixels from, check sheets of marks, and the
compile step that turns mark files into a truth file.

    paths = marking_images(run, [frame_of(info, "70")], out, MarkView("main"), around, size)
    paths = check_sheets(run, read_marks(Path("marks_2839.csv"), info), out)
    report = compile_truth(run, [Path("a.csv"), Path("b.csv")], truth_path, tol_px=12)

Everything here reads only the source video, `video.json` and the broadcast panel layout in
`match.json` (rectangles, roster and match times; all read from the overlay). It never reads
tracks, detections, identity or the overlay video, so marks made with it stay independent
of the tracker they are used to test.

A mark file is CSV with the header `team,at,x,y,state,note`. `at` is `f<frame>` or video
seconds; `x`, `y` are full-frame source pixels (empty for `skip`). States:
- `visible` the robot is seen and its identity confirmed;
- `estimated` hidden, but its place is known;
- `unsure` placed, but its identity rests on a guess (left out of truth);
- `skip` an assigned time the annotator could not mark (left out of truth, counted).
Lines starting with `#` are comments.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from scout.games.rebuilt2026 import BUMPER_CENTER_HEIGHT_M
from scout.run import RunDir
from scout.schemas.common import PixelRect
from scout.schemas.match import MatchInfo
from scout.schemas.truth import Truth, TruthRobot, TruthSample, TruthState
from scout.schemas.video import VideoInfo
from scout.video import frame_at, frame_time

Image = NDArray[np.uint8]
MarkState = Literal["visible", "estimated", "unsure", "skip"]
MARK_STATES: dict[str, MarkState] = {s: s for s in ("visible", "estimated", "unsure", "skip")}
MarkView = Literal["main", "corner_left", "corner_right", "scoreboard", "full"]
PROTOCOL = Path(__file__).with_name("protocol.txt")
HEADER = ("team", "at", "x", "y", "state", "note")

FONT = cv2.FONT_HERSHEY_SIMPLEX
MARGIN = 46
"""Pixels of label margin around every gridded tile."""
GRID_STEPS = (5, 10, 20, 25, 50, 100, 200)
GRID_MIN_OUT_PX = 40
"""Grid lines are at least this many output pixels apart."""
GRID_COLOR = (0, 255, 255)
RING_COLOR = (255, 0, 255)


@dataclass(frozen=True)
class Mark:
    """One line of a mark file. `u`, `v` are None only for `skip`."""

    team: int
    frame: int
    u: float | None
    v: float | None
    state: MarkState
    note: str
    where: str
    """`file:line`, for messages."""


def frame_of(info: VideoInfo, token: str) -> int:
    """Frame index named by `f<frame>` or by video seconds (the frame on screen then)."""
    token = token.strip()
    f = token.startswith("f")
    index = int(token[1:]) if f else math.floor(float(token) * info.fps + 1e-6)
    if not 0 <= index < info.frame_count:
        raise ValueError(f"{token!r} is outside the video (0..{info.frame_count - 1} frames)")
    return index


def frames_of(info: VideoInfo, tokens: list[str]) -> list[int]:
    """Frames for CLI tokens: `f<frame>`, seconds, or a range `start:end[:step]` in seconds
    (end included, step 1 s by default)."""
    out: list[int] = []
    for token in tokens:
        parts = token.split(":")
        if len(parts) == 1:
            out.append(frame_of(info, token))
            continue
        start, end = float(parts[0]), float(parts[1])
        step = float(parts[2]) if len(parts) > 2 else 1.0
        n = math.floor((end - start) / step + 1e-6)
        out += [frame_of(info, f"{start + i * step:.6f}") for i in range(n + 1)]
    return out


def view_rect(match: MatchInfo, info: VideoInfo, view: MarkView) -> PixelRect:
    """Source rectangle of a broadcast panel, the scoreboard, or the whole frame."""
    if view == "full":
        return PixelRect(x=0, y=0, w=info.width, h=info.height)
    if view == "scoreboard":
        return match.scoreboard
    rect = match.views.get(view)
    if rect is None:
        raise ValueError(f"this broadcast has no {view} panel")
    return rect


def read_marks(path: Path, info: VideoInfo) -> list[Mark]:
    """Parse a mark file; raises ValueError naming the bad line."""
    marks: list[Mark] = []
    with path.open(newline="") as f:
        rows = [(i, r) for i, r in enumerate(csv.reader(f), 1) if r and not r[0].startswith("#")]
    if not rows or tuple(c.strip() for c in rows[0][1]) != HEADER:
        raise ValueError(f"{path}: first line must be the header {','.join(HEADER)}")
    for i, row in rows[1:]:
        where = f"{path}:{i}"
        cells = [c.strip() for c in row] + [""] * (len(HEADER) - len(row))
        team, at, x, y, state = cells[:5]
        note = ",".join(cells[5:]).strip(", ")
        try:
            st = MARK_STATES.get(state)
            if st is None:
                raise ValueError(f"state must be one of {', '.join(MARK_STATES)}")
            u = float(x) if x else None
            v = float(y) if y else None
            if st != "skip" and (u is None or v is None):
                raise ValueError("x and y are needed unless the state is skip")
            marks.append(Mark(int(team), frame_of(info, at), u, v, st, note, where))
        except ValueError as e:
            raise ValueError(f"{where}: {e}") from e
    return marks


# --- drawing ------------------------------------------------------------------


def _text(img: Image, s: str, org: tuple[int, int], scale: float = 0.5) -> None:
    """White text with a black outline, readable on any background."""
    cv2.putText(img, s, org, FONT, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)


def _centered(cx: float, cy: float, w: int, h: int, bound: PixelRect) -> PixelRect:
    """A w x h rectangle centred on (cx, cy), slid (not shrunk) to stay inside `bound`."""
    w, h = min(w, bound.w), min(h, bound.h)
    x = min(max(round(cx - w / 2), bound.x), bound.x + bound.w - w)
    y = min(max(round(cy - h / 2), bound.y), bound.y + bound.h - h)
    return PixelRect(x=x, y=y, w=w, h=h)


def grid_tile(
    image: Image,
    rect: PixelRect,
    zoom: float,
    grid: int | None = None,
    rings: list[tuple[float, float]] | None = None,
) -> Image:
    """`rect` of a full source frame scaled by `zoom`, with grid lines every `grid` source
    pixels (auto when None) labelled with their source coordinate on all four margins, and a
    hollow ring with outside ticks at each source point in `rings`."""
    crop = image[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w]
    w, h = round(rect.w * zoom), round(rect.h * zoom)
    big = cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC if zoom > 1 else cv2.INTER_AREA)
    g = grid or next((s for s in GRID_STEPS if s * zoom >= GRID_MIN_OUT_PX), GRID_STEPS[-1])
    out = np.asarray(
        cv2.copyMakeBorder(
            big, MARGIN, MARGIN, MARGIN, MARGIN, cv2.BORDER_CONSTANT, value=(40,) * 3
        ),
        np.uint8,
    )
    lines = big.copy()
    xs = range(math.ceil(rect.x / g) * g, rect.x + rect.w, g)
    ys = range(math.ceil(rect.y / g) * g, rect.y + rect.h, g)
    for gx in xs:
        px = round((gx - rect.x) * zoom)
        cv2.line(lines, (px, 0), (px, h - 1), GRID_COLOR, 1)
    for gy in ys:
        py = round((gy - rect.y) * zoom)
        cv2.line(lines, (0, py), (w - 1, py), GRID_COLOR, 1)
    out[MARGIN : MARGIN + h, MARGIN : MARGIN + w] = cv2.addWeighted(lines, 0.45, big, 0.55, 0)
    for gx in xs:
        px = MARGIN + round((gx - rect.x) * zoom)
        tw = cv2.getTextSize(str(gx), FONT, 0.5, 1)[0][0]
        _text(out, str(gx), (px - tw // 2, MARGIN - 12))
        _text(out, str(gx), (px - tw // 2, MARGIN + h + 28))
        cv2.line(out, (px, MARGIN - 8), (px, MARGIN - 1), GRID_COLOR, 1)
        cv2.line(out, (px, MARGIN + h), (px, MARGIN + h + 7), GRID_COLOR, 1)
    for gy in ys:
        py = MARGIN + round((gy - rect.y) * zoom)
        _text(out, str(gy), (2, py + 5), 0.42)
        _text(out, str(gy), (MARGIN + w + 3, py + 5), 0.42)
    for u, v in rings or []:
        cx = MARGIN + round((u - rect.x) * zoom)
        cy = MARGIN + round((v - rect.y) * zoom)
        r = max(7, round(2.5 * zoom))
        cv2.circle(out, (cx, cy), r, RING_COLOR, 2, cv2.LINE_AA)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p0 = (cx + dx * (r + 3), cy + dy * (r + 3))
            p1 = (cx + dx * (r + 12), cy + dy * (r + 12))
            cv2.line(out, p0, p1, RING_COLOR, 2, cv2.LINE_AA)
    return out


def _caption(tile: Image, text: str, height: int = 30) -> Image:
    bar = np.full((height, tile.shape[1], 3), 20, np.uint8)
    _text(bar, text, (6, height - 9), 0.6)
    return np.vstack([bar, tile])


def _stack(tiles: list[Image], cols: int) -> Image:
    """Tiles in rows of `cols`, padded to a common cell size."""
    ch = max(t.shape[0] for t in tiles)
    cw = max(t.shape[1] for t in tiles)
    cells = [
        np.asarray(
            cv2.copyMakeBorder(
                t, 0, ch - t.shape[0] + 6, 0, cw - t.shape[1] + 6, cv2.BORDER_CONSTANT
            ),
            np.uint8,
        )
        for t in tiles
    ]
    cells += [np.zeros_like(cells[0])] * (-len(cells) % cols)
    return np.vstack([np.hstack(cells[i : i + cols]) for i in range(0, len(cells), cols)])


# --- the three jobs ---------------------------------------------------------------

MAX_OUT_W = 1400
"""Widest a gridded tile is drawn when `zoom` is automatic."""


def marking_images(
    run: RunDir,
    frames: list[int],
    out: Path,
    view: MarkView = "main",
    around: tuple[float, float] | None = None,
    size: tuple[int, int] | None = None,
    zoom: float | None = None,
    per_image: int = 1,
    context: bool = False,
) -> list[Path]:
    """Write gridded images of `frames` to read source pixels from; returns their paths.

    The crop is `size` (w, h) source pixels centred on `around` (default: the whole `view`),
    scaled by `zoom` (default: as large as fits MAX_OUT_W, at most 6x). `context` puts the
    whole view beside each crop, with the crop outlined, to find the robot again.
    `per_image` moments are stacked into one image."""
    info = VideoInfo.load(run.video_json)
    bound = view_rect(MatchInfo.load(run.match_json), info, view)
    if size is None:
        rect = bound
    else:
        cx, cy = around or (bound.x + bound.w / 2, bound.y + bound.h / 2)
        rect = _centered(cx, cy, size[0], size[1], bound)
    z = zoom or min(6.0, MAX_OUT_W / rect.w)
    out.mkdir(parents=True, exist_ok=True)
    tiles: list[Image] = []
    for index in frames:
        image = frame_at(run.video, info, frame_time(info, index) + 1e-4).image
        tile = grid_tile(image, rect, z)
        if context and rect != bound:
            wide = image.copy()
            cv2.rectangle(wide, (rect.x, rect.y), (rect.x + rect.w, rect.y + rect.h), RING_COLOR, 3)
            ctx = grid_tile(wide, bound, (tile.shape[1] - 2 * MARGIN) / bound.w, grid=100)
            tile = np.vstack([tile, ctx])
        tiles.append(_caption(tile, f"t={frame_time(info, index):.3f} s   frame f{index}   {view}"))
    paths: list[Path] = []
    for i in range(0, len(tiles), per_image):
        group = frames[i : i + per_image]
        path = out / f"mark_{view}_f{group[0]}{f'-f{group[-1]}' if len(group) > 1 else ''}.png"
        cv2.imwrite(str(path), _stack(tiles[i : i + per_image], 1))
        paths.append(path)
    return paths


CHECK_HALF = 60
"""Half the source side of a check cell's zoomed crop."""
CHECK_ZOOM = 3.0
CHECK_COLS = 3
CHECK_PER_SHEET = 6


def check_sheets(run: RunDir, marks: list[Mark], out: Path) -> list[Path]:
    """Contact sheets with each placed mark drawn as a ring on its own frame: a gridded
    zoom (to judge the exact point) above a wider crop (to judge which robot). `skip`
    marks have no point and are not drawn."""
    info = VideoInfo.load(run.video_json)
    main = view_rect(MatchInfo.load(run.match_json), info, "main")
    out.mkdir(parents=True, exist_ok=True)
    cells: list[Image] = []
    names: list[str] = []
    side = 2 * CHECK_HALF
    for m in sorted(marks, key=lambda m: (m.frame, m.team)):
        if m.u is None or m.v is None:
            continue
        image = frame_at(run.video, info, frame_time(info, m.frame) + 1e-4).image
        zoom = grid_tile(
            image, _centered(m.u, m.v, side, side, main), CHECK_ZOOM, None, [(m.u, m.v)]
        )
        wide = grid_tile(
            image, _centered(m.u, m.v, 4 * side, 2 * side, main), 1.0, 50, [(m.u, m.v)]
        )
        cell = _stack([zoom, wide], 1)
        head = f"{m.team} t={frame_time(info, m.frame):.2f} f{m.frame} {m.state}"
        cell = _caption(cell, f"({m.u:.0f},{m.v:.0f}) {m.note}"[:52], 26)
        cells.append(_caption(cell, head, 30))
        names.append(f"f{m.frame}")
    paths: list[Path] = []
    for i in range(0, len(cells), CHECK_PER_SHEET):
        path = out / f"check_{names[i]}-{names[min(i + CHECK_PER_SHEET, len(names)) - 1]}.png"
        cv2.imwrite(str(path), _stack(cells[i : i + CHECK_PER_SHEET], CHECK_COLS))
        paths.append(path)
    return paths


def compile_truth(run: RunDir, files: list[Path], truth_path: Path, tol_px: float) -> str:
    """Merge mark files into `truth_path` (`source: "ai"`) and return a report.

    Where two files mark the same team on the same frame, marks within `tol_px` pixels and
    of the same state are averaged; any other pair is a disagreement: it is reported and the
    frame is left out of the truth. `unsure` and `skip` marks never enter the truth."""
    info = VideoInfo.load(run.video_json)
    match = MatchInfo.load(run.match_json)
    main = view_rect(match, info, "main")
    roster = {s.team for s in match.roster}
    by: dict[tuple[int, int], list[Mark]] = defaultdict(list)
    counts: dict[str, dict[MarkState, int]] = {}
    for path in files:
        marks = read_marks(path, info)
        c: dict[MarkState, int] = defaultdict(int)
        for m in marks:
            if m.team not in roster:
                raise ValueError(f"{m.where}: team {m.team} is not in the roster {sorted(roster)}")
            if (
                m.u is not None
                and m.v is not None
                and not (main.x <= m.u < main.x + main.w and main.y <= m.v < main.y + main.h)
            ):
                raise ValueError(f"{m.where}: ({m.u}, {m.v}) is outside the main view")
            c[m.state] += 1
            by[(m.team, m.frame)].append(m)
        counts[str(path)] = c
    lines = [
        f"{p}: " + ", ".join(f"{n} {s}" for s, n in sorted(c.items())) for p, c in counts.items()
    ]
    samples: dict[int, list[TruthSample]] = defaultdict(list)
    pairs: list[float] = []
    disagree: list[str] = []
    for (team, frame), ms in sorted(by.items()):
        if len({m.where.rsplit(":", 1)[0] for m in ms}) < len(ms):
            raise ValueError(
                f"team {team} f{frame} is marked twice in one file: {[m.where for m in ms]}"
            )
        pts = [(m.u, m.v) for m in ms if m.u is not None and m.v is not None]
        d = max((math.dist(a, b) for a in pts for b in pts), default=0.0)
        if len(ms) > 1:
            if len(pts) == len(ms):
                pairs.append(d)
            why = "states differ" if len({m.state for m in ms}) > 1 else f"{d:.0f} px apart"
            if why == "states differ" or d > tol_px:
                said = ", ".join(f"{m.where} {m.state} ({m.u}, {m.v})" for m in ms)
                disagree.append(
                    f"team {team} t={frame_time(info, frame):.2f} f{frame}: {why}; {said}"
                )
                continue
        state: TruthState
        if ms[0].state == "visible":
            state = "visible"
        elif ms[0].state == "estimated":
            state = "estimated"
        else:
            continue
        samples[team].append(
            TruthSample(
                frame=frame,
                t=frame_time(info, frame),
                view="main",
                u=round(sum(u for u, _ in pts) / len(pts), 1),
                v=round(sum(v for _, v in pts) / len(pts), 1),
                state=state,
            )
        )
    order = [s.team for s in match.roster]
    truth = Truth(
        video_sha256=info.sha256,
        video_name=Path(info.source_path).name,
        source="ai",
        click_height_m=BUMPER_CENTER_HEIGHT_M,
        notes=(
            "AI annotators marked bumper-ring centres about once a second with `scout "
            "mark-frames` and `scout mark-check`, following annotate/protocol.txt; compiled "
            f"from {len(files)} mark file(s) by `scout mark-compile` (two marks on one frame "
            f"within {tol_px:g} px averaged, others left out)."
        ),
        robots=[TruthRobot(team=t, samples=samples[t]) for t in sorted(samples, key=order.index)],
    )
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    truth.save(truth_path)
    lines.append(
        f"wrote {truth_path}: "
        + ", ".join(f"{r.team} {len(r.samples)} samples" for r in truth.robots)
    )
    if pairs:
        med = float(np.median(pairs))
        lines.append(
            f"{len(pairs)} frames marked by two files: median {med:.1f} px apart, "
            f"max {max(pairs):.1f} px"
        )
    lines += [f"{len(disagree)} disagreements (left out of the truth):"] + [
        f"  {d}" for d in disagree
    ]
    return "\n".join(lines)
