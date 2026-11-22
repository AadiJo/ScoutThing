"""match: reads video, video.json, settings.json.

Writes match.json: overlay layout and panel rectangles, the six-team roster, target team and
alliance, auto/teleop times in video seconds, and match phases with active hubs. Also
writes `match_check.jpg`, a contact sheet for a human to check the result.

Knows two FIRST 2026 overlay layouts (`scout.games.first_overlay_2026`). A few frames
spread over the video pick the layout whose clock box shows a readable, ticking clock.
Then one decode of that layout's scoreboard band at the full frame rate does all the
reading:

1. The clock box is OCRed only when its pixels change, so each read lands on the first
   frame of a new clock value: a tick moment, exact to one frame.
2. Video time against elapsed period time is a local line through the nearest
   LOCAL_TICKS ticks, separately in auto and teleop, after dropping misreads one at a
   time. On some official broadcasts the clock runs 1 to 2% slow against video time and
   the rate wanders within a period (2026hop_qm45 teleop: 1.01 to 1.04 per 10 s), so one
   line per period puts shift boundaries up to 0.2 s off; the local line stays within
   the ticks' own frame jitter. Auto start and teleop start are the local lines at
   elapsed 0; the plain line per period and its residuals go into `clock_fits`.
3. The six team numbers are read every 10 clock seconds and voted per slot.
4. The fuel counters are read at the end of the auto-teleop pause, and the active-hub
   arrows are sampled on every frame. The arrows name the auto winner (its hub is off in
   shifts 1 and 3); the counters must not contradict them. They also switch on the clock
   tick at each shift boundary, which checks the timeline.
"""

from __future__ import annotations

import queue
import re
import threading
from collections import Counter, deque
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray
from rapidocr_onnxruntime import RapidOCR  # pyright: ignore[reportMissingTypeStubs]

from scout.games import first_overlay_2026 as ov
from scout.games.first_overlay_2026 import Layout
from scout.games.rebuilt2026 import AUTO_S, PHASES, TELEOP_S, active_hubs, other
from scout.run import NeedsInput, RunDir, Stage
from scout.schemas.common import ALLIANCES, VIEW_NAMES, Alliance, PixelRect
from scout.schemas.match import ClockFit, MatchInfo, PhaseSpan, TeamSlot
from scout.schemas.video import VideoInfo
from scout.video import THREADS, Frame, frame_at, frames

Image = NDArray[np.uint8]
Period = Literal["auto", "teleop"]
ScanState = Literal["pre", "auto", "pause", "teleop", "done"]

PAUSE_CLOCK = int(TELEOP_S)  # the clock holds 2:20 through the auto-teleop pause
CHANGE_LEVEL = 4.0  # mean absolute grey difference that means a new clock face
MIN_CONFIDENCE = 0.8
GIVE_UP_FRAMES = 10  # OCR attempts on a changed clock face before treating it as no clock
TEAM_READ_EVERY = 10  # read the team numbers when the clock shows a multiple of this
OUTLIER_S = 0.15  # tick residual from the local line that marks a misread (jitter is 0.07)
# Ticks per local line: 8 clock seconds either side. Over the six dev videos this kept the
# CapCity starts within 0.02 s of a whole-period line and cut 2026hop_qm45's shift
# boundary error against the hub arrows from 0.2 s to 0.08 s; 5 or fewer ticks follows
# frame jitter, 41 or more misses the rate drift.
LOCAL_TICKS = 17
MIN_TICKS: dict[Period, int] = {"auto": 8, "teleop": 30}
# Mean clock rate a real broadcast can have, in video seconds per clock second. Official
# broadcasts measure up to 1.022; anything far outside is not a game clock.
RATE_RANGE = (0.95, 1.05)
# Layout choice: OCR every layout's clock box on this many frames spread over the middle
# of the video (the match fills most of a clip). The layout needs this many "m:ss" reads,
# with at least this many distinct values, to count as a ticking clock.
LAYOUT_SAMPLES = 6
LAYOUT_SPAN = (0.2, 0.7)
LAYOUT_MIN_READS = 3
# Fuel is read this many frames before teleop's first tick. Hubs keep counting auto fuel
# through the pause, so only the last frames hold the final auto count.
FUEL_BACK_FRAMES = (1, 2, 3)
ROSTER_KEY = "match.roster"


class _Ocr:
    """Recognition-only RapidOCR for boxes that hold one line of text."""

    def __init__(self) -> None:
        self._ocr = RapidOCR(intra_op_num_threads=THREADS, inter_op_num_threads=1)

    def read(self, crops: Sequence[Image]) -> list[tuple[str, float]]:
        """(text, confidence) for each crop, in order."""
        found, _ = self._ocr.text_rec(list(crops))
        return [(str(text), float(score)) for text, score in found]


@dataclass
class _Scan:
    """Everything the scoreboard pass reads."""

    ticks: dict[Period, list[tuple[float, float]]] = field(
        default_factory=lambda: {"auto": [], "teleop": []}
    )
    """(video t, elapsed period seconds) of each clock tick."""
    clock_reads: int = 0
    t: list[float] = field(default_factory=list[float])
    lit: dict[Alliance, list[bool]] = field(default_factory=lambda: {"red": [], "blue": []})
    team_texts: list[list[str]] = field(default_factory=lambda: [[] for _ in range(6)])
    fuel_texts: dict[Alliance, list[str]] = field(default_factory=lambda: {"red": [], "blue": []})


class _UnreadableError(Exception):
    """The overlay could not be read where the layout puts it."""


def _cut(image: Image, rect: PixelRect, origin: PixelRect) -> Image:
    """The part of `image` (a crop at `origin`) under full-frame `rect`."""
    x, y = rect.x - origin.x, rect.y - origin.y
    return image[y : y + rect.h, x : x + rect.w]


PREFETCH_FRAMES = 240  # frames decoded ahead of the reader: 90 MB official, 200 MB CapCity


def _prefetched(source: Iterator[Frame]) -> Iterator[Frame]:
    """`source` read on a thread, so decoding runs while the caller OCRs. Without it the
    pipe from ffmpeg holds less than one frame and the two take turns."""
    buffer: queue.Queue[Frame | BaseException | None] = queue.Queue(PREFETCH_FRAMES)
    stop = threading.Event()

    def pump() -> None:
        try:
            for f in source:
                if stop.is_set():
                    break
                buffer.put(f)
            buffer.put(None)
        except BaseException as e:  # handed to the reader to re-raise
            buffer.put(e)

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    try:
        while (item := buffer.get()) is not None:
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        while thread.is_alive():  # unblock a pump waiting on a full buffer
            try:
                buffer.get_nowait()
            except queue.Empty:
                thread.join(0.05)


def _around(rects: Sequence[PixelRect]) -> PixelRect:
    """The smallest rectangle holding all of `rects`."""
    x0, y0 = min(r.x for r in rects), min(r.y for r in rects)
    x1, y1 = max(r.x + r.w for r in rects), max(r.y + r.h for r in rects)
    return PixelRect(x=x0, y=y0, w=x1 - x0, h=y1 - y0)


def _mean_bgr(image: Image) -> tuple[float, float, float]:
    """Mean colour of an arrow box, from every other pixel (the arrows are flat colour)."""
    b, g, r = image[::2, ::2].reshape(-1, 3).mean(axis=0).tolist()
    return (b, g, r)


def _parse_clock(text: str) -> int | None:
    """Clock text "m:ss" -> seconds shown, or None."""
    m = re.fullmatch(r"(\d):([0-5]\d)", text.replace(" ", ""))
    return int(m[1]) * 60 + int(m[2]) if m else None


def _parse_int(text: str) -> int | None:
    """Leading integer of an OCR string ("61/120" -> 61), reading O as 0."""
    m = re.match(r"\d+", text.strip().replace("O", "0").replace("o", "0"))
    return int(m[0]) if m else None


def _find_layout(video: Path, info: VideoInfo, ocr: _Ocr) -> tuple[Layout | None, str]:
    """The layout whose clock box reads as a ticking clock on sampled frames, with a note
    of the reads per layout. None when no layout qualifies."""
    w, h = info.width, info.height
    duration = info.frame_count / info.fps
    values: dict[ov.LayoutName, list[int]] = {lay.name: [] for lay in ov.LAYOUTS}
    for t in np.linspace(*LAYOUT_SPAN, LAYOUT_SAMPLES) * duration:
        image = frame_at(video, info, float(t)).image
        full = PixelRect(x=0, y=0, w=w, h=h)
        crops = [_cut(image, ov.scaled(lay.clock, w, h), full) for lay in ov.LAYOUTS]
        for lay, (text, conf) in zip(ov.LAYOUTS, ocr.read(crops), strict=True):
            value = _parse_clock(text) if conf >= MIN_CONFIDENCE else None
            if value is not None:
                values[lay.name].append(value)
    note = " ".join(f"{k}:{len(v)} reads/{len(set(v))} values" for k, v in values.items())
    ok = [
        lay
        for lay in ov.LAYOUTS
        if len(values[lay.name]) >= LAYOUT_MIN_READS
        and len(set(values[lay.name])) >= LAYOUT_MIN_READS
    ]
    return max(ok, key=lambda lay: len(values[lay.name]), default=None), note


def _scan(video: Path, info: VideoInfo, ocr: _Ocr, lay: Layout) -> _Scan:
    """Decode the scoreboard at full rate once and read it (steps 1, 3 and 4). Only the
    rectangle around the read boxes is decoded: the pipe from ffmpeg, not OCR, bounds the
    stage's time."""
    w, h = info.width, info.height
    clock = ov.scaled(lay.clock, w, h)
    active = {a: ov.scaled(lay.active[a], w, h) for a in ALLIANCES}
    fuel = {a: ov.scaled(lay.fuel[a], w, h) for a in ALLIANCES}
    slots = [ov.scaled(b, w, h) for a in ALLIANCES for b in lay.teams[a]]
    band = _around([clock, *active.values(), *fuel.values(), *slots])

    scan = _Scan()
    state: ScanState = "pre"
    shown: int | None = None  # clock value on screen, None when unknown
    ref: NDArray[np.int16] | None = None  # grey clock face of `shown`
    pending_t: float | None = None  # first frame of a changed face not yet read
    attempts = 0
    fuel_history: deque[tuple[Image, Image]] = deque(maxlen=max(FUEL_BACK_FRAMES))

    def tick(period: Period, t: float, elapsed: float, is_tick: bool) -> None:
        if is_tick:
            scan.ticks[period].append((t, elapsed))

    for f in _prefetched(frames(video, info, crop=band)):
        img = f.image
        scan.t.append(f.t)
        for a in ALLIANCES:
            scan.lit[a].append(ov.is_lit(_mean_bgr(_cut(img, active[a], band))))
        if state == "pause":
            fuel_history.append((_cut(img, fuel["red"], band), _cut(img, fuel["blue"], band)))

        face = _cut(img, clock, band)
        grey = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)[::2, ::2].astype(np.int16)
        if pending_t is None:
            if ref is not None and float(np.abs(grey - ref).mean()) <= CHANGE_LEVEL:
                continue
            pending_t, attempts = f.t, 0
        text, conf = ocr.read([face])[0]
        value = _parse_clock(text) if conf >= MIN_CONFIDENCE else None
        attempts += 1
        if value is None:
            if attempts >= GIVE_UP_FRAMES:
                ref, pending_t, shown = grey, None, None
            continue
        scan.clock_reads += 1
        t, ref, pending_t = pending_t, grey, None
        if value == shown:
            continue
        is_tick = shown is not None
        shown = value

        if state == "pre" and value < AUTO_S:
            state = "auto"
        if state == "auto":
            if value < AUTO_S:
                tick("auto", t, AUTO_S - value, is_tick)
            elif value == PAUSE_CLOCK:
                state = "pause"
            elif value < PAUSE_CLOCK:
                state = "teleop"
        elif state == "pause" and value < PAUSE_CLOCK:
            state = "teleop"
            for back in FUEL_BACK_FRAMES:
                if back <= len(fuel_history):
                    red, blue = fuel_history[-back]
                    (rt, _), (bt, _) = ocr.read([red, blue])
                    scan.fuel_texts["red"].append(rt)
                    scan.fuel_texts["blue"].append(bt)
        if state == "teleop" and value < PAUSE_CLOCK:
            tick("teleop", t, TELEOP_S - value, is_tick)
            if value == 0:
                state = "done"
        if state in ("auto", "teleop") and value > 0 and value % TEAM_READ_EVERY == 0:
            for i, (txt, _) in enumerate(ocr.read([_cut(img, r, band) for r in slots])):
                scan.team_texts[i].append(txt)
    return scan


@dataclass(frozen=True)
class _Timeline:
    """Video time of elapsed period time in one period, from that period's clock ticks."""

    t: NDArray[np.float64]
    """Video time of each kept tick."""
    e: NDArray[np.float64]
    """Elapsed period seconds of each kept tick."""
    fit: ClockFit

    def video_t(self, elapsed: float) -> float:
        """Local line through the LOCAL_TICKS ticks nearest `elapsed`, evaluated there."""
        return _local(self.t, self.e, elapsed)


def _local(t: NDArray[np.float64], e: NDArray[np.float64], elapsed: float) -> float:
    near = np.argsort(np.abs(e - elapsed), kind="stable")[:LOCAL_TICKS]
    slope, start = np.polyfit(e[near], t[near], 1)
    return float(start + slope * elapsed)


def _timeline(points: list[tuple[float, float]], period: Period) -> _Timeline:
    """Drop the tick furthest from its local line while any is over OUTLIER_S (step 2),
    then summarise the kept ticks with one line. Raises _UnreadableError when too few
    ticks remain or the clock does not run near video speed."""
    t = np.array([p[0] for p in points], dtype=np.float64)
    e = np.array([p[1] for p in points], dtype=np.float64)
    keep = np.ones(len(points), dtype=bool)
    while True:
        if int(keep.sum()) < MIN_TICKS[period]:
            raise _UnreadableError(
                f"{period}: only {int(keep.sum())} consistent clock ticks "
                f"(of {len(points)} read), need {MIN_TICKS[period]}"
            )
        model = np.array([_local(t[keep], e[keep], x) for x in e])
        resid = np.abs(t - model)
        worst = int(np.argmax(np.where(keep, resid, -1.0)))
        if resid[worst] <= OUTLIER_S:
            break
        keep[worst] = False
    slope, start = (float(v) for v in np.polyfit(e[keep], t[keep], 1))
    if not RATE_RANGE[0] < slope < RATE_RANGE[1]:
        raise _UnreadableError(f"{period}: clock runs at {slope:.3f} s per second, expected 1")
    line = (t - (start + slope * e))[keep]
    fit = ClockFit(
        start_t=round(start, 4),
        slope=round(slope, 5),
        ticks=int(keep.sum()),
        dropped=int((~keep).sum()),
        rms_s=round(float(np.sqrt(np.mean(line**2))), 4),
        max_abs_s=round(float(np.abs(line).max()), 4),
        timeline_max_abs_s=round(float(resid[keep].max()), 4),
    )
    return _Timeline(t=t[keep], e=e[keep], fit=fit)


def _clock_rate(auto: _Timeline, teleop: _Timeline) -> float:
    """Video seconds per clock second: one slope over both periods, one offset each."""
    e = np.concatenate([auto.e, teleop.e])
    is_auto = np.concatenate([np.ones_like(auto.e), np.zeros_like(teleop.e)])
    a = np.column_stack([e, is_auto, 1.0 - is_auto])
    solution, *_ = np.linalg.lstsq(a, np.concatenate([auto.t, teleop.t]), rcond=None)
    return round(float(solution[0]), 5)


def _vote(texts: list[str], min_votes: int) -> int | None:
    """The integer most of `texts` agree on, or None without a clear majority."""
    values = [v for v in map(_parse_int, texts) if v is not None]
    if not values:
        return None
    best, count = Counter(values).most_common(1)[0]
    return best if count >= min_votes and count * 2 > len(values) else None


def _roster(run: RunDir, scan: _Scan, info: VideoInfo, lay: Layout) -> list[TeamSlot]:
    """Six slots from the votes, or from the `match.roster` answer when one is saved."""
    answer = run.answer(ROSTER_KEY)
    if answer is not None:
        teams = [int(x) for x in answer.split(",")]
        if len(teams) != 6:
            raise ValueError(f"{ROSTER_KEY} must list six teams, got {answer!r}")
    else:
        voted = [_vote(texts, 3) for texts in scan.team_texts]
        teams = [t for t in voted if t is not None]
        if len(teams) != 6 or len(set(teams)) != 6:
            image = _layout_image(run, info, "match_roster.jpg", [lay])
            raise NeedsInput(
                f"could not agree on all six team numbers (read {voted}, from "
                f"{[len(x) for x in scan.team_texts]} reads per slot)",
                image,
                f"scout answer {run.path} {ROSTER_KEY}=R1,R2,R3,B1,B2,B3 "
                "(red then blue, left to right as on the scoreboard)",
            )
    return [
        TeamSlot(team=team, alliance=ALLIANCES[i // 3], station=(1, 2, 3)[i % 3])
        for i, team in enumerate(teams)
    ]


def _phases(auto: _Timeline, teleop: _Timeline, winner: Alliance | None) -> list[PhaseSpan]:
    """Game phases in video seconds: auto on the auto timeline, every other phase on
    teleop's. The auto-teleop pause (robots disabled) lies between auto's end and
    transition's start and belongs to no phase."""
    spans: list[PhaseSpan] = []
    for p in PHASES:
        hubs = active_hubs(p.name, winner)
        line, offset = (auto, 0.0) if p.name == "auto" else (teleop, AUTO_S)
        spans.append(
            PhaseSpan(
                name=p.name,
                t_start=round(line.video_t(p.start - offset), 4),
                t_end=round(line.video_t(p.end - offset), 4),
                active_hubs=None if hubs is None else sorted(hubs),
            )
        )
    return spans


def _arrow_winner(scan: _Scan, phases: list[PhaseSpan]) -> tuple[Alliance | None, str]:
    """Auto winner from the arrows in each shift on screen, with a note on what was seen.
    None unless every shift shows exactly one lit arrow and they all agree."""
    t = np.array(scan.t)
    lit = {a: np.array(scan.lit[a]) for a in ALLIANCES}
    votes: list[Alliance | None] = []
    seen: list[str] = []
    for p in phases:
        if not p.name.startswith("shift"):
            continue
        inside = (t > p.t_start + 0.5) & (t < p.t_end - 0.5)
        if not inside.any():
            continue
        frac = {a: float(lit[a][inside].mean()) for a in ALLIANCES}
        on: list[Alliance] = [a for a in ALLIANCES if frac[a] > 0.8]
        off = [a for a in ALLIANCES if frac[a] < 0.2]
        seen.append(f"{p.name}:{'+'.join(on) or '-'}")
        if len(on) != 1 or len(off) != 1:
            votes.append(None)
            continue
        winner_inactive = p.name in ("shift1", "shift3")
        votes.append(other(on[0]) if winner_inactive else on[0])
    note = " ".join(seen) or "no shift on screen"
    if not votes or None in votes or len(set(votes)) != 1:
        return None, note
    return votes[0], note


def _arrow_timing(scan: _Scan, phases: list[PhaseSpan]) -> list[str]:
    """Independent check of the clock fits: when the arrows first light (FMS enables the
    hubs at auto start) and when they switch at each shift and endgame boundary, relative
    to the fitted times. Returns printable notes."""
    t = np.array(scan.t)
    red, blue = np.array(scan.lit["red"]), np.array(scan.lit["blue"])
    notes: list[str] = []
    auto_start = phases[0].t_start
    lit_on = t[(t > auto_start - 3) & red & blue]
    if lit_on.size:
        notes.append(f"arrows light at {lit_on[0]:.3f} ({lit_on[0] - auto_start:+.3f} vs clock)")
    switched = np.flatnonzero((red[1:] != red[:-1]) | (blue[1:] != blue[:-1])) + 1
    offsets: list[float] = []
    for p in phases:
        if p.name.startswith("shift") or p.name == "endgame":
            near = [t[i] - p.t_start for i in switched if abs(t[i] - p.t_start) < 1.0]
            if near:
                offsets.append(float(near[0]))
    if offsets:
        notes.append(
            f"arrows switch {min(offsets):+.3f}..{max(offsets):+.3f} s from the "
            f"{len(offsets)} timeline shift boundaries"
        )
    return notes


def _fuel_winner(scan: _Scan) -> tuple[Alliance | None, dict[Alliance, int | None]]:
    fuel: dict[Alliance, int | None] = {a: _vote(scan.fuel_texts[a], 2) for a in ALLIANCES}
    red, blue = fuel["red"], fuel["blue"]
    if red is None or blue is None or red == blue:
        return None, fuel
    return ("red" if red > blue else "blue"), fuel


# Box colours (BGR) per layout, so a sheet with every layout drawn tells them apart.
BOX_COLORS: dict[ov.LayoutName, tuple[int, int, int]] = {
    "capcity_2026": (0, 255, 255),
    "official_2026": (255, 0, 255),
}


def _draw_boxes(image: Image, info: VideoInfo, layouts: Sequence[Layout]) -> Image:
    """A copy of a full frame with every overlay box and camera panel of `layouts` on it."""
    out = image.copy()
    w, h = info.width, info.height
    for lay in layouts:
        color = BOX_COLORS[lay.name]
        boxes = [lay.clock, *lay.fuel.values(), *lay.active.values()]
        boxes += [b for a in ALLIANCES for b in lay.teams[a]]
        for b in boxes:
            r = ov.scaled(b, w, h)
            cv2.rectangle(out, (r.x, r.y), (r.x + r.w, r.y + r.h), color, 2)
        c = ov.scaled(lay.clock, w, h)
        label_pos = (c.x + c.w + 6, c.y + c.h + 26)
        cv2.putText(out, lay.name, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        for name in VIEW_NAMES:
            r = ov.scaled(lay.views[name], w, h)
            p0, p1 = (r.x + 2, r.y + 2), (r.x + r.w - 3, r.y + r.h - 3)
            cv2.rectangle(out, p0, p1, (255, 160, 0), 3)
            cv2.putText(
                out,
                name,
                (r.x + 10, r.y + r.h - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 160, 0),
                2,
                cv2.LINE_AA,
            )
    return out


def _layout_image(run: RunDir, info: VideoInfo, name: str, layouts: Sequence[Layout]) -> Path:
    """Save a mid-video frame with the boxes of `layouts` drawn; returns its path."""
    frame = frame_at(run.video, info, info.frame_count / info.fps / 2)
    path = run.path / name
    image = _draw_boxes(frame.image, info, layouts)
    cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path


def _put_lines(image: Image, lines: Sequence[str], x: int, y: int, scale: float) -> None:
    step = round(34 * scale / 0.9)
    for i, line in enumerate(lines):
        pos = (x, y + i * step)
        cv2.putText(image, line, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(
            image, line, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA
        )


def _check_sheet(
    run: RunDir, info: VideoInfo, ocr: _Ocr, lay: Layout, m: MatchInfo, notes: list[str]
) -> None:
    """Write match_check.jpg: frames at auto start, teleop start and teleop end with the boxes
    drawn and what OCR reads there, plus a panel of the results."""
    w, h = info.width, info.height
    clock = ov.scaled(lay.clock, w, h)
    full = PixelRect(x=0, y=0, w=w, h=h)
    cells: list[Image] = []
    moments = [
        ("auto start", m.auto_start_t),
        ("teleop start", m.teleop_start_t),
        ("teleop end", m.teleop_end_t),
    ]
    for label, t in moments:
        frame = frame_at(run.video, info, t)
        crops = [_cut(frame.image, clock, full)]
        crops += [_cut(frame.image, ov.scaled(lay.fuel[a], w, h), full) for a in ALLIANCES]
        (ct, _), (rt, _), (bt, _) = ocr.read(crops)
        lit = [
            a
            for a in ALLIANCES
            if ov.is_lit(_mean_bgr(_cut(frame.image, ov.scaled(lay.active[a], w, h), full)))
        ]
        cell = np.asarray(
            cv2.resize(
                _draw_boxes(frame.image, info, [lay]), (960, 540), interpolation=cv2.INTER_AREA
            ),
            dtype=np.uint8,
        )
        _put_lines(
            cell,
            [
                f"{label}  t={frame.t:.3f} (frame {frame.frame})",
                f"clock '{ct}'  fuel red '{rt}' blue '{bt}'",
                f"lit arrows: {', '.join(lit) or 'none'}",
            ],
            12,
            300,
            0.8,
        )
        cells.append(cell)
    panel = np.zeros((540, 960, 3), np.uint8)
    red = " ".join(str(s.team) for s in m.roster if s.alliance == "red")
    blue = " ".join(str(s.team) for s in m.roster if s.alliance == "blue")
    a, tl = m.clock_fits["auto"], m.clock_fits["teleop"]
    lines = [
        f"layout {m.layout}   clock rate {m.clock_rate} video s per clock s",
        f"red {red}   blue {blue}",
        f"target {m.target_team} ({m.target_alliance})   auto winner {m.auto_winner}",
        f"auto   {m.auto_start_t:.3f} - {m.auto_end_t:.3f}",
        f"teleop {m.teleop_start_t:.3f} - {m.teleop_end_t:.3f}",
        f"auto: {a.ticks} ticks, {a.dropped} dropped, line max {a.max_abs_s:.3f}, "
        f"timeline max {a.timeline_max_abs_s:.3f}",
        f"teleop: {tl.ticks} ticks, {tl.dropped} dropped, line max {tl.max_abs_s:.3f}, "
        f"timeline max {tl.timeline_max_abs_s:.3f}",
        *notes,
    ]
    _put_lines(panel, lines, 16, 40, 0.65)
    cells.append(panel)
    sheet = np.vstack([np.hstack(cells[:2]), np.hstack(cells[2:])])
    cv2.imwrite(str(run.path / "match_check.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])


def match(run: RunDir) -> None:
    cv2.setNumThreads(THREADS)
    info = VideoInfo.load(run.video_json)
    target = run.settings().team
    if abs(info.width / info.height - ov.REF_W / ov.REF_H) > 0.01:
        image = _layout_image(run, info, "match_layout.jpg", ov.LAYOUTS)
        raise NeedsInput(
            f"video is {info.width}x{info.height}; the FIRST 2026 overlays are 16:9",
            image,
            "no answer can fix this; use a video of a FIRST 2026 broadcast",
        )
    ocr = _Ocr()
    lay, layout_note = _find_layout(run.video, info, ocr)
    if lay is None:
        image = _layout_image(run, info, "match_layout.jpg", ov.LAYOUTS)
        raise NeedsInput(
            f"no known overlay layout shows a ticking clock ({layout_note})",
            image,
            "no answer can fix this; the overlay is neither known FIRST 2026 layout "
            "(both are drawn on the image)",
        )
    scan = _scan(run.video, info, ocr, lay)
    try:
        auto = _timeline(scan.ticks["auto"], "auto")
        teleop = _timeline(scan.ticks["teleop"], "teleop")
    except _UnreadableError as e:
        image = _layout_image(run, info, "match_layout.jpg", [lay])
        raise NeedsInput(
            f"could not time the match from the clock: {e}. Read with the {lay.name} "
            f"layout ({layout_note}); {scan.clock_reads} clock reads in total",
            image,
            "no answer can fix this; check the drawn boxes against the video's overlay",
        ) from e

    roster = _roster(run, scan, info, lay)
    slot = next((s for s in roster if s.team == target), None)
    if slot is None:
        raise ValueError(
            f"team {target} is not in this match: red "
            f"{[s.team for s in roster if s.alliance == 'red']}, blue "
            f"{[s.team for s in roster if s.alliance == 'blue']}"
        )

    provisional = _phases(auto, teleop, None)
    arrows, arrow_note = _arrow_winner(scan, provisional)
    fuel_winner, fuel = _fuel_winner(scan)
    winner = arrows if fuel_winner in (None, arrows) else None
    notes = [
        f"layout {lay.name} ({layout_note})",
        f"arrows: {arrow_note} -> {arrows}",
        f"fuel at end of pause: red {fuel['red']} blue {fuel['blue']} -> {fuel_winner}",
    ]
    notes += _arrow_timing(scan, provisional)

    w, h = info.width, info.height
    m = MatchInfo(
        layout=lay.name,
        scoreboard=ov.scaled(lay.scoreboard, w, h),
        views={v: ov.scaled(lay.views[v], w, h) for v in VIEW_NAMES},
        roster=roster,
        target_team=target,
        target_alliance=slot.alliance,
        auto_start_t=round(auto.video_t(0.0), 4),
        auto_end_t=round(auto.video_t(AUTO_S), 4),
        teleop_start_t=round(teleop.video_t(0.0), 4),
        teleop_end_t=round(teleop.video_t(TELEOP_S), 4),
        auto_winner=winner,
        phases=_phases(auto, teleop, winner),
        clock_fits={"auto": auto.fit, "teleop": teleop.fit},
        clock_rate=_clock_rate(auto, teleop),
    )
    m.save(run.match_json)
    _check_sheet(run, info, ocr, lay, m, notes)
    print(f"match: red {[s.team for s in roster[:3]]} blue {[s.team for s in roster[3:]]}")
    starts = {"auto": m.auto_start_t, "teleop": m.teleop_start_t}
    for name, fit in m.clock_fits.items():
        print(
            f"match: {name} starts {starts[name]:.3f} s ({fit.ticks} ticks, {fit.dropped} "
            f"dropped; line slope {fit.slope}, max {fit.max_abs_s:.3f} s; timeline max "
            f"{fit.timeline_max_abs_s} s)"
        )
    print(f"match: clock rate {m.clock_rate} video s per clock s")
    for note in notes:
        print(f"match: {note}")
    print(f"match: auto winner {winner}; team {target} is {slot.alliance} {slot.station}")


STAGE = Stage(
    "match",
    inputs=lambda r: (r.video, r.video_json, r.settings_json),
    outputs=lambda r: (r.match_json,),
    fn=match,
)
