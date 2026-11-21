"""match: reads video, video.json, settings.json.

Writes match.json: overlay layout and panel rectangles, the six-team roster, target team and
alliance, auto/teleop times in video seconds, and match phases with active hubs. Also
writes `match_check.jpg`, a contact sheet for a human to check the result.

Assumes the standard FIRST 2026 overlay (`scout.games.first_overlay_2026`). One decode of
the scoreboard band at the full frame rate does all the reading:

1. The clock box is OCRed only when its pixels change, so each read lands on the first
   frame of a new clock value: a tick moment, exact to one frame.
2. Ticks in auto and in teleop are fitted separately with a line (video time against
   elapsed period time), dropping misreads one at a time. The intercepts are auto start
   and teleop start; residuals go into `clock_fits`.
3. The six team numbers are read every 10 clock seconds and voted per slot.
4. The fuel counters are read at the end of the auto-teleop pause, and the active-hub
   arrows are sampled on every frame. The arrows name the auto winner (its hub is off in
   shifts 1 and 3); the counters must not contradict them.
"""

from __future__ import annotations

import re
from collections import Counter, deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray
from rapidocr_onnxruntime import RapidOCR  # pyright: ignore[reportMissingTypeStubs]

from scout.games import first_overlay_2026 as ov
from scout.games.rebuilt2026 import AUTO_S, PHASES, TELEOP_S, active_hubs, other
from scout.run import NeedsInput, RunDir, Stage
from scout.schemas.common import ALLIANCES, VIEW_NAMES, Alliance, PixelRect
from scout.schemas.match import ClockFit, MatchInfo, PhaseSpan, TeamSlot
from scout.schemas.video import VideoInfo
from scout.video import THREADS, frame_at, frames

Image = NDArray[np.uint8]
Period = Literal["auto", "teleop"]
ScanState = Literal["pre", "auto", "pause", "teleop", "done"]

PAUSE_CLOCK = int(TELEOP_S)  # the clock holds 2:20 through the auto-teleop pause
CHANGE_LEVEL = 4.0  # mean absolute grey difference that means a new clock face
MIN_CONFIDENCE = 0.8
GIVE_UP_FRAMES = 10  # OCR attempts on a changed clock face before treating it as no clock
TEAM_READ_EVERY = 10  # read the team numbers when the clock shows a multiple of this
OUTLIER_S = 0.15
MIN_TICKS: dict[Period, int] = {"auto": 8, "teleop": 30}
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
    """The overlay could not be read where the standard layout puts it."""


def _cut(image: Image, rect: PixelRect, origin: PixelRect) -> Image:
    """The part of `image` (a crop at `origin`) under full-frame `rect`."""
    x, y = rect.x - origin.x, rect.y - origin.y
    return image[y : y + rect.h, x : x + rect.w]


def _mean_bgr(image: Image) -> tuple[float, float, float]:
    b, g, r = image.reshape(-1, 3).mean(axis=0).tolist()
    return (b, g, r)


def _parse_clock(text: str) -> int | None:
    """Clock text "m:ss" -> seconds shown, or None."""
    m = re.fullmatch(r"(\d):([0-5]\d)", text.replace(" ", ""))
    return int(m[1]) * 60 + int(m[2]) if m else None


def _parse_int(text: str) -> int | None:
    """Leading integer of an OCR string ("61/120" -> 61), reading O as 0."""
    m = re.match(r"\d+", text.strip().replace("O", "0").replace("o", "0"))
    return int(m[0]) if m else None


def _scan(video: Path, info: VideoInfo, ocr: _Ocr) -> _Scan:
    """Decode the scoreboard band at full rate once and read it (steps 1, 3 and 4)."""
    w, h = info.width, info.height
    band = ov.scaled(ov.SCOREBOARD, w, h)
    clock = ov.scaled(ov.CLOCK, w, h)
    active = {a: ov.scaled(ov.ACTIVE[a], w, h) for a in ALLIANCES}
    fuel = {a: ov.scaled(ov.FUEL[a], w, h) for a in ALLIANCES}
    slots = [ov.scaled(b, w, h) for a in ALLIANCES for b in ov.TEAMS[a]]

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

    for f in frames(video, info, crop=band):
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


def _fit(points: list[tuple[float, float]], period: Period) -> ClockFit:
    """Fit `t = start + slope * elapsed`, dropping the worst tick while any residual is over
    OUTLIER_S (step 2). Raises _UnreadableError when too few ticks remain."""
    t = np.array([p[0] for p in points], dtype=np.float64)
    e = np.array([p[1] for p in points], dtype=np.float64)
    keep = np.ones(len(points), dtype=bool)
    while True:
        if int(keep.sum()) < MIN_TICKS[period]:
            raise _UnreadableError(
                f"{period}: only {int(keep.sum())} consistent clock ticks "
                f"(of {len(points)} read), need {MIN_TICKS[period]}"
            )
        slope, start = (float(v) for v in np.polyfit(e[keep], t[keep], 1))
        resid = t - (start + slope * e)
        worst = int(np.argmax(np.where(keep, np.abs(resid), -1.0)))
        if abs(resid[worst]) <= OUTLIER_S:
            break
        keep[worst] = False
    if not 0.98 < slope < 1.02:
        raise _UnreadableError(f"{period}: clock runs at {slope:.3f} s per second, expected 1")
    used = resid[keep]
    return ClockFit(
        start_t=round(start, 4),
        slope=round(slope, 5),
        ticks=int(keep.sum()),
        dropped=int((~keep).sum()),
        rms_s=round(float(np.sqrt(np.mean(used**2))), 4),
        max_abs_s=round(float(np.abs(used).max()), 4),
    )


def _vote(texts: list[str], min_votes: int) -> int | None:
    """The integer most of `texts` agree on, or None without a clear majority."""
    values = [v for v in map(_parse_int, texts) if v is not None]
    if not values:
        return None
    best, count = Counter(values).most_common(1)[0]
    return best if count >= min_votes and count * 2 > len(values) else None


def _roster(run: RunDir, scan: _Scan, info: VideoInfo) -> list[TeamSlot]:
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
            image = _layout_image(run, info, "match_roster.jpg")
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


def _phases(auto: ClockFit, teleop: ClockFit, winner: Alliance | None) -> list[PhaseSpan]:
    """Game phases in video seconds: auto on the auto line, every other phase on teleop's.
    The auto-teleop pause (robots disabled) lies between auto's end and transition's start
    and belongs to no phase."""

    def video_t(fit: ClockFit, period_t: float) -> float:
        return round(fit.start_t + fit.slope * period_t, 4)

    spans: list[PhaseSpan] = []
    for p in PHASES:
        hubs = active_hubs(p.name, winner)
        fit, offset = (auto, 0.0) if p.name == "auto" else (teleop, AUTO_S)
        spans.append(
            PhaseSpan(
                name=p.name,
                t_start=video_t(fit, p.start - offset),
                t_end=video_t(fit, p.end - offset),
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


def _arrow_timing(scan: _Scan, auto: ClockFit, phases: list[PhaseSpan]) -> list[str]:
    """Independent check of the clock fits: when the arrows first light (FMS enables the
    hubs at auto start) and when they switch at each shift and endgame boundary, relative
    to the fitted times. Returns printable notes."""
    t = np.array(scan.t)
    red, blue = np.array(scan.lit["red"]), np.array(scan.lit["blue"])
    notes: list[str] = []
    lit_on = t[(t > auto.start_t - 3) & red & blue]
    if lit_on.size:
        notes.append(f"arrows light at {lit_on[0]:.3f} ({lit_on[0] - auto.start_t:+.3f} vs fit)")
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
            f"{len(offsets)} fitted shift boundaries"
        )
    return notes


def _fuel_winner(scan: _Scan) -> tuple[Alliance | None, dict[Alliance, int | None]]:
    fuel: dict[Alliance, int | None] = {a: _vote(scan.fuel_texts[a], 2) for a in ALLIANCES}
    red, blue = fuel["red"], fuel["blue"]
    if red is None or blue is None or red == blue:
        return None, fuel
    return ("red" if red > blue else "blue"), fuel


def _draw_boxes(image: Image, info: VideoInfo) -> Image:
    """A copy of a full frame with every overlay box and camera panel drawn on it."""
    out = image.copy()
    w, h = info.width, info.height
    boxes = [ov.CLOCK, *ov.FUEL.values(), *ov.ACTIVE.values()]
    boxes += [b for a in ALLIANCES for b in ov.TEAMS[a]]
    for b in boxes:
        r = ov.scaled(b, w, h)
        cv2.rectangle(out, (r.x, r.y), (r.x + r.w, r.y + r.h), (0, 255, 255), 2)
    for name in VIEW_NAMES:
        r = ov.scaled(ov.VIEWS[name], w, h)
        cv2.rectangle(out, (r.x + 2, r.y + 2), (r.x + r.w - 3, r.y + r.h - 3), (255, 160, 0), 3)
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


def _layout_image(run: RunDir, info: VideoInfo, name: str) -> Path:
    """Save a mid-video frame with the expected boxes drawn; returns its path."""
    frame = frame_at(run.video, info, info.frame_count / info.fps / 2)
    path = run.path / name
    cv2.imwrite(str(path), _draw_boxes(frame.image, info), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path


def _put_lines(image: Image, lines: Sequence[str], x: int, y: int, scale: float) -> None:
    step = round(34 * scale / 0.9)
    for i, line in enumerate(lines):
        pos = (x, y + i * step)
        cv2.putText(image, line, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(
            image, line, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA
        )


def _check_sheet(run: RunDir, info: VideoInfo, ocr: _Ocr, m: MatchInfo, notes: list[str]) -> None:
    """Write match_check.jpg: frames at auto start, teleop start and teleop end with the boxes
    drawn and what OCR reads there, plus a panel of the results."""
    w, h = info.width, info.height
    clock = ov.scaled(ov.CLOCK, w, h)
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
        crops += [_cut(frame.image, ov.scaled(ov.FUEL[a], w, h), full) for a in ALLIANCES]
        (ct, _), (rt, _), (bt, _) = ocr.read(crops)
        lit = [
            a
            for a in ALLIANCES
            if ov.is_lit(_mean_bgr(_cut(frame.image, ov.scaled(ov.ACTIVE[a], w, h), full)))
        ]
        cell = np.asarray(
            cv2.resize(_draw_boxes(frame.image, info), (960, 540), interpolation=cv2.INTER_AREA),
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
        f"red {red}   blue {blue}",
        f"target {m.target_team} ({m.target_alliance})   auto winner {m.auto_winner}",
        f"auto   {m.auto_start_t:.3f} - {m.auto_end_t:.3f}",
        f"teleop {m.teleop_start_t:.3f} - {m.teleop_end_t:.3f}",
        f"auto fit: {a.ticks} ticks, {a.dropped} dropped, rms {a.rms_s:.3f} max {a.max_abs_s:.3f}",
        f"teleop fit: {tl.ticks} ticks, {tl.dropped} dropped, rms {tl.rms_s:.3f} "
        f"max {tl.max_abs_s:.3f}",
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
        image = _layout_image(run, info, "match_layout.jpg")
        raise NeedsInput(
            f"video is {info.width}x{info.height}; the standard FIRST 2026 overlay is 16:9",
            image,
            "no answer can fix this; use a video of the standard FIRST broadcast",
        )
    ocr = _Ocr()
    scan = _scan(run.video, info, ocr)
    try:
        auto = _fit(scan.ticks["auto"], "auto")
        teleop = _fit(scan.ticks["teleop"], "teleop")
    except _UnreadableError as e:
        image = _layout_image(run, info, "match_layout.jpg")
        raise NeedsInput(
            f"could not time the match from the clock: {e}. Expected the standard FIRST 2026 "
            f"overlay with the game clock in the yellow box under the score bar; "
            f"{scan.clock_reads} clock reads in total",
            image,
            "no answer can fix this; check the drawn boxes against the video's overlay",
        ) from e

    roster = _roster(run, scan, info)
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
        f"arrows: {arrow_note} -> {arrows}",
        f"fuel at end of pause: red {fuel['red']} blue {fuel['blue']} -> {fuel_winner}",
    ]
    notes += _arrow_timing(scan, auto, provisional)

    m = MatchInfo(
        scoreboard=ov.scaled(ov.SCOREBOARD, info.width, info.height),
        views={v: ov.scaled(ov.VIEWS[v], info.width, info.height) for v in VIEW_NAMES},
        roster=roster,
        target_team=target,
        target_alliance=slot.alliance,
        auto_start_t=auto.start_t,
        auto_end_t=round(auto.start_t + auto.slope * AUTO_S, 4),
        teleop_start_t=teleop.start_t,
        teleop_end_t=round(teleop.start_t + teleop.slope * TELEOP_S, 4),
        auto_winner=winner,
        phases=_phases(auto, teleop, winner),
        clock_fits={"auto": auto, "teleop": teleop},
    )
    m.save(run.match_json)
    _check_sheet(run, info, ocr, m, notes)
    print(f"match: red {[s.team for s in roster[:3]]} blue {[s.team for s in roster[3:]]}")
    for name, fit in m.clock_fits.items():
        print(
            f"match: {name} starts {fit.start_t:.3f} s ({fit.ticks} ticks, {fit.dropped} "
            f"dropped, rms {fit.rms_s:.3f} s, max {fit.max_abs_s:.3f} s, slope {fit.slope})"
        )
    for note in notes:
        print(f"match: {note}")
    print(f"match: auto winner {winner}; team {target} is {slot.alliance} {slot.station}")


STAGE = Stage(
    "match",
    inputs=lambda r: (r.video, r.video_json, r.settings_json),
    outputs=lambda r: (r.match_json,),
    fn=match,
)
