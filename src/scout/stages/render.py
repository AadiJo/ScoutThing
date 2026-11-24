"""render: reads video, tracks.parquet, identity.json, corner_tracks.parquet,
corner_identity.json, calibration.json, match.json.

Writes overlay.mp4: the full match video with the tracking drawn on the main panel, robots
placed through a corner camera boxed on that corner panel, and a top-down field map with
every tracked robot in place of the sponsor card between the two corner views (corner-camera
positions as squares, main-view ones as dots). Also writes `overlay_check.jpg`, a contact
sheet of a few rendered frames at moments chosen from the data (start, auto, the most crowded
sample, the longest interpolated stretch, a robot behind a hub, the target's longest
corner-camera stretch, teleop, endgame, end).

One `scout.video.frames` decode pass at OUTPUT_HZ runs in a reader thread, the main
thread draws, and a writer thread converts each frame to I420 and pipes it into an
`ffmpeg` libx264 encoder, which also copies the source audio. Queues between them hold
at most QUEUE_FRAMES frames each, so the three steps overlap in a few frames of memory.
Tracks are 10 Hz samples; positions and boxes are interpolated between consecutive
samples of one track, never across a gap. Robots the tracks do not have are not drawn.
"""

from __future__ import annotations

import bisect
import json
import math
import queue
import resource
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO

import cv2
import numpy as np
import polars as pl
from numpy.typing import NDArray

from scout.assets import asset
from scout.camera import Camera
from scout.games.rebuilt2026 import FIELD_LENGTH_M, FIELD_WIDTH_M, HUBS, ROBOT_FOOTPRINT_M
from scout.positions import load_positions
from scout.run import RunDir, Stage
from scout.schemas.common import Alliance, PixelRect
from scout.schemas.detections import SAMPLE_HZ
from scout.schemas.identity import LabelSegment
from scout.schemas.match import MatchInfo
from scout.schemas.video import VideoInfo
from scout.video import FFMPEG, Frame, frames

F64 = NDArray[np.float64]
Image = NDArray[np.uint8]
BGR = tuple[int, int, int]

OUTPUT_HZ = 30.0
"""On Qualification 16, 30 fps renders in 27 s and 60 fps in 45 s (twice the drawing and
encoding), for motion nobody needs to inspect."""
THREADS = 8
X264_PRESET = "veryfast"
X264_CRF = 23
TRAIL_S = 3.0
LOW_CONFIDENCE = 0.5
"""A label segment below this confidence is shown with a trailing "?"."""

FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.45
SMALL_SCALE = 0.38
PAD = 3

COLORS: dict[str, BGR] = {"red": (60, 60, 235), "blue": (235, 140, 30), "unknown": (190, 190, 190)}
TARGET: BGR = (0, 225, 255)
WHITE: BGR = (255, 255, 255)
GREY: BGR = (165, 165, 165)
DIM: BGR = (120, 120, 120)
BLACK: BGR = (0, 0, 0)
BACKING: BGR = (20, 20, 20)
CARD_BG: BGR = (18, 18, 18)

BOX_THICK = 2
"""Every box; 1 px boxes vanish in some frames after x264 at CRF 23."""
TARGET_RING = 2
HUB_RING = 3
DASH_ON, DASH_OFF = 6, 5
DOT_R = 6
CORNER_R = 5
"""Half side of the square marking a corner-camera position on the map."""
TARGET_DOT_R = 8
STATUS_H = 22
LEGEND_H = 18
MAP_MARGIN = 6
MAP_CONTEXT_M = 0.4
"""Carpet shown around the field walls on the map."""
CHECK_SCALE = 0.5
QUEUE_FRAMES = 3
"""Frames buffered between decoder, drawing and encoder (6 MB each)."""


@dataclass(frozen=True)
class _Track:
    """One track's samples as arrays, for lookups at any source frame."""

    track_id: int
    color: str
    view: str
    """`main`, or the corner view that placed it (its boxes are on that panel)."""
    frames: NDArray[np.int64]
    xy: F64
    box: F64
    """(n, 4) x1 y1 x2 y2 in full-frame pixels, NaN on interpolated samples."""
    measured: NDArray[np.bool_]
    run_start: NDArray[np.int64]
    """Index of the first sample of the gap-free run each sample belongs to."""


@dataclass(frozen=True)
class _Pose:
    """A track drawn at one output frame: field position, box (None when the samples
    around it are not both measured), the recent trail, and its label."""

    track: _Track
    xy: F64
    box: F64 | None
    trail: F64
    label: str | None
    is_target: bool


@dataclass(frozen=True)
class _MapGeom:
    """The field map inside the sponsor card: a pre-drawn background and the transform
    from field metres to card pixels (flipped so it reads like the main camera)."""

    background: Image
    scale: float
    origin: tuple[float, float]
    flip_x: bool
    flip_y: bool

    def to_px(self, xy: F64) -> F64:
        """Field (..., 2) metres to card pixels (..., 2)."""
        fx = FIELD_LENGTH_M - xy[..., 0] if self.flip_x else xy[..., 0]
        fy = xy[..., 1] if self.flip_y else FIELD_WIDTH_M - xy[..., 1]
        return np.stack([self.origin[0] + fx * self.scale, self.origin[1] + fy * self.scale], -1)


@dataclass
class _Timer:
    s: dict[str, float]

    def add(self, key: str, since: float) -> float:
        now = time.perf_counter()
        self.s[key] = self.s.get(key, 0.0) + now - since
        return now


def _load_tracks(df: pl.DataFrame, gap_frames: int) -> list[_Track]:
    """Every track (`scout.positions` columns), sorted by frame, with gap-free runs marked."""
    df = df.sort("track_id", "frame")
    out: list[_Track] = []
    for (tid,), g in df.group_by("track_id", maintain_order=True):
        fr = g["frame"].to_numpy().astype(np.int64)
        breaks = np.flatnonzero(np.diff(fr) > gap_frames) + 1
        starts = np.zeros(len(fr), np.int64)
        for b in breaks:
            starts[b:] = b
        out.append(
            _Track(
                track_id=int(str(tid)),
                color=str(g["color"][0]),
                view=str(g["view"].drop_nulls().first() or "main"),
                frames=fr,
                xy=g.select("x", "y").to_numpy().astype(np.float64),
                box=g.select("x1", "y1", "x2", "y2")
                .fill_null(np.nan)
                .to_numpy()
                .astype(np.float64),
                measured=(g["state"] == "measured").to_numpy(),
                run_start=starts,
            )
        )
    return out


def _label(segments: Sequence[LabelSegment], frame: int) -> tuple[str | None, int | None]:
    """Label text and the single team for a track at `frame`. Several teams covering the
    frame show as "a/b?" with no team; a low-confidence label gets a "?"."""
    hits = [s for s in segments if s.frame_start <= frame <= s.frame_end and s.team is not None]
    teams = sorted({s.team for s in hits if s.team is not None})
    if not teams:
        return None, None
    if len(teams) > 1:
        return "/".join(map(str, teams)) + "?", None
    unsure = min(s.confidence for s in hits) < LOW_CONFIDENCE
    return f"{teams[0]}{'?' if unsure else ''}", teams[0]


def _pose(
    tr: _Track, frame: int, gap: int, trail_frames: int, segs: Sequence[LabelSegment], target: int
) -> _Pose | None:
    """The track at source `frame`, interpolated between the samples around it, or None
    outside the track or inside a gap."""
    if frame < tr.frames[0] or frame > tr.frames[-1]:
        return None
    i = int(np.searchsorted(tr.frames, frame, "right")) - 1
    fa = int(tr.frames[i])
    if fa == frame:
        xy = tr.xy[i]
        box = tr.box[i] if tr.measured[i] else None
    else:
        fb = int(tr.frames[i + 1])
        if fb - fa > gap:
            return None
        w = (frame - fa) / (fb - fa)
        xy = tr.xy[i] + w * (tr.xy[i + 1] - tr.xy[i])
        both = tr.measured[i] and tr.measured[i + 1]
        box = tr.box[i] + w * (tr.box[i + 1] - tr.box[i]) if both else None
    first = max(int(tr.run_start[i]), int(np.searchsorted(tr.frames, frame - trail_frames)))
    trail = np.vstack([tr.xy[first : i + 1], xy[None]])
    text, team = _label(segs, frame)
    return _Pose(tr, xy, box, trail, text, team == target)


def _map_geom(card: PixelRect, cam: Camera) -> _MapGeom:
    """Fit the AdvantageScope field image into the card below the status line, turned so
    left and near on the map are left and near in the main view."""
    meta = json.loads(asset("field2d_meta").read_text())
    image = cv2.imread(str(asset("field2d_image")))
    if image is None:
        raise FileNotFoundError(asset("field2d_image"))
    (l0, t0), (r0, b0) = meta["field_top_left_px"], meta["field_bottom_right_px"]
    src_ppm = (float(r0) - float(l0)) / float(meta["field_length_m"])
    ends = cam.project(
        np.array([[0.0, FIELD_WIDTH_M / 2, 0.0], [FIELD_LENGTH_M, FIELD_WIDTH_M / 2, 0.0]])
    )
    flip_x = bool(ends[1, 0] < ends[0, 0])
    flip_y = bool(cam.center[1] > FIELD_WIDTH_M / 2)

    avail_w = card.w - 2 * MAP_MARGIN
    avail_h = card.h - STATUS_H - LEGEND_H - 2 * MAP_MARGIN
    span_l, span_w = FIELD_LENGTH_M + 2 * MAP_CONTEXT_M, FIELD_WIDTH_M + 2 * MAP_CONTEXT_M
    scale = min(avail_w / span_l, avail_h / span_w)
    out_w, out_h = round(span_l * scale), round(span_w * scale)
    ctx = MAP_CONTEXT_M * src_ppm
    crop = image[
        max(0, round(float(t0) - ctx)) : round(float(b0) + ctx),
        max(0, round(float(l0) - ctx)) : round(float(r0) + ctx),
    ]
    field = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)
    if flip_x or flip_y:
        field = cv2.flip(field, -1 if flip_x and flip_y else (1 if flip_x else 0))
    background = np.full((card.h, card.w, 3), CARD_BG, np.uint8)
    ox = (card.w - out_w) // 2
    oy = STATUS_H + MAP_MARGIN + (avail_h - out_h) // 2
    background[oy : oy + out_h, ox : ox + out_w] = field
    origin = (ox + MAP_CONTEXT_M * scale, oy + MAP_CONTEXT_M * scale)
    return _MapGeom(background, scale, origin, flip_x, flip_y)


def _text(img: Image, text: str, org: tuple[int, int], fg: BGR, bg: BGR, scale: float) -> int:
    """Draw `text` with its baseline-left at `org` on a filled backing; returns the x
    just past it."""
    (w, h), base = cv2.getTextSize(text, FONT, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - PAD, y - h - PAD), (x + w + PAD, y + base), bg, -1)
    cv2.putText(img, text, (x, y), FONT, scale, fg, 1, cv2.LINE_AA)
    return x + w + 2 * PAD


def _pieces(img: Image, pieces: Sequence[tuple[str, BGR]], org: tuple[int, int]) -> int:
    """Draw coloured text pieces left to right on the card background; returns end x."""
    x, y = org
    for s, color in pieces:
        cv2.putText(img, s, (x, y), FONT, TEXT_SCALE, color, 1, cv2.LINE_AA)
        x += cv2.getTextSize(s, FONT, TEXT_SCALE, 1)[0][0]
    return x


def _dashed(img: Image, pts: F64, color: BGR, thick: int) -> None:
    """Closed dashed polygon through `pts` (n, 2)."""
    for a, b in zip(pts, np.roll(pts, -1, axis=0), strict=True):
        length = float(np.hypot(*(b - a)))
        d = 0.0
        while d < length:
            p = a + (b - a) * (d / length)
            q = a + (b - a) * (min(d + DASH_ON, length) / length)
            cv2.line(
                img,
                (round(p[0]), round(p[1])),
                (round(q[0]), round(q[1])),
                color,
                thick,
                cv2.LINE_AA,
            )
            d += DASH_ON + DASH_OFF


def _draw_video(panel: Image, offset: PixelRect, poses: Sequence[_Pose], cam: Camera) -> None:
    """Boxes on the main panel: solid box for a measured stretch, dashed carpet footprint
    (from the field position) for an interpolated one. Target last, so it sits on top."""
    half = ROBOT_FOOTPRINT_M / 2
    corners = np.array(
        [[-half, -half, 0.0], [half, -half, 0.0], [half, half, 0.0], [-half, half, 0.0]]
    )
    shift = np.array([offset.x, offset.y], np.float64)
    for p in sorted(poses, key=lambda p: (p.is_target, p.label is not None)):
        color = COLORS[p.track.color]
        if p.box is not None:
            x1, y1, x2, y2 = (round(v) for v in p.box - np.tile(shift, 2))
            if p.is_target:
                r = TARGET_RING + BOX_THICK
                cv2.rectangle(panel, (x1 - r, y1 - r), (x2 + r, y2 + r), TARGET, TARGET_RING)
            cv2.rectangle(panel, (x1, y1), (x2, y2), color, BOX_THICK)
            anchor_x, top, bottom = x1, y1, y2
        else:
            foot = cam.project(np.hstack([p.xy, [0.0]]) + corners) - shift
            if not np.isfinite(foot).all():
                continue
            if p.is_target:
                _dashed(panel, foot, TARGET, TARGET_RING + 1)
            _dashed(panel, foot, color, BOX_THICK)
            anchor_x, top, bottom = (
                round(foot[:, 0].min()),
                round(foot[:, 1].min()),
                round(foot[:, 1].max()),
            )
        tag = p.label if p.label else f"t{p.track.track_id}"
        if p.box is None:
            tag = "~" + tag
        scale = TEXT_SCALE if p.label else SMALL_SCALE
        (_, h), _ = cv2.getTextSize(tag, FONT, scale, 1)
        y = top - PAD - TARGET_RING - 2
        if y - h - PAD < 0:
            y = bottom + h + PAD + TARGET_RING + 2
        fg, bg = (
            (BLACK, TARGET) if p.is_target else ((WHITE, color) if p.label else (GREY, BACKING))
        )
        _text(panel, tag, (max(PAD, anchor_x), y), fg, bg, scale)


def _draw_corner(panel: Image, offset: PixelRect, poses: Sequence[_Pose]) -> None:
    """Boxes on a corner panel for the robots that corner camera placed, labelled like the
    main panel's. Target last."""
    shift = np.tile(np.array([offset.x, offset.y], np.float64), 2)
    for p in sorted(poses, key=lambda p: (p.is_target, p.label is not None)):
        if p.box is None:
            continue
        color = COLORS[p.track.color]
        x1, y1, x2, y2 = (round(v) for v in p.box - shift)
        if p.is_target:
            r = TARGET_RING + BOX_THICK
            cv2.rectangle(panel, (x1 - r, y1 - r), (x2 + r, y2 + r), TARGET, TARGET_RING)
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, BOX_THICK)
        tag = p.label if p.label else f"t{p.track.track_id}"
        scale = TEXT_SCALE if p.label else SMALL_SCALE
        (_, h), _ = cv2.getTextSize(tag, FONT, scale, 1)
        y = y1 - PAD - TARGET_RING - 2
        if y - h - PAD < 0:
            y = y2 + h + PAD + TARGET_RING + 2
        fg, bg = (
            (BLACK, TARGET) if p.is_target else ((WHITE, color) if p.label else (GREY, BACKING))
        )
        _text(panel, tag, (max(PAD, x1), y), fg, bg, scale)


def _draw_map(
    card: Image, geom: _MapGeom, poses: Sequence[_Pose], hubs: Sequence[Alliance] | None
) -> None:
    """Active hubs outlined (white outside the alliance colour, which alone vanishes on
    the map art's coloured stripes), then each robot's trail and dot, filled when measured and
    hollow when interpolated, or a white-edged square when a corner camera placed it; labels
    beside them. Target last."""
    for name in hubs or ():
        hub = HUBS[name]
        c = geom.to_px(np.array([[hub.x0, hub.y0], [hub.x1, hub.y1]]))
        (x1, y1), (x2, y2) = np.round(c.min(0)).astype(int), np.round(c.max(0)).astype(int)
        for ring, color in ((HUB_RING + 2, WHITE), (HUB_RING, COLORS[name])):
            cv2.rectangle(card, (x1 - ring, y1 - ring), (x2 + ring, y2 + ring), color, 2)
    for p in sorted(poses, key=lambda p: (p.is_target, p.label is not None)):
        color = COLORS[p.track.color]
        trail = np.round(geom.to_px(p.trail)).astype(np.int32)
        if len(trail) > 1:
            cv2.polylines(
                card,
                [trail],
                False,
                TARGET if p.is_target else color,
                2 if p.is_target else 1,
                cv2.LINE_AA,
            )
        cx, cy = (int(v) for v in trail[-1])
        r = TARGET_DOT_R if p.is_target else DOT_R
        if p.is_target:
            cv2.circle(card, (cx, cy), r + TARGET_RING + 1, TARGET, TARGET_RING, cv2.LINE_AA)
        if p.track.view != "main":
            q = CORNER_R + (TARGET_DOT_R - DOT_R if p.is_target else 0)
            cv2.rectangle(card, (cx - q - 1, cy - q - 1), (cx + q + 1, cy + q + 1), BLACK, 2)
            cv2.rectangle(card, (cx - q, cy - q), (cx + q, cy + q), color, -1)
            cv2.rectangle(card, (cx - q, cy - q), (cx + q, cy + q), WHITE, 1)
        else:
            cv2.circle(card, (cx, cy), r, BLACK, -1 if p.box is not None else 2, cv2.LINE_AA)
            cv2.circle(card, (cx, cy), r - 1, color, -1 if p.box is not None else 2, cv2.LINE_AA)
        fg, bg = (BLACK, TARGET) if p.is_target else (WHITE if p.label else GREY, BACKING)
        _text(card, p.label or f"t{p.track.track_id}", (cx + r + 5, cy + 4), fg, bg, SMALL_SCALE)


def _clock(match: MatchInfo, t: float) -> tuple[str, str, Sequence[Alliance] | None]:
    """Phase name, match clock (time left as the scoreboard shows it) and active hubs at
    video time `t`."""
    phase = next((p for p in match.phases if p.t_start <= t < p.t_end), None)
    if t < match.auto_start_t:
        return "PRE-MATCH", "", None
    if t >= match.teleop_end_t:
        return "POST-MATCH", "0:00", None
    if match.auto_end_t <= t < match.teleop_start_t:
        return "AUTO > TELEOP", "", None
    period, fit = (
        (20.0, match.clock_fits.get("auto"))
        if t < match.auto_end_t
        else (140.0, match.clock_fits.get("teleop"))
    )
    start, slope = (
        (fit.start_t, fit.slope)
        if fit
        else ((match.auto_start_t, 1.0) if period == 20.0 else (match.teleop_start_t, 1.0))
    )
    left = max(0, min(int(period), math.ceil(period - (t - start) / slope - 1e-6)))
    name = phase.name.upper().replace("SHIFT", "SHIFT ") if phase else "?"
    return name, f"{left // 60}:{left % 60:02d}", phase.active_hubs if phase else None


def _draw_status(
    card: Image, match: MatchInfo, t: float, frame: int, poses: Sequence[_Pose]
) -> Sequence[Alliance] | None:
    """Status line (phase, clock, active hub, robots tracked, video time) and legend line
    (target team and how it is shown). Returns the active hubs for the map."""
    name, clock, hubs = _clock(match, t)
    y = STATUS_H - 7
    pieces: list[tuple[str, BGR]] = [(f"{name}  {clock}  ".rstrip() + "   ", WHITE)]
    if name not in ("PRE-MATCH", "POST-MATCH", "AUTO > TELEOP"):
        pieces.append(("hub ", GREY))
        if hubs is None:
            pieces.append(("unknown", GREY))
        elif len(hubs) == 2:
            pieces += [("RED", COLORS["red"]), ("+", GREY), ("BLUE", COLORS["blue"])]
        else:
            pieces.append((hubs[0].upper(), COLORS[hubs[0]]))
    _pieces(card, pieces, (8, y))
    right = f"{len(poses)}/6 tracked   t {t:6.1f}s  f {frame}"
    w = cv2.getTextSize(right, FONT, SMALL_SCALE, 1)[0][0]
    cv2.putText(card, right, (card.shape[1] - w - 8, y), FONT, SMALL_SCALE, GREY, 1, cv2.LINE_AA)

    y = card.shape[0] - 6
    shown = any(p.is_target for p in poses)
    x = _text(card, f"target {match.target_team}", (8, y), BLACK, TARGET, SMALL_SCALE)
    note = "" if shown else "not labelled now   "
    legend = (
        f"{note}solid = measured   dashed / hollow = interpolated   square = corner camera   "
        "t# = unlabelled"
    )
    cv2.putText(card, legend, (x + 4, y), FONT, SMALL_SCALE, GREY, 1, cv2.LINE_AA)
    return hubs


def _check_frames(
    tracks: Sequence[_Track],
    match: MatchInfo,
    info: VideoInfo,
    cam: Camera,
    step: int,
    segments: dict[int, list[LabelSegment]],
) -> dict[int, str]:
    """Output frames worth inspecting, with a caption each: start, end, auto, the most
    crowded sample, the longest interpolated stretch, a robot just behind a hub as the
    main camera sees it, the middle of the target's longest corner-camera stretch, mid teleop
    and endgame."""

    def at(t: float) -> int:
        return min(info.frame_count - 1, max(0, round(t * info.fps / step) * step))

    picks: dict[int, str] = {0: "start", at(info.duration_s - 0.5): "end"}
    picks[at(match.auto_start_t + 5)] = "auto +5s"
    picks[at((match.teleop_start_t + match.teleop_end_t) / 2)] = "mid teleop"
    picks[at(match.teleop_end_t - 15)] = "endgame"
    counts: dict[int, int] = {}
    best_run: tuple[int, int] = (0, -1)
    behind: tuple[float, int] = (math.inf, -1)
    far_sign = -1.0 if cam.center[1] > FIELD_WIDTH_M / 2 else 1.0
    for tr in tracks:
        for f in tr.frames.tolist():
            counts[f] = counts.get(f, 0) + 1
        run = 0
        for k, m in enumerate(tr.measured.tolist()):
            run = 0 if m else run + 1
            if run > best_run[0]:
                best_run = (run, int(tr.frames[k - run // 2]))
        for hub in HUBS.values() if tr.view == "main" else ():
            hx, hy = (hub.x0 + hub.x1) / 2, (hub.y0 + hub.y1) / 2
            ok = (
                tr.measured
                & (np.abs(tr.xy[:, 0] - hx) < (hub.x1 - hub.x0) / 2)
                & ((tr.xy[:, 1] - hy) * far_sign > 0)
            )
            if ok.any():
                d = np.hypot(tr.xy[:, 0] - hx, tr.xy[:, 1] - hy)
                k = int(np.argmin(np.where(ok, d, np.inf)))
                if d[k] < behind[0]:
                    behind = (float(d[k]), int(tr.frames[k]))
    if counts:
        picks.setdefault(max(counts, key=lambda f: (counts[f], -f)), "most robots")
    if best_run[1] >= 0:
        picks.setdefault(best_run[1], "longest interpolated")
    if behind[1] >= 0:
        picks.setdefault(behind[1], "behind a hub")
    corner = [
        s
        for tr in tracks
        if tr.view != "main"
        for s in segments.get(tr.track_id, [])
        if s.team == match.target_team
    ]
    if corner:
        s = max(corner, key=lambda s: s.frame_end - s.frame_start)
        picks.setdefault(round((s.frame_start + s.frame_end) / 2 / step) * step, "target, corner")
    return dict(sorted(picks.items()))


def _contact_sheet(path: Path, shots: dict[int, tuple[str, Image]], fps: float) -> None:
    """Two columns of half-size frames, each captioned with its moment and time."""
    tiles: list[Image] = []
    for frame, (caption, im) in sorted(shots.items()):
        tile = im.copy()
        _text(tile, f"{caption}  t={frame / fps:.1f}s  f{frame}", (8, 20), WHITE, BACKING, 0.55)
        tiles.append(tile)
    if not tiles:
        return
    if len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)]
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def _prefetch(source: Iterator[Frame]) -> Iterator[Frame]:
    """Run the decoder in a thread, at most QUEUE_FRAMES ahead, and hand over writable
    copies, so decoding overlaps drawing."""
    q: queue.Queue[Frame | BaseException | None] = queue.Queue(maxsize=QUEUE_FRAMES)
    stop = threading.Event()

    def put(item: Frame | BaseException | None) -> None:
        while not stop.is_set():
            try:
                q.put(item, timeout=0.2)
                return
            except queue.Full:
                continue

    def work() -> None:
        try:
            for f in source:
                if stop.is_set():
                    return
                put(Frame(f.frame, f.t, f.image.copy()))
            put(None)
        except BaseException as e:
            put(e)

    thread = threading.Thread(target=work, name="render-decode", daemon=True)
    thread.start()
    try:
        while (item := q.get()) is not None:
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        thread.join()


def _write_all(q: queue.Queue[Image | None], sink: IO[bytes], failed: list[BaseException]) -> None:
    """Writer thread: convert queued BGR frames to I420 (multithreaded in OpenCV, half
    the bytes, and spares ffmpeg its single-threaded conversion, which made encoding the
    slowest step) and pipe them into the encoder until None. On a write error, record it
    in `failed` and keep draining so the producer never blocks."""
    while (img := q.get()) is not None:
        if failed:
            continue
        try:
            sink.write(cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420).data)
        except BaseException as e:
            failed.append(e)


def _encoder(
    out: Path, source: Path, info: VideoInfo, step: int, log: IO[bytes]
) -> subprocess.Popen[bytes]:
    """An ffmpeg process reading raw I420 frames on stdin at fps/step and the source's
    audio (if any), writing H.264 mp4 to `out`."""
    cmd = [
        FFMPEG, "-nostdin", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{info.width}x{info.height}",
        "-framerate", f"{info.fps_num}/{info.fps_den * step}", "-i", "pipe:0",
        "-i", str(source), "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", X264_PRESET, "-crf", str(X264_CRF), "-pix_fmt", "yuv420p",
        "-threads", str(THREADS), "-c:a", "copy", "-movflags", "+faststart", "-f", "mp4", str(out),
    ]  # fmt: skip
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log)


def render(run: RunDir) -> None:
    cv2.setNumThreads(THREADS)
    info = VideoInfo.load(run.video_json)
    match = MatchInfo.load(run.match_json)
    cam = Camera.load(run.calibration_json, "main")
    pos = load_positions(run)
    main = match.views["main"]
    left, right = match.views["corner_left"], match.views["corner_right"]
    card = PixelRect(
        x=left.x + left.w, y=left.y, w=right.x - (left.x + left.w), h=info.height - left.y
    )

    gap = max(1, round(info.fps / SAMPLE_HZ))
    step = max(1, round(info.fps / OUTPUT_HZ))
    trail_frames = round(TRAIL_S * info.fps)
    tracks = _load_tracks(pos.tracks, gap)
    starts = [int(tr.frames[0]) for tr in tracks]
    order = sorted(range(len(tracks)), key=lambda k: starts[k])
    tracks = [tracks[k] for k in order]
    starts = [starts[k] for k in order]
    segments: dict[int, list[LabelSegment]] = {}
    for s in pos.segments:
        segments.setdefault(s.track_id, []).append(s)
    geom = _map_geom(card, cam)
    checks = _check_frames(tracks, match, info, cam, step, segments)
    shots: dict[int, tuple[str, Image]] = {}
    target = match.target_team

    timer = _Timer({})
    tmp = run.overlay.with_name(f".{run.overlay.stem}.tmp.mp4")
    n_out = 0
    wall = time.perf_counter()
    with tempfile.TemporaryFile() as log:
        enc = _encoder(tmp, run.video, info, step, log)
        assert enc.stdin is not None
        out: queue.Queue[Image | None] = queue.Queue(maxsize=QUEUE_FRAMES)
        failed: list[BaseException] = []
        writer = threading.Thread(target=_write_all, args=(out, enc.stdin, failed), daemon=True)
        writer.start()
        try:
            mark = time.perf_counter()
            for f in _prefetch(frames(run.video, info, sample_hz=info.fps / step)):
                mark = timer.add("decode wait", mark)
                img = f.image
                live = bisect.bisect_right(starts, f.frame)
                segs = segments.get
                poses = [
                    p
                    for tr in tracks[:live]
                    if (p := _pose(tr, f.frame, gap, trail_frames, segs(tr.track_id, ()), target))
                ]
                _draw_video(
                    img[main.y : main.y + main.h, main.x : main.x + main.w],
                    main,
                    [p for p in poses if p.track.view == "main"],
                    cam,
                )
                for name, r in (("corner_left", left), ("corner_right", right)):
                    _draw_corner(
                        img[r.y : r.y + r.h, r.x : r.x + r.w],
                        r,
                        [p for p in poses if p.track.view == name],
                    )
                card_img = geom.background.copy()
                hubs = _draw_status(card_img, match, f.t, f.frame, poses)
                _draw_map(card_img, geom, poses, hubs)
                img[card.y : card.y + card.h, card.x : card.x + card.w] = card_img
                if f.frame in checks:
                    small = cv2.resize(
                        img, None, fx=CHECK_SCALE, fy=CHECK_SCALE, interpolation=cv2.INTER_AREA
                    )
                    shots[f.frame] = (checks[f.frame], np.asarray(small, np.uint8))
                mark = timer.add("draw", mark)
                if failed:
                    raise failed[0]
                out.put(img)
                n_out += 1
                mark = timer.add("encode wait", mark)
            out.put(None)
            writer.join()
            if failed:
                raise failed[0]
            enc.stdin.close()
            code = enc.wait()
            timer.add("encode wait", mark)
        except BaseException:
            enc.kill()
            enc.wait()
            tmp.unlink(missing_ok=True)
            raise
        if code != 0:
            log.seek(0)
            tmp.unlink(missing_ok=True)
            err = log.read().decode(errors="replace").strip()
            raise RuntimeError(f"ffmpeg encoder failed ({code}): {err}")
    tmp.replace(run.overlay)
    _contact_sheet(run.path / "overlay_check.jpg", shots, info.fps)

    total = time.perf_counter() - wall
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**20
    print(
        f"render: {n_out} frames at {info.fps / step:g} fps, {len(tracks)} tracks, "
        f"{len(pos.segments)} label segments"
    )
    print("  main thread: " + "  ".join(f"{k} {v:.1f}s" for k, v in timer.s.items()))
    print(f"  wall {total:.1f}s; peak RSS {rss:.2f} GiB here, {child:.2f} GiB largest ffmpeg child")


STAGE = Stage(
    "render",
    inputs=lambda r: (
        r.video,
        r.tracks,
        r.identity_json,
        r.corner_tracks,
        r.corner_identity_json,
        r.calibration_json,
        r.match_json,
    ),
    outputs=lambda r: (r.overlay,),
    fn=render,
)
