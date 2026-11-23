"""The `frames` tool. Renders a contact sheet of the main camera (the target ringed, other
tracked robots boxed with their team) or a corner panel, logs each sheet to
`grade_work/frames.jsonl` (`served` reads it back) and refuses past IMAGE_CAP sheets.
`serve_frames` runs it as the stdio MCP server the grader session calls."""

from __future__ import annotations

import base64
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from scout.camera import Camera
from scout.games.rebuilt2026 import ROBOT_FOOTPRINT_M
from scout.run import RunDir
from scout.schemas.facts import Facts
from scout.schemas.grade import FrameSheet, TileRecord, View
from scout.schemas.match import MatchInfo
from scout.schemas.video import VideoInfo
from scout.stages.grade.base import IMAGE_CAP, JSON, SERVER, TOOL, frame_log, overlap
from scout.stages.grade.clock import clock as _clock
from scout.stages.grade.clock import home_view, relation
from scout.stages.grade.clock import phase as _phase
from scout.stages.grade.tracks import TRACKED_WITHIN_S, Others, Target, load_tracks
from scout.video import frame_at

Image = NDArray[np.uint8]

NEAR_BOX_M = 2.5
"""A boxed robot this close to the target can be the defender or partner a text names."""
NEARBY_S = 5.0
"""When untracked, the crop centres on the target's position within this much time."""
CROP_W, CROP_H = 800, 450
TILE_W = 640
FULL_W = 960
CORNER_SCALE = 2
CORNER_MAX_TILES = 2
"""Corner panels are small (652 x 344): they are served at twice their size, two per sheet."""
HEADER_H = 26
LEGEND_H = 24
RING: tuple[int, int, int] = (0, 225, 255)
RING_EST: tuple[int, int, int] = (0, 140, 255)
MARK: dict[str, tuple[int, int, int]] = {
    "opponent": (255, 0, 255),
    "partner": (80, 220, 80),
    "unknown": (230, 230, 230),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX
LEGEND = (
    "Yellow arc under it: target. Boxes: robots the tool tracked (team PARTNER/OPP, or "
    "'unlabelled'). A robot with NO box was not tracked."
)
CORNER_LEGEND = (
    "CORNER camera, uncalibrated: no ring and no boxes. Find the target by its bumper number."
)


def served(run: RunDir) -> list[FrameSheet]:
    """Every sheet the frame tool served for this run, from its log."""
    log = frame_log(run)
    if not log.exists():
        return []
    lines = log.read_text().splitlines()
    return [FrameSheet.model_validate_json(line) for line in lines if line.strip()]


class FrameRequest(BaseModel):
    """Arguments of the `frames` tool."""

    model_config = ConfigDict(extra="forbid")

    t: float = Field(ge=0.0, description="Video seconds of the first tile.")
    count: int = Field(
        default=4, ge=1, le=4, description="Tiles, one per moment (corner views: at most 2)."
    )
    step_s: float = Field(
        default=1.0, ge=0.2, le=5.0, description="Seconds between consecutive tiles."
    )
    view: View = Field(
        default="target",
        description="'target': crop around the target robot. 'full': the whole main camera. "
        "'corner_left' / 'corner_right': a corner camera panel, uncalibrated (no ring, no "
        "boxes), to find a target hidden from the main camera and read its bumper.",
    )
    zoom: Literal[1, 2] = Field(
        default=1,
        description="target view: 2 crops half as wide around the target, to read bumpers and "
        "the hopper.",
    )


@dataclass(frozen=True)
class _Spot:
    """Where to look for the target in one frame."""

    center: tuple[float, float] | None
    """Full-frame pixel to centre the crop on; None when nothing is known nearby."""
    box: tuple[float, float, float, float] | None
    """Box to ring; None when the target is not tracked here."""
    measured: bool
    status: str
    xy: tuple[float, float] | None
    """Field position when tracked at this moment."""


def _spot(target: Target, cam: Camera, t: float) -> _Spot:
    """Look up the target at video time `t`: its box if measured, its projected footprint if
    interpolated, else the nearest known position within NEARBY_S for centring only."""
    if len(target.t) == 0:
        return _Spot(None, None, False, "target never tracked", None)
    i = int(np.argmin(np.abs(target.t - t)))
    dt = float(target.t[i] - t)
    half = ROBOT_FOOTPRINT_M / 2
    x, y = float(target.xy[i][0]), float(target.xy[i][1])
    foot = cam.project(
        np.array([[x + dx, y + dy, 0.0] for dx in (-half, half) for dy in (-half, half)])
    )
    fbox = (
        (
            float(foot[:, 0].min()),
            float(foot[:, 1].min()),
            float(foot[:, 0].max()),
            float(foot[:, 1].max()),
        )
        if np.isfinite(foot).all()
        else None
    )
    if abs(dt) <= TRACKED_WITHIN_S:
        if np.isfinite(target.box[i]).all():
            b = tuple(float(v) for v in target.box[i])
            box = (b[0], b[1], b[2], b[3])
            c = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            return _Spot(c, box, True, "tracked", (x, y))
        if fbox is not None:
            c = ((fbox[0] + fbox[2]) / 2, (fbox[1] + fbox[3]) / 2)
            return _Spot(c, fbox, False, "tracked (estimated box)", (x, y))
    if abs(dt) <= NEARBY_S and fbox is not None:
        when = "next seen" if dt > 0 else "last seen"
        c = ((fbox[0] + fbox[2]) / 2, (fbox[1] + fbox[3]) / 2)
        return _Spot(c, None, False, f"NOT tracked ({when} {abs(dt):.1f}s away)", None)
    return _Spot(None, None, False, "NOT tracked nearby", None)


@dataclass(frozen=True)
class _Mark:
    """Another tracked robot on a tile."""

    team: int | None
    relation: Literal["partner", "opponent", "unknown"]
    box: tuple[float, float, float, float]
    dist_m: float | None

    @property
    def label(self) -> str:
        if self.team is None:
            return "unlabelled"
        return f"{self.team} {'PARTNER' if self.relation == 'partner' else 'OPP'}"


def _marks(others: Others, match: MatchInfo, spot: _Spot, t: float, same_m: float) -> list[_Mark]:
    """Other robots with a measured main-view box within TRACKED_WITHIN_S of `t`, nearest
    sample per track. Unlabelled tracks on the target's own spot are the target itself."""
    idx = np.flatnonzero(
        (np.abs(others.t - t) <= TRACKED_WITHIN_S) & np.isfinite(others.box).all(axis=1)
    )
    best: dict[int, int] = {}
    for i in idx.tolist():
        k = int(others.track[i])
        if k not in best or abs(others.t[i] - t) < abs(others.t[best[k]] - t):
            best[k] = i
    out: list[_Mark] = []
    for i in best.values():
        team = int(others.team[i])
        ox, oy = float(others.xy[i][0]), float(others.xy[i][1])
        dist = None if spot.xy is None else math.hypot(ox - spot.xy[0], oy - spot.xy[1])
        if team < 0 and dist is not None and dist < same_m:
            continue
        b = others.box[i]
        out.append(
            _Mark(
                team=None if team < 0 else team,
                relation="unknown" if team < 0 else relation(match, team),
                box=(float(b[0]), float(b[1]), float(b[2]), float(b[3])),
                dist_m=dist,
            )
        )
    return sorted(out, key=lambda m: m.dist_m if m.dist_m is not None else 99.0)


def _tile(
    image: Image,
    view: Literal["target", "full"],
    zoom: int,
    spot: _Spot,
    marks: list[_Mark],
    panel: tuple[int, int, int, int],
) -> tuple[Image, list[_Mark]]:
    """Crop (or shrink) one main-view frame, box the other tracked robots and ring the
    target; returns the tile and the marks drawn on it. `panel` is the main view x, y, w, h in
    full-frame pixels; `image` is that panel."""
    px, py, pw, ph = panel
    if view == "target" and spot.center is not None:
        cw, ch = min(CROP_W // zoom, pw), min(CROP_H // zoom, ph)
        x0 = int(np.clip(spot.center[0] - px - cw / 2, 0, pw - cw))
        y0 = int(np.clip(spot.center[1] - py - ch / 2, 0, ph - ch))
        out_w = TILE_W
    else:
        x0, y0, cw, ch = 0, 0, pw, ph
        out_w = FULL_W if view == "full" else TILE_W
    crop = image[y0 : y0 + ch, x0 : x0 + cw]
    scale = out_w / cw
    size = (out_w, round(ch * scale))
    tile = np.asarray(cv2.resize(crop, size, interpolation=cv2.INTER_CUBIC), np.uint8)

    def to_tile(u: float, v: float) -> tuple[int, int]:
        return round((u - px - x0) * scale), round((v - py - y0) * scale)

    drawn: list[_Mark] = []
    for m in marks:
        (ax, ay), (bx, by) = to_tile(m.box[0], m.box[1]), to_tile(m.box[2], m.box[3])
        if not (0 <= (ax + bx) / 2 <= size[0] and 0 <= (ay + by) / 2 <= size[1]):
            continue
        drawn.append(m)
        color = MARK[m.relation]
        cv2.rectangle(tile, (ax, ay), (bx, by), color, 2)
        (tw, th), _ = cv2.getTextSize(m.label, FONT, 0.5, 1)
        ty = max(th + 4, ay - 3)
        cv2.rectangle(tile, (ax, ty - th - 4), (ax + tw + 4, ty + 2), (0, 0, 0), -1)
        cv2.putText(tile, m.label, (ax + 2, ty - 1), FONT, 0.5, color, 1, cv2.LINE_AA)
    if spot.box is not None:
        bx1, by1, bx2, by2 = spot.box
        cx, cy = to_tile((bx1 + bx2) / 2, (by1 + by2) / 2)
        ax_ = max(14.0, (bx2 - bx1) / 2 * scale + 10)
        ay_ = max(14.0, (by2 - by1) / 2 * scale + 10)
        color = RING if spot.measured else RING_EST
        # lower half only: a whole ring crossed the hopper above the bumper in fuel's colour
        cv2.ellipse(tile, (cx, cy), (round(ax_), round(ay_)), 0, 0, 180, color, 3)
    return tile, drawn


def _bar(width: int, height: int, text: str, scale: float = 0.55) -> Image:
    bar = np.zeros((height, width, 3), np.uint8)
    cv2.putText(bar, text, (6, height - 8), FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return bar


def _sheet(tiles: list[Image], view: View) -> Image:
    """Tiles in a 2-wide grid for crops, one column for full and corner views, with the
    legend."""
    if view != "target" or len(tiles) == 1:
        grid = np.vstack(tiles)
    else:
        h, w = tiles[0].shape[:2]
        while len(tiles) % 2:
            tiles = [*tiles, np.zeros((h, w, 3), np.uint8)]
        grid = np.vstack([np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)])
    legend = CORNER_LEGEND if view.startswith("corner") else LEGEND
    return np.vstack([grid, _bar(grid.shape[1], LEGEND_H, legend, 0.5)])


FACT_KINDS_LISTED = frozenset(
    {"gap", "crossing", "crossing_in_gap", "crossing_turned_back", "stop",
     "left_own_zone_while_active", "contact", "hub_on"}
)  # fmt: skip


class _FrameServer:
    """Answers `frames` calls for one run folder, one sheet at a time."""

    def __init__(self, run: RunDir) -> None:
        self.run = run
        self.info = VideoInfo.load(run.video_json)
        self.match = MatchInfo.load(run.match_json)
        self.cam = Camera.load(run.calibration_json, "main")
        self.target, self.others = load_tracks(run, self.match)
        self.facts = Facts.load(run.facts_json)
        self.same_m = self.facts.thresholds.get("SAME_ROBOT_M", 0.7)
        self.home = home_view(self.cam, self.match)
        m = self.match.views["main"]
        self.panel = (m.x, m.y, m.w, m.h)

    def call(self, req: FrameRequest) -> tuple[str, Image | None]:
        """Render one contact sheet; returns the text for the agent and the image, or a
        refusal and None."""
        done = len(served(self.run))
        if done >= IMAGE_CAP:
            return f"Refused: the image budget of {IMAGE_CAP} sheets is used up.", None
        corner = req.view in ("corner_left", "corner_right")
        if corner and req.count > CORNER_MAX_TILES:
            return f"Refused: corner views show at most {CORNER_MAX_TILES} tiles per sheet.", None
        fps = self.info.fps
        last = (self.info.frame_count - 1) / fps
        start, step = round(req.t, 1), round(req.step_s, 1)
        times = [round(start + k * step, 1) for k in range(req.count)]
        times = [t for t in times if 0.0 <= t <= last]
        if not times:
            return f"Refused: t must be between 0 and {last:.1f} video seconds.", None
        tiles: list[Image] = []
        lines: list[str] = []
        records: list[TileRecord] = []
        for t in times:
            period, clock = _clock(self.match, t)
            phase = _phase(self.match, t)
            if req.view == "corner_left" or req.view == "corner_right":
                frame = frame_at(
                    self.run.video, self.info, t + 0.5 / fps, crop=self.match.views[req.view]
                )
                h, w = frame.image.shape[:2]
                size = (w * CORNER_SCALE, h * CORNER_SCALE)
                image = np.asarray(
                    cv2.resize(frame.image, size, interpolation=cv2.INTER_CUBIC), np.uint8
                )
                label = f"t={t:.1f}s  {clock} {period} {phase}  {req.view} (uncalibrated)"
                tiles.append(np.vstack([_bar(image.shape[1], HEADER_H, label), image]))
                lines.append(f"- frame:{t:.1f}  clock {clock} ({period}, {phase}), {req.view}")
                records.append(
                    TileRecord(
                        t=t, tracked=False, teams_marked=[], teams_near=[], unlabelled_marked=0
                    )
                )
                continue
            frame = frame_at(
                self.run.video, self.info, t + 0.5 / fps, crop=self.match.views["main"]
            )
            spot = _spot(self.target, self.cam, t)
            marks = _marks(self.others, self.match, spot, t, self.same_m)
            label = f"t={t:.1f}s  {clock} {period} {phase}  target {spot.status}"
            image, marks = _tile(frame.image, req.view, req.zoom, spot, marks, self.panel)
            tiles.append(np.vstack([_bar(image.shape[1], HEADER_H, label), image]))
            boxed = ", ".join(
                m.label.replace("OPP", "opponent").replace("PARTNER", "partner")
                + (f" {m.dist_m:.1f} m" if m.dist_m is not None else "")
                for m in marks
            )
            lines.append(
                f"- frame:{t:.1f}  clock {clock} ({period}, {phase}), {spot.status}; "
                f"boxed: {boxed or 'none'}"
                + ("; an unlabelled box here may be the target itself" if spot.xy is None else "")
            )
            records.append(
                TileRecord(
                    t=t,
                    tracked=spot.box is not None,
                    teams_marked=sorted({m.team for m in marks if m.team is not None}),
                    teams_near=sorted(
                        {
                            m.team
                            for m in marks
                            if m.team is not None
                            and m.dist_m is not None
                            and m.dist_m <= NEAR_BOX_M
                        }
                    ),
                    unlabelled_marked=sum(m.team is None for m in marks),
                )
            )
        n = done + 1
        rel = f"grade_work/sheet-{n:02d}.jpg"
        sheet = _sheet(tiles, req.view)
        cv2.imwrite(str(self.run.path / rel), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        record = FrameSheet(image=rel, view=req.view, zoom=req.zoom, tiles=records)
        with frame_log(self.run).open("a") as f:
            f.write(record.model_dump_json() + "\n")
        near = [
            f"{f.id} ({f.t_start:.1f}-{f.t_end:.1f})"
            for f in self.facts.facts
            if f.kind in FACT_KINDS_LISTED
            and overlap(f.t_start, f.t_end, times[0] - 0.5, times[-1] + 0.5)
        ]
        end = "the target's own end" if req.view == self.home else "the opponents' end"
        how = (
            f"{req.view} is a corner camera showing {end}. "
            "It is not calibrated: the tool cannot ring the target or box robots there. Name the "
            "target only where you read its bumper number, and give that number as "
            "corner_bumper in any call that cites these tiles."
            if corner
            else "Yellow arc under a robot: the target's measured box; orange arc: its position "
            "estimated "
            "between detections. Boxes are the other robots the tool tracked: team and "
            "PARTNER/OPP from the roster, or 'unlabelled' when the tool does not know the team. "
            "A robot with no box was NOT tracked: no fact knows about it."
        )
        text = (
            f"Sheet {n} of at most {IMAGE_CAP} ({rel}). Tiles, citable as written:\n"
            + "\n".join(lines)
            + f"\nFacts overlapping these tiles: {', '.join(near) or 'none'}.\n"
            + how
        )
        return text, sheet


def serve_frames(run_path: Path) -> None:
    """Run the `frames` tool as a stdio MCP server for `run_path` (newline-delimited
    JSON-RPC). The grade stage has Claude Code start it as `python -m scout.stages.grade
    <run>`, which imports only this stage rather than the whole CLI (torch included)."""
    server = _FrameServer(RunDir(run_path.resolve()))
    tool = {
        "name": TOOL,
        "description": (
            "Show the main broadcast camera at up to four consecutive moments, cropped around "
            "the target robot (ringed) or the whole view, labelled with video time and match "
            "clock; other tracked robots are boxed with their team and partner/opponent, or "
            "'unlabelled'. Or show a corner camera panel (unmarked) at up to two moments. "
            f"At most {IMAGE_CAP} sheets per grade."
        ),
        "inputSchema": FrameRequest.model_json_schema(),
    }
    adapter = TypeAdapter(dict[str, object])
    for line in sys.stdin:
        if not line.strip():
            continue
        msg = adapter.validate_json(line)
        method, mid = msg.get("method"), msg.get("id")
        if mid is None:
            continue
        result: JSON
        if method == "initialize":
            params = msg.get("params")
            version = params.get("protocolVersion") if isinstance(params, dict) else None
            result = {
                "protocolVersion": version or "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": SERVER, "version": "1"},
            }
        elif method == "tools/list":
            result = {"tools": [tool]}
        elif method == "tools/call":
            result = _tool_call(server, msg.get("params"))
        elif method == "ping":
            result = {}
        else:
            err = {"code": -32601, "message": f"unknown method {method}"}
            _send({"jsonrpc": "2.0", "id": mid, "error": err})
            continue
        _send({"jsonrpc": "2.0", "id": mid, "result": result})


def _tool_call(server: _FrameServer, params: object) -> JSON:
    args = params.get("arguments") if isinstance(params, dict) else None
    try:
        req = FrameRequest.model_validate(args or {})
        text, image = server.call(req)
    except (ValidationError, RuntimeError, ValueError) as e:
        return {"content": [{"type": "text", "text": f"Error: {e}"}], "isError": True}
    content: list[JSON] = [{"type": "text", "text": text}]
    if image is None:
        return {"content": content, "isError": True}
    ok, jpg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if ok:
        data = base64.b64encode(jpg.tobytes()).decode()
        content.append({"type": "image", "data": data, "mimeType": "image/jpeg"})
    return {"content": content}


def _send(msg: JSON) -> None:
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()
