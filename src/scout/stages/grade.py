"""grade: reads facts.json, match.json, tracks.parquet, identity.json, calibration.json, the
video and settings.json.

Writes grade.json (see `scout.schemas.grade`): 1-10 scores for three driving skills with
cited evidence, timestamped notes and an unscored auto report, written by Claude Code run
headless (`claude -p`). Also writes grade.md, the same grade as a short readable report, and
`grade_work/` with the prompts, every contact sheet served, the frame log and each session's
stream.

Steps:

1. Build the prompt from files: `grade_system.md` and `grade_task.md` beside this module
   (game-independent), and the game guide, limits and rubric under
   `scout/games/rebuilt2026_grade/`. The task carries the match timeline and all of facts.json.
2. Run `claude -p` sealed: our system prompt replaces the default one, no built-in tools, no
   settings files, skills or other MCP servers, and one MCP tool, `frames`, served by
   `python -m scout.stages.grade <run>` (`serve_frames` below). The tool crops the main camera
   around the target, rings it, tiles up to four moments with the match clock, logs every sheet to
   `grade_work/frames.jsonl` and refuses past IMAGE_CAP sheets. The session's init message is
   checked so a leak fails the stage instead of tainting the grade.
3. Check the structured answer: schema, every citation against the fact ids and the frame
   log, note times against what they cite, and frame evidence for defense. On failure the
   session is resumed once with the exact errors; a second failure fails the stage.
"""

from __future__ import annotations

import base64
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import polars as pl
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from scout.camera import Camera
from scout.games.rebuilt2026 import ROBOT_FOOTPRINT_M
from scout.run import RunDir, Stage
from scout.schemas.facts import Facts
from scout.schemas.grade import (
    SKILLS,
    FrameSheet,
    Grade,
    GradeAnswer,
    Note,
    RunRecord,
)
from scout.schemas.identity import Identity
from scout.schemas.match import MatchInfo
from scout.schemas.tracks import read_tracks
from scout.schemas.video import VideoInfo
from scout.video import frame_at

F64 = NDArray[np.float64]
Image = NDArray[np.uint8]
JSON = dict[str, object]

GAME_DIR = "rebuilt2026_grade"
"""Folder under `scout.games` with the game guide, limits and rubric for this game."""
IMAGE_CAP = 20
EFFORT = "medium"
"""Claude Code effort level. Medium keeps a grade near two minutes on Opus."""
MAX_BUDGET_USD = 8.0
"""Stops a runaway session; one grade lists at about 1 to 3 USD."""
TIMEOUT_S = 600.0
TOOL = "frames"
SERVER = "scout"
TOOL_ID = f"mcp__{SERVER}__{TOOL}"
SESSION_TOOLS = (TOOL_ID, "StructuredOutput")
"""The only tools a grader session may load: ours, and the one `--json-schema` adds."""
NOTE_REACH_S = 5.0
"""A note's `t` must be within this of something it cites."""
FRAME_MATCH_S = 0.051

TRACKED_WITHIN_S = 0.15
"""A target sample this close in time counts as the target's position at a tile."""
NEARBY_S = 5.0
"""When untracked, the crop centres on the target's position within this much time."""
CROP_W, CROP_H = 800, 450
TILE_W = 640
FULL_W = 960
HEADER_H = 26
RING: tuple[int, int, int] = (0, 225, 255)
RING_EST: tuple[int, int, int] = (0, 140, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX

ENV_KEEP = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CONFIG_DIR",
)
"""Claude and Anthropic variables the child keeps: credentials and where they live. Every
other `CLAUDE*`, `ANTHROPIC_*` and agent marker from a parent session is dropped."""
ENV_DROP_PREFIXES = ("CLAUDE", "ANTHROPIC_", "AI_AGENT", "CODEX_", "MCP_")


def _work(run: RunDir) -> Path:
    return run.path / "grade_work"


def _grade_md(run: RunDir) -> Path:
    return run.path / "grade.md"


def _frame_log(run: RunDir) -> Path:
    return _work(run) / "frames.jsonl"


# --- match clock ------------------------------------------------------------------------


def _clock(match: MatchInfo, t: float) -> tuple[str, str]:
    """Period and scoreboard clock at video time `t`, as facts.json writes them."""
    if t < match.auto_start_t:
        return "pre_match", "0:20"
    if t < match.auto_end_t:
        fit, length, period = match.clock_fits["auto"], 20.0, "auto"
    elif t < match.teleop_start_t:
        return "pause", "2:20"
    elif t <= match.teleop_end_t:
        fit, length, period = match.clock_fits["teleop"], 140.0, "teleop"
    else:
        return "post_match", "0:00"
    s = max(0, math.ceil(length - (t - fit.start_t) / fit.slope - 1e-6))
    return period, f"{s // 60}:{s % 60:02d}"


def _phase(match: MatchInfo, t: float) -> str:
    for p in match.phases:
        if p.t_start <= t < p.t_end:
            return p.name
    return "-"


# --- the frame tool ---------------------------------------------------------------------


def _served(run: RunDir) -> list[FrameSheet]:
    """Every sheet the frame tool served for this run, from its log."""
    log = _frame_log(run)
    if not log.exists():
        return []
    lines = log.read_text().splitlines()
    return [FrameSheet.model_validate_json(line) for line in lines if line.strip()]


class FrameRequest(BaseModel):
    """Arguments of the `frames` tool."""

    model_config = ConfigDict(extra="forbid")

    t: float = Field(ge=0.0, description="Video seconds of the first tile.")
    count: int = Field(default=4, ge=1, le=4, description="Tiles, one per moment.")
    step_s: float = Field(
        default=1.0, ge=0.2, le=5.0, description="Seconds between consecutive tiles."
    )
    view: Literal["target", "full"] = Field(
        default="target",
        description="'target': crop around the target robot. 'full': the whole main camera.",
    )


@dataclass(frozen=True)
class _Target:
    """The target team's samples (what facts.json calls its path), for looking it up at
    any moment."""

    t: F64
    xy: F64
    box: F64
    """(n, 4) main-view box in full-frame pixels; NaN when interpolated or from a corner view."""


def _load_target(run: RunDir, match: MatchInfo) -> _Target:
    """Track samples inside identity segments labelled with the target, minus frames two
    tracks both claim."""
    ident = Identity.load(run.identity_json)
    target = match.target_team
    segs = [s for s in ident.segments if s.team == target]
    tracks = read_tracks(run.tracks)
    parts = [
        tracks.filter(
            (pl.col("track_id") == s.track_id)
            & pl.col("frame").is_between(s.frame_start, s.frame_end)
        )
        for s in segs
    ]
    rows = pl.concat(parts) if parts else tracks.clear()
    rows = rows.unique(["track_id", "frame"]).filter(pl.col("frame").is_unique()).sort("frame")
    main = (rows["state"] == "measured") & (rows["view"] == "main")
    box = rows.select("x1", "y1", "x2", "y2").to_numpy().astype(np.float64)
    box[~main.to_numpy()] = np.nan
    return _Target(
        t=rows["t"].to_numpy().astype(np.float64),
        xy=rows.select("x", "y").to_numpy().astype(np.float64),
        box=box,
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


def _spot(target: _Target, cam: Camera, t: float) -> _Spot:
    """Look up the target at video time `t`: its box if measured, its projected footprint if
    interpolated, else the nearest known position within NEARBY_S for centring only."""
    if len(target.t) == 0:
        return _Spot(None, None, False, "target never tracked")
    i = int(np.argmin(np.abs(target.t - t)))
    dt = float(target.t[i] - t)
    half = ROBOT_FOOTPRINT_M / 2
    x, y = target.xy[i]
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
            return _Spot(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), box, True, "tracked")
        if fbox is not None:
            c = ((fbox[0] + fbox[2]) / 2, (fbox[1] + fbox[3]) / 2)
            return _Spot(c, fbox, False, "tracked (estimated box)")
    if abs(dt) <= NEARBY_S and fbox is not None:
        when = "next seen" if dt > 0 else "last seen"
        c = ((fbox[0] + fbox[2]) / 2, (fbox[1] + fbox[3]) / 2)
        return _Spot(c, None, False, f"NOT tracked ({when} {abs(dt):.1f}s away, centred there)")
    return _Spot(None, None, False, "NOT tracked nearby")


def _tile(
    image: Image, view: Literal["target", "full"], spot: _Spot, panel: tuple[int, int, int, int]
) -> Image:
    """Crop (or shrink) one main-view frame and ring the target. `panel` is the main view
    x, y, w, h in full-frame pixels; `image` is that panel."""
    px, py, pw, ph = panel
    if view == "target" and spot.center is not None:
        cw, ch = min(CROP_W, pw), min(CROP_H, ph)
        x0 = int(np.clip(spot.center[0] - px - cw / 2, 0, pw - cw))
        y0 = int(np.clip(spot.center[1] - py - ch / 2, 0, ph - ch))
        out_w = TILE_W
    else:
        x0, y0, cw, ch = 0, 0, pw, ph
        out_w = FULL_W if view == "full" else TILE_W
    crop = image[y0 : y0 + ch, x0 : x0 + cw]
    scale = out_w / cw
    size = (out_w, round(ch * scale))
    tile = np.asarray(cv2.resize(crop, size, interpolation=cv2.INTER_AREA), np.uint8)
    if spot.box is not None:
        bx1, by1, bx2, by2 = spot.box
        cx = ((bx1 + bx2) / 2 - px - x0) * scale
        cy = ((by1 + by2) / 2 - py - y0) * scale
        ax = max(14.0, (bx2 - bx1) / 2 * scale + 10)
        ay = max(14.0, (by2 - by1) / 2 * scale + 10)
        color = RING if spot.measured else RING_EST
        cv2.ellipse(tile, (round(cx), round(cy)), (round(ax), round(ay)), 0, 0, 360, color, 3)
    return tile


def _with_header(tile: Image, text: str) -> Image:
    head = np.zeros((HEADER_H, tile.shape[1], 3), np.uint8)
    cv2.putText(head, text, (6, HEADER_H - 8), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([head, tile])


def _sheet(tiles: list[Image], view: Literal["target", "full"]) -> Image:
    """Tiles in a 2-wide grid for crops, one column for full views."""
    if view == "full" or len(tiles) == 1:
        return np.vstack(tiles)
    h, w = tiles[0].shape[:2]
    while len(tiles) % 2:
        tiles = [*tiles, np.zeros((h, w, 3), np.uint8)]
    rows = [np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)]
    return np.vstack(rows)


class _FrameServer:
    """Answers `frames` calls for one run folder, one sheet at a time."""

    def __init__(self, run: RunDir) -> None:
        self.run = run
        self.info = VideoInfo.load(run.video_json)
        self.match = MatchInfo.load(run.match_json)
        self.cam = Camera.load(run.calibration_json, "main")
        self.target = _load_target(run, self.match)
        m = self.match.views["main"]
        self.panel = (m.x, m.y, m.w, m.h)

    def call(self, req: FrameRequest) -> tuple[str, Image | None]:
        """Render one contact sheet; returns the text for the agent and the image, or a
        refusal and None."""
        done = len(_served(self.run))
        if done >= IMAGE_CAP:
            return f"Refused: the image budget of {IMAGE_CAP} sheets is used up.", None
        fps = self.info.fps
        last = (self.info.frame_count - 1) / fps
        start, step = round(req.t, 1), round(req.step_s, 1)
        times = [round(start + k * step, 1) for k in range(req.count)]
        times = [t for t in times if 0.0 <= t <= last]
        if not times:
            return f"Refused: t must be between 0 and {last:.1f} video seconds.", None
        main = self.match.views["main"]
        tiles: list[Image] = []
        lines: list[str] = []
        for t in times:
            frame = frame_at(self.run.video, self.info, t + 0.5 / fps, crop=main)
            spot = _spot(self.target, self.cam, t)
            period, clock = _clock(self.match, t)
            phase = _phase(self.match, t)
            label = f"t={t:.1f}s  {clock} {period} {phase}  target {spot.status}"
            tiles.append(_with_header(_tile(frame.image, req.view, spot, self.panel), label))
            lines.append(f"- frame:{t:.1f}  clock {clock} ({period}, {phase}), {spot.status}")
        n = done + 1
        rel = f"grade_work/sheet-{n:02d}.jpg"
        sheet = _sheet(tiles, req.view)
        cv2.imwrite(str(self.run.path / rel), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        record = FrameSheet(image=rel, view=req.view, times=times)
        with _frame_log(self.run).open("a") as f:
            f.write(record.model_dump_json() + "\n")
        text = (
            f"Sheet {n} of at most {IMAGE_CAP} ({rel}). Tiles, citable as written:\n"
            + "\n".join(lines)
            + "\nYellow ring: the target's measured box. Orange ring: its position estimated "
            "between detections."
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
            f"clock. At most {IMAGE_CAP} sheets per grade."
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


# --- prompt -----------------------------------------------------------------------------


def _text(name: str) -> str:
    """A prompt file: `grade_*.md` beside this module, or a game file under GAME_DIR."""
    if name.startswith("grade_"):
        return resources.files("scout.stages").joinpath(name).read_text()
    return resources.files("scout.games").joinpath(GAME_DIR, name).read_text()


def _rubric_version(rubric: str) -> str:
    first = rubric.splitlines()[0]
    key, _, value = first.partition(":")
    if key.strip() != "Rubric version" or not value.strip():
        raise ValueError(f"rubric.md must start with 'Rubric version: <name>', got {first!r}")
    return value.strip()


def _prompts(match: MatchInfo, facts_json: str) -> tuple[str, str]:
    """System prompt (rules, game, limits, rubric) and task (match and facts)."""
    system = "\n\n".join(
        [
            _text("grade_system.md").replace("{{image_cap}}", str(IMAGE_CAP)),
            _text("game.md"),
            _text("limits.md"),
            _text("rubric.md"),
        ]
    )
    roster = ", ".join(f"{s.team} ({s.alliance} {s.station})" for s in match.roster)
    phases = "\n".join(
        f"- {p.name}: {p.t_start:.1f}-{p.t_end:.1f} s, active: "
        f"{', '.join(p.active_hubs) if p.active_hubs is not None else 'unknown'}"
        for p in match.phases
    )
    own = [
        p.name
        for p in match.phases
        if p.active_hubs is not None and match.target_alliance in p.active_hubs
    ]
    compact = json.dumps(json.loads(facts_json), separators=(",", ":"))
    fill = {
        "team": str(match.target_team),
        "alliance": match.target_alliance,
        "roster": roster,
        "auto_winner": match.auto_winner or "unknown",
        "phases": phases,
        "own_active": ", ".join(own),
        "facts": compact,
    }
    task = _text("grade_task.md")
    for key, value in fill.items():
        task = task.replace("{{" + key + "}}", value)
    return system, task


# --- claude -----------------------------------------------------------------------------


def _child_env() -> dict[str, str]:
    """This process's environment without the variables a parent Claude Code session
    leaks (session ids, entrypoint, effort, messaging sockets), keeping credentials."""
    return {
        k: v for k, v in os.environ.items() if k in ENV_KEEP or not k.startswith(ENV_DROP_PREFIXES)
    }


def _claude() -> str:
    """Path to `claude`, checked to be installed and logged in."""
    exe = shutil.which("claude")
    if exe is None:
        raise RuntimeError(
            "grade needs Claude Code: `claude` is not on PATH. Install it "
            "(https://docs.claude.com/claude-code) and run `claude` once to log in."
        )
    out = subprocess.run(
        [exe, "auth", "status"], capture_output=True, text=True, env=_child_env(), timeout=60
    )
    try:
        status = TypeAdapter(dict[str, object]).validate_json(out.stdout)
    except ValidationError:
        status = {}
    if status.get("loggedIn") is not True:
        raise RuntimeError(
            "grade needs Claude Code logged in: `claude auth status` says it is not. "
            "Run `claude` and log in (or set ANTHROPIC_API_KEY), then rerun."
        )
    return exe


def _command(exe: str, model: str, run: RunDir, system: str, resume: str | None) -> list[str]:
    """The sealed `claude -p` command line: our prompt and our one tool, no built-in tools,
    settings files, skills or other MCP servers. `resume` continues a session for the retry."""
    mcp = {
        "mcpServers": {
            SERVER: {
                "command": sys.executable,
                "args": ["-m", "scout.stages.grade", str(run.path.resolve())],
            }
        }
    }
    cmd = [
        exe, "-p",
        "--model", model,
        "--effort", EFFORT,
        "--output-format", "stream-json", "--verbose",
        "--json-schema", json.dumps(GradeAnswer.model_json_schema()),
        "--system-prompt", system,
        "--tools", "",
        "--mcp-config", json.dumps(mcp), "--strict-mcp-config",
        "--allowedTools", TOOL_ID,
        "--permission-mode", "dontAsk",
        "--setting-sources", "",
        "--disable-slash-commands",
        "--max-budget-usd", str(MAX_BUDGET_USD),
    ]  # fmt: skip
    if resume is not None:
        cmd += ["--resume", resume]
    return cmd


@dataclass(frozen=True)
class _Session:
    """What one `claude -p` call returned."""

    session_id: str
    answer: object
    result: JSON
    init: JSON


def _as_dict(v: object) -> JSON:
    return TypeAdapter(dict[str, object]).validate_python(v) if isinstance(v, dict) else {}


def _run_claude(cmd: list[str], prompt: str, cwd: Path, log: Path) -> _Session:
    """Run one session, save its stream to `log`, check the sealing, return the result."""
    with log.open("w") as out:
        proc = subprocess.run(
            cmd,
            input=prompt,
            stdout=out,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            env=_child_env(),
            timeout=TIMEOUT_S,
        )
    adapter = TypeAdapter(dict[str, object])
    init: JSON = {}
    result: JSON = {}
    for line in log.read_text().splitlines():
        if not line.startswith("{"):
            continue
        msg = adapter.validate_json(line)
        if msg.get("type") == "system" and msg.get("subtype") == "init":
            init = msg
        elif msg.get("type") == "result":
            result = msg
    if not result:
        raise RuntimeError(
            f"claude exited {proc.returncode} without a result; stderr: {proc.stderr[-2000:]}"
        )
    _check_sealed(init)
    if result.get("is_error") or result.get("subtype") != "success":
        raise RuntimeError(
            f"claude session failed ({result.get('subtype')}): {str(result.get('result'))[:500]}"
        )
    return _Session(str(result.get("session_id")), result.get("structured_output"), result, init)


def _check_sealed(init: JSON) -> None:
    """Fail unless the session loaded exactly SESSION_TOOLS, our MCP server and no skills,
    slash commands or user plugins."""
    tools = init.get("tools")
    servers = init.get("mcp_servers")
    plugins = init.get("plugins")
    problems: list[str] = []
    if not isinstance(tools, list) or sorted(map(str, tools)) != sorted(SESSION_TOOLS):
        problems.append(f"tools {tools}")
    if not isinstance(servers, list) or [_as_dict(s).get("name") for s in servers] != [SERVER]:
        problems.append(f"mcp servers {servers}")
    elif _as_dict(servers[0]).get("status") != "connected":
        problems.append(f"frame tool not connected: {servers}")
    if init.get("skills") or init.get("slash_commands"):
        problems.append(f"skills {init.get('skills')} commands {init.get('slash_commands')}")
    if isinstance(plugins, list) and any(_as_dict(p).get("path") != "builtin" for p in plugins):
        problems.append(f"plugins {plugins}")
    if problems:
        raise RuntimeError("grader session is not sealed: " + "; ".join(problems))


# --- checking the answer ----------------------------------------------------------------


def check_answer(
    raw: object, facts: Facts, sheets: list[FrameSheet]
) -> tuple[GradeAnswer | None, list[str]]:
    """Validate the agent's answer against the schema, the fact ids and the frames the tool
    served. Returns the answer when it passes, and every problem found."""
    try:
        answer = GradeAnswer.model_validate(raw)
    except ValidationError as e:
        problems = [
            f"schema: {'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()
        ]
        return None, problems
    spans = {f.id: (f.t_start, f.t_end) for f in facts.facts}
    served = sorted({t for s in sheets for t in s.times})
    problems: list[str] = []

    def where(cite: str, at: str) -> tuple[float, float] | None:
        if cite.startswith("frame:"):
            try:
                t = float(cite.removeprefix("frame:"))
            except ValueError:
                problems.append(f"{at}: {cite!r} is not frame:<video seconds>")
                return None
            if not any(abs(t - s) <= FRAME_MATCH_S for s in served):
                problems.append(f"{at}: {cite} was never served by the frame tool")
                return None
            return (t, t)
        if cite not in spans:
            problems.append(f"{at}: {cite!r} is not a fact id in facts.json")
            return None
        return spans[cite]

    names = [g.skill for g in answer.skills]
    if sorted(names) != sorted(SKILLS):
        problems.append(f"skills: need exactly one each of {list(SKILLS)}, got {names}")
    for g in answer.skills:
        for c in g.cites:
            where(c, f"skill {g.skill}")
        if (
            g.skill == "handling_defense"
            and g.score is not None
            and not any(c.startswith("frame:") for c in g.cites)
        ):
            problems.append("skill handling_defense: a score needs at least one frame citation")
    for c in answer.auto.cites:
        where(c, "auto")
    for k, n in enumerate(answer.notes):
        at = f"note {k + 1} (t={n.t})"
        found = [s for c in n.cites if (s := where(c, at)) is not None]
        if found and not any(a - NOTE_REACH_S <= n.t <= b + NOTE_REACH_S for a, b in found):
            problems.append(f"{at}: t is more than {NOTE_REACH_S:g} s from everything it cites")
        if n.kind == "played_through_defense" and not any(c.startswith("frame:") for c in n.cites):
            problems.append(f"{at}: played_through_defense needs a frame citation")
    return (answer if not problems else None), problems


def _retry_prompt(problems: list[str]) -> str:
    return (
        "Your answer was rejected by the checker. Fix exactly these problems and return the "
        "whole answer again (you may open more frames within the budget):\n"
        + "\n".join(f"- {p}" for p in problems)
    )


# --- the stage --------------------------------------------------------------------------


def _usage(results: list[JSON]) -> tuple[int, int, int, int, float | None, int, list[str]]:
    """Summed input, output, cache read, cache write tokens, cost, turns and models."""
    ins = outs = reads = writes = turns = 0
    cost: float | None = None
    models: list[str] = []
    for r in results:
        u = _as_dict(r.get("usage"))
        ins += _int(u.get("input_tokens"))
        outs += _int(u.get("output_tokens"))
        reads += _int(u.get("cache_read_input_tokens"))
        writes += _int(u.get("cache_creation_input_tokens"))
        turns += _int(r.get("num_turns"))
        c = r.get("total_cost_usd")
        if isinstance(c, int | float):
            cost = (cost or 0.0) + float(c)
        models += [m for m in _as_dict(r.get("modelUsage")) if m not in models]
    return ins, outs, reads, writes, cost, turns, models


def _int(v: object) -> int:
    return int(v) if isinstance(v, int | float) else 0


def grade(run: RunDir) -> None:
    exe = _claude()
    model = run.settings().grader_model
    match = MatchInfo.load(run.match_json)
    facts_text = run.facts_json.read_text()
    facts = Facts.model_validate_json(facts_text)
    work = _work(run)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    system, task = _prompts(match, facts_text)
    (work / "system_prompt.md").write_text(system)
    (work / "task_prompt.md").write_text(task)
    start = time.perf_counter()

    first = _run_claude(
        _command(exe, model, run, system, None), task, run.path, work / "session-1.jsonl"
    )
    results = [first.result]
    sheets = _served(run)
    answer, problems = check_answer(first.answer, facts, sheets)
    rejection: list[str] | None = None
    if answer is None:
        rejection = problems
        (work / "rejection-1.json").write_text(
            json.dumps({"answer": first.answer, "problems": problems}, indent=2)
        )
        second = _run_claude(
            _command(exe, model, run, system, first.session_id),
            _retry_prompt(problems),
            run.path,
            work / "session-2.jsonl",
        )
        results.append(second.result)
        sheets = _served(run)
        answer, problems = check_answer(second.answer, facts, sheets)
        if answer is None:
            (work / "rejection-2.json").write_text(
                json.dumps({"answer": second.answer, "problems": problems}, indent=2)
            )
            raise RuntimeError(
                "grader answer rejected twice; no grade written:\n" + "\n".join(problems)
            )

    ins, outs, reads, writes, cost, turns, models = _usage(results)
    notes = []
    for n in sorted(answer.notes, key=lambda n: n.t):
        period, clock = _clock(match, n.t)
        notes.append(Note(**n.model_dump(), clock=clock, period=period))
    result = Grade(
        team=match.target_team,
        rubric_version=_rubric_version(_text("rubric.md")),
        skills=sorted(answer.skills, key=lambda g: SKILLS.index(g.skill)),
        notes=notes,
        auto=answer.auto,
        run=RunRecord(
            model=model,
            models_used=models,
            effort=EFFORT,
            claude_version=_str(first.init.get("claude_code_version")),
            wall_s=round(time.perf_counter() - start, 1),
            turns=turns,
            images_served=len(sheets),
            image_cap=IMAGE_CAP,
            sheets=sheets,
            input_tokens=ins,
            output_tokens=outs,
            cache_read_tokens=reads,
            cache_write_tokens=writes,
            cost_usd=None if cost is None else round(cost, 4),
            retried=rejection is not None,
            rejection=rejection,
        ),
    )
    result.save(run.grade_json)
    _grade_md(run).write_text(render_md(result, facts))


def _str(v: object) -> str | None:
    return v if isinstance(v, str) else None


# --- grade.md ---------------------------------------------------------------------------


def render_md(g: Grade, facts: Facts) -> str:
    """grade.json as a short report a person can read."""
    by_id = {f.id: f for f in facts.facts}

    def cites(cs: list[str]) -> str:
        out: list[str] = []
        for c in cs:
            f = by_id.get(c)
            out.append(f"`{c}` ({f.clock})" if f is not None else f"`{c}`")
        return ", ".join(out)

    lines = [
        f"# Driving grade: team {g.team}",
        "",
        f"Rubric `{g.rubric_version}`. Scores are absolute, 1 to 10. Auto is reported, not scored.",
        "",
        "| Skill | Score |",
        "|---|---|",
    ]
    lines += [f"| {s.skill} | {s.score if s.score is not None else 'null'} |" for s in g.skills]
    for s in g.skills:
        lines += ["", f"## {s.skill}: {s.score if s.score is not None else 'null'}", ""]
        if s.null_reason:
            lines += [f"No score: {s.null_reason}", ""]
        lines += [s.summary, "", f"Evidence: {cites(s.cites)}"]
    lines += [
        "",
        "## Notes",
        "",
        "| Clock | t (s) | Kind | Note | Cites |",
        "|---|---|---|---|---|",
    ]
    for n in g.notes:
        text = n.text.replace("|", "/")
        lines.append(f"| {n.clock} {n.period} | {n.t:.1f} | {n.kind} | {text} | {cites(n.cites)} |")
    r = g.run
    cost = f"{r.cost_usd:.2f} USD (list price)" if r.cost_usd is not None else "not reported"
    lines += [
        "",
        "## Auto (not scored)",
        "",
        g.auto.text,
        "",
        f"Evidence: {cites(g.auto.cites)}",
        "",
        "## Run",
        "",
        f"- Model: {r.model} (used: {', '.join(r.models_used) or 'unknown'}), effort {r.effort}, "
        f"Claude Code {r.claude_version or 'unknown'}",
        f"- Wall time: {r.wall_s:.0f} s over {r.turns} turns",
        f"- Contact sheets: {r.images_served} of {r.image_cap}: "
        + ", ".join(f"{s.image} ({s.view}, t {s.times[0]:.1f})" for s in r.sheets),
        f"- Input tokens: {r.input_tokens + r.cache_read_tokens + r.cache_write_tokens} "
        f"({r.cache_read_tokens} cache read, {r.cache_write_tokens} cache write, "
        f"{r.input_tokens} uncached); output tokens: {r.output_tokens}; cost {cost}",
        f"- Retry needed: {'yes' if r.retried else 'no'}",
    ]
    if r.rejection:
        lines += [f"  - {p}" for p in r.rejection]
    return "\n".join(lines) + "\n"


STAGE = Stage(
    "grade",
    inputs=lambda r: (
        r.facts_json,
        r.match_json,
        r.tracks,
        r.identity_json,
        r.calibration_json,
        r.video,
        r.settings_json,
    ),
    outputs=lambda r: (r.grade_json, _grade_md(r)),
    fn=grade,
)


if __name__ == "__main__":
    serve_frames(Path(sys.argv[1]))
