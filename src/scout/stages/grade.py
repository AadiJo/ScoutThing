"""grade: reads facts.json, match.json, tracks.parquet, identity.json, calibration.json, the
video and settings.json.

Writes grade.json (see `scout.schemas.grade`): 1-10 scores for three driving skills with
cited evidence, the ledgers and defense episodes they rest on, timestamped notes and an
unscored auto report, written by Claude Code run headless (`claude -p`). Also writes
grade.md, the same grade as a short readable report, and `grade_work/` with the prompts,
every contact sheet served, the frame log and each session's stream.

Steps:

1. Code builds the driving ledger (barrier lines and trips out of its own zone in its own
   active time), the readiness ledger and the list of defense items from the facts
   (`ledgers`), so the grader decides per line instead of doing arithmetic.
2. Build the prompt from files: `grade_system.md` and `grade_task.md` beside this module
   (game-independent), and the game guide, limits and rubric under
   `scout/games/rebuilt2026_grade/`. The task carries the match timeline, the ledgers, the
   defense items and all of facts.json.
3. Run `claude -p` sealed: our system prompt replaces the default one, no built-in tools, no
   settings files, skills or other MCP servers, and one MCP tool, `frames`, served by
   `python -m scout.stages.grade <run>` (`serve_frames` below). The tool crops the main camera
   around the target, rings it, boxes every other tracked robot with its team and partner or
   opponent (or "unlabelled"), or shows a corner camera panel unmarked; it tiles up to four
   moments with the match clock, lists the facts overlapping them, logs every sheet to
   `grade_work/frames.jsonl` and refuses past IMAGE_CAP sheets. The session's init message is
   checked so a leak fails the stage.
4. Check the structured answer (`check_answer`): citations, one call per ledger line with
   the evidence each call needs, episodes covering every defense item, numbers only as
   references to fact fields, team names only when labelled or read, no purpose words, places
   that agree with where the target was, no text reading a stretch against its ledger call,
   and each score inside the band the rubric computes from the charged seconds and episode
   costs (`charges`, `episode_costs`). On failure the session is resumed with the exact
   errors, up to ATTEMPTS answers in all; then the stage fails.
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Sequence
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
from scout.games.rebuilt2026 import (
    ALLIANCE_ZONE_DEPTH_M,
    BARRIER_X,
    CENTER_LINE_X_M,
    FIELD_LENGTH_M,
    FIELD_WIDTH_M,
    HUBS,
    ROBOT_FOOTPRINT_M,
    guardrail_m,
    own_wall_m,
    rect_distance_m,
    zone_of,
)
from scout.run import RunDir, Stage
from scout.schemas.common import Alliance
from scout.schemas.facts import (
    ContactFact,
    CrossingFact,
    CrossingTurnedBackFact,
    Fact,
    Facts,
    GapFact,
    HubOnFact,
    NearInGap,
    OtherRobot,
    PhaseFact,
    StopFact,
    TripFact,
    ZoneVisitFact,
)
from scout.schemas.grade import (
    SKILLS,
    Call,
    DefenseEpisode,
    DefenseItem,
    DrivingKind,
    DrivingLine,
    FrameSheet,
    Grade,
    GradeAnswer,
    LineCall,
    Note,
    ReadinessLine,
    RunRecord,
    ScoredCall,
    ScoredEpisode,
    ScoredSkill,
    SkillName,
    TileRecord,
    View,
)
from scout.schemas.identity import Identity
from scout.schemas.match import MatchInfo
from scout.schemas.tracks import read_tracks
from scout.schemas.video import VideoInfo
from scout.video import frame_at

F64 = NDArray[np.float64]
I64 = NDArray[np.int64]
Image = NDArray[np.uint8]
JSON = dict[str, object]
Band = tuple[int, int]
Corner = Literal["corner_left", "corner_right"]

GAME_DIR = "rebuilt2026_grade"
"""Folder under `scout.games` with the game guide, limits and rubric for this game."""
IMAGE_CAP = 20
EFFORT = "medium"
"""Claude Code effort level. Medium keeps a grade between 40 and 120 s on Opus."""
MAX_BUDGET_USD = 8.0
"""Stops a runaway session; one grade lists at about 0.3 to 1 USD."""
TIMEOUT_S = 600.0
TOOL = "frames"
SERVER = "scout"
TOOL_ID = f"mcp__{SERVER}__{TOOL}"
SESSION_TOOLS = (TOOL_ID, "StructuredOutput")
"""The only tools a grader session may load: ours, and the one `--json-schema` adds."""
ATTEMPTS = 3
"""Answers checked per grade: the first and up to two retries in the same session, each told
exactly what failed. The checks are strict, so one retry was not always enough."""
NOTE_REACH_S = 5.0
"""A note's `t` must be within this of something it cites."""
FRAME_MATCH_S = 0.051
SPAN_PAD_S = 2.0
"""A frame this close to an episode or ledger line counts as evidence for it."""
MIN_TRACKED_SHARE = 0.5
"""Below this share of teleop tracked, driving may be null."""
CENTRE_NEAR_M = 1.5
BARRIER_NEAR_M = 2.5
"""A place word is supported when the target was this close to the centre line, or to a
barrier's bumps (people say "at the red hub" for a robot a metre or two in front of it)."""
HUB_NEAR_M = 2.0
WALL_NEAR_M = 1.5
""""Beside its hub" needs the centre this close to its own hub's base; "at its wall" or "at the
guardrail" this close to that wall."""
MIN_LOST_S = 1.0
"""Barrier lines costing less active time than this are touches, not ledger lines."""
TRIP_ALLOW_S = 3.0
HOME_CROSS_S = 1.2
HOME_MPS = 2.0
"""Seconds a heaped robot out of its own zone is allowed to get home before the rest of its
trip is charged: a normal crossing plus its distance outside its zone at HOME_MPS (robots
drive home at 2 to 2.5 m/s), and never less than TRIP_ALLOW_S."""
LOOK_TILES = 4
LOOK_FIRST_S = 1.0
"""Hopper tiles of a trip: up to LOOK_TILES, from LOOK_FIRST_S after it left to TRIP_ALLOW_S
before it ended (heaped any later costs nothing)."""
PIN_ALLOW_S = 3.0
"""Still seconds beside an opponent an episode may hold before they cost (pinning longer
than 3 s is a foul in this game)."""
PIN_GAP_S = 2.0
"""Opponent episodes this close in time are one hold: they share one allowance."""
TOUCH_M = 1.0
"""Centre distance under which two robots (0.95 m bumpers) were very probably touching."""
APART_STILL_S = 1.0
"""A contact with the target still this long cannot be called `apart`."""

NEAR_BOX_M = 2.5
"""A boxed robot this close to the target can be the defender or partner a text names."""
TRACKED_WITHIN_S = 0.15
"""A sample this close in time counts as a robot's position at a tile."""
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
    "Yellow ring: target. Boxes: robots the tool tracked (team PARTNER/OPP, or 'unlabelled'). "
    "A robot with NO box was not tracked."
)
CORNER_LEGEND = (
    "CORNER camera, uncalibrated: no ring and no boxes. Find the target by its bumper number."
)

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


def _fmt(v: float) -> str:
    return f"{v:.2f}" if abs(v) < 1 else f"{v:.1f}"


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


def _own_active(match: MatchInfo) -> list[tuple[float, float]]:
    """Teleop spans with the target's hub active."""
    return [
        (p.t_start, p.t_end)
        for p in match.phases
        if p.name != "auto" and p.active_hubs is not None and match.target_alliance in p.active_hubs
    ]


def _relation(match: MatchInfo, team: int) -> Literal["partner", "opponent", "unknown"]:
    slot = next((s for s in match.roster if s.team == team), None)
    if slot is None:
        return "unknown"
    return "partner" if slot.alliance == match.target_alliance else "opponent"


def _home_view(cam: Camera, match: MatchInfo) -> Corner:
    """The corner panel showing the target's own end: the side of the main view its own
    zone projects to."""
    x = 2.0 if match.target_alliance == "blue" else FIELD_LENGTH_M - 2.0
    u = float(cam.project(np.array([[x, FIELD_WIDTH_M / 2, 0.0]]))[0, 0])
    main = match.views["main"]
    return "corner_left" if u < main.x + main.w / 2 else "corner_right"


# --- tracks -----------------------------------------------------------------------------


@dataclass(frozen=True)
class _Target:
    """The target team's samples (what facts.json calls its path)."""

    t: F64
    xy: F64
    box: F64
    """(n, 4) main-view box in full-frame pixels; NaN when interpolated or from a corner view."""

    def near(self, a: float, b: float) -> F64:
        """Positions (k, 2) tracked between video times a and b."""
        return self.xy[(self.t >= a) & (self.t <= b)]


@dataclass(frozen=True)
class _Others:
    """Every other track's samples with its team label (-1 when unlabelled)."""

    t: F64
    track: I64
    team: I64
    xy: F64
    box: F64
    """(n, 4) main-view box; NaN when interpolated or from a corner view."""


def _load_tracks(run: RunDir, match: MatchInfo) -> tuple[_Target, _Others]:
    """The target: samples inside identity segments labelled with it, minus frames two
    tracks both claim. Others: every other sample, with its label."""
    ident = Identity.load(run.identity_json)
    tracks = read_tracks(run.tracks)
    target = match.target_team
    named = [s for s in ident.segments if s.team is not None]
    seg = pl.DataFrame(
        {
            "track_id": [s.track_id for s in named],
            "f0": [s.frame_start for s in named],
            "f1": [s.frame_end for s in named],
            "team": [s.team for s in named],
        },
        schema={"track_id": pl.Int32, "f0": pl.Int64, "f1": pl.Int64, "team": pl.Int64},
    )
    labels = (
        tracks.select("track_id", "frame")
        .join(seg, on="track_id")
        .filter(pl.col("frame").is_between(pl.col("f0"), pl.col("f1")))
        .unique(["track_id", "frame"], keep="first")
        .select("track_id", "frame", "team")
    )
    rows = tracks.join(labels, on=["track_id", "frame"], how="left")

    def boxes(df: pl.DataFrame) -> F64:
        main = ((df["state"] == "measured") & (df["view"] == "main")).fill_null(False)
        box = df.select("x1", "y1", "x2", "y2").to_numpy().astype(np.float64)
        box[~main.to_numpy()] = np.nan
        return box

    mine = rows.filter(pl.col("team") == target)
    mine = mine.unique(["track_id", "frame"]).filter(pl.col("frame").is_unique()).sort("frame")
    rest = rows.filter(pl.col("team").is_null() | (pl.col("team") != target)).sort("frame")
    return (
        _Target(
            t=mine["t"].to_numpy().astype(np.float64),
            xy=mine.select("x", "y").to_numpy().astype(np.float64),
            box=boxes(mine),
        ),
        _Others(
            t=rest["t"].to_numpy().astype(np.float64),
            track=rest["track_id"].to_numpy().astype(np.int64),
            team=rest["team"].fill_null(-1).to_numpy().astype(np.int64),
            xy=rest.select("x", "y").to_numpy().astype(np.float64),
            box=boxes(rest),
        ),
    )


# --- ledgers ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Ledgers:
    driving: list[DrivingLine]
    readiness: list[ReadinessLine]
    items: list[DefenseItem]


def _overlap(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 <= b1 + 1e-6 and b0 <= a1 + 1e-6


def _overlap_s(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _union_s(a: float, b: float, spans: Iterable[tuple[float, float]]) -> float:
    """Seconds of [a, b] covered by the union of `spans`."""
    total, at = 0.0, a
    for s0, s1 in sorted((max(a, s0), min(b, s1)) for s0, s1 in spans):
        if s1 > at and s1 > s0:
            total += s1 - max(at, s0)
            at = s1
    return total


def _in_spans(t: float, spans: Sequence[tuple[float, float]]) -> bool:
    return any(a - 1e-6 <= t < b for a, b in spans)


def _contacts_in(facts: Sequence[Fact], a: float, b: float) -> tuple[list[str], list[str]]:
    """(contacts with an opponent or unlabelled robot, contacts with only partners) in [a, b]."""
    opp: list[str] = []
    partner: list[str] = []
    for f in facts:
        if isinstance(f, ContactFact) and _overlap(f.t_start, f.t_end, a, b):
            if all(o.relation == "partner" for o in f.data.others):
                partner.append(f.id)
            else:
                opp.append(f.id)
    return opp, partner


def _per_team[T: (OtherRobot, NearInGap)](rows: Sequence[T]) -> list[T]:
    """Rows nearest first, one per labelled team (a team on several tracks is listed once)."""
    seen: set[int] = set()
    out: list[T] = []
    for o in rows:
        if o.team is None or o.team not in seen:
            out.append(o)
        if o.team is not None:
            seen.add(o.team)
    return out


def _end_text(end: str) -> str:
    return {
        "target_moved_away": "the target moved away",
        "other_moved_away": "the other robot moved away",
        "unknown_other_track_lost": "tracking of the other robot ended: how it ended is unknown",
        "unknown_target_track_lost": "tracking of the target ended: how it ended is unknown",
        "match_end": "the match ended",
        "reached_far_side": "it reached the far side",
        "tried_again": "its next attempt on this barrier began",
        "hub_went_inactive": "its hub went inactive",
        "track_lost": "tracking lost it (it took at least this long)",
    }.get(end, end)


def _look_times(a: float, b: float) -> list[float]:
    """Hopper tile times for a trip from a to b, evenly spaced, at most 5 s apart."""
    first, last = a + LOOK_FIRST_S, max(a + LOOK_FIRST_S, b - TRIP_ALLOW_S)
    n = max(1, min(LOOK_TILES, 1 + math.floor((last - first) / 1.5)))
    step = min(5.0, round((last - first) / (n - 1), 1)) if n > 1 else 0.0
    return [round(first + k * step, 1) for k in range(n)]


@dataclass(frozen=True)
class _Raw:
    """A driving line before numbering."""

    a: float
    b: float
    kind: DrivingKind
    facts: list[str]
    lost: float
    unseen: float
    text: str
    bound: bool = False
    corner: bool = False
    look: tuple[float, ...] = ()


def ledgers(facts: Facts, match: MatchInfo) -> Ledgers:
    """The driving and readiness ledgers and the defense items, from the facts alone."""
    fs = facts.facts
    teleop = match.teleop_start_t - 1e-6
    active = _own_active(match)
    contact_m = facts.thresholds.get("CONTACT_M", 1.6)
    raw: list[_Raw] = []

    def taken(a: float, b: float) -> bool:
        return any(_overlap(a, b, r.a, r.b) for r in raw)

    for f in fs:
        if not (isinstance(f, CrossingTurnedBackFact) and f.t_start >= teleop):
            continue
        d = f.data
        if not d.toward_home:
            continue
        folded = [
            s
            for s in fs
            if isinstance(s, StopFact)
            and _overlap(s.t_start, s.t_end, f.t_start, f.t_end)
            and s.data.barrier_edge_m <= ROBOT_FOOTPRINT_M
        ]
        arrive = (
            f"first tracked in the {d.far_side} zone {_fmt(d.seconds_to_far_side)} s after "
            "entering" + (" (after a gap: at most that)" if d.far_side_after_gap else "")
            if d.seconds_to_far_side is not None
            else f"never tracked in the {d.far_side} zone after it"
        )
        stops = "".join(
            f"; {s.id}: stopped {_fmt(s.data.duration_s)} s ({_fmt(s.data.unseen_s)} s inferred "
            f"across gaps), {_fmt(s.data.barrier_edge_m)} m from the bumps' depth"
            for s in folded
        )
        text = (
            f"onto the {d.barrier} {d.lane} from the {d.side} zone toward the {d.far_side} zone "
            f"and back out (reach {d.reach}, centre {_fmt(d.nearest_edge_m)} m from the bumps' "
            f"depth at nearest); on the barrier {_fmt(d.time_on_barrier_s)} s, stopped "
            f"{_fmt(d.stopped_s)} s, {_fmt(d.unseen_s)} s unseen; {arrive}; costed "
            f"{_fmt(d.cost_s)} s until {_end_text(d.cost_end)}, {_fmt(d.own_active_s)} s of it in "
            f"its own active time{stops}"
        )
        raw.append(
            _Raw(
                f.t_start, f.t_end, "turned_back", [f.id, *(s.id for s in folded)],
                d.own_active_s, d.unseen_s, text,
                bound=d.far_side_after_gap and d.cost_end == "reached_far_side",
                corner=d.reach == "corner",
            )
        )  # fmt: skip
    for f in fs:
        if (
            isinstance(f, StopFact)
            and f.t_start >= teleop
            and f.data.on_barrier is not None
            and not taken(f.t_start, f.t_end)
        ):
            d = f.data
            text = (
                f"stopped on the {d.on_barrier} {d.lane or 'barrier'} {_fmt(d.duration_s)} s: "
                f"{_fmt(d.seen_s)} s seen, {_fmt(d.unseen_s)} s inferred across gaps (start "
                f"{d.start}, end {d.end}); {_fmt(d.own_active_s)} s in its own active time"
            )
            raw.append(
                _Raw(f.t_start, f.t_end, "barrier_stop", [f.id], d.own_active_s, d.unseen_s, text)
            )
    for f in fs:
        if (
            isinstance(f, ZoneVisitFact)
            and f.t_start >= teleop
            and f.data.zone == "on_barrier"
            and not taken(f.t_start, f.t_end)
        ):
            d = f.data
            excess = d.excess_s or 0.0
            text = (
                f"stayed on the {d.barrier} {d.lane} {_fmt(d.duration_s)} s "
                f"({_fmt(d.untracked_s)} s untracked), from {d.came_from} to {d.went_to}: "
                f"{_fmt(excess)} s beyond a normal crossing, {_fmt(d.own_active_s)} s in its own "
                "active time"
            )
            raw.append(
                _Raw(f.t_start, f.t_end, "barrier_stay", [f.id], min(excess, d.own_active_s),
                     d.untracked_s, text)
            )  # fmt: skip
    for f in fs:
        if not isinstance(f, CrossingFact) or f.t_start < teleop or taken(f.t_start, f.t_end):
            continue
        d = f.data
        on = d.time_on_barrier_s if d.time_on_barrier_s is not None else d.observed_s
        text = (
            f"crossed the {d.barrier} {d.lane} from {d.from_zone} to {d.to_zone} in {_fmt(on)} s, "
            f"stopped {_fmt(d.stopped_s)} s: {_fmt(d.excess_s)} s beyond a normal crossing, "
            f"{_fmt(d.own_active_s)} s in its own active time"
        )
        raw.append(
            _Raw(f.t_start, f.t_end, "slow_crossing", [f.id], min(d.excess_s, d.own_active_s),
                 0.0, text)
        )  # fmt: skip
    raw = [r for r in raw if r.lost >= MIN_LOST_S]
    for f in fs:
        if (
            not isinstance(f, TripFact)
            or f.data.duration_s < LOOK_FIRST_S + TRIP_ALLOW_S + MIN_LOST_S
        ):
            continue
        d = f.data
        past = d.past_centre_line_m
        text = (
            f"out of its own zone in its own active time {_fmt(d.duration_s)} s "
            f"({_fmt(d.unseen_s)} s unseen), furthest {_fmt(abs(past))} m "
            f"{'past' if past >= 0 else 'short of'} the centre line, "
            + ("then back home" if d.back is not None else "not back before its active time ended")
            + (f"; stops out: {', '.join(d.stop_ids)}" if d.stop_ids else "")
        )
        look = tuple(_look_times(f.t_start, f.t_end))
        raw.append(
            _Raw(f.t_start, f.t_end, "loaded_trip", [f.id], d.duration_s, d.unseen_s, text,
                 look=look)
        )  # fmt: skip
    driving: list[DrivingLine] = []
    for n, r in enumerate(sorted(raw, key=lambda r: r.a), 1):
        opp, partner = _contacts_in(fs, r.a, r.b)
        driving.append(
            DrivingLine(
                id=f"D{n}",
                kind=r.kind,
                facts=r.facts,
                t_start=r.a,
                t_end=r.b,
                phase=_phase(match, r.a),
                lost_s=round(r.lost, 2),
                bound=r.bound,
                unseen_s=round(r.unseen, 2),
                corner_reach=r.corner,
                look_at=list(r.look),
                opponent_contacts=opp,
                partner_contacts=partner,
                text=r.text,
            )
        )

    readiness: list[ReadinessLine] = []
    for f in fs:
        if not isinstance(f, HubOnFact):
            continue
        d = f.data
        ph = next(p for p in match.phases if p.name == d.phase)
        never = d.in_own_zone is not True and d.reached_own_zone_after_s is None
        late = (
            0.0
            if d.in_own_zone
            else (ph.t_end - ph.t_start if never else d.reached_own_zone_after_s or 0.0)
        )
        bound = bool(not d.in_own_zone and d.reached_after_gap)
        where = (
            "in its own zone"
            if d.in_own_zone
            else "outside its own zone"
            if d.in_own_zone is False
            else "untracked at that moment"
        )
        arrive = (
            ""
            if d.in_own_zone
            else "; never tracked in its own zone in this phase"
            if never
            else f"; first tracked in its own zone {_fmt(late)} s later"
            + (" (first seen after a gap: at most that late)" if bound else "")
        )
        readiness.append(
            ReadinessLine(
                id=f"R{len(readiness) + 1}",
                fact=f.id,
                phase=d.phase,
                t=f.t_start,
                in_zone=d.in_own_zone,
                late_s=round(late, 2),
                late_is_bound=bound,
                never_home=never,
                text=f"its hub turned on for {d.phase}: {where}{arrive}",
            )
        )

    items: list[DefenseItem] = []
    for f in fs:
        if f.t_start < teleop:
            continue
        in_active = _in_spans(f.t_start, active)
        if isinstance(f, ContactFact):
            d = f.data
            if all(o.relation == "partner" for o in d.others):
                continue
            who = ", ".join(
                (
                    f"{o.team} {o.relation} (track label)"
                    if o.team is not None
                    else f"unlabelled track {o.track_id} (detector bumper colour {o.bumper_seen})"
                )
                + f" {_fmt(o.min_distance_m)} m"
                + (
                    f", closing speeds: target {_fmt(o.target_closing_mps)}, other "
                    f"{_fmt(o.other_closing_mps)} m/s"
                    if o.target_closing_mps is not None and o.other_closing_mps is not None
                    else ""
                )
                for o in _per_team(d.others)
            )
            text = (
                f"contact, target {d.target_zone} ({_fmt(d.where.own_hub_m)} m from its hub, "
                f"{_fmt(d.where.own_wall_m)} m from its wall), own hub "
                f"{'active' if in_active else 'inactive'}; {who}; {_fmt(d.seen_s)} s seen, "
                f"{_fmt(d.unseen_s)} s unseen; target nearly still {_fmt(d.still_s)} s "
                f"({_fmt(d.still_unseen_s)} s of it inferred across gaps, longest run "
                f"{_fmt(d.longest_still_s)} s); end: {_end_text(d.end)}"
            )
            items.append(
                DefenseItem(
                    fact=f.id,
                    t_start=f.t_start,
                    t_end=f.t_end,
                    home_active=in_active and d.target_zone == "own",
                    pin_s=round(d.still_s - d.still_unseen_s / 2, 2),
                    text=text,
                )
            )
        elif isinstance(f, GapFact) and any(
            _in_spans(t, active) for t in (f.t_start, f.t_end - 1e-3)
        ):
            d = f.data
            near = [o for o in d.others_near if o.relation != "partner"]
            if not near:
                continue
            who = ", ".join(
                (
                    f"{o.team} {o.relation}"
                    if o.team is not None
                    else f"unlabelled track {o.track_id}"
                )
                + f" {_fmt(o.min_distance_m)} m"
                for o in _per_team(d.others_near)
            )
            zones = f"{d.before.zone if d.before else '?'} to {d.after.zone if d.after else '?'}"
            text = (
                f"target unseen {_fmt(d.duration_s)} s ({zones}, inference {d.inference}); "
                f"robots tracked near its last or next seen spot: {who}"
            )
            home = bool(d.before and d.after and d.before.zone == d.after.zone == "own")
            items.append(
                DefenseItem(
                    fact=f.id, t_start=f.t_start, t_end=f.t_end, home_active=home, pin_s=0.0,
                    text=text,
                )
            )  # fmt: skip
        elif (
            isinstance(f, StopFact)
            and f.data.unseen_s > 0
            and in_active
            and f.data.nearest_tracked_m is not None
            and f.data.nearest_tracked_m <= contact_m
        ):
            d = f.data
            text = (
                f"stop {_fmt(d.duration_s)} s in the {d.where.zone} zone, {_fmt(d.unseen_s)} s "
                f"inferred across gaps; nearest tracked robot {_fmt(d.nearest_tracked_m or 0.0)} m"
            )
            items.append(
                DefenseItem(
                    fact=f.id,
                    t_start=f.t_start,
                    t_end=f.t_end,
                    home_active=d.where.zone == "own" and d.on_barrier is None,
                    pin_s=round(d.seen_s + d.unseen_s / 2, 2),
                    text=text,
                )
            )
    return Ledgers(driving, readiness, items)


def _ledger_text(led: Ledgers, match: MatchInfo, home: Corner) -> tuple[str, str, str]:
    """The driving ledger, readiness ledger and defense items as prompt text. Lines that
    depend on a moment the target was untracked point at the corner view of its own end."""

    def span(a: float, b: float) -> str:
        return f"{a:.1f}-{b:.1f} s ({_clock(match, a)[1]}, {_phase(match, a)})"

    corner = f"untracked for part of it: look in the {home} view (its own end) too"

    def drive_line(d: DrivingLine) -> str:
        flags = [
            *(["lost_s is an upper bound"] if d.bound else []),
            *(["corner reach only: may be a drive past with no stall"] if d.corner_reach else []),
            *([corner] if d.unseen_s >= 1.0 or d.bound else []),
        ]
        look = (
            f"; HOPPER TILES: frames t={d.look_at[0]:.1f} count {len(d.look_at)} step_s "
            f"{(d.look_at[1] - d.look_at[0]) if len(d.look_at) > 1 else 1.0:.1f} zoom 2"
            if d.look_at
            else ""
        )
        return (
            f"- {d.id} {d.kind} {span(d.t_start, d.t_end)}: lost_s {_fmt(d.lost_s)}, unseen_s "
            f"{_fmt(d.unseen_s)}{''.join(f' ({x})' for x in flags)}; facts {', '.join(d.facts)}; "
            f"{d.text}; contacts with opponent or unlabelled robots: "
            f"{', '.join(d.opponent_contacts) or 'none'}; with partners only: "
            f"{', '.join(d.partner_contacts) or 'none'}{look}"
        )

    drive = "\n".join(drive_line(d) for d in led.driving)
    ready = "\n".join(
        f"- {r.id} {r.fact} at {r.t:.1f} s ({_clock(match, r.t)[1]}): late_s {_fmt(r.late_s)}"
        f"{' (bound)' if r.late_is_bound else ''}; {r.text}"
        + (f" ({corner})" if r.in_zone is None or r.late_is_bound else "")
        for r in led.readiness
    )
    items = "\n".join(
        f"- {i.fact} {span(i.t_start, i.t_end)}{', HOME IN ACTIVE TIME' if i.home_active else ''}"
        f"{f', pin_s {_fmt(i.pin_s)}' if i.pin_s > 0 else ''}: {i.text}"
        for i in led.items
    )
    return drive or "(no lines)", ready or "(no lines)", items or "(no items)"


# --- costs and bands --------------------------------------------------------------------


def _band_of(x: float, edges: Sequence[float]) -> Band:
    """9-10 at or below edges[0], then one band lower past each edge."""
    k = sum(x > e + 1e-9 for e in edges)
    return (9 - 2 * k, 10 - 2 * k)


DRIVING_EDGES = (0.5, 3.0, 6.0, 12.0)
READINESS_EDGES = (1.0, 3.0, 8.0)
DEFENSE_EDGES = (1.0, 4.0, 8.0, 15.0)
"""Band edges in seconds: 9-10 at or under the first, one band lower past each. Set from the
spread of charged seconds over the development matches (see rubric.md)."""
HELD_SHARE: dict[str, float] = {"not_shooting": 1.0, "cannot_see": 0.5, "shooting": 0.0}


def _episode_span(e: DefenseEpisode, by_id: dict[str, Fact]) -> tuple[float, float] | None:
    spans = [(f.t_start, f.t_end) for i in e.facts if (f := by_id.get(i)) is not None]
    if not spans:
        return None
    return (min(a for a, _ in spans), max(b for _, b in spans))


@dataclass(frozen=True)
class _Cost:
    """One episode's cost: still seconds beyond the allowance, the pin cost (those scaled by
    `while_held`) and the total the defense score counts."""

    held: float
    pin: float
    total: float


def episode_costs(episodes: Sequence[DefenseEpisode], items: Sequence[DefenseItem]) -> list[_Cost]:
    """Cost per episode. For `opponent_on_target`: its items' still seconds (a stop item
    overlapping a contact in the same episode counted once) beyond PIN_ALLOW_S, scaled by
    `while_held`; the total is the larger of that and the grader's `cost_s`. Episodes that
    follow each other within PIN_GAP_S share one allowance, so splitting a hold costs the same."""
    by_fact = {i.fact: i for i in items}
    out: dict[int, _Cost] = {}
    held: list[tuple[float, float, int, float]] = []
    for k, e in enumerate(episodes):
        mine = [by_fact[f] for f in e.facts if f in by_fact]
        if e.verdict != "opponent_on_target" or not mine:
            out[k] = _Cost(0.0, 0.0, 0.0)
            continue
        contacts = [i for i in mine if i.fact.startswith("contact")]
        still = sum(
            i.pin_s
            for i in mine
            if i in contacts
            or not any(_overlap(i.t_start, i.t_end, c.t_start, c.t_end) for c in contacts)
        )
        held.append((min(i.t_start for i in mine), max(i.t_end for i in mine), k, still))
    left, last = PIN_ALLOW_S, -math.inf
    for a, b, k, still in sorted(held):
        if a - last > PIN_GAP_S:
            left = PIN_ALLOW_S
        beyond = max(0.0, still - left)
        left, last = max(0.0, left - still), max(last, b)
        pin = beyond * HELD_SHARE.get(episodes[k].while_held or "", 0.0)
        out[k] = _Cost(round(beyond, 2), round(pin, 2), round(max(episodes[k].cost_s, pin), 2))
    return [out[k] for k in range(len(episodes))]


@dataclass(frozen=True)
class Charges:
    """What code charges from the calls and episodes."""

    line: dict[str, float]
    """Seconds per driving line, lateness per readiness line."""
    driving_s: float
    blind: bool
    """A trip's hopper could not be seen: the driving score is held to its band's lower end."""
    episode: list[_Cost]


def _home_allow(target: _Target, alliance: Alliance, t: float) -> float:
    """Seconds a heaped robot at video time `t` needs to get home (see TRIP_ALLOW_S)."""
    near = target.near(t - 1.0, t + 1.0)
    if not len(near):
        return TRIP_ALLOW_S
    out_m = max(0.0, own_wall_m(float(np.median(near[:, 0])), alliance) - ALLIANCE_ZONE_DEPTH_M)
    return max(TRIP_ALLOW_S, HOME_CROSS_S + out_m / HOME_MPS)


def charges(
    led: Ledgers,
    calls: dict[str, LineCall],
    episodes: Sequence[DefenseEpisode],
    by_id: dict[str, Fact],
    target: _Target,
    alliance: Alliance,
) -> Charges:
    """Charged seconds per line. Barrier lines: lost_s (or a tile-backed charged_s) less
    the time opponent episodes cost inside them, half when uncertain. Heaped trips: the time
    from `heaped_at` to the end of the trip, less charged or excused barrier lines and
    opponent episodes inside it, less the time to get home from there (`_home_allow`).
    Readiness: late_s, half when uncertain."""
    costs = episode_costs(episodes, led.items)
    opp = [
        (s, cost.total)
        for e, cost in zip(episodes, costs, strict=True)
        if e.verdict == "opponent_on_target" and (s := _episode_span(e, by_id)) is not None
    ]
    out: dict[str, float] = {}
    held: list[tuple[float, float]] = []
    for d in led.driving:
        c = calls.get(d.id)
        if d.kind == "loaded_trip" or c is None:
            continue
        cut = sum(min(_overlap_s(d.t_start, d.t_end, s0, s1), t) for (s0, s1), t in opp)
        base = max(0.0, (c.charged_s if c.charged_s is not None else d.lost_s) - cut)
        out[d.id] = base if c.call == "charged" else base / 2 if c.call == "uncertain" else 0.0
        if c.call in ("charged", "uncertain", "opponent", "partner"):
            held.append((d.t_start, d.t_end))
    blind = False
    for d in led.driving:
        c = calls.get(d.id)
        if d.kind != "loaded_trip" or c is None:
            continue
        blind |= c.call == "cannot_see"
        if c.call != "heaped" or c.heaped_at is None:
            out[d.id] = 0.0
            continue
        a = max(d.t_start, c.heaped_at)
        excused = _union_s(a, d.t_end, [*held, *(s for s, _ in opp)])
        out[d.id] = max(0.0, d.t_end - a - excused - _home_allow(target, alliance, a))
    for r in led.readiness:
        c = calls.get(r.id)
        if c is not None:
            out[r.id] = (
                r.late_s if c.call == "charged" else r.late_s / 2 if c.call == "uncertain" else 0.0
            )
    out = {k: round(v, 2) for k, v in out.items()}
    total = sum(out[d.id] for d in led.driving if d.id in out)
    return Charges(out, round(total, 2), blind, costs)


def driving_band(ch: Charges) -> Band:
    """From the charged seconds; held to its lower end when a hopper could not be seen."""
    lo, hi = _band_of(ch.driving_s, DRIVING_EDGES)
    return (lo, lo) if ch.blind else (lo, hi)


def readiness_band(
    lines: Sequence[ReadinessLine], calls: dict[str, LineCall], ch: Charges
) -> Band | None:
    """From the latest charged activation; two late ones cap it at 5-6."""
    if not lines:
        return None
    for r in lines:
        c = calls.get(r.id)
        if c is not None and c.call == "charged" and r.never_home:
            return (1, 2)
    late = [ch.line.get(r.id, 0.0) for r in lines]
    band = _band_of(max(late, default=0.0), READINESS_EDGES)
    if sum(x > READINESS_EDGES[0] for x in late) >= 2:
        band = min(band, (5, 6))
    return band


def defense_band(episodes: Sequence[DefenseEpisode], ch: Charges) -> Band | None:
    """From the summed cost of the `opponent_on_target` episodes; None when there is none."""
    costs = [
        c.total
        for e, c in zip(episodes, ch.episode, strict=True)
        if e.verdict == "opponent_on_target"
    ]
    if not costs:
        return None
    return _band_of(sum(costs), DEFENSE_EDGES)


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


def _spot(target: _Target, cam: Camera, t: float) -> _Spot:
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


def _marks(others: _Others, match: MatchInfo, spot: _Spot, t: float, same_m: float) -> list[_Mark]:
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
                relation="unknown" if team < 0 else _relation(match, team),
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
        cv2.ellipse(tile, (cx, cy), (round(ax_), round(ay_)), 0, 0, 360, color, 3)
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
        self.target, self.others = _load_tracks(run, self.match)
        self.facts = Facts.load(run.facts_json)
        self.same_m = self.facts.thresholds.get("SAME_ROBOT_M", 0.7)
        self.home = _home_view(self.cam, self.match)
        m = self.match.views["main"]
        self.panel = (m.x, m.y, m.w, m.h)

    def call(self, req: FrameRequest) -> tuple[str, Image | None]:
        """Render one contact sheet; returns the text for the agent and the image, or a
        refusal and None."""
        done = len(_served(self.run))
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
        with _frame_log(self.run).open("a") as f:
            f.write(record.model_dump_json() + "\n")
        near = [
            f"{f.id} ({f.t_start:.1f}-{f.t_end:.1f})"
            for f in self.facts.facts
            if f.kind in FACT_KINDS_LISTED
            and _overlap(f.t_start, f.t_end, times[0] - 0.5, times[-1] + 0.5)
        ]
        end = "the target's own end" if req.view == self.home else "the opponents' end"
        how = (
            f"{req.view} is a corner camera showing {end}. "
            "It is not calibrated: the tool cannot ring the target or box robots there. Name the "
            "target only where you read its bumper number, and give that number as "
            "corner_bumper in any call that cites these tiles."
            if corner
            else "Yellow ring: the target's measured box; orange ring: its position estimated "
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


def _prompts(match: MatchInfo, facts_json: str, led: Ledgers, home: Corner) -> tuple[str, str]:
    """System prompt (rules, game, limits, rubric) and task (match, ledgers and facts)."""
    system = "\n\n".join(
        [
            _text("grade_system.md").replace("{{image_cap}}", str(IMAGE_CAP)),
            _text("game.md"),
            _text("limits.md"),
            _text("rubric.md"),
        ]
    )
    roster = ", ".join(
        f"{s.team} ({s.alliance} {s.station}, "
        f"{'target' if s.team == match.target_team else _relation(match, s.team)})"
        for s in match.roster
    )
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
    drive, ready, items = _ledger_text(led, match, home)
    compact = json.dumps(json.loads(facts_json), separators=(",", ":"))
    fill = {
        "team": str(match.target_team),
        "alliance": match.target_alliance,
        "roster": roster,
        "auto_winner": match.auto_winner or "unknown",
        "phases": phases,
        "own_active": ", ".join(own),
        "home_view": home,
        "driving_ledger": drive,
        "readiness_ledger": ready,
        "defense_items": items,
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

REF = re.compile(r"\{([^{}]*)\}")
SHIFT_WORD = re.compile(r"\bshifts? ?[1-4](?:(?:, ?| and | or | to )[1-4])*\b", re.IGNORECASE)
NUMBER = re.compile(r"\d+(?:\.\d+)?")
NEGATED = re.compile(r"\b(no|not|without|never|nor)\W+(?:\w+\W+){0,3}$", re.IGNORECASE)
DEFENDER_WORD = re.compile(r"\b(defenders?|defending|defended|pinn(?:ed|ing)|pins?)\b", re.I)
PARTNER_WORD = re.compile(r"\bpartners?\b", re.IGNORECASE)
MOTIVE_WORD = re.compile(
    r"\b(re-?load\w*|refill\w*|collect\w*|gather\w*|feed\w*|fed|stockpil\w*|wait\w*|intak\w*|"
    r"scoop\w*|tr(?:y|ies|ied|ying) to|want\w*|decid\w*|in order to|"
    r"looking for|hunting)\b",
    re.IGNORECASE,
)
FUEL_WORD = re.compile(
    r"\b(fuel|balls?|hopper|heaped|loaded|shots?|shoot\w*|fir(?:e|es|ed|ing)|volleys?|scored|"
    r"scoring)\b",
    re.IGNORECASE,
)
CLEAN_WORD = re.compile(
    r"\b(clean(?:ly)?|smooth(?:ly)?|no hesitation|unhindered|staging|staged|kept moving|"
    r"keeps moving|kept working|kept its route|played through|free(?:d)? itself)\b",
    re.IGNORECASE,
)
"""Words reading a stretch as fine: rejected over a charged line or a costed pin."""
FAULT_WORD = re.compile(
    r"\b(stall\w*|stuck|hesitat\w*|backed off|backing off|abort\w*|wasted|beached|wedged)\b",
    re.IGNORECASE,
)
"""Words reading a stretch as a driving fault: rejected over a line a tile cleared."""
PLACE_WORDS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bcent(?:re|er)[ -]?line\b|\bmidfield\b", re.I), "centre line"),
    (re.compile(r"\bred (?:hub|bump|ramp|trench)", re.I), "red barrier"),
    (re.compile(r"\bblue (?:hub|bump|ramp|trench)", re.I), "blue barrier"),
    (re.compile(r"\b(?:own|its|alliance) zone\b", re.I), "own zone"),
    (re.compile(r"\bneutral zone\b", re.I), "neutral zone"),
)
POINT_WORDS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(?:beside|next to|by|at|against|near|under|in front of|up to) "
            r"(?:its |the |their )?(?:own )?hub\b",
            re.I,
        ),
        "beside its hub",
    ),
    (re.compile(r"\b(?:alliance|driver[- ]station|its own|its|back) wall\b", re.I), "at its wall"),
    (re.compile(r"\bguard ?rail\b|\bside wall\b", re.I), "at the guardrail"),
)
"""Place words checked against distances: the cited stop's or contact's `where`, else the
target's tracked positions over what the item cites."""
ATTACK = frozenset({"opponent_on_target", "target_on_opponent"})
BARRIER_CALLS: frozenset[Call] = frozenset(
    {"charged", "opponent", "partner", "frame_contradicts", "uncertain"}
)
TRIP_CALLS: frozenset[Call] = frozenset({"heaped", "not_heaped", "cannot_see", "frame_contradicts"})
READINESS_CALLS: frozenset[Call] = frozenset(
    {"charged", "opponent", "frame_contradicts", "uncertain"}
)


@dataclass(frozen=True)
class GradeContext:
    """Everything `check_answer` checks against, built once per run."""

    facts: Facts
    match: MatchInfo
    led: Ledgers
    target: _Target
    teleop_tracked_share: float

    @property
    def by_id(self) -> dict[str, Fact]:
        return {f.id: f for f in self.facts.facts}


def grade_context(run: RunDir) -> GradeContext:
    """Load what the checker needs from a run folder."""
    match = MatchInfo.load(run.match_json)
    facts = Facts.load(run.facts_json)
    target, _ = _load_tracks(run, match)
    teleop = [f for f in facts.facts if isinstance(f, PhaseFact) and f.phase != "auto"]
    dur = sum(f.data.coverage.duration_s for f in teleop)
    tracked = sum(f.data.coverage.tracked_s for f in teleop)
    return GradeContext(facts, match, ledgers(facts, match), target, tracked / dur if dur else 0.0)


@dataclass
class _Item:
    """One answer item with text, for the text checks."""

    at: str
    text: str
    cites: list[str]
    bumpers_read: list[int]
    roles: bool
    """Check role words and places (notes, calls, episodes)."""


@dataclass(frozen=True)
class _At:
    """What was served at one cited moment: the main-camera tile, and whether a corner
    panel was shown then too."""

    main: TileRecord | None
    corner: bool


class _Checker:
    """Collects problems while checking one answer."""

    def __init__(self, ctx: GradeContext, sheets: list[FrameSheet]) -> None:
        self.ctx = ctx
        self.by_id = ctx.by_id
        self.tiles: dict[float, _At] = {}
        for s in sheets:
            for tile in s.tiles:
                t = round(tile.t, 1)
                was = self.tiles.get(t, _At(None, False))
                corner = s.view.startswith("corner")
                self.tiles[t] = _At(was.main if corner else tile, was.corner or corner)
        self.problems: list[str] = []
        self.lines: dict[str, DrivingLine | ReadinessLine] = {
            **{d.id: d for d in ctx.led.driving},
            **{r.id: r for r in ctx.led.readiness},
        }
        self.episodes: list[DefenseEpisode] = []
        self.charged_lines: list[DrivingLine] = []
        self.cleared_lines: list[tuple[DrivingLine, str]] = []
        self.pinned: list[tuple[float, float]] = []

    def bad(self, msg: str) -> None:
        self.problems.append(msg)

    def frame_t(self, cite: str) -> float | None:
        try:
            t = float(cite.removeprefix("frame:"))
        except ValueError:
            return None
        return next((s for s in self.tiles if abs(t - s) <= FRAME_MATCH_S), None)

    def span(self, cite: str, at: str) -> tuple[float, float] | None:
        """Time span of a citation, or None (with a problem) when it is not valid."""
        if cite.startswith("frame:"):
            t = self.frame_t(cite)
            if t is None:
                self.bad(f"{at}: {cite} was never served by the frame tool (cite as frame:<t>)")
                return None
            return (t, t)
        f = self.by_id.get(cite)
        if f is None:
            self.bad(f"{at}: {cite!r} is not a fact id in facts.json")
            return None
        return (f.t_start, f.t_end)

    def frames_in(
        self, cites: Iterable[str], a: float, b: float, pad: float = SPAN_PAD_S
    ) -> list[float]:
        out: list[float] = []
        for c in cites:
            t = self.frame_t(c) if c.startswith("frame:") else None
            if t is not None and a - pad <= t <= b + pad:
                out.append(t)
        return out

    def corner_only(self, t: float) -> bool:
        at = self.tiles[t]
        return at.corner and at.main is None

    # who is who

    def labelled(self, cites: Iterable[str], near_only: bool = False) -> set[int]:
        """Teams labelled in cited facts or boxed on cited tiles (`near_only`: only boxes
        within NEAR_BOX_M of the target)."""
        out: set[int] = set()
        for c in cites:
            if c.startswith("frame:"):
                t = self.frame_t(c)
                tile = self.tiles[t].main if t is not None else None
                if tile is not None:
                    out |= set(tile.teams_near if near_only else tile.teams_marked)
                continue
            f = self.by_id.get(c)
            if isinstance(f, ContactFact):
                out |= {o.team for o in f.data.others if o.team is not None}
            elif isinstance(f, GapFact):
                out |= {o.team for o in f.data.others_near if o.team is not None}
        return out

    def spans_of(self, cites: Iterable[str]) -> list[tuple[float, float]]:
        return [s for c in cites if (s := self._quiet_span(c)) is not None]

    def episode_roles(self, cites: Iterable[str]) -> set[str]:
        """Verdicts of episodes overlapping the cited spans."""
        spans = self.spans_of(cites)
        out: set[str] = set()
        for e in self.episodes:
            es = _episode_span(e, self.by_id)
            if es is not None and any(_overlap(a, b, es[0], es[1]) for a, b in spans):
                out.add(e.verdict)
        return out

    def _quiet_span(self, cite: str) -> tuple[float, float] | None:
        if cite.startswith("frame:"):
            t = self.frame_t(cite)
            return None if t is None else (t, t)
        f = self.by_id.get(cite)
        return None if f is None else (f.t_start, f.t_end)

    # text

    def resolve(self, item: _Item) -> str:
        """Check one text field and return it with its references filled in."""
        at, text = item.at, item.text
        cited = set(item.cites)
        out = text

        def value(ref: str) -> str | None:
            if ref.startswith("frame:"):
                t = self.frame_t(ref)
                if t is None:
                    self.bad(f"{at}: {{{ref}}} quotes a frame the tool never served")
                    return None
                return _clock(self.ctx.match, t)[1]
            head, _, path = ref.partition(".")
            if head in self.lines:
                obj: object = self.lines[head].model_dump()
            elif head in self.by_id:
                if head not in cited:
                    self.bad(f"{at}: {{{ref}}} quotes {head}; add {head} to this item's cites")
                    return None
                dump = self.by_id[head].model_dump()
                data = dump.get("data")
                first = path.split(".")[0]
                obj = data if isinstance(data, dict) and first in data else dump
            else:
                self.bad(f"{at}: {{{ref}}} names no fact id, ledger line or frame")
                return None
            for part in path.split(".") if path else []:
                if isinstance(obj, dict) and part in obj:
                    obj = TypeAdapter(dict[str, object]).validate_python(obj)[part]
                elif isinstance(obj, list) and part.isdigit():
                    items = TypeAdapter(list[object]).validate_python(obj)
                    obj = items[int(part)] if int(part) < len(items) else None
                else:
                    self.bad(f"{at}: {{{ref}}}: {head} has no field {path!r}")
                    return None
            if isinstance(obj, bool):
                return "yes" if obj else "no"
            if isinstance(obj, int):
                return str(obj)
            if isinstance(obj, float):
                return _fmt(obj)
            if isinstance(obj, str):
                return obj
            self.bad(f"{at}: {{{ref}}} is {obj!r}, not a single value to quote")
            return None

        for m in REF.finditer(text):
            v = value(m.group(1).strip())
            if v is not None:
                out = out.replace(m.group(0), v, 1)
        bare = SHIFT_WORD.sub(" ", REF.sub(" ", text))
        roster = {s.team for s in self.ctx.match.roster}
        named: set[int] = set()
        for m in NUMBER.finditer(bare):
            tok = m.group(0)
            if tok.isdigit() and int(tok) in roster:
                named.add(int(tok))
                continue
            self.bad(
                f"{at}: the number {tok!r} is typed into the text; numbers must be references "
                "like {stop-03.duration_s}, {D1.lost_s} or {frame:92.8}, or a team number"
            )
        has_frame = any(c.startswith("frame:") for c in item.cites)
        known = self.labelled(item.cites) | {self.ctx.match.target_team}
        if has_frame:
            known |= set(item.bumpers_read)
        for team in sorted(named - known):
            self.bad(
                f"{at}: names {team}, which no cited fact labels and no cited tile boxes; cite "
                "the fact or tile that labels it, or a tile where you read its bumper number "
                "(and list it in bumpers_read), or call it 'an unidentified robot'"
            )
        if (m := MOTIVE_WORD.search(bare)) is not None:
            self.bad(
                f"{at}: {m.group(0)!r} claims a purpose positions cannot show; say where it went "
                "and what a frame shows, not why"
            )
        if not has_frame and (m := FUEL_WORD.search(bare)) is not None:
            self.bad(
                f"{at}: {m.group(0)!r} is about fuel or shooting, which positions cannot show; "
                "cite the tile that shows it or remove it"
            )
        self.readings(item, bare, has_frame)
        if item.roles:
            self.roles(item, bare)
            self.places(item, bare)
        return out

    def readings(self, item: _Item, bare: str, has_frame: bool) -> None:
        """One stretch, one reading: no text calls fine what the ledger charged, or a fault
        what a tile cleared; and a claim that it kept moving needs a tile."""
        spans = self.spans_of(item.cites)

        def hits(a: float, b: float) -> bool:
            return any(_overlap(a, b, s0, s1) for s0, s1 in spans)

        charged = [d.id for d in self.charged_lines if hits(d.t_start, d.t_end)]
        pinned = any(hits(a, b) for a, b in self.pinned)
        for m in CLEAN_WORD.finditer(bare):
            if NEGATED.search(bare[: m.start()]):
                continue
            if charged or pinned:
                what = f"line {', '.join(charged)} is charged" if charged else "a pin is costed"
                self.bad(
                    f"{item.at}: {m.group(0)!r} reads the stretch it cites as fine, but {what} "
                    "there; describe it as the ledger calls it, or cite only what you mean"
                )
                return
            if not has_frame:
                self.bad(f"{item.at}: {m.group(0)!r} needs a cited tile that shows it")
                return
        cleared = [(d.id, why) for d, why in self.cleared_lines if hits(d.t_start, d.t_end)]
        if cleared and not charged:
            for m in FAULT_WORD.finditer(bare):
                if NEGATED.search(bare[: m.start()]):
                    continue
                self.bad(
                    f"{item.at}: {m.group(0)!r} reads the stretch as a driving fault, but "
                    + ", ".join(f"{i} was called {why}" for i, why in cleared)
                )
                return

    def roles(self, item: _Item, bare: str) -> None:
        near = self.labelled(item.cites, near_only=True) | set(item.bumpers_read)
        rel = {_relation(self.ctx.match, t) for t in near if t != self.ctx.match.target_team}
        verdicts = self.episode_roles(item.cites)
        for word, ok, need in (
            (DEFENDER_WORD, "opponent" in rel or bool(verdicts & ATTACK), "an opponent"),
            (PARTNER_WORD, "partner" in rel or "partner" in verdicts, "a partner"),
        ):  # fmt: skip
            for m in word.finditer(bare):
                if NEGATED.search(bare[: m.start()]) or ok:
                    continue
                self.bad(
                    f"{item.at}: {m.group(0)!r} needs {need} established by a cited fact's "
                    "label, a box on a cited tile, a bumper you read, or an overlapping "
                    "defense episode with that verdict"
                )
                break

    def places(self, item: _Item, bare: str) -> None:
        claims = [name for pat, name in PLACE_WORDS if pat.search(bare)]
        points = [name for pat, name in POINT_WORDS if pat.search(bare)]
        if not claims and not points:
            return
        alliance: Alliance = self.ctx.match.target_alliance
        xs: list[float] = []
        xy: list[tuple[float, float]] = []
        near = set[str]()
        for c in item.cites:
            s = self._quiet_span(c)
            if s is None:
                continue
            f = self.by_id.get(c)
            if f is not None:
                data = f.model_dump().get("data")
                near |= _declared_places(data)
                where = getattr(f.data, "where", None)
                if where is not None:
                    xy.append((float(where.x), float(where.y)))
                    xs.append(float(where.x))
                    continue
            pad = 0.5 if c.startswith("frame:") else SPAN_PAD_S
            got = self.ctx.target.near(s[0] - pad, s[1] + pad)
            if c.startswith("frame:") and not len(got):
                return
            xs += [float(v) for v in got[:, 0]]
            xy += [(float(p[0]), float(p[1])) for p in got]
        if not xs and not near:
            return
        for x in xs:
            for dx in (-0.5, 0.0, 0.5):
                near.add(f"{zone_of(x + dx, alliance)} zone")
            if abs(x - CENTER_LINE_X_M) <= CENTRE_NEAR_M:
                near.add("centre line")
            for colour, (x0, x1) in BARRIER_X.items():
                if x0 - BARRIER_NEAR_M <= x <= x1 + BARRIER_NEAR_M:
                    near.add(f"{colour} barrier")
        for x, y in xy:
            if rect_distance_m(HUBS[alliance], x, y) <= HUB_NEAR_M:
                near.add("beside its hub")
            if own_wall_m(x, alliance) <= WALL_NEAR_M:
                near.add("at its wall")
            if guardrail_m(y) <= WALL_NEAR_M:
                near.add("at the guardrail")
        span = f"tracked at x {min(xs):.1f}-{max(xs):.1f} m" if xs else "not tracked"
        for claim in claims:
            if claim not in near:
                self.bad(
                    f"{item.at}: says {claim!r}, but the target was {span} over what it cites "
                    f"and those facts name only: {', '.join(sorted(near)) or 'nothing'}; fix "
                    "the place or cite the tile that shows it"
                )
        if not xy:
            return
        hub = min(rect_distance_m(HUBS[alliance], x, y) for x, y in xy)
        wall = min(own_wall_m(x, alliance) for x, _ in xy)
        rail = min(guardrail_m(y) for _, y in xy)
        for claim in points:
            if claim not in near:
                self.bad(
                    f"{item.at}: says {claim!r}, but over what it cites the target's centre was "
                    f"at least {_fmt(hub)} m from its hub, {_fmt(wall)} m from its wall and "
                    f"{_fmt(rail)} m from a guardrail (a stop or contact counts by its `where`); "
                    "say where it was from those distances"
                )


ZONE_KEYS = frozenset(
    {"zone", "from_zone", "to_zone", "side", "target_zone", "came_from", "went_to"}
)


def _declared_places(data: object) -> set[str]:
    """Places a fact names itself: its zones (as "own zone", ...) and barriers ("red
    barrier"), anywhere in its data."""
    out: set[str] = set()
    if isinstance(data, dict):
        for k, v in TypeAdapter(dict[str, object]).validate_python(data).items():
            if k in ZONE_KEYS and v in ("own", "neutral", "opponent"):
                out.add(f"{v} zone")
            elif k in ("barrier", "on_barrier") and v in ("red", "blue"):
                out.add(f"{v} barrier")
            else:
                out |= _declared_places(v)
    elif isinstance(data, list):
        for v in TypeAdapter(list[object]).validate_python(data):
            out |= _declared_places(v)
    return out


@dataclass(frozen=True)
class Checked:
    """A checked answer: the answer when it passed, every problem, the bands, the charges
    and every text field resolved (keyed by its location)."""

    answer: GradeAnswer | None
    problems: list[str]
    bands: dict[SkillName, Band | None]
    charges: Charges | None
    resolved: dict[str, str]


def _check_episodes(ck: _Checker, answer: GradeAnswer, sheets: list[FrameSheet]) -> None:
    ctx = ck.ctx
    items = {i.fact: i for i in ctx.led.items}
    driving_spans = [(d.t_start, d.t_end) for d in ctx.led.driving if d.kind != "loaded_trip"]
    costs = episode_costs(answer.episodes, ctx.led.items)
    covered: set[str] = set()
    for k, e in enumerate(answer.episodes):
        at = f"episode {k + 1} ({', '.join(e.facts)})"
        for c in e.cites:
            ck.span(c, at)
        bad_ids = [i for i in e.facts if i not in ck.by_id]
        if bad_ids:
            ck.bad(f"{at}: {bad_ids} are not fact ids")
            continue
        covered |= set(e.facts)
        span = _episode_span(e, ck.by_id)
        assert span is not None
        frames = ck.frames_in(e.cites, *span)
        team = e.other_team
        rel: str = "unknown"
        if e.identified_by == "track_label":
            if team is None or team not in ck.labelled([*e.facts, *e.cites]):
                ck.bad(
                    f"{at}: track_label needs other_team to be a team labelled in its facts "
                    "or boxed on a cited tile"
                )
            else:
                rel = _relation(ctx.match, team)
        elif e.identified_by == "bumper_number":
            if team is None or team not in {s.team for s in ctx.match.roster} or not frames:
                ck.bad(
                    f"{at}: bumper_number needs a roster team and a cited tile within "
                    f"{SPAN_PAD_S:g} s of the episode"
                )
            else:
                rel = _relation(ctx.match, team)
        elif e.identified_by in ("bumper_red", "bumper_blue"):
            if not frames:
                ck.bad(
                    f"{at}: {e.identified_by} needs a cited tile within {SPAN_PAD_S:g} s of "
                    "the episode"
                )
            else:
                rel = (
                    "partner"
                    if e.identified_by == f"bumper_{ctx.match.target_alliance}"
                    else "opponent"
                )
        elif team is not None:
            ck.bad(f"{at}: identified_by none but other_team is set")
        contacts = [f for i in e.facts if isinstance(f := ck.by_id[i], ContactFact)]
        if e.identified_by != "none":
            for f in contacts:
                teams = {o.team for o in f.data.others}
                rels = {o.relation for o in f.data.others if o.team is not None}
                boxed = {
                    m
                    for c in e.cites
                    if (t := ck.frame_t(c) if c.startswith("frame:") else None) is not None
                    and f.t_start - SPAN_PAD_S <= t <= f.t_end + SPAN_PAD_S
                    and (tile := ck.tiles[t].main) is not None
                    for m in tile.teams_near
                }
                fits = (
                    team in teams or rel in rels or (None in teams and team in boxed)
                    if e.identified_by == "track_label"
                    else None in teams or team in teams
                    if e.identified_by == "bumper_number"
                    else None in teams or rel in rels
                )
                if not fits:
                    ck.bad(
                        f"{at}: no robot in {f.id} can be {team or e.identified_by} (its robots "
                        f"are {sorted(str(t or 'unlabelled') for t in teams)}); give it its own "
                        "episode"
                    )
        need = {
            "opponent_on_target": "opponent",
            "target_on_opponent": "opponent",
            "partner": "partner",
        }
        if e.verdict in need and rel != need[e.verdict]:
            ck.bad(
                f"{at}: {e.verdict} needs the other robot established as {need[e.verdict]} "
                f"(it is {rel}: from {e.identified_by}); otherwise use could_not_tell"
            )
        if e.verdict != "could_not_tell" and not frames:
            ck.bad(f"{at}: {e.verdict} needs a cited tile within {SPAN_PAD_S:g} s of the episode")
        long = [i.fact for i in (items.get(x) for x in e.facts) if i and i.t_end - i.t_start >= 1.0]
        if long and not frames and len(sheets) < IMAGE_CAP:
            ck.bad(
                f"{at}: {', '.join(long)} lasted a second or more; open and cite a tile within "
                f"{SPAN_PAD_S:g} s of it before deciding (could_not_tell without a tile is only "
                "allowed once the sheet budget is used up)"
            )
        if e.verdict == "apart" or (e.verdict == "could_not_tell" and len(sheets) < IMAGE_CAP):
            for f in contacts:
                d = f.data
                if not (rel == "opponent" or any(o.relation == "opponent" for o in d.others)):
                    continue
                why = (
                    "the target was in its own zone"
                    if d.target_zone == "own" or d.where.zone == "own"
                    else f"they came within {_fmt(d.min_distance_m)} m (touching)"
                    if d.min_distance_m < TOUCH_M
                    else f"the target was still {_fmt(d.longest_still_s)} s beside it"
                    if d.longest_still_s >= APART_STILL_S
                    else "it happened during a driving ledger line"
                    if any(_overlap(f.t_start, f.t_end, a, b) for a, b in driving_spans)
                    else None
                )
                if why is not None:
                    ck.bad(
                        f"{at}: {f.id} cannot be {e.verdict}: an opponent is established in it and "
                        f"{why}. An opponent there is on the target: open a tile and use "
                        "opponent_on_target (cost_s 0 when it cost nothing) or target_on_opponent"
                    )
        if e.verdict != "opponent_on_target" and e.cost_s:
            ck.bad(f"{at}: cost_s must be 0 unless the verdict is opponent_on_target")
        mine = [items[i] for i in e.facts if i in items]
        away = [i for i in mine if not i.home_active]
        if e.cost_s and mine and not away:
            ck.bad(
                f"{at}: cost_s must be 0: the target was home in its own active time, where only "
                "its pin time costs (code computes it from pin_s and while_held)"
            )
        elif away and len(away) < len(mine):
            most = sum(i.t_end - i.t_start for i in away)
            if e.cost_s > most + 1e-6:
                ck.bad(
                    f"{at}: cost_s {e.cost_s:g} is more than the {_fmt(most)} s of its items away "
                    "from home or outside its active time"
                )
        if e.cost_s > span[1] - span[0] + SPAN_PAD_S:
            ck.bad(f"{at}: cost_s {e.cost_s:g} is longer than the episode's facts span")
        if costs[k].held > 0 and e.while_held is None:
            ck.bad(
                f"{at}: its items hold {_fmt(costs[k].held)} s still beyond the allowance; set "
                "while_held from the tiles "
                "across the still stretch (not_shooting, shooting or cannot_see)"
            )
    ck.episodes = list(answer.episodes)
    missing = [i.fact for i in ctx.led.items if i.fact not in covered]
    if missing:
        ck.bad(
            f"episodes: these defense items are not accounted for: {', '.join(missing)}; put each "
            "in an episode (could_not_tell is allowed)"
        )


def _check_call(
    ck: _Checker,
    c: LineCall,
    line: DrivingLine | ReadinessLine,
    opp_spans: list[tuple[float, float]],
    sheets: list[FrameSheet],
) -> None:
    at = f"line call {c.line}"
    ctx = ck.ctx
    for x in c.cites:
        ck.span(x, at)
    trip = isinstance(line, DrivingLine) and line.kind == "loaded_trip"
    allowed = (
        TRIP_CALLS if trip else BARRIER_CALLS if isinstance(line, DrivingLine) else READINESS_CALLS
    )
    if c.call not in allowed:
        ck.bad(f"{at}: {c.call} is not a call for this line; use one of {sorted(allowed)}")
        return
    a, b = (
        (line.t_start, line.t_end)
        if isinstance(line, DrivingLine)
        else (line.t, line.t + line.late_s)
    )
    frames = ck.frames_in(c.cites, a, b, pad=0.05 if trip else SPAN_PAD_S)
    corner = [t for t in ck.frames_in(c.cites, a, b) if ck.corner_only(t)]
    if corner and c.corner_bumper != ctx.match.target_team:
        ck.bad(
            f"{at}: it cites corner tile(s) {', '.join(f'frame:{t:.1f}' for t in corner)}; a "
            "corner tile counts only where you read the target's bumper number there: set "
            f"corner_bumper to {ctx.match.target_team} if you read it, else do not rely on it"
        )
    if c.heaped_at is not None and c.call != "heaped":
        ck.bad(f"{at}: heaped_at is only for heaped")
    if c.charged_s is not None and (trip or not isinstance(line, DrivingLine)):
        ck.bad(f"{at}: charged_s is only for charged barrier lines")
    if isinstance(line, DrivingLine) and trip:
        seen = [
            t
            for t in frames
            if ((m := ck.tiles[t].main) is not None and m.tracked) or ck.corner_only(t)
        ]
        if c.call == "heaped":
            t = c.heaped_at
            if t is None or not any(abs(t - s) <= FRAME_MATCH_S for s in seen):
                ck.bad(
                    f"{at}: heaped needs heaped_at = the time of a cited tile inside the trip "
                    f"({a:.1f}-{b:.1f} s) where the target is ringed (or read on a corner tile) "
                    "and its hopper is heaped"
                )
        elif c.call == "not_heaped":
            late = line.look_at[-1] - 1.0
            if not any(t >= late for t in seen):
                ck.bad(
                    f"{at}: not_heaped needs a cited tile with the target ringed at or after "
                    f"{late:.1f} s (its last hopper tile is {line.look_at[-1]:.1f} s): a robot "
                    "can fill up out there. If no late tile shows the hopper, say cannot_see"
                )
        elif c.call == "cannot_see" and not frames and len(sheets) < IMAGE_CAP:
            ck.bad(f"{at}: cannot_see needs the hopper tiles you looked at cited (inside the trip)")
        elif c.call == "frame_contradicts" and not frames:
            ck.bad(f"{at}: frame_contradicts needs a cited tile inside the trip")
        return
    if c.call == "partner" and isinstance(line, DrivingLine):
        stops = [
            (f.t_start, f.t_end) for i in line.facts if isinstance(f := ck.by_id.get(i), StopFact)
        ] or [(line.t_start, line.t_end)]
        held = [
            i
            for i in line.partner_contacts
            if any(_overlap(ck.by_id[i].t_start, ck.by_id[i].t_end, a0, b0) for a0, b0 in stops)
        ]
        if not held:
            ck.bad(
                f"{at}: partner needs a partner contact fact while it was stopped "
                f"({', '.join(f'{a0:.1f}-{b0:.1f}' for a0, b0 in stops)} s); a partner nearby "
                "is not in the way"
            )
    if c.call in ("partner", "frame_contradicts") and not frames:
        ck.bad(f"{at}: {c.call} needs a cited tile within {SPAN_PAD_S:g} s of {a:.1f}-{b:.1f} s")
    if c.call == "opponent" and not any(
        _overlap(a - SPAN_PAD_S, b + SPAN_PAD_S, s0, s1) for s0, s1 in opp_spans
    ):
        ck.bad(f"{at}: opponent needs an opponent_on_target episode overlapping the line")
    if c.call == "charged" and isinstance(line, DrivingLine) and not frames:
        why = [
            *([f"partly inferred ({_fmt(line.unseen_s)} s unseen)"] if line.unseen_s > 0 else []),
            *(["an upper bound"] if line.bound else []),
            *(["a corner reach only"] if line.corner_reach else []),
        ]
        if why:
            ck.bad(
                f"{at}: {line.id} is {' and '.join(why)}; charge it only with a cited tile in its "
                "span that shows the stall, else call it uncertain (or frame_contradicts)"
            )
    if c.charged_s is not None and isinstance(line, DrivingLine):
        if c.call != "charged":
            ck.bad(f"{at}: charged_s is only for charged lines")
        elif (
            not line.bound
            or c.charged_s > line.lost_s
            or not any(abs(t - line.t_start - c.charged_s) <= 0.6 for t in frames)
        ):
            ck.bad(
                f"{at}: charged_s is only for a line whose cost is an upper bound (arrival first "
                "seen after a gap): the seconds from the line's start to a cited tile that shows "
                f"the target home, at most lost_s ({_fmt(line.lost_s)}). Otherwise charge the "
                "line in full, or call it uncertain"
            )


def check_answer(raw: object, ctx: GradeContext, sheets: list[FrameSheet]) -> Checked:
    """Validate the agent's answer and compute the charges and bands it implies."""
    try:
        answer = GradeAnswer.model_validate(raw)
    except ValidationError as e:
        problems = [
            f"schema: {'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()
        ]
        return Checked(None, problems, {}, None, {})
    ck = _Checker(ctx, sheets)
    led = ctx.led
    resolved: dict[str, str] = {}

    def text(
        at: str, value: str, cites: list[str], bumpers: list[int] | None = None, roles: bool = True
    ) -> None:
        resolved[at] = ck.resolve(_Item(at, value, cites, bumpers or [], roles))

    _check_episodes(ck, answer, sheets)
    calls: dict[str, LineCall] = {}
    for c in answer.line_calls:
        if c.line in calls:
            ck.bad(f"line_calls: {c.line} has more than one call")
        calls[c.line] = c
    for line_id in ck.lines:
        if line_id not in calls:
            ck.bad(f"line_calls: ledger line {line_id} has no call")
    opp_spans = [
        s for e in answer.episodes if e.verdict == "opponent_on_target"
        if (s := _episode_span(e, ck.by_id)) is not None
    ]  # fmt: skip
    for c in answer.line_calls:
        line = ck.lines.get(c.line)
        if line is None:
            ck.bad(f"line call {c.line}: no ledger line {c.line}")
            continue
        _check_call(ck, c, line, opp_spans, sheets)

    # what code charges, then the text checks that lean on it
    ch = charges(led, calls, answer.episodes, ck.by_id, ctx.target, ctx.match.target_alliance)
    for d in led.driving:
        c = calls.get(d.id)
        if c is None:
            continue
        if ch.line.get(d.id, 0.0) > 0:
            ck.charged_lines.append(d)
        elif c.call in ("frame_contradicts", "not_heaped"):
            ck.cleared_lines.append((d, c.call))
    ck.pinned = [
        s
        for e, cost in zip(answer.episodes, ch.episode, strict=True)
        if cost.pin > 0 and (s := _episode_span(e, ck.by_id)) is not None
    ]
    for k, e in enumerate(answer.episodes):
        text(
            f"episode {k + 1} ({', '.join(e.facts)})",
            e.text,
            [*e.facts, *e.cites],
            [e.other_team] if e.identified_by == "bumper_number" and e.other_team else [],
        )
    for c in answer.line_calls:
        line = ck.lines.get(c.line)
        if line is not None:
            own = line.facts if isinstance(line, DrivingLine) else [line.fact]
            text(f"line call {c.line}", c.text, [*c.cites, *own])

    bands: dict[SkillName, Band | None] = {
        "driving": driving_band(ch),
        "shift_readiness": readiness_band(led.readiness, calls, ch),
        "handling_defense": defense_band(answer.episodes, ch),
    }
    names = [g.skill for g in answer.skills]
    if sorted(names) != sorted(SKILLS):
        ck.bad(f"skills: need exactly one each of {list(SKILLS)}, got {names}")
    for g in answer.skills:
        at = f"skill {g.skill}"
        for c in g.cites:
            ck.span(c, at)
        band = bands[g.skill]
        null_ok = band is None or (
            g.skill == "driving" and ctx.teleop_tracked_share < MIN_TRACKED_SHARE
        )
        if g.score is None and not null_ok:
            ck.bad(
                f"{at}: null is not allowed; the rubric band from your calls and episodes is {band}"
            )
        elif g.score is not None and band is None:
            ck.bad(
                f"{at}: must be null: "
                + (
                    "there is no hub_on fact"
                    if g.skill == "shift_readiness"
                    else "no episode is opponent_on_target"
                )
            )
        elif g.score is not None and band is not None and not band[0] <= g.score <= band[1]:
            ck.bad(
                f"{at}: score {g.score} is outside the band {band[0]}-{band[1]} the rubric "
                "gives for your calls and episodes"
                + (
                    f" ({_fmt(ch.driving_s)} charged s"
                    + (
                        "; a hopper you could not see holds it to the lower score"
                        if ch.blind
                        else ""
                    )
                    + ")"
                    if g.skill == "driving"
                    else ""
                )
            )
        text(f"{at} summary", g.summary, g.cites, roles=False)
        if g.null_reason is not None:
            text(f"{at} null_reason", g.null_reason, g.cites, roles=False)
    for c in answer.auto.cites:
        ck.span(c, "auto")
    text("auto", answer.auto.text, answer.auto.cites, roles=False)
    for k, n in enumerate(answer.notes):
        at = f"note {k + 1} (t={n.t})"
        found = [s for c in n.cites if (s := ck.span(c, at)) is not None]
        if found and not any(a - NOTE_REACH_S <= n.t <= b + NOTE_REACH_S for a, b in found):
            ck.bad(f"{at}: t is more than {NOTE_REACH_S:g} s from everything it cites")
        if n.kind == "played_through_defense" and not any(c.startswith("frame:") for c in n.cites):
            ck.bad(f"{at}: played_through_defense needs a frame citation")
        spans = [s for c in n.cites if (s := ck._quiet_span(c)) is not None]  # pyright: ignore[reportPrivateUsage]
        hit = [
            d.id
            for d in ck.charged_lines
            if any(_overlap(d.t_start, d.t_end, a, b) for a, b in spans)
        ]
        if n.kind in ("went_well", "played_through_defense") and hit:
            ck.bad(f"{at}: kind {n.kind} over {', '.join(hit)}, which is charged")
        cleared = [
            i.id
            for i, _ in ck.cleared_lines
            if any(_overlap(i.t_start, i.t_end, a, b) for a, b in spans)
        ]
        if n.kind == "went_wrong" and cleared and not hit:
            ck.bad(f"{at}: kind went_wrong over {', '.join(cleared)}, which a tile cleared")
        text(at, n.text, n.cites, n.bumpers_read)
    problems = ck.problems
    return Checked(answer if not problems else None, problems, bands, ch, resolved)


def _retry_prompt(problems: list[str]) -> str:
    return (
        "Your answer was rejected by the checker. Fix exactly these problems and return the "
        "whole answer again (you may open more frames within the budget):\n"
        + "\n".join(f"- {p}" for p in problems)
    )


# --- the stage --------------------------------------------------------------------------


def _usage(results: list[JSON]) -> tuple[int, int, int, int, float | None, int, list[str]]:
    """Summed input, output, cache read, cache write tokens, the cost (the last result's
    total: the retry resumes the same session), turns and models."""
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
            cost = float(c)  # a resumed session reports its running total
        models += [m for m in _as_dict(r.get("modelUsage")) if m not in models]
    return ins, outs, reads, writes, cost, turns, models


def _int(v: object) -> int:
    return int(v) if isinstance(v, int | float) else 0


def grade(run: RunDir) -> None:
    exe = _claude()
    model = run.settings().grader_model
    ctx = grade_context(run)
    match = ctx.match
    work = _work(run)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    home = _home_view(Camera.load(run.calibration_json, "main"), match)
    system, task = _prompts(match, run.facts_json.read_text(), ctx.led, home)
    (work / "system_prompt.md").write_text(system)
    (work / "task_prompt.md").write_text(task)
    start = time.perf_counter()

    first = _run_claude(
        _command(exe, model, run, system, None), task, run.path, work / "session-1.jsonl"
    )
    results = [first.result]
    session = first
    rejection: list[str] | None = None
    for attempt in range(1, ATTEMPTS + 1):
        sheets = _served(run)
        checked = check_answer(session.answer, ctx, sheets)
        if checked.answer is not None:
            break
        rejection = rejection or checked.problems
        (work / f"rejection-{attempt}.json").write_text(
            json.dumps({"answer": session.answer, "problems": checked.problems}, indent=2)
        )
        if attempt == ATTEMPTS:
            raise RuntimeError(
                f"grader answer rejected {ATTEMPTS} times; no grade written:\n"
                + "\n".join(checked.problems)
            )
        session = _run_claude(
            _command(exe, model, run, system, session.session_id),
            _retry_prompt(checked.problems),
            run.path,
            work / f"session-{attempt + 1}.jsonl",
        )
        results.append(session.result)
    else:
        raise AssertionError("unreachable")
    answer, resolved, ch = checked.answer, checked.resolved, checked.charges
    assert ch is not None

    ins, outs, reads, writes, cost, turns, models = _usage(results)
    notes: list[Note] = []
    for k, n in enumerate(answer.notes):
        period, clock = _clock(match, n.t)
        text = resolved[f"note {k + 1} (t={n.t})"]
        notes.append(
            Note.model_validate(n.model_dump() | {"text": text, "clock": clock, "period": period})
        )
    notes.sort(key=lambda n: n.t)
    skills = [
        ScoredSkill.model_validate(
            g.model_dump()
            | {
                "summary": resolved[f"skill {g.skill} summary"],
                "null_reason": resolved.get(f"skill {g.skill} null_reason"),
                "band": checked.bands[g.skill],
            }
        )
        for g in sorted(answer.skills, key=lambda g: SKILLS.index(g.skill))
    ]
    result = Grade(
        team=match.target_team,
        rubric_version=_rubric_version(_text("rubric.md")),
        skills=skills,
        driving_ledger=ctx.led.driving,
        readiness_ledger=ctx.led.readiness,
        defense_items=ctx.led.items,
        line_calls=[
            ScoredCall.model_validate(
                c.model_dump()
                | {
                    "text": resolved[f"line call {c.line}"],
                    "charged_total_s": ch.line.get(c.line, 0.0),
                }
            )
            for c in answer.line_calls
        ],
        episodes=[
            ScoredEpisode.model_validate(
                e.model_dump()
                | {
                    "text": resolved[f"episode {k + 1} ({', '.join(e.facts)})"],
                    "pin_cost_s": cost.pin,
                    "total_cost_s": cost.total,
                }
            )
            for k, (e, cost) in enumerate(zip(answer.episodes, ch.episode, strict=True))
        ],
        notes=notes,
        auto=answer.auto.model_copy(update={"text": resolved["auto"]}),
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
    _grade_md(run).write_text(render_md(result, ctx.facts))


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

    def cell(s: str) -> str:
        return s.replace("|", "/")

    lines = [
        f"# Driving grade: team {g.team}",
        "",
        f"Rubric `{g.rubric_version}`. Scores are absolute, 1 to 10, inside the band the rubric "
        "gives from the charged seconds and episode costs. Auto is reported, not scored.",
        "",
        "| Skill | Score | Band |",
        "|---|---|---|",
    ]
    lines += [
        f"| {s.skill} | {s.score if s.score is not None else 'null'} | "
        f"{f'{s.band[0]}-{s.band[1]}' if s.band else 'null'} |"
        for s in g.skills
    ]
    for s in g.skills:
        lines += ["", f"## {s.skill}: {s.score if s.score is not None else 'null'}", ""]
        if s.null_reason:
            lines += [f"No score: {s.null_reason}", ""]
        lines += [s.summary, "", f"Evidence: {cites(s.cites)}"]
    calls = {c.line: c for c in g.line_calls}
    lines += [
        "",
        "## Driving ledger",
        "",
        "| Line | Time | Lost (s) | Call | Charged (s) | Why |",
        "|---|---|---|---|---|---|",
    ]
    for d in g.driving_ledger:
        c = calls.get(d.id)
        lines.append(
            f"| {d.id} {d.kind} | {d.t_start:.1f}-{d.t_end:.1f} | {_fmt(d.lost_s)} | "
            f"{c.call if c else '-'} | {_fmt(c.charged_total_s) if c else '-'} | "
            f"{cell(c.text) if c else ''} |"
        )
    lines += [
        "",
        "## Readiness ledger",
        "",
        "| Line | Phase | Late (s) | Call | Why |",
        "|---|---|---|---|---|",
    ]
    for r in g.readiness_ledger:
        c = calls.get(r.id)
        lines.append(
            f"| {r.id} | {r.phase} | {_fmt(r.late_s)} | {c.call if c else '-'} | "
            f"{cell(c.text) if c else ''} |"
        )
    lines += [
        "",
        "## Defense episodes",
        "",
        "| Facts | Verdict | Other | Cost (s) | What |",
        "|---|---|---|---|---|",
    ]
    for e in g.episodes:
        who = f"{e.other_team or '?'} ({e.identified_by})"
        lines.append(
            f"| {', '.join(e.facts)} | {e.verdict} | {who} | {e.total_cost_s:g} | {cell(e.text)} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "| Clock | t (s) | Kind | Note | Cites |",
        "|---|---|---|---|---|",
    ]
    for n in g.notes:
        lines.append(
            f"| {n.clock} {n.period} | {n.t:.1f} | {n.kind} | {cell(n.text)} | {cites(n.cites)} |"
        )
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
        + ", ".join(f"{s.image} ({s.view}, t {s.tiles[0].t:.1f})" for s in r.sheets),
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
