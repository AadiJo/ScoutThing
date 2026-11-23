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
   (`ledgers`). Each barrier and readiness line gets a default ruling computed from measured
   features (`_turned_back_default`, `_readiness`), and a charged barrier line the part a
   robot standing on its way home cost (`_split`), so a line is decided the same way on every
   run.
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
4. Check the structured answer (`check_answer`): citations, a hopper reading per trip tile
   (code turns them into the trip's ruling, `_hopper_ruling`), each
   overrule of a default backed by the checkable contradiction it names, episodes covering
   every defense item, numbers only as references to fact fields, team names only when
   labelled or read, no purpose words, places that agree with where the target was, and no
   text reading a stretch against its ruling. On failure the session is resumed with the exact
   errors, up to ATTEMPTS answers in all; then the stage fails.
5. Code computes the charged seconds (`charges`) and every score (`scores`); the grader only
   writes why.
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
    Arrival,
    ContactFact,
    CrossingFact,
    CrossingTurnedBackData,
    CrossingTurnedBackFact,
    Fact,
    Facts,
    GapFact,
    GapInference,
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
    Coverage,
    DefenseEpisode,
    DefenseItem,
    DrivingKind,
    DrivingLine,
    FrameSheet,
    Grade,
    GradeAnswer,
    Hopper,
    LineCall,
    Note,
    OpponentSplit,
    PhaseCoverage,
    ReadinessLine,
    ReadinessRuling,
    Ruling,
    RunRecord,
    ScoredCall,
    ScoredEpisode,
    ScoredSkill,
    SkillName,
    TileRead,
    TileRecord,
    Untracked,
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
DRIVING_UNSEEN_MAX = 0.25
"""Driving is withheld when more than this share of its own active time was untracked in gaps
that nothing puts in its own zone, unless what was seen already charges more than
DRIVING_EDGES[1]. Charges come from barrier passes and trips out of its zone, and only unseen
time away from home can hide them. On the sixteen matches that share is 1 to 21% except two
blind matches (33%, 51%); a quarter of 90 s is about 22 s, room for two loaded trips or a stall
and a reroute, enough to move the score three bands."""
PHASE_UNSEEN_MAX = 0.5
"""An own-active phase more than this share unseen (as above) flags the driving score: a whole
shift can be blind in a match that is well tracked overall."""
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
STALL_MPS = 0.8
BACKED_OFF_M = 3.0
"""A turned-back attempt is charged by default when its `slowest_mps` is at most STALL_MPS or
it went at least BACKED_OFF_M back from the barrier (`furthest_from_barrier_m`); otherwise it
is a drive past. On the development matches the 12 frame- or film-checked stalls in active time
read at most 0.36 m/s, except one that backed off 4.65 m; the one drive past read 1.26 m/s and
went 1.38 m back. STALL_MPS sits between them (0.44 and 0.46 m/s of margin) at the facts'
below-driving-pace speed; BACKED_OFF_M halfway (1.6 m either side). Only one drive past was in
active time, so the cleared side rests on little evidence."""
STALLED_S = 1.0
"""A turned-back attempt is called a stall only when the robot was measured stopped on the band:
its slowest speed under the facts' STOP_MPS, or stopped (or unseen, so possibly stopped) this
long. The audit's frame-checked turn-away without a stall read 0.46 m/s and 0 s stopped; the
human's stalls read 0.01 to 0.36 m/s (one, at 0.36, also stopped 1.1 s)."""
AHEAD_M = 0.5
"""A robot in a contact inside a charged barrier line is in the target's way when, while within
the facts' CONTACT_M, its centre came at least this much nearer the target's own wall than the
target's (half a robot: it stood on the target's way home, not beside or behind it). Its split
(`_split`) is the line's time after it came close beyond the time to get home from there
(`_home_allow`), at most the contact's length; all of it from then when it came close before
the target left the barrier. On the sixteen matches four charged lines had contacts with a
robot not known to be a partner: the opponent the film study shows cutting in on the way home
read 1.51 m (split 3.2 s); an opponent that met it as it reached home 0.30 m, an opponent
behind it -0.22 m, unlabelled robots 0.65 m (0.1 s long) and 1.57 m (0.4 s before it got
home). All but the first cost 0 s through the time home or the contact's length anyway, so
the threshold has a 1.0 m margin to the one positive and is not pinned from below."""
HOME_INFERENCES: frozenset[GapInference] = frozenset(
    {"in_own_zone_beyond_barrier", "entered_own_zone", "own_zone_both_sides"}
)
"""Gap inferences that put an untracked robot in its own zone (with `stayed_in_place` when it
was last seen there)."""
TRIP_ALLOW_S = 3.0
HOME_CROSS_S = 1.2
HOME_MPS = 2.0
"""Seconds a heaped robot out of its own zone is allowed to get home before the rest of its
trip is charged: a normal crossing plus its distance outside its zone at HOME_MPS (robots
drive home at 2 to 2.5 m/s), and never less than TRIP_ALLOW_S."""
TRIP_GRACE_S = 3.0
"""Seconds of a heaped trip never charged on top of the time to get home: hopper reads are
right about four times in five, so a remainder of a few seconds rests on one doubtful tile.
Subtracted, not a cut-off, so moving `heaped_at` one tile moves the charge by that step only."""
LOOK_TILES = 4
LOOK_FIRST_S = 1.0
LOOK_STEP_S = 1.0
"""Hopper tiles of a trip: up to LOOK_TILES, at least LOOK_STEP_S apart, from LOOK_FIRST_S
after it left to the last moment from which a heaped hopper could still cost MIN_LOST_S
(`_look_times`). A trip with no such moment is no ledger line: a later heap costs nothing, so
reading it could only flip a ruling that charges 0 s. On the sixteen development matches that
drops 15 of 28 trips, 6 of the 8 whose rulings flipped between two saved gradings (5 of them
charged 0 s either way, one 0.24 s from a heap on its first tile)."""
HEAPED_TILES = 2
"""A trip is heaped only when at least this many readable tiles in a row, through its last
readable one, are heaped: a heap on one tile rests on one reading. On the development matches a
lone heap on a far-side last tile (fuel piled on the floor behind the robot at the hopper's
height) charged 1.1 s in two gradings of a trip the film study logs as part full; the one heaped
trip the film study calls wasted read heaped on three tiles in both. A trip with fewer hopper
tiles than this is no ledger line."""
PIN_ALLOW_S = 3.0
"""Still seconds beside an opponent an episode may hold before they cost (pinning longer
than 3 s is a foul in this game)."""
PIN_GAP_S = 2.0
"""Opponent-held items this close in time are one hold: they share one allowance."""
PIN_SHARE = 0.5
"""Share of held still time beyond the allowance that costs. Whether it kept shooting while
held is rarely readable on a tile (the grader said `cannot_see` on almost every hold), so
code charges half instead of asking."""
MISPLACED_TILES = 2
"""`misplaced` needs this many cited tiles in the line's span with the target ringed."""
NOT_HOME_S = 1.0
"""`not_home` needs a cited tile this close to the activation."""
INTENT_WORD = re.compile(
    r"\b(attempt\w*|intend\w*|intent\w*|meant|chose|choice|choos\w*|route choice|lane change|"
    r"rerout\w*|drive past|drove past|driving past|instead)\b",
    re.IGNORECASE,
)
"""Words reading what the robot meant to do: refused in an overrule, whose grounds must be
what a tile shows."""
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
    "Yellow arc under it: target. Boxes: robots the tool tracked (team PARTNER/OPP, or "
    "'unlabelled'). A robot with NO box was not tracked."
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


def _look_times(a: float, b: float, target: _Target, alliance: Alliance) -> list[float]:
    """Hopper tile times for a trip from a to b, evenly spaced (see LOOK_STEP_S); empty when a
    heap seen at any moment of it could not cost MIN_LOST_S (`charges`, before excused time),
    or fewer than HEAPED_TILES tiles fit."""
    first = round(a + LOOK_FIRST_S, 1)
    last = round(b - TRIP_ALLOW_S - TRIP_GRACE_S - MIN_LOST_S, 1)
    while last >= first and b - last - _home_allow(target, alliance, last) < (
        TRIP_GRACE_S + MIN_LOST_S
    ):
        last = round(last - 0.1, 1)
    n = min(LOOK_TILES, 1 + math.floor((last - first) / LOOK_STEP_S + 1e-9))
    if last < first or n < HEAPED_TILES:
        return []
    step = round((last - first) / (n - 1), 1)
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
    default: Ruling = "charged"
    why: str = "on the barrier in its own active time"
    bound: bool = False
    look: tuple[float, ...] = ()
    left: float | None = None
    """turned_back: when it left the barrier back to its own side; other barrier lines are on
    the barrier throughout."""


@dataclass(frozen=True)
class _Tracks:
    """What `_split` measures positions on."""

    target: _Target
    others: _Others
    alliance: Alliance
    near_m: float
    """The facts' CONTACT_M."""

    def ahead_m(self, f: ContactFact, o: OtherRobot) -> float | None:
        """How much nearer the target's own wall than the target's the other robot's centre
        came while within `near_m` of it in contact `f` (negative: never level with it); None
        when the two were never tracked together that close."""
        tg, ot, al = self.target, self.others, self.alliance
        sel = (ot.track == o.track_id) & (ot.t >= f.t_start - 1e-6) & (ot.t <= f.t_end + 1e-6)
        best: float | None = None
        for t, xy in zip(ot.t[sel], ot.xy[sel], strict=True):
            if not len(tg.t):
                break
            k = int(np.argmin(np.abs(tg.t - t)))
            p = tg.xy[k]
            if abs(tg.t[k] - t) > TRACKED_WITHIN_S or float(np.hypot(*(xy - p))) > self.near_m:
                continue
            ahead = own_wall_m(float(p[0]), al) - own_wall_m(float(xy[0]), al)
            best = ahead if best is None else max(best, ahead)
        return best


def _split(r: _Raw, fs: Sequence[Fact], tr: _Tracks) -> OpponentSplit | None:
    """The part of a charged barrier line a robot in its way cost (see AHEAD_M): from the
    earliest contact inside the line with a robot not known to be a partner that came AHEAD_M
    nearer its own wall than the target, the line's time beyond what getting home from there
    takes (`_home_allow`), at most the contact's length; when it came close before the target
    left the barrier, all of the line from then. A labelled opponent's split is preferred over
    an unlabelled robot's; splits of no seconds are dropped."""
    left = r.b if r.left is None else r.left
    found: list[tuple[bool, float, OpponentSplit]] = []
    for f in fs:
        if not (isinstance(f, ContactFact) and _overlap(f.t_start, f.t_end, r.a, r.b)):
            continue
        for o in f.data.others:
            if o.relation == "partner":
                continue
            ahead = tr.ahead_m(f, o)
            t = max(r.a, o.first_close_t)
            if ahead is None or ahead < AHEAD_M or t >= r.b:
                continue
            who = (
                f"{o.team} {o.relation}" if o.team is not None else f"unlabelled track {o.track_id}"
            )
            if t < left:
                secs = r.b - t
                how = "before it left the barrier: all of the line from then"
            else:
                allow = _home_allow(tr.target, tr.alliance, t)
                secs = min(r.b - t - allow, f.t_end - t)
                how = (
                    f"after it left the barrier: the line's {_fmt(r.b - t)} s from then less "
                    f"{_fmt(allow)} s to get home from there, at most the contact's "
                    f"{_fmt(f.t_end - t)} s"
                )
            secs = round(min(secs, r.lost), 2)
            if secs <= 0:
                continue
            opp = o.relation == "opponent"
            why = (
                f"{f.id}: {who} came {_fmt(ahead)} m nearer its own wall than the target (in its "
                f"way at {AHEAD_M:g} m or more) at {t:.1f} s, {how}: {_fmt(secs)} s"
            )
            found.append((not opp, t, OpponentSplit(
                contact=f.id, team=o.team if opp else None, t=round(t, 2), seconds=secs, why=why
            )))  # fmt: skip
    return min(found, key=lambda x: (x[0], x[1]))[2] if found else None


def _stalled(d: CrossingTurnedBackData, stop_mps: float) -> bool:
    """Measured stopped on the band (see STALLED_S)."""
    slow = d.slowest_mps
    return (slow is not None and slow < stop_mps) or d.stopped_s + d.unseen_s >= STALLED_S


def _turned_back_default(d: CrossingTurnedBackData, stop_mps: float) -> tuple[Ruling, str]:
    """A turned-back attempt is charged when it slowed on the band or went well back from the
    barrier afterwards; a drive past keeps its pace and stays close. The reason names the
    measure that fired, and calls it a stall only when the robot was measured stopped."""
    slow, far = d.slowest_mps, d.furthest_from_barrier_m
    kind = (
        f"stopped {_fmt(d.stopped_s)} s on the band: a stall"
        if _stalled(d, stop_mps)
        else f"never under {stop_mps:g} m/s there and stopped {_fmt(d.stopped_s)} s: a slow "
        "turn-away, not a stall"
    )
    if slow is None or slow <= STALL_MPS:
        what = (
            "no speed could be read on the band"
            if slow is None
            else (f"slowed to {_fmt(slow)} m/s on the band")
        )
        return "charged", f"{what} (charged at or under {STALL_MPS:g} m/s); {kind}"
    if far is not None and far >= BACKED_OFF_M:
        return "charged", (
            f"went {_fmt(far)} m back from the barrier (charged at {BACKED_OFF_M:g} m or more), "
            f"slowest {_fmt(slow)} m/s: a back-off or turn-away, not a stall"
        )
    return "cleared", (
        f"drive past: slowest {_fmt(slow)} m/s (over {STALL_MPS:g}) and at most "
        f"{_fmt(far or 0.0)} m back from the barrier (under {BACKED_OFF_M:g})"
    )


def _arrival_text(a: Arrival, t0: float) -> str:
    """How an arrival is known, for a ledger line's text."""
    if a.by == "seen":
        return "seen arriving"
    if a.by == "after_gap":
        return "first tracked there after a gap: at most that"
    how = f"{a.fact} saw it crossing" if a.by == "crossing" else f"{a.fact}'s inference"
    return (
        f"estimated: {how}, plus the rest of the way at the speed it was going; first tracked "
        f"there {_fmt(a.first_tracked_t - t0)} s after"
    )


def _home_gap(g: GapFact) -> bool:
    """A gap whose inference puts the untracked robot in its own zone."""
    d = g.data
    return d.inference in HOME_INFERENCES or (
        d.inference == "stayed_in_place" and d.before is not None and d.before.zone == "own"
    )


def _readiness(fs: Sequence[Fact], match: MatchInfo) -> list[ReadinessLine]:
    """One line per `hub_on` fact, with its default: tracked in its own zone, untracked inside
    a gap whose inference puts it there, or shown home by then by a crossing or inference
    (`reached`), is cleared. Untracked with its last sighting in its own zone and nothing
    putting it home, it is `not_judged`: it may have stayed or left, and positions cannot
    say which. Otherwise its lateness is charged (at most `late_s` when first seen home after
    a gap)."""
    gaps = [f for f in fs if isinstance(f, GapFact)]
    out: list[ReadinessLine] = []
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
        bound = bool(not d.in_own_zone and d.reached and d.reached.by == "after_gap")
        gap = next((g for g in gaps if g.t_start <= f.t_start <= g.t_end), None)
        last = d.last_seen
        default: ReadinessRuling
        if d.in_own_zone:
            default, why = "cleared", "tracked in its own zone"
        elif d.in_own_zone is None and gap is not None and _home_gap(gap):
            default = "cleared"
            why = f"untracked in {gap.id}, whose inference {gap.data.inference} puts it home"
        elif d.reached is not None and d.reached.by in ("crossing", "inference") and late == 0:
            default = "cleared"
            why = f"untracked; {d.reached.fact} puts it home by the activation"
        elif d.in_own_zone is None and (last is None or last.zone == "own"):
            default = "not_judged"
            why = (
                "untracked, last seen in its own zone "
                f"{_fmt(f.t_start - last.t)} s before and nothing seen since puts it out or home: "
                "where it was is unknown, so it is not judged"
                if last is not None
                else "untracked since the match began: not judged"
            )
        else:
            default = "charged"
            why = (
                "tracked outside its own zone"
                if d.in_own_zone is False
                else f"untracked, last seen outside its own zone {_fmt(f.t_start - last.t)} s "
                "before"
                if last is not None
                else "untracked"
            )
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
            else f"; in its own zone {_fmt(late)} s later"
            + (f" ({_arrival_text(d.reached, f.t_start)})" if d.reached is not None else "")
        )
        out.append(
            ReadinessLine(
                id=f"R{len(out) + 1}",
                fact=f.id,
                phase=d.phase,
                t=f.t_start,
                in_zone=d.in_own_zone,
                late_s=round(late, 2),
                late_is_bound=bound,
                never_home=never,
                default=default,
                why=why,
                text=f"its hub turned on for {d.phase}: {where}{arrive}",
            )
        )
    return out


def ledgers(facts: Facts, match: MatchInfo, target: _Target, others: _Others) -> Ledgers:
    """The driving and readiness ledgers and the defense items, from the facts; the tracks
    only place a robot in a contact against the target's way home (`_split`)."""
    fs = facts.facts
    teleop = match.teleop_start_t - 1e-6
    active = _own_active(match)
    contact_m = facts.thresholds.get("CONTACT_M", 1.6)
    stop_mps = facts.thresholds.get("STOP_MPS", 0.3)
    readiness = _readiness(fs, match)
    late = [(r.t, r.t + r.late_s) for r in readiness if r.default == "charged" and r.late_s > 0]
    raw: list[_Raw] = []

    def taken(a: float, b: float) -> bool:
        return any(_overlap(a, b, r.a, r.b) for r in raw)

    def active_s(own_active: float, a: float, b: float) -> float:
        """A turned-back attempt's own-active seconds less the lateness a readiness line
        already charges: an attempt in inactive time that got it home late is a readiness
        fault, charged once."""
        return max(0.0, own_active - _union_s(a, b, late))

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
            f"in the {d.far_side} zone {_fmt(d.seconds_to_far_side)} s after entering "
            f"({_arrival_text(d.arrived, f.t_start)})"
            if d.arrived is not None and d.seconds_to_far_side is not None
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
        default, why = _turned_back_default(d, stop_mps)
        raw.append(
            _Raw(
                f.t_start, f.t_end, "turned_back", [f.id, *(s.id for s in folded)],
                active_s(d.own_active_s, f.t_start, f.t_end), d.unseen_s, text, default, why,
                bound=bool(d.arrived and d.arrived.by == "after_gap")
                and d.cost_end == "reached_far_side",
                left=f.t_start + d.time_on_barrier_s,
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
            lost = min(excess, d.own_active_s)
            raw.append(_Raw(f.t_start, f.t_end, "barrier_stay", [f.id], lost, d.untracked_s, text))
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
        lost = min(d.excess_s, d.own_active_s)
        raw.append(_Raw(f.t_start, f.t_end, "slow_crossing", [f.id], lost, 0.0, text))
    raw = [r for r in raw if r.lost >= MIN_LOST_S]
    for f in fs:
        if not isinstance(f, TripFact):
            continue
        look = tuple(_look_times(f.t_start, f.t_end, target, match.target_alliance))
        if not look:
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
        raw.append(
            _Raw(f.t_start, f.t_end, "loaded_trip", [f.id], d.duration_s, d.unseen_s, text,
                 "cleared", "no hopper reading yet", look=look)
        )  # fmt: skip
    driving: list[DrivingLine] = []
    tracks = _Tracks(target, others, match.target_alliance, contact_m)
    for n, r in enumerate(sorted(raw, key=lambda r: r.a), 1):
        opp, partner = _contacts_in(fs, r.a, r.b)
        split = (
            _split(r, fs, tracks) if r.kind != "loaded_trip" and r.default == "charged" else None
        )
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
                default=r.default,
                why=r.why,
                look_at=list(r.look),
                opponent_contacts=opp,
                split=split,
                partner_contacts=partner,
                text=r.text,
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


def _ledger_text(
    led: Ledgers, match: MatchInfo, home: Corner, by_id: dict[str, Fact]
) -> tuple[str, str, str]:
    """The driving ledger, readiness ledger and defense items as prompt text. Lines that
    depend on a moment the target was untracked point at the corner view of its own end."""

    def span(a: float, b: float) -> str:
        return f"{a:.1f}-{b:.1f} s ({_clock(match, a)[1]}, {_phase(match, a)})"

    corner = f"untracked for part of it: look in the {home} view (its own end) too"

    def drive_line(d: DrivingLine) -> str:
        flags = [
            *(["lost_s is an upper bound"] if d.bound else []),
            *([corner] if d.unseen_s >= 1.0 or d.bound else []),
        ]
        rule = (
            f"HOPPER TILES {', '.join(f'{t:.1f}' for t in d.look_at)} (frames t="
            f"{d.look_at[0]:.1f} count {len(d.look_at)} step_s "
            f"{(d.look_at[1] - d.look_at[0]) if len(d.look_at) > 1 else 1.0:.1f} zoom 2): read "
            "each one"
            if d.look_at
            else f"DEFAULT {d.default} ({d.why})"
        )
        s = d.split
        if s is not None:
            rule += (
                f"; OPPONENT SPLIT {_fmt(s.seconds)} s of it count against handling_defense "
                f"({s.why}): {s.contact}'s episode is opponent_on_target"
                if s.team is not None
                else f"; UNLABELLED SPLIT {_fmt(s.seconds)} s ({s.why}): they count against "
                f"handling_defense only if a tile identifies that robot as an opponent (call "
                f"opponent)"
            )
        return (
            f"- {d.id} {d.kind} {span(d.t_start, d.t_end)}: {rule}; lost_s {_fmt(d.lost_s)}, "
            f"unseen_s {_fmt(d.unseen_s)}{''.join(f' ({x})' for x in flags)}; facts "
            f"{', '.join(d.facts)}; {d.text}; contacts with opponent or unlabelled robots: "
            f"{', '.join(d.opponent_contacts) or 'none'}; with partners only: "
            f"{', '.join(d.partner_contacts) or 'none'}"
        )

    drive = "\n".join(drive_line(d) for d in led.driving)
    ready = "\n".join(
        f"- {r.id} {r.fact} at {r.t:.1f} s ({_clock(match, r.t)[1]}): DEFAULT {r.default} "
        f"({r.why}); late_s {_fmt(r.late_s)}{' (bound)' if r.late_is_bound else ''}; {r.text}"
        + (f" ({corner})" if r.in_zone is None or r.late_is_bound else "")
        for r in led.readiness
    )
    spans = [(d.t_start, d.t_end) for d in led.driving if d.kind != "loaded_trip"]

    def pressed(fact: str) -> str:
        f = by_id.get(fact)
        if not isinstance(f, ContactFact) or not any(
            o.relation == "opponent" for o in f.data.others
        ):
            return ""
        why = _pressed(f, spans)
        return f", OPPONENT ON IT ({why}: opponent_on_target or target_on_opponent)" if why else ""

    items = "\n".join(
        f"- {i.fact} {span(i.t_start, i.t_end)}{', HOME IN ACTIVE TIME' if i.home_active else ''}"
        f"{f', pin_s {_fmt(i.pin_s)}' if i.pin_s > 0 else ''}{pressed(i.fact)}: {i.text}"
        for i in led.items
    )
    return drive or "(no lines)", ready or "(no lines)", items or "(no items)"


# --- costs and scores -------------------------------------------------------------------


DRIVING_EDGES = (0.5, 3.0, 6.0, 12.0)
READINESS_EDGES = (1.0, 3.0, 8.0)
DEFENSE_EDGES = (1.0, 4.0, 8.0, 15.0)
"""Band edges in seconds: 9-10 at or under the first, one band lower past each. Set from the
spread of charged seconds over the development matches (see rubric.md)."""


def _band_of(x: float, edges: Sequence[float]) -> Band:
    """9-10 at or below edges[0], then one band lower past each edge."""
    k = sum(x > e + 1e-9 for e in edges)
    return (9 - 2 * k, 10 - 2 * k)


def _score(x: float, edges: Sequence[float]) -> int:
    """The higher score of x's band in the lower half of its span, else the lower one. The
    first band spans 0 to edges[0]; the last runs to twice the last edge."""
    lo, hi = _band_of(x, edges)
    k = (10 - hi) // 2
    a = 0.0 if k == 0 else edges[k - 1]
    b = edges[k] if k < len(edges) else 2 * edges[-1]
    return hi if x <= (a + b) / 2 + 1e-9 else lo


def _episode_span(e: DefenseEpisode, by_id: dict[str, Fact]) -> tuple[float, float] | None:
    spans = [(f.t_start, f.t_end) for i in e.facts if (f := by_id.get(i)) is not None]
    if not spans:
        return None
    return (min(a for a, _ in spans), max(b for _, b in spans))


@dataclass(frozen=True)
class Charges:
    """What code charges from the defaults, the calls and the episodes."""

    line: dict[str, float]
    """Seconds per driving line, lateness per readiness line."""
    unjudged: list[str]
    """Readiness lines `not_judged` by default that no call settled: not scored."""
    driving_s: float
    pin: list[float]
    """Held cost per episode."""
    taken: list[float]
    """Split seconds of barrier lines (`OpponentSplit`) that apply, per episode."""

    @property
    def defense_s(self) -> float:
        return round(sum(self.pin) + sum(self.taken), 2)


def _pins(
    episodes: Sequence[DefenseEpisode],
    items: Sequence[DefenseItem],
    moved: dict[str, tuple[float, float]],
) -> list[float]:
    """Held cost per episode: the `pin_s` of items in `opponent_on_target` episodes, joined into
    holds of items within PIN_GAP_S of each other (a stop item overlapping a contact in the same
    hold counted once), each hold's still time beyond PIN_ALLOW_S times PIN_SHARE. A hold's
    cost goes to the episode of its longest item. Grouping is code's, so how the grader splits
    episodes changes nothing. A contact whose seconds a split already moved (`moved`: contact
    to the moved span) keeps only the share of its `pin_s` outside that span, so no second is
    costed twice."""

    def pin(i: DefenseItem) -> float:
        span = moved.get(i.fact)
        if span is None or i.t_end <= i.t_start:
            return i.pin_s
        inside = max(0.0, min(i.t_end, span[1]) - max(i.t_start, span[0]))
        return i.pin_s * (1.0 - inside / (i.t_end - i.t_start))

    by_fact = {i.fact: i.model_copy(update={"pin_s": pin(i)}) for i in items}
    owner: dict[str, int] = {}
    for k, e in enumerate(episodes):
        if e.verdict == "opponent_on_target":
            owner.update({f: k for f in e.facts if f in by_fact})
    held = sorted((by_fact[f] for f in owner), key=lambda i: i.t_start)
    holds: list[list[DefenseItem]] = []
    for i in held:
        if holds and i.t_start - max(j.t_end for j in holds[-1]) <= PIN_GAP_S:
            holds[-1].append(i)
        else:
            holds.append([i])
    out = [0.0] * len(episodes)
    for hold in holds:
        contacts = [i for i in hold if i.fact.startswith("contact")]
        still = sum(
            i.pin_s
            for i in hold
            if i in contacts
            or not any(_overlap(i.t_start, i.t_end, c.t_start, c.t_end) for c in contacts)
        )
        top = max(hold, key=lambda i: i.pin_s)
        out[owner[top.fact]] += max(0.0, still - PIN_ALLOW_S) * PIN_SHARE
    return [round(x, 2) for x in out]


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
    heaped: dict[str, float],
    episodes: Sequence[DefenseEpisode],
    by_id: dict[str, Fact],
    target: _Target,
    alliance: Alliance,
) -> Charges:
    """Charged seconds per line. Barrier lines: `lost_s` when charged by default, unless the
    grader overruled it: `home_earlier` charges its `charged_s`, `partner` and `misplaced`
    nothing. Of that, the line's `split` seconds (less any time `home_earlier` took off) go to
    the episode holding the split's contact instead, when the split applies
    (`_split_applies`). Heaped trips (`heaped`: trip line to its `heaped_at`, `_hopper_ruling`):
    from `heaped_at` to the end of the trip, less barrier lines and opponent episodes inside it,
    the time to get home from there (`_home_allow`) and TRIP_GRACE_S. Readiness: `late_s` when
    charged by default or called `not_home`, `charged_s` when `home_earlier` (or `not_home`
    with it), else 0; a `not_judged` line no call settled is not scored."""
    opp = [
        (k, s)
        for k, e in enumerate(episodes)
        if e.verdict == "opponent_on_target" and (s := _episode_span(e, by_id)) is not None
    ]
    taken = [0.0] * len(episodes)
    moved: dict[str, tuple[float, float]] = {}
    out: dict[str, float] = {}
    held: list[tuple[float, float]] = []
    for d in led.driving:
        if d.kind == "loaded_trip":
            continue
        c = calls.get(d.id)
        call = c.call if c is not None else None
        out[d.id] = (
            (c.charged_s or 0.0)
            if c is not None and call == "home_earlier"
            else d.lost_s
            if call in (None, "opponent", "no_opponent") and d.default == "charged"
            else 0.0
        )
        s, secs = d.split, 0.0
        if s is not None and _split_applies(d, call):
            k = next(
                (k for k, e in enumerate(episodes)
                 if e.verdict == "opponent_on_target" and s.contact in e.facts),
                None,
            )  # fmt: skip
            if k is not None:
                # a split is the line's tail: time `home_earlier` takes off comes off it first
                secs = max(0.0, min(s.seconds - (d.lost_s - out[d.id]), out[d.id]))
                out[d.id] -= secs
                taken[k] += secs
                moved[s.contact] = (s.t, s.t + secs)
        if out[d.id] > 0 or secs > 0 or call == "partner":
            held.append((d.t_start, d.t_end))
    for d in led.driving:
        if d.kind != "loaded_trip":
            continue
        if (h := heaped.get(d.id)) is None:
            out[d.id] = 0.0
            continue
        a = max(d.t_start, h)
        excused = _union_s(a, d.t_end, [*held, *(s for _, s in opp)])
        allow = _home_allow(target, alliance, a) + TRIP_GRACE_S
        out[d.id] = max(0.0, d.t_end - a - excused - allow)
    unjudged: list[str] = []
    for r in led.readiness:
        c = calls.get(r.id)
        call = c.call if c is not None else None
        if c is not None and call in ("home_earlier", "not_home") and c.charged_s is not None:
            out[r.id] = c.charged_s
        elif call == "not_home" or (call is None and r.default == "charged"):
            out[r.id] = r.late_s
        else:
            out[r.id] = 0.0
        if r.default == "not_judged" and call not in ("home", "not_home"):
            unjudged.append(r.id)
    out = {k: round(v, 2) for k, v in out.items()}
    total = sum(out[d.id] for d in led.driving)
    pins = _pins(episodes, led.items, moved)
    return Charges(out, unjudged, round(total, 2), pins, [round(x, 2) for x in taken])


def _split_applies(d: DrivingLine, call: Call | None) -> bool:
    """A labelled opponent's split applies unless the grader called `no_opponent`, `partner`
    or `misplaced`; an unlabelled robot's only when it called `opponent`."""
    if d.split is None or d.default != "charged":
        return False
    return call in (None, "home_earlier") if d.split.team is not None else call == "opponent"


@dataclass(frozen=True)
class Scored:
    """A skill's score and band (None when withheld, with `null_reason`), and whether it
    stands on part of the match only."""

    score: int | None
    band: Band | None
    null_reason: str | None = None
    low_coverage: bool = False


def scores(
    led: Ledgers, episodes: Sequence[DefenseEpisode], ch: Charges, cov: Coverage
) -> dict[SkillName, Scored]:
    """Each skill's score from the charged seconds (see rubric.md): code, not the grader,
    picks the score inside the band. Driving is withheld when too much of its own active time
    went unseen (DRIVING_UNSEEN_MAX) unless what was seen already costs more than one short
    hesitation, and flagged when a whole phase was mostly unseen (PHASE_UNSEEN_MAX): unseen
    time can only add charges."""
    out: dict[SkillName, Scored] = {}
    blind = [p for p in cov.phases if p.unseen_s > PHASE_UNSEEN_MAX * p.duration_s]
    low = cov.own_active_unseen_share > DRIVING_UNSEEN_MAX
    if low and ch.driving_s <= DRIVING_EDGES[1]:
        out["driving"] = Scored(
            None,
            None,
            f"{cov.own_active_unseen_share:.0%} of its own active time was untracked with "
            f"nothing to say where it was (over {DRIVING_UNSEEN_MAX:.0%}): a stall or a loaded "
            "trip there would not be seen",
            True,
        )
    else:
        out["driving"] = Scored(
            _score(ch.driving_s, DRIVING_EDGES),
            _band_of(ch.driving_s, DRIVING_EDGES),
            None,
            low or bool(blind),
        )
    judged = [r for r in led.readiness if r.id not in ch.unjudged]
    late = [ch.line.get(r.id, 0.0) for r in judged]
    if not led.readiness:
        out["shift_readiness"] = Scored(None, None, "its hub never turned on in teleop")
    elif not judged:
        out["shift_readiness"] = Scored(
            None, None, "it was untracked at every activation of its hub: none could be judged"
        )
    elif any(r.never_home and late[k] >= r.late_s > 0 for k, r in enumerate(judged)):
        out["shift_readiness"] = Scored(1, (1, 2), None, len(judged) < len(led.readiness))
    else:
        worst = max(late)
        score, band = _score(worst, READINESS_EDGES), _band_of(worst, READINESS_EDGES)
        if sum(x > READINESS_EDGES[0] for x in late) >= 2:
            score, band = min(score, 6), min(band, (5, 6))
        out["shift_readiness"] = Scored(score, band, None, len(judged) < len(led.readiness))
    if not any(e.verdict == "opponent_on_target" for e in episodes):
        out["handling_defense"] = Scored(None, None, "no opponent was established on it")
    else:
        d = ch.defense_s
        out["handling_defense"] = Scored(_score(d, DEFENSE_EDGES), _band_of(d, DEFENSE_EDGES))
    return out


def coverage(facts: Facts, match: MatchInfo, led: Ledgers) -> Coverage:
    """What the scores rest on: tracked shares of its own active time and of each own-active
    phase, the gaps there (with whether an inference puts it home), activations judged, and
    match-wide tracking of it and of the other robots."""
    gaps = [f for f in facts.facts if isinstance(f, GapFact)]
    active = _own_active(match)
    home = [(g.t_start, g.t_end) for g in gaps if _home_gap(g)]
    unknown = [(g.t_start, g.t_end) for g in gaps if not _home_gap(g)]
    phases: list[PhaseCoverage] = []
    for f in facts.facts:
        if not isinstance(f, PhaseFact) or f.phase == "auto" or not f.data.own_hub_active:
            continue
        c = f.data.coverage
        phases.append(
            PhaseCoverage(
                phase=f.phase or "-",
                duration_s=c.duration_s,
                tracked_share=c.tracked_share,
                unseen_s=round(_union_s(f.t_start, f.t_end, unknown), 1),
                home_s=round(_union_s(f.t_start, f.t_end, home), 1),
            )
        )
    dur = sum(p.duration_s for p in phases)
    auto = next(
        (f.data.coverage.tracked_share for f in facts.facts if isinstance(f, PhaseFact)
         and f.phase == "auto"), 0.0,
    )  # fmt: skip
    return Coverage(
        own_active_s=round(dur, 1),
        own_active_tracked_share=round(
            sum(p.tracked_share * p.duration_s for p in phases) / dur if dur else 0.0, 3
        ),
        own_active_unseen_share=round(sum(p.unseen_s for p in phases) / dur if dur else 0.0, 3),
        phases=phases,
        untracked=[
            Untracked(fact=g.id, t_start=g.t_start, t_end=g.t_end, inferred_home=_home_gap(g))
            for g in gaps
            if any(_overlap(g.t_start, g.t_end, a, b) for a, b in active)
        ],
        activations=len(led.readiness),
        activations_judged=sum(r.default != "not_judged" for r in led.readiness),
        match_tracked_share=facts.coverage.tracked_share,
        auto_tracked_share=auto,
        others_tracked_mean=facts.coverage.others_tracked_mean,
    )


def coverage_text(
    cov: Coverage, led: Ledgers, sc: dict[SkillName, Scored] | None, unjudged: Sequence[str]
) -> dict[SkillName, str]:
    """One sentence per skill on what its score rests on, for grade.json, grade.md and the
    grader's task (`sc` None: before scoring)."""
    phases = ", ".join(f"{p.phase} {p.tracked_share:.0%}" for p in cov.phases)
    unseen = sum(p.unseen_s for p in cov.phases)
    drive = (
        f"Driving is judged on its own active time ({cov.own_active_s:.0f} s): the target was "
        f"tracked {cov.own_active_tracked_share:.0%} of it ({phases}), and {unseen:.0f} s of "
        f"it ({cov.own_active_unseen_share:.0%}) was untracked with nothing to say where it was"
    )
    d = sc.get("driving") if sc else None
    if d is not None and d.score is None:
        drive += ", too much to score"
    elif d is not None and d.low_coverage:
        drive += ", so the score counts only what was seen and may be too high"
    skipped = [r for r in led.readiness if r.id in unjudged]
    ready = (
        f"Readiness judged {cov.activations_judged} of {cov.activations} activations of its hub"
        + (
            "; not judged (untracked, last seen home, nothing seen since): "
            + ", ".join(f"{r.phase} at {r.t:.1f} s" for r in skipped)
            if skipped
            else ""
        )
    )
    defense = (
        f"The target was tracked {cov.match_tracked_share:.0%} of the match, and on average "
        f"{cov.others_tracked_mean:.1f} of the other five robots were tracked: contact with a "
        "robot the tracker missed is not in the facts"
    )
    return {
        "driving": drive + ".",
        "shift_readiness": ready + ".",
        "handling_defense": defense + ".",
    }


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


def _coverage_prompt(cov: Coverage, led: Ledgers, match: MatchInfo) -> str:
    """What was seen, for the task: the per-skill sentences, auto's tracked share and every
    gap in its own active time."""
    said = coverage_text(cov, led, None, [r.id for r in led.readiness if r.default == "not_judged"])
    own = [p for p in match.phases if p.name != "auto" and p.active_hubs
           and match.target_alliance in p.active_hubs]  # fmt: skip
    gaps = "\n".join(
        f"- {u.fact} {u.t_start:.1f}-{u.t_end:.1f} s "
        f"({', '.join(p.name for p in own if _overlap(u.t_start, u.t_end, p.t_start, p.t_end))}): "
        + ("an inference puts it in its own zone" if u.inferred_home else "where it was is unknown")
        for u in cov.untracked
    )
    return "\n".join(
        [
            *(f"- {skill}: {text}" for skill, text in said.items()),
            f"- auto: the target was tracked {cov.auto_tracked_share:.0%} of auto.",
            "",
            "Untracked stretches in its own active time:",
            gaps or "(none)",
        ]
    )


def _prompts(
    match: MatchInfo, facts: Facts, facts_json: str, led: Ledgers, home: Corner, cov: Coverage
) -> tuple[str, str]:
    """System prompt (rules, game, limits, rubric) and task (match, coverage, ledgers and
    facts)."""
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
    drive, ready, items = _ledger_text(led, match, home, {f.id: f for f in facts.facts})
    compact = json.dumps(json.loads(facts_json), separators=(",", ":"))
    fill = {
        "team": str(match.target_team),
        "alliance": match.target_alliance,
        "roster": roster,
        "auto_winner": match.auto_winner or "unknown",
        "phases": phases,
        "own_active": ", ".join(own),
        "home_view": home,
        "coverage": _coverage_prompt(cov, led, match),
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
_QUANT = (
    r"(?:no|never|none|nothing|not once|without (?:a|an|any)|every|each|all|(?:the|its) only|"
    r"(?:only|just) one|the one|a single|two|three|four|five|sole|both|any)"
)
_EVENT = (
    r"(?:trips?|crossings?|stalls?|stops?|attempts?|turn(?:ed)?-?backs?|touch(?:es)?|faults?|"
    r"mistakes?|problems?|hesitations?|loops?|detours?|excursions?|outings?|exits?|holds?|pins?|"
    r"contacts?|hits?)"
)
ABSENCE_WORD = re.compile(
    rf"\b{_QUANT}\b(?!\W+of\W+(?:these|those|them|its charged)\b)(?:\W+\w+){{0,3}}?\W+"
    rf"{_EVENT}\b|\b(?:stayed|remained|kept) (?:well )?(?:behind|inside|in its|at home|home|"
    r"short of|on its own side)\b|\b(?:twice|thrice|(?:two|three|four|five) times|throughout|"
    r"the whole|the entire|never (?:left|crossed|"
    r"went|got out|stalled)|did not (?:leave|cross|go out|stall))\b",
    re.IGNORECASE,
)
"""Claims of absence, counts or staying put ("no trip out", "the one trip", "every crossing",
"stayed behind the centre line"): over time with untracked stretches they need SEEN_WORD.
Counts of listed things ("none of these holds") are not claims about unseen time."""
SEEN_WORD = re.compile(
    r"\b(seen|tracked|visible|in view|observed|inferred|positions? show)\b", re.IGNORECASE
)
UNKNOWN_WORD = re.compile(
    r"\b(unknown|not known|untracked|unseen|not (?:tracked|seen)|cannot (?:say|tell|show)|"
    r"do(?:es)? not show)\b",
    re.IGNORECASE,
)
"""A sentence that says the time was not seen is not claiming what happened in it."""
SENTENCE = re.compile(r"(?<=[.;!?])\s+")
CLAIM_MIN_SHARE = 0.5
"""Below this tracked share of what an item covers, no absence or count claim stands, even
qualified: "as far as tracked" over a fifth of auto says nothing."""
STALL_WORD = re.compile(r"\b(stall\w*|stuck|wedged|beached|stopped dead)\b", re.IGNORECASE)
LOST_WORD = re.compile(
    r"\b(lost time|time lost|wasted|idle\w*|doing nothing|loiter\w*)\b", re.IGNORECASE
)
"""Words reading a stop as lost time: refused over a stop in its own zone in its own active
time, where a robot shoots (the rubric does not charge it)."""
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
    {"opponent", "no_opponent", "partner", "misplaced", "home_earlier"}
)
READINESS_CALLS: frozenset[Call] = frozenset({"opponent", "home_earlier", "home", "not_home"})


@dataclass(frozen=True)
class GradeContext:
    """Everything `check_answer` checks against, built once per run."""

    facts: Facts
    match: MatchInfo
    led: Ledgers
    target: _Target
    coverage: Coverage

    @property
    def by_id(self) -> dict[str, Fact]:
        return {f.id: f for f in self.facts.facts}


def grade_context(run: RunDir) -> GradeContext:
    """Load what the checker needs from a run folder."""
    match = MatchInfo.load(run.match_json)
    facts = Facts.load(run.facts_json)
    target, others = _load_tracks(run, match)
    led = ledgers(facts, match, target, others)
    return GradeContext(facts, match, led, target, coverage(facts, match, led))


@dataclass
class _Item:
    """One answer item with text, for the text checks."""

    at: str
    text: str
    cites: list[str]
    bumpers_read: list[int]
    roles: bool
    """Check role words and places (notes, calls, episodes)."""
    scope: list[tuple[float, float]] | None = None
    """Time its absence and count claims cover (summaries, auto); None: what it cites."""
    claims: bool = True
    """Check absence and count claims (notes, summaries, auto). Episodes and calls judge what
    their tiles show by design."""


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
        self.gaps = [f for f in ctx.facts.facts if isinstance(f, GapFact)]

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
        self.unseen(item, text)
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

    def unseen(self, item: _Item, text: str) -> None:
        """No claim beyond what was seen: absence and count claims over time with untracked
        stretches say "seen" or "tracked" (and need half of it tracked); a quoted stop or hold
        that was partly inferred quotes its inferred part too; "stall" needs the robot measured
        stopped (`_stalled`); a stop at home in its own active time is not lost time."""
        at = item.at
        cited = [f for c in item.cites if (f := self.by_id.get(c)) is not None]
        scope = item.scope if item.scope is not None else self.spans_of(item.cites)
        gaps = [
            (g.t_start, g.t_end)
            for g in self.gaps
            if any(_overlap(g.t_start, g.t_end, a, b) for a, b in scope)
        ]
        if gaps and item.claims:
            total = sum(b - a for a, b in scope)
            hidden = sum(_union_s(a, b, gaps) for a, b in scope)
            share = 1.0 - hidden / total if total > 0 else 0.0
            for sentence in SENTENCE.split(REF.sub("#", text)):
                m = ABSENCE_WORD.search(sentence)
                if (
                    m is None
                    or UNKNOWN_WORD.search(sentence)
                    or (SEEN_WORD.search(sentence) and share >= CLAIM_MIN_SHARE)
                ):
                    continue
                spans = ", ".join(f"{a:.1f}-{b:.1f} s" for a, b in gaps[:6])
                self.bad(
                    f"{at}: {m.group(0)!r} claims what happened over time that was partly "
                    f"untracked ({spans}; {share:.0%} tracked)"
                    + (
                        ": say how much was tracked instead"
                        if share < CLAIM_MIN_SHARE
                        else ': say it was what was seen or tracked ("no trip out was seen")'
                    )
                )
                break
        refs = {m.group(1).strip() for m in REF.finditer(text)}
        for f in cited:
            if isinstance(f, StopFact) and f.data.unseen_s > 0:
                quoted, need = {"duration_s"}, ("unseen_s", "seen_s")
            elif isinstance(f, ContactFact) and f.data.still_unseen_s > 0:
                quoted, need = {"still_s", "longest_still_s"}, ("still_unseen_s",)
            else:
                continue
            if any(f"{f.id}.{q}" in refs for q in quoted) and not any(
                f"{f.id}.{n}" in refs for n in need
            ):
                self.bad(
                    f"{at}: it quotes {f.id}'s seconds but part of them was inferred across a "
                    f"gap; quote {{{f.id}.{need[0]}}} too and say that part was inferred"
                )
        bare = REF.sub(" ", text)
        turned = [f for f in cited if isinstance(f, CrossingTurnedBackFact)]
        m = STALL_WORD.search(bare)
        if (
            m is not None
            and not NEGATED.search(bare[: m.start()])
            and turned
            and not any(isinstance(f, StopFact) for f in cited)
            and not any(
                _stalled(f.data, self.ctx.facts.thresholds.get("STOP_MPS", 0.3)) for f in turned
            )
        ):
            self.bad(
                f"{at}: {m.group(0)!r}, but {', '.join(f.id for f in turned)} was never measured "
                f"stopped on the band; say what fired, e.g. slowed to "
                f"{{{turned[0].id}.slowest_mps}} m/s, or that it turned away"
            )
        home = [
            f
            for f in cited
            if isinstance(f, StopFact)
            and f.data.where.zone == "own"
            and f.data.on_barrier is None
            and f.data.own_active_s > 0
        ]
        if home and (m := LOST_WORD.search(bare)) is not None:
            self.bad(
                f"{at}: {m.group(0)!r} over {home[0].id}, a stop in its own zone in its own "
                "active time: that is where it shoots, not lost time"
            )

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
    """A checked answer: the answer when it passed, every problem, the scores, the charges
    and every text field resolved (keyed by its location)."""

    answer: GradeAnswer | None
    problems: list[str]
    scores: dict[SkillName, Scored]
    charges: Charges | None
    resolved: dict[str, str]


def _pressed(f: ContactFact, driving_spans: Sequence[tuple[float, float]]) -> str | None:
    """Why a contact with an opponent in it cannot be `apart` or `could_not_tell`, or None."""
    d = f.data
    return (
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


def _check_episodes(
    ck: _Checker, answer: GradeAnswer, sheets: list[FrameSheet], reread: dict[str, int]
) -> None:
    """Episodes: identification, verdicts that need an established robot, tiles, and every
    defense item covered. `reread`: contacts whose labelled robot a `no_opponent` call read as
    another team (checked there); that team fits them."""
    ctx = ck.ctx
    items = {i.fact: i for i in ctx.led.items}
    driving_spans = [(d.t_start, d.t_end) for d in ctx.led.driving if d.kind != "loaded_trip"]
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
                    else None in teams or team in teams or reread.get(f.id) == team
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
                why = _pressed(f, driving_spans)
                if why is not None:
                    ck.bad(
                        f"{at}: {f.id} cannot be {e.verdict}: an opponent is established in it and "
                        f"{why}. An opponent there is on the target: open a tile and use "
                        "opponent_on_target or target_on_opponent"
                    )
    ck.episodes = list(answer.episodes)
    missing = [i.fact for i in ctx.led.items if i.fact not in covered]
    if missing:
        ck.bad(
            f"episodes: these defense items are not accounted for: {', '.join(missing)}; put each "
            "in an episode (could_not_tell is allowed)"
        )


def _check_split_call(ck: _Checker, c: LineCall, line: DrivingLine, at: str) -> None:
    """`opponent` applies an unlabelled robot's split once its episode establishes it as an
    opponent; `no_opponent` drops a labelled one's when the grader read a partner's number on
    the robot boxed with that label beside the target (`other_bumper`, also its episode's
    team). Either needs the call to cite a tile in that contact. Where the robot was is
    measured, so whether it was in the way is never grounds."""
    s = line.split
    want = "opponent" if c.call == "opponent" else "no_opponent"
    if s is None or line.default != "charged" or (s.team is None) != (want == "opponent"):
        ck.bad(
            f"{at}: {want} "
            + (
                "applies an UNLABELLED split only: this line has none"
                if want == "opponent"
                else "drops a labelled OPPONENT SPLIT only: this line has none"
            )
            + "; where the other robots were is measured, so whether one was in its way is not "
            "an overrule"
        )
        return
    f = ck.by_id[s.contact]
    e = next((e for e in ck.episodes if s.contact in e.facts), None)
    tiles = ck.frames_in(c.cites, f.t_start, f.t_end)
    match = ck.ctx.match
    if want == "opponent":
        ok = e is not None and e.verdict == "opponent_on_target"
        need = (
            f"the episode holding {s.contact} to be opponent_on_target, the robot identified "
            "as an opponent there by its bumper number or colour on a cited tile"
        )
    else:
        read = c.other_bumper
        boxed = [t for t in tiles if (m := ck.tiles[t].main) is not None and s.team in m.teams_near]
        ok = (
            read is not None
            and read != match.target_team
            and _relation(match, read) == "partner"
            and bool(boxed)
            and e is not None
            and e.identified_by == "bumper_number"
            and e.other_team == read
        )
        need = (
            f"other_bumper = the partner number you read on the robot boxed {s.team} beside the "
            f"target on a cited tile, and the episode holding {s.contact} identified by that "
            "bumper_number"
        )
    if not ok or not tiles:
        ck.bad(
            f"{at}: {want} needs {need}, and this call to cite a tile within {SPAN_PAD_S:g} s of "
            f"{s.contact} ({f.t_start:.1f}-{f.t_end:.1f} s)"
        )


def _hopper_ruling(tiles: Sequence[TileRead]) -> tuple[Hopper, float | None]:
    """A trip's ruling from its tile readings: `heaped` from the earliest readable tile from
    which every later readable tile is heaped (a hopper rarely empties out of its zone, and an
    unreadable tile in between says nothing), `not_heaped` when the last readable tile is not,
    `cannot_see` when none is readable. A heap on fewer than HEAPED_TILES readable tiles is
    `not_heaped`."""
    seen = sorted((r for r in tiles if r.hopper != "cannot_see"), key=lambda r: r.t)
    if not seen:
        return "cannot_see", None
    run: list[float] = []
    for r in reversed(seen):
        if r.hopper != "heaped":
            break
        run.append(r.t)
    return ("heaped", run[-1]) if len(run) >= HEAPED_TILES else ("not_heaped", None)


def _check_hoppers(ck: _Checker, answer: GradeAnswer, sheets: list[FrameSheet]) -> None:
    """One reading per trip line, one per HOPPER TILE of it and none elsewhere; `heaped` and
    `not_heaped` only on a served tile with the target ringed (or a corner tile with its bumper
    read); `cannot_see` on a served tile, unless the sheet budget ran out."""
    trips = {d.id: d for d in ck.ctx.led.driving if d.kind == "loaded_trip"}
    team = ck.ctx.match.target_team
    done: set[str] = set()
    for r in answer.hoppers:
        at = f"hoppers {r.line}"
        line = trips.get(r.line)
        if line is None or r.line in done:
            ck.bad(f"{at}: not a loaded_trip line, or read twice: one reading per trip line")
            continue
        done.add(r.line)
        want = ", ".join(f"{t:.1f}" for t in sorted(line.look_at))
        got = ", ".join(f"{x.t:.1f}" for x in sorted(r.tiles, key=lambda x: x.t))
        if got != want:
            ck.bad(f"{at}: read each HOPPER TILE once, exactly at {want} (got {got})")
        for x in r.tiles:
            t = ck.frame_t(f"frame:{x.t}")
            tile = ck.tiles[t] if t is not None else None
            if tile is None:
                if x.hopper != "cannot_see" or len(sheets) < IMAGE_CAP:
                    ck.bad(f"{at}: the tile at {x.t:.1f} s was never served: open it")
                continue
            if x.hopper == "cannot_see":
                continue
            ringed = tile.main is not None and tile.main.tracked
            if not ringed and not (tile.corner and r.corner_bumper == team):
                ck.bad(
                    f"{at}: {x.hopper} at {x.t:.1f} s needs the target ringed there, or a corner "
                    f"tile at that time with its bumper read (corner_bumper {team}); else "
                    "cannot_see"
                )
    for d in trips:
        if d not in done:
            ck.bad(f"hoppers: trip line {d} has no reading")


def _check_call(ck: _Checker, c: LineCall, line: DrivingLine | ReadinessLine) -> None:
    """An overrule needs the contradiction its call names, on tiles code can check, and no
    reading of what the robot meant."""
    at = f"line call {c.line}"
    ctx = ck.ctx
    for x in c.cites:
        ck.span(x, at)
    drive = isinstance(line, DrivingLine)
    if drive and line.kind == "loaded_trip":
        ck.bad(f"{at}: {c.line} is a loaded_trip line: read its hopper tiles in `hoppers`")
        return
    allowed = BARRIER_CALLS if drive else READINESS_CALLS
    if c.call not in allowed:
        ck.bad(f"{at}: {c.call} is not a call for this line; use one of {sorted(allowed)}")
        return
    a, b = (line.t_start, line.t_end) if drive else (line.t, line.t + line.late_s)
    frames = ck.frames_in(c.cites, a, b)
    corner = [t for t in ck.frames_in(c.cites, a, b) if ck.corner_only(t)]
    if corner and c.corner_bumper != ctx.match.target_team:
        ck.bad(
            f"{at}: it cites corner tile(s) {', '.join(f'frame:{t:.1f}' for t in corner)}; a "
            "corner tile counts only where you read the target's bumper number there: set "
            f"corner_bumper to {ctx.match.target_team} if you read it, else do not rely on it"
        )
    if c.other_bumper is not None and c.call != "no_opponent":
        ck.bad(f"{at}: other_bumper is only for no_opponent")
    if c.charged_s is not None and c.call not in ("home_earlier", "not_home"):
        ck.bad(f"{at}: charged_s is only for home_earlier and not_home")
    if (m := INTENT_WORD.search(c.text)) is not None:
        ck.bad(
            f"{at}: {m.group(0)!r} reads what the robot meant; an overrule stands only on what a "
            "tile shows (where the target was, who held it, when it was home)"
        )
    if c.call in ("opponent", "no_opponent") and isinstance(line, DrivingLine):
        _check_split_call(ck, c, line, at)
    elif c.call == "opponent" and not any(
        e.verdict == "opponent_on_target"
        and any(
            isinstance(f := ck.by_id.get(i), ContactFact) and _overlap(a, b, f.t_start, f.t_end)
            for i in e.facts
        )
        for e in ck.episodes
    ):
        ck.bad(
            f"{at}: opponent needs a contact fact in {a:.1f}-{b:.1f} s in an "
            "opponent_on_target episode"
        )
    elif c.call == "partner" and isinstance(line, DrivingLine):
        stops = [
            (f.t_start, f.t_end) for i in line.facts if isinstance(f := ck.by_id.get(i), StopFact)
        ] or [(line.t_start, line.t_end)]
        held = [
            i
            for i in line.partner_contacts
            if any(_overlap(ck.by_id[i].t_start, ck.by_id[i].t_end, a0, b0) for a0, b0 in stops)
        ]
        if not held or not frames:
            ck.bad(
                f"{at}: partner needs a cited tile in the line and a partner contact fact while "
                f"it was stopped ({', '.join(f'{a0:.1f}-{b0:.1f}' for a0, b0 in stops)} s); a "
                "partner nearby is not in the way"
            )
    elif c.call == "misplaced":
        ringed = [t for t in frames if (m := ck.tiles[t].main) is not None and m.tracked]
        if line.default != "charged" or len(ringed) < MISPLACED_TILES:
            ck.bad(
                f"{at}: misplaced overrules a charged default only, with at least "
                f"{MISPLACED_TILES} cited tiles in {a:.1f}-{b:.1f} s with the target ringed that "
                "show the ring on another robot or the target away from the barrier"
            )
    elif c.call == "home_earlier":
        cap = line.lost_s if isinstance(line, DrivingLine) else line.late_s
        bound = line.bound if isinstance(line, DrivingLine) else line.late_is_bound
        cs = c.charged_s
        if (
            not bound
            or line.default != "charged"
            or cs is None
            or cs >= cap
            or not any(abs(t - a - cs) <= 0.6 for t in frames)
        ):
            ck.bad(
                f"{at}: home_earlier is only for a charged line whose time is an upper bound "
                "(arrival first seen after a gap): charged_s = the seconds from its start to a "
                f"cited tile that shows the target home, under {_fmt(cap)}"
            )
    elif c.call in ("home", "not_home") and isinstance(line, ReadinessLine):
        near = any(abs(t - a) <= NOT_HOME_S for t in ck.frames_in(c.cites, a, a, pad=NOT_HOME_S))
        unknown = line.default == "not_judged"
        read = c.corner_bumper == ctx.match.target_team
        at_home = c.charged_s is None or any(
            abs(t - a - c.charged_s) <= 0.6 for t in ck.frames_in(c.cites, a, a + c.charged_s)
        )
        if c.call == "home" and not (unknown and near and read):
            ck.bad(
                f"{at}: home settles a not_judged readiness line only, with a cited tile within "
                f"{NOT_HOME_S:g} s of the activation where you read the target's bumper number "
                "(corner_bumper) in its own zone"
            )
        elif c.call == "not_home" and not (
            near and at_home and (read if unknown else line.default == "cleared")
        ):
            ck.bad(
                f"{at}: not_home overrules a cleared readiness line, or settles a not_judged one "
                "with the target's bumper read (corner_bumper), with a cited tile within "
                f"{NOT_HOME_S:g} s of the activation that shows the target out of its own zone; "
                "charged_s, if given, is the seconds to a cited tile that shows it home"
            )


def check_answer(raw: object, ctx: GradeContext, sheets: list[FrameSheet]) -> Checked:
    """Validate the agent's answer and compute the charges and scores it implies."""
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
        at: str,
        value: str,
        cites: list[str],
        bumpers: list[int] | None = None,
        roles: bool = True,
        scope: list[tuple[float, float]] | None = None,
        claims: bool = True,
    ) -> None:
        resolved[at] = ck.resolve(_Item(at, value, cites, bumpers or [], roles, scope, claims))

    m = ctx.match
    scopes: dict[SkillName, list[tuple[float, float]]] = {
        "driving": _own_active(m),
        "shift_readiness": _own_active(m),
        "handling_defense": [(m.teleop_start_t, m.teleop_end_t)],
    }

    calls: dict[str, LineCall] = {}
    for c in answer.line_calls:
        if c.line in calls:
            ck.bad(f"line_calls: {c.line} has more than one call")
        calls[c.line] = c
    reread = {
        d.split.contact: c.other_bumper
        for d in led.driving
        if d.split is not None
        and (c := calls.get(d.id)) is not None
        and c.call == "no_opponent"
        and c.other_bumper is not None
    }
    _check_episodes(ck, answer, sheets, reread)
    _check_hoppers(ck, answer, sheets)
    for c in answer.line_calls:
        line = ck.lines.get(c.line)
        if line is None:
            ck.bad(f"line call {c.line}: no ledger line {c.line}")
            continue
        _check_call(ck, c, line)
    for d in led.driving:
        c = calls.get(d.id)
        s = d.split
        if s is None or not _split_applies(d, c.call if c else None):
            continue
        holding = [e for e in answer.episodes if s.contact in e.facts]
        if not holding or any(e.verdict != "opponent_on_target" for e in holding):
            ck.bad(
                f"episodes: {s.contact} carries {d.id}'s OPPONENT SPLIT (a robot between the "
                "target and its own wall, measured): its episode must be opponent_on_target"
                + ("" if s.team is None else f", unless you call no_opponent on {d.id}")
            )

    # what code charges, then the text checks that lean on it
    rulings = {r.line: _hopper_ruling(r.tiles) for r in answer.hoppers}
    heaped = {k: t for k, (_, t) in rulings.items() if t is not None}
    al = ctx.match.target_alliance
    ch = charges(led, calls, heaped, answer.episodes, ck.by_id, ctx.target, al)
    for d in led.driving:
        c = calls.get(d.id)
        trip = rulings.get(d.id, ("cannot_see", None))[0] if d.kind == "loaded_trip" else None
        if ch.line.get(d.id, 0.0) > 0:
            ck.charged_lines.append(d)
        elif c is None and d.default == "cleared" and trip is None:
            ck.cleared_lines.append((d, "cleared by its default"))
        elif c is not None and c.call == "misplaced":
            ck.cleared_lines.append((d, c.call))
        elif trip == "not_heaped":
            ck.cleared_lines.append((d, trip))
    ck.pinned = [
        s
        for e, cost in zip(answer.episodes, ch.pin, strict=True)
        if cost > 0 and (s := _episode_span(e, ck.by_id)) is not None
    ]
    for k, e in enumerate(answer.episodes):
        text(
            f"episode {k + 1} ({', '.join(e.facts)})",
            e.text,
            [*e.facts, *e.cites],
            [e.other_team] if e.identified_by == "bumper_number" and e.other_team else [],
            claims=False,
        )
    for c in answer.line_calls:
        line = ck.lines.get(c.line)
        if line is not None:
            own = line.facts if isinstance(line, DrivingLine) else [line.fact]
            read = [c.other_bumper] if c.other_bumper is not None else []
            text(f"line call {c.line}", c.text, [*c.cites, *own], read, claims=False)
    for r in answer.hoppers:
        if (line := ck.lines.get(r.line)) is not None and isinstance(line, DrivingLine):
            times = [ck.frame_t(f"frame:{x.t}") for x in r.tiles]
            served = [f"frame:{t:.1f}" for t in times if t is not None]
            text(f"hoppers {r.line}", r.text, [*served, *line.facts], claims=False)
    names = [g.skill for g in answer.skills]
    if sorted(names) != sorted(SKILLS):
        ck.bad(f"skills: need exactly one each of {list(SKILLS)}, got {names}")
    for g in answer.skills:
        for c in g.cites:
            ck.span(c, f"skill {g.skill}")
        text(f"skill {g.skill} summary", g.summary, g.cites, roles=False, scope=scopes[g.skill])
    for c in answer.auto.cites:
        ck.span(c, "auto")
    text("auto", answer.auto.text, answer.auto.cites, roles=False,
         scope=[(m.auto_start_t, m.auto_end_t)])  # fmt: skip
    for k, n in enumerate(answer.notes):
        at = f"note {k + 1} (t={n.t})"
        found = [s for c in n.cites if (s := ck.span(c, at)) is not None]
        if found and not any(a - NOTE_REACH_S <= n.t <= b + NOTE_REACH_S for a, b in found):
            ck.bad(f"{at}: t is more than {NOTE_REACH_S:g} s from everything it cites")
        if n.kind == "played_through_defense" and not any(c.startswith("frame:") for c in n.cites):
            ck.bad(f"{at}: played_through_defense needs a frame citation")
        spans = ck.spans_of(n.cites)
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
            ck.bad(f"{at}: kind went_wrong over {', '.join(cleared)}, which is cleared")
        home = [
            i
            for i in n.cites
            if isinstance(f := ck.by_id.get(i), StopFact)
            and f.data.where.zone == "own"
            and f.data.on_barrier is None
            and f.data.own_active_s > 0
        ]
        if n.kind == "went_wrong" and home and not hit:
            ck.bad(
                f"{at}: kind went_wrong over {home[0]}, a stop in its own zone in its own active "
                "time, which no line charges: that is where it shoots"
            )
        text(at, n.text, n.cites, n.bumpers_read)
    problems = ck.problems
    sc = scores(led, answer.episodes, ch, ctx.coverage)
    return Checked(answer if not problems else None, problems, sc, ch, resolved)


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
    system, task = _prompts(
        match, ctx.facts, run.facts_json.read_text(), ctx.led, home, ctx.coverage
    )
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
    cov = ctx.coverage.model_copy(
        update={"activations_judged": len(ctx.led.readiness) - len(ch.unjudged)}
    )
    said = coverage_text(cov, ctx.led, checked.scores, ch.unjudged)
    notes: list[Note] = []
    for k, n in enumerate(answer.notes):
        period, clock = _clock(match, n.t)
        text = resolved[f"note {k + 1} (t={n.t})"]
        notes.append(
            Note.model_validate(n.model_dump() | {"text": text, "clock": clock, "period": period})
        )
    notes.sort(key=lambda n: n.t)
    skills = [
        ScoredSkill(
            skill=g.skill,
            summary=resolved[f"skill {g.skill} summary"],
            cites=g.cites,
            score=checked.scores[g.skill].score,
            band=checked.scores[g.skill].band,
            null_reason=checked.scores[g.skill].null_reason,
            coverage=said[g.skill],
            low_coverage=checked.scores[g.skill].low_coverage,
        )
        for g in sorted(answer.skills, key=lambda g: SKILLS.index(g.skill))
    ]
    calls = {c.line: c for c in answer.line_calls}
    hoppers = {r.line: r for r in answer.hoppers}
    scored: list[ScoredCall] = []
    for line in [*ctx.led.driving, *ctx.led.readiness]:
        if (r := hoppers.get(line.id)) is not None:
            call, at = _hopper_ruling(r.tiles)
            scored.append(
                ScoredCall(
                    line=line.id,
                    default=line.default,
                    call=call,
                    charged_total_s=ch.line.get(line.id, 0.0),
                    heaped_at=at,
                    hopper_tiles=r.tiles,
                    corner_bumper=r.corner_bumper,
                    text=resolved[f"hoppers {line.id}"],
                    cites=[f"frame:{x.t:.1f}" for x in r.tiles],
                )
            )
            continue
        c = calls.get(line.id)
        scored.append(
            ScoredCall(
                line=line.id,
                default=line.default,
                call=c.call if c else None,
                charged_total_s=ch.line.get(line.id, 0.0),
                charged_s=c.charged_s if c else None,
                corner_bumper=c.corner_bumper if c else None,
                text=resolved[f"line call {line.id}"]
                if c
                else line.why
                + (
                    f"; {line.split.why} count against handling_defense"
                    if isinstance(line, DrivingLine)
                    and line.split is not None
                    and _split_applies(line, None)
                    else ""
                ),
                cites=c.cites if c else [],
            )
        )
    result = Grade(
        team=match.target_team,
        rubric_version=_rubric_version(_text("rubric.md")),
        coverage=cov,
        skills=skills,
        driving_ledger=ctx.led.driving,
        readiness_ledger=ctx.led.readiness,
        defense_items=ctx.led.items,
        line_calls=scored,
        episodes=[
            ScoredEpisode.model_validate(
                e.model_dump()
                | {
                    "text": resolved[f"episode {k + 1} ({', '.join(e.facts)})"],
                    "pin_cost_s": pin,
                    "total_cost_s": round(pin + took, 2),
                }
            )
            for k, (e, pin, took) in enumerate(zip(answer.episodes, ch.pin, ch.taken, strict=True))
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
        "## What the scores rest on",
        "",
        *(f"- {s.skill}: {s.coverage}" for s in g.skills),
        f"- auto: the target was tracked {g.coverage.auto_tracked_share:.0%} of auto.",
        "",
        f"Rubric `{g.rubric_version}`. Scores are absolute, 1 to 10, computed from the charged "
        "seconds and episode costs. Auto is reported, not scored.",
        "",
        "| Skill | Score | Band |",
        "|---|---|---|",
    ]
    lines += [
        f"| {s.skill} | {s.score if s.score is not None else 'null'}"
        f"{' (low coverage: may be too high)' if s.low_coverage and s.score is not None else ''} | "
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
        "| Line | Time | Lost (s) | Default | Call | Charged (s) | Why |",
        "|---|---|---|---|---|---|---|",
    ]
    for d in g.driving_ledger:
        c = calls[d.id]
        lines.append(
            f"| {d.id} {d.kind} | {d.t_start:.1f}-{d.t_end:.1f} | {_fmt(d.lost_s)} | {d.default} | "
            f"{c.call or 'kept'} | {_fmt(c.charged_total_s)} | {cell(c.text)} |"
        )
    lines += [
        "",
        "## Readiness ledger",
        "",
        "| Line | Phase | Late (s) | Default | Call | Charged (s) | Why |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in g.readiness_ledger:
        c = calls[r.id]
        lines.append(
            f"| {r.id} | {r.phase} | {_fmt(r.late_s)} | {r.default} | {c.call or 'kept'} | "
            f"{_fmt(c.charged_total_s)} | {cell(c.text)} |"
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
