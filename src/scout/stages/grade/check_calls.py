"""Checks of the ledger calls and hopper readings: each overrule of a default must be backed by
the checkable contradiction it names (`check_call`), and each trip's hopper tiles must be read
(`check_hoppers`); code turns the readings into the trip's ruling (`hopper_ruling`)."""

from __future__ import annotations

import re
from collections.abc import Sequence

from scout.schemas.facts import ContactFact, StopFact
from scout.schemas.grade import (
    Call,
    DrivingLine,
    FrameSheet,
    GradeAnswer,
    Hopper,
    LineCall,
    ReadinessLine,
    TileRead,
)
from scout.stages.grade.base import IMAGE_CAP, fmt, overlap
from scout.stages.grade.checker import SPAN_PAD_S, Checker
from scout.stages.grade.clock import relation
from scout.stages.grade.ledgers import HEAPED_TILES

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
BARRIER_CALLS: frozenset[Call] = frozenset(
    {"opponent", "no_opponent", "partner", "misplaced", "home_earlier"}
)
READINESS_CALLS: frozenset[Call] = frozenset({"opponent", "home_earlier", "home", "not_home"})


def _check_split_call(ck: Checker, c: LineCall, line: DrivingLine, at: str) -> None:
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
            and relation(match, read) == "partner"
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


def hopper_ruling(tiles: Sequence[TileRead]) -> tuple[Hopper, float | None]:
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


def check_hoppers(ck: Checker, answer: GradeAnswer, sheets: list[FrameSheet]) -> None:
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


def check_call(ck: Checker, c: LineCall, line: DrivingLine | ReadinessLine) -> None:
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
            isinstance(f := ck.by_id.get(i), ContactFact) and overlap(a, b, f.t_start, f.t_end)
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
            if any(overlap(ck.by_id[i].t_start, ck.by_id[i].t_end, a0, b0) for a0, b0 in stops)
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
                f"cited tile that shows the target home, under {fmt(cap)}"
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
