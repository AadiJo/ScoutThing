"""The ledgers code builds from the facts before the grader runs (`ledgers`): driving lines
(barrier passes and trips out of its own zone in its own active time) with their default
rulings, hopper tiles and opponent splits, readiness lines with their default rulings, and the
defense items. A line is decided the same way on every run; the grader can only overrule a
default with a checkable reason."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from scout.games.rebuilt2026 import ROBOT_FOOTPRINT_M, own_wall_m
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
    StopFact,
    TripFact,
    ZoneVisitFact,
)
from scout.schemas.grade import (
    DefenseItem,
    DrivingKind,
    DrivingLine,
    OpponentSplit,
    ReadinessLine,
    ReadinessRuling,
    Ruling,
)
from scout.schemas.match import MatchInfo
from scout.stages.grade.base import fmt, overlap, union_s
from scout.stages.grade.clock import own_active as _own_active
from scout.stages.grade.clock import phase
from scout.stages.grade.tracks import TRACKED_WITHIN_S, TRIP_ALLOW_S, Others, Target, home_allow

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
(`tracks.home_allow`), at most the contact's length; all of it from then when it came close before
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


@dataclass(frozen=True)
class Ledgers:
    driving: list[DrivingLine]
    readiness: list[ReadinessLine]
    items: list[DefenseItem]


def _in_spans(t: float, spans: Sequence[tuple[float, float]]) -> bool:
    return any(a - 1e-6 <= t < b for a, b in spans)


def _contacts_in(facts: Sequence[Fact], a: float, b: float) -> tuple[list[str], list[str]]:
    """(contacts with an opponent or unlabelled robot, contacts with only partners) in [a, b]."""
    opp: list[str] = []
    partner: list[str] = []
    for f in facts:
        if isinstance(f, ContactFact) and overlap(f.t_start, f.t_end, a, b):
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


def _look_times(a: float, b: float, target: Target, alliance: Alliance) -> list[float]:
    """Hopper tile times for a trip from a to b, evenly spaced (see LOOK_STEP_S); empty when a
    heap seen at any moment of it could not cost MIN_LOST_S (`charges`, before excused time),
    or fewer than HEAPED_TILES tiles fit."""
    first = round(a + LOOK_FIRST_S, 1)
    last = round(b - TRIP_ALLOW_S - TRIP_GRACE_S - MIN_LOST_S, 1)
    while last >= first and b - last - home_allow(target, alliance, last) < (
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

    target: Target
    others: Others
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
    takes (`tracks.home_allow`), at most the contact's length; when it came close before the target
    left the barrier, all of the line from then. A labelled opponent's split is preferred over
    an unlabelled robot's; splits of no seconds are dropped."""
    left = r.b if r.left is None else r.left
    found: list[tuple[bool, float, OpponentSplit]] = []
    for f in fs:
        if not (isinstance(f, ContactFact) and overlap(f.t_start, f.t_end, r.a, r.b)):
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
                allow = home_allow(tr.target, tr.alliance, t)
                secs = min(r.b - t - allow, f.t_end - t)
                how = (
                    f"after it left the barrier: the line's {fmt(r.b - t)} s from then less "
                    f"{fmt(allow)} s to get home from there, at most the contact's "
                    f"{fmt(f.t_end - t)} s"
                )
            secs = round(min(secs, r.lost), 2)
            if secs <= 0:
                continue
            opp = o.relation == "opponent"
            why = (
                f"{f.id}: {who} came {fmt(ahead)} m nearer its own wall than the target (in its "
                f"way at {AHEAD_M:g} m or more) at {t:.1f} s, {how}: {fmt(secs)} s"
            )
            found.append((not opp, t, OpponentSplit(
                contact=f.id, team=o.team if opp else None, t=round(t, 2), seconds=secs, why=why
            )))  # fmt: skip
    return min(found, key=lambda x: (x[0], x[1]))[2] if found else None


def stalled(d: CrossingTurnedBackData, stop_mps: float) -> bool:
    """Measured stopped on the band (see STALLED_S)."""
    slow = d.slowest_mps
    return (slow is not None and slow < stop_mps) or d.stopped_s + d.unseen_s >= STALLED_S


def _turned_back_default(d: CrossingTurnedBackData, stop_mps: float) -> tuple[Ruling, str]:
    """A turned-back attempt is charged when it slowed on the band or went well back from the
    barrier afterwards; a drive past keeps its pace and stays close. The reason names the
    measure that fired, and calls it a stall only when the robot was measured stopped; it never
    denies a stall, which a slow speed read over a second can hide."""
    slow, far = d.slowest_mps, d.furthest_from_barrier_m
    kind = (
        f"stopped {fmt(d.stopped_s)} s on the band: a stall"
        if stalled(d, stop_mps)
        else f"not measured stopped (never under {stop_mps:g} m/s there, stopped "
        f"{fmt(d.stopped_s)} s)"
    )
    if slow is None or slow <= STALL_MPS:
        what = (
            "no speed could be read on the band"
            if slow is None
            else (f"slowed to {fmt(slow)} m/s on the band")
        )
        return "charged", f"{what} (charged at or under {STALL_MPS:g} m/s); {kind}"
    if far is not None and far >= BACKED_OFF_M:
        return "charged", (
            f"went {fmt(far)} m back from the barrier (charged at {BACKED_OFF_M:g} m or more), "
            f"slowest {fmt(slow)} m/s: a back-off or turn-away, not measured stopped"
        )
    return "cleared", (
        f"drive past: slowest {fmt(slow)} m/s (over {STALL_MPS:g}) and at most "
        f"{fmt(far or 0.0)} m back from the barrier (under {BACKED_OFF_M:g})"
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
        f"there {fmt(a.first_tracked_t - t0)} s after"
    )


def home_gap(g: GapFact) -> bool:
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
        elif d.in_own_zone is None and gap is not None and home_gap(gap):
            default = "cleared"
            why = f"untracked in {gap.id}, whose inference {gap.data.inference} puts it home"
        elif d.reached is not None and d.reached.by in ("crossing", "inference") and late == 0:
            default = "cleared"
            why = f"untracked; {d.reached.fact} puts it home by the activation"
        elif d.in_own_zone is None and (last is None or last.zone == "own"):
            default = "not_judged"
            why = (
                "untracked, last seen in its own zone "
                f"{fmt(f.t_start - last.t)} s before and nothing seen since puts it out or home: "
                "where it was is unknown, so it is not judged"
                if last is not None
                else "untracked since the match began: not judged"
            )
        else:
            default = "charged"
            why = (
                "tracked outside its own zone"
                if d.in_own_zone is False
                else f"untracked, last seen outside its own zone {fmt(f.t_start - last.t)} s before"
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
            else f"; in its own zone {fmt(late)} s later"
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


def ledgers(facts: Facts, match: MatchInfo, target: Target, others: Others) -> Ledgers:
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
        return any(overlap(a, b, r.a, r.b) for r in raw)

    def active_s(own_active: float, a: float, b: float) -> float:
        """A turned-back attempt's own-active seconds less the lateness a readiness line
        already charges: an attempt in inactive time that got it home late is a readiness
        fault, charged once."""
        return max(0.0, own_active - union_s(a, b, late))

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
            and overlap(s.t_start, s.t_end, f.t_start, f.t_end)
            and s.data.barrier_edge_m <= ROBOT_FOOTPRINT_M
        ]
        arrive = (
            f"in the {d.far_side} zone {fmt(d.seconds_to_far_side)} s after entering "
            f"({_arrival_text(d.arrived, f.t_start)})"
            if d.arrived is not None and d.seconds_to_far_side is not None
            else f"never tracked in the {d.far_side} zone after it"
        )
        stops = "".join(
            f"; {s.id}: stopped {fmt(s.data.duration_s)} s ({fmt(s.data.unseen_s)} s inferred "
            f"across gaps), {fmt(s.data.barrier_edge_m)} m from the bumps' depth"
            for s in folded
        )
        text = (
            f"onto the {d.barrier} {d.lane} from the {d.side} zone toward the {d.far_side} zone "
            f"and back out (reach {d.reach}, centre {fmt(d.nearest_edge_m)} m from the bumps' "
            f"depth at nearest); on the barrier {fmt(d.time_on_barrier_s)} s, stopped "
            f"{fmt(d.stopped_s)} s, {fmt(d.unseen_s)} s unseen; {arrive}; costed "
            f"{fmt(d.cost_s)} s until {_end_text(d.cost_end)}, {fmt(d.own_active_s)} s of it in "
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
                f"stopped on the {d.on_barrier} {d.lane or 'barrier'} {fmt(d.duration_s)} s: "
                f"{fmt(d.seen_s)} s seen, {fmt(d.unseen_s)} s inferred across gaps (start "
                f"{d.start}, end {d.end}); {fmt(d.own_active_s)} s in its own active time"
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
                f"stayed on the {d.barrier} {d.lane} {fmt(d.duration_s)} s "
                f"({fmt(d.untracked_s)} s untracked), from {d.came_from} to {d.went_to}: "
                f"{fmt(excess)} s beyond a normal crossing, {fmt(d.own_active_s)} s in its own "
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
            f"crossed the {d.barrier} {d.lane} from {d.from_zone} to {d.to_zone} in {fmt(on)} s, "
            f"stopped {fmt(d.stopped_s)} s: {fmt(d.excess_s)} s beyond a normal crossing, "
            f"{fmt(d.own_active_s)} s in its own active time"
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
        ending = {
            "home": "then back home",
            "track_lost": f"then tracking lost it outside its zone ({d.end_gap}): not seen home "
            "and not seen staying out, so the trip ends where it was last seen out",
            "hub_went_inactive": "still tracked out when its active time ended",
            "match_end": "still tracked out when the match ended",
        }[d.end]
        text = (
            f"out of its own zone in its own active time {fmt(d.duration_s)} s "
            f"({fmt(d.unseen_s)} s unseen)"
            + (f", from the end of {d.start_gap} (long enough to have gone home)"
               if d.start_gap else "")
            + f", furthest {fmt(abs(past))} m {'past' if past >= 0 else 'short of'} the centre "
            f"line, {ending}"
            + (f"; stops out: {', '.join(d.stop_ids)}" if d.stop_ids else "")
        )  # fmt: skip
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
                phase=phase(match, r.a),
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
                + f" {fmt(o.min_distance_m)} m"
                + (
                    f", closing speeds: target {fmt(o.target_closing_mps)}, other "
                    f"{fmt(o.other_closing_mps)} m/s"
                    if o.target_closing_mps is not None and o.other_closing_mps is not None
                    else ""
                )
                for o in _per_team(d.others)
            )
            text = (
                f"contact, target {d.target_zone} ({fmt(d.where.own_hub_m)} m from its hub, "
                f"{fmt(d.where.own_wall_m)} m from its wall), own hub "
                f"{'active' if in_active else 'inactive'}; {who}; {fmt(d.seen_s)} s seen, "
                f"{fmt(d.unseen_s)} s unseen; target nearly still {fmt(d.still_s)} s "
                f"({fmt(d.still_unseen_s)} s of it inferred across gaps, longest run "
                f"{fmt(d.longest_still_s)} s); end: {_end_text(d.end)}"
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
                + f" {fmt(o.min_distance_m)} m"
                for o in _per_team(d.others_near)
            )
            zones = f"{d.before.zone if d.before else '?'} to {d.after.zone if d.after else '?'}"
            text = (
                f"target unseen {fmt(d.duration_s)} s ({zones}, inference {d.inference}); "
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
                f"stop {fmt(d.duration_s)} s in the {d.where.zone} zone, {fmt(d.unseen_s)} s "
                f"inferred across gaps; nearest tracked robot {fmt(d.nearest_tracked_m or 0.0)} m"
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
