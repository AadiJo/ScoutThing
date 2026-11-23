"""What code charges and scores once the grader's calls are in: charged seconds per line and
per episode (`charges`), the 1-10 scores and their bands (`scores`), and how much of the match
the tracking saw (`coverage`, `coverage_text`)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from scout.schemas.common import Alliance
from scout.schemas.facts import Fact, Facts, GapFact, PhaseFact
from scout.schemas.grade import (
    Call,
    Coverage,
    DefenseEpisode,
    DefenseItem,
    DrivingLine,
    LineCall,
    PhaseCoverage,
    SkillName,
    Untracked,
)
from scout.schemas.match import MatchInfo
from scout.stages.grade.base import overlap, union_s
from scout.stages.grade.clock import own_active
from scout.stages.grade.ledgers import TRIP_GRACE_S, Ledgers, home_gap
from scout.stages.grade.tracks import Target, home_allow

Band = tuple[int, int]
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
PIN_ALLOW_S = 3.0
"""Still seconds beside an opponent an episode may hold before they cost (pinning longer
than 3 s is a foul in this game)."""
PIN_GAP_S = 2.0
"""Opponent-held items this close in time are one hold: they share one allowance."""
PIN_SHARE = 0.5
"""Share of held still time beyond the allowance that costs. Whether it kept shooting while
held is rarely readable on a tile (the grader said `cannot_see` on almost every hold), so
code charges half instead of asking."""


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


def episode_span(e: DefenseEpisode, by_id: dict[str, Fact]) -> tuple[float, float] | None:
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
            or not any(overlap(i.t_start, i.t_end, c.t_start, c.t_end) for c in contacts)
        )
        top = max(hold, key=lambda i: i.pin_s)
        out[owner[top.fact]] += max(0.0, still - PIN_ALLOW_S) * PIN_SHARE
    return [round(x, 2) for x in out]


def charges(
    led: Ledgers,
    calls: dict[str, LineCall],
    heaped: dict[str, float],
    episodes: Sequence[DefenseEpisode],
    by_id: dict[str, Fact],
    target: Target,
    alliance: Alliance,
) -> Charges:
    """Charged seconds per line. Barrier lines: `lost_s` when charged by default, unless the
    grader overruled it: `home_earlier` charges its `charged_s`, `partner` and `misplaced`
    nothing. Of that, the line's `split` seconds (less any time `home_earlier` took off) go to
    the episode holding the split's contact instead, when the split applies
    (`split_applies`). Heaped trips (`heaped`: trip line to its `heaped_at`,
    `check_calls.hopper_ruling`): from `heaped_at` to the end of the trip, less barrier lines and
    opponent episodes inside it, the time to get home from there (`tracks.home_allow`) and
    TRIP_GRACE_S. Readiness: `late_s` when charged by default or called `not_home`, `charged_s`
    when `home_earlier` (or `not_home` with it), else 0; a `not_judged` line no call settled is
    not scored."""
    opp = [
        (k, s)
        for k, e in enumerate(episodes)
        if e.verdict == "opponent_on_target" and (s := episode_span(e, by_id)) is not None
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
        if s is not None and split_applies(d, call):
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
        excused = union_s(a, d.t_end, [*held, *(s for _, s in opp)])
        allow = home_allow(target, alliance, a) + TRIP_GRACE_S
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


def split_applies(d: DrivingLine, call: Call | None) -> bool:
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
    active = own_active(match)
    home = [(g.t_start, g.t_end) for g in gaps if home_gap(g)]
    unknown = [(g.t_start, g.t_end) for g in gaps if not home_gap(g)]
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
                unseen_s=round(union_s(f.t_start, f.t_end, unknown), 1),
                home_s=round(union_s(f.t_start, f.t_end, home), 1),
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
            Untracked(fact=g.id, t_start=g.t_start, t_end=g.t_end, inferred_home=home_gap(g))
            for g in gaps
            if any(overlap(g.t_start, g.t_end, a, b) for a, b in active)
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
