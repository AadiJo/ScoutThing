"""`check_answer` runs every check over one structured answer and computes the charges and
scores it implies; `retry_prompt` turns the problems into the message for the retry."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from scout.schemas.facts import AutoRouteFact, StopFact
from scout.schemas.grade import SKILLS, DrivingLine, FrameSheet, GradeAnswer, LineCall, SkillName
from scout.stages.grade.base import fmt, overlap
from scout.stages.grade.check_calls import check_call, check_hoppers, hopper_ruling
from scout.stages.grade.check_episodes import check_episodes
from scout.stages.grade.checker import (
    CHARGE_WORD,
    DOUBT_WORD,
    NEGATED,
    NONE_WORD,
    REF,
    SENTENCE,
    START_WORD,
    Checker,
    GradeContext,
    Item,
)
from scout.stages.grade.clock import own_active
from scout.stages.grade.scores import Charges, Scored, charges, episode_span, scores, split_applies

NOTE_REACH_S = 5.0
"""A note's `t` must be within this of something it cites."""


@dataclass(frozen=True)
class Checked:
    """A checked answer: the answer when it passed, every problem, the scores, the charges
    and every text field resolved (keyed by its location)."""

    answer: GradeAnswer | None
    problems: list[str]
    scores: dict[SkillName, Scored]
    charges: Charges | None
    resolved: dict[str, str]


def check_answer(raw: object, ctx: GradeContext, sheets: list[FrameSheet]) -> Checked:
    """Validate the agent's answer and compute the charges and scores it implies."""
    try:
        answer = GradeAnswer.model_validate(raw)
    except ValidationError as e:
        problems = [
            f"schema: {'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()
        ]
        return Checked(None, problems, {}, None, {})
    ck = Checker(ctx, sheets)
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
        resolved[at] = ck.resolve(Item(at, value, cites, bumpers or [], roles, scope, claims))

    m = ctx.match
    scopes: dict[SkillName, list[tuple[float, float]]] = {
        "driving": own_active(m),
        "shift_readiness": own_active(m),
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
    check_episodes(ck, answer, sheets, reread)
    check_hoppers(ck, answer, sheets)
    for c in answer.line_calls:
        line = ck.lines.get(c.line)
        if line is None:
            ck.bad(f"line call {c.line}: no ledger line {c.line}")
            continue
        check_call(ck, c, line)
    for d in led.driving:
        c = calls.get(d.id)
        s = d.split
        if s is None or not split_applies(d, c.call if c else None):
            continue
        holding = [e for e in answer.episodes if s.contact in e.facts]
        if not holding or any(e.verdict != "opponent_on_target" for e in holding):
            ck.bad(
                f"episodes: {s.contact} carries {d.id}'s OPPONENT SPLIT (a robot between the "
                "target and its own wall, measured): its episode must be opponent_on_target"
                + ("" if s.team is None else f", unless you call no_opponent on {d.id}")
            )

    # what code charges, then the text checks that lean on it
    rulings = {r.line: hopper_ruling(r.tiles) for r in answer.hoppers}
    heaped = {k: t for k, (_, t) in rulings.items() if t is not None}
    al = ctx.match.target_alliance
    ch = charges(led, calls, heaped, answer.episodes, ctx.target, al)
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
        if cost > 0 and (s := episode_span(e, ck.by_id)) is not None
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
    sc = scores(led, answer.episodes, ch, ctx.coverage)
    charged: dict[SkillName, bool] = {
        "driving": ch.driving_s > 0,
        "shift_readiness": any(ch.line.get(r.id, 0.0) > 0 for r in led.readiness),
        "handling_defense": True,
    }
    for g in answer.skills:
        for c in g.cites:
            ck.span(c, f"skill {g.skill}")
        text(f"skill {g.skill} summary", g.summary, g.cites, roles=False, scope=scopes[g.skill])
        flagged = (s := sc.get(g.skill)) is not None and s.low_coverage
        _summary_claims(ck, f"skill {g.skill} summary", g.summary, charged[g.skill], flagged)
    for c in answer.auto.cites:
        ck.span(c, "auto")
    text("auto", answer.auto.text, answer.auto.cites, roles=False,
         scope=[(m.auto_start_t, m.auto_end_t)])  # fmt: skip
    route = next((f.data for f in ck.by_id.values() if isinstance(f, AutoRouteFact)), None)
    begun = route is None or route.first_tracked is None or route.first_tracked_after_s > 0
    if begun and (hit := START_WORD.search(REF.sub(" ", answer.auto.text))) is not None:
        when = f"{fmt(route.first_tracked_after_s)} s into auto" if route else "never"
        ck.bad(
            f"auto: {hit.group(0)!r} says where it started, but it was first tracked {when}: "
            "say where it was first tracked and when ({auto-route.first_tracked_after_s} s into "
            "auto, citing auto-route), not where it started"
        )
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
            if any(overlap(d.t_start, d.t_end, a, b) for a, b in spans)
        ]
        if n.kind in ("went_well", "played_through_defense") and hit:
            ck.bad(f"{at}: kind {n.kind} over {', '.join(hit)}, which is charged")
        cleared = [
            i.id
            for i, _ in ck.cleared_lines
            if any(overlap(i.t_start, i.t_end, a, b) for a, b in spans)
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
    return Checked(answer if not problems else None, problems, sc, ch, resolved)


def _summary_claims(ck: Checker, at: str, summary: str, charged: bool, flagged: bool) -> None:
    """A skill summary says something was charged or cost only when its skill charged seconds,
    and doubts its score ("the true score may be lower") only when code flagged the score as
    low coverage: unseen time is never charged, and the coverage sentence says what was seen."""
    for sentence in [] if charged else SENTENCE.split(REF.sub("#", summary)):
        hit = CHARGE_WORD.search(sentence)
        if hit and not NEGATED.search(sentence[: hit.start()]) and not NONE_WORD.search(sentence):
            ck.bad(
                f"{at}: {hit.group(0)!r} says something was charged, but no line of this skill "
                "charged any seconds: say nothing was charged"
            )
            break
    if not flagged and (hit := DOUBT_WORD.search(summary)) is not None:
        ck.bad(
            f"{at}: {hit.group(0)!r} doubts the score, but code did not flag it as low "
            "coverage; unseen time is never charged and the coverage sentence already says "
            "what was seen: drop the doubt"
        )


def retry_prompt(problems: list[str]) -> str:
    return (
        "Your answer was rejected by the checker. Fix exactly these problems and return the "
        "whole answer again (you may open more frames within the budget):\n"
        + "\n".join(f"- {p}" for p in problems)
    )
