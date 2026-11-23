"""The grader's system prompt and task (`prompts`): `grade_system.md` and `grade_task.md` in
this package, the game guide, limits and rubric under GAME_DIR, and the match, coverage,
ledgers and facts filled into the task."""

from __future__ import annotations

import json
from importlib import resources

from scout.schemas.facts import ContactFact, Fact, Facts
from scout.schemas.grade import Coverage, DrivingLine
from scout.schemas.match import MatchInfo
from scout.stages.grade.base import IMAGE_CAP, fmt, overlap
from scout.stages.grade.check_episodes import pressed as _pressed
from scout.stages.grade.clock import Corner, clock, phase, relation
from scout.stages.grade.ledgers import Ledgers
from scout.stages.grade.scores import coverage_text

GAME_DIR = "rebuilt2026_grade"
"""Folder under `scout.games` with the game guide, limits and rubric for this game."""


def _ledger_text(
    led: Ledgers, match: MatchInfo, home: Corner, by_id: dict[str, Fact]
) -> tuple[str, str, str]:
    """The driving ledger, readiness ledger and defense items as prompt text. Lines that
    depend on a moment the target was untracked point at the corner view of its own end."""

    def span(a: float, b: float) -> str:
        return f"{a:.1f}-{b:.1f} s ({clock(match, a)[1]}, {phase(match, a)})"

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
                f"; OPPONENT SPLIT {fmt(s.seconds)} s of it count against handling_defense "
                f"({s.why}): {s.contact}'s episode is opponent_on_target"
                if s.team is not None
                else f"; UNLABELLED SPLIT {fmt(s.seconds)} s ({s.why}): they count against "
                f"handling_defense only if a tile identifies that robot as an opponent (call "
                f"opponent)"
            )
        return (
            f"- {d.id} {d.kind} {span(d.t_start, d.t_end)}: {rule}; lost_s {fmt(d.lost_s)}, "
            f"unseen_s {fmt(d.unseen_s)}{''.join(f' ({x})' for x in flags)}; facts "
            f"{', '.join(d.facts)}; {d.text}; contacts with opponent or unlabelled robots: "
            f"{', '.join(d.opponent_contacts) or 'none'}; with partners only: "
            f"{', '.join(d.partner_contacts) or 'none'}"
        )

    drive = "\n".join(drive_line(d) for d in led.driving)
    ready = "\n".join(
        f"- {r.id} {r.fact} at {r.t:.1f} s ({clock(match, r.t)[1]}): DEFAULT {r.default} "
        f"({r.why}); late_s {fmt(r.late_s)}{' (bound)' if r.late_is_bound else ''}; {r.text}"
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
        f"{f', pin_s {fmt(i.pin_s)}' if i.pin_s > 0 else ''}{pressed(i.fact)}: {i.text}"
        for i in led.items
    )
    return drive or "(no lines)", ready or "(no lines)", items or "(no items)"


def text(name: str) -> str:
    """A prompt file: `grade_*.md` in this package, or a game file under GAME_DIR."""
    if name.startswith("grade_"):
        return resources.files("scout.stages.grade").joinpath(name).read_text()
    return resources.files("scout.games").joinpath(GAME_DIR, name).read_text()


def rubric_version(rubric: str) -> str:
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
        f"({', '.join(p.name for p in own if overlap(u.t_start, u.t_end, p.t_start, p.t_end))}): "
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


def prompts(
    match: MatchInfo, facts: Facts, facts_json: str, led: Ledgers, home: Corner, cov: Coverage
) -> tuple[str, str]:
    """System prompt (rules, game, limits, rubric) and task (match, coverage, ledgers and
    facts)."""
    system = "\n\n".join(
        [
            text("grade_system.md").replace("{{image_cap}}", str(IMAGE_CAP)),
            text("game.md"),
            text("limits.md"),
            text("rubric.md"),
        ]
    )
    roster = ", ".join(
        f"{s.team} ({s.alliance} {s.station}, "
        f"{'target' if s.team == match.target_team else relation(match, s.team)})"
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
    task = text("grade_task.md")
    for key, value in fill.items():
        task = task.replace("{{" + key + "}}", value)
    return system, task
