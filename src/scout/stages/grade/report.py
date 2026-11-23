"""grade.md: grade.json as a short report a person can read (`render_md`)."""

from __future__ import annotations

from scout.schemas.facts import Facts
from scout.schemas.grade import Grade
from scout.stages.grade.base import fmt


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
            f"| {d.id} {d.kind} | {d.t_start:.1f}-{d.t_end:.1f} | {fmt(d.lost_s)} | {d.default} | "
            f"{c.call or 'kept'} | {fmt(c.charged_total_s)} | {cell(c.text)} |"
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
            f"| {r.id} | {r.phase} | {fmt(r.late_s)} | {r.default} | {c.call or 'kept'} | "
            f"{fmt(c.charged_total_s)} | {cell(c.text)} |"
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
