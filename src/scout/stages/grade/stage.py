"""The grade stage (`STAGE`, `grade`): build the ledgers and prompts, run the sealed session,
check its answer and resume it with the problems up to ATTEMPTS times, then write grade.json,
grade.md and `grade_work/`."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from scout.camera import Camera
from scout.run import RunDir, Stage
from scout.schemas.grade import (
    SKILLS,
    DrivingLine,
    Grade,
    Note,
    RunRecord,
    ScoredCall,
    ScoredEpisode,
    ScoredSkill,
)
from scout.stages.grade.base import IMAGE_CAP, JSON
from scout.stages.grade.base import work as _work
from scout.stages.grade.check import check_answer, retry_prompt
from scout.stages.grade.check_calls import hopper_ruling
from scout.stages.grade.checker import grade_context
from scout.stages.grade.claude import EFFORT, as_dict, claude, command, run_claude
from scout.stages.grade.clock import clock as _clock
from scout.stages.grade.clock import home_view
from scout.stages.grade.frames import served
from scout.stages.grade.prompt import prompts, rubric_version
from scout.stages.grade.prompt import text as _text
from scout.stages.grade.report import render_md
from scout.stages.grade.scores import coverage_text, split_applies

ATTEMPTS = 3
"""Answers checked per grade: the first and up to two retries in the same session, each told
exactly what failed. The checks are strict, so one retry was not always enough."""


def _grade_md(run: RunDir) -> Path:
    return run.path / "grade.md"


def _usage(results: list[JSON]) -> tuple[int, int, int, int, float | None, int, list[str]]:
    """Summed input, output, cache read, cache write tokens, the cost (the last result's
    total: the retry resumes the same session), turns and models."""
    ins = outs = reads = writes = turns = 0
    cost: float | None = None
    models: list[str] = []
    for r in results:
        u = as_dict(r.get("usage"))
        ins += _int(u.get("input_tokens"))
        outs += _int(u.get("output_tokens"))
        reads += _int(u.get("cache_read_input_tokens"))
        writes += _int(u.get("cache_creation_input_tokens"))
        turns += _int(r.get("num_turns"))
        c = r.get("total_cost_usd")
        if isinstance(c, int | float):
            cost = float(c)  # a resumed session reports its running total
        models += [m for m in as_dict(r.get("modelUsage")) if m not in models]
    return ins, outs, reads, writes, cost, turns, models


def _int(v: object) -> int:
    return int(v) if isinstance(v, int | float) else 0


def grade(run: RunDir) -> None:
    exe = claude()
    model = run.settings().grader_model
    ctx = grade_context(run)
    match = ctx.match
    work = _work(run)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    home = home_view(Camera.load(run.calibration_json, "main"), match)
    system, task = prompts(
        match, ctx.facts, run.facts_json.read_text(), ctx.led, home, ctx.coverage
    )
    (work / "system_prompt.md").write_text(system)
    (work / "task_prompt.md").write_text(task)
    start = time.perf_counter()

    first = run_claude(
        command(exe, model, run, system, None), task, run.path, work / "session-1.jsonl"
    )
    results = [first.result]
    session = first
    rejection: list[str] | None = None
    for attempt in range(1, ATTEMPTS + 1):
        sheets = served(run)
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
        session = run_claude(
            command(exe, model, run, system, session.session_id),
            retry_prompt(checked.problems),
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
            call, at = hopper_ruling(r.tiles)
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
                    and split_applies(line, None)
                    else ""
                ),
                cites=c.cites if c else [],
            )
        )
    result = Grade(
        team=match.target_team,
        rubric_version=rubric_version(_text("rubric.md")),
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
