"""Checks of the defense episodes (`check_episodes`): identification, verdicts that need an
established robot, cited tiles, and every defense item covered. `pressed` says why a contact
with an opponent cannot be `apart`; the prompt shows the same reason on the defense items."""

from __future__ import annotations

from collections.abc import Sequence

from scout.schemas.facts import ContactFact
from scout.schemas.grade import FrameSheet, GradeAnswer
from scout.stages.grade.base import IMAGE_CAP, fmt, overlap
from scout.stages.grade.checker import SPAN_PAD_S, Checker
from scout.stages.grade.clock import relation
from scout.stages.grade.scores import episode_span

TOUCH_M = 1.0
"""Centre distance under which two robots (0.95 m bumpers) were very probably touching."""
APART_STILL_S = 1.0
"""A contact with the target still this long cannot be called `apart`."""


def pressed(f: ContactFact, driving_spans: Sequence[tuple[float, float]]) -> str | None:
    """Why a contact with an opponent in it cannot be `apart` or `could_not_tell`, or None."""
    d = f.data
    return (
        "the target was in its own zone"
        if d.target_zone == "own" or d.where.zone == "own"
        else f"they came within {fmt(d.min_distance_m)} m (touching)"
        if d.min_distance_m < TOUCH_M
        else f"the target was still {fmt(d.longest_still_s)} s beside it"
        if d.longest_still_s >= APART_STILL_S
        else "it happened during a driving ledger line"
        if any(overlap(f.t_start, f.t_end, a, b) for a, b in driving_spans)
        else None
    )


def check_episodes(
    ck: Checker, answer: GradeAnswer, sheets: list[FrameSheet], reread: dict[str, int]
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
        span = episode_span(e, ck.by_id)
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
                rel = relation(ctx.match, team)
        elif e.identified_by == "bumper_number":
            if team is None or team not in {s.team for s in ctx.match.roster} or not frames:
                ck.bad(
                    f"{at}: bumper_number needs a roster team and a cited tile within "
                    f"{SPAN_PAD_S:g} s of the episode"
                )
            else:
                rel = relation(ctx.match, team)
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
                why = pressed(f, driving_spans)
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
