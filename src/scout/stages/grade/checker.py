"""The answer checker's state and its text checks. `GradeContext` is what every check runs
against. `Checker` collects the problems in one answer, resolves citations and who is who, and
checks each text field (`Checker.resolve`): references filled in, no typed numbers, team names
only when labelled or read, no purpose words, no reading against the ledger, no claim over
unseen time, and roles and places that agree with what was tracked."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import TypeAdapter

from scout.games.rebuilt2026 import (
    BARRIER_X,
    CENTER_LINE_X_M,
    HUBS,
    guardrail_m,
    own_wall_m,
    rect_distance_m,
    zone_of,
)
from scout.run import RunDir
from scout.schemas.common import Alliance
from scout.schemas.facts import ContactFact, CrossingTurnedBackFact, Fact, Facts, GapFact, StopFact
from scout.schemas.grade import (
    Coverage,
    DefenseEpisode,
    DrivingLine,
    FrameSheet,
    ReadinessLine,
    TileRecord,
)
from scout.schemas.match import MatchInfo
from scout.stages.grade.base import fmt, overlap, union_s
from scout.stages.grade.clock import clock, relation
from scout.stages.grade.ledgers import Ledgers, ledgers, stalled
from scout.stages.grade.scores import coverage, episode_span
from scout.stages.grade.tracks import Target, load_tracks

FRAME_MATCH_S = 0.051
SPAN_PAD_S = 2.0
"""A frame this close to an episode or ledger line counts as evidence for it."""
CENTRE_NEAR_M = 1.5
BARRIER_NEAR_M = 2.5
"""A place word is supported when the target was this close to the centre line, or to a
barrier's bumps (people say "at the red hub" for a robot a metre or two in front of it)."""
HUB_NEAR_M = 2.0
WALL_NEAR_M = 1.5
""""Beside its hub" needs the centre this close to its own hub's base; "at its wall" or "at the
guardrail" this close to that wall."""


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


@dataclass(frozen=True)
class GradeContext:
    """Everything `check_answer` checks against, built once per run."""

    facts: Facts
    match: MatchInfo
    led: Ledgers
    target: Target
    coverage: Coverage

    @property
    def by_id(self) -> dict[str, Fact]:
        return {f.id: f for f in self.facts.facts}


def grade_context(run: RunDir) -> GradeContext:
    """Load what the checker needs from a run folder."""
    match = MatchInfo.load(run.match_json)
    facts = Facts.load(run.facts_json)
    target, others = load_tracks(run, match)
    led = ledgers(facts, match, target, others)
    return GradeContext(facts, match, led, target, coverage(facts, match, led))


@dataclass
class Item:
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


class Checker:
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
        within `frames.NEAR_BOX_M` of the target)."""
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
            es = episode_span(e, self.by_id)
            if es is not None and any(overlap(a, b, es[0], es[1]) for a, b in spans):
                out.add(e.verdict)
        return out

    def _quiet_span(self, cite: str) -> tuple[float, float] | None:
        if cite.startswith("frame:"):
            t = self.frame_t(cite)
            return None if t is None else (t, t)
        f = self.by_id.get(cite)
        return None if f is None else (f.t_start, f.t_end)

    # text

    def resolve(self, item: Item) -> str:
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
                return clock(self.ctx.match, t)[1]
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
                return fmt(obj)
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

    def readings(self, item: Item, bare: str, has_frame: bool) -> None:
        """One stretch, one reading: no text calls fine what the ledger charged, or a fault
        what a tile cleared; and a claim that it kept moving needs a tile."""
        spans = self.spans_of(item.cites)

        def hits(a: float, b: float) -> bool:
            return any(overlap(a, b, s0, s1) for s0, s1 in spans)

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

    def unseen(self, item: Item, text: str) -> None:
        """No claim beyond what was seen: absence and count claims over time with untracked
        stretches say "seen" or "tracked" (and need half of it tracked); a quoted stop or hold
        that was partly inferred quotes its inferred part too; "stall" needs the robot measured
        stopped (`ledgers.stalled`); a stop at home in its own active time is not lost time."""
        at = item.at
        cited = [f for c in item.cites if (f := self.by_id.get(c)) is not None]
        scope = item.scope if item.scope is not None else self.spans_of(item.cites)
        gaps = [
            (g.t_start, g.t_end)
            for g in self.gaps
            if any(overlap(g.t_start, g.t_end, a, b) for a, b in scope)
        ]
        if gaps and item.claims:
            total = sum(b - a for a, b in scope)
            hidden = sum(union_s(a, b, gaps) for a, b in scope)
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
                stalled(f.data, self.ctx.facts.thresholds.get("STOP_MPS", 0.3)) for f in turned
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

    def roles(self, item: Item, bare: str) -> None:
        near = self.labelled(item.cites, near_only=True) | set(item.bumpers_read)
        rel = {relation(self.ctx.match, t) for t in near if t != self.ctx.match.target_team}
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

    def places(self, item: Item, bare: str) -> None:
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
                    f"at least {fmt(hub)} m from its hub, {fmt(wall)} m from its wall and "
                    f"{fmt(rail)} m from a guardrail (a stop or contact counts by its `where`); "
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
