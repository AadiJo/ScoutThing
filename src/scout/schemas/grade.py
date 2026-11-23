"""`grade.json`, written by the grade stage: absolute 1-10 driving scores with evidence.

Before the grader runs, code builds two ledgers from facts.json: the driving ledger (every
turned-back attempt toward home, stop or stay on a barrier, slow crossing and trip out of its
own zone in its own active time, with the active seconds it could cost) and the readiness
ledger (every real hub activation, with how late the target was). Each barrier and readiness
line carries a default ruling that code computed from measured features, and the rule that
set it. Code also lists the defense items: every teleop contact with a robot not known to be
a partner, and every gap or inferred stop in the target's own active time with such a robot
near.

A coding agent writes `GradeAnswer` (its JSON schema is handed to `claude --json-schema`): a
hopper reading for each trip, an overrule only where a tile shows a checkable contradiction
of a default, defense episodes that account for every defense item, a summary per skill,
notes and an auto report. The stage checks it, computes each line's charged seconds, each
episode's cost and each skill's score, resolves the `{...}` references in the text, and wraps
it into `Grade`.

Citations (`cites`) are fact ids from facts.json (`stop-03`) or `frame:<video seconds>`
for a moment the frame tool really served (`frame:92.8`). Free text may not contain numbers:
it writes `{<fact id>.<field>}`, `{<ledger line>.<field>}` or `{frame:<t>}`, which the stage
replaces with the value. This schema only checks the shape; `scout.stages.grade` checks the
rest.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from scout.schemas.common import StageFile

SkillName = Literal["driving", "shift_readiness", "handling_defense"]
SKILLS: tuple[SkillName, ...] = ("driving", "shift_readiness", "handling_defense")
NoteKind = Literal[
    "went_well", "went_wrong", "played_through_defense", "outside_driver_control", "observation"
]
Cites = list[str]
Text = str
"""Free text: no digits except shift numbers and team numbers the grader may name; every
number comes from a `{...}` reference to a fact field, a ledger field or a served frame."""
View = Literal["target", "full", "corner_left", "corner_right"]
"""Frame tool views: a crop of the main camera around the target, the whole main camera, or
one of the two uncalibrated corner camera panels (no ring, no boxes)."""
Ruling = Literal["charged", "cleared"]
"""A ledger line's default: `charged` costs its `lost_s` (readiness: `late_s`), `cleared`
costs nothing."""
ReadinessRuling = Literal["charged", "cleared", "not_judged"]
"""A readiness line's default; `not_judged`: where the robot was at the activation is unknown,
so the line costs nothing and does not count toward the score unless a tile settles it."""


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- ledgers (computed by code) ---------------------------------------------------------

DrivingKind = Literal["turned_back", "barrier_stop", "barrier_stay", "slow_crossing", "loaded_trip"]


class OpponentSplit(_Model):
    """The part of a charged barrier line an opponent cost: a robot in a contact inside the
    line came between the target and its own wall. Code computes it from positions."""

    contact: str
    """The contact fact (the earliest such one in the line)."""
    team: int | None
    """That robot's team label; None when unlabelled: then the split applies only when the
    grader establishes it as an opponent on a tile (`opponent`). A labelled opponent's split
    applies by default."""
    t: float
    """When that robot came close (or the line's start, if later): `seconds` run from here."""
    seconds: float
    """Seconds of the line's `lost_s` that count against handling_defense instead of driving."""
    why: str
    """The measures and thresholds that set `seconds`."""


class DrivingLine(_Model):
    """One stretch the driving rubric may charge, from the facts."""

    id: str
    """`D1`, `D2`, ... in time order."""
    kind: DrivingKind
    facts: list[str]
    """The fact behind the line first, then facts folded into it (a stop during a turned-back
    attempt)."""
    t_start: float
    t_end: float
    phase: str | None
    lost_s: float
    """Barrier lines: seconds of its own active time the line costs when charged (a
    turned-back attempt's `own_active_s`, a stop's `own_active_s`, the active part of a stay's
    or slow crossing's `excess_s`); a turned-back attempt's less the lateness a readiness line
    already charges.
    `loaded_trip`: the trip's seconds out of its own zone in its own active time, the most it
    could cost."""
    bound: bool
    """True when `lost_s` is an upper bound: the arrival home was first seen after a gap."""
    unseen_s: float
    """Part of the line's time inferred across tracking gaps."""
    default: Ruling
    """What the line costs unless the grader overrules it on a tile. `loaded_trip` lines are
    `cleared` until a hopper reading charges them."""
    why: str
    """The measured features and thresholds that set `default`."""
    look_at: list[float]
    """`loaded_trip` only: the tile times at which to read the hopper (served at zoom 2)."""
    opponent_contacts: list[str]
    """Contact facts in the line's span with an opponent or unlabelled robot."""
    split: OpponentSplit | None
    """Charged barrier lines: the seconds a robot between it and home cost (see
    `OpponentSplit`); None when no such robot was measured."""
    partner_contacts: list[str]
    """Contact facts in the line's span with only labelled partners."""
    text: str
    """What the facts say, in one line."""


class ReadinessLine(_Model):
    """One real activation of the target's hub (a `hub_on` fact)."""

    id: str
    """`R1`, `R2`, ..."""
    fact: str
    phase: str
    t: float
    in_zone: bool | None
    """In its own zone at the moment; None when untracked then."""
    late_s: float
    """Seconds until first tracked in its own zone (0 when in it); the phase's length when
    it never got there."""
    late_is_bound: bool
    """True when the arrival was first seen after a gap, so it was at most `late_s` late."""
    never_home: bool
    default: ReadinessRuling
    """`cleared` when tracked in its own zone, untracked inside a gap whose `inference` puts it
    there, or put home by then by a crossing or inference; `not_judged` when untracked with its
    last sighting in its own zone and nothing putting it home; otherwise `charged`
    (`late_s`)."""
    why: str
    text: str


class DefenseItem(_Model):
    """A fact the defense episodes must account for."""

    fact: str
    t_start: float
    t_end: float
    home_active: bool
    """The target was in its own zone during its own active time."""
    pin_s: float
    """Seconds the target was nearly still in this item (inferred still time counted half);
    held time beyond the rubric's allowance costs."""
    text: str


class PhaseCoverage(_Model):
    """One teleop phase with its own hub active: how much of it the target was tracked."""

    phase: str
    duration_s: float
    tracked_share: float
    unseen_s: float
    """Untracked in gaps that nothing puts in its own zone: where it was is unknown."""
    home_s: float
    """Untracked in gaps whose `inference` puts it in its own zone."""


class Untracked(_Model):
    """A gap fact in its own active time."""

    fact: str
    t_start: float
    t_end: float
    inferred_home: bool


class Coverage(_Model):
    """What the scores rest on, computed by code from facts.json."""

    own_active_s: float
    """Teleop seconds with its own hub active: what driving is judged on."""
    own_active_tracked_share: float
    own_active_unseen_share: float
    """Share of that time untracked in gaps that nothing puts in its own zone."""
    phases: list[PhaseCoverage]
    untracked: list[Untracked]
    activations: int
    activations_judged: int
    match_tracked_share: float
    auto_tracked_share: float
    others_tracked_mean: float
    """Other robots (of five) tracked per frame over the match, on average."""


# --- the agent's answer -----------------------------------------------------------------

Call = Literal[
    "opponent",
    "no_opponent",
    "partner",
    "misplaced",
    "home_earlier",
    "home",
    "not_home",
    "heaped",
    "not_heaped",
    "cannot_see",
]
"""Overrules of a barrier or readiness line's default, each needing a checkable contradiction:
`opponent` (barrier lines: the unlabelled robot of the line's `split` is established as an
opponent on a tile, so the split applies; readiness lines: an opponent held it),
`no_opponent` (a labelled robot's split: its bumper read on a tile, `other_bumper`, is not an
opponent's, so the split does not apply), `partner` (barrier lines: its own partner was
in the way; needs a tile and a partner contact), `misplaced` (barrier lines charged by
default: ringed tiles show the facts put the target in the wrong place, or the ring on another
robot), `home_earlier` (a line whose time is an upper bound: a tile shows it home sooner;
`charged_s` gives the seconds), `home` (a `not_judged` readiness line: a tile at the activation
shows it in its zone, its bumper read), `not_home` (a readiness line cleared by a gap inference,
or `not_judged` with its bumper read: a tile at the activation shows it out of its zone;
`charged_s`, optional, the seconds to a cited tile showing it home). `loaded_trip` lines
(always called): `heaped` (`heaped_at` says from when), `not_heaped` (empty or part full on its
late tiles) or `cannot_see`."""


class LineCall(_Model):
    """The grader's decision on one ledger line: every trip, and only the barrier and
    readiness lines whose default it overrules."""

    line: str = Field(description="Ledger line id, e.g. D1 or R2.")
    call: Call
    charged_s: float | None = Field(
        default=None,
        ge=0.0,
        description="home_earlier (and optionally not_home): seconds from the line's start "
        "(readiness: the activation) to the cited tile that shows the target home.",
    )
    heaped_at: float | None = Field(
        default=None,
        description="heaped only: video time of the earliest cited tile in the trip from "
        "which every later hopper tile is heaped.",
    )
    corner_bumper: int | None = Field(
        default=None,
        description="Required when the call cites a corner camera tile, or calls home or "
        "not_home on a not_judged line: the target's bumper number as you read it there.",
    )
    other_bumper: int | None = Field(
        default=None,
        description="no_opponent only: the team number you read on the bumper of the robot the "
        "line's OPPONENT SPLIT names, on a cited tile where the tool boxed it beside the target.",
    )
    text: Text = Field(description="One sentence: what the tiles show.")
    cites: Cites


Verdict = Literal["opponent_on_target", "target_on_opponent", "partner", "apart", "could_not_tell"]
IdentifiedBy = Literal["track_label", "bumper_number", "bumper_red", "bumper_blue", "none"]


class DefenseEpisode(_Model):
    """One episode that accounts for one or more defense items. Its cost is computed by code."""

    facts: list[str] = Field(min_length=1, description="Defense item fact ids it covers.")
    verdict: Verdict
    other_team: int | None = Field(description="The other robot's team, when identified.")
    identified_by: IdentifiedBy = Field(
        description="track_label: a team label in a cited fact or a box on a cited tile. "
        "bumper_number: you read the number on a cited tile. bumper_red / bumper_blue: you saw "
        "a clearly red or blue bumper on a cited tile. none: not identified."
    )
    text: Text
    cites: Cites = Field(min_length=1)


class SkillSummary(_Model):
    """Why one skill scored what code computed."""

    skill: SkillName
    summary: Text = Field(description="Two or three plain sentences on what drove this skill.")
    cites: Cites = Field(min_length=1)


class AgentNote(_Model):
    """One timestamped moment as the agent writes it."""

    t: float = Field(description="Video seconds of the moment.")
    kind: NoteKind
    skill: SkillName | None = Field(description="The skill this bears on, or null.")
    text: Text = Field(description="What happened and why it matters, in one or two sentences.")
    cites: Cites = Field(min_length=1)
    bumpers_read: list[int] = Field(
        default_factory=list[int],
        description="Teams named in the text whose bumper number you read on a cited tile.",
    )


class AutoReport(_Model):
    """What the robot's autonomous routine did. Described, never scored."""

    text: Text
    cites: Cites = Field(min_length=1)


class GradeAnswer(_Model):
    """The agent's whole answer."""

    line_calls: list[LineCall] = Field(
        description="One per loaded_trip line, plus one per barrier or readiness line whose "
        "default you overrule."
    )
    episodes: list[DefenseEpisode] = Field(
        description="Together they cover every defense item fact id."
    )
    skills: list[SkillSummary] = Field(min_length=3, max_length=3)
    notes: list[AgentNote] = Field(min_length=1)
    auto: AutoReport


# --- grade.json -------------------------------------------------------------------------


class ScoredSkill(SkillSummary):
    """A skill in grade.json: the agent's summary plus the score code computed."""

    score: int | None
    """1-10 from the charged seconds (see rubric.md); None when the rubric requires it."""
    band: tuple[int, int] | None
    """The rubric's two-score band around `score`; None when `score` is None."""
    null_reason: str | None
    """Why `score` is None."""
    coverage: str
    """What the score rests on, in one sentence a person can read (computed by code)."""
    low_coverage: bool
    """The score stands on part of the match only: unseen time can only add charges, so the
    true score may be lower."""


class ScoredCall(_Model):
    """One ledger line's outcome in grade.json: its default, the grader's call if any, and the
    seconds code charged."""

    line: str
    default: ReadinessRuling
    call: Call | None
    """The grader's call; None when it kept the default."""
    charged_total_s: float
    """Seconds this line adds to the driving total, or the readiness lateness."""
    charged_s: float | None = None
    heaped_at: float | None = None
    corner_bumper: int | None = None
    text: str
    """The grader's sentence, or the default's rule when it kept the default."""
    cites: Cites


class ScoredEpisode(DefenseEpisode):
    """An episode in grade.json, with its cost as code computed it."""

    pin_cost_s: float
    """Its share of held still time beyond the allowance (see rubric.md)."""
    total_cost_s: float
    """`pin_cost_s` plus the `split` seconds of barrier lines whose split contact it holds:
    what the defense score counts."""


class Note(AgentNote):
    """A note in grade.json: the agent's note with its references resolved, plus the match
    clock the stage computed."""

    clock: str
    """Scoreboard clock at `t` ("1:20"), with `period` saying which clock it is."""
    period: str


class TileRecord(_Model):
    """One tile of a served sheet."""

    t: float
    """Video seconds, as the agent may cite it."""
    tracked: bool
    """The target was tracked at this moment and ringed (main-camera tiles only)."""
    teams_marked: list[int]
    """Labelled teams the tool boxed on this tile."""
    teams_near: list[int]
    """Those boxed within 2.5 m of the target (when it is tracked there)."""
    unlabelled_marked: int
    """Unlabelled tracked robots boxed on this tile."""


class FrameSheet(_Model):
    """One contact sheet the frame tool served."""

    image: str
    """Path relative to the run folder."""
    view: View
    zoom: int
    tiles: list[TileRecord]


class RunRecord(_Model):
    """How the grade was produced."""

    model: str
    """Requested model (`--grader-model`)."""
    models_used: list[str]
    """Models the CLI reports having called."""
    effort: str
    claude_version: str | None
    wall_s: float
    turns: int
    """Agent turns over every attempt."""
    images_served: int
    image_cap: int
    sheets: list[FrameSheet]
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost_usd: float | None
    """List-price cost the CLI reports; on a subscription plan nothing is billed per call."""
    retried: bool
    rejection: list[str] | None
    """What the first answer failed on, when a retry was needed."""


class Grade(StageFile):
    """The grade for the target team. Auto is reported but not scored. Text fields have
    their `{...}` references resolved."""

    schema_version: Literal[7] = 7
    team: int
    rubric_version: str
    coverage: Coverage
    skills: list[ScoredSkill]
    driving_ledger: list[DrivingLine]
    readiness_ledger: list[ReadinessLine]
    defense_items: list[DefenseItem]
    line_calls: list[ScoredCall]
    episodes: list[ScoredEpisode]
    notes: list[Note]
    auto: AutoReport
    run: RunRecord
