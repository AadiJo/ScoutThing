"""`grade.json`, written by the grade stage: absolute 1-10 driving scores with evidence.

Before the grader runs, code builds two ledgers from facts.json: the driving ledger (every
turned-back attempt toward home, stop or stay on a barrier, slow crossing and trip out of its
own zone in its own active time, with the active seconds it could cost) and the readiness
ledger (every real hub activation, with how late the target was). It also lists the defense
items: every teleop contact with a robot not known to be a partner, and every gap or inferred
stop in the target's own active time with such a robot near.

A coding agent writes `GradeAnswer` (its JSON schema is handed to `claude --json-schema`):
one call per ledger line (charged or not, and why; a hopper reading for each trip), defense
episodes that account for every defense item, the three scores, notes and an auto report. The
stage checks it, computes each line's charged seconds and each episode's cost, and from them
each skill's band (the score must sit inside it), resolves the `{...}` references in the text,
and wraps it into `Grade`.

Citations (`cites`) are fact ids from facts.json (`stop-03`) or `frame:<video seconds>`
for a moment the frame tool really served (`frame:92.8`). Free text may not contain numbers:
it writes `{<fact id>.<field>}`, `{<ledger line>.<field>}` or `{frame:<t>}`, which the stage
replaces with the value. This schema only checks the shape; `scout.stages.grade` checks the
rest.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

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


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- ledgers (computed by code) ---------------------------------------------------------

DrivingKind = Literal["turned_back", "barrier_stop", "barrier_stay", "slow_crossing", "loaded_trip"]


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
    or slow crossing's `excess_s`). `loaded_trip`: the trip's seconds out of its own zone in
    its own active time, the most it could cost."""
    bound: bool
    """True when `lost_s` is an upper bound: the arrival home was first seen after a gap."""
    unseen_s: float
    """Part of the line's time inferred across tracking gaps."""
    corner_reach: bool
    """A turned-back attempt where only a corner of the robot could have touched the
    barrier: it may be a curving drive past it with no stall."""
    look_at: list[float]
    """`loaded_trip` only: the tile times at which to read the hopper (served at zoom 2)."""
    opponent_contacts: list[str]
    """Contact facts in the line's span with an opponent or unlabelled robot."""
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
    text: str


class DefenseItem(_Model):
    """A fact the defense episodes must account for."""

    fact: str
    t_start: float
    t_end: float
    home_active: bool
    """The target was in its own zone during its own active time: an opponent pressing it
    there costs only its pin time (`pin_s`), never route time."""
    pin_s: float
    """Seconds the target was nearly still in this item (inferred still time counted half);
    an episode's pin cost is its items' sum beyond the rubric's allowance."""
    text: str


# --- the agent's answer -----------------------------------------------------------------

Call = Literal[
    "charged",
    "opponent",
    "partner",
    "frame_contradicts",
    "uncertain",
    "heaped",
    "not_heaped",
    "cannot_see",
]
"""Barrier and readiness lines: `charged` (self-inflicted), `opponent` (an opponent held it;
needs an overlapping `opponent_on_target` episode), `partner` (its own partner was in the way;
barrier lines only, needs a tile and a partner contact), `frame_contradicts` (a tile shows the
facts' shape is wrong), `uncertain` (the evidence cannot decide; counts half).
`loaded_trip` lines: `heaped` (a tile shows a heaped hopper; `heaped_at` says from when),
`not_heaped` (empty or part full on the trip's late tiles), `cannot_see` (the hopper cannot
be read on any tile), or `frame_contradicts`."""


class LineCall(_Model):
    """The grader's decision on one ledger line."""

    line: str = Field(description="Ledger line id, e.g. D1 or R2.")
    call: Call
    charged_s: float | None = Field(
        default=None,
        ge=0.0,
        description="Barrier lines whose lost_s is an upper bound only: the seconds from the "
        "line's start to a cited tile that shows the target home, when that is earlier. Null "
        "charges lost_s.",
    )
    heaped_at: float | None = Field(
        default=None,
        description="heaped only: video time of the earliest cited tile in the trip where the "
        "hopper is heaped.",
    )
    corner_bumper: int | None = Field(
        default=None,
        description="Required when the call cites a corner camera tile: the target's bumper "
        "number as you read it there.",
    )
    text: Text = Field(description="One sentence: what the evidence shows.")
    cites: Cites


Verdict = Literal["opponent_on_target", "target_on_opponent", "partner", "apart", "could_not_tell"]
IdentifiedBy = Literal["track_label", "bumper_number", "bumper_red", "bumper_blue", "none"]
WhileHeld = Literal["not_shooting", "shooting", "cannot_see"]


class DefenseEpisode(_Model):
    """One episode that accounts for one or more defense items."""

    facts: list[str] = Field(min_length=1, description="Defense item fact ids it covers.")
    verdict: Verdict
    other_team: int | None = Field(description="The other robot's team, when identified.")
    identified_by: IdentifiedBy = Field(
        description="track_label: a team label in a cited fact or a box on a cited tile. "
        "bumper_number: you read the number on a cited tile. bumper_red / bumper_blue: you saw "
        "a clearly red or blue bumper on a cited tile. none: not identified."
    )
    cost_s: float = Field(
        ge=0.0,
        description="opponent_on_target only: seconds the opponent cost it in getting home or "
        "onto its route (0 when it kept its route, and 0 when home in its own active time). "
        "Pin time is computed by code from pin_s and while_held. 0 for every other verdict.",
    )
    while_held: WhileHeld | None = Field(
        default=None,
        description="opponent_on_target with pin_s in its items: what the tiles across the "
        "still stretch show. Null otherwise.",
    )
    text: Text
    cites: Cites = Field(min_length=1)


class SkillGrade(_Model):
    """One skill on the absolute scale. `score` is None when the evidence cannot support
    one, and then `null_reason` says why."""

    skill: SkillName
    score: int | None = Field(ge=1, le=10)
    null_reason: Text | None
    summary: Text = Field(description="Two or three plain sentences on why this score.")
    cites: Cites = Field(min_length=1)

    @model_validator(mode="after")
    def _reason_iff_null(self) -> Self:
        if (self.score is None) != (self.null_reason is not None):
            raise ValueError("null_reason is required exactly when score is null")
        return self


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

    line_calls: list[LineCall] = Field(description="Exactly one per ledger line.")
    episodes: list[DefenseEpisode] = Field(
        description="Together they cover every defense item fact id."
    )
    skills: list[SkillGrade] = Field(min_length=3, max_length=3)
    notes: list[AgentNote] = Field(min_length=1)
    auto: AutoReport


# --- grade.json -------------------------------------------------------------------------


class ScoredSkill(SkillGrade):
    """A skill in grade.json: the agent's grade plus the band the stage computed."""

    band: tuple[int, int] | None
    """Lowest and highest score the rubric allows from the calls and episodes; None when the
    score must be null."""


class ScoredCall(LineCall):
    """A line call in grade.json, with the seconds code charged for it."""

    charged_total_s: float
    """Seconds this call adds to the driving total or is the readiness lateness."""


class ScoredEpisode(DefenseEpisode):
    """An episode in grade.json, with its cost as code computed it."""

    pin_cost_s: float
    """Its items' `pin_s` beyond the allowance, scaled by `while_held`."""
    total_cost_s: float
    """The larger of `cost_s` and `pin_cost_s`: what the defense score counts."""


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

    schema_version: Literal[4] = 4
    team: int
    rubric_version: str
    skills: list[ScoredSkill]
    driving_ledger: list[DrivingLine]
    readiness_ledger: list[ReadinessLine]
    defense_items: list[DefenseItem]
    line_calls: list[ScoredCall]
    episodes: list[ScoredEpisode]
    notes: list[Note]
    auto: AutoReport
    run: RunRecord
