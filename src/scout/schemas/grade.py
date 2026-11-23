"""`grade.json`, written by the grade stage: absolute 1-10 driving scores with evidence.

Before the grader runs, code builds two ledgers from facts.json: the driving ledger (every
turned-back crossing, stop or stay on a barrier and slow crossing in teleop, with the seconds
it cost) and the readiness ledger (every real hub activation, with how late the target was).
It also lists the defense items: every teleop contact with a robot not known to be a partner,
and every gap or inferred stop in the target's own active time with such a robot near.

A coding agent writes `GradeAnswer` (its JSON schema is handed to `claude --json-schema`):
one call per ledger line (charged or not, and why), defense episodes that account for every
defense item, the three scores, notes and an auto report. The stage checks it, computes each
skill's band from the calls and episodes (the score must sit inside it), resolves the
`{...}` references in the text, and wraps it into `Grade`.

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


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --- ledgers (computed by code) ---------------------------------------------------------

DrivingKind = Literal["turned_back", "barrier_stop", "barrier_stay", "slow_crossing"]


class DrivingLine(_Model):
    """One moment the driving rubric may charge, from the facts."""

    id: str
    """`D1`, `D2`, ... in time order."""
    kind: DrivingKind
    facts: list[str]
    """The fact behind the line first, then facts folded into it (a stop on the barrier
    during a turned-back crossing)."""
    t_start: float
    t_end: float
    phase: str | None
    own_hub_active: bool | None
    lost_s: float
    """Seconds the rubric charges if the line is charged: `seconds_to_across` of a turned-back
    crossing (its time on the barrier when it never got across), the stop or stay's duration,
    or a slow crossing's time on the barrier."""
    unseen_s: float
    """Part of the line's time inferred across tracking gaps."""
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
    """The target was in its own zone during its own active time: being pressed there is
    playing through, so an episode made only of such items costs 0 s."""
    text: str


# --- the agent's answer -----------------------------------------------------------------

Call = Literal["charged", "opponent", "hub_inactive", "partner", "frame_contradicts", "uncertain"]
"""`charged`: self-inflicted. `opponent`: an opponent held it (needs an overlapping
`opponent_on_target` episode). `hub_inactive`: staging while its hub was off (driving only,
needs `own_hub_active` false). `partner`: its own partner was in the way (driving only, needs a
frame). `frame_contradicts`: a frame shows the facts' shape is wrong (needs a frame in the
line's span). `uncertain`: the evidence cannot decide; counts half."""


class LineCall(_Model):
    """The grader's decision on one ledger line."""

    line: str = Field(description="Ledger line id, e.g. D1 or R2.")
    call: Call
    charged_s: float | None = Field(
        default=None,
        ge=0.0,
        description="Driving lines only: seconds charged when less than lost_s (a frame in "
        "the line's span must show why). Null charges lost_s.",
    )
    text: Text = Field(description="One sentence: what the evidence shows.")
    cites: Cites


Verdict = Literal[
    "opponent_on_target", "target_on_opponent", "partner", "pass_no_holdup", "could_not_tell"
]
IdentifiedBy = Literal["track_label", "bumper_number", "bumper_red", "bumper_blue", "none"]


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
        description="opponent_on_target only: seconds the opponent cost the target (0 when "
        "it kept shooting or kept its route). 0 for every other verdict.",
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
    view: Literal["target", "full"]
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

    schema_version: Literal[3] = 3
    team: int
    rubric_version: str
    skills: list[ScoredSkill]
    driving_ledger: list[DrivingLine]
    readiness_ledger: list[ReadinessLine]
    defense_items: list[DefenseItem]
    line_calls: list[LineCall]
    episodes: list[DefenseEpisode]
    notes: list[Note]
    auto: AutoReport
    run: RunRecord
