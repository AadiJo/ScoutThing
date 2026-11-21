"""`grade.json`, written by the grade stage: absolute 1-10 driving scores with evidence.

A coding agent writes `GradeAnswer` (its JSON schema is handed to `claude --json-schema`);
the stage checks every citation and wraps the answer into `Grade` with the match clock on
each note and a record of the run.

Citations (`cites`) are fact ids from facts.json (`stop-03`) or `frame:<video seconds>`
for a moment the frame tool really served (`frame:92.8`). The stage rejects anything
else; this schema only checks the shape.
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


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SkillGrade(_Model):
    """One skill on the absolute scale. `score` is None when the evidence cannot support
    one, and then `null_reason` says why."""

    skill: SkillName
    score: int | None = Field(ge=1, le=10)
    null_reason: str | None
    summary: str = Field(description="Two or three plain sentences on why this score.")
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
    text: str = Field(description="What happened and why it matters, in one or two sentences.")
    cites: Cites = Field(min_length=1)


class AutoReport(_Model):
    """What the robot's autonomous routine did. Described, never scored."""

    text: str
    cites: Cites = Field(min_length=1)


class GradeAnswer(_Model):
    """The agent's whole answer: one grade per skill, the notes, and the auto report."""

    skills: list[SkillGrade] = Field(min_length=3, max_length=3)
    notes: list[AgentNote] = Field(min_length=1)
    auto: AutoReport


class Note(AgentNote):
    """A note in grade.json: the agent's note plus the match clock the stage computed."""

    clock: str
    """Scoreboard clock at `t` ("1:20"), with `period` saying which clock it is."""
    period: str


class FrameSheet(_Model):
    """One contact sheet the frame tool served."""

    image: str
    """Path relative to the run folder."""
    view: Literal["target", "full"]
    times: list[float]
    """Video seconds of each tile, as the agent may cite them."""


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
    """The grade for the target team. Auto is reported but not scored."""

    schema_version: Literal[2] = 2
    team: int
    rubric_version: str
    skills: list[SkillGrade]
    notes: list[Note]
    auto: AutoReport
    run: RunRecord
