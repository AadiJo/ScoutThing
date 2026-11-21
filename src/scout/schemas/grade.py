"""`grade.json`, written by the grade stage: absolute 1-10 scores with evidence.

Every note must cite at least one fact id or frame; the schema rejects one that cites nothing.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from scout.schemas.common import StageFile

Category = Literal["driving", "shift_readiness", "defense"]


class Note(BaseModel):
    """A timestamped observation. `cites` holds fact ids from facts.json or `frame:<n>`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    t: float
    text: str
    cites: list[str] = Field(min_length=1)


class Score(BaseModel):
    """One category score on the absolute scale, with the notes that justify it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    category: Category
    score: int = Field(ge=1, le=10)
    summary: str
    notes: list[Note] = Field(min_length=1)


class Grade(StageFile):
    """The grade for the target team. Auto is described but not scored."""

    schema_version: Literal[1] = 1
    team: int
    scores: list[Score]
    auto_notes: list[Note]
