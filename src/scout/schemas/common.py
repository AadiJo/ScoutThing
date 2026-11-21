"""Types shared by every stage file: the JSON base model, the parquet column spec, and
small value types.

Conventions used by every schema:
- field positions are WPILib field metres (blue alliance wall at x=0, x along the
  field's length, y across, z up), the frame of `assets/data/field/2026-rebuilt-welded.json`;
- pixel coordinates are full-frame pixels of the source video, origin top left;
- `t` is video seconds (`frame / fps`, first frame at 0) and `frame` the source frame index.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import polars as pl
from pydantic import BaseModel, ConfigDict

Alliance = Literal["red", "blue"]
ViewName = Literal["main", "corner_left", "corner_right"]
"""The three camera panels of the standard FIRST broadcast layout."""

VIEW_NAMES: tuple[ViewName, ...] = ("main", "corner_left", "corner_right")
ALLIANCES: tuple[Alliance, ...] = ("red", "blue")


def atomic_write(path: Path, data: bytes) -> None:
    """Write `data` to `path` through a temporary file, so readers never see half a file."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


class StageFile(BaseModel):
    """Base for every JSON stage file. Subclasses pin `schema_version` with a Literal.

    Use `Model.load(path)` and `model.save(path)`; unknown keys are rejected.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def load(cls, path: Path) -> Self:
        return cls.model_validate_json(path.read_bytes())

    def save(self, path: Path) -> None:
        atomic_write(path, self.model_dump_json(indent=2).encode() + b"\n")


class PixelRect(BaseModel):
    """Axis-aligned rectangle in full-frame pixels: left `x`, top `y`, width `w`, height `h`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    x: int
    y: int
    w: int
    h: int


class SchemaError(ValueError):
    """A stage file does not match its declared schema."""


@dataclass(frozen=True)
class TableSpec:
    """Declared columns of one parquet table. `read` and `write` both reject a frame
    whose column names or dtypes differ from the spec, and return/write columns in
    spec order. Enum columns also reject values outside their categories.
    """

    name: str
    columns: Mapping[str, pl.DataType]

    def check(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return `df` with columns in spec order, or raise SchemaError."""
        got = dict(df.schema)
        missing = [c for c in self.columns if c not in got]
        extra = [c for c in got if c not in self.columns]
        wrong = [
            f"{c}: {got[c]} (want {want})"
            for c, want in self.columns.items()
            if c in got and got[c] != want
        ]
        if missing or extra or wrong:
            raise SchemaError(f"{self.name}: missing={missing} extra={extra} wrong_dtype={wrong}")
        return df.select(list(self.columns))

    def read(self, path: Path) -> pl.DataFrame:
        return self.check(pl.read_parquet(path))

    def write(self, path: Path, df: pl.DataFrame) -> None:
        checked = self.check(df)
        tmp = path.with_name(f".{path.name}.tmp")
        checked.write_parquet(tmp)
        tmp.replace(path)
