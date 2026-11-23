"""The stage list, in run order. Adding a stage means writing its module and adding it here."""

from __future__ import annotations

from scout.run import Stage
from scout.stages import (
    calibrate,
    corners,
    detect,
    facts,
    grade,
    identify,
    match,
    probe,
    render,
    track,
)

STAGES: tuple[Stage, ...] = (
    probe.STAGE,
    match.STAGE,
    calibrate.STAGE,
    detect.STAGE,
    track.STAGE,
    identify.STAGE,
    corners.STAGE,
    facts.STAGE,
    render.STAGE,
    grade.STAGE,
)

BY_NAME: dict[str, Stage] = {s.name: s for s in STAGES}
