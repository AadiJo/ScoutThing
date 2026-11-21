"""Where things sit in the standard FIRST 2026 broadcast overlay.

Boxes are in a 1920x1080 frame and come from the frc-film-review skill's
`layout-capcity-2026.json` (checked on CapCity Showdown 2026 qualifications and playoffs).
The match stage scales them to the video with `scaled`:

    rect = scaled(CLOCK, info.width, info.height)  # -> PixelRect in video pixels

Each text box holds exactly one line, so it is read with recognition-only OCR.
"""

from __future__ import annotations

from dataclasses import dataclass

from scout.schemas.common import Alliance, PixelRect, ViewName

REF_W, REF_H = 1920, 1080


@dataclass(frozen=True)
class Box:
    """Corner box `[x0, x1) x [y0, y1)` in 1920x1080 pixels."""

    x0: int
    y0: int
    x1: int
    y1: int


SCOREBOARD = Box(0, 0, 1920, 256)
"""Title bar plus score bar: everything the match stage reads lies inside it."""

CLOCK = Box(850, 150, 1072, 250)
"""Game clock, "m:ss". Shows 0:20 before auto, 0:19..0:00 in auto, 2:20 in the pause,
then 2:19..0:00 in teleop."""

FUEL: dict[Alliance, Box] = {
    "red": Box(135, 160, 400, 240),
    "blue": Box(1520, 160, 1785, 240),
}
"""Fuel counters, "scored / goal"."""

ACTIVE: dict[Alliance, Box] = {
    "red": Box(470, 162, 545, 237),
    "blue": Box(1375, 162, 1448, 237),
}
"""Arrow boxes that are yellow while that alliance's hub is active, dark otherwise."""

TEAMS: dict[Alliance, tuple[Box, Box, Box]] = {
    "red": (Box(80, 98, 202, 142), Box(282, 98, 405, 142), Box(485, 98, 607, 142)),
    "blue": (Box(1390, 98, 1514, 142), Box(1595, 98, 1716, 142), Box(1798, 98, 1920, 142)),
}
"""Team number of each slot, left to right (driver stations 1, 2, 3), right of the logo."""

VIEWS: dict[ViewName, Box] = {
    "main": Box(0, 256, 1920, 736),
    "corner_left": Box(0, 736, 652, 1080),
    "corner_right": Box(1268, 736, 1920, 1080),
}
"""Camera panels: the wide main shot and the two corner cameras below it."""


def scaled(box: Box, width: int, height: int) -> PixelRect:
    """`box` in the pixels of a `width` x `height` video."""
    sx, sy = width / REF_W, height / REF_H
    x0, y0 = round(box.x0 * sx), round(box.y0 * sy)
    return PixelRect(x=x0, y=y0, w=round(box.x1 * sx) - x0, h=round(box.y1 * sy) - y0)


def is_lit(bgr_mean: tuple[float, float, float]) -> bool:
    """True when an `ACTIVE` box's mean colour is the arrow's yellow (BGR about 8, 218, 230;
    unlit red is about 16, 12, 129 and unlit blue 113, 63, 0)."""
    b, g, r = bgr_mean
    return r > 180 and g > 170 and b < 120
