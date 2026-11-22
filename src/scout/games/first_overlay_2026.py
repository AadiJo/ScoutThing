"""Where things sit in the two known FIRST 2026 broadcast overlays.

Boxes are in a 1920x1080 frame. The match stage picks the layout whose clock box shows a
readable, ticking clock, then scales its boxes to the video with `scaled`:

    rect = scaled(layout.clock, info.width, info.height)  # -> PixelRect in video pixels

Each text box holds exactly one line, so it is read with recognition-only OCR.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from scout.schemas.common import Alliance, PixelRect, ViewName

REF_W, REF_H = 1920, 1080

LayoutName = Literal["capcity_2026", "official_2026"]


@dataclass(frozen=True)
class Box:
    """Corner box `[x0, x1) x [y0, y1)` in 1920x1080 pixels."""

    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True)
class Layout:
    """One overlay layout. Every layout's clock shows "m:ss": 0:20 before auto, 0:19..0:00
    in auto, 2:20 in the pause, then 2:19..0:00 in teleop."""

    name: LayoutName
    scoreboard: Box
    """Everything the match stage reads lies inside it."""
    clock: Box
    fuel: dict[Alliance, Box]
    """Fuel counters, "scored / goal"."""
    active: dict[Alliance, Box]
    """Arrow boxes that are yellow while that alliance's hub is active, dark otherwise."""
    teams: dict[Alliance, tuple[Box, Box, Box]]
    """Team number of each slot, driver stations 1, 2, 3 left to right."""
    views: dict[ViewName, Box]
    """Camera panels. A panel may lie partly under the scoreboard."""


CAPCITY = Layout(
    # From the frc-film-review skill's `layout-capcity-2026.json`, checked on CapCity
    # Showdown 2026 qualifications and playoffs. Red on the left; a 256 px title and score
    # bar sits above the main camera panel.
    name="capcity_2026",
    scoreboard=Box(0, 0, 1920, 256),
    clock=Box(850, 150, 1072, 250),
    fuel={"red": Box(135, 160, 400, 240), "blue": Box(1520, 160, 1785, 240)},
    active={"red": Box(470, 162, 545, 237), "blue": Box(1375, 162, 1448, 237)},
    teams={
        "red": (Box(80, 98, 202, 142), Box(282, 98, 405, 142), Box(485, 98, 607, 142)),
        "blue": (Box(1390, 98, 1514, 142), Box(1595, 98, 1716, 142), Box(1798, 98, 1920, 142)),
    },
    views={
        "main": Box(0, 256, 1920, 736),
        "corner_left": Box(0, 736, 652, 1080),
        "corner_right": Box(1268, 736, 1920, 1080),
    },
)

OFFICIAL = Layout(
    # The 2026 official broadcast (Championship divisions, district events), checked on
    # 2026cur_qm8, 2026hop_qm45 and 2026casnd_qm14. Blue on the left; one 160 px score bar
    # is drawn over the top of the main camera picture, which starts at y = 0.
    name="official_2026",
    scoreboard=Box(0, 0, 1920, 160),
    clock=Box(902, 64, 1018, 130),
    fuel={"red": Box(1740, 70, 1885, 112), "blue": Box(85, 70, 228, 112)},
    active={"red": Box(1578, 72, 1614, 108), "blue": Box(308, 72, 343, 108)},
    teams={
        "red": (Box(1180, 78, 1258, 120), Box(1288, 78, 1360, 120), Box(1398, 78, 1468, 120)),
        "blue": (Box(490, 78, 562, 120), Box(600, 78, 667, 120), Box(708, 78, 778, 120)),
    },
    views={
        "main": Box(0, 0, 1920, 736),
        "corner_left": Box(0, 736, 652, 1080),
        "corner_right": Box(1268, 736, 1920, 1080),
    },
)

LAYOUTS: tuple[Layout, ...] = (CAPCITY, OFFICIAL)


def scaled(box: Box, width: int, height: int) -> PixelRect:
    """`box` in the pixels of a `width` x `height` video."""
    sx, sy = width / REF_W, height / REF_H
    x0, y0 = round(box.x0 * sx), round(box.y0 * sy)
    return PixelRect(x=x0, y=y0, w=round(box.x1 * sx) - x0, h=round(box.y1 * sy) - y0)


def is_lit(bgr_mean: tuple[float, float, float]) -> bool:
    """True when an `active` box's mean colour is the arrow's yellow (BGR about 8, 218, 230;
    unlit red is about 16, 12, 129 and unlit blue 113, 63, 0)."""
    b, g, r = bgr_mean
    return r > 180 and g > 170 and b < 120
