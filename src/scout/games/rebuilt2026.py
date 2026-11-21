"""Everything specific to the 2026 game REBUILT: field geometry, zones and match timing.

Coordinates are WPILib field metres: blue alliance wall at x=0, red wall at
x=FIELD_LENGTH_M, y across the field. The field is point-symmetric about its centre.
Sources are files in Paradox2102/frc-vision-scorer at commit dfe74cae; the geometry
numbers are what that code computes from the welded tag layout, rounded to 0.1 mm.

Game time (`game_t`) is game-clock seconds since auto start with the auto-to-teleop
pause removed, as in the reference `web/scouting/clock.py`: auto 0..20, teleop 20..160.
The match stage maps it to video seconds using the measured auto and teleop starts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from scout.schemas.common import Alliance

# --- field ------------------------------------------------------------------

FIELD_LENGTH_M = 16.541  # data/field/2026-rebuilt-welded.json, "field.length"
FIELD_WIDTH_M = 8.069  # data/field/2026-rebuilt-welded.json, "field.width"


@dataclass(frozen=True)
class Rect:
    """Axis-aligned ground rectangle in field metres."""

    x0: float
    y0: float
    x1: float
    y1: float

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


HUB_BASE_WIDTH_M = 47.0 * 0.0254  # web/field/overlay_geometry.py, HUB_BASE_WIDTH_M
HUB_HEIGHT_M = 120.36 * 0.0254  # web/field/overlay_geometry.py, HUB_HEIGHT_M

# Hub cabinet centres: bounding-box centre of each hub's tag ring.
# web/field/overlay_geometry.py hub_center() over web/field/occlusion.py hub_prisms().
HUB_CENTERS: dict[Alliance, tuple[float, float]] = {
    "blue": (4.6255, 4.0346),
    "red": (11.9155, 4.0346),
}

# Hub cabinet bases, HUB_BASE_WIDTH_M square about each centre.
# web/field/overlay_geometry.py structure_footprints()["hubs"].
HUBS: dict[Alliance, Rect] = {
    "blue": Rect(4.0286, 3.4377, 5.2224, 4.6315),
    "red": Rect(11.3186, 3.4377, 12.5124, 4.6315),
}

# The four drivable bumps, one either side of each hub (1.128 m deep, 1.854 m wide).
# web/field/overlay_geometry.py structure_footprints()["bumps"], BUMP_* constants.
BUMPS: tuple[Rect, ...] = (
    Rect(4.0615, 1.5772, 5.1895, 3.4312),
    Rect(4.0615, 4.6380, 5.1895, 6.4920),
    Rect(11.3515, 1.5772, 12.4795, 3.4312),
    Rect(11.3515, 4.6380, 12.4795, 6.4920),
)
BUMP_HEIGHT_M = 0.1654  # web/field/overlay_geometry.py, BUMP_HEIGHT_M

# Alliance zone depth: own wall to the near face of the own hub cabinet. The neutral
# zone is everything between the two alliance zones (hubs, bumps and trenches included).
# web/scouting/field_rules.py AutoRules.alliance_zone_x.
ALLIANCE_ZONE_DEPTH_M = 4.0286

# --- robots -----------------------------------------------------------------

# Square robot footprint including bumpers. web/field/bumper_colour.py HALF_FOOTPRINT_M x 2.
ROBOT_FOOTPRINT_M = 0.95
# Bumper band heights above the carpet. web/field/bumper_colour.py BAND_Z0_M, BAND_Z1_M.
BUMPER_Z0_M = 0.03
BUMPER_Z1_M = 0.16
BUMPER_CENTER_HEIGHT_M = (BUMPER_Z0_M + BUMPER_Z1_M) / 2

# --- zones ------------------------------------------------------------------

Zone = Literal["own", "neutral", "opponent"]


def other(alliance: Alliance) -> Alliance:
    """The opposing alliance."""
    return "blue" if alliance == "red" else "red"


def zone_of(x: float, alliance: Alliance) -> Zone:
    """Zone containing field x, seen from `alliance`: its own alliance zone, the neutral
    zone, or the opponent's alliance zone."""
    from_own_wall = x if alliance == "blue" else FIELD_LENGTH_M - x
    if from_own_wall < ALLIANCE_ZONE_DEPTH_M:
        return "own"
    if from_own_wall > FIELD_LENGTH_M - ALLIANCE_ZONE_DEPTH_M:
        return "opponent"
    return "neutral"


# --- match timing -------------------------------------------------------------

AUTO_S = 20.0
AUTO_TELEOP_PAUSE_S = 3.0  # nominal; not part of game time, measured per match in video
TELEOP_S = 140.0
TRANSITION_S = 10.0
SHIFT_S = 25.0
ENDGAME_S = 30.0
GAME_S = AUTO_S + TELEOP_S

PhaseName = Literal["auto", "transition", "shift1", "shift2", "shift3", "shift4", "endgame"]


@dataclass(frozen=True)
class Phase:
    """One phase in game time, `start <= game_t < end` (the last phase includes `end`)."""

    name: PhaseName
    start: float
    end: float


# Teleop countdown 140..130 transition, then four 25 s shifts, then a 30 s endgame.
# web/scouting/games/rebuilt2026.py SHIFTS.
PHASES: tuple[Phase, ...] = (
    Phase("auto", 0.0, AUTO_S),
    Phase("transition", AUTO_S, AUTO_S + TRANSITION_S),
    Phase("shift1", AUTO_S + TRANSITION_S, AUTO_S + TRANSITION_S + SHIFT_S),
    Phase("shift2", AUTO_S + TRANSITION_S + SHIFT_S, AUTO_S + TRANSITION_S + 2 * SHIFT_S),
    Phase("shift3", AUTO_S + TRANSITION_S + 2 * SHIFT_S, AUTO_S + TRANSITION_S + 3 * SHIFT_S),
    Phase("shift4", AUTO_S + TRANSITION_S + 3 * SHIFT_S, GAME_S - ENDGAME_S),
    Phase("endgame", GAME_S - ENDGAME_S, GAME_S),
)


def phase_at(game_t: float) -> Phase | None:
    """The phase at game time `game_t`, or None before auto or after the match."""
    for phase in PHASES:
        if phase.start <= game_t < phase.end:
            return phase
    if game_t == GAME_S:
        return PHASES[-1]
    return None


def active_hubs(phase: PhaseName, auto_winner: Alliance | None) -> frozenset[Alliance] | None:
    """Alliances whose hub scores in `phase`. Both are active in auto, transition and
    endgame. In shifts 1 and 3 the auto winner (more fuel in auto) is inactive, in shifts
    2 and 4 the other alliance is. Returns None in a shift when the winner is unknown
    (a tie is broken by the FMS and is not visible). web/scouting/games/rebuilt2026.py.
    """
    if phase in ("auto", "transition", "endgame"):
        return frozenset(("red", "blue"))
    if auto_winner is None:
        return None
    winner_inactive = phase in ("shift1", "shift3")
    return frozenset((other(auto_winner),) if winner_inactive else (auto_winner,))
