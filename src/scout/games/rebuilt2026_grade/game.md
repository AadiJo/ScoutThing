# The game: REBUILT (FRC 2026)

Cut down from the scouting guide to what robot positions and video frames can show.

- Match: AUTO 20 s (clock 0:20 to 0:00), a pause of about 3 s (clock holds 2:20), then
  TELEOP 2:20 to 0:00. Teleop phases by clock: TRANSITION 2:20-2:10, SHIFT 1 2:10-1:45,
  SHIFT 2 1:45-1:20, SHIFT 3 1:20-0:55, SHIFT 4 0:55-0:30, END GAME 0:30-0:00.
- Each alliance has a HUB (a tall box with a funnel near the middle of the field). Robots
  collect FUEL (yellow foam balls) and shoot it into their own hub. Fuel only counts while
  that alliance's hub is ACTIVE.
- Both hubs are active in AUTO, TRANSITION and END GAME. In SHIFTS 1-4 exactly one hub is
  active and it alternates. The alliance that scored more fuel in AUTO is inactive in
  SHIFTS 1 and 3 and active in SHIFTS 2 and 4. The match data below says which shifts are
  the target's, and the `hub_on` facts mark each moment its hub turns on.
- A robot may shoot into its hub only while its bumpers are at least partly inside its own
  ALLIANCE ZONE, which runs from its alliance wall to the front edge of its hub and bumps.
  So a loaded robot in the NEUTRAL ZONE must first get home, over a BUMP (the sloped ramps
  either side of each hub) or under a TRENCH (the low arm between the outer bump and the
  guardrail). Each alliance's line of bump, hub, bump and trenches is a "barrier" in the
  facts.
- Good play: shoot hard while your hub is active; while it is inactive, collect fuel in the
  neutral zone, stockpile, feed fuel toward your side, or play defense, so you are home and
  loaded when your hub turns on. Being away from home when your hub is active is a mistake
  only if the robot could have got home and chose not to. If it was blocked, that is the
  defender's doing.
- Leaving home during an active shift to reload is good play, not a mistake.
- AUTO is the robot's code: the drive team does not control it. A robot may not completely
  cross the centre line in auto.
- Defense is legal. PINNING an opponent for more than 3 s is a foul. Contact is common.
- Fuel sources: the centre pile in the neutral zone, floor depots near the walls, and chutes
  fed by human players at the alliance walls. Fuel that goes through a hub drops out into the
  neutral zone.
