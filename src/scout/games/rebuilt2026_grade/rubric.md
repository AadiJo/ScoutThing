Rubric version: draft-2

# The three skills and the scale

Scores are absolute, 1 to 10, so a 7 means the same in any match. Grade from this match only.
Code turns your calls into a band of two scores for each skill; pick the score inside it (the
higher one when nothing outside the ledger counts against the robot, the lower one when
something does, such as an uncertain line you lean toward charging). An answer outside the band
is rejected.

## driving
Getting over bumps and through trenches without stalling or backing off. Judged only from the
driving ledger. For each line decide:
- `charged`: the robot's own driving cost the time (stalled, backed off, beached).
- `opponent`: an opponent held it there (you also list it as an `opponent_on_target` episode).
- `hub_inactive`: its hub was off, so standing on or turning back from a barrier was staging,
  not lost time.
- `partner`: its own partner was in the way (a tile shows it).
- `frame_contradicts`: a tile shows the facts' shape is wrong (for example an inferred stop
  where the robot was elsewhere).
- `uncertain`: no tile shows the robot in the line's span, so you cannot tell; it counts half.

A bump or trench is not a place to stand: time stopped or turned back on one while its hub is
active is charged unless an opponent or partner held it there, even if it might have been
shooting.

Charged seconds T = the sum of `lost_s` (or a smaller `charged_s` a tile justifies) over charged
lines, plus half of `lost_s` over uncertain lines.
- 9-10: T = 0.
- 7-8: T under 3 s.
- 5-6: T 3 to 6 s.
- 3-4: T 6 to 10 s.
- 1-2: T over 10 s.

## shift_readiness
The position part only: in its own zone when its hub turns on. Judged only at the readiness
ledger's lines, one per `hub_on` fact; a phase where its hub was already on has no line and is
never judged here. Lateness L is the line's `late_s` when charged, half of it when uncertain, 0
when an opponent held it (`opponent`, with an episode) or a tile contradicts the facts.
- 9-10: every L at most 1 s.
- 7-8: worst L 1 to 3 s.
- 5-6: worst L 3 to 8 s, or two or more activations with L over 1 s.
- 3-4: worst L over 8 s.
- 1-2: a charged line where it never got home in that phase.
- Null when the ledger has no line.

## handling_defense
How much opponents cost it, from the episodes marked `opponent_on_target`. Each gets `cost_s`:
the seconds the opponent cost it in getting home or onto its route. Being pressed while home
during its own active time is playing through: `cost_s` 0. Time it spent defending an opponent
(`target_on_opponent`) is not part of this score.
- 9-10: every cost at most 2 s.
- 7-8: one episode cost 2 to 5 s.
- 5-6: one cost 5 to 8 s, or two or more over 2 s.
- 3-4: one cost over 8 s, or most episodes cost over 3 s.
- 1-2: over 30 s lost to opponents in all.
- Null when no episode is `opponent_on_target`: with no confirmed opponent on it, there is
  nothing to judge.
