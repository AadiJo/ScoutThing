Rubric version: draft-3.1

# The three skills and the scale

Scores are absolute, 1 to 10, so a 7 means the same in any match. Grade from this match only.
Code turns your calls into charged seconds and a band of two scores for each skill; pick the
score inside it (the higher one when nothing outside the ledger counts against the robot, the
lower one when something does, such as an uncertain line you lean toward charging). An answer
outside the band is rejected. Every charge is time in its own active time: time while its hub
is off costs nothing here.

## driving
Getting home over bumps and through trenches without stalling or backing off, and going home
when loaded instead of moving around out of its zone. Judged only from the driving ledger.

Barrier lines (`turned_back`, `barrier_stop`, `barrier_stay`, `slow_crossing`):
- `charged`: its own driving cost the time. At a barrier on its way home it stalled, beached,
  backed off, or turned away (back toward the centre, or along the barrier to another lane) and
  got home later: a turn-away is charged whether or not it ever stopped, because the line's
  cost is the time until it got home. Standing on a bump or trench while its hub is active is
  charged even if it might have been shooting, unless a tile shows balls leaving it from inside
  its own zone.
- `opponent`: an opponent held it there (you also list it as an `opponent_on_target` episode).
- `partner`: its own partner was in the way (a tile shows it).
- `frame_contradicts`: a tile shows the facts' shape is wrong: it never turned away (it kept
  driving toward home and crossed, for example over the ramp corner in one curving pass), it
  was not where the facts put it, or it arrived earlier than the bound.
- `uncertain`: no tile shows the robot in the line's span; it counts half.

Loaded trips (`loaded_trip`, a trip out of its own zone in its own active time): read the
hopper on the HOPPER TILES. A robot out of its zone with a heaped hopper should be going home.
- `heaped`: `heaped_at` is the first hopper tile from which every later tile shows a heaped
  hopper. Code charges the trip from then to its end, less any time a barrier line or an
  `opponent_on_target` episode already covers, less the time to get home from where it was
  (1.2 s plus half a second per metre outside its zone, at least 3 s).
- `not_heaped`: empty or part full up to its late tiles (a trip out to get more fuel). Not
  charged.
- `cannot_see`: the hopper cannot be read on any tile (behind the hub, a referee in front, too
  far). It adds no seconds but holds the driving score to the lower score of its band.

Charged seconds T = charged barrier lines (`lost_s` less time an opponent episode cost inside
them; on a line whose cost is an upper bound, `charged_s` may instead give the seconds from
its start to a cited tile that shows it home) + half of uncertain ones + heaped
trips as above.
- 9-10: T at most 0.5 s.
- 7-8: T up to 3 s (one short hesitation).
- 5-6: T up to 6 s (one real stall or a loaded detour).
- 3-4: T up to 12 s (a stall and a reroute, or several stalls).
- 1-2: T over 12 s.

## shift_readiness
The position part only: in its own zone when its hub turns on. Judged only at the readiness
ledger's lines, one per `hub_on` fact; a phase where its hub was already on has no line and is
never judged here. Lateness L is the line's `late_s` when charged, half of it when uncertain, 0
when an opponent held it (`opponent`, with an episode) or a tile contradicts the facts (a
corner tile where you read its bumper in its zone counts).
- 9-10: every L at most 1 s.
- 7-8: worst L 1 to 3 s.
- 5-6: worst L 3 to 8 s, or two or more activations with L over 1 s.
- 3-4: worst L over 8 s.
- 1-2: a charged line where it never got home in that phase.
- Null when the ledger has no line.

## handling_defense
How much opponents cost it, from the episodes marked `opponent_on_target`: every episode where
an established opponent touched, pressed, pinned or blocked the target, or stood between it and
where it was driving, even when that cost nothing. Each episode's cost is the larger of:
- `cost_s`, your seconds for what the opponent cost it in getting home or onto its route (0
  when home in its own active time: there it was already where it scores), and
- its pin cost, from code: the items' `pin_s` (seconds nearly still, inferred time counted
  half) beyond 3 s (episodes within 2 s of each other share one 3 s), in full when `while_held` is `not_shooting`, half when `cannot_see`, 0 when
  `shooting` (balls leaving it on most tiles across the still stretch).
A press while it keeps shooting or keeps driving its route costs nothing. Time it spent
defending an opponent (`target_on_opponent`) is not part of this score.

D = the summed cost of its `opponent_on_target` episodes.
- 9-10: D at most 1 s.
- 7-8: D up to 4 s.
- 5-6: D up to 8 s.
- 3-4: D up to 15 s.
- 1-2: D over 15 s.
- Null only when no episode is `opponent_on_target`: no opponent was ever on it.
