Rubric version: draft-6

# The three skills and the scale

Scores are absolute, 1 to 10, so a 7 means the same in any match. Grade from this match only.
Code computes every score from charged seconds: each band of two scores covers a span of
seconds, and the higher score goes to the lower half of the span. Every charge is time in its
own active time: time while its hub is off costs nothing here.

## driving
Getting home over bumps and through trenches without stalling or backing off, and going home
when loaded instead of moving around out of its zone. Judged only from the driving ledger.

Barrier lines (`turned_back`, `barrier_stop`, `barrier_stay`, `slow_crossing`) carry a default:
- A `turned_back` attempt is `charged` when it slowed to 0.8 m/s or less on the band (a stall, a
  reversal or a slow turn-away), or went 3 m or more back from the barrier before getting home
  (a back-off or a turn-away); its cost is the time until it got home, at the earliest evidence
  of that (a crossing or a gap inference, else the first tracked point there). Otherwise it is a
  drive past, `cleared`. Its line's `why` names which measure fired; it is a stall only when the
  robot was measured stopped on the band for a second or more.
  Time after its hub turned on that a readiness line already charges as lateness is not charged
  here again.
- Stops, stays and slow crossings on a barrier in its own active time are `charged`.
Opponent split. When a robot in a contact inside a charged barrier line came half a metre or
more nearer the target's own wall than the target (it stood on its way home), part of the line
counts against handling_defense instead: when it came close before the target left the
barrier, all of the line from then; after, the line's time from then beyond what getting home
from there takes (1.2 s plus half a second per metre outside its zone, at least 3 s), at most
the contact's length. A robot that met it as it reached home cost nothing, and a stall and the
back-off before the contact stay here. A labelled opponent's split applies by default; an
unlabelled robot's only when a tile identifies it as an opponent (`opponent`), and a labelled
one's is dropped when a bumper read shows it is not an opponent (`no_opponent`).
A default changes only on a tile that contradicts it: those two, its partner was in the way
(`partner`), the facts had the target somewhere else (`misplaced`), or it was home before an
upper-bound time (`home_earlier`).

Loaded trips (`loaded_trip`, a trip out of its own zone in its own active time): read the
hopper on the HOPPER TILES. A robot out of its zone with a heaped hopper should be going home.
- `heaped`: `heaped_at` is the first hopper tile from which every later tile shows a heaped
  hopper. Code charges the trip from then to its end, less any time a barrier line or an
  `opponent_on_target` episode already covers, less the time to get home from where it was
  (1.2 s plus half a second per metre outside its zone, at least 3 s), less 3 s more for the
  doubt in one hopper reading.
- `not_heaped` (empty or part full up to its late tiles) and `cannot_see`: not charged.

Charged seconds T = charged barrier lines + heaped trips as above.
- 9-10: T at most 0.5 s.
- 7-8: T up to 3 s (one short hesitation).
- 5-6: T up to 6 s (one real stall or a loaded detour).
- 3-4: T up to 12 s (a stall and a reroute, or several stalls).
- 1-2: T over 12 s.

Coverage. T counts only what was seen, and unseen time can only add to it. When more than a
quarter of its own active time was untracked with nothing putting it in its own zone, driving is
null, unless what was seen already charges more than 3 s (then the score stands, flagged as
low coverage: the true score may be lower). A phase of its own active time more than half
unseen flags the score the same way.

## shift_readiness
The position part only: in its own zone when its hub turns on. Judged only at the readiness
ledger's lines, one per `hub_on` fact; a phase where its hub was already on has no line and is
never judged here. Each line's default: `cleared` when tracked in its own zone, or untracked
inside a gap whose `inference` puts it there (`in_own_zone_beyond_barrier`,
`entered_own_zone`, `own_zone_both_sides`, or `stayed_in_place` last seen in its zone), or
shown home by then by a crossing or inference; `not_judged` when untracked with its last
sighting in its own zone and nothing seen since (it may have stayed or left: not charged, not
scored, unless a tile with its bumper read settles it: `home` or `not_home`); otherwise
`charged` with its `late_s` (at most that when it is a bound). Lateness L is the line's charged seconds: 0 when an
opponent held it (`opponent`), the tile's time when a tile shows it home sooner
(`home_earlier`), `late_s` when a tile shows it out of its zone against a cleared default
(`not_home`).
- 9-10: every L at most 1 s.
- 7-8: worst L 1 to 3 s.
- 5-6: worst L 3 to 8 s, or two or more activations with L over 1 s.
- 3-4: worst L over 8 s.
- 1: a charged line where it never got home in that phase.
- Only judged lines count. Null when the ledger has no line or none was judged.

## handling_defense
How much opponents cost it, from the defense items in episodes marked `opponent_on_target`:
every episode where an established opponent touched, pressed, pinned or blocked the target, or
stood between it and where it was driving, even when that cost nothing. Code computes the cost:
- held time: items within 2 s of each other form one hold; a hold's `pin_s` (seconds nearly
  still, inferred time counted half) beyond 3 s, counted half (whether it kept shooting while
  held is rarely readable);
- plus the opponent split seconds of barrier lines (see driving), counted once: held still time
  inside those seconds is not counted again.
Time it spent defending an opponent (`target_on_opponent`) is not part of this score.

D = the summed cost.
- 9-10: D at most 1 s.
- 7-8: D up to 4 s.
- 5-6: D up to 8 s.
- 3-4: D up to 15 s.
- 1-2: D over 15 s.
- Null only when no episode is `opponent_on_target`: no opponent was ever on it.
