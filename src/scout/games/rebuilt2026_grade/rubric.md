Rubric version: draft-1

# The three skills and the scale

Scores are absolute, 1 to 10, so a 7 means the same in any match. Grade from this match only.
A score may be null for any skill when the evidence cannot support one; then `null_reason` is
required and says why.

## driving
Bump and trench crossings, getting stuck, wasted movement.
- 9-10: every crossing clean and at speed, no self-inflicted stops on a barrier, no aborted
  crossings, direct paths.
- 7-8: one hesitation or slow crossing costing under about 2 s, otherwise direct.
- 5-6: one self-inflicted stall or aborted crossing costing 3 to 5 s, or a couple of wasted
  trips.
- 3-4: two or three stalls or aborts, around 10 s lost to its own driving.
- 1-2: stuck or beached for a large part of the match by its own driving.
- Not charged: time lost while an opponent is on it, and anything inside a tracking gap.

## shift_readiness
The position part only: being home when its hub turns on. Judge each moment its own hub turns
on (the `hub_on` facts): was it inside its own alliance zone, and if not, how long until it
was. Weight the two shift activations more than END GAME.
- 9-10: in its zone at or within about 1 s of every activation.
- 7-8: one activation late by up to 3 s.
- 5-6: one late by 3 to 8 s, or all slightly late.
- 3-4: one late by more than 8 s, or late at every one.
- 1-2: away from its zone for most of its active time by choice.
- Lateness caused by a defender holding it is listed, not charged. Leaving its zone during an
  active shift is not charged by itself, since reload trips are good play.

## handling_defense
What it did while an opponent was on it. Judge only episodes you have confirmed by opening a
frame; the grade must cite those frames.
- 9-10: each time, it got free within about 2 s or kept its route and reached its zone.
- 7-8: mostly quick escapes, one episode costing 3 to 5 s.
- 5-6: mixed, with some 5 to 8 s losses.
- 3-4: usually held up, or drives into the defender repeatedly.
- 1-2: neutralised for most of the match while it had options.
- When you found no defense episode you could confirm with a frame, the score is null with the
  reason, never a high score.
