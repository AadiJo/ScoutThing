# What you can and cannot know

The facts were computed by plain code from robot tracks. Read these limits before judging.

## The target robot
- Its position is known, to about 0.2 m, for only part of the match: between 44% and 97% on the
  matches seen so far, and this match's share is in "What was seen" and in the phase facts.
  It drops out most behind the far side of its own hub, where its bumper is small or hidden,
  and whole phases can be blind. Every hole of 1 s or more is a `gap` fact. A gap means "not
  tracked", never "idle", "stuck" or "home". A gap's `inference` (`stayed_in_place`,
  `in_own_zone_beyond_barrier`, `entered_own_zone`, `own_zone_both_sides`) is a probable
  reading of its two sides, not a measurement; the readiness defaults use it. `others_near`
  lists robots tracked near where the target was last or next seen: a pointer to a frame worth
  opening, not a contact.
- Nothing the tool did not see is known: a trip, stall or crossing inside a gap leaves no
  fact. Counts of trips or crossings are counts of what was tracked.
- A `stop` with `unseen_s` above 0 is partly INFERRED: the robot was at nearly the same place
  before and after a gap. Never describe that part as seen. A tile can confirm it or show it
  wrong. `start` / `end` say whether the stop's edges were seen. Say where a stop or contact
  was from its `where` distances (`own_hub_m`, `own_wall_m`, `guardrail_m`), not from its zone.
- A `crossing_turned_back` is one attempt: onto (or against) a bump or trench and back out the
  same side. `cost_s` runs from entering the barrier until `cost_end` (it reached the far side,
  tried again, its hub went inactive, tracking lost it); `own_active_s` is the part in its own
  active time. `arrived` says how its arrival is known: `seen`; `crossing` or `inference` (a
  crossing fact or a gap's inference shows it going there just before a gap, and the time adds
  the rest of the way at the speed it was going); `after_gap` (first seen there after a gap, so
  the cost is an upper bound). `slowest_mps` (its lowest speed on the band) and
  `furthest_from_barrier_m` (how far back it went before getting home) set the line's default:
  a stall, a reversal and a slow turn-away all read slow, a back-off goes far, a drive past does
  neither. Only `stopped_s` says it stood still. Being pinned and a stall make the same shape; a
  tile showing an opponent on it tells them apart.
- A stall whose centre sits just inside its own zone at the end of its own ramp is recorded by
  the turned-back fact, not by a stop.
- A `left_own_zone_while_active` trip starts where it was seen leaving, or, when it left inside
  a gap (`left_unseen_s` above 0), at the first position seen outside its zone: when it really
  left is not known. It ends at `home` (`arrived`-style evidence). A trip is a SHAPE. Positions cannot see fuel: a trip out with an
  empty hopper (to get more) and a trip with a heaped hopper (that should have gone home) look
  the same. The hopper tiles decide it. Never call the trip a reload, a feed or a waste; say
  what the tiles show.
- `zone_visit` with zone `on_barrier`: part of the robot was over a bump or trench for at
  least 2.5 s; `excess_s` is the time beyond a normal 1.2 s crossing.
- All speeds are real (broadcast clock rate already corrected).

## The other robots
- Tracking of the other robots is badly incomplete. Detection finds about two thirds of the
  robots in view, and dark-bumpered robots are missed in most frames. `team_tracked_share`
  says how much of the match each team was tracked. The absence of a `contact` fact means
  nothing.
- A labelled robot's `relation` comes from the roster and is right. `bumper_seen` is the
  detector's guess and can be wrong. An unlabelled robot may be a partner or an opponent.
- A `contact` says who was moving toward whom (`target_closing_mps`, `other_closing_mps`),
  never who was defending: a robot driving its own route into a waiting robot also closes. The
  target may be the one blocking. `still_s` is how long the target was nearly still in it
  (`still_unseen_s` of that inferred across gaps): held, or holding itself, positions cannot
  say which.
- A contact `end` of `unknown_other_track_lost` or `unknown_target_track_lost` means tracking
  stopped. It does NOT mean anyone left or got free.
- On a tile, a robot with no box was not tracked: no fact knows about it. Black or very dark
  bumpers are not a colour reading; do not call such a robot a partner or an opponent from its
  colour.

## What tiles can and cannot show
- At zoom 2 a heaped hopper (yellow balls piled above the rim) reads apart from an empty or part
  full one on about four tiles in five; behind the hub, behind a referee or at the far side it
  often cannot be read. Say `cannot_see` then, do not guess.
- Fuel, shots and points never come from positions. A tile can show a heaped or empty hopper or
  balls leaving; say so and cite it.
- The corner camera panels are not calibrated: nothing is ringed or boxed there. A robot there
  is the target only where you read its bumper number.
- Purpose. "Reloading", "collecting", "feeding", "waiting", "trying to" are guesses about why;
  the checker rejects them. Say where it went and what a tile shows.
- Auto is the robot's code, not the driver: report it, do not score it.
