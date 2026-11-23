# What you can and cannot know

The facts were computed by plain code from robot tracks. Read these limits before judging.

## The target robot
- Its position is known for about 80% of the match, to about 0.2 m. Every hole of 1 s or more
  is a `gap` fact. A gap means "not tracked", never "idle" or "stuck". A gap's `inference`
  (`stayed_in_place`, `in_own_zone_beyond_barrier`) is a probable reading of its two sides,
  not a measurement. `others_near` lists robots tracked near where the target was last or next
  seen: a pointer to a frame worth opening, not a contact.
- A `stop` with `unseen_s` above 0 is partly INFERRED: the robot was at nearly the same place
  before and after a gap. Never describe that part as seen. A tile can confirm it or show it
  wrong. `start` / `end` say whether the stop's edges were seen.
- A `crossing_turned_back` is a SHAPE: onto a bump or trench and back out the same side, then
  `seconds_to_across` until it got across some other way (null: not within 8 s). Staging while
  its hub is off and being pinned make the same shape as a stall. Several tries can be folded
  into one fact, which overstates `seconds_to_across`; a tile can show that.
- A `left_own_zone_while_active` trip is a SHAPE too. A trip to get more fuel looks exactly
  like a wasted trip; the facts cannot tell them apart. Describe where it went, how long it
  took and, if a tile shows the hopper, what the tile shows. Never call the trip itself good,
  wasted, a reload or a feed.
- `zone_visit` with zone `on_barrier`: part of the robot was over a bump or trench for at
  least 2.5 s.
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
  target may be the one blocking.
- A contact `end` of `unknown_other_track_lost` or `unknown_target_track_lost` means tracking
  stopped. It does NOT mean anyone left or got free.
- On a tile, a robot with no box was not tracked: no fact knows about it. Black or very dark
  bumpers are not a colour reading; do not call such a robot a partner or an opponent from its
  colour.

## What positions cannot show
- Fuel, shots, whether a shot went in, or points. A tile can show a heaped or empty hopper or
  balls leaving; say so and cite it. Never claim fuel or shots without a tile.
- Purpose. "Reloading", "collecting", "feeding", "waiting", "trying to" are guesses about why;
  the checker rejects them. Say where it went and what a tile shows.
- Defense on a moving target inside a gap, or a robot that only pokes its front onto a ramp.
- Auto is the robot's code, not the driver: report it, do not score it.
