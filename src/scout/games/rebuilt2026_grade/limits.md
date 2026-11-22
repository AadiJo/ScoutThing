# What you can and cannot know

The facts were computed by plain code from robot tracks. Read these limits before judging.

## The target robot
- Its position is known for about 80% of the match, to about 0.2 m. Every hole of 1 s or
  more is a `gap` fact. A gap means "not tracked", never "idle" or "stuck". Nothing that
  happened inside a gap may be charged to the driver unless a frame you opened shows it.
- Speed along the main camera's line of sight is less reliable on the far side of the field.

## The other robots
- Tracking of the other robots is badly incomplete. Detection finds about two thirds of the
  robots in view, and robots with dark bumpers are missed in most frames. `team_tracked_share`
  says how much of the match each team was tracked; a low share means that robot is mostly
  invisible to the facts.
- So the absence of a `contact` fact means nothing. When these facts were checked against a
  human's review of a match, they caught only a small fraction of the real defense episodes.
- The best indirect signs of defense are `stop` facts, slow or `returned` crossings, and gaps.
  Each needs a frame to confirm what was there.
- A `contact` with an unlabelled robot may be an alliance partner (for example a partner
  sitting beside the target while both shoot). Check the bumper colour in a frame.

## What positions cannot show
- Whether the robot was carrying fuel, whether a shot went in, or how many points anyone
  scored. Do not grade shooting, and do not make claims about fuel or scoring.
- Stops inside the robot's own zone while its hub is active are usually shooting, which is
  good play, not a stall. Never treat standing still at home during an active shift as wasted
  time.
- Auto is the robot's code, not the driver: report it, do not score it.
