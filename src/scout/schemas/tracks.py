"""`tracks.parquet`, written by the track stage: robot tracks in field metres, one row per
track per sampled frame where the track exists. `corner_tracks.parquet`, written by the follow
stage, has the same columns: robots placed through a corner camera in frames where the main
view had no sample of them (`view` is the corner view, `track_id` at least
`CORNER_TRACK_ID_BASE`). Read both through `scout.positions.load_positions`.

A track is one robot for as long as the track stage is sure it is the same robot. Tracks
end wherever that is in doubt (crossings, occlusion, long gaps), so one robot usually has
several tracks over a match and `track_id` is not limited to six values; the identify
stage labels each piece with a team. At most six tracks should be live at once, but that
is a consequence of the tracking, not a guarantee.

- `color` is the track's bumper colour (confidence-weighted majority of its boxes), not its
  alliance: at offseason events a robot can play for one alliance in the other's bumpers.
- `x`, `y` are the smoothed robot centre on the carpet. When a whole-robot box was paired
  with the source bumper box, the measured position mixes both boxes' placements, so it need
  not sit at the source box's centre.
- `state` is `measured` when a box backs the sample, `interpolated` inside a short gap.
- `view`, `x1`..`y2` (full-frame pixels) and `conf` are the measured sample's source box,
  for cropping and weighting; null on interpolated rows. When duplicate boxes on one robot
  were merged, the box is their union and `conf` their maximum.
- `kind` is what that source box frames: `bumper`, or `robot` when only a whole-robot box saw
  the robot in that frame (its colour does not count toward the track's `color`; a track
  seen only that way has colour `unknown`); null on interpolated rows.
- `partial` is true when the source box framed only part of the robot (a bumper box much
  narrower than the whole-robot box on the same robot, a box much narrower than a robot at
  that spot, or a box cut by the edge of the picture); null on interpolated rows. When no
  second box backed a partial sample, its position is mostly carried by the track's other
  samples (it is weighted down in smoothing); it is still `measured`, but less sure.
"""

from __future__ import annotations

import polars as pl

from scout.schemas.common import ALLIANCES, VIEW_NAMES, TableSpec
from scout.schemas.detections import KIND

COLOR = pl.Enum([*ALLIANCES, "unknown"])
STATE = pl.Enum(["measured", "interpolated"])
VIEW = pl.Enum(VIEW_NAMES)

TRACKS = TableSpec(
    "tracks",
    {
        "track_id": pl.Int32(),
        "color": COLOR,
        "frame": pl.Int64(),
        "t": pl.Float64(),
        "x": pl.Float64(),
        "y": pl.Float64(),
        "state": STATE,
        "view": VIEW,
        "x1": pl.Float32(),
        "y1": pl.Float32(),
        "x2": pl.Float32(),
        "y2": pl.Float32(),
        "conf": pl.Float32(),
        "kind": KIND,
        "partial": pl.Boolean(),
    },
)

CORNER_TRACKS_FILE = "corner_tracks.parquet"
CORNER_TRACK_ID_BASE = 1_000_000
"""Corner track ids start here, clear of the track stage's."""

read_tracks = TRACKS.read
write_tracks = TRACKS.write
