"""`video.json`, written by the probe stage: what the source video is."""

from __future__ import annotations

from typing import Literal

from scout.schemas.common import StageFile


class VideoInfo(StageFile):
    """Facts about the source video from ffprobe. `fps` is `fps_num / fps_den`.

    Frames are slots on an even grid: frame `i` is at `t = i / fps` for every frame. A slot
    with no picture of its own (listed in `held_frames`) shows the previous picture, as a
    player does.
    """

    schema_version: Literal[1] = 1
    source_path: str
    sha256: str
    size_bytes: int
    codec: str
    pix_fmt: str
    width: int
    height: int
    fps_num: int
    fps_den: int
    fps: float
    frame_count: int
    duration_s: float
    start_time_s: float
    """Presentation time of frame 0 in the container. Video seconds are relative to it."""
    field_order: str
    interlaced: bool
    cfr: bool
    """Constant frame rate: every grid slot has its own picture (`held_frames` is empty)."""
    held_frames: tuple[int, ...] = ()
    """Grid slots with no picture of their own (the container skips that timestamp). The
    reader returns the previous frame's picture there."""
