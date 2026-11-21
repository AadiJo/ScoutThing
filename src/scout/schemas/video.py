"""`video.json`, written by the probe stage: what the source video is."""

from __future__ import annotations

from typing import Literal

from scout.schemas.common import StageFile


class VideoInfo(StageFile):
    """Facts about the source video from ffprobe. `fps` is `fps_num / fps_den`.

    `t = frame / fps` for every frame; the frame reader refuses a video with `cfr` false.
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
    """Constant frame rate: r_frame_rate equals avg_frame_rate and frame_count matches."""
