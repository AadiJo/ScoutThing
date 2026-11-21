"""The one frame reader every stage uses.

Frames come from a single sequential `ffmpeg` decode piped as raw BGR, so memory holds
one frame at a time and nothing seeks frame by frame:

    info = VideoInfo.load(run.video_json)
    for f in frames(run.video, info, sample_hz=10, crop=panel, start=20.0, end=40.0):
        f.frame, f.t, f.image  # source frame index, video seconds, HxWx3 uint8 BGR

Frame `i` has `t = i / fps` exactly (video seconds, frame 0 at t=0). ffmpeg picks frames
by that index (`select` on the frame timestamp), so indices are exact on any constant
frame rate video; `frames` refuses a video whose probe found `cfr` false.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from scout.schemas.common import PixelRect
from scout.schemas.video import VideoInfo

FFMPEG = "ffmpeg"
THREADS = 8
_EPS = 1e-6


class Frame(NamedTuple):
    frame: int
    """Source frame index."""
    t: float
    """Video seconds, `frame / fps`."""
    image: NDArray[np.uint8]
    """HxWx3 BGR, full frame or the requested crop."""


def frame_time(info: VideoInfo, index: int) -> float:
    """Video seconds of frame `index`."""
    return index * info.fps_den / info.fps_num


def frames(
    video: Path,
    info: VideoInfo,
    *,
    sample_hz: float | None = None,
    crop: PixelRect | None = None,
    start: float = 0.0,
    end: float | None = None,
) -> Iterator[Frame]:
    """Yield frames with `start <= t < end` in order.

    `sample_hz` keeps every `round(fps / sample_hz)`-th frame counted from the first
    frame in range (10 Hz on a 60 fps video keeps every 6th); None keeps every frame.
    `crop` is a rectangle in full-frame pixels.
    """
    first = max(0, math.ceil(start * info.fps - _EPS))
    last = info.frame_count - 1
    if end is not None:
        last = min(last, math.ceil(end * info.fps - _EPS) - 1)
    step = 1 if sample_hz is None else max(1, round(info.fps / sample_hz))
    return _decode(video, info, first, last, step, crop)


def frame_at(video: Path, info: VideoInfo, t: float, crop: PixelRect | None = None) -> Frame:
    """The frame on screen at video time `t` (the last frame with `frame.t <= t`)."""
    index = min(max(0, math.floor(t * info.fps + _EPS)), info.frame_count - 1)
    return next(_decode(video, info, index, index, 1, crop))


def _decode(
    video: Path, info: VideoInfo, first: int, last: int, step: int, crop: PixelRect | None
) -> Iterator[Frame]:
    """Decode frames `first, first+step, ... <= last` with one ffmpeg process."""
    if not info.cfr:
        raise ValueError(f"{video}: variable frame rate; frame indices would not be exact")
    if last < first:
        return
    wanted = range(first, last + 1, step)
    # Frame index from the original timestamp (-copyts keeps it through the input seek).
    index = f"round((t-{info.start_time_s})*{info.fps_num}/{info.fps_den})"
    select = (
        f"select='st(0,{index})*0+between(ld(0),{first},{last})*not(mod(ld(0)-{first},{step}))'"
    )
    filters = [select, "format=bgr24"]
    width, height = info.width, info.height
    if crop is not None:
        filters.append(f"crop={crop.w}:{crop.h}:{crop.x}:{crop.y}")
        width, height = crop.w, crop.h
    seek = max(0.0, frame_time(info, first) - 1.0)
    cmd = [
        FFMPEG, "-nostdin", "-loglevel", "error", "-threads", str(THREADS),
        "-ss", f"{seek:.6f}", "-copyts", "-i", str(video),
        "-vf", ",".join(filters), "-fps_mode", "passthrough",
        "-frames:v", str(len(wanted)), "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
    ]  # fmt: skip
    size = width * height * 3
    with tempfile.TemporaryFile() as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log)
        assert proc.stdout is not None
        try:
            for i in wanted:
                buf = proc.stdout.read(size)
                if len(buf) < size:
                    proc.wait()
                    log.seek(0)
                    err = log.read().decode(errors="replace").strip()
                    raise RuntimeError(f"ffmpeg ended before frame {i} of {video}: {err}")
                image = np.frombuffer(buf, np.uint8).reshape(height, width, 3)
                yield Frame(i, frame_time(info, i), image)
        finally:
            proc.kill()
            proc.wait()
            proc.stdout.close()
