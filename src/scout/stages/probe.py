"""probe: source video -> video.json.

Reads the container with ffprobe and hashes the file. The frame count and the constant
frame rate check come from the packet timestamps, not the container's `nb_frames`:
broadcast clips often carry an edit list whose leading packets sit before t=0 and are
never shown (the Qualification_16 clip has 180 of them).
"""

from __future__ import annotations

import itertools
import subprocess
from fractions import Fraction

from pydantic import BaseModel

from scout.assets import sha256_file
from scout.run import RunDir, Stage
from scout.schemas.video import VideoInfo

FFPROBE = "ffprobe"


class _Stream(BaseModel):
    codec_name: str
    pix_fmt: str
    width: int
    height: int
    r_frame_rate: str
    avg_frame_rate: str
    time_base: str
    start_time: str = "0"
    duration: str
    field_order: str = "unknown"


class _Probe(BaseModel):
    streams: list[_Stream]


def _ffprobe(*args: str) -> str:
    return subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def probe(run: RunDir) -> None:
    video = run.video.resolve()
    entries = "stream=" + ",".join(_Stream.model_fields)
    s = _Probe.model_validate_json(
        _ffprobe("-show_entries", entries, "-of", "json", str(video))
    ).streams[0]
    fps = Fraction(s.r_frame_rate)
    time_base = Fraction(s.time_base)
    start = Fraction(s.start_time)

    pts = sorted(
        int(line)
        for line in _ffprobe("-show_entries", "packet=pts", "-of", "csv=p=0", str(video)).split()
        if line != "N/A"
    )
    shown = [p for p in pts if p * time_base >= start]
    steps = {b - a for a, b in itertools.pairwise(shown)}
    cfr = (
        Fraction(s.avg_frame_rate) == fps
        and len(steps) == 1
        and next(iter(steps)) * time_base == 1 / fps
    )
    interlaced = s.field_order not in ("progressive", "unknown")

    VideoInfo(
        source_path=str(video),
        sha256=sha256_file(video),
        size_bytes=video.stat().st_size,
        codec=s.codec_name,
        pix_fmt=s.pix_fmt,
        width=s.width,
        height=s.height,
        fps_num=fps.numerator,
        fps_den=fps.denominator,
        fps=float(fps),
        frame_count=len(shown),
        duration_s=float(s.duration),
        start_time_s=float(start),
        field_order=s.field_order,
        interlaced=interlaced,
        cfr=cfr,
    ).save(run.video_json)
    print(f"probe: {len(shown)} frames at {float(fps):g} fps, cfr={cfr}, interlaced={interlaced}")


STAGE = Stage(
    "probe",
    inputs=lambda r: (r.video,),
    outputs=lambda r: (r.video_json,),
    fn=probe,
)
