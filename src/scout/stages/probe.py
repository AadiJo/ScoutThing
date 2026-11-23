"""probe: source video -> video.json.

Reads the container with ffprobe and hashes the file. The frame count and the frame
grid come from the packet timestamps, not the container's `nb_frames`: broadcast clips
often carry an edit list whose leading packets sit before t=0 and are never shown (the
Qualification_16 clip has 180 of them).

Every shown picture must sit on the `1 / r_frame_rate` grid (within one time-base tick).
A grid slot with no picture is a held frame: the reader repeats the previous picture
there, as a player does (Qualification_2 and 13 skip one slot just before their last
picture). A picture off the grid, two pictures in one slot, or no picture at t=0 cannot
be read exactly, and the stage stops with NeedsInput.
"""

from __future__ import annotations

import shlex
import subprocess
from fractions import Fraction

from pydantic import BaseModel

from scout.assets import sha256_file
from scout.run import NeedsInput, RunDir, Stage
from scout.schemas.video import VideoInfo

FFPROBE = "ffprobe"


class _Stream(BaseModel):
    codec_name: str
    pix_fmt: str
    width: int
    height: int
    r_frame_rate: str
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


def _grid(shown: list[int], time_base: Fraction, start: Fraction, fps: Fraction) -> list[int]:
    """Grid slot of each sorted shown pts. Raises ValueError when a pts is more than one
    tick off the grid, two share a slot, or slot 0 is empty."""
    slots: list[int] = []
    for p in shown:
        exact = (p * time_base - start) * fps
        slot = round(exact)
        if abs(exact - slot) / fps >= time_base:
            raise ValueError(
                f"the picture at {float(p * time_base):.6f} s is "
                f"{float(abs(exact - slot)):.2f} frames off the {float(fps):g} fps grid "
                "(a truly variable frame rate)"
            )
        if slots and slot == slots[-1]:
            raise ValueError(f"two pictures share frame slot {slot}")
        slots.append(slot)
    if not slots or slots[0] != 0:
        raise ValueError("no picture at the start of the video (t=0)")
    return slots


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
    try:
        slots = _grid(shown, time_base, start, fps)
    except ValueError as e:
        fixed = video.with_name(f"{video.stem}_cfr.mp4")
        cmd = shlex.join(
            ["ffmpeg", "-i", str(video), "-fps_mode", "cfr", "-r", s.r_frame_rate, "-c:a", "copy",
             str(fixed)]
        )  # fmt: skip
        raise NeedsInput(
            f"{video.name} cannot be read frame-exactly: {e}",
            None,
            f"no answer can fix this; re-encode to a constant frame rate with `{cmd}` "
            f"and run scout on {fixed.name}",
        ) from e
    taken = set(slots)
    held = tuple(i for i in range(slots[-1] + 1) if i not in taken)
    cfr = not held
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
        frame_count=slots[-1] + 1,
        duration_s=float(s.duration),
        start_time_s=float(start),
        field_order=s.field_order,
        interlaced=interlaced,
        cfr=cfr,
        held_frames=held,
    ).save(run.video_json)
    note = f", {len(held)} held frames {held[:5]} repeat the previous picture" if held else ""
    print(
        f"probe: {slots[-1] + 1} frames at {float(fps):g} fps, cfr={cfr}, "
        f"interlaced={interlaced}{note}"
    )


STAGE = Stage(
    "probe",
    inputs=lambda r: (r.video,),
    outputs=lambda r: (r.video_json,),
    fn=probe,
)
