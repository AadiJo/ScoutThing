"""Import frc-vision-scorer hand-tracked truth into one `truth/<match>.json`.

Their truth is one file per robot, `data/eval/ground_truth/<match_key>/<team>.json`, indexed
by frame in their copy of the match video. This converter keeps every `visible` and
`estimated` keyframe (pixel, view, state), drops `unknown` ones, and maps each frame to the
video described by a scout `video.json` as `our_frame = their_frame + frame_offset`:

    uv run --no-sync python -m scout.importers.scorer_truth \\
        <their_match_dir> runs/<run>/video.json truth/<match>.json \\
        [--frame-offset N] [--mapping-note "how the offset was established"]

It refuses a video whose picture size or frame rate differs from theirs, since the pixels and
frame indices would then not carry over.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from scout.schemas.common import ViewName
from scout.schemas.truth import Truth, TruthRobot, TruthSample
from scout.schemas.video import VideoInfo
from scout.video import frame_time


class _Keyframe(BaseModel):
    model_config = ConfigDict(extra="ignore")

    frame: int
    state: Literal["visible", "estimated", "unknown"]
    uv: tuple[float, float] | None = None
    view: ViewName | None = None


class _Video(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str
    fingerprint: str
    fps: float
    width: int
    height: int
    frame_count: int


class _Doc(BaseModel):
    """The fields of one of their per-robot files that the import uses."""

    model_config = ConfigDict(extra="ignore")

    match_key: str
    team: int
    video: _Video
    match_t0_frame: int | None
    click_height_m: float
    keyframes: list[_Keyframe]


def _robot(doc: _Doc, info: VideoInfo, offset: int) -> TruthRobot:
    samples: list[TruthSample] = []
    for k in doc.keyframes:
        if k.state == "unknown":
            continue
        if k.uv is None or k.view is None:
            raise ValueError(f"team {doc.team}: {k.state} keyframe {k.frame} has no uv or view")
        frame = k.frame + offset
        if not 0 <= frame < info.frame_count:
            raise ValueError(f"team {doc.team}: frame {k.frame} maps outside the video")
        samples.append(
            TruthSample(
                frame=frame,
                t=frame_time(info, frame),
                view=k.view,
                u=k.uv[0],
                v=k.uv[1],
                state=k.state,
            )
        )
    return TruthRobot(team=doc.team, samples=samples)


def convert(
    match_dir: Path, info: VideoInfo, frame_offset: int = 0, mapping_note: str = ""
) -> Truth:
    """One Truth from every `<team>.json` in `match_dir`, frames mapped onto `info`'s video."""
    docs = [_Doc.model_validate_json(p.read_bytes()) for p in sorted(match_dir.glob("[0-9]*.json"))]
    if not docs:
        raise ValueError(f"no <team>.json files in {match_dir}")
    for field in ("match_key", "video", "click_height_m", "match_t0_frame"):
        if len({repr(getattr(d, field)) for d in docs}) != 1:
            raise ValueError(f"{match_dir}: files disagree on {field}")
    doc = docs[0]
    theirs = doc.video
    if (theirs.width, theirs.height) != (info.width, info.height):
        raise ValueError(f"picture {theirs.width}x{theirs.height} != {info.width}x{info.height}")
    if Fraction(theirs.fps).limit_denominator(1001) != Fraction(info.fps_num, info.fps_den):
        raise ValueError(f"frame rate {theirs.fps} != {info.fps_num}/{info.fps_den}")
    t0 = "none" if doc.match_t0_frame is None else str(doc.match_t0_frame + frame_offset)
    notes = (
        f"Imported from frc-vision-scorer data/eval/ground_truth/{doc.match_key}/ "
        f"({len(docs)} robots, hand-followed). Their video {theirs.name} "
        f"(fingerprint {theirs.fingerprint}, {theirs.frame_count} frames at {theirs.fps:g} fps). "
        f"Frames mapped as ours = theirs + {frame_offset}; their auto start frame maps to {t0}. "
        f"Unknown keyframes dropped; uv, view and state kept as recorded. {mapping_note}"
    ).strip()
    return Truth(
        video_sha256=info.sha256,
        video_name=Path(info.source_path).name,
        source="imported",
        click_height_m=doc.click_height_m,
        notes=notes,
        robots=[_robot(d, info, frame_offset) for d in sorted(docs, key=lambda d: d.team)],
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("match_dir", type=Path)
    p.add_argument("video_json", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--frame-offset", type=int, default=0)
    p.add_argument("--mapping-note", default="")
    a = p.parse_args()
    match_dir: Path = a.match_dir
    video_json: Path = a.video_json
    out: Path = a.out
    offset: int = a.frame_offset
    note: str = a.mapping_note
    truth = convert(match_dir, VideoInfo.load(video_json), offset, note)
    out.parent.mkdir(parents=True, exist_ok=True)
    truth.save(out)
    for r in truth.robots:
        print(f"{out.name}: team {r.team}: {len(r.samples)} samples")


if __name__ == "__main__":
    main()
