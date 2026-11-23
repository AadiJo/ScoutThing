"""The field's AprilTags: where they are, and finding them in median images of a camera panel.

Shared by the calibrate stage (every view, from tags alone) and the corners stage (the corner
views, from tags and robots). A panel is read from `window_medians`: the median of each
quarter of the match, in which robots and game pieces have moved away and the bare field
shows. A tag counts only if it is found in at least `MIN_WINDOWS` of those quarters at
points that agree (`merge`).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray
from pupil_apriltags import Detection, Detector  # pyright: ignore[reportMissingTypeStubs]

from scout.assets import asset
from scout.camera import Camera
from scout.schemas.common import PixelRect
from scout.schemas.video import VideoInfo
from scout.video import THREADS, frames

F64 = NDArray[np.float64]
U8 = NDArray[np.uint8]

SAMPLE_HZ = 0.5
WINDOWS = 4
MIN_WINDOWS = 2
SCALES = (2, 3, 4)
MIN_MARGIN = 15.0
AGREE_PX = 2.0
TAG_SIZE_M = 0.1651  # black square side (6.5 in); fits rule out the 0.2064 m outer size


@dataclass(frozen=True)
class Tag:
    """One tag seen in a view: image points (N, 2) in full-frame pixels and the field
    points (N, 3) they show. Decoded tags give 4 corners, located tags their centre."""

    id: int
    img: F64
    obj: F64
    located: bool


Hits = dict[int, list[tuple[int, F64]]]
"""Per tag id, (window index, image points) of every sighting."""


# --- field geometry ------------------------------------------------------------


def tag_corners() -> dict[int, F64]:
    """Every field tag's black-square corners (4, 3) in field metres. Order: bottom-left,
    bottom-right, top-right, top-left seen from the front, matching pupil_apriltags."""
    data = json.loads(asset("field_tags").read_text())
    s = TAG_SIZE_M / 2
    local = np.array([[0.0, -s, -s], [0.0, s, -s], [0.0, s, s], [0.0, -s, s]])
    out: dict[int, F64] = {}
    for t in data["tags"]:
        p = t["pose"]["translation"]
        q = t["pose"]["rotation"]["quaternion"]
        r = _quat_matrix(float(q["W"]), float(q["X"]), float(q["Y"]), float(q["Z"]))
        out[int(t["ID"])] = local @ r.T + np.array([p["x"], p["y"], p["z"]], np.float64)
    return out


def _quat_matrix(w: float, x: float, y: float, z: float) -> F64:
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def normal(c: F64) -> F64:
    """Outward normal of a tag from its corners (front face)."""
    n = np.cross(c[1] - c[0], c[3] - c[0])
    return n / np.linalg.norm(n)


def faces(c: F64, cam: Camera) -> bool:
    """The tag with corners `c` shows its front to `cam`."""
    return bool(normal(c) @ (cam.center - c.mean(0)) > 0)


def clusters(
    ids: list[int], corners: dict[int, F64], dist_m: float, same_facing: bool
) -> list[list[int]]:
    """Tags whose centres lie within `dist_m` of each other, transitively; with
    `same_facing`, only tags facing the same way join."""
    groups: list[list[int]] = []
    for t in sorted(ids):
        c, n = corners[t].mean(0), normal(corners[t])
        near = [
            g
            for g in groups
            if any(
                np.linalg.norm(corners[u].mean(0) - c) < dist_m
                and (not same_facing or normal(corners[u]) @ n > 0.95)
                for u in g
            )
        ]
        groups = [g for g in groups if g not in near] + [sorted([t, *(u for g in near for u in g)])]
    return sorted(groups)


# --- finding tags --------------------------------------------------------------


def detector() -> Detector:
    return Detector(
        families="tag36h11",
        nthreads=THREADS,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
    )


def decode_view(det: Detector, medians: list[U8], rect: PixelRect) -> Hits:
    """Every tag decoded in each window median of the panel at each of SCALES."""
    hits: Hits = {}
    off = np.array([rect.x, rect.y], np.float64)
    for w, med in enumerate(medians):
        crop = med[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w]
        for s in SCALES:
            up = np.asarray(
                cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC), np.uint8
            )
            # The bindings annotate detect() as one Detection and decision_margin as None.
            for d in cast(list[Detection], det.detect(up)):
                if d.hamming <= 1 and cast(float, d.decision_margin) > MIN_MARGIN:
                    corners = np.asarray(d.corners, np.float64) / s + off
                    hits.setdefault(int(d.tag_id), []).append((w, corners))
    return hits


def merge(hits: Hits, obj: dict[int, F64], located: bool) -> tuple[dict[int, Tag], list[int]]:
    """Trusted tags (seen in MIN_WINDOWS windows with agreeing points) and suspect ids."""
    tags: dict[int, Tag] = {}
    suspect: list[int] = []
    for tid, hs in hits.items():
        stack = np.stack([p for _, p in hs])
        med = np.median(stack, 0)
        good = np.max(np.linalg.norm(stack - med, axis=2), axis=1) <= AGREE_PX
        windows = len({w for (w, _), g in zip(hs, good, strict=True) if g})
        if windows >= MIN_WINDOWS and tid in obj:
            tags[tid] = Tag(tid, np.median(stack[good], 0), obj[tid], located)
        else:
            suspect.append(tid)
    return tags, sorted(suspect)


# --- median images ---------------------------------------------------------------


def window_medians(
    video: Path, info: VideoInfo, rects: Iterable[PixelRect], start: float, end: float
) -> tuple[list[U8], U8]:
    """Grey full-frame medians of each of WINDOWS quarters of `start`..`end` (video
    seconds, sampled at SAMPLE_HZ) and of the whole span. Outside the match an official
    broadcast shows intro cards, results and other cameras, so a median over the whole file
    is not the field. Only the rows holding `rects` are decoded; the rest stay black."""
    rows = list(rects)
    y0 = min(r.y for r in rows)
    y1 = max(r.y + r.h for r in rows)
    band = PixelRect(x=0, y=y0, w=info.width, h=y1 - y0)
    grey = [
        cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY)
        for f in frames(
            video,
            info,
            sample_hz=SAMPLE_HZ,
            crop=band,
            start=max(0.0, start),
            end=min(info.duration_s, end),
        )
    ]
    stack = np.stack(grey)
    n = len(stack)

    def full(img: NDArray[np.floating]) -> U8:
        out = np.zeros((info.height, info.width), np.uint8)
        out[y0:y1] = img.astype(np.uint8)
        return out

    quarters = [
        full(np.median(stack[i * n // WINDOWS : (i + 1) * n // WINDOWS], 0)) for i in range(WINDOWS)
    ]
    return quarters, full(np.median(stack, 0))
