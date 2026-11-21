"""identify: reads video, video.json, detections.parquet, match.json (and, for the labelling
half, tracks.parquet).

Writes reads.parquet: the bumper number reader's verdict on every detector box in every
view, scored against the six roster teams. Also writes `reads_check.jpg`, a grid of crops
exactly as the reader sees them, sampled across views and confidence, for a human to read
against the captions.

One ffmpeg decode pass at the detector's SAMPLE_HZ. Each box is cut and resized on the
CPU (a few hundred pixels each); full batches of crops go to the GPU in half precision.
Only one decoded frame is held at a time.

The labelling half (reads joined to tracks into identity.json) is not built yet; the stage
raises NotImplementedError after writing reads.parquet, so the pipeline stops there.
"""

from __future__ import annotations

import random
import resource
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import torch
from numpy.typing import NDArray
from scipy.spatial import KDTree  # pyright: ignore[reportMissingTypeStubs]

from scout.assets import asset
from scout.readers.digits import Crop, DigitReader
from scout.run import RunDir, Stage
from scout.schemas.common import ViewName
from scout.schemas.detections import SAMPLE_HZ, read_detections
from scout.schemas.identity import READER_NAME, write_reads
from scout.schemas.match import MatchInfo
from scout.schemas.video import VideoInfo
from scout.video import frames

BATCH = 512
CPU_THREADS = 8
CROP = Crop(pad=0.0, trim=0.0)
"""The reader reads the detector box itself. It was trained on bumper boxes padded 25% then
trimmed 10% (`Crop(0.25, 0.1)`), but this detector boxes the whole robot in the main view, so
padding mostly adds robot body; on the hand-marked boxes of Qualification 16 and Playoff 12
the bare box doubled the share of free reads that spell the true team."""

ACCEPT_P = 0.99
"""Abstain unless the best roster team's probability (softmax over the six roster teams'
log-likelihoods, temperature 1) is at least this (the checkpoint's `plain` rule, whose
0.98 top-two margin this implies)..."""
ACCEPT_FREE_READ = True
"""...and the unconstrained free read spells that same team number. This stops a number
outside the roster (a crate, a banner) or a half-hidden one from being forced onto the
nearest roster team. On the hand-marked boxes of Qualification 16 and Playoff 12 the rule
named 43 of 189 boxes with no wrong team; p >= 0.99 alone named 110, one of them wrong (a
2468 half hidden by a sign, read as 2687 at p = 0.994)."""
STATIC_RADIUS_PX = 6.0
STATIC_SIZE_TOL = 0.2
STATIC_WINDOW_S = 10.0
STATIC_SHARE = 0.5
"""A box is static when boxes of about its size (width and height within STATIC_SIZE_TOL)
centred within STATIC_RADIUS_PX of it, in the same view, appear in at least STATIC_SHARE of
the video's STATIC_WINDOW_S windows. Static boxes always abstain: the driver station team
number displays read as real roster teams at p = 1.0. On Qualification 16 and Playoff 12
this flagged no hand-marked robot box, but missed 23 of 720 confident display reads (Q16),
where the display's box jitters."""

CHECK_BINS = (0.0, 0.5, 0.9, ACCEPT_P, 1.01)
CHECK_PER_CELL = 5
"""reads_check.jpg samples this many crops per (view, confidence bin) cell."""
CHECK_SCALE = 3
CHECK_POOL = 400
"""Crops per view kept (uniformly at random) as candidates for reads_check.jpg."""


@dataclass
class _Timer:
    """Wall seconds per step, printed at the end of the stage."""

    s: dict[str, float] = field(default_factory=dict[str, float])

    def add(self, key: str, since: float) -> float:
        now = time.perf_counter()
        self.s[key] = self.s.get(key, 0.0) + now - since
        return now


@dataclass
class _Out:
    """Per-box results in detection-row order, filled one batch at a time."""

    rows: list[int] = field(default_factory=list[int])
    lp: list[NDArray[np.float64]] = field(default_factory=list[NDArray[np.float64]])
    text: list[str] = field(default_factory=list[str])
    text_lp: list[NDArray[np.float64]] = field(default_factory=list[NDArray[np.float64]])


def _static(dets: pl.DataFrame) -> NDArray[np.bool_]:
    """Per detection row: is this a box at a fixed spot (see STATIC_SHARE)?"""
    box = dets.select("x1", "y1", "x2", "y2").to_numpy().astype(np.float64)
    centre = (box[:, :2] + box[:, 2:]) / 2
    size = box[:, 2:] - box[:, :2]
    window = (dets["t"].to_numpy() // STATIC_WINDOW_S).astype(np.int64)
    view = dets["view"].to_physical().to_numpy()
    n_windows = int(window.max()) + 1 if len(window) else 1
    pairs = [np.c_[np.arange(len(dets)), np.arange(len(dets))]]
    for v in np.unique(view):
        idx = np.flatnonzero(view == v)
        found: NDArray[np.int64] = KDTree(centre[idx]).query_pairs(
            STATIC_RADIUS_PX, output_type="ndarray"
        )
        ij = idx[found]
        pairs += [ij, ij[:, ::-1]]
    ij = np.concatenate(pairs)
    i, j = ij[:, 0], ij[:, 1]
    similar = np.all(np.abs(size[j] - size[i]) <= STATIC_SIZE_TOL * size[i], axis=1)
    windows = (
        pl.DataFrame({"i": i[similar], "w": window[j[similar]]})
        .group_by("i")
        .agg(n=pl.col("w").n_unique())
    )
    share = np.zeros(len(dets))
    share[windows["i"].to_numpy()] = windows["n"].to_numpy() / n_windows
    return share >= STATIC_SHARE


def _decide(
    lp: NDArray[np.float64], text: list[str], static: NDArray[np.bool_], teams: list[str]
) -> pl.DataFrame:
    """Probabilities, roster mass, best team and abstain flag for `[N, K]` log-likelihoods."""
    z = lp - lp.max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    best = p.argmax(1)
    top = p.max(1)
    ok = (top >= ACCEPT_P) & ~static
    if ACCEPT_FREE_READ:
        ok &= np.array([t == teams[b] for t, b in zip(text, best, strict=True)], dtype=bool)
    roster = np.array([int(t) for t in teams], np.int32)
    return pl.DataFrame(
        {
            "probs": pl.Series(p.astype(np.float32).tolist(), dtype=pl.List(pl.Float32)),
            "roster_mass": np.exp(lp).sum(1).astype(np.float32),
            "conf": top.astype(np.float32),
            "best": roster[best],
            "static": static,
            "abstain": ~ok,
        }
    )


def _check_cell(view: ViewName, conf: float) -> tuple[ViewName, int]:
    return view, int(np.searchsorted(CHECK_BINS, conf, side="right")) - 1


def _draw_check(path: Path, crops: dict[int, NDArray[np.uint8]], reads: pl.DataFrame) -> None:
    """Write reads_check.jpg: one row per (view, confidence bin) cell, each crop at
    CHECK_SCALE times model size with its free read, best team, probability and verdict."""
    rng = random.Random(0)
    cells: dict[tuple[ViewName, int], list[int]] = {}
    sub = reads.with_row_index("i").filter(pl.col("i").is_in(list(crops)))
    for r in sub.iter_rows(named=True):
        cells.setdefault(_check_cell(r["view"], r["conf"]), []).append(r["i"])
    by_i = {r["i"]: r for r in sub.iter_rows(named=True)}
    tw, th, cap = 128 * CHECK_SCALE, 32 * CHECK_SCALE, 44
    tiles_rows: list[NDArray[np.uint8]] = []
    for key in sorted(cells):
        picks = rng.sample(cells[key], min(CHECK_PER_CELL, len(cells[key])))
        row: list[NDArray[np.uint8]] = []
        for i in picks:
            r = by_i[i]
            tile = np.zeros((th + cap, tw, 3), np.uint8)
            rgb = cv2.resize(crops[i], (tw, th), interpolation=cv2.INTER_NEAREST)
            tile[:th] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            verdict = "static" if r["static"] else "abstain" if r["abstain"] else f"= {r['team']}"
            colour = (160, 160, 160) if r["abstain"] else (0, 255, 0)
            line1 = f"{r['view'][:8]} t={r['t']:.1f} read '{r['text']}'"
            line2 = f"best {r['best']} p={r['conf']:.3f} {verdict}"
            cv2.putText(tile, line1, (4, th + 18), 0, 0.5, (255, 255, 255), 1)
            cv2.putText(tile, line2, (4, th + 38), 0, 0.5, colour, 1)
            row.append(tile)
        while len(row) < CHECK_PER_CELL:
            row.append(np.zeros((th + cap, tw, 3), np.uint8))
        tiles_rows.append(np.hstack([np.pad(t, ((2, 2), (2, 2), (0, 0))) for t in row]))
    if tiles_rows:
        cv2.imwrite(str(path), np.vstack(tiles_rows), [cv2.IMWRITE_JPEG_QUALITY, 90])


def identify(run: RunDir) -> None:
    torch.set_num_threads(CPU_THREADS)
    cv2.setNumThreads(CPU_THREADS)
    info = VideoInfo.load(run.video_json)
    if info.interlaced:
        raise ValueError("interlaced video: the reader needs de-interlaced crops (not ported)")
    match = MatchInfo.load(run.match_json)
    teams = [str(s.team) for s in match.roster]
    dets = read_detections(run.detections).sort(
        "frame", "view", "conf", descending=[False, False, True]
    )
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    timer = _Timer()
    start = time.perf_counter()
    reader = DigitReader.load(asset("digits"), device)
    mark = timer.add("load model", start)

    frame_col = dets["frame"].to_numpy()
    uniq, starts = np.unique(frame_col, return_index=True)
    stops = [*starts[1:].tolist(), len(dets)]
    spans = {int(f): (int(a), b) for f, a, b in zip(uniq, starts, stops, strict=True)}
    views = dets["view"].to_list()
    boxes = dets.select("x1", "y1", "x2", "y2").to_numpy().astype(np.float64)

    rng = random.Random(0)
    seen_per_view: dict[str, int] = {}
    pool: dict[str, list[int]] = {}
    check_rows: dict[int, NDArray[np.uint8]] = {}
    out = _Out()
    pending_rows: list[int] = []
    pending: list[NDArray[np.uint8]] = []

    def flush() -> float:
        nonlocal pending_rows, pending
        t0 = time.perf_counter()
        x = reader.batch(pending)
        s = reader.score(x, teams)
        out.rows += pending_rows
        out.lp.append(s.lp)
        out.text += s.text
        out.text_lp.append(s.text_lp)
        pending_rows, pending = [], []
        return timer.add("gpu read", t0)

    last = int(frame_col.max()) if len(frame_col) else -1
    end_t = (last + 1) / info.fps
    n_frames = 0
    for f in frames(run.video, info, sample_hz=SAMPLE_HZ, end=end_t):
        mark = timer.add("decode", mark)
        span = spans.get(f.frame)
        if span is None:
            continue
        n_frames += 1
        for i in range(*span):
            view = views[i]
            img = reader.model_input(f.image, boxes[i].tolist(), match.views[view], CROP)
            pending_rows.append(i)
            pending.append(img)
            # uniform reservoir of CHECK_POOL candidate check crops per view
            k = seen_per_view.get(view, 0)
            seen_per_view[view] = k + 1
            slots = pool.setdefault(view, [])
            if k < CHECK_POOL:
                slots.append(i)
                check_rows[i] = img
            elif (j := rng.randrange(k + 1)) < CHECK_POOL:
                del check_rows[slots[j]]
                slots[j] = i
                check_rows[i] = img
        mark = timer.add("crop", mark)
        if len(pending) >= BATCH:
            mark = flush()
    if pending:
        mark = flush()
    if n_frames != len(spans):
        raise RuntimeError(f"decoded {n_frames} of {len(spans)} frames with boxes")

    order = np.argsort(np.asarray(out.rows))
    lp = np.concatenate(out.lp)[order]
    text = [out.text[i] for i in order]
    text_lp = np.concatenate(out.text_lp)[order]
    static = _static(dets)
    mark = timer.add("static boxes", mark)
    decided = _decide(lp, text, static, teams)
    reads = pl.concat(
        [
            dets.select(
                "frame",
                "t",
                "view",
                pl.lit(None, pl.Int32).alias("track_id"),
                "x1",
                "y1",
                "x2",
                "y2",
            ),
            pl.DataFrame(
                {
                    "text": pl.Series(text, dtype=pl.String),
                    "text_p": np.exp(text_lp).astype(np.float32),
                    "roster": pl.Series(
                        [[int(t) for t in teams]] * len(text), dtype=pl.List(pl.Int32)
                    ),
                }
            ),
            decided,
        ],
        how="horizontal",
    ).with_columns(
        team=pl.when(pl.col("abstain")).then(None).otherwise(pl.col("best")),
        reader=pl.lit(READER_NAME),
    )
    write_reads(run.reads, reads)
    mark = timer.add("write parquet", mark)
    _draw_check(run.path / "reads_check.jpg", check_rows, reads)
    timer.add("check sheet", mark)

    total = time.perf_counter() - start
    named = reads.filter(~pl.col("abstain"))
    print(f"identify: {len(reads)} boxes read over {n_frames} frames, {len(named)} named")
    print("  " + "  ".join(f"{k} {v:.1f}s" for k, v in timer.s.items()) + f"  total {total:.1f}s")
    gpu = torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"  peak RSS {rss:.2f} GiB, peak torch GPU {gpu:.2f} GiB")
    raise NotImplementedError("labelling reads onto tracks waits for the track stage")


STAGE = Stage(
    "identify",
    inputs=lambda r: (r.video, r.video_json, r.detections, r.match_json),
    outputs=lambda r: (r.identity_json, r.reads),
    fn=identify,
)
