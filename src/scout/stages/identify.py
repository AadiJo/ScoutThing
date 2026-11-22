"""identify: reads video, video.json, detections.parquet, tracks.parquet, match.json.

Writes reads.parquet: the bumper number reader's verdict on every detector box in every
view, scored against the six roster teams, with the track each main-view box joins. Writes
identity.json: which stretch of which track is which team, or null when unsure. Also writes
`reads_check.jpg` (crops exactly as the reader sees them, sampled across views and
confidence) and `identity_check.jpg` (one timeline row per track, coloured by label, with a
tick per named read).

Reading: one ffmpeg decode pass at the detector's SAMPLE_HZ. Each box is cut and resized on
the CPU (a few hundred pixels each); full batches of crops go to the GPU in half precision.
Only one decoded frame is held at a time.

Labelling: named reads are sparse anchors on tracks (tracks are main view only, so corner
reads never label). A few agreeing reads label a stretch of a track, which may carry on
past them while the weaker reads do not contradict; named reads of two teams on one track
mark a change of robot, and the stretch between them stays null. No team is left on two
tracks at once. Unread tracks stay null: no colour, position or elimination guesses.
"""

from __future__ import annotations

import itertools
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
from scout.schemas.identity import (
    READER_NAME,
    ClaimConflict,
    Identity,
    LabelSegment,
    LabelSource,
    write_reads,
)
from scout.schemas.match import MatchInfo
from scout.schemas.tracks import read_tracks
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

JOIN_COVER = 0.8
"""A read joins a track sample when the sample's box covers this share of the read's box
(track boxes can be unions of merged duplicates, so equality is not required)."""
LABEL_MIN_READS = 3
LABEL_MIN_SPAN_S = 0.3
"""A label needs this many agreeing named reads on one track, first to last at least this
far apart, with no named read of another team between them. Never one read."""
EXTEND_MAX_S = 5.0
EXTEND_STOP_P = 0.05
EXTEND_STOP_READS = 3
EXTEND_MIN_READS = 5
"""A label backed by at least EXTEND_MIN_READS named reads extends past its first and last
reads toward the track's ends (never toward a named read of another team) for at most
EXTEND_MAX_S, and stops after EXTEND_STOP_READS reads in a row give the team less than
EXTEND_STOP_P. On stretches pinned by named reads, an unnamed read gives the robot's own team
p >= 0.05 93-95% of the time and another robot's team 9-15% of the time, so three such reads
in a row mark a likely change of robot the tracker missed. On Qualification 16 and Playoff 12
this stop removed one of two same-team clashes; extending 3- and 4-read labels caused the
other clash and the one wrong hand-marked label (a pin where one box covers two robots)."""

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


def _read_boxes(
    run: RunDir, info: VideoInfo, match: MatchInfo, timer: _Timer
) -> tuple[pl.DataFrame, dict[int, NDArray[np.uint8]]]:
    """Read every detector box in one decode pass. Returns reads (track_id still null) in
    detection order and the candidate crops for reads_check.jpg by row index."""
    teams = [str(s.team) for s in match.roster]
    dets = read_detections(run.detections).sort(
        "frame", "view", "conf", descending=[False, False, True]
    )
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    mark = time.perf_counter()
    reader = DigitReader.load(asset("digits"), device)
    mark = timer.add("load model", mark)

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
    timer.add("decide", mark)
    return reads, check_rows


def _join_tracks(reads: pl.DataFrame, tracks: pl.DataFrame) -> pl.Series:
    """track_id per read: the measured track sample in the same frame whose box covers at
    least JOIN_COVER of the (main-view) read box, best cover then best IoU; else null."""
    r = reads.with_row_index("i").filter(pl.col("view") == "main")
    t = tracks.filter(pl.col("state") == "measured").select(
        "frame", "track_id", tx1="x1", ty1="y1", tx2="x2", ty2="y2"
    )
    w = pl.min_horizontal("x2", "tx2") - pl.max_horizontal("x1", "tx1")
    h = pl.min_horizontal("y2", "ty2") - pl.max_horizontal("y1", "ty1")
    inter = w.clip(lower_bound=0) * h.clip(lower_bound=0)
    area = (pl.col("x2") - pl.col("x1")) * (pl.col("y2") - pl.col("y1"))
    t_area = (pl.col("tx2") - pl.col("tx1")) * (pl.col("ty2") - pl.col("ty1"))
    best = (
        r.select("i", "frame", "x1", "y1", "x2", "y2")
        .join(t, on="frame")
        .with_columns(cover=inter / area, iou=inter / (area + t_area - inter))
        .filter(pl.col("cover") >= JOIN_COVER)
        .sort("i", "cover", "iou", descending=[False, True, True])
        .unique("i", keep="first")
        .select("i", "track_id")
    )
    joined = reads.with_row_index("i").join(best, on="i", how="left", maintain_order="left")
    return joined["track_id_right"].alias("track_id")


@dataclass(frozen=True)
class _Mark:
    """The label of one track sample before segments are formed."""

    team: int | None
    source: LabelSource
    n_reads: int = 0


def _reach(
    edge: float, direction: int, read_t: NDArray[np.float64], p: NDArray[np.float64]
) -> float:
    """How far a label backed by reads up to `edge` extends in `direction` (+1 later, -1
    earlier): over the track's later (earlier) reads until EXTEND_STOP_READS in a row give the
    team less than EXTEND_STOP_P, or EXTEND_MAX_S. Returns the last time still labelled."""
    sel = read_t > edge if direction > 0 else read_t < edge
    ts, ps = read_t[sel], p[sel]
    if direction < 0:
        ts, ps = ts[::-1], ps[::-1]
    limit = edge + direction * EXTEND_MAX_S
    last_ok, bad = edge, 0
    for t, pt in zip(ts.tolist(), ps.tolist(), strict=True):
        if (t - limit) * direction > 0:
            break
        if pt < EXTEND_STOP_P:
            bad += 1
            if bad >= EXTEND_STOP_READS:
                return last_ok
        else:
            bad, last_ok = 0, t
    return last_ok if bad else limit


def _label_track(
    sample_t: NDArray[np.float64], reads: pl.DataFrame, roster: list[int]
) -> list[_Mark]:
    """Label each sample of one track from its joined, non-static reads (sorted by t).

    Agreeing named reads form runs; a run of at least LABEL_MIN_READS spanning
    LABEL_MIN_SPAN_S labels its own stretch (`digits`) and, with EXTEND_MIN_READS reads and on
    a side with no named read of another team, extends with `_reach` (`digits_extended`).
    Between named reads of two teams nothing is labelled (`conflict`): the track changed
    robot somewhere in there.
    """
    marks = [_Mark(None, "below_rule" if len(reads) else "no_reads")] * len(sample_t)
    named = reads.filter(~pl.col("abstain"))
    if named.is_empty():
        return marks
    read_t = reads["t"].to_numpy()
    probs = np.asarray(reads["probs"].to_list(), np.float64)
    runs: list[tuple[int, list[float]]] = []
    for t, team in zip(named["t"].to_list(), named["team"].to_list(), strict=True):
        if runs and runs[-1][0] == team:
            runs[-1][1].append(t)
        else:
            runs.append((team, [t]))

    def paint(lo: float, hi: float, mark: _Mark, closed: bool = True) -> None:
        for i, t in enumerate(sample_t.tolist()):
            if (lo <= t <= hi) if closed else (lo < t < hi):
                marks[i] = mark

    for (_, a), (_, b) in itertools.pairwise(runs):
        paint(a[-1], b[0], _Mark(None, "conflict"), closed=False)
    for i, (team, ts) in enumerate(runs):
        if len(ts) < LABEL_MIN_READS or ts[-1] - ts[0] < LABEL_MIN_SPAN_S - 1e-6:
            continue
        p = probs[:, roster.index(team)]
        ext = _Mark(team, "digits_extended", len(ts))
        extend = len(ts) >= EXTEND_MIN_READS
        if extend and i == 0:
            paint(_reach(ts[0], -1, read_t, p), ts[0], ext)
        if extend and i == len(runs) - 1:
            paint(ts[-1], _reach(ts[-1], +1, read_t, p), ext)
        paint(ts[0], ts[-1], _Mark(team, "digits", len(ts)))
    return marks


def _resolve_claims(labels: pl.DataFrame) -> tuple[pl.DataFrame, list[ClaimConflict]]:
    """Make sure no team is on two tracks at once. Labels are compared as time intervals
    (one per run of equal labels on a track). Where two tracks' intervals for one team
    overlap, a `digits` interval beats a `digits_extended` one, which loses all of itself
    (an extension that ran into another track's claim is not trusted past that point);
    two `digits` intervals both lose the overlap, and two extended ones both lose all.
    Losers become `conflict`. Returns the labels and the conflicts found."""
    runs = labels.with_row_index("i").with_columns(
        run=pl.struct("team", "source").rle_id().over("track_id")
    )
    spans = (
        runs.filter(pl.col("team").is_not_null())
        .group_by("track_id", "run")
        .agg(
            team=pl.col("team").first(),
            source=pl.col("source").first(),
            s=pl.col("t").min(),
            e=pl.col("t").max(),
            rows=pl.col("i"),
        )
        .to_dicts()
    )
    lost: set[int] = set()
    conflicts: list[ClaimConflict] = []
    for a, b in itertools.combinations(spans, 2):
        if a["team"] != b["team"] or a["track_id"] == b["track_id"]:
            continue
        lo, hi = max(a["s"], b["s"]), min(a["e"], b["e"])
        if lo > hi:
            continue
        backed = [x for x in (a, b) if x["source"] == "digits"]
        kept = backed[0] if len(backed) == 1 else None
        for x in (a, b):
            if x is kept:
                continue
            if x["source"] == "digits_extended":
                lost.update(x["rows"])
            else:
                lost.update(
                    i
                    for i, t in zip(x["rows"], runs["t"].gather(x["rows"]), strict=True)
                    if lo <= t <= hi
                )
        pair = sorted((a["track_id"], b["track_id"]))
        conflicts.append(
            ClaimConflict(
                team=a["team"],
                track_ids=(pair[0], pair[1]),
                t_start=lo,
                t_end=hi,
                kept=None if kept is None else kept["track_id"],
            )
        )
    is_lost = pl.Series(np.isin(np.arange(len(labels)), np.fromiter(lost, np.int64, len(lost))))
    out = labels.with_columns(
        source=pl.when(is_lost).then(pl.lit("conflict")).otherwise(pl.col("source")),
        n_reads=pl.when(is_lost).then(0).otherwise(pl.col("n_reads")),
        team=pl.when(is_lost).then(None).otherwise(pl.col("team")),
    )
    conflicts.sort(key=lambda c: (c.t_start, c.team))
    return out, conflicts


def _label(reads: pl.DataFrame, tracks: pl.DataFrame, roster: list[int]) -> Identity:
    """identity.json from reads (with track_id) and tracks."""
    joined = reads.filter(pl.col("track_id").is_not_null() & ~pl.col("static")).sort("t")
    by_track = {int(k): g for (k,), g in joined.group_by("track_id")}
    empty = joined.clear()
    cols: dict[str, list[int | float | str | None]] = {
        k: [] for k in ("track_id", "frame", "t", "team", "source", "n_reads")
    }
    for (tid,), g in tracks.sort("t").group_by("track_id", maintain_order=True):
        marks = _label_track(g["t"].to_numpy(), by_track.get(int(tid), empty), roster)
        cols["track_id"] += [int(tid)] * len(g)
        cols["frame"] += g["frame"].to_list()
        cols["t"] += g["t"].to_list()
        cols["team"] += [m.team for m in marks]
        cols["source"] += [m.source for m in marks]
        cols["n_reads"] += [m.n_reads for m in marks]
    labels = pl.DataFrame(
        cols,
        schema={
            "track_id": pl.Int64,
            "frame": pl.Int64,
            "t": pl.Float64,
            "team": pl.Int64,
            "source": pl.String,
            "n_reads": pl.Int64,
        },
    )
    labels, conflicts = _resolve_claims(labels)
    segments: list[LabelSegment] = []
    runs = labels.sort("track_id", "frame").with_columns(
        run=pl.struct("team", "source", "n_reads").rle_id().over("track_id")
    )
    for (tid, _), g in runs.group_by("track_id", "run", maintain_order=True):
        first, last = g.row(0, named=True), g.row(-1, named=True)
        team: int | None = first["team"]
        conf = 0.0
        if team is not None:
            rs = by_track.get(int(tid), empty).filter(pl.col("t").is_between(first["t"], last["t"]))
            if len(rs):
                k = roster.index(team)
                conf = float(np.median([p[k] for p in rs["probs"].to_list()]))
        segments.append(
            LabelSegment(
                track_id=int(tid),
                frame_start=first["frame"],
                frame_end=last["frame"],
                t_start=first["t"],
                t_end=last["t"],
                team=team,
                confidence=min(1.0, max(0.0, conf)),
                source=first["source"],
                n_reads=first["n_reads"],
            )
        )
    return Identity(segments=segments, conflicts=conflicts)


def _draw_identity(
    path: Path, ident: Identity, reads: pl.DataFrame, match: MatchInfo, duration: float
) -> None:
    """Write identity_check.jpg: one row per track over match time, coloured by label (the
    target team orange, other teams in their own colours, null grey, `conflict` dark red),
    a short white tick at the top of the row at each named read joined to the track, and a
    strip on top that is orange wherever some track carries the target team."""
    left, width, strip = 70, 1800, 34
    by_track: dict[int, list[LabelSegment]] = {}
    for s in ident.segments:
        by_track.setdefault(s.track_id, []).append(s)
    order = sorted(by_track, key=lambda k: by_track[k][0].t_start)
    rh = max(4, min(14, 1000 // max(1, len(order))))
    top = strip + 70
    img = np.full((top + rh * len(order) + 40, left + width + 20, 3), 20, np.uint8)

    def x(t: float) -> int:
        return left + round(t / duration * width)

    others = [s.team for s in match.roster if s.team != match.target_team]
    palette = [(255, 160, 60), (90, 200, 90), (220, 120, 220), (60, 220, 220), (200, 200, 90)]
    colour: dict[int, tuple[int, int, int]] = {match.target_team: (0, 140, 255)}
    colour |= {team: palette[i % len(palette)] for i, team in enumerate(others)}
    for s in ident.segments:
        if s.team == match.target_team:
            cv2.rectangle(
                img, (x(s.t_start), 6), (x(s.t_end) + 1, strip), colour[match.target_team], -1
            )
    cv2.putText(img, f"{match.target_team} labelled", (4, strip - 8), 0, 0.4, (255, 255, 255), 1)
    named = reads.filter(~pl.col("abstain") & pl.col("track_id").is_not_null())
    ticks = {int(k): g["t"].to_list() for (k,), g in named.group_by("track_id")}
    for row, tid in enumerate(order):
        y0 = top + row * rh
        for s in by_track[tid]:
            c = colour.get(s.team, (90, 90, 90)) if s.team is not None else (90, 90, 90)
            if s.source == "conflict":
                c = (40, 40, 150)
            cv2.rectangle(img, (x(s.t_start), y0), (x(s.t_end) + 1, y0 + rh - 2), c, -1)
        for t in ticks.get(tid, []):
            cv2.line(img, (x(t), y0), (x(t), y0 + max(1, rh // 3)), (255, 255, 255), 1)
        if rh >= 8 or row % 5 == 0:
            cv2.putText(img, str(tid), (4, y0 + rh - 2), 0, 0.3, (200, 200, 200), 1)
    bottom = top + rh * len(order)
    for t in range(0, int(duration) + 1, 10):
        cv2.line(img, (x(t), top - 4), (x(t), bottom), (60, 60, 60), 1)
        cv2.putText(img, str(t), (x(t) - 8, bottom + 16), 0, 0.4, (200, 200, 200), 1)
    for t in (match.auto_start_t, match.auto_end_t, match.teleop_start_t, match.teleop_end_t):
        cv2.line(img, (x(t), strip + 2), (x(t), bottom), (255, 255, 255), 1)
    legend = [(str(k), v) for k, v in colour.items()]
    legend += [("null", (90, 90, 90)), ("conflict", (40, 40, 150)), ("| named read", (20, 20, 20))]
    lx = left
    for name, c in legend:
        cv2.rectangle(img, (lx, strip + 18), (lx + 14, strip + 32), c, -1)
        cv2.putText(img, name, (lx + 18, strip + 30), 0, 0.45, (255, 255, 255), 1)
        lx += 30 + 10 * len(name)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def identify(run: RunDir) -> None:
    torch.set_num_threads(CPU_THREADS)
    cv2.setNumThreads(CPU_THREADS)
    info = VideoInfo.load(run.video_json)
    if info.interlaced:
        raise ValueError("interlaced video: the reader needs de-interlaced crops (not ported)")
    match = MatchInfo.load(run.match_json)
    tracks = read_tracks(run.tracks)
    timer = _Timer()
    start = time.perf_counter()
    reads, check_rows = _read_boxes(run, info, match, timer)
    mark = time.perf_counter()
    reads = reads.with_columns(_join_tracks(reads, tracks).cast(pl.Int32))
    write_reads(run.reads, reads)
    mark = timer.add("join + write reads", mark)
    _draw_check(run.path / "reads_check.jpg", check_rows, reads)
    mark = timer.add("reads sheet", mark)
    roster = [s.team for s in match.roster]
    ident = _label(reads, tracks, roster)
    ident.save(run.identity_json)
    mark = timer.add("label", mark)
    _draw_identity(run.path / "identity_check.jpg", ident, reads, match, info.duration_s)
    timer.add("identity sheet", mark)

    total = time.perf_counter() - start
    named = reads.filter(~pl.col("abstain"))
    joined = named.filter(pl.col("track_id").is_not_null())
    target = [s for s in ident.segments if s.team == match.target_team]
    print(
        f"identify: {len(reads)} boxes read, {len(named)} named, {len(joined)} named on a track;"
        f" {len(ident.segments)} segments, {len(target)} for {match.target_team},"
        f" {len(ident.conflicts)} same-team conflicts"
    )
    print("  " + "  ".join(f"{k} {v:.1f}s" for k, v in timer.s.items()) + f"  total {total:.1f}s")
    gpu = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"  peak RSS {rss:.2f} GiB, peak torch GPU {gpu:.2f} GiB")


STAGE = Stage(
    "identify",
    inputs=lambda r: (r.video, r.video_json, r.detections, r.tracks, r.match_json),
    outputs=lambda r: (r.identity_json, r.reads),
    fn=identify,
)
