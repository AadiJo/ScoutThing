"""identify: reads video, video.json, detections.parquet, tracks.parquet, match.json.

Writes reads.parquet: the bumper number reader's verdict on every detector box in every
view, scored against the six roster teams, with the track each main-view box joins. Writes
identity.json: which stretch of which track is which team, or null when unsure. Also writes
`reads_check.jpg` (crops exactly as the reader sees them, sampled across views and
confidence) and `identity_check.jpg` (one timeline row per track, coloured by label, hatched
where the label is not from number reads, with a tick per named read).

Reading: one ffmpeg decode pass at the detector's SAMPLE_HZ. Each detector box is cut for the
number reader and each measured track box for the appearance embedder on the CPU; full
batches go to the GPU in half precision. Only one decoded frame is held at a time.

Labelling: named reads are sparse anchors on tracks (tracks are main view only, so corner
reads never label). A few agreeing reads label a stretch of a track, which may carry on
past them while the weaker reads do not contradict; named reads of two teams on one track
mark a change of robot, and the stretch between them stays null. No team is left on two
tracks at once. Then each remaining null stretch (not `conflict`) may be named from its
appearance against this match's number-read boxes, among the teams not labelled elsewhere
at the time, with continuity loosening the bar (`_label_unread`). Those labels never
replace a number-read label and are marked `appearance*` in `source`.
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
from scout.readers.embedder import Embedder
from scout.run import RunDir, Stage
from scout.schemas.common import ViewName
from scout.schemas.detections import SAMPLE_HZ, read_detections
from scout.schemas.identity import (
    READER_NAME,
    AppearanceEvidence,
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

EMBED_BATCH = 256
"""Appearance crops per GPU batch (224x224 each)."""
APPEAR_TOPK = 10
"""A box's score for a team is the mean of its APPEAR_TOPK highest cosines to that team's
gallery (a nearest-neighbour score; a single mean vector per team picked the true team for
83 of 102 clean Qualification 8 stretches against 90 for this)."""
GALLERY_MIN = 20
"""A team needs this many embedded `digits` boxes to be scored. A candidate team below it
(never read, like 9066 in Qualification 8) is an unseen robot: appearance cannot rule it out,
so the stretch must then reach UNSEEN_MIN_COS."""
APPEAR_MIN_BOXES = 5
"""An `appearance` label needs this many embedded boxes (0.5 s). On the development runs,
stretches of 1-4 boxes held 9 of the 13 wrong appearance-only stretches at margin 0.1."""
APPEAR_MARGIN = 0.10
"""cosine units: the chosen team must beat every other candidate by this. At 0.05 the
development runs gained 4% more correct samples and 1 more wrong stretch; at 0.2, 15% fewer
correct and the same wrong ones (those come from wrong eliminations, not close calls)."""
APPEAR_MIN_COS = 0.60
"""cosine: below this the stretch looks like none of the gallery (cross-match, 0.59 passed
89% of true pairs and 9% of false ones). Barely binding at margin 0.1."""
UNSEEN_MIN_COS = 0.75
"""cosine: the bar when some candidate team has no gallery. 0.75 kept 29 of 31 judged
Qualification 8 stretches correct; with no bar, Playoff 12 labelled 9506 and 9988 (both
unread, 9506 an open-top lookalike of 2468) as 2468 in 7 short stretches."""
CONT_MARGIN = 0.05
CONT_MIN_BOXES = 1
"""`appearance_continuity` labels: appearance bars when continuity agrees (cosine margin;
embedded boxes). Continuity alone was 91% precise (553/605 samples), appearance at these
looser bars alone 96-97%; agreeing, 103 of 104 samples were right."""
CONT_GAP_S = 2.0
CONT_SPEED_MPS = 5.5
CONT_NOISE_M = 0.35
"""A track that ended at most CONT_GAP_S before a stretch starts (or began at most that
after it ends) continues it when the gap is drivable at CONT_SPEED_MPS plus CONT_NOISE_M of
position noise; only a sole such track counts. The track stage's JOIN_MAX_S, MAX_SPEED_MPS
and NOISE_M (repeated here: stages do not import each other)."""

HATCH = {"appearance": 5, "appearance_continuity": 3}
"""identity_check.jpg stripe spacing in pixels for labels not from number reads."""
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


@dataclass
class _Embeds:
    """Appearance vectors of measured track samples: `index[(track_id, frame)]` is a row of
    `vecs`."""

    index: dict[tuple[int, int], int]
    vecs: NDArray[np.float32]


def _read_boxes(
    run: RunDir, info: VideoInfo, match: MatchInfo, tracks: pl.DataFrame, timer: _Timer
) -> tuple[pl.DataFrame, dict[int, NDArray[np.uint8]], _Embeds]:
    """Read every detector box and embed every measured track box in one decode pass.
    Returns reads (track_id still null) in detection order, the candidate crops for
    reads_check.jpg by row index, and the track box embeddings."""
    teams = [str(s.team) for s in match.roster]
    dets = (
        read_detections(run.detections)
        .filter(pl.col("kind") == "bumper")
        .sort("frame", "view", "conf", descending=[False, False, True])
    )
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    mark = time.perf_counter()
    reader = DigitReader.load(asset("digits"), device)
    embedder = Embedder.load(asset("embedder"), device)
    mark = timer.add("load model", mark)
    main_panel = match.views["main"]
    to_embed: dict[int, list[tuple[int, list[float]]]] = {}
    for tid, fr, x1, y1, x2, y2 in (
        tracks.filter(pl.col("state") == "measured")
        .select("track_id", "frame", "x1", "y1", "x2", "y2")
        .iter_rows()
    ):
        box = [float(x1), float(y1), float(x2), float(y2)]
        if Embedder.usable(box):
            to_embed.setdefault(int(fr), []).append((int(tid), box))
    emb_keys: list[tuple[int, int]] = []
    emb_out: list[NDArray[np.float32]] = []
    emb_pending: list[NDArray[np.uint8]] = []

    def flush_embed() -> float:
        nonlocal emb_pending
        t0 = time.perf_counter()
        emb_out.append(embedder.embed(emb_pending))
        emb_pending = []
        return timer.add("gpu embed", t0)

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

    last = max([int(frame_col.max()) if len(frame_col) else -1, *to_embed])
    end_t = (last + 1) / info.fps
    n_frames = 0
    for f in frames(run.video, info, sample_hz=SAMPLE_HZ, end=end_t):
        mark = timer.add("decode", mark)
        for tid, box in to_embed.get(f.frame, []):
            emb_keys.append((tid, f.frame))
            emb_pending.append(embedder.model_input(f.image, box, main_panel))
        mark = timer.add("embed crop", mark)
        if len(emb_pending) >= EMBED_BATCH:
            mark = flush_embed()
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
    if emb_pending:
        mark = flush_embed()
    if len(emb_keys) != sum(len(v) for v in to_embed.values()):
        raise RuntimeError("not every track box was embedded")
    embeds = _Embeds(
        {k: i for i, k in enumerate(emb_keys)},
        np.concatenate(emb_out) if emb_out else np.zeros((0, embedder.dim), np.float32),
    )
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
    return reads, check_rows, embeds


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


@dataclass(frozen=True)
class _End:
    """One end of a track: time, field position and its label there (None if unlabelled)."""

    track_id: int
    t: float
    x: float
    y: float
    team: int | None


def _continuers(stretch_end: _End, ends: list[_End], after: bool) -> list[_End]:
    """Track ends that could be this robot: ends of other tracks within CONT_GAP_S before
    (`after` False: tracks ending before the stretch starts) or after it, drivable at
    CONT_SPEED_MPS plus CONT_NOISE_M."""
    out: list[_End] = []
    for e in ends:
        if e.track_id == stretch_end.track_id:
            continue
        gap = e.t - stretch_end.t if after else stretch_end.t - e.t
        if not 0 < gap <= CONT_GAP_S + 1e-9:
            continue
        if (
            np.hypot(e.x - stretch_end.x, e.y - stretch_end.y)
            <= CONT_SPEED_MPS * gap + CONT_NOISE_M
        ):
            out.append(e)
    return out


def _label_unread(
    labels: pl.DataFrame, tracks: pl.DataFrame, embeds: _Embeds, roster: list[int]
) -> tuple[pl.DataFrame, dict[tuple[int, int], AppearanceEvidence]]:
    """Name null stretches (not `conflict`) of tracks from appearance, elimination and
    continuity. Returns the labels and the evidence per (track_id, first frame) of each named
    stretch.

    Per stretch: candidates are roster teams not labelled on any other track at any of its
    frames (one robot, one place). Each candidate with a gallery (GALLERY_MIN `digits` boxes)
    gets the stretch's mean APPEAR_TOPK score; the best must beat the runner-up by
    APPEAR_MARGIN, reach APPEAR_MIN_COS (UNSEEN_MIN_COS if a candidate has no gallery), be the
    best over the whole roster too (a team labelled elsewhere that looks more alike vetoes),
    and have APPEAR_MIN_BOXES boxes: `appearance`. Continuity names a team when the track is
    labelled it just before or after the stretch, or the sole track that ended (began) next
    to its start (end) is. When every continuity team equals the appearance pick, the looser
    CONT_MARGIN and CONT_MIN_BOXES apply (a continuity team that disagrees did not need to
    veto an `appearance` label: on the development runs that veto removed 67 right samples
    and no wrong ones):
    `appearance_continuity`. Two new stretches of one team that overlap in time both lose."""
    xy = tracks.select("track_id", "frame", "x", "y")
    lab = labels.join(xy, on=["track_id", "frame"], how="left").sort("track_id", "frame")
    lab = lab.with_row_index("i").with_columns(
        run=pl.struct(pl.col("team").is_null(), "source").rle_id().over("track_id")
    )

    gallery: dict[int, NDArray[np.float32]] = {}
    for team in roster:
        keys = lab.filter((pl.col("source") == "digits") & (pl.col("team") == team))
        rows = [
            embeds.index[k]
            for k in keys.select("track_id", "frame").iter_rows()
            if k in embeds.index
        ]
        if len(rows) >= GALLERY_MIN:
            gallery[team] = embeds.vecs[rows]
    named = lab.filter(pl.col("team").is_not_null())
    busy = {int(team): set(g["frame"].to_list()) for (team,), g in named.group_by("team")}

    def end(g: pl.DataFrame, k: int) -> _End:
        r = g.row(k, named=True)
        return _End(r["track_id"], r["t"], r["x"], r["y"], r["team"])

    by_track = {int(tid): g for (tid,), g in lab.group_by("track_id")}
    starts = [end(g, 0) for g in by_track.values()]
    stops = [end(g, -1) for g in by_track.values()]

    picks: list[tuple[int, list[int], int, str, AppearanceEvidence]] = []
    open_runs = lab.filter(pl.col("team").is_null() & (pl.col("source") != "conflict"))
    for (tid, _), g in open_runs.group_by("track_id", "run", maintain_order=True):
        frames_ = g["frame"].to_list()
        whole = by_track[int(tid)]
        cands = [t for t in roster if not (busy.get(t, set()) & set(frames_))]
        rows = [embeds.index[(int(tid), f)] for f in frames_ if (int(tid), f) in embeds.index]
        if not cands or not rows:
            continue
        sims = embeds.vecs[rows]
        score = {
            t: float(np.sort(sims @ gal.T, axis=1)[:, -APPEAR_TOPK:].mean())
            for t, gal in gallery.items()
        }
        scored = sorted((t for t in cands if t in score), key=lambda t: -score[t])
        if not scored or max(score, key=lambda t: score[t]) != scored[0]:
            continue
        best = scored[0]
        runner = scored[1] if len(scored) > 1 else None
        margin = score[best] - score[runner] if runner is not None else 9.0
        min_cos = UNSEEN_MIN_COS if len(scored) < len(cands) else APPEAR_MIN_COS
        if score[best] < min_cos:
            continue
        before = whole.filter(pl.col("frame") < frames_[0])
        after = whole.filter(pl.col("frame") > frames_[-1])
        pred = [end(before, -1)] if len(before) else _continuers(end(g, 0), stops, after=False)
        succ = [end(after, 0)] if len(after) else _continuers(end(g, -1), starts, after=True)
        cont = {e[0].team for e in (pred, succ) if len(e) == 1 and e[0].team is not None}
        if margin >= APPEAR_MARGIN and len(rows) >= APPEAR_MIN_BOXES:
            source = "appearance"
        elif cont == {best} and margin >= CONT_MARGIN and len(rows) >= CONT_MIN_BOXES:
            source = "appearance_continuity"
        else:
            continue
        ev = AppearanceEvidence(
            cosine=score[best],
            margin=margin,
            runner_up=runner,
            n_boxes=len(rows),
            candidates=cands,
        )
        picks.append((int(tid), frames_, best, source, ev))

    clash: set[int] = set()
    for (i, a), (j, b) in itertools.combinations(enumerate(picks), 2):
        if a[2] == b[2] and set(a[1]) & set(b[1]):
            clash |= {i, j}
    team_col = labels["team"].to_list()
    source_col = labels["source"].to_list()
    where = {
        (t, f): k for k, (t, f) in enumerate(zip(labels["track_id"], labels["frame"], strict=True))
    }
    evidence: dict[tuple[int, int], AppearanceEvidence] = {}
    for k, (tid, frames_, team, source, ev) in enumerate(picks):
        if k in clash:
            continue
        for f in frames_:
            team_col[where[(tid, f)]] = team
            source_col[where[(tid, f)]] = source
        evidence[(tid, frames_[0])] = ev
    out = labels.with_columns(
        team=pl.Series(team_col, dtype=pl.Int64), source=pl.Series(source_col, dtype=pl.String)
    )
    return out, evidence


def _label(
    reads: pl.DataFrame, tracks: pl.DataFrame, roster: list[int], embeds: _Embeds
) -> Identity:
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
    labels, evidence = _label_unread(labels, tracks, embeds, roster)
    segments: list[LabelSegment] = []
    runs = labels.sort("track_id", "frame").with_columns(
        run=pl.struct("team", "source", "n_reads").rle_id().over("track_id")
    )
    for (tid, _), g in runs.group_by("track_id", "run", maintain_order=True):
        first, last = g.row(0, named=True), g.row(-1, named=True)
        team: int | None = first["team"]
        conf = 0.0
        ev = evidence.get((int(tid), first["frame"]))
        if ev is not None:
            conf = ev.cosine
        elif team is not None:
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
                appearance=ev,
            )
        )
    return Identity(segments=segments, conflicts=conflicts)


def _draw_identity(
    path: Path, ident: Identity, reads: pl.DataFrame, match: MatchInfo, duration: float
) -> None:
    """Write identity_check.jpg: one row per track over match time, coloured by label (the
    target team orange, other teams in their own colours, null grey, `conflict` dark red),
    a short white tick at the top of the row at each named read joined to the track, and a
    strip on top that is orange wherever some track carries the target team. Labels from
    number reads are solid; `appearance*` labels are hatched with dark stripes (sparse for
    `appearance`, dense for `appearance_continuity`), in the strip too."""
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

    def fill(x0: int, y0: int, x1: int, y1: int, c: tuple[int, int, int], source: str) -> None:
        cv2.rectangle(img, (x0, y0), (x1, y1), c, -1)
        step = HATCH.get(source)
        if step:
            for xx in range(x0 - (y1 - y0), x1, step):
                cv2.line(img, (xx, y1), (xx + (y1 - y0), y0), (20, 20, 20), 1)
            cv2.rectangle(img, (x0, y0), (x1, y1), c, 1)

    others = [s.team for s in match.roster if s.team != match.target_team]
    palette = [(255, 160, 60), (90, 200, 90), (220, 120, 220), (60, 220, 220), (200, 200, 90)]
    colour: dict[int, tuple[int, int, int]] = {match.target_team: (0, 140, 255)}
    colour |= {team: palette[i % len(palette)] for i, team in enumerate(others)}
    for s in ident.segments:
        if s.team == match.target_team:
            fill(x(s.t_start), 6, x(s.t_end) + 1, strip, colour[match.target_team], s.source)
    cv2.putText(img, f"{match.target_team} labelled", (4, strip - 8), 0, 0.4, (255, 255, 255), 1)
    named = reads.filter(~pl.col("abstain") & pl.col("track_id").is_not_null())
    ticks = {int(k): g["t"].to_list() for (k,), g in named.group_by("track_id")}
    for row, tid in enumerate(order):
        y0 = top + row * rh
        for s in by_track[tid]:
            c = colour.get(s.team, (90, 90, 90)) if s.team is not None else (90, 90, 90)
            if s.source == "conflict":
                c = (40, 40, 150)
            fill(x(s.t_start), y0, x(s.t_end) + 1, y0 + rh - 2, c, s.source)
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
    legend += [("appearance", (200, 200, 200)), ("appearance_continuity", (200, 200, 200))]
    lx = left
    for name, c in legend:
        fill(lx, strip + 18, lx + 14, strip + 32, c, name)
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
    reads, check_rows, embeds = _read_boxes(run, info, match, tracks, timer)
    mark = time.perf_counter()
    reads = reads.with_columns(_join_tracks(reads, tracks).cast(pl.Int32))
    write_reads(run.reads, reads)
    mark = timer.add("join + write reads", mark)
    _draw_check(run.path / "reads_check.jpg", check_rows, reads)
    mark = timer.add("reads sheet", mark)
    roster = [s.team for s in match.roster]
    ident = _label(reads, tracks, roster, embeds)
    ident.save(run.identity_json)
    mark = timer.add("label", mark)
    _draw_identity(run.path / "identity_check.jpg", ident, reads, match, info.duration_s)
    timer.add("identity sheet", mark)

    total = time.perf_counter() - start
    named = reads.filter(~pl.col("abstain"))
    joined = named.filter(pl.col("track_id").is_not_null())
    target = [s for s in ident.segments if s.team == match.target_team]
    by_source: dict[str, float] = {}
    for s in ident.segments:
        if s.team is not None:
            by_source[s.source] = by_source.get(s.source, 0.0) + s.t_end - s.t_start + 1 / SAMPLE_HZ
    print(
        f"identify: {len(reads)} boxes read, {len(named)} named, {len(joined)} named on a track,"
        f" {len(embeds.index)} track boxes embedded; {len(ident.segments)} segments,"
        f" {len(target)} for {match.target_team}, {len(ident.conflicts)} same-team conflicts"
    )
    print("  labelled track time: " + ", ".join(f"{k} {v:.0f}s" for k, v in by_source.items()))
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
