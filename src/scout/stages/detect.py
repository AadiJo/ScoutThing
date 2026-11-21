"""detect: reads video, video.json, match.json.

Writes detections.parquet: raw bumper boxes per sampled frame per view in full-frame
pixels, unfiltered apart from a low confidence floor. Also writes
`detections_check.jpg`, a contact sheet of 12 frames across the match with every box
drawn, for a human to count against.

One ffmpeg decode pass at `SAMPLE_HZ` feeds a reader thread that hands over batches of
full frames. Each batch goes to the GPU once; the three camera panels are cut, resized
and padded there and run through the bumper detector in half precision, one batched
call per view.
"""

from __future__ import annotations

import queue
import resource
import threading
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray
from torchvision.ops import batched_nms  # pyright: ignore[reportMissingTypeStubs]
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

from scout.assets import AssetName, asset
from scout.run import RunDir, Stage
from scout.schemas.common import Alliance, PixelRect, ViewName
from scout.schemas.detections import COLOR, SAMPLE_HZ, VIEW, write_detections
from scout.schemas.match import MatchInfo
from scout.schemas.video import VideoInfo
from scout.video import frames

DETECTOR: AssetName = "bumpers_offpalette"
CONF_FLOOR = 0.10
NMS_IOU = 0.5
"""Per-class NMS overlap, the Ultralytics predict default."""
VIEW_SCALE: dict[ViewName, float] = {"main": 0.5, "corner_left": 1.0, "corner_right": 1.0}
"""Panel resize before inference. The main panel (1920x480) runs at 960x240, which found
more robots and fewer static false positives than 640, 1280 or 1920 wide on Qualification
16; corners run at native size, which beat 1024 and 1344."""
BATCH = 32
CHECK_FRAMES = 12
STRIDE = 32
PAD_VALUE = 114 / 255
CPU_THREADS = 8


@dataclass
class _Timer:
    """Wall seconds per pipeline step, printed at the end of the stage."""

    s: dict[str, float] = field(default_factory=dict[str, float])

    def add(self, key: str, since: float) -> float:
        now = time.perf_counter()
        self.s[key] = self.s.get(key, 0.0) + now - since
        return now


@dataclass
class _Rows:
    """Column buffers for detections.parquet, filled one batch at a time."""

    frame: list[NDArray[np.int64]] = field(default_factory=list[NDArray[np.int64]])
    view: list[NDArray[np.int8]] = field(default_factory=list[NDArray[np.int8]])
    box: list[NDArray[np.float32]] = field(default_factory=list[NDArray[np.float32]])
    cls: list[NDArray[np.int64]] = field(default_factory=list[NDArray[np.int64]])


class _Batch:
    """Sampled frames `index` with their video times `t` and the stacked BGR images."""

    def __init__(self, index: list[int], t: list[float], images: NDArray[np.uint8]) -> None:
        self.index = index
        self.t = t
        self.images = images


def _color_of(name: str) -> Alliance | None:
    """Alliance colour of a detector class name such as `blue_bumper`, or None."""
    for alliance in ("red", "blue"):
        if alliance in name.lower():
            return alliance
    return None


def _read_batches(video: Path, info: VideoInfo) -> Iterator[_Batch]:
    """Decode the video once at SAMPLE_HZ in a thread and yield batches of BATCH frames.

    At most two batches wait in the queue, so memory holds about three batches.
    """
    q: queue.Queue[_Batch | BaseException | None] = queue.Queue(maxsize=2)
    stop = threading.Event()

    def put(item: _Batch | BaseException | None) -> None:
        while not stop.is_set():
            try:
                q.put(item, timeout=0.2)
                return
            except queue.Full:
                continue

    def work() -> None:
        try:
            index: list[int] = []
            ts: list[float] = []
            images: list[NDArray[np.uint8]] = []
            for f in frames(video, info, sample_hz=SAMPLE_HZ):
                if stop.is_set():
                    return
                index.append(f.frame)
                ts.append(f.t)
                images.append(f.image)
                if len(images) == BATCH:
                    put(_Batch(index, ts, np.stack(images)))
                    index, ts, images = [], [], []
            if images:
                put(_Batch(index, ts, np.stack(images)))
            put(None)
        except BaseException as e:
            put(e)

    thread = threading.Thread(target=work, name="detect-decode", daemon=True)
    thread.start()
    try:
        while (item := q.get()) is not None:
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        thread.join()


def _panel_tensor(rgb: torch.Tensor, rect: PixelRect, scale: float) -> torch.Tensor:
    """Cut one panel from a BCHW 0-1 batch, resize by `scale`, and pad right and bottom
    to a multiple of STRIDE, as the detector's tensor input requires."""
    p = rgb[:, :, rect.y : rect.y + rect.h, rect.x : rect.x + rect.w]
    if scale != 1.0:
        size = (round(rect.h * scale), round(rect.w * scale))
        p = F.interpolate(p, size=size, mode="bilinear", align_corners=False)
    h, w = p.shape[2], p.shape[3]
    return F.pad(p, (0, -w % STRIDE, 0, -h % STRIDE), value=PAD_VALUE)


def _load_net(device: torch.device, half: bool) -> DetectionModel:
    """The bumper detector's network, fused and in eval mode on `device`, for direct
    batched calls. Skipping `YOLO.predict` drops its per-call CPU work, which left the GPU
    idle more than half the time."""
    net = YOLO(str(asset(DETECTOR))).model
    if not isinstance(net, DetectionModel):
        raise TypeError(f"{DETECTOR} is not a detection model")
    net = net.fuse(verbose=False).to(device).eval()
    return net.half() if half else net


def _nms(raw: object) -> tuple[torch.Tensor, torch.Tensor]:
    """Boxes above CONF_FLOOR after per-image, per-class NMS, for a whole batch in one call.

    `raw` is the network output: a `(B, 4 + classes, anchors)` tensor of centre-size boxes
    and class scores, or a tuple starting with one. Returns each kept box's image index in
    the batch and its `x1 y1 x2 y2 conf cls` row. One batched call instead of Ultralytics'
    per-image loop.
    """
    pred = raw[0] if isinstance(raw, tuple) else raw
    if not isinstance(pred, torch.Tensor):
        raise TypeError(f"unexpected detector output {type(pred).__name__}")
    p = pred.transpose(1, 2).float()
    n_classes = p.shape[2] - 4
    conf, cls = p[..., 4:].max(2)
    img, anchor = (conf > CONF_FLOOR).nonzero(as_tuple=True)
    xy, wh = p[img, anchor, :2], p[img, anchor, 2:4]
    xyxy = torch.cat([xy - wh / 2, xy + wh / 2], 1)
    conf, cls = conf[img, anchor], cls[img, anchor]
    keep: torch.Tensor = batched_nms(xyxy, conf, img * n_classes + cls, NMS_IOU)
    return img[keep], torch.cat([xyxy[keep], conf[keep, None], cls[keep, None].float()], 1)


def _check_targets(match: MatchInfo, sampled: Sequence[int], info: VideoInfo) -> set[int]:
    """CHECK_FRAMES sampled frame indices spread evenly from auto start to teleop end."""
    times = np.linspace(match.auto_start_t + 2.0, match.teleop_end_t - 2.0, CHECK_FRAMES)
    arr = np.asarray(sampled)
    return {int(arr[np.abs(arr / info.fps - t).argmin()]) for t in times}


def _draw_sheet(
    path: Path, shots: dict[int, NDArray[np.uint8]], df: pl.DataFrame, info: VideoInfo
) -> None:
    """Write the contact sheet: each check frame at half size with every box in its
    colour (thick at conf >= 0.35, thin below) labelled with its confidence."""
    bgr = {"red": (0, 0, 255), "blue": (255, 128, 0), "unknown": (0, 255, 0)}
    tiles: list[NDArray[np.uint8]] = []
    for idx in sorted(shots):
        im = shots[idx].copy()
        for r in df.filter(pl.col("frame") == idx).iter_rows(named=True):
            p1 = (round(r["x1"]), round(r["y1"]))
            p2 = (round(r["x2"]), round(r["y2"]))
            cv2.rectangle(im, p1, p2, bgr[r["color"]], 3 if r["conf"] >= 0.35 else 1)
            cv2.putText(im, f"{r['conf']:.2f}", (p1[0], p1[1] - 4), 0, 0.8, (255, 255, 255), 2)
        im = cv2.resize(im, (info.width // 2, info.height // 2), interpolation=cv2.INTER_AREA)
        im = np.asarray(cv2.copyMakeBorder(im, 40, 4, 2, 2, cv2.BORDER_CONSTANT), np.uint8)
        label = f"frame {idx}  t={idx / info.fps:.1f}s"
        cv2.putText(im, label, (10, 30), 0, 1.0, (0, 255, 255), 2)
        tiles.append(im)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i : i + 3]) for i in range(0, len(tiles), 3)]
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def detect(run: RunDir) -> None:
    torch.set_num_threads(CPU_THREADS)
    cv2.setNumThreads(CPU_THREADS)
    info = VideoInfo.load(run.video_json)
    match = MatchInfo.load(run.match_json)
    views = list(match.views.items())
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    dtype = torch.float16 if cuda else torch.float32
    net = _load_net(device, half=cuda)
    names: dict[int, str] = {int(k): str(v) for k, v in net.names.items()}
    colors = [_color_of(names[k]) or "unknown" for k in sorted(names)]

    step = max(1, round(info.fps / SAMPLE_HZ))
    targets = _check_targets(match, range(0, info.frame_count, step), info)
    shots: dict[int, NDArray[np.uint8]] = {}
    rows = _Rows()
    timer = _Timer()
    n_frames = 0
    times: dict[int, float] = {}
    start = time.perf_counter()
    mark = start
    for batch in _read_batches(run.video, info):
        mark = timer.add("decode wait", mark)
        n_frames += len(batch.index)
        times.update(zip(batch.index, batch.t, strict=True))
        for i, idx in enumerate(batch.index):
            if idx in targets:
                shots[idx] = batch.images[i].copy()
        frame_ids = torch.tensor(batch.index, device=device)
        rgb = torch.from_numpy(batch.images).to(device).permute(0, 3, 1, 2).flip(1)
        rgb = rgb.to(dtype).div_(255)
        for code, (name, rect) in enumerate(views):
            scale = VIEW_SCALE[name]
            x = _panel_tensor(rgb, rect, scale)
            if cuda:
                torch.cuda.synchronize()
            mark = timer.add("gpu preprocess", mark)
            with torch.inference_mode():
                raw = net(x)
                if cuda:
                    torch.cuda.synchronize()
                mark = timer.add("inference", mark)
                img, data = _nms(raw)
            mark = timer.add("nms", mark)
            offset = torch.tensor([rect.x, rect.y, rect.x, rect.y], device=device)
            box = torch.cat([data[:, :4] / scale + offset, data[:, 4:5]], 1)
            rows.frame.append(frame_ids[img].cpu().numpy())
            rows.view.append(np.full(len(data), code, np.int8))
            rows.box.append(box.cpu().numpy().astype(np.float32))
            rows.cls.append(data[:, 5].long().cpu().numpy())
            mark = timer.add("collect", mark)
        del rgb
    if n_frames != len(range(0, info.frame_count, step)):
        raise RuntimeError(f"decoded {n_frames} sampled frames, expected every {step}th")

    box = np.concatenate(rows.box) if rows.box else np.zeros((0, 5), np.float32)
    frame_col = np.concatenate(rows.frame) if rows.frame else np.zeros(0, np.int64)
    view_names = [n for n, _ in views]
    df = pl.DataFrame(
        {
            "frame": frame_col,
            "t": pl.Series([times[f] for f in frame_col.tolist()], dtype=pl.Float64),
            "view": pl.Series(
                [view_names[c] for c in np.concatenate(rows.view)] if rows.view else [],
                dtype=pl.String,
            ),
            "x1": box[:, 0],
            "y1": box[:, 1],
            "x2": box[:, 2],
            "y2": box[:, 3],
            "conf": box[:, 4],
            "color": pl.Series(
                [colors[c] for c in np.concatenate(rows.cls)] if rows.cls else [],
                dtype=pl.String,
            ),
            "detector": pl.Series([DETECTOR] * len(box), dtype=pl.String),
        }
    ).with_columns(pl.col("view").cast(VIEW), pl.col("color").cast(COLOR))
    df = df.sort("frame", "view", "conf", descending=[False, False, True])
    write_detections(run.detections, df)
    mark = timer.add("write parquet", mark)
    _draw_sheet(run.path / "detections_check.jpg", shots, df, info)
    timer.add("contact sheet", mark)

    total = time.perf_counter() - start
    print(f"detect: {len(df)} boxes over {n_frames} sampled frames x {len(views)} views")
    print("  " + "  ".join(f"{k} {v:.1f}s" for k, v in timer.s.items()) + f"  total {total:.1f}s")
    gpu = torch.cuda.max_memory_allocated() / 2**30 if cuda else 0.0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    print(f"  peak RSS {rss:.2f} GiB, peak torch GPU {gpu:.2f} GiB")


STAGE = Stage(
    "detect",
    inputs=lambda r: (r.video, r.video_json, r.match_json),
    outputs=lambda r: (r.detections,),
    fn=detect,
)
