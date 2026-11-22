"""The bumper number reader: a PARSeq text recogniser fine-tuned on FRC bumper crops.

    reader = DigitReader.load(asset("digits"), device)
    x = reader.batch([reader.model_input(image, box, panel, Crop(0.0, 0.0)) for ...])
    scores = reader.score(x, ["2468", "6800", ...])

`model_input` cuts a region around a detector box (`Crop`; training used `Crop(0.25, 0.1)`
on bumper boxes), clamped to its camera panel, and resizes it to 32x128 RGB with the same
OpenCV interpolation as training. Crops are prepared on the CPU (each is a few hundred
pixels) and scored in GPU batches.

Ported from frc-vision-scorer's web/students/digit_reader.py. The source de-interlaces
each crop first (bumper_register.deinterlace_crop, when `comb_ratio` > 0.9). That step is
not ported: a woven interlaced video would need it at the top of `model_input`, and the
identify stage refuses interlaced video until it exists.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple, Self

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from torch import Tensor

from scout.readers.parseq import PARSeq, encode_strings
from scout.schemas.common import PixelRect

IMG_H, IMG_W = 32, 128
CHECKPOINT_ARCH = "parseq-base digits"


class _Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_label_length: int
    img_size: tuple[int, int]
    patch_size: tuple[int, int]
    embed_dim: int
    enc_num_heads: int
    enc_mlp_ratio: int
    enc_depth: int
    dec_num_heads: int
    dec_mlp_ratio: int
    dec_depth: int
    dropout: float


class _Meta(BaseModel):
    """The checkpoint fields the reader uses; the rest (rules, report) is ignored."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    arch: str
    config: _Config


class Crop(NamedTuple):
    """How much of the frame around a detector box the model reads: `pad` of the box size
    added per side, then `trim` of the padded size removed per side."""

    pad: float
    trim: float


class Scores(NamedTuple):
    """Reader output for a batch of N crops against K candidate strings."""

    lp: NDArray[np.float64]
    """`[N, K]` log P(candidate string | crop), teacher forced."""
    text: list[str]
    """Free (unconstrained greedy) read per crop; may be empty."""
    text_lp: NDArray[np.float64]
    """`[N]` log P of each free read."""


class DigitReader:
    """The loaded network plus its crop geometry. Build with `DigitReader.load`."""

    def __init__(self, model: PARSeq, meta: _Meta, device: torch.device, dtype: torch.dtype):
        self.model = model
        self.max_len = meta.config.max_label_length
        self.device = device
        self.dtype = dtype

    @classmethod
    def load(cls, path: Path, device: torch.device) -> Self:
        """Load the checkpoint; half precision on CUDA, float32 on CPU."""
        raw: object = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(raw, dict):
            raise TypeError(f"{path}: checkpoint is not a dict")
        ck: dict[str, object] = {str(k): v for k, v in raw.items()}
        meta = _Meta.model_validate(ck)
        if meta.arch != CHECKPOINT_ARCH or meta.config.img_size != (IMG_H, IMG_W):
            raise ValueError(f"{path}: unexpected reader {meta.arch} {meta.config.img_size}")
        sd = ck["state_dict"]
        if not isinstance(sd, dict):
            raise TypeError(f"{path}: state_dict is not a dict")
        state = {str(k): v for k, v in sd.items() if isinstance(v, Tensor)}
        model = PARSeq(**meta.config.model_dump())
        model.load_state_dict(state)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        model = model.to(device=device, dtype=dtype).eval()
        return cls(model, meta, device, dtype)

    def crop_rect(
        self, box: Sequence[float], panel: PixelRect, crop: Crop
    ) -> tuple[int, int, int, int]:
        """`x0 y0 x1 y1` the model reads for detector box `x1 y1 x2 y2`: padded by `crop.pad`
        per side (clamped to `panel`), then `crop.trim` of that cut off every side when at
        least 4 px remain."""
        bw, bh = box[2] - box[0], box[3] - box[1]
        x0 = max(panel.x, math.floor(box[0] - crop.pad * bw))
        y0 = max(panel.y, math.floor(box[1] - crop.pad * bh))
        x1 = min(panel.x + panel.w, math.ceil(box[2] + crop.pad * bw))
        y1 = min(panel.y + panel.h, math.ceil(box[3] + crop.pad * bh))
        w, h = x1 - x0, y1 - y0
        dy, dx = int(h * crop.trim), int(w * crop.trim)
        if h - 2 * dy >= 4 and w - 2 * dx >= 4:
            x0, y0, x1, y1 = x0 + dx, y0 + dy, x1 - dx, y1 - dy
        return x0, y0, x1, y1

    def model_input(
        self, image: NDArray[np.uint8], box: Sequence[float], panel: PixelRect, crop: Crop
    ) -> NDArray[np.uint8]:
        """The 32x128 RGB uint8 image the model reads for one box of a full BGR frame."""
        x0, y0, x1, y1 = self.crop_rect(box, panel, crop)
        cut = image[y0:y1, x0:x1]
        h, w = cut.shape[:2]
        if h < 1 or w < 1:
            return np.zeros((IMG_H, IMG_W, 3), np.uint8)
        interp = cv2.INTER_AREA if (w > IMG_W and h > IMG_H) else cv2.INTER_CUBIC
        rgb = cv2.cvtColor(cut, cv2.COLOR_BGR2RGB)
        return np.asarray(cv2.resize(rgb, (IMG_W, IMG_H), interpolation=interp), np.uint8)

    def batch(self, images: Sequence[NDArray[np.uint8]]) -> Tensor:
        """Stack `model_input` images into a `[N, 3, 32, 128]` tensor in [-1, 1] on device."""
        x = torch.from_numpy(np.stack(images)).to(self.device).permute(0, 3, 1, 2)
        return x.to(self.dtype).div_(127.5).sub_(1.0)

    def score(self, x: Tensor, candidates: Sequence[str]) -> Scores:
        """log P of every candidate digit string and the free read, for one batch."""
        tokens = encode_strings(candidates, self.max_len, self.device)
        with torch.inference_mode():
            memory = self.model.encode(x)
            lp = self.model.score_strings(memory, tokens)
            text, text_lp = self.model.greedy(memory)
        return Scores(lp.double().cpu().numpy(), text, text_lp.double().cpu().numpy())
