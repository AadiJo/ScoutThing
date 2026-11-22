"""The robot appearance embedder: DINOv3 ViT-S/16 plus a small head, fine-tuned contrastively
on pairs of the same robot. Output is a 128-d L2-normalised vector per robot box.

    emb = Embedder.load(asset("embedder"), device)
    vecs = emb.embed([emb.model_input(image, box, panel) for ...])  # [N, 128] float32

`model_input` pads the box 10% per side (clamped to its panel), letterboxes it onto a
mean-grey square and resizes to 224 RGB uint8 on the CPU; `embed` normalises on the GPU.
Cosines are only comparable within one camera view (main against corner is about 0.62 for
the same robot).

Ported from frc-vision-scorer's web/students/s4_embedder.py (`build_net`, `load_net`,
`letterbox`, `robot_crop`). Not ported: the field de-interlace (the identify stage refuses
interlaced video). The letterbox resizes the crop into the square directly instead of
resizing a padded canvas; the two differ only at the border pixels.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import cv2
import numpy as np
import timm  # pyright: ignore[reportMissingTypeStubs]
import torch
import torch.nn.functional as F  # noqa: N812
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn

from scout.schemas.common import PixelRect

BACKBONE = "vit_small_patch16_dinov3"
"""timm 1.0.31 builds this with the checkpoint's exact keys and shapes, no download."""
SIZE = 224
PAD = 0.10
MIN_BOX_PX = 10
"""Boxes narrower or shorter than this give no embedding (the source's floor)."""
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
_GREY = tuple(round(m * 255) for m in MEAN)


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    train_blocks: int
    dim: int


class _Input(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    pad: float
    mode: str
    size: int


class _Meta(BaseModel):
    """The checkpoint fields the embedder checks; the rest (validation report) is ignored."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    model: str
    cfg: _Cfg
    input: _Input


class _Net(nn.Module):
    """Backbone CLS token -> MLP -> L2-normalised `dim` vector."""

    def __init__(self, dim: int, hidden: int = 512) -> None:
        super().__init__()
        self.backbone: nn.Module = timm.create_model(BACKBONE, pretrained=False, num_classes=0)
        width = int(self.backbone.num_features)  # pyright: ignore[reportArgumentType]
        self.proj = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: Tensor) -> Tensor:
        feats: Tensor = self.backbone.forward_features(x)  # pyright: ignore[reportCallIssue]
        return F.normalize(self.proj(feats[:, 0]).float(), dim=1)


class Embedder:
    """The loaded network. Build with `Embedder.load`."""

    def __init__(self, net: _Net, dim: int, device: torch.device, dtype: torch.dtype) -> None:
        self.net = net
        self.dim = dim
        self.device = device
        self.dtype = dtype
        self._mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self._std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    @classmethod
    def load(cls, path: Path, device: torch.device) -> Self:
        """Load the checkpoint (saved on cuda:0; mapped to CPU first); half precision on CUDA."""
        raw: object = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(raw, dict):
            raise TypeError(f"{path}: checkpoint is not a dict")
        ck: dict[str, object] = {str(k): v for k, v in raw.items()}
        meta = _Meta.model_validate(ck)
        arch = meta.model.removeprefix("timm/").split(".")[0]
        if arch != BACKBONE or meta.input.size != SIZE or meta.input.pad != PAD:
            raise ValueError(f"{path}: unexpected embedder {meta.model} {meta.input}")
        sd = ck["state_dict"]
        if not isinstance(sd, dict):
            raise TypeError(f"{path}: state_dict is not a dict")
        state = {str(k): v.float() for k, v in sd.items() if isinstance(v, Tensor)}
        net = _Net(meta.cfg.dim)
        net.load_state_dict(state)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        net = net.to(device=device, dtype=dtype).eval()
        return cls(net, meta.cfg.dim, device, dtype)

    @staticmethod
    def usable(box: Sequence[float]) -> bool:
        """Whether a box is big enough to embed (MIN_BOX_PX on both sides)."""
        return box[2] - box[0] >= MIN_BOX_PX and box[3] - box[1] >= MIN_BOX_PX

    @staticmethod
    def model_input(
        image: NDArray[np.uint8], box: Sequence[float], panel: PixelRect
    ) -> NDArray[np.uint8]:
        """The SIZE x SIZE RGB uint8 letterboxed crop of box `x1 y1 x2 y2` padded by PAD."""
        bw, bh = box[2] - box[0], box[3] - box[1]
        x0 = max(panel.x, math.floor(box[0] - PAD * bw))
        y0 = max(panel.y, math.floor(box[1] - PAD * bh))
        x1 = min(panel.x + panel.w, math.ceil(box[2] + PAD * bw))
        y1 = min(panel.y + panel.h, math.ceil(box[3] + PAD * bh))
        out = np.empty((SIZE, SIZE, 3), np.uint8)
        out[:] = _GREY
        cut = image[y0:y1, x0:x1]
        h, w = cut.shape[:2]
        if h < 1 or w < 1:
            return out
        s = max(h, w)
        nw, nh = max(1, round(w * SIZE / s)), max(1, round(h * SIZE / s))
        interp = cv2.INTER_CUBIC if s < SIZE else cv2.INTER_AREA
        small = cv2.resize(cut, (nw, nh), interpolation=interp)
        oy, ox = (SIZE - nh) // 2, (SIZE - nw) // 2
        out[oy : oy + nh, ox : ox + nw] = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        return out

    def embed(self, images: Sequence[NDArray[np.uint8]]) -> NDArray[np.float32]:
        """`[N, dim]` L2-normalised vectors for `model_input` images, one batch."""
        x = torch.from_numpy(np.stack(images)).to(self.device).permute(0, 3, 1, 2)
        x = (x.float().div_(255.0) - self._mean) / self._std
        with torch.inference_mode():
            v = self.net(x.to(self.dtype))
        return v.float().cpu().numpy()
