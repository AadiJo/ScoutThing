# PARSeq model code ported from https://github.com/baudm/parseq (strhub/models/parseq/model.py
# and modules.py) by way of frc-vision-scorer's web/students/parseq.py.
#
#   Scene Text Recognition Model Hub
#   Copyright 2022 Darwin Bautista
#   Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
#   except in compliance with the License. You may obtain a copy of the License at
#   https://www.apache.org/licenses/LICENSE-2.0
#   Unless required by applicable law or agreed to in writing, software distributed under the
#   License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
#
# Changes from upstream: inference only (no training loss, no weight init, no pretrained
# vocabulary slicing); a digits-only vocabulary; teacher-forced string scoring.
"""PARSeq (Bautista & Atienza, ECCV 2022) restricted to digit strings, inference only.

Build with `PARSeq(**config)` from the checkpoint, load its state dict, then
`memory = model.encode(images)` and either `model.score_strings(memory, tokens)` for
log P(given strings) or `model.greedy(memory)` for the free read.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F  # noqa: N812
from timm.layers.patch_embed import PatchEmbed  # pyright: ignore[reportMissingTypeStubs]
from timm.models.vision_transformer import (  # pyright: ignore[reportMissingTypeStubs]
    VisionTransformer,
)
from torch import Tensor, nn

CHARSET = "0123456789"
EOS, BOS, PAD = 0, len(CHARSET) + 1, len(CHARSET) + 2
"""Token ids: [E]=0, '0'..'9'=1..10, [B]=11, [P]=12 (upstream layout)."""


def encode_strings(labels: Sequence[str], length: int, device: torch.device) -> Tensor:
    """`[B] digits [E] [P]...` token rows of width `length + 2` for digit strings."""
    out = torch.full((len(labels), length + 2), PAD, dtype=torch.long)
    for i, s in enumerate(labels):
        ids = [BOS, *(1 + int(c) for c in s), EOS]
        out[i, : len(ids)] = torch.tensor(ids)
    return out.to(device)


def decode_ids(ids: Sequence[int]) -> str:
    """Digit string from predicted token ids, stopping at [E]."""
    chars: list[str] = []
    for i in ids:
        if i == EOS or i > len(CHARSET):
            break
        chars.append(CHARSET[i - 1])
    return "".join(chars)


class DecoderLayer(nn.Module):
    """Pre-LN transformer decoder layer with two-stream attention (XLNet)."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_c = nn.LayerNorm(d_model)

    def _stream(
        self, tgt: Tensor, tgt_norm: Tensor, kv: Tensor, memory: Tensor, mask: Tensor | None
    ) -> Tensor:
        tgt = tgt + self.self_attn(tgt_norm, kv, kv, attn_mask=mask, need_weights=False)[0]
        tgt = tgt + self.cross_attn(self.norm1(tgt), memory, memory, need_weights=False)[0]
        return tgt + self.linear2(F.gelu(self.linear1(self.norm2(tgt))))

    def forward(
        self,
        query: Tensor,
        content: Tensor,
        memory: Tensor,
        query_mask: Tensor | None,
        content_mask: Tensor | None,
        update_content: bool,
    ) -> tuple[Tensor, Tensor]:
        content_norm = self.norm_c(content)
        query = self._stream(query, self.norm_q(query), content_norm, memory, query_mask)
        if update_content:
            content = self._stream(content, content_norm, content_norm, memory, content_mask)
        return query, content


class Decoder(nn.Module):
    """Stack of decoder layers; the last one skips the content stream update."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, depth: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            DecoderLayer(d_model, nhead, dim_feedforward) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query: Tensor,
        content: Tensor,
        memory: Tensor,
        query_mask: Tensor | None,
        content_mask: Tensor | None,
    ) -> Tensor:
        for i, layer in enumerate(self.layers):
            last = i == len(self.layers) - 1
            query, content = layer(query, content, memory, query_mask, content_mask, not last)
        out: Tensor = self.norm(query)
        return out


class TokenEmbedding(nn.Module):
    """Token embedding scaled by sqrt(embed_dim)."""

    def __init__(self, charset_size: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(charset_size, embed_dim)
        self.scale = math.sqrt(embed_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        out: Tensor = self.embedding(tokens)
        return self.scale * out


class Encoder(VisionTransformer):
    """ViT over the 32x128 crop; `forward_features` gives all patch tokens (no class token)."""

    def __init__(
        self,
        img_size: tuple[int, int],
        patch_size: tuple[int, int],
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=True,
            embed_layer=PatchEmbed,
            num_classes=0,
            global_pool="",
            class_token=False,
        )


class PARSeq(nn.Module):
    """The digit reader network. Constructor arguments are the checkpoint's `config`."""

    def __init__(
        self,
        max_label_length: int,
        img_size: tuple[int, int],
        patch_size: tuple[int, int],
        embed_dim: int,
        enc_num_heads: int,
        enc_mlp_ratio: float,
        enc_depth: int,
        dec_num_heads: int,
        dec_mlp_ratio: int,
        dec_depth: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        del dropout  # training only
        self.max_label_length = max_label_length
        self.encoder = Encoder(
            img_size, patch_size, embed_dim, enc_depth, enc_num_heads, enc_mlp_ratio
        )
        self.decoder = Decoder(embed_dim, dec_num_heads, embed_dim * dec_mlp_ratio, dec_depth)
        self.head = nn.Linear(embed_dim, len(CHARSET) + 1)
        self.text_embed = TokenEmbedding(len(CHARSET) + 3, embed_dim)
        self.pos_queries = nn.Parameter(torch.zeros(1, max_label_length + 1, embed_dim))

    def encode(self, img: Tensor) -> Tensor:
        """Patch memory `[N, 128, D]` for a `[N, 3, 32, 128]` batch scaled to [-1, 1]."""
        out: Tensor = self.encoder.forward_features(img)
        return out

    def _decode(
        self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor, query: Tensor, query_mask: Tensor
    ) -> Tensor:
        length = tgt.shape[1]
        content = torch.cat(
            [
                self.text_embed(tgt[:, :1]),
                self.pos_queries[:, : length - 1] + self.text_embed(tgt[:, 1:]),
            ],
            dim=1,
        )
        return self.decoder(query, content, memory, query_mask, tgt_mask)

    def score_strings(self, memory: Tensor, tokens: Tensor) -> Tensor:
        """Teacher-forced log P(string, [E] | image) of K candidate strings for every image.

        `tokens` is `encode_strings(candidates, max_label_length)`, `[K, L + 2]`; returns
        `[N, K]` float32 log-likelihoods.
        """
        n, k = memory.shape[0], tokens.shape[0]
        tgt_in, tgt_out = tokens[:, :-1].repeat(n, 1), tokens[:, 1:].repeat(n, 1)
        length = tgt_in.shape[1]
        mask = torch.triu(torch.ones(length, length, dtype=torch.bool, device=memory.device), 1)
        mem = memory.repeat_interleave(k, dim=0)
        query = self.pos_queries[:, :length].expand(n * k, -1, -1)
        lp = self.head(self._decode(tgt_in, mem, mask, query, mask)).float().log_softmax(-1)
        valid = tgt_out != PAD
        tok_lp = lp.gather(2, tgt_out.clamp(max=lp.shape[-1] - 1)[..., None])[..., 0]
        return (tok_lp * valid).sum(1).view(n, k)

    def greedy(self, memory: Tensor) -> tuple[list[str], Tensor]:
        """Free autoregressive read: (digit string per image, its sequence log-prob `[N]`)."""
        bs, dev = memory.shape[0], memory.device
        steps = self.max_label_length + 1
        mask = torch.triu(torch.ones(steps, steps, dtype=torch.bool, device=dev), 1)
        tgt_in = torch.full((bs, steps), PAD, dtype=torch.long, device=dev)
        tgt_in[:, 0] = BOS
        lp_tot = torch.zeros(bs, device=dev)
        done = torch.zeros(bs, dtype=torch.bool, device=dev)
        pos = self.pos_queries[:, :steps].expand(bs, -1, -1)
        out_ids: list[Tensor] = []
        for i in range(steps):
            j = i + 1
            o = self._decode(tgt_in[:, :j], memory, mask[:j, :j], pos[:, i:j], mask[i:j, :j])
            lp = self.head(o)[:, 0].float().log_softmax(-1)
            if i == steps - 1:
                nxt = torch.full((bs,), EOS, dtype=torch.long, device=dev)
            else:
                nxt = lp.argmax(-1)
            step_lp = lp.gather(1, nxt[:, None])[:, 0]
            lp_tot = lp_tot + torch.where(done, torch.zeros_like(lp_tot), step_lp)
            done = done | (nxt == EOS)
            out_ids.append(nxt)
            if j < steps:
                tgt_in[:, j] = nxt
            if bool(done.all()):
                break
        rows: list[list[int]] = torch.stack(out_ids, 1).tolist()
        return [decode_ids(r) for r in rows], lp_tot
