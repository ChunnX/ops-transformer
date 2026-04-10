"""
CPU 金标准与 ops-transformer `interleave_rope/README.md` 公式一致：
先对最后一维做 interleave（view 为 [..., D/2, 2] 再交换最后两维），再
y = q * cos + RotateHalf(q) * sin，其中 RotateHalf 为 concat([-q[..., D/2:], q[..., :D/2]], dim=-1)。
在 float32 上计算后 cast 回输入 dtype；cos/sin 按广播规则扩展到 q 的形状。
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


def _interleave_last_dim(x: torch.Tensor) -> torch.Tensor:
    *rest, d = x.shape
    assert d % 2 == 0
    return (
        x.reshape(*rest, d // 2, 2)
        .transpose(-1, -2)
        .contiguous()
        .reshape(*rest, d)
    )


def _rotate_half(q: torch.Tensor) -> torch.Tensor:
    d = q.shape[-1]
    half = d // 2
    return torch.cat([-q[..., half:], q[..., :half]], dim=-1)


def golden_interleave_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    od = x.dtype
    q = _interleave_last_dim(x.float())
    cos_b = torch.broadcast_to(cos.float(), q.shape)
    sin_b = torch.broadcast_to(sin.float(), q.shape)
    y = q * cos_b + _rotate_half(q) * sin_b
    return y.to(od)


class Model(nn.Module):
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, _out: torch.Tensor) -> List[torch.Tensor]:
        del _out
        return [golden_interleave_rope(x, cos, sin)]


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, out: torch.Tensor) -> List[torch.Tensor]:
        import kernel_gen_ops

        kernel_gen_ops.interleave_rope(x, cos, sin, out)
        return [out]
