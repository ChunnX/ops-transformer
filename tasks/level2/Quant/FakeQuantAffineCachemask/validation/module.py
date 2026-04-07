"""
CPU golden：torch.fake_quantize_per_channel_affine（axis=0，与 tiling 中 headNum=dim0 一致）。
mask：round(x/scale)+zero_point（float 路径）落在 [quant_min, quant_max] 内（与核心里比较逻辑近似）。
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


def _mask_in_quant_range(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    axis: int,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    """按通道广播后，判断 round(x/s + zp) 是否在整型量化网格范围内。"""
    if axis != 0:
        raise ValueError("golden 仅对齐本算子 tiling：headNum = x.shape[0]，axis 须为 0")
    c = x.shape[0]
    sc = scale.reshape(c, *([1] * (x.ndim - 1))).float()
    zp = zero_point.reshape(c, *([1] * (x.ndim - 1))).float()
    t = torch.round(x.float() / sc + zp)
    return (t >= qmin) & (t <= qmax)


def golden(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    axis: int,
    quant_min: int,
    quant_max: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    zp = zero_point.to(torch.int32)
    y = torch.fake_quantize_per_channel_affine(x, scale.float(), zp, axis, quant_min, quant_max)
    m = _mask_in_quant_range(x, scale, zero_point, axis, quant_min, quant_max)
    return y, m


class Model(nn.Module):
    def __init__(self, axis: int, quant_min: int, quant_max: int):
        super().__init__()
        self.axis = int(axis)
        self.quant_min = int(quant_min)
        self.quant_max = int(quant_max)

    def forward(self, x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor):
        y, m = golden(x, scale, zero_point, self.axis, self.quant_min, self.quant_max)
        return [y, m]


class ModelNew(nn.Module):
    def __init__(self, axis: int, quant_min: int, quant_max: int):
        super().__init__()
        self.axis = int(axis)
        self.quant_min = int(quant_min)
        self.quant_max = int(quant_max)

    def forward(self, x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor):
        import kernel_gen_ops

        return kernel_gen_ops.fake_quant_affine_cachemask(
            x, scale, zero_point, self.axis, self.quant_min, self.quant_max
        )
