"""
CPU golden：与 trans_quant_param_v2.h 中 round_mode==0 路径一致：
pack_scale = (float32_bits(scale) & 0xFFFFE000) | (1<<46)；
offset 按 CAST_RINT 到 int32 后 clamp 到 [-256,255]，取低 9 bit 左移 37 后与 pack_scale 按位或。
"""

from __future__ import annotations

import struct
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

DEQ_SCALE_MUL = 0xFFFFE000
QUANT_SCALE = 1 << 46
QUANT_MASK_0 = 0x1FF
OFFSET_DEVIATION = 37


def _pack_scale_rm0(f: float) -> np.uint64:
    u32 = struct.unpack("I", struct.pack("f", np.float32(f)))[0]
    return np.uint64((u32 & DEQ_SCALE_MUL) | QUANT_SCALE)


def _offset_bits(f: float) -> np.uint64:
    v = int(np.rint(np.float32(f)))
    v = max(-256, min(255, v))
    return (np.uint64(v) & np.uint64(QUANT_MASK_0)) << np.uint64(OFFSET_DEVIATION)


def golden(scale: torch.Tensor, offset: Optional[torch.Tensor], round_mode: int) -> torch.Tensor:
    if int(round_mode) != 0:
        raise ValueError("validation 仅对齐 aclnnTransQuantParamV2（内部 roundMode 固定为 0）")
    s = scale.detach().cpu().numpy().astype(np.float32).reshape(-1)
    n = s.size
    out = np.zeros(n, dtype=np.uint64)
    if offset is None:
        for i in range(n):
            out[i] = _pack_scale_rm0(float(s[i]))
        return torch.from_numpy(out).to(device=scale.device)

    o = offset.detach().cpu().numpy().astype(np.float32).reshape(-1)
    m = o.size
    if m == 1 and n > 1:
        ob = _offset_bits(float(o[0]))
        for i in range(n):
            out[i] = _pack_scale_rm0(float(s[i])) | ob
    elif m == n:
        for i in range(n):
            out[i] = _pack_scale_rm0(float(s[i])) | _offset_bits(float(o[i]))
    else:
        raise ValueError(f"unsupported scale len {n} vs offset len {m}")
    return torch.from_numpy(out).to(device=scale.device)


class Model(nn.Module):
    def __init__(self, round_mode: int):
        super().__init__()
        self.round_mode = int(round_mode)

    def forward(self, scale: torch.Tensor, offset=None):
        return golden(scale, offset, self.round_mode)


class ModelNew(nn.Module):
    def __init__(self, round_mode: int):
        super().__init__()
        self.round_mode = int(round_mode)

    def forward(self, scale: torch.Tensor, offset=None):
        import kernel_gen_ops

        return kernel_gen_ops.trans_quant_param_v2(scale, offset)
