"""
CPU 金标准对齐 kernel group_quant_base.h::VecCompute：
x*scale(+offset) 在 fp32 上先 CAST_RINT→int32，再经 fp16 回读，最后 CAST_RINT→int8；
与 ops-nn-dev executor_aclnnGroupQuant 的分组边界语义一致。
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

DTYPE_INT8 = 2
DTYPE_INT4 = 29


def _ascend_int8_from_float(y_fp32: np.ndarray) -> np.ndarray:
    i32 = np.rint(y_fp32).astype(np.int32)
    h = i32.astype(np.float16).astype(np.float32)
    out = np.rint(h)
    return np.clip(out, -128, 127).astype(np.int8)


def _ascend_int4_from_float(y_fp32: np.ndarray) -> np.ndarray:
    i32 = np.rint(y_fp32).astype(np.int32)
    h = i32.astype(np.float16).astype(np.float32)
    out = np.rint(h)
    return np.clip(out, -8, 7).astype(np.int8)


def golden_group_quant(
    x: torch.Tensor,
    scale: torch.Tensor,
    group_index: torch.Tensor,
    offset: Optional[torch.Tensor],
    dst_type: int,
) -> torch.Tensor:
    x_np = x.detach().float().cpu().numpy()
    scale_np = scale.detach().float().cpu().numpy()
    gi = group_index.detach().cpu().numpy().astype(np.int64)
    dim_s, dim_h = x_np.shape
    dim_e, dim_h2 = scale_np.shape
    if dim_h != dim_h2 or gi.shape[0] != dim_e:
        raise ValueError("shape mismatch x/scale/group_index")
    if int(gi[-1]) != dim_s:
        raise ValueError("group_index[-1] must equal S")

    off = 0.0
    if offset is not None:
        off = float(offset.detach().float().cpu().reshape(-1)[0])

    parts = []
    for row_scale in range(dim_e):
        r0 = 0 if row_scale == 0 else int(gi[row_scale - 1])
        r1 = int(gi[row_scale])
        if r0 < r1:
            blk = x_np[r0:r1] * scale_np[row_scale : row_scale + 1]
            blk = blk + off
            parts.append(blk)
    y_fp32 = np.concatenate(parts, axis=0)

    if int(dst_type) == DTYPE_INT8:
        y_np = _ascend_int8_from_float(y_fp32)
    elif int(dst_type) == DTYPE_INT4:
        y_np = _ascend_int4_from_float(y_fp32)
    else:
        raise ValueError("dst_type must be 2 (int8) or 29 (int4)")

    return torch.from_numpy(y_np).to(device=x.device)


class Model(nn.Module):
    def __init__(self, dst_type: int):
        super().__init__()
        self.dst_type = int(dst_type)

    def forward(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        group_index: torch.Tensor,
        offset: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        y = golden_group_quant(x, scale, group_index, offset, self.dst_type)
        return [y]


class ModelNew(nn.Module):
    def __init__(self, dst_type: int):
        super().__init__()
        self.dst_type = int(dst_type)

    def forward(
        self,
        x: torch.Tensor,
        scale: torch.Tensor,
        group_index: torch.Tensor,
        offset: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        import kernel_gen_ops

        return kernel_gen_ops.group_quant(x, scale, group_index, offset, self.dst_type)
