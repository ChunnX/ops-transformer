"""
CPU golden 与 AICore 对称动态量化路径一致（见 answer/0/op_kernel/dynamic_quant.h::Compute）：
Cast(x)->fp32；可选乘 smooth；Abs 后沿最后一维求 max；scale_fp = 127/max_abs；
输出 scale 张量为 max_abs/127（与内核 scaleLocal.SetValue(i, 1 / scale) 一致）；
量化：tempFp32 * (127/max_abs) 后 CAST_RINT 再落到 int8。
Golden 用 float32 + numpy.rint 逼近 CAST_RINT。
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def golden_dynamic_quant_int8(
    x: torch.Tensor, smooth: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_f = x.detach().float()
    if smooth is not None:
        x_f = x_f * smooth.detach().float()
    last = x_f.shape[-1]
    flat = x_f.reshape(-1, last)
    max_abs = flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    inv = 127.0 / max_abs
    qf = flat * inv
    q_np = np.rint(qf.cpu().numpy())
    q = torch.from_numpy(q_np).to(device=x.device, dtype=torch.float32).clamp(-128, 127).to(torch.int8)
    y = q.reshape_as(x)
    scale = (max_abs / 127.0).squeeze(-1).reshape(x.shape[:-1]).to(torch.float32)
    return y, scale


class Model(nn.Module):
    def forward(self, x: torch.Tensor, smooth: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        y, scale = golden_dynamic_quant_int8(x, smooth)
        return [y, scale]


class ModelNew(nn.Module):
    def forward(self, x: torch.Tensor, smooth: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        import kernel_gen_ops

        return kernel_gen_ops.dynamic_quant(x, smooth)
