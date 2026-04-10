"""
CPU 金标准与 ascendc ApplyRotaryPosEmb（half rotary_mode，与 aclnn 默认一致）对齐：
对最后一维前半 sin 取负，将 x 后半与前半交换后与 sin' 相乘，再与 x*cos 相加（见 apply_rotary_pos_emb_small.h::ComputeTotary）。
"""

from typing import List, Tuple

import torch
import torch.nn as nn


def golden_apply_rotary_pos_emb_half(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """与内核 half 模式一致；在 float32 上计算再 cast 回输入 dtype。"""
    od = q.dtype
    qf = q.float()
    kf = k.float()
    cos_b = torch.broadcast_to(cos.float(), qf.shape)
    sin_b = torch.broadcast_to(sin.float(), qf.shape)
    d = qf.shape[-1]
    half = d // 2
    sin_neg = sin_b.clone()
    sin_neg[..., :half] = -sin_neg[..., :half]

    def one(x: torch.Tensor) -> torch.Tensor:
        x_rot = torch.cat([x[..., half:], x[..., :half]], dim=-1)
        return x * cos_b + x_rot * sin_neg

    qo = one(qf).to(od)
    ko = one(kf).to(od)
    return qo, ko


class Model(nn.Module):
    def __init__(self, layout: int = 1):
        super().__init__()
        self.layout = layout

    def forward(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> List[torch.Tensor]:
        qo, ko = golden_apply_rotary_pos_emb_half(q, k, cos, sin)
        return [qo, ko]


class ModelNew(nn.Module):
    def __init__(self, layout: int = 1):
        super().__init__()
        self.layout = layout

    def forward(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> List[torch.Tensor]:
        import kernel_gen_ops

        return kernel_gen_ops.apply_rotary_pos_emb(q, k, cos, sin, int(self.layout))
