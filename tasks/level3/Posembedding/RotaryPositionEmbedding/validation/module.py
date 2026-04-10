"""
CPU 金标准与 ops-transformer `posembedding/rotary_position_embedding/README.md` 中四种 mode 公式一致：
0 half，1 interleave，2 quarter（D 能被 4 整除），3 interleave-half（x_part1*cos + x_part2*sin）。
在 float32 上计算再 cast 回输入 dtype；cos/sin 广播到与 x 同形。
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def golden_rotary_position_embedding(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mode: int,
) -> torch.Tensor:
    od = x.dtype
    xf = x.float()
    cos_b = torch.broadcast_to(cos.float(), xf.shape)
    sin_b = torch.broadcast_to(sin.float(), xf.shape)
    d = xf.shape[-1]

    if mode == 0:
        half = d // 2
        x1 = xf[..., :half]
        x2 = xf[..., half:]
        x_rot = torch.cat((-x2, x1), dim=-1)
        y = xf * cos_b + x_rot * sin_b
    elif mode == 1:
        x1 = xf[..., ::2]
        x2 = xf[..., 1::2]
        # 与 README：cat((-x2,x1).view(-1,1), dim=-1).view(x.shape) 等价，最后一维为 ...,-x2[k],x1[k],...
        x_rot = torch.stack((-x2, x1), dim=-1).reshape(*xf.shape)
        y = xf * cos_b + x_rot * sin_b
    elif mode == 2:
        q = d // 4
        x1 = xf[..., :q]
        x2 = xf[..., q : 2 * q]
        x3 = xf[..., 2 * q : 3 * q]
        x4 = xf[..., 3 * q :]
        x_rot = torch.cat((-x2, x1, -x4, x3), dim=-1)
        y = xf * cos_b + x_rot * sin_b
    elif mode == 3:
        x1 = xf[..., ::2]
        x2 = xf[..., 1::2]
        x_part1 = torch.cat((x1, x2), dim=-1)
        x_part2 = torch.cat((-x2, x1), dim=-1)
        # 与 README：y = x_part1 * cos + x_part2 * sin（cos/sin 与 x broadcast 同形；自然序 D）
        y = x_part1 * cos_b + x_part2 * sin_b
    else:
        raise ValueError(f"unsupported mode {mode}")

    return y.to(od)


class Model(nn.Module):
    def __init__(self, mode: int = 0):
        super().__init__()
        self.mode = int(mode)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, _out: torch.Tensor) -> List[torch.Tensor]:
        del _out
        return [golden_rotary_position_embedding(x, cos, sin, self.mode)]


class ModelNew(nn.Module):
    def __init__(self, mode: int = 0):
        super().__init__()
        self.mode = int(mode)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, out: torch.Tensor) -> List[torch.Tensor]:
        import kernel_gen_ops

        kernel_gen_ops.rotary_position_embedding(x, cos, sin, int(self.mode), out)
        return [out]
