"""
CPU 金标准与 `posembedding/rotary_position_embedding_grad/README.md` / aclnn 文档中 half、interleave 的反向公式一致。
当前 Bench 目标核（如 910B）与 RotaryPositionEmbedding 一致，仅覆盖 mode 0、1；dcos/dsin 对 broadcast 轴求和。
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


def _sum_grad_to_broadcast_input(grad_full: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """将 [B,N,S,D] 上的梯度规约到 ref（如 cos/sin）的 shape。"""
    g = grad_full
    for d in range(grad_full.dim()):
        if ref.shape[d] == 1 and g.shape[d] != 1:
            g = g.sum(dim=d, keepdim=True)
    return g


def golden_rotary_position_embedding_grad(
    dy: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    x: torch.Tensor,
    mode: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dy_f = dy.float()
    cos_b = torch.broadcast_to(cos.float(), dy.shape)
    sin_b = torch.broadcast_to(sin.float(), dy.shape)
    xf = torch.broadcast_to(x.float(), dy.shape)
    d = dy.shape[-1]
    if mode == 0:
        half = d // 2
        dy1, dy2 = dy_f[..., :half], dy_f[..., half:]
        cos1, cos2 = cos_b[..., :half], cos_b[..., half:]
        sin1, sin2 = sin_b[..., :half], sin_b[..., half:]
        x1, x2 = xf[..., :half], xf[..., half:]
        dx = torch.cat((cos1 * dy1 + sin2 * dy2, cos2 * dy2 - sin1 * dy1), dim=-1)
        g_dcos = dy_f * xf
        g_dsin = dy_f * torch.cat((-x2, x1), dim=-1)
    elif mode == 1:
        dy1, dy2 = dy_f[..., ::2], dy_f[..., 1::2]
        cos1, cos2 = cos_b[..., ::2], cos_b[..., 1::2]
        sin1, sin2 = sin_b[..., ::2], sin_b[..., 1::2]
        x1, x2 = xf[..., ::2], xf[..., 1::2]
        dx = torch.stack((cos1 * dy1 + sin2 * dy2, cos2 * dy2 - sin1 * dy1), dim=-1).reshape(dy.shape)
        g_dcos = dy_f * xf
        rot_x = torch.stack((-x2, x1), dim=-1).reshape(xf.shape)
        g_dsin = dy_f * rot_x
    else:
        raise ValueError(f"unsupported mode {mode} for this validation")

    dcos = _sum_grad_to_broadcast_input(g_dcos, cos.float())
    dsin = _sum_grad_to_broadcast_input(g_dsin, sin.float())
    return (
        dx.to(dy.dtype),
        dcos.to(cos.dtype),
        dsin.to(sin.dtype),
    )


class Model(nn.Module):
    def __init__(self, mode: int = 0):
        super().__init__()
        self.mode = int(mode)

    def forward(
        self,
        dy: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        x: torch.Tensor,
        _dx: torch.Tensor,
        _dcos: torch.Tensor,
        _dsin: torch.Tensor,
    ) -> List[torch.Tensor]:
        del _dx, _dcos, _dsin
        dx, dcos, dsin = golden_rotary_position_embedding_grad(dy, cos, sin, x, self.mode)
        return [dx, dcos, dsin]


class ModelNew(nn.Module):
    def __init__(self, mode: int = 0):
        super().__init__()
        self.mode = int(mode)

    def forward(
        self,
        dy: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        x: torch.Tensor,
        dx: torch.Tensor,
        dcos: torch.Tensor,
        dsin: torch.Tensor,
    ) -> List[torch.Tensor]:
        import kernel_gen_ops

        kernel_gen_ops.rotary_position_embedding_grad(dy, cos, sin, x, int(self.mode), dx, dcos, dsin)
        return [dx, dcos, dsin]
