from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn


def exp_segsum_reference(x: torch.Tensor) -> torch.Tensor:
    """
    ExpSegsum 参考实现：
    输入 x (..., N)，输出 (..., N, N)。
    下三角（不含对角）为 cumsum 后 exp，对角为 1，上三角为 0（-inf 再 exp）。
    """
    # 支持 3D (batch, ..., N) 或 4D
    shape = list(x.shape)
    n = shape[-1]
    prefix = shape[:-1]
    # 构造 (..., N, N)：位置 (i,j) 取 x[..., i] 当 i>j，否则 0
    x_expand = x.unsqueeze(-1)  # (..., N, 1)
    x_broadcast = x_expand.expand(prefix + [n, n])  # (..., N, N)
    # A: 上三角+对角为 True -> 填 0；即保留下三角（i>j）
    mask_keep = torch.tril(torch.ones(n, n, device=x.device, dtype=torch.bool), -1)  # 严格下三角
    mask_keep = mask_keep.reshape([1] * len(prefix) + [n, n]).expand(prefix + [n, n])
    mat = torch.where(mask_keep, x_broadcast, torch.zeros_like(x_broadcast, device=x.device))
    # 沿倒数第二维 cumsum
    cum = mat.cumsum(dim=-2)
    # B: 上三角（不含对角）为 True -> 填 -inf
    mask_upper = torch.triu(torch.ones(n, n, device=x.device, dtype=torch.bool), 1)
    mask_upper = mask_upper.reshape([1] * len(prefix) + [n, n]).expand(prefix + [n, n])
    cum = torch.where(mask_upper, torch.full_like(cum, float('-inf')), cum)
    # 对角保持 cum 的值：对 i=j 为 cum[..., i, i] = sum_{k<i} mat[..., k, i] = sum_{k<i} (x[...,k] if k<i else 0) = sum_{k=0}^{i-1} x[...,k]
    # 文档说对角线为 0 的矩阵，cumsum 后对角就是 0，再 exp(0)=1
    out = torch.exp(cum)
    return out.to(x.dtype)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return exp_segsum_reference(x)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.segsum(x)
