from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn


def _ref_transform_bias_rescale_qkv(qkv: torch.Tensor, qkv_bias: torch.Tensor, num_heads: int) -> List[torch.Tensor]:
    """Reference implementation: split qkv to q,k,v, add bias, rescale q by 1/sqrt(dim_per_head)."""
    batch, token, triple_dim = qkv.shape
    dim = triple_dim // 3
    dim_per_head = dim // num_heads
    scale = dim_per_head ** (-0.5)
    qkv_flat = (qkv + qkv_bias).float()
    qkv_flat = qkv_flat.view(batch, token, 3, num_heads, dim_per_head)
    q = qkv_flat[:, :, 0, :, :].permute(0, 2, 1, 3)   # [batch, num_heads, token, dim_per_head]
    k = qkv_flat[:, :, 1, :, :].permute(0, 2, 1, 3)
    v = qkv_flat[:, :, 2, :, :].permute(0, 2, 1, 3)
    q = (q * scale).to(qkv.dtype)
    # 返回 list 以便框架 check_precision 按多输出逐项比较，无需修改框架
    return [q, k, v]


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, qkv: torch.Tensor, qkv_bias: torch.Tensor, num_heads: int) -> List[torch.Tensor]:
        return _ref_transform_bias_rescale_qkv(qkv, qkv_bias, num_heads)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, qkv: torch.Tensor, qkv_bias: torch.Tensor, num_heads: int) -> List[torch.Tensor]:
        import kernel_gen_ops
        # pybind 返回 tuple，转为 list 与 ref 一致，便于框架按 list 多输出比较
        return list(kernel_gen_ops.transform_bias_rescale_qkv(qkv, qkv_bias, num_heads))
