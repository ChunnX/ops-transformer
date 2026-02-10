# module.py
from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


def _layer_norm_grad_ref_pytorch(
    dy: torch.Tensor,
    x: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    normalized_shape: List[int],
) -> List[torch.Tensor]:
    """标杆：直接用 PyTorch 官方 native_layer_norm_backward，公式与舍入与 PyTorch 完全一致."""
    output_mask = (True, True, True)
    dx, dgamma, dbeta = torch.ops.aten.native_layer_norm_backward(
        dy, x, normalized_shape, mean, rstd, weight, bias, output_mask
    )
    dtype_out = weight.dtype if weight is not None else x.dtype
    dev = x.device
    return [
        dx,
        dgamma if dgamma is not None else torch.zeros(normalized_shape, dtype=dtype_out, device=dev),
        dbeta if dbeta is not None else torch.zeros(normalized_shape, dtype=dtype_out, device=dev),
    ]


def _layer_norm_grad_ref_fallback(
    dy: torch.Tensor,
    x: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    weight: Optional[torch.Tensor],
    normalized_shape: List[int],
) -> List[torch.Tensor]:
    """无 aten 时的回退：手写公式，与 aclnn 对齐——dgamma/dbeta 在 float32 下 sum 再 cast 到输出 dtype."""
    input_dim = x.dim()
    normalized_dim = len(normalized_shape)
    reduction_dims = tuple(range(input_dim - normalized_dim, input_dim))
    N = 1
    for i in reduction_dims:
        N *= x.shape[i]

    dtype_orig = x.dtype
    compute_dtype = dtype_orig if dtype_orig in (torch.float16, torch.bfloat16) else torch.float32
    dy_c = dy.to(compute_dtype)
    x_c = x.to(compute_dtype)
    mean_c = mean.to(compute_dtype)
    rstd_c = rstd.to(compute_dtype)
    weight_c = weight.to(compute_dtype) if weight is not None else None

    x_norm = (x_c - mean_c) * rstd_c
    dy_weighted = (dy_c * weight_c) if weight_c is not None else dy_c

    N_t = torch.tensor(N, dtype=compute_dtype, device=x.device)
    sum1 = dy_weighted.sum(dim=reduction_dims, keepdim=True).to(compute_dtype)
    sum2 = (dy_weighted * x_norm).sum(dim=reduction_dims, keepdim=True).to(compute_dtype)
    c1 = sum1 / N_t
    c2 = sum2 / N_t
    dx = (dy_weighted - c1 - x_norm * c2) * rstd_c
    dx = dx.to(dtype_orig)

    batch_dims = tuple(range(0, input_dim - normalized_dim))
    # 与 kernel 一致：dgamma/dbeta 在 float32 下累加再 cast 到输出 dtype
    dgamma = (dy_weighted.float() * x_norm.float()).sum(dim=batch_dims)
    dbeta = dy_c.float().sum(dim=batch_dims)
    dtype_out = weight.dtype if weight is not None else dtype_orig
    dgamma = dgamma.to(dtype_out)
    dbeta = dbeta.to(dtype_out)
    return [dx, dgamma, dbeta]


def _layer_norm_grad_ref(
    dy: torch.Tensor,
    x: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    weight: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    normalized_shape: List[int],
) -> List[torch.Tensor]:
    """优先用 PyTorch 官方 backward，否则回退到手写公式."""
    try:
        return _layer_norm_grad_ref_pytorch(dy, x, mean, rstd, weight, bias, normalized_shape)
    except Exception:
        return _layer_norm_grad_ref_fallback(dy, x, mean, rstd, weight, normalized_shape)


class Model(nn.Module):
    def __init__(self, normalized_shape: List[int]):
        super(Model, self).__init__()
        self.normalized_shape = normalized_shape

    def forward(
        self,
        dy: torch.Tensor,
        x: torch.Tensor,
        mean: torch.Tensor,
        rstd: torch.Tensor,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
    ) -> List[torch.Tensor]:
        return _layer_norm_grad_ref(dy, x, mean, rstd, weight, bias, self.normalized_shape)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: List[int]):
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape

    def forward(
        self,
        dy: torch.Tensor,
        x: torch.Tensor,
        mean: torch.Tensor,
        rstd: torch.Tensor,
        weight: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
    ) -> List[torch.Tensor]:
        return kernel_gen_ops.layer_norm_grad_v3(
            dy, x, tuple(self.normalized_shape), mean, rstd, weight, bias
        )
