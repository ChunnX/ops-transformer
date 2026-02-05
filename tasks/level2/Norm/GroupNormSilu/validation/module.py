# module.py
from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


def _group_norm_silu_ref(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                         num_groups: int, eps: float, activate_silu: bool) -> List[torch.Tensor]:
    """PyTorch reference: GroupNorm + optional SiLU (x * sigmoid(x)). Returns [y] for compare."""
    input_dtype = x.dtype
    N, C = x.shape[0], x.shape[1]
    remaining = x.shape[2:]
    HxW = 1
    for s in remaining:
        HxW *= s

    x_fp32 = x.to(torch.float32)
    gamma_fp32 = gamma.to(torch.float32)
    beta_fp32 = beta.to(torch.float32)

    gn_out, mean_out, rstd_out = torch.ops.aten.native_group_norm(
        x_fp32, gamma_fp32, beta_fp32, N, C, HxW, num_groups, eps
    )

    if activate_silu:
        final_out = gn_out * torch.sigmoid(gn_out)
    else:
        final_out = gn_out

    return [final_out.to(input_dtype)]


class Model(nn.Module):
    def __init__(self, num_channels: int, num_groups: int, eps: float, activate_silu: bool):
        super(Model, self).__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        self.activate_silu = activate_silu

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> List[torch.Tensor]:
        return _group_norm_silu_ref(x, gamma, beta, self.num_groups, self.eps, self.activate_silu)


class ModelNew(nn.Module):
    def __init__(self, num_channels: int, num_groups: int, eps: float, activate_silu: bool):
        super(ModelNew, self).__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        self.activate_silu = activate_silu

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> List[torch.Tensor]:
        y_out_npu, _, _ = kernel_gen_ops.group_norm_silu(
            x, gamma, beta, self.num_groups, self.eps, self.activate_silu
        )
        return [y_out_npu]
