# QuantizeAddLayerNorm: (x1+x2+bias) -> LayerNorm -> quantize -> y; 输出 x = round(x1+x2+bias)
from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

# kernel tiling: axis == -65535 为 mul_scale 模式 (y_norm * scale + zp)，否则为 div 模式 (y_norm / scale + zp)
AXIS_MUL_MODE = -65535


def _layer_norm_ref(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float, axis: int = -1) -> torch.Tensor:
    dim = axis if axis >= 0 else x.dim() + axis
    mean = x.mean(dim=dim, keepdim=True)
    var = x.var(dim=dim, keepdim=True, unbiased=False) + eps
    rstd = torch.rsqrt(var)
    return (x - mean) * rstd * gamma + beta


def _quantize_per_tensor(
    x: torch.Tensor,
    scale: torch.Tensor,
    zero_point: Optional[torch.Tensor],
    div_mode: bool,
) -> torch.Tensor:
    """与 kernel 一致：div_mode 时 y=round(x/scale+zp)，mul 时 y=round(x*scale+zp)。"""
    if div_mode:
        q = x / scale
    else:
        q = x * scale
    if zero_point is not None:
        q = q + zero_point.float()
    return torch.round(q).clamp(-128, 127).to(torch.int8)


class Model(nn.Module):
    def __init__(
        self,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        bias: torch.Tensor,
        scales: torch.Tensor,
        zero_points: Optional[torch.Tensor] = None,
        dtype_attr: int = 0,
        axis: int = -1,
        epsilon: float = 1e-5,
        additional_output: bool = False,
    ):
        super(Model, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.bias = bias
        self.scales = scales
        self.zero_points = zero_points
        self.dtype_attr = dtype_attr
        self.axis = axis
        self.epsilon = epsilon
        self.additional_output = additional_output

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> List[torch.Tensor]:
        dtype = x1.dtype
        x1_f = x1.float()
        x2_f = x2.float()
        x_sum = x1_f + x2_f + self.bias.float()
        x_norm = _layer_norm_ref(x_sum, self.gamma.float(), self.beta.float(), self.epsilon, self.axis)
        div_mode = self.axis != AXIS_MUL_MODE
        zp = self.zero_points.float() if self.zero_points is not None else None
        y = _quantize_per_tensor(x_norm, self.scales.float(), zp, div_mode)
        # kernel 始终输出 x = round(x1+x2+bias)
        x_out = torch.round(x_sum).to(dtype)
        return [y, x_out]


class ModelNew(nn.Module):
    def __init__(
        self,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        bias: torch.Tensor,
        scales: torch.Tensor,
        zero_points: Optional[torch.Tensor] = None,
        dtype_attr: int = 0,
        axis: int = -1,
        epsilon: float = 1e-5,
        additional_output: bool = False,
    ):
        super(ModelNew, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.bias = bias
        self.scales = scales
        self.zero_points = zero_points
        self.dtype_attr = dtype_attr
        self.axis = axis
        self.epsilon = epsilon
        self.additional_output = additional_output

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> List[torch.Tensor]:
        import kernel_gen_ops
        return list(kernel_gen_ops.quantize_add_layer_norm(
            x1, x2, self.gamma, self.beta, self.bias,
            self.scales, self.zero_points,
            self.dtype_attr, self.axis, self.epsilon, self.additional_output,
        ))
