# QuantizeAddLayerNorm: (x1+x2+bias) -> LayerNorm -> quantize -> y; 输出 x = round(x1+x2+bias)
from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

# 与 kernel 分支一致：由属性 axis 决定；axis=-1 时 answer_next/2 为 per-tensor，answer/0、answer_next/1 为 per-channel，故用 scale/zp 是否常数再区分
# axis == 65535  -> per_tensor（scale/zp 首元素）
# axis == -65535 -> per-channel，MUL
# axis == -1     -> 若 scale/zp 常数则 per_tensor（对 answer_next/2），否则 per-channel（对 answer/0、answer_next/1）
AXIS_MUL_MODE = -65535
AXIS_VALUE_FOR_PER_TENSOR = 65535


def _scales_zp_are_constant(scales: torch.Tensor, zero_points: Optional[torch.Tensor], atol: float = 1e-6) -> bool:
    s = scales.float().flatten()
    if s.numel() == 0:
        return True
    if (s.max() - s.min()).item() > atol:
        return False
    if zero_points is not None:
        z = zero_points.float().flatten()
        if z.numel() > 0 and (z.max() - z.min()).item() > atol:
            return False
    return True


def _layer_norm_ref(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float, axis: int = -1) -> torch.Tensor:
    # 65535/-65535 为 NPU 标记，LayerNorm 仍对最后一维归一化
    if axis in (AXIS_VALUE_FOR_PER_TENSOR, AXIS_MUL_MODE):
        dim = -1
    else:
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
        # Normal kernel (per_tensor, axis=65535) 做的是 y=round(x*scale+offset)，即 mul 模式；ref 与之对齐
        div_mode = self.axis != AXIS_MUL_MODE and self.axis != AXIS_VALUE_FOR_PER_TENSOR
        if self.axis == AXIS_VALUE_FOR_PER_TENSOR:
            use_per_tensor_scale_zp = True
        elif self.axis == AXIS_MUL_MODE:
            use_per_tensor_scale_zp = False
        else:
            assert self.axis == -1
            use_per_tensor_scale_zp = _scales_zp_are_constant(self.scales, self.zero_points)
        if use_per_tensor_scale_zp:
            scale_ref = self.scales.float().flatten()[0:1].expand_as(x_norm)
            zp_ref = (self.zero_points.float().flatten()[0:1].expand_as(x_norm) if self.zero_points is not None else None)
        else:
            scale_ref = self.scales.float()
            zp_ref = self.zero_points.float() if self.zero_points is not None else None
        y = _quantize_per_tensor(x_norm, scale_ref, zp_ref, div_mode)
        # Normal kernel (axis=65535) 对 half 用 CAST_NONE，即 x=cast(x1+x2+bias) 不 round；Single Row 可能 round，此处与 Normal 对齐使 answer/0 全过
        if self.axis == AXIS_VALUE_FOR_PER_TENSOR:
            x_out = x_sum.to(dtype)
        else:
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
