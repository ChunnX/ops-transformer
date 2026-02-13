from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn


def _calc_output(output_y: torch.Tensor) -> tuple:
    """Golden formula: out_scales = max(|y|, dim=-1, keepdim)/127, y_int8 = round(y/out_scales)."""
    max_y_0 = torch.max(torch.abs(output_y), dim=-1, keepdim=True)[0]
    out_scales = max_y_0 / 127.0
    x_quant = output_y / (out_scales + 1e-12)
    y_int8 = torch.round(x_quant).clamp(-128, 127).to(torch.int8)
    return y_int8, out_scales.squeeze(-1)


class Model(nn.Module):
    """Reference aligned with TBE golden: add -> layernorm -> quant(y*scale) with out_scale = max(|z|)/127."""

    def __init__(
        self,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        scales1: Optional[torch.Tensor] = None,
        scales2: Optional[torch.Tensor] = None,
        zero_points1: Optional[torch.Tensor] = None,
        zero_points2: Optional[torch.Tensor] = None,
        epsilon: float = 1e-5,
        additional_output: bool = True,
        div_mode: bool = True,
    ):
        super(Model, self).__init__()
        self.gamma = gamma.to(torch.float32).to('cpu')
        self.beta = beta.to(torch.float32).to('cpu')
        self.bias = bias.to(torch.float32).to('cpu') if bias is not None else None
        self.scales1 = scales1
        self.scales2 = scales2
        self.zero_points1 = zero_points1
        self.zero_points2 = zero_points2
        self.epsilon = epsilon
        self.additional_output = additional_output
        self.div_mode = div_mode

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> List[torch.Tensor]:
        dtype_hp = torch.float32
        x1 = x1.to(dtype_hp)
        x2 = x2.to(dtype_hp)
        gamma = self.gamma.to(x1.device).to(dtype_hp)
        beta = self.beta.to(x1.device).to(dtype_hp)

        # x = x1 + x2 + bias (same as golden)
        x = x1 + x2
        if self.bias is not None:
            x = x + self.bias.to(x.device).to(dtype_hp)

        # LayerNorm: mean, var, rstd, y = (x-mean)*rstd*gamma + beta
        mean = x.mean(dim=-1, keepdim=True)
        var = torch.mean(torch.pow(x - mean, 2), dim=-1, keepdim=True)
        rstd = 1.0 / torch.sqrt(var + self.epsilon)
        y = (x - mean) * rstd * gamma + beta

        # Quant: golden uses float32 for calc_output
        y = y.to(torch.float32)
        if self.scales1 is not None and self.scales2 is not None:
            s1 = self.scales1.to(y.device).to(torch.float32)
            s2 = self.scales2.to(y.device).to(torch.float32)
            output_y1 = y * s1
            output_y2 = y * s2
            y1_int, out_scale1 = _calc_output(output_y1)
            y2_int, out_scale2 = _calc_output(output_y2)
        else:
            # dynamic (non-dual): kernel only fills y1; y2 is dummy (shape {1} in L0), effectively zeros on framework side
            y1_int, out_scale1 = _calc_output(y)
            y2_int = torch.zeros_like(y, dtype=torch.int8)
            out_scale2 = out_scale1

        # Third output: x (pre-norm sum), same dtype as input
        x_out = x.to(x1.dtype) if x1.dtype != torch.float32 else x
        return [y1_int, y2_int, x_out, out_scale1, out_scale2]


class ModelNew(nn.Module):
    def __init__(
        self,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        scales1: Optional[torch.Tensor] = None,
        scales2: Optional[torch.Tensor] = None,
        zero_points1: Optional[torch.Tensor] = None,
        zero_points2: Optional[torch.Tensor] = None,
        epsilon: float = 1e-5,
        additional_output: bool = True,
        div_mode: bool = True,
    ):
        super(ModelNew, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.bias = bias
        self.scales1 = scales1
        self.scales2 = scales2
        self.zero_points1 = zero_points1
        self.zero_points2 = zero_points2
        self.epsilon = epsilon
        self.additional_output = additional_output
        self.div_mode = div_mode

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> List[torch.Tensor]:
        import kernel_gen_ops
        quant_mode = "dynamic" if (self.scales1 is None or self.scales1.numel() == 0) else "static"
        return kernel_gen_ops.add_layer_norm_quant(
            x1, x2, self.gamma, self.beta,
            self.bias, self.scales1, self.scales2, self.zero_points1, self.zero_points2,
            quant_mode, self.epsilon, self.additional_output, self.div_mode)
