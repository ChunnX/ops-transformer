from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """与 ATK npu_add_rms_norm_quant_golden 一致: factor = 1/sqrt(mean(x^2)+eps), return x * factor * weight + bias."""
    square_sum = torch.sum(torch.square(x), dim=-1, keepdim=True)
    factor = 1.0 / torch.sqrt(square_sum / x.shape[-1] + eps)
    return x * factor * weight + bias


class Model(nn.Module):
    """
    标杆与 ATK function_rms_norm_quant 完全一致:
    quant_in = rms_norm(x, gamma, beta, eps=epsilon);
    y = quantize_per_tensor(quant_in, scale, offset, qint8).int_repr()，scale/offset 取标量。
    误差来源说明：当某行 mean(x^2) 很小时，rms≈sqrt(epsilon)。若 kernel 侧 epsilon 与标杆不一致
    （如未注入 attr 时用默认 1e-12），kernel 的 factor 会远大于标杆，导致该行 quant_in 被放大，
    出现 72/127 等异常值，从而产生 max_abs_error=130 量级的误差。
    """

    def __init__(self, gamma: torch.Tensor, beta: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor, epsilon: float):
        super(Model, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.scale = scale
        self.offset = offset
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        input_x = x.float()
        input_gamma = self.gamma.float()
        input_beta = self.beta.float()
        # scale/offset 标量，与 ATK golden 及 kernel GetScaleAndOffset(0) 一致
        input_scale = self.scale.float().flatten()[0].item()
        input_offset = self.offset.flatten()[0].item()

        quant_in = _rms_norm(input_x, weight=input_gamma, bias=input_beta, eps=self.epsilon)
        # 与 ATK 一致：quantize_per_tensor -> int_repr -> int8 reshape
        output_q = torch.quantize_per_tensor(quant_in, input_scale, input_offset, torch.qint8)
        y_np = output_q.int_repr().detach().clone().cpu()
        out_int8 = y_np.to(torch.int8).reshape(x.shape)
        return [out_int8]


class ModelNew(nn.Module):
    def __init__(self, gamma: torch.Tensor, beta: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor, epsilon: float):
        super(ModelNew, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.scale = scale
        self.offset = offset
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return kernel_gen_ops.rms_norm_quant(x, self.gamma, self.beta, self.scale, self.offset, self.epsilon)
