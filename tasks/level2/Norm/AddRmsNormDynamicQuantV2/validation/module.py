# -*- coding: utf-8 -*-
# 标杆与 ATK aclnn_add_rms_norm_dynamic_quant_v2 对齐：
# goldenAddRmsNorm: x=x1+x2, rstd=rsqrt(mean(x^2)+eps), y=x*rstd*gamma+beta
# goldenDynamicQuant: smooth_x = x or x*smooth; scale=127/max(|smooth_x|), gq=round(smooth_x/scale).int8

from typing import List, Optional

import torch
import torch.nn as nn


def golden_add_rms_norm(
    x1: torch.Tensor,
    x2: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
) -> tuple:
    """与 ATK goldenAddRmsNorm 一致: x=x1+x2, rstd=rsqrt(mean(x^2)+eps), y=x*rstd*gamma+beta, return (y, x)."""
    ori_dtype = x1.dtype
    if ori_dtype != torch.float32:
        x1 = x1.float()
        x2 = x2.float()
        gamma = gamma.float()
        beta = beta.float()
    x = x1 + x2
    rstd = torch.rsqrt(x.pow(2).mean(axis=-1, keepdim=True) + eps)
    y = x * rstd * gamma + beta
    if ori_dtype != torch.float32:
        return y, x.to(ori_dtype)
    return y, x


def golden_dynamic_quant(
    x: torch.Tensor,
    smooth: Optional[torch.Tensor],
) -> tuple:
    """与 ATK goldenDynamicQuant 一致: smooth_x = x 或 x*smooth; gs_rev=127/max(|smooth_x|), gs=1/gs_rev, gq=round(smooth_x*gs_rev).int8."""
    x = x.float() if x.dtype != torch.float32 else x
    if smooth is not None:
        smooth = smooth.float() if smooth.dtype != torch.float32 else smooth
    else:
        smooth = None
    smooth_x = x if smooth is None else x * smooth
    x_max = torch.max(torch.abs(smooth_x), dim=-1, keepdim=True)[0].clamp(min=1e-8)
    gs_rev = 127.0 / x_max
    gs = 1.0 / gs_rev
    sx = smooth_x * gs_rev
    gq = torch.round(sx).to(torch.int8)
    return gq, gs.squeeze(-1).float()


class Model(nn.Module):
    """Reference: add + RMSNorm(+beta) + dynamic quant，与 ATK FunctionRmsNormGradApi 标杆一致。"""

    def __init__(
        self,
        gamma: torch.Tensor,
        epsilon: float = 1e-6,
        beta: Optional[torch.Tensor] = None,
    ):
        super(Model, self).__init__()
        self.gamma = gamma.detach().to(torch.float32)
        self.epsilon = epsilon
        self.beta = beta.detach().to(torch.float32) if beta is not None else None

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        smooth1: Optional[torch.Tensor] = None,
        smooth2: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        device = x1.device
        dtype = x1.dtype
        gamma = self.gamma.to(device).to(dtype)
        beta = (
            self.beta.to(device).to(dtype)
            if self.beta is not None
            else torch.zeros(gamma.shape, device=device, dtype=dtype)
        )
        # 与 ATK 一致: gyFp32, gx = goldenAddRmsNorm(x1, x2, gamma, beta, epsilon)
        gy_fp32, gx = golden_add_rms_norm(x1, x2, gamma, beta, self.epsilon)
        smooth1_exist = smooth1 is not None
        smooth2_exist = smooth2 is not None

        if smooth1_exist and smooth2_exist:
            gq1, gs1 = golden_dynamic_quant(gy_fp32, smooth1)
            gq2, gs2 = golden_dynamic_quant(gy_fp32, smooth2)
        elif smooth1_exist and (not smooth2_exist):
            gq1, gs1 = golden_dynamic_quant(gy_fp32, smooth1)
            gq2 = torch.zeros_like(gq1, device=device, dtype=gq1.dtype)
            gs2 = torch.zeros_like(gs1, device=device, dtype=gs1.dtype)
        elif (not smooth1_exist) and (not smooth2_exist):
            gq1, gs1 = golden_dynamic_quant(gy_fp32, None)
            gq2 = torch.zeros_like(gq1, device=device, dtype=gq1.dtype)
            gs2 = torch.zeros_like(gs1, device=device, dtype=gs1.dtype)
        else:
            # only smooth2 exist
            gq1, gs1 = golden_dynamic_quant(gy_fp32, None)
            gq2, gs2 = golden_dynamic_quant(gy_fp32, smooth2)

        # 输出顺序与 kernel 一致: (y1, y2, y3, y4, x, scale1, scale2)
        y3 = gy_fp32
        y4 = gy_fp32.to(dtype) if dtype != torch.float32 else gy_fp32
        x_out = gx
        N = gq1.shape[0]
        scale1_out = gs1 if gs1.dim() >= 1 else gs1.unsqueeze(0).expand(N)
        scale2_out = gs2 if gs2.dim() >= 1 else gs2.unsqueeze(0).expand(N)
        return [gq1, gq2, y3, y4, x_out, scale1_out, scale2_out]


class ModelNew(nn.Module):
    """调用 kernel 的封装，与 pybind 接口一致。"""

    def __init__(self, gamma: torch.Tensor, epsilon: float = 1e-6):
        super(ModelNew, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        smooth1: Optional[torch.Tensor] = None,
        smooth2: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        import kernel_gen_ops

        return list(
            kernel_gen_ops.add_rms_norm_dynamic_quant_v2(
                x1, x2, self.gamma, smooth1, smooth2, self.epsilon
            )
        )
