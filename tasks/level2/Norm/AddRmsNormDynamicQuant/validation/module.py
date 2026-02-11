from typing import List, Optional

import torch
import torch.nn as nn


def _dynamic_quant_per_row(y: torch.Tensor, smooth: Optional[torch.Tensor] = None, dim: int = -1) -> tuple:
    """Per-row dynamic quant. If smooth is not None: quantize y*smooth (smooth broadcast to y); else quantize y.
    scale = max(abs(smooth_x))/127, q = round(smooth_x/scale). Aligned with ATK goldenDynamicQuant."""
    y_flat = y.float()
    if smooth is not None:
        smooth_f = smooth.float().to(y_flat.device)
        smooth_x = y_flat * smooth_f  # broadcast [D] to [N, D]
    else:
        smooth_x = y_flat
    scale = smooth_x.abs().amax(dim=dim, keepdim=True).clamp(min=1e-8) / 127.0
    q = (smooth_x / scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale.squeeze(dim).float()


def _golden_add_rms_norm(x1: torch.Tensor, x2: torch.Tensor, gamma: torch.Tensor, eps: float):
    """x = x1+x2, rstd = rsqrt(mean(x^2,-1)+eps), y = x * rstd * gamma. Aligned with ATK goldenAddRmsNorm (no beta)."""
    ori_dtype = x1.dtype
    x1_f = x1.float()
    x2_f = x2.float()
    gamma_f = gamma.float().to(x1_f.device)
    x = x1_f + x2_f
    rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    y = x * rstd * gamma_f
    x_out = x if ori_dtype == torch.float32 else x.to(ori_dtype)
    return y, x_out


class Model(nn.Module):
    """Reference: add + RMSNorm + dynamic quant. Supports optional smooth1/smooth2. Outputs: y1, y2, x, scale1, scale2.
    Aligned with ATK goldenAddRmsNorm + goldenDynamicQuant."""

    def __init__(
        self,
        gamma: torch.Tensor,
        epsilon: float = 1e-6,
        smooth_scale1: Optional[torch.Tensor] = None,
        smooth_scale2: Optional[torch.Tensor] = None,
    ):
        super(Model, self).__init__()
        self.gamma = gamma.detach().to(torch.float32)
        self.epsilon = epsilon
        self.smooth_scale1 = smooth_scale1
        self.smooth_scale2 = smooth_scale2

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> List[torch.Tensor]:
        gamma = self.gamma.to(x1.device).to(x1.dtype)
        y, x_out = _golden_add_rms_norm(x1, x2, gamma, self.epsilon)
        smooth1 = self.smooth_scale1.to(y.device).to(y.dtype) if self.smooth_scale1 is not None else None
        smooth2 = self.smooth_scale2.to(y.device).to(y.dtype) if self.smooth_scale2 is not None else None

        if smooth1 is not None and smooth2 is not None:
            y1, scale1 = _dynamic_quant_per_row(y, smooth1, dim=-1)
            y2, scale2 = _dynamic_quant_per_row(y, smooth2, dim=-1)
        elif smooth1 is not None:
            y1, scale1 = _dynamic_quant_per_row(y, smooth1, dim=-1)
            y2 = torch.zeros_like(y1, device=y.device, dtype=torch.int8)
            scale2 = torch.zeros_like(scale1, device=y.device, dtype=torch.float32)
        else:
            y1, scale1 = _dynamic_quant_per_row(y, None, dim=-1)
            y2 = torch.zeros_like(y1, device=y.device, dtype=torch.int8)
            # Kernel outputs scale2=0 when no smooth; align ref for precision check.
            scale2 = torch.zeros_like(scale1, device=y.device, dtype=torch.float32)

        scale1_out = scale1 if scale1.dim() >= 1 else scale1.unsqueeze(0)
        return [y1, y2, x_out, scale1_out, scale2]


class ModelNew(nn.Module):
    def __init__(
        self,
        gamma: torch.Tensor,
        epsilon: float = 1e-6,
        smooth_scale1: Optional[torch.Tensor] = None,
        smooth_scale2: Optional[torch.Tensor] = None,
    ):
        super(ModelNew, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon
        self.smooth_scale1 = smooth_scale1
        self.smooth_scale2 = smooth_scale2

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> List[torch.Tensor]:
        import kernel_gen_ops
        return list(kernel_gen_ops.add_rms_norm_dynamic_quant(
            x1, x2, self.gamma, self.smooth_scale1, self.smooth_scale2, self.epsilon
        ))
