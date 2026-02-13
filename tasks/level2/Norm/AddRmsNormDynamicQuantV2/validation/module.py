from typing import List, Optional

import torch
import torch.nn as nn


def _dynamic_quant_per_row(y: torch.Tensor, dim: int = -1) -> tuple:
    """Per-row dynamic quantization: scale = max(abs(y))/127, q = round(y/scale)."""
    y_flat = y.float()
    scale = y_flat.abs().amax(dim=dim, keepdim=True).clamp(min=1e-8) / 127.0
    q = (y_flat / scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale.squeeze(dim).float()


class Model(nn.Module):
    """Reference: add + RMSNorm + dynamic quant (per-row scale)."""

    def __init__(self, gamma: torch.Tensor, epsilon: float = 1e-6):
        super(Model, self).__init__()
        self.gamma = gamma.detach().to(torch.float32)
        self.epsilon = epsilon

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> List[torch.Tensor]:
        x1 = x1.float()
        x2 = x2.float()
        gamma = self.gamma.to(x1.device).to(x1.dtype)
        x = x1 + x2
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        x_norm = x / rms
        y = x_norm * gamma
        y1, scale1 = _dynamic_quant_per_row(y, dim=-1)
        N = y.shape[0]
        y2 = torch.zeros_like(y1, device=y1.device, dtype=y1.dtype)
        scale2_out = torch.ones(N, device=y.device, dtype=torch.float32)
        y3 = y.float()
        y4 = y
        x_out = x
        scale1_out = scale1 if scale1.dim() >= 1 else scale1.unsqueeze(0).expand(N)
        return [y1, y2, y3, y4, x_out, scale1_out, scale2_out]


class ModelNew(nn.Module):
    def __init__(self, gamma: torch.Tensor, epsilon: float = 1e-6):
        super(ModelNew, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> List[torch.Tensor]:
        import kernel_gen_ops
        return list(kernel_gen_ops.add_rms_norm_dynamic_quant_v2(
            x1, x2, self.gamma, None, None, self.epsilon
        ))
