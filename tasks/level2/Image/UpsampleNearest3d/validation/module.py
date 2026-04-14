import torch
import torch.nn as nn
import kernel_gen_ops
from typing import List


class Model(nn.Module):
    """PyTorch native reference implementation (golden model)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, output_size: List[int],
                scale_d: float = 0.0, scale_h: float = 0.0, scale_w: float = 0.0) -> torch.Tensor:
        N, C, iD, iH, iW = x.shape
        oD, oH, oW = output_size

        s_d = iD / oD if scale_d <= 0.0 else scale_d
        s_h = iH / oH if scale_h <= 0.0 else scale_h
        s_w = iW / oW if scale_w <= 0.0 else scale_w

        d_idx = torch.arange(oD, device=x.device, dtype=torch.float32)
        h_idx = torch.arange(oH, device=x.device, dtype=torch.float32)
        w_idx = torch.arange(oW, device=x.device, dtype=torch.float32)

        d_src = (d_idx * s_d).floor().long().clamp(0, iD - 1)
        h_src = (h_idx * s_h).floor().long().clamp(0, iH - 1)
        w_src = (w_idx * s_w).floor().long().clamp(0, iW - 1)

        return x[:, :, d_src.unsqueeze(1).unsqueeze(1), h_src.unsqueeze(1), w_src]


class ModelNew(nn.Module):
    """Custom kernel version."""
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor, output_size: List[int],
                scale_d: float = 0.0, scale_h: float = 0.0, scale_w: float = 0.0) -> torch.Tensor:
        return kernel_gen_ops.upsample_nearest3d(
            x, output_size, scale_d, scale_h, scale_w)
