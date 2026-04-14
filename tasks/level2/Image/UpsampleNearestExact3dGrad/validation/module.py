from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import kernel_gen_ops


class Model(nn.Module):
    """PyTorch native reference implementation (golden model)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output: torch.Tensor, input_size: List[int],
                output_size: List[int], scales_d: float = 0.0,
                scales_h: float = 0.0, scales_w: float = 0.0) -> torch.Tensor:
        # Manual backward of nearest_exact 3D upsampling
        # For nearest_exact: src_idx = floor((dst_idx + 0.5) * scale)
        # where scale = input_size / output_size
        orig_dtype = grad_output.dtype
        grad_output = grad_output.to(torch.float32)

        N, C, oD, oH, oW = grad_output.shape
        _, _, iD, iH, iW = input_size

        scale_d = iD / oD
        scale_h = iH / oH
        scale_w = iW / oW

        # Compute source indices for each output position
        d_idx = torch.arange(oD, device=grad_output.device, dtype=torch.float32)
        h_idx = torch.arange(oH, device=grad_output.device, dtype=torch.float32)
        w_idx = torch.arange(oW, device=grad_output.device, dtype=torch.float32)
        d_src = ((d_idx + 0.5) * scale_d).floor().long().clamp(0, iD - 1)
        h_src = ((h_idx + 0.5) * scale_h).floor().long().clamp(0, iH - 1)
        w_src = ((w_idx + 0.5) * scale_w).floor().long().clamp(0, iW - 1)

        # Build 3D source index grid and flatten
        src_3d = d_src.unsqueeze(1).unsqueeze(2) * (iH * iW) + h_src.unsqueeze(0).unsqueeze(2) * iW + w_src.unsqueeze(0).unsqueeze(0)  # [oD, oH, oW]
        src_flat = src_3d.reshape(-1)  # [oD * oH * oW]

        # Expand for batch/channel and scatter_add
        grad_out_flat = grad_output.reshape(N, C, oD * oH * oW)
        src_expanded = src_flat.unsqueeze(0).unsqueeze(0).expand(N, C, -1)

        grad_input_flat = torch.zeros(N, C, iD * iH * iW, device=grad_output.device, dtype=torch.float32)
        grad_input_flat.scatter_add_(2, src_expanded, grad_out_flat)

        result = grad_input_flat.reshape(N, C, iD, iH, iW)
        if orig_dtype != torch.float32:
            result = result.to(orig_dtype)
        return result


class ModelNew(nn.Module):
    """Custom kernel version."""
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, grad_output: torch.Tensor, input_size: List[int],
                output_size: List[int], scales_d: float = 0.0,
                scales_h: float = 0.0, scales_w: float = 0.0) -> torch.Tensor:
        return kernel_gen_ops.upsample_nearest_exact3d_grad(
            grad_output, input_size, output_size, scales_d, scales_h, scales_w)
