from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import kernel_gen_ops


class Model(nn.Module):
    """PyTorch native reference implementation (golden model)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output: torch.Tensor, output_size: List[int],
                input_size: List[int], scales_h: float = 0.0, scales_w: float = 0.0) -> torch.Tensor:
        # Manual backward of nearest_exact 2D upsampling
        # For nearest_exact: src_idx = floor((dst_idx + 0.5) * scale)
        # where scale = input_size / output_size
        orig_dtype = grad_output.dtype
        N, C, oH, oW = grad_output.shape
        _, _, iH, iW = input_size

        scale_h = iH / oH
        scale_w = iW / oW

        # Compute source indices for each output position
        h_idx = torch.arange(oH, device=grad_output.device, dtype=torch.float32)
        w_idx = torch.arange(oW, device=grad_output.device, dtype=torch.float32)
        h_src = ((h_idx + 0.5) * scale_h).floor().long().clamp(0, iH - 1)
        w_src = ((w_idx + 0.5) * scale_w).floor().long().clamp(0, iW - 1)

        # Build 2D source index grid and flatten
        src_2d = h_src.unsqueeze(1) * iW + w_src.unsqueeze(0)  # [oH, oW]
        src_flat = src_2d.reshape(-1)  # [oH * oW]

        # Expand for batch/channel and scatter_add
        # Use float32 for accumulation to avoid precision loss in half types
        grad_out_flat = grad_output.reshape(N, C, oH * oW).float()
        src_expanded = src_flat.unsqueeze(0).unsqueeze(0).expand(N, C, -1)

        grad_input_flat = torch.zeros(N, C, iH * iW, device=grad_output.device, dtype=torch.float32)
        grad_input_flat.scatter_add_(2, src_expanded, grad_out_flat)

        result = grad_input_flat.reshape(N, C, iH, iW)
        if orig_dtype != torch.float32:
            result = result.to(orig_dtype)
        return result


class ModelNew(nn.Module):
    """Custom kernel version."""
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, grad_output: torch.Tensor, output_size: List[int],
                input_size: List[int], scales_h: float = 0.0, scales_w: float = 0.0) -> torch.Tensor:
        return kernel_gen_ops.upsample_nearest_exact2d_grad(
            grad_output, output_size, input_size, scales_h, scales_w)
