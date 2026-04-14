from typing import List
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import torch.nn.functional as F
import kernel_gen_ops


class Model(nn.Module):
    """Golden model: use PyTorch autograd to compute the backward of bilinear 2d upsampling."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output, output_size, input_size, align_corners=False):
        # Create a dummy input matching the forward pass input shape
        dummy_input = torch.zeros(input_size, dtype=grad_output.dtype, device=grad_output.device, requires_grad=True)
        # Forward upsample
        upsampled = F.interpolate(dummy_input, size=output_size[2:], mode='bilinear', align_corners=align_corners)
        # Backward pass with the provided grad_output
        upsampled.backward(grad_output)
        return dummy_input.grad


class ModelNew(nn.Module):
    """Custom kernel version."""
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, grad_output, output_size, input_size, align_corners=False):
        ac = align_corners
        if isinstance(ac, int):
            ac = bool(ac)

        # Compute scales following the same logic as compute_scale_value in tiling
        def compute_scale(in_size, out_size, align_corner):
            if align_corner:
                if in_size > 1:
                    return float(out_size - 1) / float(in_size - 1)
                else:
                    return 0.0
            else:
                return float(out_size) / float(in_size) if in_size != 0 else 1.0

        scale_h = compute_scale(output_size[2], input_size[2], ac)
        scale_w = compute_scale(output_size[3], input_size[3], ac)

        return kernel_gen_ops.upsample_bilinear2d_grad(
            grad_output, output_size, input_size, ac, scale_h, scale_w
        )
