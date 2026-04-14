from typing import List
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import torch.nn.functional as F
import kernel_gen_ops


class Model(nn.Module):
    """Reference implementation using PyTorch native ops (golden model)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output, output_size, input_size, align_corners):
        # Use PyTorch autograd to compute the backward pass of bilinear 2D AA upsampling
        # input_size: [N, C, H_in, W_in] - original input shape
        # output_size: [H_out, W_out] - upsampled output spatial size
        N, C, H_in, W_in = input_size
        H_out, W_out = output_size

        # F.interpolate with antialias=True only supports float32,
        # so compute reference in float32 and cast back
        orig_dtype = grad_output.dtype
        compute_dtype = torch.float32

        dummy_input = torch.randn(N, C, H_in, W_in, dtype=compute_dtype, device=grad_output.device, requires_grad=True)
        grad_output_f32 = grad_output.to(compute_dtype)

        # Forward: bilinear 2D AA upsample
        upsampled = F.interpolate(
            dummy_input,
            size=(H_out, W_out),
            mode='bilinear',
            align_corners=align_corners,
            antialias=True
        )

        # Backward
        upsampled.backward(grad_output_f32)

        result = dummy_input.grad
        if orig_dtype != compute_dtype:
            result = result.to(orig_dtype)
        return result


class ModelNew(nn.Module):
    """Custom kernel version."""
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, grad_output, output_size, input_size, align_corners):
        N, C, H_in, W_in = input_size
        H_out, W_out = output_size

        # Compute scales matching the tiling code's ComputeScaleValue logic
        if align_corners:
            if H_out > 1:
                scales_h = float(H_in - 1) / float(H_out - 1)
            else:
                scales_h = 0.0
            if W_out > 1:
                scales_w = float(W_in - 1) / float(W_out - 1)
            else:
                scales_w = 0.0
        else:
            scales_h = float(H_in) / float(H_out)
            scales_w = float(W_in) / float(W_out)

        full_input_size = [N, C, H_in, W_in]

        return kernel_gen_ops.upsample_bilinear2d_aa_backward(
            grad_output,
            output_size,
            full_input_size,
            align_corners,
            scales_h,
            scales_w
        )
