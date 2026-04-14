from typing import List
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import torch.nn.functional as F
import kernel_gen_ops


class Model(nn.Module):
    """使用 PyTorch 原生算子的参考实现（golden model）。"""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output, output_size, input_size, align_corners=False):
        orig_dtype = grad_output.dtype
        compute_dtype = torch.float32
        # Create a leaf tensor matching the original input shape in float32
        x = torch.randn(input_size, dtype=compute_dtype, device=grad_output.device, requires_grad=True)
        # Forward pass: upsample from input_size spatial dims to output_size
        y = F.interpolate(x, size=output_size, mode='bicubic', align_corners=align_corners, antialias=True)
        # Backward pass with the given grad_output (cast to float32 for computation)
        y.backward(grad_output.to(compute_dtype))
        result = x.grad
        if orig_dtype != compute_dtype:
            result = result.to(orig_dtype)
        return result


class ModelNew(nn.Module):
    """自定义 kernel 版本。"""
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, grad_output, output_size, input_size, align_corners=False):
        # Pass 0.0 for scales; tiling code will compute correct scale from shapes
        return kernel_gen_ops.upsample_bicubic2d_aa_grad(grad_output, output_size, input_size, align_corners, 0.0, 0.0)
