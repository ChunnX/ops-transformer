from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu
from torch_npu.contrib import transfer_to_npu
import kernel_gen_ops


class Model(nn.Module):
    """使用 PyTorch autograd 的参考实现（golden model）。"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output, input_size, align_corners):
        N, C, H_grad, W_grad = grad_output.shape
        H_in, W_in = input_size

        # Compute golden via autograd on CPU (NPU bicubic autograd may not be supported)
        # Must use enable_grad() because the framework wraps in torch.no_grad()
        with torch.enable_grad():
            grad_cpu = grad_output.detach().float().cpu()
            dummy_input = torch.randn(
                N, C, H_in, W_in,
                dtype=torch.float32,
                device="cpu",
                requires_grad=True,
            )
            output = F.interpolate(
                dummy_input,
                size=(H_grad, W_grad),
                mode="bicubic",
                align_corners=align_corners,
            )
            output.backward(grad_cpu)
        return dummy_input.grad.to(grad_output.dtype).to(grad_output.device)


class ModelNew(nn.Module):
    """自定义 kernel 版本。"""

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, grad_output, input_size, align_corners):
        N, C, H_grad, W_grad = grad_output.shape
        H_in, W_in = input_size

        # Compute scales matching tiling code logic
        if align_corners:
            scales_h = float(H_in - 1) / float(H_grad - 1) if H_grad > 1 else 0.0
            scales_w = float(W_in - 1) / float(W_grad - 1) if W_grad > 1 else 0.0
        else:
            scales_h = float(H_in) / float(H_grad)
            scales_w = float(W_in) / float(W_grad)

        return kernel_gen_ops.upsample_bicubic2d_grad(
            grad_output, align_corners, scales_h, scales_w, input_size
        )
