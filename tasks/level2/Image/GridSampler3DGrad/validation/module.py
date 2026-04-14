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

    def forward(self, grad, x, grid, interpolation_mode='bilinear', padding_mode='zeros', align_corners=False):
        # PyTorch CPU grid_sample backward does not support float16/bfloat16 well,
        # so cast to float32 for computation
        input_dtype = x.dtype
        x_float = x.detach().float().requires_grad_(True)
        grid_float = grid.detach().float().requires_grad_(True)
        grad_float = grad.float()

        output = F.grid_sample(
            x_float, grid_float,
            mode=interpolation_mode,
            padding_mode=padding_mode,
            align_corners=align_corners
        )
        output.backward(grad_float)
        dx = x_float.grad.to(input_dtype)
        dgrid = grid_float.grad.to(input_dtype)
        return dx, dgrid


class ModelNew(nn.Module):
    """自定义 kernel 版本。"""
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, grad, x, grid, interpolation_mode='bilinear', padding_mode='zeros', align_corners=False):
        input_dtype = x.dtype
        # aclnnGridSampler3DGrad may not support float16/bfloat16, cast to float32
        if input_dtype != torch.float32:
            grad = grad.float()
            x = x.float()
            grid = grid.float()
        dx, dgrid = kernel_gen_ops.grid_sampler3_d_grad(
            grad, x, grid,
            interpolation_mode, padding_mode, align_corners
        )
        if input_dtype != torch.float32:
            dx = dx.to(input_dtype)
            dgrid = dgrid.to(input_dtype)
        return dx, dgrid
