from typing import List
import torch
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """使用 PyTorch 原生算子的参考实现（golden model）。"""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad, x, grid,
                interpolation_mode='bilinear', padding_mode='zeros', align_corners=False):
        # PyTorch CPU grid_sample backward does not support float16/bfloat16,
        # so cast to float32 for computation
        input_dtype = x.dtype
        x_float = x.detach().float().requires_grad_(True)
        grid_float = grid.detach().float().requires_grad_(True)
        grad_float = grad.float()

        output = torch.nn.functional.grid_sample(
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

    def forward(self, grad, x, grid,
                interpolation_mode='bilinear', padding_mode='zeros', align_corners=False):
        return kernel_gen_ops.grid_sampler2_d_grad(grad, x, grid,
                                                  interpolation_mode, padding_mode, align_corners)
