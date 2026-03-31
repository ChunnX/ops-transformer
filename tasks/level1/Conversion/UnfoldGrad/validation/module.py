from typing import List

import torch
import torch.nn as nn

import kernel_gen_ops


class Model(nn.Module):
    def __init__(self, input_shape: List[int], dim: int, size: int, step: int):
        super().__init__()
        self.input_shape = list(input_shape)
        self.dim = int(dim)
        self.size = int(size)
        self.step = int(step)

    def forward(self, grad_out: torch.Tensor) -> torch.Tensor:
        x = torch.zeros(
            self.input_shape,
            dtype=grad_out.dtype,
            device=grad_out.device,
            requires_grad=True,
        )
        u = x.unfold(self.dim, self.size, self.step)
        u.backward(grad_out)
        return x.grad


class ModelNew(nn.Module):
    def __init__(self, input_shape: List[int], dim: int, size: int, step: int):
        super().__init__()
        self.input_shape = list(input_shape)
        self.dim = int(dim)
        self.size = int(size)
        self.step = int(step)

    def forward(self, grad_out: torch.Tensor) -> torch.Tensor:
        return kernel_gen_ops.unfold_grad(grad_out, self.input_shape, self.dim, self.size, self.step)
