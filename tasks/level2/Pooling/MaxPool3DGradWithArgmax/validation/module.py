from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import math

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, grad: torch.Tensor, indices: torch.Tensor, ksize: List[int], strides: List[int], pads: List[int], dilation: List[int], ceil_mode: bool) -> torch.Tensor:
        m = torch.nn.MaxPool3d(kernel_size=ksize, stride=strides, padding=pads,
                           dilation=dilation, return_indices=True, ceil_mode=ceil_mode)
        out, indices = m(x.to(torch.float32))

        gradOut_float = grad.to(torch.float32)
        x_float = x.to(torch.float32)
        grad_in = torch.ops.aten.max_pool3d_with_indices_backward(
            gradOut_float, x_float, ksize, strides, pads, dilation, ceil_mode, indices.to(torch.int64))
        return grad_in.to(x.dtype)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor, grad: torch.Tensor, indices: torch.Tensor, ksize: List[int], strides: List[int], pads: List[int], dilation: List[int], ceil_mode: bool) -> torch.Tensor:
        import kernel_gen_ops
        m = torch.nn.MaxPool3d(kernel_size=ksize, stride=strides, padding=pads,
                           dilation=dilation, return_indices=True, ceil_mode=ceil_mode)
        out, indices = m(x.cpu().to(torch.float32))

        return kernel_gen_ops.max_pool3d_grad_with_argmax(x, grad, indices.to(torch.int32).to(x.device), ksize, strides, pads, dilation, ceil_mode)