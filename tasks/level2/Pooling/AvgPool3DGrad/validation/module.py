from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import math

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_output: torch.Tensor, x: torch.Tensor, ksize: List[int], strides: List[int], pads: List[int], ceil_mode: bool, count_include_pad: bool, divisor_override: int, data_format: str) -> torch.Tensor:
        res = torch.ops.aten.avg_pool3d_backward(
            grad_output.to(torch.float32).npu(),
            x.to(torch.float32).npu(),
            ksize,
            strides,
            pads,
            ceil_mode,
            count_include_pad,
            divisor_override)
        return res.to(x.dtype).cpu()

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, grad_output: torch.Tensor, x: torch.Tensor, ksize: List[int], strides: List[int], pads: List[int], ceil_mode: bool, count_include_pad: bool, divisor_override: int, data_format: str) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.avg_pool3_d_grad(grad_output, x, ksize, strides, pads, ceil_mode, count_include_pad, divisor_override)