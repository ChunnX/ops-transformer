from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import math

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, ksize: List[int], strides: List[int], pads: List[int], dilation: List[int], ceil_mode: bool, data_format: str) -> torch.Tensor:
        m = torch.nn.MaxPool3d(kernel_size=ksize, stride=strides, padding=pads, dilation=dilation,
                               return_indices=True, ceil_mode=ceil_mode)
        ori_dtype = x.dtype
        if x.dtype != torch.float:
            x_float = x.to(torch.float)
        else:
            x_float = x
        output, indices = m(x_float)
        if ori_dtype != torch.float:
            output = output.to(ori_dtype)
        indices = indices.to(torch.int32)
        return [output.cpu(), indices.cpu()]

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor, ksize: List[int], strides: List[int], pads: List[int], dilation: List[int], ceil_mode: bool, data_format: str) -> torch.Tensor:
        import kernel_gen_ops
        return list(kernel_gen_ops.max_pool3d_with_argmax_v2(x, ksize, strides, pads, dilation, ceil_mode))