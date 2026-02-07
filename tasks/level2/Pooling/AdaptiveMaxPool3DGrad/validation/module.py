from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import math

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, y_grad: torch.Tensor, x: torch.Tensor, argmax: torch.Tensor) -> torch.Tensor:
        res = torch.ops.aten.adaptive_max_pool3d_backward(
            y_grad.to(torch.float32).npu(),
            x.to(torch.float32).npu(),
            argmax.to(torch.int64).npu())
        return res.to(x.dtype).cpu()

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, y_grad: torch.Tensor, x: torch.Tensor, argmax: torch.Tensor) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.adaptive_max_pool3d_grad(y_grad, x, argmax)