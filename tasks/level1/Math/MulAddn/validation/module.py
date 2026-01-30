from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import math

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x1: List[torch.Tensor], x2: List[torch.Tensor], n: int) -> torch.Tensor:
        out_shape = [x1[0].shape[0], x1[0].shape[1], x2[0].shape[2]]
        if x1[0].dtype != torch.float16:
            out = torch.zeros(out_shape).to(torch.float32)
            for i in range(n):
                out = out + torch.matmul(x1[i].to(torch.float32), x2[i].to(torch.float32))
            return out.to(x1[0].dtype)
        else:
            out = torch.zeros(out_shape).to(x1[0].dtype)
            for i in range(n):
                out = out + torch.matmul(x1[i], x2[i])
            return out

class ModelNew(nn.Module):
    def __init__(self):
        
        super(ModelNew, self).__init__()

    def forward(self, x1: List[torch.Tensor], x2: List[torch.Tensor], n: int) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.mul_addn(x1, x2, n)