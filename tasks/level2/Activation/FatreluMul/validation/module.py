from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import math

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, threshlod: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.chunk(input.to(torch.float32), chunks=2, dim=-1)
        res = torch.nn.functional.threshold(x1, threshlod.to(torch.float32)[0].item(), float(0)) * x2
        return res.to(input.dtype)

class ModelNew(nn.Module):
    def __init__(self):
        
        super(ModelNew, self).__init__()

    def forward(self, input: torch.Tensor, threshlod: torch.Tensor) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.fatrelu_mul(input, threshlod)