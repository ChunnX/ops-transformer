from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import math

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, min: torch.Tensor, max: torch.Tensor, bins: int) -> torch.Tensor:
        return torch.histc(input.to(torch.float32), bins=bins, min=min[0].item(), max=max[0].item()).to(torch.int32)

class ModelNew(nn.Module):
    def __init__(self):
        
        super(ModelNew, self).__init__()

    def forward(self, input: torch.Tensor, min: torch.Tensor, max: torch.Tensor, bins: int) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.histogram_v2(input, min, max, bins)