from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import math

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, start: torch.Tensor, end: torch.Tensor, steps: int) -> torch.Tensor:
        return torch.linspace(float(start[0].item()), float(end[0].item()), steps, device="npu").to(start.dtype).cpu()

class ModelNew(nn.Module):
    def __init__(self):
        
        super(ModelNew, self).__init__()

    def forward(self, start: torch.Tensor, end: torch.Tensor, steps: int) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.lin_space(start, end, steps)