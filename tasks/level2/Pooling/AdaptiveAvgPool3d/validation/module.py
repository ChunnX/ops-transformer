from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import math

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, output_size: List[int]) -> torch.Tensor:
        m = torch.nn.AdaptiveAvgPool3d(output_size)
        return m(x.npu()).cpu()

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor, output_size: List[int]) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.adaptive_avg_pool3d(x, output_size)