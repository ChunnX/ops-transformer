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
        output, indices = torch.nn.functional.adaptive_max_pool3d(x.npu().to(torch.float32), output_size, True)
        return [output.to(x.dtype).cpu(), indices.to(torch.int32).cpu()]

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor, output_size: List[int]) -> torch.Tensor:
        import kernel_gen_ops
        return list(kernel_gen_ops.adaptive_max_pool3d(x, output_size))