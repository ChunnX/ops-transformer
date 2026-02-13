from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

# Rfft1D: norm 1=backward, 2=forward, 3=ortho (same as torch.fft)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, n: Optional[int], norm: int) -> torch.Tensor:
        norm_str = {1: 'backward', 2: 'forward', 3: 'ortho'}.get(norm, 'backward')
        out = torch.fft.rfft(input.to(torch.float32), n=n, dim=-1, norm=norm_str)
        # rfft returns complex; real-view for comparison if kernel outputs interleaved real/imag
        return torch.view_as_real(out).flatten(-2)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input: torch.Tensor, n: Optional[int], norm: int) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.rfft1_d(input, n, norm)
