from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

# 标杆与 execute_rfft1d 对齐：
#   norm: 1=backward, 2=forward, 3=ortho
#   output = torch.fft.rfft(x, n, dim, norm)
#   output = torch.stack([output.real, output.imag], dim=-1)
# NPUKernelBench 固定 dim=-1；返回时 flatten(-2) 以与 kernel 交错 [real,imag,real,imag,...] 布局一致便于精度比较

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, n: Optional[int], norm: int) -> torch.Tensor:
        if norm == 1:
            norm_str = "backward"
        elif norm == 2:
            norm_str = "forward"
        elif norm == 3:
            norm_str = "ortho"
        else:
            norm_str = "backward"
        x = input.to(torch.float32)
        dim = -1
        output = torch.fft.rfft(x, n, dim, norm_str)
        output = torch.stack([output.real, output.imag], dim=-1)
        return output.flatten(-2)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input: torch.Tensor, n: Optional[int], norm: int) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.rfft1_d(input, n, norm)
