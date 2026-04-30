import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x1, x2, bias=None, transpose_x1=False, transpose_x2=False):
        if transpose_x1:
            x1 = x1.transpose(-2, -1)
        if transpose_x2:
            x2 = x2.transpose(-2, -1)
        result = torch.matmul(x1.float(), x2.float())
        if bias is not None:
            result = result + bias.float()
        return result.to(x1.dtype)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x1, x2, bias=None, transpose_x1=False, transpose_x2=False):
        offset_w = torch.empty(0, dtype=torch.int8, device=x1.device)
        offset_x = 0
        opImplMode = 1
        if bias is None:
            bias = torch.empty(0, dtype=x1.dtype, device=x1.device)
        return kernel_gen_ops.mat_mul_v3(
            x1, x2, bias, offset_w,
            transpose_x1, transpose_x2, offset_x, opImplMode
        )
