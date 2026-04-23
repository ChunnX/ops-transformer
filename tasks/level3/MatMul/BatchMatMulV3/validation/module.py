import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x1, x2, bias=None, adj_x1=False, adj_x2=False):
        if adj_x1:
            x1 = x1.transpose(-2, -1)
        if adj_x2:
            x2 = x2.transpose(-2, -1)
        result = torch.bmm(x1, x2)
        if bias is not None:
            result = result + bias
        return result


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x1, x2, bias=None, adj_x1=False, adj_x2=False):
        offset_w = torch.empty(0, dtype=torch.int8, device=x1.device)
        enable_hf32 = False
        offset_x = 0
        if bias is None:
            bias = torch.empty(0, dtype=x1.dtype, device=x1.device)
        result = kernel_gen_ops.batch_mat_mul_v3(
            x1, x2, bias, offset_w,
            adj_x1, adj_x2, offset_x, enable_hf32
        )
        return result
