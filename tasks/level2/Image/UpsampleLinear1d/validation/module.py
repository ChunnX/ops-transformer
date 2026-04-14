from typing import List
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import torch.nn.functional as F
import kernel_gen_ops


class Model(nn.Module):
    """使用 PyTorch 原生算子的参考实现（golden model）。"""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, output_size, align_corners):
        return F.interpolate(x, size=output_size, mode='linear', align_corners=align_corners)


class ModelNew(nn.Module):
    """自定义 kernel 版本。"""
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x, output_size, align_corners):
        scale = 0.0
        return kernel_gen_ops.upsample_linear1d(x, [output_size], align_corners, scale)
