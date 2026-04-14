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

    def forward(self, x, output_size, align_corners=False):
        orig_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        result = F.interpolate(x_f32, size=output_size, mode='bicubic', align_corners=align_corners, antialias=True)
        if orig_dtype != torch.float32:
            result = result.to(orig_dtype)
        return result


class ModelNew(nn.Module):
    """自定义 kernel 版本。"""
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x, output_size, align_corners=False):
        # Pass 0.0 for scales; tiling code will compute correct scale from shapes
        return kernel_gen_ops.upsample_bicubic2d_aa(x, output_size, align_corners, 0.0, 0.0)
