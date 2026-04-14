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
        return F.interpolate(x, size=output_size, mode='trilinear', align_corners=align_corners)


class ModelNew(nn.Module):
    """自定义 kernel 版本。"""
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x, output_size, align_corners=False):
        # Compute scales for the CANN op (input_size / output_size)
        scales_d = float(x.shape[2]) / float(output_size[0])
        scales_h = float(x.shape[3]) / float(output_size[1])
        scales_w = float(x.shape[4]) / float(output_size[2])
        return kernel_gen_ops.resize_upsample_trilinear(
            x, list(output_size), align_corners, scales_d, scales_h, scales_w
        )
