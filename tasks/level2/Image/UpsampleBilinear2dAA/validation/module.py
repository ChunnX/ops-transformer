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

    def forward(self, input, output_size, align_corners):
        orig_dtype = input.dtype
        input_f32 = input.to(torch.float32)
        result = F.interpolate(input_f32, size=output_size, mode="bilinear", align_corners=align_corners, antialias=True)
        if orig_dtype != torch.float32:
            result = result.to(orig_dtype)
        return result


class ModelNew(nn.Module):
    """自定义 kernel 版本。"""
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input, output_size, align_corners):
        # Compute scales matching the tiling compute_scale_value logic
        input_h = input.shape[2]
        input_w = input.shape[3]
        scales_h = float(input_h) / float(output_size[0])
        scales_w = float(input_w) / float(output_size[1])
        return kernel_gen_ops.upsample_bilinear2d_aa(input, output_size, align_corners, scales_h, scales_w)
