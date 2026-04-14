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

    def forward(self, input, size, align_corners=False):
        # size 是 [out_h, out_w] 的 int32 tensor
        out_h, out_w = size[0].item(), size[1].item()
        return F.interpolate(input, size=[out_h, out_w], mode='bilinear', align_corners=align_corners)


class ModelNew(nn.Module):
    """自定义 kernel 版本。"""
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input, size, align_corners=False):
        # 从输入/输出形状推导 scales
        out_h, out_w = size[0].item(), size[1].item()
        in_h, in_w = input.shape[2], input.shape[3]

        # compute_scale_value 逻辑与 tiling 代码一致
        def compute_scale(input_size, output_size, align_corners, scale=0.0):
            if output_size == input_size:
                return 1.0
            if align_corners:
                if output_size > 1:
                    return float(input_size - 1) / (output_size - 1)
                else:
                    return 0.0
            else:
                return float(scale) if scale > 0 else float(input_size) / output_size

        scales_h = compute_scale(in_h, out_h, align_corners)
        scales_w = compute_scale(in_w, out_w, align_corners)

        return kernel_gen_ops.upsample_bilinear2d(input, size, align_corners, scales_h, scales_w)
