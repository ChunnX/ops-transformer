from typing import List
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import torch.nn.functional as F
import kernel_gen_ops


class Model(nn.Module):
    """Reference implementation using PyTorch native ops (golden model)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input, output_size, align_corners):
        return F.interpolate(input, size=output_size, mode='bicubic',
                             align_corners=align_corners)


class ModelNew(nn.Module):
    """Custom kernel version."""
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input, output_size, align_corners):
        # Compute scales for the CANN op
        h_in = input.shape[2]
        w_in = input.shape[3]
        h_out = output_size[0]
        w_out = output_size[1]

        if align_corners and h_out > 1:
            scales_h = float(h_in - 1) / float(h_out - 1)
        else:
            scales_h = float(h_in) / float(h_out)

        if align_corners and w_out > 1:
            scales_w = float(w_in - 1) / float(w_out - 1)
        else:
            scales_w = float(w_in) / float(w_out)

        return kernel_gen_ops.upsample_bicubic2d(input, output_size, align_corners,
                                                  scales_h, scales_w)
