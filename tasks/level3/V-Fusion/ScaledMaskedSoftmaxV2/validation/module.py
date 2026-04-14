from typing import List, Optional
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                scale: float = 1.0) -> torch.Tensor:
        # Replicate kernel: scale * x -> mask apply -> softmax (on last dim)
        orig_dtype = x.dtype
        x_f = x.to(torch.float32)
        x_f = x_f * scale
        # Apply mask: set masked positions to -10000.0
        mask_f = mask.to(torch.float32)
        x_f = x_f * (1.0 - mask_f) + (-10000.0) * mask_f
        # Softmax along last dimension
        y_f = torch.softmax(x_f, dim=-1)
        return y_f.to(orig_dtype)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                scale: float = 1.0) -> torch.Tensor:
        import kernel_gen_ops
        fixed_triu_mask = False
        return kernel_gen_ops.scaled_masked_softmax_v2(x, mask, scale, fixed_triu_mask)
