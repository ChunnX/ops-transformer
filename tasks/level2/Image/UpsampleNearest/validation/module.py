from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import kernel_gen_ops


class Model(nn.Module):
    """PyTorch native reference implementation (golden model)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, output_size: List[int], exact_mode: bool = True) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        mode = 'nearest-exact' if exact_mode else 'nearest'
        result = F.interpolate(x_float, size=output_size, mode=mode)
        return result.to(input_dtype)


class ModelNew(nn.Module):
    """Custom kernel version."""
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor, output_size: List[int], exact_mode: bool = True) -> torch.Tensor:
        scales_h = 0.0
        scales_w = 0.0
        return kernel_gen_ops.upsample_nearest(x, output_size, scales_h, scales_w, exact_mode)
