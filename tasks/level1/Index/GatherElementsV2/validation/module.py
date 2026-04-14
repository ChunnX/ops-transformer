import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """PyTorch native torch.gather_elements reference implementation."""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, index: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.gather(x, dim, index.long())


class ModelNew(nn.Module):
    """Custom kernel version."""

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor, index: torch.Tensor, dim: int) -> torch.Tensor:
        return kernel_gen_ops.gather_elements_v2(x, index, dim)
