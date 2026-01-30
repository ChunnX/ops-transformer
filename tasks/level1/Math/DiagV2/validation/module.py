from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """
    实现AngleV2算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, diagonal: int = 0) -> torch.Tensor:
        """
        实现AngleV2算子功能。
        Args:
            x: 输入张量
        Returns:
            计算输入张量逐元素的角度
        """
        return torch.diag(x, diagonal)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor, diagonal: int = 0) -> torch.Tensor:
        return kernel_gen_ops.diag_v2(x, diagonal)