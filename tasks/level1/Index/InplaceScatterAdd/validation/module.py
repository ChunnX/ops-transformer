from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """
    实现InplaceScatterAdd算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, var: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
        """
        实现InplaceScatterAdd算子功能。

        Args:
            var: 被更新的张量 [M, N]
            indices: 索引张量 [K]
            updates: 更新值张量 [K, N]

        Returns:
            更新后的var张量
        """
        # 使用PyTorch的scatter_add实现标杆
        result = var.clone()
        # scatter_add_(dim, index, src)
        # dim=0表示在第0维度上scatter
        result.scatter_add_(0, indices.unsqueeze(1).expand_as(updates), updates)
        
        return result


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, var: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor) -> torch.Tensor:
        return kernel_gen_ops.inplace_scatter_add(var, indices, updates)