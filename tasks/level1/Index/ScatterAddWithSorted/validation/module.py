from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """
    实现ScatterAddWithSorted算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, var: torch.Tensor, value: torch.Tensor, sorted_index: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        实现ScatterAddWithSorted算子功能。

        Args:
            var: 目标张量
            value: 更新值张量
            sorted_index: 已排序的索引张量
            pos: 可选的位置信息张量

        Returns:
            更新后的结果张量
        """
        # 克隆var以避免修改原始数据
        result = var.clone()
        
        # 获取索引对应的位置
        for i in range(sorted_index.shape[0]):
            idx = sorted_index[i].item()
            result[idx] = result[idx] + value[i]
        
        return result


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, var: torch.Tensor, value: torch.Tensor, sorted_index: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        return kernel_gen_ops.scatter_add_with_sorted(var, value, sorted_index, pos)