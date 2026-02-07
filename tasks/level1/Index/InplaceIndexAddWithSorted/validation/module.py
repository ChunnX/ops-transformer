from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """
    实现InplaceIndexAddWithSorted算子功能的模型(torch标杆)。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, var: torch.Tensor, value: torch.Tensor, sorted_indices: torch.Tensor, 
                pos: torch.Tensor, axis: int, alpha: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        实现InplaceIndexAddWithSorted算子功能。

        Args:
            var: 待更新的张量
            value: 更新值张量
            sorted_indices: 已排序的索引张量
            pos: 位置索引张量
            axis: 操作的维度
            alpha: 可选的缩放因子

        Returns:
            更新后的var张量
        """
        # 克隆var以避免修改原始输入
        result = var.clone()
        
        # 获取alpha值
        alpha_value = alpha.item() if alpha is not None else 1.0
        
        # 根据sorted_indices和pos进行索引加法
        # result[sorted_indices] += alpha * value[pos]
        for i in range(len(sorted_indices)):
            idx = sorted_indices[i].item()
            p = pos[i].item()
            if axis == 0:
                result[idx] = result[idx] + alpha_value * value[p]
            else:
                raise NotImplementedError("Only axis=0 is supported")
        
        return result


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, var: torch.Tensor, value: torch.Tensor, sorted_indices: torch.Tensor, 
                pos: torch.Tensor, axis: int, alpha: Optional[torch.Tensor] = None) -> torch.Tensor:
        return kernel_gen_ops.inplace_index_add_with_sorted(var, value, sorted_indices, pos, axis, alpha)