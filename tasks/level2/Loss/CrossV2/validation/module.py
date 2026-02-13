from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn



class Model(nn.Module):
    """
    实现CrossV2功能的golden模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, dim: int):
        """
        实现CrossV2功能。

        Args:
            x1 (Tensor): 第一个输入张量，形状为 [B, D1, D2, ..., Dk]，其中最后一个维度必须 ≥ 3。
            x2 (Tensor): 第二个输入张量，形状与 x1 完全一致，用于与 x1 进行叉积运算。
            dim (int): 指定进行叉积运算的维度索引。支持负索引（如 -1 表示倒数第一维）。必须为整数。
                   - 叉积在该维度的最后三个元素上执行。
                   - 若 dim 为 -1，则在倒数第一维进行叉积。
                   - 若维度大小 < 3，将引发错误。

        Returns:
            y (Tensor): 输出张量，形状与输入张量 x1 相同，表示在指定维度上的叉积结果。
        """
        # Golden: torch.cross
        compute_dtype = x1.dtype
        if compute_dtype in (torch.float16, torch.bfloat16):
            x1 = x1.to(torch.float)
            x2 = x2.to(torch.float)
        y = torch.cross(x1, x2, dim=dim)
        if compute_dtype in (torch.float16, torch.bfloat16):
            y = y.to(compute_dtype)
        return y


class ModelNew(nn.Module):
    def __init__(self):
        """
        初始化模型。
        """
        super(ModelNew, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, dim: int):
        """
        调用算子。

        Args:
            x1 (Tensor): 第一个输入张量，形状为 [B, D1, D2, ..., Dk]，其中最后一个维度必须 ≥ 3。
            x2 (Tensor): 第二个输入张量，形状与 x1 完全一致，用于与 x1 进行叉积运算。
            dim (int): 指定进行叉积运算的维度索引。支持负索引（如 -1 表示倒数第一维）。必须为整数。
                   - 叉积在该维度的最后三个元素上执行。
                   - 若 dim 为 -1，则在倒数第一维进行叉积。
                   - 若维度大小 < 3，将引发错误。

        Returns:
            y (Tensor): 输出张量，形状与输入张量 x1 相同，表示在指定维度上的叉积结果。
        """
        import kernel_gen_ops
        return kernel_gen_ops.cross_v2(x1, x2, dim)