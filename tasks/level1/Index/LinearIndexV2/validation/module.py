from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """
    实现LinearIndexV2算子功能的模型 - 将多维索引转换为线性索引。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, indices_list: List[torch.Tensor], stride: torch.Tensor, 
                value_size: torch.Tensor) -> torch.Tensor:
        """
        实现LinearIndexV2算子功能。
        
        计算公式: output += (indices[i] % value_size[i]) * stride[i]

        Args:
            indices_list: 索引张量列表
            stride: 步长张量 [dim_num]
            value_size: 维度大小张量 [dim_num]

        Returns:
            线性索引张量 (int32)
        """
        # 初始化输出
        output = torch.zeros_like(indices_list[0], dtype=torch.int32)
        
        # 对每个维度的索引进行处理
        for i, indices in enumerate(indices_list):
            # 转换为int64进行计算以避免溢出
            indices_int64 = indices.to(torch.int64)
            value_size_val = value_size[i].item()
            stride_val = stride[i].item()
            
            # 计算: (indices % value_size) * stride
            # 向下取整除法: indices // value_size
            quotient = torch.div(indices_int64, value_size_val, rounding_mode='floor')
            # 取余: indices - quotient * value_size
            remainder = indices_int64 - quotient * value_size_val
            # 乘以步长
            contribution = (remainder * stride_val).to(torch.int32)
            
            # 累加到输出
            output = output + contribution

        return output


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, indices_list: List[torch.Tensor], stride: torch.Tensor, 
                value_size: torch.Tensor) -> torch.Tensor:
        return kernel_gen_ops.linear_index_v2(tuple(indices_list), stride, value_size)