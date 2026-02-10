from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """
    实现LinearIndex算子功能的模型（CPU参考实现）。
    """

    def __init__(self, axis=-1, combine=False):
        """
        初始化模型。
        
        Args:
            axis: 目标维度
            combine: 是否合轴
        """
        super(Model, self).__init__()
        self.axis = axis
        self.combine = combine

    def forward(self, indices: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
        """
        实现LinearIndex算子功能。

        Args:
            indices: 索引张量
            shape: 形状张量

        Returns:
            线性索引张量（int32）
        """
        # 转换为int32进行计算
        indices_int32 = indices.to(torch.int32)
        shape_list = shape.tolist()
        
        # 处理axis
        axis = self.axis
        if axis < 0:
            axis += len(shape_list)
        
        target = shape_list[axis]
        
        # 处理负数索引
        mask = indices_int32 < 0
        indices_int32 = torch.where(mask, indices_int32 + target, indices_int32)
        
        # 基础模式：不合轴
        if not self.combine:
            return indices_int32
        
        # 合轴模式
        indices_shape = indices.shape
        
        # 二维场景或三维场景
        if len(shape_list) == 2:
            # 二维场景，合轴和不合轴结果相同
            return indices_int32
        elif len(shape_list) == 3:
            self_stride = shape_list[1]
            
            if len(indices_shape) == 2:
                # 三维场景，indices是二维
                indices_stride = indices_shape[1]
                
                # 生成偏移量
                offset = torch.arange(indices_int32.numel(), dtype=torch.int32, device=indices.device)
                offset = offset.reshape(indices_shape)
                
                if axis == 0:
                    # MODE 1: dim=0
                    # 计算余数
                    remainder = offset % indices_stride
                    # indices * self_stride + remainder
                    result = indices_int32 * self_stride + remainder
                else:  # axis == 1
                    # MODE 2: dim=1
                    # 计算商
                    quotient = offset // indices_stride
                    # indices + self_stride * quotient
                    result = indices_int32 + self_stride * quotient
                
                return result
        
        return indices_int32


class ModelNew(nn.Module):
    def __init__(self, axis=-1, combine=False):
        super(ModelNew, self).__init__()
        self.axis = axis
        self.combine = combine

    def forward(self, indices: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
        return kernel_gen_ops.linear_index(indices, shape, self.axis, self.combine)