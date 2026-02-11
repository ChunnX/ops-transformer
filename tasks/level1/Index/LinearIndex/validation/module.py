import torch
import torch.nn as nn
import torch_npu
from torch_npu.contrib import transfer_to_npu
import kernel_gen_ops

class Model(nn.Module):
    """
    实现 LinearIndex 算子功能的标杆模型。
    """
    def __init__(self, axis=-1, combine=False):
        super(Model, self).__init__()
        self.axis = axis
        self.combine = combine

    def forward(self, indices: torch.Tensor, shape_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: 输入索引张量
            shape_tensor: 表示维度的 1D Int32 张量
        Returns:
            计算后的 int32 索引张量
        """
        # 获取 shape 数值
        shape_list = shape_tensor.tolist()
        rank = len(shape_list)
        
        # 处理负数 axis
        axis = self.axis
        if axis < 0:
            axis += rank
            
        target_dim = shape_list[axis]
        
        # 1. 标准化：处理负索引 (idx < 0 ? idx + dim : idx)
        out = torch.where(indices < 0, indices + target_dim, indices)
        
        # 2. 合轴逻辑 (Combine Mode)
        # 根据 Tiling 代码逻辑，仅当 shape 长度为 3 且 axis 为 0 或 1 时，Combine 才会生效
        if self.combine and rank == 3:
            stride = shape_list[1]
            
            # 确保输入至少是 2D，以便获取行列信息 (参考 tiling 中 GetDim(1) 的逻辑)
            if indices.dim() >= 2:
                if axis == 0:
                    # Mode 1: dim=0 (相当于 out = indices * stride + col_idx)
                    # 构造列索引矩阵
                    cols = indices.shape[1]
                    col_indices = torch.arange(cols, device=indices.device, dtype=indices.dtype).unsqueeze(0)
                    # 广播到与 indices 相同的形状
                    col_indices = col_indices.expand_as(indices)
                    
                    out = out * stride + col_indices
                    
                elif axis == 1:
                    # Mode 2: dim=1 (相当于 out = indices + stride * row_idx)
                    # 构造行索引矩阵
                    rows = indices.shape[0]
                    row_indices = torch.arange(rows, device=indices.device, dtype=indices.dtype).unsqueeze(1)
                    # 广播到与 indices 相同的形状
                    row_indices = row_indices.expand_as(indices)
                    
                    out = out + stride * row_indices

        # 算子输出固定为 int32
        return out.to(torch.int32)

class ModelNew(nn.Module):
    def __init__(self, axis=-1, combine=False):
        super(ModelNew, self).__init__()
        self.axis = axis
        self.combine = combine

    def forward(self, indices: torch.Tensor, shape_tensor: torch.Tensor) -> torch.Tensor:
        return kernel_gen_ops.linear_index(indices, shape_tensor, self.axis, self.combine)