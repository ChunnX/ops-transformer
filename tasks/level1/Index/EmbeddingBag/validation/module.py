from typing import List, Optional, Tuple

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """
    实现EmbeddingBag算子功能的模型（PyTorch标杆）。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(
        self, 
        weight: torch.Tensor, 
        indices: torch.Tensor, 
        offsets: torch.Tensor,
        per_sample_weights: Optional[torch.Tensor],
        mode: str,
        scale_grad_by_freq: bool,
        sparse: bool,
        include_last_offset: bool,
        padding_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        实现EmbeddingBag算子功能。

        Args:
            weight: 嵌入权重矩阵 (num_embeddings, embedding_dim)
            indices: 索引张量
            offsets: 偏移量张量
            per_sample_weights: 可选的每样本权重
            mode: 聚合模式 ('sum', 'mean', 'max')
            scale_grad_by_freq: 是否按频率缩放梯度
            sparse: 是否使用稀疏梯度
            include_last_offset: offsets是否包含最后一个偏移
            padding_idx: 填充索引

        Returns:
            (y, offset2bag, bag_size, max_indices)
        """
        # 使用 torch.nn.functional.embedding_bag
        num_bags = offsets.size(0)
        if include_last_offset:
            num_bags -= 1
        
        # PyTorch的embedding_bag返回4个输出
        y, offset2bag, bag_size, max_indices = torch.ops.aten._embedding_bag_forward_only(
            weight, 
            indices, 
            offsets,
            scale_grad_by_freq,
            0 if mode == 'sum' else (1 if mode == 'mean' else 2),  # mode转换
            sparse,
            per_sample_weights,
            include_last_offset,
            padding_idx
        )
        
        return y, offset2bag, bag_size, max_indices


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(
        self, 
        weight: torch.Tensor, 
        indices: torch.Tensor, 
        offsets: torch.Tensor,
        per_sample_weights: Optional[torch.Tensor],
        mode: str,
        scale_grad_by_freq: bool,
        sparse: bool,
        include_last_offset: bool,
        padding_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return kernel_gen_ops.custom_pybind_api(
            weight, 
            indices, 
            offsets, 
            per_sample_weights,
            mode,
            scale_grad_by_freq,
            sparse,
            include_last_offset,
            padding_idx
        )