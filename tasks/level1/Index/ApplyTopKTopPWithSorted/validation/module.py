from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import torch.nn.functional as F
import kernel_gen_ops


class Model(nn.Module):
    """
    实现ApplyTopKTopPWithSorted算子功能的模型（PyTorch标杆）。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(
        self, 
        sorted_value: torch.Tensor, 
        sorted_indices: torch.Tensor,
        p: Optional[torch.Tensor] = None,
        k: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        实现ApplyTopKTopPWithSorted算子功能。

        Args:
            sorted_value: 已排序的概率值张量 [batch, vocab]
            sorted_indices: 对应的索引张量 [batch, vocab]
            p: Top-P阈值 [batch] (可选)
            k: Top-K数量 [batch] (可选)

        Returns:
            输出张量，满足top-k/top-p条件的保留原值，其余位置为-inf
        """
            
        # Top-K过滤
        if not k.dim() == 0:
            kth_idx = sorted_value.size(1) - k.long()
            kth_value = sorted_value.gather(1, kth_idx.unsqueeze(dim=1))
            top_k_mask = sorted_value < kth_value
            sorted_value.masked_fill_(top_k_mask, -float("inf"))
        # Top-P过滤
        if not p.dim() == 0:
            softmax_res = sorted_value.to(float32).softmax(dim=-1)
            cumsum_res = softmax_res.cumsum(dim=-1)
            top_p_mask = cumsum_res <= 1 - p.unsqueeze(dim=1)
            top_p_mask[:, -1] = False
            sorted_value.masked_fill_(top_p_mask, -float("inf"))
        out = torch.empty_like(sorted_value).scatter_(dim=-1, index=sorted_indices.long(), src=sorted_value)
        return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(
        self, 
        sorted_value: torch.Tensor, 
        sorted_indices: torch.Tensor,
        p: Optional[torch.Tensor] = None,
        k: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return kernel_gen_ops.apply_top_k_top_p_with_sorted(sorted_value, sorted_indices, p, k)