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
        batch_size, vocab_size = sorted_value.shape
        device = sorted_value.device
        dtype = sorted_value.dtype
        
        # 初始化输出为-inf
        out = torch.full_like(sorted_value, float('-inf'))
        
        for b in range(batch_size):
            # 获取当前batch的值和索引
            values = sorted_value[b].float()
            indices = sorted_indices[b]
            
            # 计算softmax（已排序，最大值在末尾）
            max_val = values[-1]
            exp_values = torch.exp(values - max_val)
            softmax_probs = exp_values / exp_values.sum()
            
            # 计算累积概率
            cumsum_probs = torch.cumsum(softmax_probs, dim=0)
            
            # 确定掩码
            mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
            
            # Top-K过滤
            if k is not None:
                k_val = k[b].item() if k.dim() > 0 else k.item()
                k_threshold_idx = vocab_size - k_val
                mask[:k_threshold_idx] = False
            
            # Top-P过滤
            if p is not None:
                p_val = p[b].item() if p.dim() > 0 else p.item()
                p_threshold = 1.0 - p_val
                # 从后往前找第一个累积概率<=p_threshold的位置
                valid_mask = cumsum_probs > p_threshold
                if valid_mask.any():
                    mask = mask & valid_mask
            
            # 应用掩码：scatter回原始索引位置
            valid_values = sorted_value[b][mask]
            valid_indices = indices[mask]
            out[b, valid_indices] = valid_values
        
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