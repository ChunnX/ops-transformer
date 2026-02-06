from typing import List, Tuple
import torch
import torch.nn as nn
import torch_npu
from torch_npu.contrib import transfer_to_npu
import kernel_gen_ops


class Model(nn.Module):
    """
    实现EmbeddingDenseGradV2算子功能的模型（PyTorch标杆）。
    """

    def __init__(self, num_weights: int, padding_idx: int = -1, scale_grad_by_freq: bool = False):
        """
        初始化模型。
        
        Args:
            num_weights: 权重数量
            padding_idx: 填充索引
            scale_grad_by_freq: 是否按频率缩放梯度
        """
        super(Model, self).__init__()
        self.num_weights = num_weights
        self.padding_idx = padding_idx
        self.scale_grad_by_freq = scale_grad_by_freq

    def forward(self, grad: torch.Tensor, sort_indices: torch.Tensor, pos_idx: torch.Tensor) -> torch.Tensor:
        """
        实现EmbeddingDenseGrad功能。

        Args:
            grad: 输入梯度张量
            sort_indices: 排序后的索引
            pos_idx: 位置索引

        Returns:
            输出梯度张量
        """
        # 获取原始 indices（通过 pos_idx 还原）
        batch_shape = list(grad.shape[:-1])
        embedding_dim = grad.shape[-1]
        
        # 初始化输出
        output = torch.zeros(self.num_weights, embedding_dim, dtype=grad.dtype, device=grad.device)
        
        # 展平 grad
        grad_flat = grad.reshape(-1, embedding_dim)
        
        # 使用 pos_idx 获取对应的梯度
        for i in range(len(sort_indices)):
            idx = sort_indices[i].item()
            if idx == self.padding_idx:
                continue
            pos = pos_idx[i].item()
            output[idx] += grad_flat[pos]
        
        # 如果需要按频率缩放
        if self.scale_grad_by_freq:
            # 统计每个索引出现的频率
            unique_indices, counts = torch.unique(sort_indices, return_counts=True)
            for idx, count in zip(unique_indices, counts):
                if idx != self.padding_idx:
                    output[idx] = output[idx] / count.float()
        
        return output


class ModelNew(nn.Module):
    def __init__(self, num_weights: int, padding_idx: int = -1, scale_grad_by_freq: bool = False):
        super(ModelNew, self).__init__()
        self.num_weights = num_weights
        self.padding_idx = padding_idx
        self.scale_grad_by_freq = scale_grad_by_freq

    def forward(self, grad: torch.Tensor, sort_indices: torch.Tensor, pos_idx: torch.Tensor) -> torch.Tensor:
        return kernel_gen_ops.embedding_dense_grad_v2(
            grad, sort_indices, pos_idx, 
            self.num_weights, self.padding_idx, self.scale_grad_by_freq
        )