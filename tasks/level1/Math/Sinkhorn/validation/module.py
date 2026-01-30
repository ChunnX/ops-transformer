from typing import List, Tuple

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops

class Model(nn.Module):
    def __init__(self, cost: torch.Tensor, tol: float = 0.0001):
        super(Model, self).__init__()
        self.cost = cost
        self.tol = tol

    def forward(self, cost: torch.Tensor, tol: float = 0.0001) -> torch.Tensor:
        """
        手动实现 Sinkhorn 算法（CPU）
        Args:
            cost: 输入成本矩阵 [S*B, num_experts]，dtype: float16/bf16/f32
            tol: 收敛容差，float32

        Returns:
            p: 输出运输方案，shape=[S*B, num_experts]
        """
        golden_p = self.sinkhorn(cost, tol)
        return golden_p

    # magetron提供的sinkhorn作为golden
    def sinkhorn(self, cost: torch.Tensor, tol: float = 0.0001):
        """Sinkhorn based MoE routing function"""
        cost = torch.exp(cost)
        d0 = torch.ones(cost.size(0), device=cost.device, dtype=cost.dtype)
        d1 = torch.ones(cost.size(1), device=cost.device, dtype=cost.dtype)

        eps = 0.00000001
        error = 1e9
        d1_old = d1
        while error > tol:
            d0 = (1 / d0.size(0)) * 1 / (torch.sum(d1 * cost, 1) + eps)
            d1 = (1 / d1.size(0)) * 1 / (torch.sum(d0.unsqueeze(1) * cost, 0) + eps)
            error = torch.mean(torch.abs(d1_old - d1))
            d1_old = d1
        return d1 * cost * d0.unsqueeze(1)

class ModelNew(nn.Module):
    def __init__(self, cost: torch.Tensor, tol: float = 0.0001):
        super(ModelNew, self).__init__()
        self.cost = cost
        self.tol = tol

    def forward(self, cost: torch.Tensor, tol: float = 0.0001) -> torch.Tensor:
        """
        调用 NPU 自定义算子
        Args:
            cost: 输入张量，shape=[S*B, num_experts]
            tol: 收敛容差，float32

        Returns:
            p: 输出张量，shape=[S*B, num_experts]
        """
        # tol 必须是 float32
        if not isinstance(tol, torch.Tensor):
            tol = torch.tensor(tol, dtype=torch.float32, device=cost.device)

        # 调用算子
        print(f"***lyy-ModuleNew调用算子开始, cost is {cost}, tol is {tol}")
        # p = kernel_gen_ops.sinkhorn(cost, tol)
        return kernel_gen_ops.sinkhorn(cost, tol)