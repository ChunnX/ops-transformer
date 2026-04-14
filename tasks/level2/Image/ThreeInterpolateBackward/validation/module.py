from typing import List
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """使用 PyTorch 原生算子的参考实现（golden model）。

    ThreeInterpolateBackward: 反向传播算子。
    给定 grad_x (B, C, N), idx (B, N, 3), weight (B, N, 3)，
    计算 grad_y (B, C, M)，其中 M 由属性 m 指定。

    计算公式:
        grad_y[b, c, idx[b, n, k]] += weight[b, n, k] * grad_x[b, c, n]
        对所有 n in [0, N) 和 k in [0, 3)
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, grad_x, idx, weight, m):
        B, C, N = grad_x.shape
        grad_y = torch.zeros(B, C, m, dtype=grad_x.dtype, device=grad_x.device)

        for k in range(3):
            idx_k = idx[:, :, k].long().unsqueeze(1).expand(-1, C, -1)  # (B, C, N)
            weight_k = weight[:, :, k].unsqueeze(1)  # (B, 1, N)
            contrib = grad_x * weight_k  # (B, C, N)
            grad_y.scatter_add_(2, idx_k, contrib)

        return grad_y


class ModelNew(nn.Module):
    """自定义 kernel 版本。

    算子 kernel 内部按 NC1HWC0 布局处理数据（C 方向按 16 对齐分组）。
    在 Python 侧将 ND (B, C, N) 重排为 (B, C1, N, 1, C0) 再传入，
    输出同理反向重排回 (B, C, M)。
    """

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, grad_x, idx, weight, m):
        B, C, N = grad_x.shape
        C0 = 16
        C1 = C // C0

        # Process each C1 group separately (kernel does not handle C1 > 1 correctly)
        results = []
        for c1 in range(C1):
            chunk = grad_x[:, c1*C0:(c1+1)*C0, :]  # (B, C0, N)
            # (B, C0, N) → (B, 1, C0, N) → permute → (B, 1, N, C0) → unsqueeze → (B, 1, N, 1, C0)
            chunk_5d = chunk.reshape(B, 1, C0, N).permute(0, 1, 3, 2).unsqueeze(3).contiguous()
            result_5d = kernel_gen_ops.three_interpolate_backward(chunk_5d, idx, weight, m)
            # (B, 1, m, 1, C0) → squeeze → (B, 1, m, C0) → permute → (B, 1, C0, m) → reshape → (B, C0, m)
            result_chunk = result_5d.squeeze(3).permute(0, 1, 3, 2).reshape(B, C0, m)
            results.append(result_chunk)

        return torch.cat(results, dim=1)  # (B, C, m)
