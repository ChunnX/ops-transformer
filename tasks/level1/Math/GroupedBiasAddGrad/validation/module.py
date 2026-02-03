import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional
import kernel_gen_ops
from typing import List


class Model(nn.Module):
    def __init__(self, grad_y: Tensor, group_idx: Tensor):
        super(Model, self).__init__()
        self.grad_y = grad_y
        self.group_idx = group_idx

    def forward(self, grad_y: Tensor, group_idx: Tensor) -> Tensor:
        """
        CPU 实现 grouped_bias_add_grad，与 golden 一致
        Args:
            grad_y: shape [C, H] 或 [G, C, H]，dtype: float32/float16/bfloat16
            group_idx: shape [G]，int32/int64，可选

        Returns:
            out: shape [G, H]，dtype 与 grad_y 一致
        """
        if group_idx is None:
            # 无可选输入：out = sum(grad_y, dim=1)
            grad_y = grad_y.float()  # 统一转 float32
            out = torch.sum(grad_y, dim=1)
            return out.to(grad_y.dtype)
        else:
            # 有可选输入：分组累加
            group_idx = group_idx.int()
            grad_y = grad_y.float()
            out = torch.tensor([]).to(grad_y.device)

            for i in range(len(group_idx)):
                start = group_idx[i-1] if i > 0 else 0
                end = group_idx[i]
                chunk = grad_y[start:end, :]
                tmp = torch.sum(chunk, dim=0, keepdim=True)
                out = torch.cat((out, tmp), dim=0)

            return out.to(grad_y.dtype)


class ModelNew(nn.Module):
    def __init__(self, grad_y: Tensor, group_idx: Tensor):
        super(ModelNew, self).__init__()
        self.grad_y = grad_y
        self.group_idx = group_idx

    def forward(self, grad_y: Tensor, group_idx: Tensor) -> List[torch.Tensor]:
        """
        调用 NPU 算子
        """
        return kernel_gen_ops.grouped_bias_add_grad(grad_y, group_idx)