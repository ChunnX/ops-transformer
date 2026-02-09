from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn



class Model(nn.Module):
    """
    实现概率到对数几率转换（log-odds)。
    """

    def __init__(self):
        """
        初始化模型。

        Args:
        """
        super(Model, self).__init__()

    def forward(self, x_tensor, dy_tensor, eps=None):
        """
        使用torch.special.logit + backward功能实现，获取标杆结果。

        Args:
            x_tensor: 输入张量，数值在(0-1)之间
            dy_tensor: 正向输出结果的梯度
            eps: double 输入x_tensor的epsilon限制边界，建议值-1

        Returns:
            out: dx, 输出张量，与输入数据类型、shape与输入x_tensor一致
        """
        x_tensor = x_tensor.requires_grad_(True)
        y = torch.special.logit(x_tensor, eps)
        y.backward(dy_tensor)
        return x_tensor.grad

class ModelNew(nn.Module):
    def __init__(self):
        """
        初始化模型。

        Args:
        """
        super(ModelNew, self).__init__()

    def forward(self, x_tensor, dy_tensor, eps=None):
        """
        使用自定义kernel实现Logit功能。

        Args:
            x_tensor: 输入张量
            dy_tensor: 正向输出结果的梯度
            eps: double 输入input的epsilon限制边界，建议值-1

        Returns:
            out: 转换后结果张量
        """
        import kernel_gen_ops
        return kernel_gen_ops.logit_grad(x_tensor, dy_tensor, eps)