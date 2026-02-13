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

    def forward(self, input_tensor, eps=None):
        """
        使用torch.logit功能实现，获取标杆结果。

        Args:
            input_tensor: 输入张量
            eps: double 输入input的epsilon限制边界，建议值-1

        Returns:
            out: 转换后结果张量
        """
        return torch.logit(input_tensor, eps)


class ModelNew(nn.Module):
    def __init__(self):
        """
        初始化模型。

        Args:
        """
        super(ModelNew, self).__init__()

    def forward(self, input_tensor, eps=None):
        """
        使用自定义kernel实现Logit功能。

        Args:
            input_tensor: 输入张量
            eps: double 输入input的epsilon限制边界，建议值-1

        Returns:
            out: 转换后结果张量
        """
        import kernel_gen_ops
        return kernel_gen_ops.logit(input_tensor, eps)