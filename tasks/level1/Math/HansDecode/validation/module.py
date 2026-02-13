import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple, Optional
import kernel_gen_ops
from typing import Dict, Any
import torch_npu
import numpy as np


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        input_tensor: torch.Tensor,
        mantissa: torch.Tensor,
        fixed: torch.Tensor,
        var: torch.Tensor,
        hist: torch.Tensor,
        reshuff: bool = False
    ) -> torch.Tensor:
        """
        使用 torch 实现 Hans 解码，还原 input_tensor

        Args:
            mantissa: [1, M], 尾数部分, float32/f16/bf16
            fixed: [1, F], 压缩后的固定部分, float32/f16/bf16
            var: [1, V],未压缩部分, float32/f16/bf16
            hist: [1, 256], int32,指数位统计
            reshuff: 是否启用内存重整（占位参数）

        Returns:
            input_tensor: [1, N], float32/f16/bf16, 原始输入
        """

        return input_tensor

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(
        self,
        input_tensor: torch.Tensor,
        mantissa: torch.Tensor,
        fixed: torch.Tensor,
        var: torch.Tensor,
        hist: torch.Tensor,
        reshuff: bool = False
    ) -> torch.Tensor:
        """
        完整编解码流程：
        1. 调用 hans_encode 对 input_tensor 压缩
        2. 将编码输出（mantissa, fixed, var, pdf）作为 hans_decode 输入
        3. 返回解码后的最终结果

        Args:
            input_tensor: 原始输入张量
            pdf_ref: 初始 PDF 参考值（[1, 256], int32），用于编码
            statistic: 是否统计 PDF
            reshuff: 是否内存重整

        Returns:
            output: 解码后结果，应与 input_tensor 一致（数值误差内）
        """
        if fixed.dim() == 1:
            fixed = fixed.unsqueeze(0)
        if var.dim() == 1:
            var = var.unsqueeze(0)
        if mantissa.dim() == 1:
            mantissa = mantissa.unsqueeze(0)
        output = kernel_gen_ops.hans_decode(
            mantissa, fixed, var, hist, reshuff
        )

        return output