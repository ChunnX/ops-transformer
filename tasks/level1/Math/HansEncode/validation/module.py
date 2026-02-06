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
        input_tensor: Tensor,
        pdf_ref: Tensor,
        statistic: bool = True,
        reshuff: bool = False
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        使用torch实现压缩做对比标杆。
        
        参数:
            input_tensor (Tensor): shape支持多维，要求数据数量是32768的倍数，dtype=float16/bfloat16/f32
            pdf_ref (Tensor): shape [1, 256]，dtype=int32
            statistic (bool): 是否进行 PDF 统计，默认为True
            reshuff (bool): 是否对各核编码后的结果进行内存重整
            
        返回:
            pdf_out（Tensor）: [1, 256], int32 指数位统计结果
            mantissa_out（Tensor）: 表示输出的尾数部分，dtype与input一致
            fixed_out（Tensor）: 表示压缩的第一段输出，dtype与input一致
            var_out（Tensor）: 表示压缩超过fixedOut后的输出，dtype与input一致
        """
        # 1. 转为 CPU 并确保内存连续（为后续 view 做准备）
        input_cpu = input_tensor.cpu().contiguous()

        # 2. 获取 dtype
        dtype = input_cpu.dtype

        # 3. 根据 dtype 提取指数位（exponent bits）
        # 以 uint8 视图访问字节，提取指定字节
        if dtype == torch.float32:
            # float32: 指数位在第 4 字节（索引 3）
            # view 为 uint8，reshape 为 [N, 4]，取第 3 个字节（索引 3）
            exp_bytes = input_cpu.view(torch.uint8).view(-1, 4)[:, 3]
        elif dtype in [torch.float16, torch.bfloat16]:
            # float16 / bfloat16: 指数位在第 2 字节（索引 1）
            # view 为 uint8，reshape 为 [N, 2]，取第 1 个字节（索引 1）
            exp_bytes = input_cpu.view(torch.uint8).view(-1, 2)[:, 1]
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        # 4. 统计 PDF：使用 torch.bincount（原生支持）
        # minlength=256，dtype=int32
        pdf_out = torch.bincount(exp_bytes, minlength=256).to(torch.int32)

        # 5. 转回 NPU 设备（如需）
        pdf_out_tensor = pdf_out.to(input_tensor.device)

        return pdf_out_tensor

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input_tensor: Tensor, pdf_ref: Tensor, statistic: bool = True, reshuff: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        调用 NPU 算子
        参数:
            input_tensor (Tensor): shape支持多维，要求数据数量是32768的倍数，dtype=float16/bfloat16/f32
            pdf_ref (Tensor): shape [1, 256]，dtype=int32
            statistic (bool): 是否进行 PDF 统计，默认为True
            reshuff (bool): 是否对各核编码后的结果进行内存重整
            
        返回:
            pdf_out（Tensor）: [1, 256], int32 指数位统计结果
            mantissa_out（Tensor）: 表示输出的尾数部分，dtype与input一致
            fixed_out（Tensor）: 表示压缩的第一段输出，dtype与input一致
            var_out（Tensor）: 表示压缩超过fixedOut后的输出，dtype与input一致
        """

        # 调用 NPU 算子
        pdf_out, mantissa_out, fixed_out, var_out = kernel_gen_ops.hans_encode(
            input_tensor, pdf_ref, statistic, reshuff
        )

        return pdf_out