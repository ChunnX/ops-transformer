"""
CPU golden 与 ops-nn-dev ST 对齐：
`ops-nn-dev/quant/dequant_bias/tests/st/aclnnDequantBias/executor_aclnnDequantBias.py`
"""

from typing import List, Optional

import torch
import torch.nn as nn

GE_DT_FP16 = 1
GE_DT_BF16 = 27


def _golden_dequant_bias(
    x: torch.Tensor,
    weight_scale: torch.Tensor,
    activate_scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    output_dtype: int,
) -> torch.Tensor:
    """复刻 executor_aclnnDequantBias.__call__ 分支逻辑。"""
    x_data = x
    weight_scale_data = weight_scale
    activate_scale_data = activate_scale
    bias_data = bias

    if bias_data is not None:
        if bias_data.dtype == torch.int32:
            if activate_scale_data is None:
                y = (x_data.to(torch.float32) + bias_data.to(torch.float32)) * weight_scale_data.to(torch.float32)
            else:
                a = activate_scale_data.to(torch.float32)[:, None]
                y = (x_data.to(torch.float32) + bias_data.to(torch.float32)) * weight_scale_data.to(
                    torch.float32
                ) * a
        elif bias_data.dtype in (torch.float16, torch.bfloat16, torch.float32):
            if activate_scale_data is None:
                y = x_data.to(torch.float32) * weight_scale_data.to(torch.float32) + bias_data.to(torch.float32)
            else:
                a = activate_scale_data.to(torch.float32)[:, None]
                y = (
                    x_data.to(torch.float32) * weight_scale_data.to(torch.float32) * a
                    + bias_data.to(torch.float32)
                )
        else:
            raise ValueError(f"Unsupported bias dtype: {bias_data.dtype}")
    else:
        if activate_scale_data is None:
            y = x_data.to(torch.float32) * weight_scale_data.to(torch.float32)
        else:
            a = activate_scale_data.to(torch.float32)[:, None]
            y = x_data.to(torch.float32) * weight_scale_data.to(torch.float32) * a

    if output_dtype == GE_DT_FP16:
        return y.to(torch.float16)
    if output_dtype == GE_DT_BF16:
        return y.to(torch.bfloat16)
    raise ValueError(f"golden 仅支持 output_dtype 1(fp16) 或 27(bf16)，收到 {output_dtype}")


class Model(nn.Module):
    def __init__(self, output_dtype: int):
        super().__init__()
        self.output_dtype = int(output_dtype)

    def forward(
        self,
        x: torch.Tensor,
        weight_scale: torch.Tensor,
        activate_scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        y = _golden_dequant_bias(x, weight_scale, activate_scale, bias, self.output_dtype)
        return [y]


class ModelNew(nn.Module):
    def __init__(self, output_dtype: int):
        super().__init__()
        self.output_dtype = int(output_dtype)

    def forward(
        self,
        x: torch.Tensor,
        weight_scale: torch.Tensor,
        activate_scale: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        import kernel_gen_ops

        return kernel_gen_ops.dequant_bias(x, weight_scale, activate_scale, bias, self.output_dtype)
