from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn.functional as F
import kernel_gen_ops


def _ref_masked_softmax_with_rel_pos_bias(
    x: torch.Tensor,
    atten_mask: Optional[torch.Tensor],
    relative_pos_bias: torch.Tensor,
    scale_value: float = 1.0,
) -> torch.Tensor:
    """Reference: out = softmax(scale_value * x + atten_mask + relative_pos_bias)."""
    scaled = x * scale_value
    out = scaled + relative_pos_bias
    if atten_mask is not None:
        # broadcast atten_mask to same shape as out if needed
        out = out + atten_mask
    return F.softmax(out, dim=-1)


class Model(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        atten_mask: Optional[torch.Tensor],
        relative_pos_bias: torch.Tensor,
        scale_value: float = 1.0,
    ) -> List[torch.Tensor]:
        y = _ref_masked_softmax_with_rel_pos_bias(x, atten_mask, relative_pos_bias, scale_value)
        return [y]


class ModelNew(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        atten_mask: Optional[torch.Tensor],
        relative_pos_bias: torch.Tensor,
        scale_value: float = 1.0,
    ) -> List[torch.Tensor]:
        y = kernel_gen_ops.masked_softmax_with_rel_pos_bias(
            x, atten_mask, relative_pos_bias, scale_value
        )
        return [y]
