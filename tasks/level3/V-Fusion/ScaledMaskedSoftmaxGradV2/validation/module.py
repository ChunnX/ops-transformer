from typing import List, Optional
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, yGrad: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
                scale: float = 1.0) -> torch.Tensor:
        # Replicate kernel computation: softmax_grad + scale + mask
        orig_dtype = yGrad.dtype
        yGrad_f = yGrad.to(torch.float32)
        y_f = y.to(torch.float32)

        # softmax backward: dx = y * (dy - sum(dy * y, dim=-1, keepdim=True))
        dy_mul_y = yGrad_f * y_f
        sum_dy_y = dy_mul_y.sum(dim=-1, keepdim=True)
        xGrad_f = y_f * (yGrad_f - sum_dy_y)

        # scale
        if scale != 1.0:
            xGrad_f = xGrad_f * scale

        # mask: zero out where mask is True
        mask_f = mask.to(torch.float32)
        xGrad_f = xGrad_f * (1.0 - mask_f)

        return xGrad_f.to(orig_dtype)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, yGrad: torch.Tensor, y: torch.Tensor, mask: torch.Tensor,
                scale: float = 1.0) -> torch.Tensor:
        import kernel_gen_ops
        fixed_triu_mask = False
        return kernel_gen_ops.scaled_masked_softmax_grad_v2(yGrad, y, mask, scale, fixed_triu_mask)
