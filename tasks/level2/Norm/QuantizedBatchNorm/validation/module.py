from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import kernel_gen_ops

class Model(nn.Module):
    def __init__(self, num_features: int, eps: float):
        super(Model, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor,
                input_scale: torch.Tensor, input_zero_point: torch.Tensor,
                output_scale: torch.Tensor, output_zero_point: torch.Tensor,
                weight: Optional[torch.Tensor], bias: Optional[torch.Tensor]) -> torch.Tensor:
        
        # Dequantize input
        x_fp32 = (x.float() - input_zero_point.item()) * input_scale.item()
        
        # Normalize: (x - mean) / sqrt(var + eps)
        normalized = (x_fp32 - mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)) / torch.sqrt(var.unsqueeze(0).unsqueeze(2).unsqueeze(3) + self.eps)
        
        # Apply affine transformation if provided
        if weight is not None:
            normalized = normalized * weight.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        if bias is not None:
            normalized = normalized + bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        
        # Quantize output
        y_fp32 = normalized / output_scale.item() + output_zero_point.item()
        y = y_fp32.round().clamp(-128, 127).to(x.dtype)
        
        return y


class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor, mean: torch.Tensor, var: torch.Tensor,
                input_scale: torch.Tensor, input_zero_point: torch.Tensor,
                output_scale: torch.Tensor, output_zero_point: torch.Tensor,
                weight: Optional[torch.Tensor], bias: Optional[torch.Tensor]) -> torch.Tensor:
        return kernel_gen_ops.quantized_batch_norm(
            x, mean, var, input_scale, input_zero_point, output_scale, output_zero_point,
            weight, bias, self.eps
        )
