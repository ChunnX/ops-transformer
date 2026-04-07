from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, expandedX: torch.Tensor, expandedRowIdx: torch.Tensor, x1Optional: torch.Tensor, 
                x2Optional: torch.Tensor, biasOptional: torch.Tensor, scalesOptional: torch.Tensor, 
                expertIdxOptional: torch.Tensor, dropPadMode: int) -> torch.Tensor:
        if len(expandedX.shape) == 2:
            num_rows = expertIdxOptional.shape[0]
            k = expertIdxOptional.shape[1]
            hidden_size = expandedX.shape[-1]
            output = torch.zeros(num_rows, hidden_size, device=expandedX.device, dtype=expandedX.dtype)
            
            # Check if bias is provided
            has_bias = biasOptional.numel() > 0
            
            for i in range(num_rows):
                temp_sum = torch.zeros(hidden_size, device=expandedX.device, dtype=expandedX.dtype)
                for j in range(k):
                    temp_sum += scalesOptional[i, j] * expandedX[expandedRowIdx[i + j * num_rows]]
                    
                    # Add bias if provided
                    if has_bias:
                        expert_id = expertIdxOptional[i, j].item()
                        temp_sum += scalesOptional[i, j] * biasOptional[expert_id]
                
                output[i] = x1Optional[i] + x2Optional[i] + temp_sum
            
            return output

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, expandedX: torch.Tensor, expandedRowIdx: torch.Tensor, x1Optional: torch.Tensor, 
                x2Optional: torch.Tensor, biasOptional: torch.Tensor, scalesOptional: torch.Tensor, 
                expertIdxOptional: torch.Tensor, dropPadMode: int) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.moe_finalize_routing_v2(expandedX, expandedRowIdx, x1Optional, x2Optional, 
                                                   biasOptional, scalesOptional, expertIdxOptional, dropPadMode)