import torch
import torch_npu
import numpy as np
import torch.nn as nn


class Model(nn.Module):
    """
    CPU golden model for MoeComputeExpertTokens operator.
    
    Function: Calculate total token count before each expert based on sorted expert indices.
    """

    def __init__(self):
        """
        Initialize model.
        """
        super(Model, self).__init__()

    def forward(self, sorted_experts: torch.Tensor, num_experts: int) -> torch.Tensor:
        """
        CPU golden implementation: Calculate total token count before each expert.
        
        Args:
            sorted_experts: Sorted expert indices, shape [num_tokens], dtype int32.
            num_experts: Number of experts.
        
        Returns:
            total_rows_before_expert: Total token count before each expert, 
                                      shape [num_experts], dtype int32.
        """
        sorted_experts_np = sorted_experts.cpu().numpy()
        num_experts = int(num_experts)
        arr_length = sorted_experts_np.shape[-1]
        res = np.arange(num_experts)
        for i in range(num_experts):
            target = i
            low = 0
            high = arr_length - 1
            target_location = -1
            while low <= high:
                mid = (low + high) // 2
                if sorted_experts_np[mid] > target:
                    high = mid - 1
                else:
                    low = mid + 1
                    target_location = mid
            res[i] = target_location + 1
        res = res.astype(np.int32)
        return torch.from_numpy(res)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, sorted_experts: torch.Tensor, num_experts: int) -> torch.Tensor:
        """
        NPU implementation: Call kernel_gen_ops.moe_compute_expert_tokens.
        """
        import kernel_gen_ops
        
        # Get result from NPU kernel
        result = kernel_gen_ops.moe_compute_expert_tokens(sorted_experts, num_experts)
        return result
