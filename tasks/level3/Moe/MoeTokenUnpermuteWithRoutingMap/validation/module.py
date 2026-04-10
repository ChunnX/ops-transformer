from typing import List, Optional
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """PyTorch native reference implementation (golden model)."""
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, permuted_tokens, sorted_indices, routing_map, probs, drop_and_pad, restore_shape):
        """
        Args:
            permuted_tokens: [num_out_tokens, hidden_size]
            sorted_indices: [total_length] - indices into permuted_tokens
            routing_map: optional [num_tokens, num_experts] bool/int8 mask
            probs: optional [num_tokens, num_experts] - expert probabilities
            drop_and_pad: bool - padded mode flag
            restore_shape: List[int] - [num_tokens] used when probs is None
        Returns:
            unpermuted_tokens: [num_tokens, hidden_size]
        """
        orig_dtype = permuted_tokens.dtype
        num_out_tokens = permuted_tokens.shape[0]
        hidden_size = permuted_tokens.shape[1]
        total_length = sorted_indices.shape[0]

        if probs is not None:
            num_tokens = probs.shape[0]
            num_experts = probs.shape[1]
            topK = num_out_tokens // num_tokens
        else:
            num_tokens = restore_shape[0] if len(restore_shape) > 0 else total_length
            topK = num_out_tokens // num_tokens

        # Compute in float32 for precision
        tokens_f = permuted_tokens.float()
        output = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=permuted_tokens.device)

        indices_cpu = sorted_indices.cpu()
        tokens_cpu = tokens_f.cpu()

        if probs is not None:
            probs_cpu = probs.float().cpu()
            if routing_map is not None:
                routing_map_cpu = routing_map.cpu()

        for i in range(num_tokens):
            for j in range(topK):
                idx = indices_cpu[i * topK + j].item()
                if idx < num_out_tokens:
                    token = tokens_cpu[idx]
                    if probs is not None:
                        # Check routing_map mask
                        if routing_map is not None:
                            if not routing_map_cpu[i, j].item():
                                continue
                        prob_val = probs_cpu[i, j].item()
                        if prob_val == 0:
                            continue
                        token = token * prob_val
                    output[i] += token

        return output.to(orig_dtype)


class ModelNew(nn.Module):
    """Custom kernel version."""
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, permuted_tokens, sorted_indices, routing_map, probs, drop_and_pad, restore_shape):
        num_out_tokens = permuted_tokens.shape[0]
        hidden_size = permuted_tokens.shape[1]
        total_length = sorted_indices.shape[0]

        if probs is not None:
            num_tokens = probs.shape[0]
            num_experts = probs.shape[1]
        else:
            num_tokens = restore_shape[0] if len(restore_shape) > 0 else total_length

        # Pre-allocate all outputs
        unpermuted_tokens = torch.zeros(num_tokens, hidden_size, dtype=permuted_tokens.dtype, device=permuted_tokens.device)
        out_index = torch.zeros(total_length, dtype=torch.int32, device=permuted_tokens.device)
        permute_token_id = torch.zeros(total_length, dtype=torch.int32, device=permuted_tokens.device)
        # permute_probs: max possible size
        if probs is not None:
            permute_probs = torch.zeros(num_tokens * num_experts, dtype=permuted_tokens.dtype, device=permuted_tokens.device)
        else:
            permute_probs = torch.zeros(0, dtype=permuted_tokens.dtype, device=permuted_tokens.device)

        return kernel_gen_ops.moe_token_unpermute_with_routing_map(
            permuted_tokens, sorted_indices, routing_map, probs,
            drop_and_pad, restore_shape,
            unpermuted_tokens, out_index, permute_token_id, permute_probs
        )
