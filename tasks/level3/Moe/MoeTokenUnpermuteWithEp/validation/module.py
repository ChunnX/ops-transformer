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

    def forward(self, permuted_tokens, sorted_indices, probs, num_topk, range_start, range_end):
        """
        Args:
            permuted_tokens: [num_local_tokens, hidden_size] - tokens in the local EP partition
            sorted_indices: [num_tokens * top_k] - indices into global token positions
            probs: [num_tokens, top_k] or None - expert probabilities
            num_topk: number of top-k experts
            range_start: start of EP range
            range_end: end of EP range
        Returns:
            unpermuted_tokens: [num_tokens, hidden_size]
        """
        orig_dtype = permuted_tokens.dtype
        num_total = sorted_indices.shape[0]
        num_tokens = num_total // num_topk
        hidden_size = permuted_tokens.shape[1]

        # Compute in float32 for precision
        tokens_f = permuted_tokens.float()
        output = torch.zeros(num_tokens, hidden_size, dtype=torch.float32, device=permuted_tokens.device)

        indices_cpu = sorted_indices.cpu()
        tokens_cpu = tokens_f.cpu()

        if probs is not None:
            probs_cpu = probs.float().cpu()

        for i in range(num_tokens):
            for j in range(num_topk):
                idx = indices_cpu[i * num_topk + j].item()
                if range_start <= idx < range_end:
                    token = tokens_cpu[idx - range_start]
                    if probs is not None:
                        prob_val = probs_cpu[i, j].item()
                        token = token * prob_val
                    output[i] += token

        return output.to(orig_dtype)


class ModelNew(nn.Module):
    """Custom kernel version."""
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, permuted_tokens, sorted_indices, probs, num_topk, range_start, range_end):
        num_total = sorted_indices.shape[0]
        num_tokens = num_total // num_topk
        hidden_size = permuted_tokens.shape[1]

        # Prepare output tensor
        out = torch.zeros(num_tokens, hidden_size, dtype=permuted_tokens.dtype, device=permuted_tokens.device)

        # Prepare range list
        range_list = [range_start, range_end]

        return kernel_gen_ops.moe_token_unpermute_with_ep(
            permuted_tokens, sorted_indices, probs,
            num_topk, range_list, out
        )
