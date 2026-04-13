import torch
import torch_npu
import numpy as np

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, indices: torch.Tensor, probs: torch.Tensor, range_vals: list, num_token_out: int, pad_mode: bool):
        # Reference implementation using PyTorch operations based on the provided CPU benchmark
        if indices.dim() == 1:
            topk = 1
        else:
            topk = indices.size(1)
        
        # Flatten indices and sort them
        flatten_indices = indices.view(-1)
        sorted_indices = torch.argsort(flatten_indices.float(), stable=True)
        sorted_indices1 = torch.argsort(sorted_indices.float(), stable=True)
        sorted_indices1 = sorted_indices1.to(torch.int32)

        # Apply range if provided
        if range_vals is not None:
            start = range_vals[0]
            end = range_vals[1]
            sorted_indices_sliced = sorted_indices[start:end]
        else:
            sorted_indices_sliced = sorted_indices

        # Expand tokens based on sorted indices
        permuted_tokens = x.index_select(0, sorted_indices_sliced // topk)

        # Expand probs if provided
        if probs is not None:
            flatten_probs = probs.view(-1)
            permuted_probs = flatten_probs.index_select(0, sorted_indices_sliced)
        else:
            permuted_probs = torch.empty(0, device=x.device, dtype=probs.dtype) if probs is not None else None

        return [
            permuted_tokens,
            sorted_indices1,
            permuted_probs
        ]

class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor, indices: torch.Tensor, probs: torch.Tensor, range_vals: list, num_token_out: int, pad_mode: bool):
        import kernel_gen_ops
        return list(kernel_gen_ops.moe_token_permute_with_ep(x, indices, probs, range_vals, num_token_out, pad_mode))