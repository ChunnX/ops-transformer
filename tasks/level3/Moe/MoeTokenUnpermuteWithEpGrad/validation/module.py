from typing import List, Tuple, Optional
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """PyTorch 原生算子参考实现（golden model）。"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        unpermuted_tokens_grad: torch.Tensor,
        sorted_indices: torch.Tensor,
        permuted_tokens: Optional[torch.Tensor],
        probs: Optional[torch.Tensor],
        padded_mode: bool,
        restore_shape: List[int],
        range_vals: List[int],
        topk_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens_num = unpermuted_tokens_grad.shape[0]
        hidden_size = unpermuted_tokens_grad.shape[1]

        # Determine range
        total_indices = sorted_indices.shape[0]
        start = range_vals[0] if range_vals[0] >= 0 else 0
        end = range_vals[1] if range_vals[1] >= 0 else total_indices

        # Output tensors
        permuted_tokens_grad = torch.zeros(
            total_indices, hidden_size, dtype=unpermuted_tokens_grad.dtype, device=unpermuted_tokens_grad.device
        )

        has_probs = probs is not None and probs.numel() > 0

        if not has_probs:
            # probs is None: directly scatter unpermuted_tokens_grad to permuted_tokens_grad
            for i in range(total_indices):
                idx = sorted_indices[i].item()
                if start <= idx < end:
                    token_idx = i // topk_num
                    permuted_tokens_grad[idx - start] = unpermuted_tokens_grad[token_idx]
            probs_grad = torch.zeros(
                tokens_num, topk_num, dtype=unpermuted_tokens_grad.dtype, device=unpermuted_tokens_grad.device
            )
        else:
            # probs is not None
            topk = probs.shape[1] if probs.dim() == 2 else topk_num
            probs_grad = torch.zeros_like(probs)

            for i in range(total_indices):
                idx = sorted_indices[i].item()
                token_idx = i // topk
                k_idx = i % topk

                if start <= idx < end:
                    # permuted_tokens_grad
                    permuted_tokens_grad[idx - start] = (
                        unpermuted_tokens_grad[token_idx] * probs[token_idx, k_idx]
                    )

            # probs_grad: sum over hidden_size of (permuted_tokens * unpermuted_tokens_grad)
            # for each (token_idx, k_idx) pair
            if permuted_tokens is not None and permuted_tokens.numel() > 0:
                for i in range(total_indices):
                    idx = sorted_indices[i].item()
                    token_idx = i // topk
                    k_idx = i % topk

                    if start <= idx < end:
                        # probs_grad[token_idx, k_idx] = sum(permuted_tokens[idx-start] * unpermuted_tokens_grad[token_idx])
                        prod = permuted_tokens[idx - start].float() * unpermuted_tokens_grad[token_idx].float()
                        probs_grad[token_idx, k_idx] = prod.sum().to(probs.dtype)

        return permuted_tokens_grad, probs_grad


class ModelNew(nn.Module):
    """自定义 kernel 版本。"""

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(
        self,
        unpermuted_tokens_grad: torch.Tensor,
        sorted_indices: torch.Tensor,
        permuted_tokens: Optional[torch.Tensor],
        probs: Optional[torch.Tensor],
        padded_mode: bool,
        restore_shape: List[int],
        range_vals: List[int],
        topk_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return kernel_gen_ops.moe_token_unpermute_with_ep_grad(
            unpermuted_tokens_grad,
            sorted_indices,
            permuted_tokens,
            probs,
            padded_mode,
            restore_shape,
            range_vals,
            topk_num,
        )
