import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, List
import kernel_gen_ops


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, permuted_tokens: Tensor, unpermuted_tokens_grad: Tensor,
                sorted_indices: Tensor, probs: Optional[Tensor], padded_mode: bool) -> List[Tensor]:
        if probs is not None:
            topk = probs.size(1)
            H = unpermuted_tokens_grad.size(0)
            D = unpermuted_tokens_grad.size(1)
            N = sorted_indices.size(0)
            permuted_tokens_float = permuted_tokens.float()
            unpermuted_tokens = torch.zeros(N, D, dtype=torch.float32)
            unpermuted_tokens = permuted_tokens_float.index_select(0, sorted_indices)
            unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
            probs_grad = torch.sum(unpermuted_tokens_grad.unsqueeze(1) * unpermuted_tokens, -1, keepdim=False)

            permuted_tokens_grad = (unpermuted_tokens_grad.unsqueeze(1) * probs.unsqueeze(-1)).reshape(-1, D)
            permuted_tokens_grad1 = torch.zeros(N, D, dtype=torch.float32)
            permuted_tokens_grad1.index_add_(0, sorted_indices, permuted_tokens_grad.to(torch.float32))

            return permuted_tokens_grad1.to(permuted_tokens.dtype), probs_grad.to(probs.dtype)
        else:
            topk = 1
            D = unpermuted_tokens_grad.size(1)
            N = sorted_indices.size(0)
            permuted_tokens_float = permuted_tokens.float()
            unpermuted_tokens = torch.zeros(N, D, dtype=torch.float32)
            unpermuted_tokens = permuted_tokens_float.index_select(0, sorted_indices)
            if permuted_tokens.size(-1) == 0:
                unpermuted_tokens = unpermuted_tokens.reshape(sorted_indices.shape[0] // topk, topk, permuted_tokens.size(-1))
            else:
                unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
            permuted_tokens_grad1 = torch.zeros(N, D, dtype=torch.float32)
            permuted_tokens_grad1.index_copy_(0, sorted_indices.to(torch.int64), unpermuted_tokens_grad.to(torch.float32))
            return permuted_tokens_grad1.to(permuted_tokens.dtype)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, permuted_tokens: Tensor, unpermuted_tokens_grad: Tensor,
                sorted_indices: Tensor, probs: Optional[Tensor], padded_mode: bool) -> List[Tensor]:
        """
        调用 NPU 算子
        """
        permuted_tokens_grad, probs_grad = kernel_gen_ops.moe_token_unpermute_grad(
            permuted_tokens, unpermuted_tokens_grad, sorted_indices, probs, padded_mode
        )
        if probs is not None:
            return permuted_tokens_grad, probs_grad
        else:
            return permuted_tokens_grad
