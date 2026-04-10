import torch
import kernel_gen_ops

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, permuted_tokens, sorted_indices, probs=None, padded_mode=False):
        permuted_tokens_fp32 = permuted_tokens.float()
        if probs is not None:
            probs_fp32 = probs.float()
            num_unpermuted_tokens = probs.numel()
            topk = probs.size(1)
        else:
            num_unpermuted_tokens = permuted_tokens.size(0)
            topk = 1

        unpermuted_tokens = torch.zeros(
            [num_unpermuted_tokens, permuted_tokens.shape[-1]],
            dtype=permuted_tokens_fp32.dtype,
            device=permuted_tokens_fp32.device
        )
        unpermuted_tokens.index_copy_(0, sorted_indices.to(torch.int64), permuted_tokens_fp32)
        unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.shape[-1])
        if probs is not None:
            unpermuted_tokens = unpermuted_tokens * probs_fp32.unsqueeze(-1)
        unpermuted_tokens = unpermuted_tokens.sum(dim=1)
        return unpermuted_tokens.to(permuted_tokens.dtype)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, permuted_tokens, sorted_indices, probs=None, padded_mode=False):
        # 调用 pybind 绑定的接口
        output = kernel_gen_ops.moe_token_unpermute(
            permuted_tokens,
            sorted_indices,
            probs,
            padded_mode
        )
        return output