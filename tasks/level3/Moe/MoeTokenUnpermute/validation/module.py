import torch
import kernel_gen_ops


class Model(torch.nn.Module):
    """参考模型：与 api_desc 一致 — sorted_indices[k] 为从 permuted_tokens 取行的下标（gather），再按 topK 聚合。"""

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, permuted_tokens, sorted_indices, probs=None, padded_mode=False):
        rows, hidden = permuted_tokens.shape
        if probs is not None:
            tokens_num, topk = probs.shape
        else:
            tokens_num = rows
            topk = 1

        if rows != tokens_num * topk:
            raise ValueError(
                f"permuted_tokens dim0 ({rows}) must equal tokens_num*topk ({tokens_num}*{topk})"
            )
        if sorted_indices.numel() != rows:
            raise ValueError("sorted_indices length must match permuted_tokens rows")

        pt = permuted_tokens.float()
        idx = sorted_indices.to(torch.int64).clamp(0, rows - 1)
        gathered = pt.index_select(0, idx.reshape(-1))

        if probs is not None:
            gathered = gathered * probs.float().reshape(-1, 1)

        out = gathered.view(tokens_num, topk, hidden).sum(dim=1)
        return out.to(permuted_tokens.dtype)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, permuted_tokens, sorted_indices, probs=None, padded_mode=False):
        output = kernel_gen_ops.moe_token_unpermute(
            permuted_tokens,
            sorted_indices,
            probs,
            padded_mode,
        )
        return output
