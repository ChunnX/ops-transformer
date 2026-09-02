import torch

from cann_ops_transformer.ops.block_attention_residuals import block_attention_residuals
from cann_ops_transformer.ops.block_attention_residuals_grad import (
    block_attention_residuals_grad,
)


class BlockAttentionResidualsFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prefix_sum, block_res, proj_weight, norm_weight, norm_eps):
        hidden, inv_norm, probs = block_attention_residuals(
            prefix_sum, block_res, proj_weight, norm_weight,
            norm_eps=norm_eps, need_backward=True
        )
        ctx.save_for_backward(
            prefix_sum, block_res, proj_weight, norm_weight, inv_norm, probs
        )
        return hidden

    @staticmethod
    def backward(ctx, grad_hidden_states):
        prefix_sum, block_res, proj_weight, norm_weight, inv_norm, probs = ctx.saved_tensors
        return (
            *block_attention_residuals_grad(
                prefix_sum, block_res, proj_weight, norm_weight, grad_hidden_states,
                inv_norm, probs, valid_block_num=block_res.size(1)
            ),
            None,
        )


def attn_res(prefix_sum, block_res, proj_weight, norm_weight, norm_eps=1e-5):
    """Attention Residual forward and backward."""
    needs_grad = any(
        tensor.requires_grad
        for tensor in (prefix_sum, block_res, proj_weight, norm_weight)
    )
    if needs_grad:
        return BlockAttentionResidualsFunction.apply(
            prefix_sum, block_res, proj_weight, norm_weight, norm_eps
        )
    return block_attention_residuals(
        prefix_sum, block_res, proj_weight, norm_weight,
        norm_eps=norm_eps, need_backward=False
    )
