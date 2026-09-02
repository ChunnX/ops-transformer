import torch
import torch_npu
from torch.library import impl
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import get_as_library

AS_LIBRARY = get_as_library()


class BlockAttentionResidualsGradOpBuilder(OpBuilder):
    def __init__(self):
        super(BlockAttentionResidualsGradOpBuilder, self).__init__(
            "block_attention_residuals_grad"
        )

    def sources(self):
        return ["ops/csrc/block_attention_residuals_grad.cpp"]

    def schema(self) -> str:
        return (
            "block_attention_residuals_grad(Tensor partial_block, Tensor block_res, "
            "Tensor proj_weight, Tensor norm_weight, Tensor grad_hidden_states, "
            "Tensor inv_norm, Tensor probs, int valid_block_num=-1) "
            "-> (Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def block_attention_residuals_grad_meta(
            partial_block, block_res, proj_weight, norm_weight, grad_hidden_states,
            inv_norm, probs, valid_block_num=-1
        ):
            token_count = partial_block.size(0)
            hidden_size = partial_block.size(1)
            block_count = block_res.size(1)
            grad_partial_block = torch.empty(
                token_count, hidden_size, dtype=partial_block.dtype, device="meta"
            )
            grad_block_res = torch.empty(
                token_count, block_count, hidden_size,
                dtype=partial_block.dtype, device="meta"
            )
            grad_proj_weight = torch.empty(
                1, hidden_size, dtype=partial_block.dtype, device="meta"
            )
            grad_norm_weight = torch.empty(
                hidden_size, dtype=partial_block.dtype, device="meta"
            )
            return grad_partial_block, grad_block_res, grad_proj_weight, grad_norm_weight


block_attention_residuals_grad_op_builder = BlockAttentionResidualsGradOpBuilder()


@impl(AS_LIBRARY, block_attention_residuals_grad_op_builder.name, "PrivateUse1")
def _block_attention_residuals_grad_dispatch(
    partial_block, block_res, proj_weight, norm_weight, grad_hidden_states,
    inv_norm, probs, valid_block_num=-1
):
    if valid_block_num < 0:
        valid_block_num = block_res.size(1)
    op_module = block_attention_residuals_grad_op_builder.load()
    return op_module.block_attention_residuals_grad(
        partial_block, block_res, proj_weight, norm_weight, grad_hidden_states,
        inv_norm, probs, valid_block_num
    )


def block_attention_residuals_grad(
    partial_block, block_res, proj_weight, norm_weight, grad_hidden_states,
    inv_norm, probs, valid_block_num=-1
):
    """Python wrapper for aclnnBlockAttentionResidualsGrad."""
    if valid_block_num < 0:
        valid_block_num = block_res.size(1)
    op_module = block_attention_residuals_grad_op_builder.load()
    return op_module.block_attention_residuals_grad(
        partial_block, block_res, proj_weight, norm_weight, grad_hidden_states,
        inv_norm, probs, valid_block_num
    )
