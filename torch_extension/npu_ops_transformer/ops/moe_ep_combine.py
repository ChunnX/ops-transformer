import torch
import torch_npu
from torch.library import impl
from npu_ops_transformer.op_builder.builder import OpBuilder
from npu_ops_transformer.op_builder.builder import AS_LIBRARY


class MoeEpCombineOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("npu_moe_ep_combine")

    def sources(self):
        return ['ops/csrc/moe_ep_combine.cpp']

    def schema(self) -> str:
        return "npu_moe_ep_combine("
               "Tensor context, Tensor x, Tensor dst_buffer_slot_idx, Tensor recv_src_metadata, "
               "Tensor num_recv_tokens_per_rank, Tensor? topk_weights, "
               "Tensor? bias_0, Tensor? bias_1, "
               "int ep_world_size, int ep_rank_id, "
               "int num_experts, int num_max_tokens_per_rank, int ccl_buffer_size"
               ") -> (Tensor, Tensor?)"

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_moe_ep_combine_meta(context, x, dst_buffer_slot_idx, recv_src_metadata,
                                    num_recv_tokens_per_rank, topk_weights,
                                    bias_0, bias_1,
                                    ep_world_size, ep_rank_id,
                                    num_experts, num_max_tokens_per_rank, ccl_buffer_size):
            num_tokens = dst_buffer_slot_idx.shape[0]
            hidden = x.shape[1]
            combined_x = x.new_empty(num_tokens, hidden, dtype=torch.bfloat16)
            if topk_weights is not None:
                top_k = dst_buffer_slot_idx.shape[1]
                combined_topk_weights = topk_weights.new_empty(num_tokens, top_k, dtype=torch.float32)
            else:
                combined_topk_weights = None
            return (combined_x, combined_topk_weights)


moe_ep_combine_op_builder = MoeEpCombineOpBuilder()
op_module_combine = moe_ep_combine_op_builder.load()


@impl(AS_LIBRARY, moe_ep_combine_op_builder.name, "PrivateUse1")
def npu_moe_ep_combine(context, x, dst_buffer_slot_idx, recv_src_metadata,
                       num_recv_tokens_per_rank, topk_weights,
                       bias_0, bias_1,
                       ep_world_size, ep_rank_id,
                       num_experts, num_max_tokens_per_rank, ccl_buffer_size):
    return op_module_combine.npu_moe_ep_combine(
        context, x, dst_buffer_slot_idx, recv_src_metadata,
        num_recv_tokens_per_rank, topk_weights,
        bias_0, bias_1,
        ep_world_size, ep_rank_id,
        num_experts, num_max_tokens_per_rank, ccl_buffer_size,
    )