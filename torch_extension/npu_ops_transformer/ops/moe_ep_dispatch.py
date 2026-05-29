import torch
import torch_npu
from torch.library import impl
from npu_ops_transformer.op_builder.builder import OpBuilder
from npu_ops_transformer.op_builder.builder import AS_LIBRARY


class MoeEpDispatchOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("npu_moe_ep_dispatch")

    def sources(self):
        return ['ops/csrc/moe_ep_dispatch.cpp']

    def schema(self) -> str:
        return "npu_moe_ep_dispatch("
               "Tensor x, Tensor? topk_idx, Tensor? topk_weights, "
               "Tensor? cached_handle_dst_buffer_slot_idx, "
               "str group_ep, int ep_world_size, int ep_rank_id, "
               "int num_experts, int num_max_tokens_per_rank, "
               "*, int expert_alignment=1, bool do_cpu_sync=True"
               ") -> (Tensor, Tensor, Tensor, Tensor)"

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_moe_ep_dispatch_meta(x, topk_idx, topk_weights,
                                     cached_handle_dst_buffer_slot_idx,
                                     group_ep, ep_world_size, ep_rank_id,
                                     num_experts, num_max_tokens_per_rank,
                                     *, expert_alignment=1, do_cpu_sync=True):
            num_tokens = x.shape[0]
            top_k = 8
            if topk_idx is not None:
                top_k = topk_idx.shape[1]
            elif cached_handle_dst_buffer_slot_idx is not None:
                top_k = cached_handle_dst_buffer_slot_idx.shape[1]
            num_local_experts = num_experts // ep_world_size
            upper_a = ep_world_size * num_max_tokens_per_rank
            return (
                x.new_empty(ep_world_size, dtype=torch.int32),
                x.new_empty(num_local_experts, dtype=torch.int64),
                x.new_empty(num_tokens, top_k, dtype=torch.int32),
                x.new_empty(upper_a, 3, dtype=torch.int32),
            )


class MoeEpDispatchEpilogueOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("npu_moe_ep_dispatch_epilogue")

    def sources(self):
        return ['ops/csrc/moe_ep_dispatch_epilogue.cpp']

    def schema(self) -> str:
        return "npu_moe_ep_dispatch_epilogue("
               "Tensor dst_buffer_slot_idx, Tensor recv_src_metadata_internal, "
               "Tensor num_recv_tokens_per_rank_host_pinned, "
               "Tensor num_recv_tokens_per_expert_host_pinned, "
               "str group_ep, int ep_world_size, int ep_rank_id, "
               "int num_experts, int num_max_tokens_per_rank, "
               "*, int expert_alignment=1"
               ") -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_moe_ep_dispatch_epilogue_meta(dst_buffer_slot_idx, recv_src_metadata_internal,
                                               num_recv_per_rank_hp, num_recv_per_expert_hp,
                                               group_ep, ep_world_size, ep_rank_id,
                                               num_experts, num_max_tokens_per_rank,
                                               *, expert_alignment=1):
            top_k = dst_buffer_slot_idx.shape[1]
            num_local_experts = num_experts // ep_world_size
            upper_a = ep_world_size * num_max_tokens_per_rank
            hidden = upper_a
            bf16 = torch.bfloat16
            return (
                dst_buffer_slot_idx.new_empty(upper_a, hidden, dtype=bf16),
                dst_buffer_slot_idx.new_empty(upper_a, top_k, dtype=torch.int32),
                dst_buffer_slot_idx.new_empty(upper_a, top_k, dtype=torch.float32),
                dst_buffer_slot_idx.new_empty(upper_a, 3, dtype=torch.int32),
                dst_buffer_slot_idx.new_empty(ep_world_size, dtype=torch.int32),
                dst_buffer_slot_idx.new_empty(num_local_experts, dtype=torch.int64),
            )


moe_ep_dispatch_op_builder = MoeEpDispatchOpBuilder()
op_module_dispatch = moe_ep_dispatch_op_builder.load()

moe_ep_dispatch_epilogue_op_builder = MoeEpDispatchEpilogueOpBuilder()
op_module_epilogue = moe_ep_dispatch_epilogue_op_builder.load()


@impl(AS_LIBRARY, moe_ep_dispatch_op_builder.name, "PrivateUse1")
def npu_moe_ep_dispatch(x, topk_idx, topk_weights,
                        cached_handle_dst_buffer_slot_idx,
                        group_ep, ep_world_size, ep_rank_id,
                        num_experts, num_max_tokens_per_rank,
                        *, expert_alignment=1, do_cpu_sync=True):
    return op_module_dispatch.npu_moe_ep_dispatch(
        x, topk_idx, topk_weights,
        cached_handle_dst_buffer_slot_idx,
        group_ep, ep_world_size, ep_rank_id,
        num_experts, num_max_tokens_per_rank,
        expert_alignment, do_cpu_sync,
    )


@impl(AS_LIBRARY, moe_ep_dispatch_epilogue_op_builder.name, "PrivateUse1")
def npu_moe_ep_dispatch_epilogue(dst_buffer_slot_idx, recv_src_metadata_internal,
                                  num_recv_per_rank_hp, num_recv_per_expert_hp,
                                  group_ep, ep_world_size, ep_rank_id,
                                  num_experts, num_max_tokens_per_rank,
                                  *, expert_alignment=1):
    return op_module_epilogue.npu_moe_ep_dispatch_epilogue(
        dst_buffer_slot_idx, recv_src_metadata_internal,
        num_recv_per_rank_hp, num_recv_per_expert_hp,
        group_ep, ep_world_size, ep_rank_id,
        num_experts, num_max_tokens_per_rank,
        expert_alignment,
    )