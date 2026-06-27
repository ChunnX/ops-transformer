# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch
import torch_npu
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY


class MoeEpDispatchOpBuilder(OpBuilder):
    def __init__(self):
        super(MoeEpDispatchOpBuilder, self).__init__("npu_moe_ep_dispatch")

    def sources(self):
        return ['ops/csrc/moe_ep_dispatch.cpp']

    def schema(self) -> str:
        return "npu_moe_ep_dispatch(Tensor context, Tensor x, Tensor topk_idx, Tensor? topk_weights, " \
               "Tensor? scales, Tensor? cached_handle_dst_buffer_slot_idx, int ep_world_size, " \
               "int ep_rank_id, int num_experts, int num_max_tokens_per_rank, int ccl_buffer_size, " \
               "*, int expert_alignment=1, bool do_cpu_sync=True, int host_pinned_counter_addr=0) " \
               "-> (Tensor, Tensor, Tensor)"

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_moe_ep_dispatch_meta(context, x, topk_idx, topk_weights, scales, 
                                     cached_handle_dst_buffer_slot_idx, ep_world_size,
                                     ep_rank_id, num_experts, num_max_tokens_per_rank,
                                     ccl_buffer_size, *, expert_alignment=1,
                                     do_cpu_sync=True, host_pinned_counter_addr=0):
            torch._check(
                ep_world_size > 1,
                lambda: (
                    f"ep_world_size should be greater than 1, "
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )

            num_tokens = x.shape[0]
            top_k = topk_idx.shape[1]
            num_local_experts = num_experts // ep_world_size
            upper_a = ep_world_size * num_max_tokens_per_rank

            num_recv_tokens_per_rank = x.new_empty((ep_world_size,), dtype=torch.int32)
            num_recv_tokens_per_expert = x.new_empty((num_local_experts,), dtype=torch.int64)
            dst_buffer_slot_idx = x.new_empty((num_tokens, top_k), dtype=torch.int32)

            return (num_recv_tokens_per_rank, num_recv_tokens_per_expert, dst_buffer_slot_idx)


class MoeEpDispatchEpilogueOpBuilder(OpBuilder):
    def __init__(self):
        super(MoeEpDispatchEpilogueOpBuilder, self).__init__("npu_moe_ep_dispatch_epilogue")

    def sources(self):
        return ['ops/csrc/moe_ep_dispatch_epilogue.cpp']

    def schema(self) -> str:
        return "npu_moe_ep_dispatch_epilogue(Tensor context, Tensor dst_buffer_slot_idx, " \
               "Tensor num_recv_tokens_per_rank, Tensor num_recv_tokens_per_expert, " \
               "Tensor? cached_recv_src_metadata, " \
               "int ep_world_size, int ep_rank_id, " \
               "int num_experts, int num_max_tokens_per_rank, int ccl_buffer_size, *, " \
               "int expert_alignment=1, Tensor recv_x, " \
               "Tensor? recv_topk_weights, Tensor recv_src_metadata, Tensor? recv_scales) " \
               "-> (Tensor, Tensor?, Tensor, Tensor?)"

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_moe_ep_dispatch_epilogue_meta(context, dst_buffer_slot_idx, num_recv_per_rank,
                                              num_recv_per_expert, cached_recv_src_metadata,
                                              ep_world_size, ep_rank_id, num_experts,
                                              num_max_tokens_per_rank, ccl_buffer_size, *, expert_alignment=1,
                                              recv_x, recv_topk_weights,
                                              recv_src_metadata, recv_scales):
            recv_topk_weights_out = recv_topk_weights if recv_topk_weights is not None else None
            recv_scales_out = recv_scales if recv_scales is not None else None
            return (recv_x, recv_topk_weights_out, recv_src_metadata, recv_scales_out)


moe_ep_dispatch_op_builder = MoeEpDispatchOpBuilder()
op_module_dispatch = moe_ep_dispatch_op_builder.load()

moe_ep_dispatch_epilogue_op_builder = MoeEpDispatchEpilogueOpBuilder()
op_module_epilogue = moe_ep_dispatch_epilogue_op_builder.load()


@impl(AS_LIBRARY, moe_ep_dispatch_op_builder.name, "PrivateUse1")
def npu_moe_ep_dispatch(context, x, topk_idx, topk_weights, scales,
                        cached_handle_dst_buffer_slot_idx, ep_world_size,
                        ep_rank_id, num_experts, num_max_tokens_per_rank,
                        ccl_buffer_size, *, expert_alignment=1,
                        do_cpu_sync=True, host_pinned_counter_addr=0):
    return op_module_dispatch.npu_moe_ep_dispatch(
        context, x, topk_idx, topk_weights, scales, cached_handle_dst_buffer_slot_idx,
        ep_world_size, ep_rank_id, num_experts, num_max_tokens_per_rank, ccl_buffer_size,
        expert_alignment, do_cpu_sync, host_pinned_counter_addr
    )


@impl(AS_LIBRARY, moe_ep_dispatch_epilogue_op_builder.name, "PrivateUse1")
def npu_moe_ep_dispatch_epilogue(context, dst_buffer_slot_idx, num_recv_per_rank, num_recv_per_expert,
                                 cached_recv_src_metadata,
                                 ep_world_size, ep_rank_id, num_experts,
                                 num_max_tokens_per_rank, ccl_buffer_size, *, expert_alignment=1,
                                 recv_x, recv_topk_weights, recv_src_metadata, recv_scales):
    return op_module_epilogue.npu_moe_ep_dispatch_epilogue(
        context, dst_buffer_slot_idx, num_recv_per_rank, num_recv_per_expert,
        cached_recv_src_metadata,
        ep_world_size, ep_rank_id, num_experts,
        num_max_tokens_per_rank, ccl_buffer_size, expert_alignment,
        recv_x, recv_topk_weights, recv_src_metadata, recv_scales
    )