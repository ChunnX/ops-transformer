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
from npu_ops_transformer.op_builder.builder import OpBuilder
from npu_ops_transformer.op_builder.builder import AS_LIBRARY


class NotifyDispatchOpBuilder(OpBuilder):
    def __init__(self):
        super(NotifyDispatchOpBuilder, self).__init__("npu_notify_dispatch")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/notify_dispatch.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return (
            "npu_notify_dispatch(Tensor context, Tensor send_data, Tensor token_per_expert_data, "
            "int send_count, int num_tokens, int ep_world_size, int ep_rank_id, int local_rank_size, "
            "int local_rank_id, int ccl_buffer_size, int round, int per_round_tokens) -> "
            "(Tensor, Tensor[], Tensor[], Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_notify_dispatch_meta(context, send_data, token_per_expert_data, send_count, num_tokens,
                                     ep_world_size, ep_rank_id, local_rank_size, local_rank_id,
                                     ccl_buffer_size, round, per_round_tokens):
            SEND_PER_GROUP = 3
            torch._check(
                send_count > 0,
                lambda: (
                    f"send_count should be greater than 0, "
                    f"but got {send_count=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                num_tokens > 0,
                lambda: (
                    f"num_tokens should be greater than 0, "
                    f"but got {num_tokens=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                ep_world_size > 0,
                lambda: (
                    f"ep_world_size should be greater than 0, "
                    f"but got {ep_world_size=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                (ep_rank_id >= 0) and (ep_rank_id < ep_world_size),
                lambda: (
                    f"ep_rank_id should be in [0, ep_world_size), "
                    f"but got {ep_world_size=}, {ep_rank_id=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                ccl_buffer_size > 0,
                lambda: (
                    f"ccl_buffer_size must be positive, "
                    f"but got {ccl_buffer_size=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                round > 0,
                lambda: (
                    f"round should be greater than 0, "
                    f"but got {round=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                per_round_tokens > 0,
                lambda: (
                    f"per_round_tokens should be greater than 0, "
                    f"but got {per_round_tokens=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                send_data.dim() == 1,
                lambda: (
                    f"send_data should be 1D tensor, "
                    f"but got {send_data.dim()}D."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                token_per_expert_data.dim() == 1,
                lambda: (
                    f"token_per_expert_data should be 1D tensor, "
                    f"but got {token_per_expert_data.dim()}D."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                context.dim() == 1,
                lambda: (
                    f"context should be 1D tensor, "
                    f"but got {context.dim()}D."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )

            num_experts = send_count // SEND_PER_GROUP // round
            num_local_experts = num_experts // ep_world_size
            torch._check(
                num_experts > 0,
                lambda: (
                    f"num_experts should be greater than 0, "
                    f"but got {num_experts=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                (num_experts % ep_world_size) == 0,
                lambda: (
                    f"num_experts should be divisible by ep_world_size, "
                    f"but got {num_experts=}, {ep_world_size=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                (local_rank_id >= 0) and (local_rank_id < local_rank_size),
                lambda: (
                    f"local_rank_id should be in [0, local_rank_size), "
                    f"but got {local_rank_size=}, {local_rank_id=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                token_per_expert_data.size(0) == round * num_experts,
                lambda: (
                    f"token_per_expert_data size(0) must equal round * num_experts "
                    f"(={round * num_experts}), but got {token_per_expert_data.size(0)}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                send_data.size(0) == send_count,
                lambda: (
                    f"send_data size(0) must equal send_count, "
                    f"but got {send_data.size(0)=}, {send_count=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )

            send_offset = send_data.new_empty((round * num_experts,), dtype=torch.int32)
            recv_data = send_data.new_empty((send_count,), dtype=torch.int32)
            recv_count = send_data.new_empty((round * num_experts,), dtype=torch.int32)
            recv_offset = send_data.new_empty((round * num_experts,), dtype=torch.int32)
            expert_global_offset = send_data.new_empty((num_local_experts,), dtype=torch.int32)
            srcrank_in_expert_offset = send_data.new_empty((num_local_experts * ep_world_size,), dtype=torch.int32)
            r_in_srcrank_offset = send_data.new_empty(
                (num_local_experts * ep_world_size * round,), dtype=torch.int32
            )
            total_recv_tokens = send_data.new_empty((1,), dtype=torch.int32)
            max_bs = send_data.new_empty((1,), dtype=torch.int32)
            recv_tokens_per_expert = send_data.new_empty((round * num_local_experts,), dtype=torch.int32)

            recv_offset_list = [recv_offset, recv_count]
            expert_output_offset_list = [
                expert_global_offset,
                srcrank_in_expert_offset,
                r_in_srcrank_offset,
            ]
            return (
                send_offset,
                recv_offset_list,
                expert_output_offset_list,
                total_recv_tokens,
                max_bs,
                recv_data,
                recv_tokens_per_expert,
            )


notify_dispatch_op_builder = NotifyDispatchOpBuilder()
op_module = notify_dispatch_op_builder.load()


@impl(AS_LIBRARY, notify_dispatch_op_builder.name, "PrivateUse1")
def npu_notify_dispatch(context, send_data, token_per_expert_data, send_count, num_tokens,
                        ep_world_size, ep_rank_id, local_rank_size, local_rank_id,
                        ccl_buffer_size, round, per_round_tokens):
    """
    MoE notify_dispatch with Mc2 context communication domain.

    Returns offsets aligned with npu_dispatch_normal inputs:
      send_offset, recv_offset[recv_offset, recv_count],
      expert_output_offset[expert_global_offset, srcrank_in_expert_offset, r_in_srcrank_offset],
      total_recv_tokens, max_bs, recv_data, recv_tokens_per_expert.
    """
    return op_module.npu_notify_dispatch(
        context, send_data, token_per_expert_data, send_count, num_tokens,
        ep_world_size, ep_rank_id, local_rank_size, local_rank_id,
        ccl_buffer_size, round, per_round_tokens,
    )
