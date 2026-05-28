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
        return "npu_notify_dispatch(Tensor send_data, Tensor token_per_expert_data, " \
            "int send_count, int num_tokens, str group_ep, int rank_size, int rank_id, " \
            "int local_rank_size, int local_rank_id, int round, int per_round_tokens) -> " \
            "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_notify_dispatch_meta(send_data, token_per_expert_data, send_count, num_tokens,
                                     group_ep, rank_size, rank_id, local_rank_size,
                                     local_rank_id, round, per_round_tokens):
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
                rank_size > 0,
                lambda: (
                    f"rank_size should be greater than 0, "
                    f"but got {rank_size=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                (rank_id >= 0) and (rank_id < rank_size),
                lambda: (
                    f"rank_id should be in [0, rank_size), "
                    f"but got {rank_size=}, {rank_id=}."
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
                num_experts > 0,
                lambda: (
                    f"num_experts should be greater than 0, "
                    f"but got {num_experts=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                (num_experts % rank_size) == 0,
                lambda: (
                    f"num_experts should be divisible by rank_size, "
                    f"but got {num_experts=}, {rank_size=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )

            SEND_PER_GROUP = 3
            num_experts = send_count // SEND_PER_GROUP // round
            num_local_experts = num_experts // rank_size

            send_data_offset = send_data.new_empty((round * num_experts,), dtype=torch.int32)
            recv_data = send_data.new_empty((send_count,), dtype=torch.int32)
            recv_count = send_data.new_empty((round * num_experts,), dtype=torch.int32)
            recv_offset = send_data.new_empty((round * num_experts,), dtype=torch.int32)
            expert_global_offset = send_data.new_empty((num_local_experts,), dtype=torch.int32)
            srcrank_in_expert_offset = send_data.new_empty((num_local_experts * rank_size,), dtype=torch.int32)
            r_in_srcrank_offset = send_data.new_empty((num_local_experts * rank_size * round,), dtype=torch.int32)
            total_recv_tokens = send_data.new_empty((1,), dtype=torch.int32)
            max_bs = send_data.new_empty((1,), dtype=torch.int32)
            recv_tokens_per_expert = send_data.new_empty((round * num_local_experts,), dtype=torch.int32)

            return (send_data_offset, recv_data, recv_count, recv_offset, expert_global_offset,
                    srcrank_in_expert_offset, r_in_srcrank_offset, total_recv_tokens, max_bs,
                    recv_tokens_per_expert)


notify_dispatch_op_builder = NotifyDispatchOpBuilder()
op_module = notify_dispatch_op_builder.load()


@impl(AS_LIBRARY, notify_dispatch_op_builder.name, "PrivateUse1")
def npu_notify_dispatch(send_data, token_per_expert_data, send_count, num_tokens,
                        group_ep, rank_size, rank_id, local_rank_size,
                        local_rank_id, round, per_round_tokens):
    """
    Dispatcher implementation for NPU.
    'PrivateUse1' is the dispatch key for custom NPU backends.
    """
    return op_module.npu_notify_dispatch(send_data, token_per_expert_data, send_count, num_tokens,
                                         group_ep, rank_size, rank_id, local_rank_size,
                                         local_rank_id, round, per_round_tokens)