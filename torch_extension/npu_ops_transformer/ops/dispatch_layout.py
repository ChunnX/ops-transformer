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


class DispatchLayoutOpBuilder(OpBuilder):
    def __init__(self):
        super(DispatchLayoutOpBuilder, self).__init__("npu_dispatch_layout")

    def sources(self):
        return ['ops/csrc/dispatch_layout.cpp']

    def schema(self) -> str:
        return "npu_dispatch_layout(Tensor topk_idx, int num_tokens, int num_ranks, int num_experts, " \
            "int num_topk, int local_ranksize, int per_round_tokens, int rank_id) -> " \
            "(Tensor, Tensor, Tensor, Tensor, Tensor)"

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_dispatch_layout_meta(topk_idx, num_tokens, num_ranks, num_experts, num_topk,
                                      local_ranksize, per_round_tokens, rank_id):
            torch._check(
                num_tokens > 0,
                lambda: (
                    f"num_tokens should be greater than 0, "
                    f"but got {num_tokens=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                num_ranks > 0,
                lambda: (
                    f"num_ranks should be greater than 0, "
                    f"but got {num_ranks=}."
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
                num_topk > 0,
                lambda: (
                    f"num_topk should be greater than 0, "
                    f"but got {num_topk=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                (num_experts % num_ranks) == 0,
                lambda: (
                    f"num_experts should be divisible by num_ranks, "
                    f"but got {num_experts=}, {num_ranks=}."
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
                (rank_id >= 0) and (rank_id < num_ranks),
                lambda: (
                    f"rank_id should be in [0, num_ranks), "
                    f"but got {num_ranks=}, {rank_id=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                topk_idx.dim() == 2,
                lambda: (
                    f"topk_idx should be 2D tensor, "
                    f"but got {topk_idx.dim()}D."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                topk_idx.size(0) == num_tokens,
                lambda: (
                    f"topk_idx size(0) should equal num_tokens, "
                    f"but got {topk_idx.size(0)=}, {num_tokens=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                topk_idx.size(1) == num_topk,
                lambda: (
                    f"topk_idx size(1) should equal num_topk, "
                    f"but got {topk_idx.size(1)=}, {num_topk=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )

            round_val = (num_tokens + per_round_tokens - 1) // per_round_tokens
            num_local_experts = num_experts // num_ranks

            num_tokens_per_rank = topk_idx.new_empty((num_ranks,), dtype=torch.int32)
            num_tokens_per_expert = topk_idx.new_empty((round_val * num_experts,), dtype=torch.int32)
            is_token_in_rank = topk_idx.new_empty((round_val * per_round_tokens, num_ranks), dtype=torch.int32)
            notify_send_data = topk_idx.new_empty((round_val * num_local_experts,), dtype=torch.int32)
            send_token_idx_small = topk_idx.new_empty((num_tokens * num_topk,), dtype=torch.int32)

            return (num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank,
                    notify_send_data, send_token_idx_small)


dispatch_layout_op_builder = DispatchLayoutOpBuilder()
op_module = dispatch_layout_op_builder.load()


@impl(AS_LIBRARY, dispatch_layout_op_builder.name, "PrivateUse1")
def npu_dispatch_layout(topk_idx, num_tokens, num_ranks, num_experts, num_topk,
                        local_ranksize, per_round_tokens, rank_id):
    """
    Dispatcher implementation for NPU.
    'PrivateUse1' is the dispatch key for custom NPU backends.
    """
    return op_module.npu_dispatch_layout(topk_idx, num_tokens, num_ranks, num_experts, num_topk,
                                         local_ranksize, per_round_tokens, rank_id)