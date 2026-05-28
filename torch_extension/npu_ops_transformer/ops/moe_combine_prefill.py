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


class MoeCombinePrefillOpBuilder(OpBuilder):
    def __init__(self):
        super(MoeCombinePrefillOpBuilder, self).__init__("npu_moe_combine_prefill")

    def sources(self):
        return ['ops/csrc/moe_combine_prefill.cpp']

    def schema(self) -> str:
        return (
            "npu_moe_combine_prefill(Tensor context, Tensor recv_x, Tensor token_src_info, Tensor ep_recv_counts, "
            "Tensor topk_weights, Tensor token_idx, int ep_world_size, int ep_rank_id, int moe_expert_num, "
            "int ccl_buffer_size, *, Tensor? tp_recv_counts=None, Tensor? combine_send_cost_stats=None, "
            "int tp_world_size=1, int tp_rank_id=0, int real_max_bs=0, int round=4, int per_round_tokens=1024) -> Tensor"
            # "npu_moe_combine_prefill(Tensor recv_x, Tensor token_src_info, Tensor ep_recv_counts, "
            # "Tensor recv_topk_weights, Tensor token_idx, str ep_group_name, int ep_world_size, int ep_rank_id, "
            # "int moe_expert_num, Tensor? tp_recv_counts=None, *, str tp_group_name='', int tp_world_size=0, "
            # "int tp_rank_id=0, int real_max_bs=0, int round=4, int per_round_tokens=1024, "
            # "Tensor? combine_send_cost_stats=None) -> Tensor"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        # def npu_moe_combine_normal_meta(
        #     recv_x, token_src_info, ep_recv_counts, recv_topk_weights, token_idx,
        #     ep_group_name, ep_world_size, ep_rank_id, moe_expert_num,
        #     tp_recv_counts=None,
        #     tp_group_name="", tp_world_size=0, tp_rank_id=0, real_max_bs=0,
        #     round=4, per_round_tokens=1024,
        #     combine_send_cost_stats=None,
        # ):
        def npu_moe_combine_prefill_meta(
            context, recv_x, token_src_info, ep_recv_counts, topk_weights, token_idx, ep_world_size,
            ep_rank_id, moe_expert_num, ccl_buffer_size, tp_recv_counts=None, combine_send_cost_stats=None,
            tp_world_size=0, tp_rank_id=0, real_max_bs=0, round=4, per_round_tokens=1024
        ):
            torch._check(
                recv_x.dim() == 2,
                lambda: f"recv_x must be 2D tensor (num_recv_tokens, H).{ops_error(ErrCode.VALUE)}",
            )
            torch._check(
                token_src_info.dim() == 1,
                lambda: f"token_src_info must be 1D tensor.{ops_error(ErrCode.VALUE)}",
            )
            torch._check(
                ep_recv_counts.dim() == 1,
                lambda: f"ep_recv_counts must be 1D tensor.{ops_error(ErrCode.VALUE)}",
            )
            torch._check(
                topk_weights.dim() == 2,
                lambda: f"topk_weights must be 2D tensor (BS, K).{ops_error(ErrCode.VALUE)}",
            )
            torch._check(
                token_idx.dim() == 2,
                lambda: f"token_idx must be 2D tensor.{ops_error(ErrCode.VALUE)}",
            )
            torch._check(
                token_idx.dtype == torch.int32,
                lambda: f"token_idx dtype must be int32.{ops_error(ErrCode.VALUE)}",
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
                (moe_expert_num % ep_world_size) == 0,
                lambda: (
                    f"moe_expert_num must be divisible by ep_world_size, "
                    f"but got {moe_expert_num=}, {ep_world_size=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                moe_expert_num > 0 and moe_expert_num <= 512,
                lambda: (
                    f"moe_expert_num must be in range (0, 512], "
                    f"but got {moe_expert_num=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                ep_world_size >= 2 and ep_world_size <= 384,
                lambda: (
                    f"ep_world_size must be in range [2, 384], "
                    f"but got {ep_world_size=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                ccl_buffer_size > 0,
                lambda: f"ccl_buffer_size must be positive.{ops_error(ErrCode.VALUE)}",
            )
            torch._check(
                round > 0,
                lambda: f"round must be greater than 0, but got {round=}.{ops_error(ErrCode.VALUE)}",
            )
            torch._check(
                per_round_tokens > 0,
                lambda: f"per_round_tokens must be greater than 0, but got {per_round_tokens=}.{ops_error(ErrCode.VALUE)}",
            )

            num_recv_tokens = recv_x.size(0)
            h = recv_x.size(1)
            bs = token_idx.size(0)
            k = token_idx.size(1)

            torch._check(
                token_src_info.numel() == num_recv_tokens * 3,
                lambda: (
                    f"token_src_info numel must be num_recv_tokens * 3, "
                    f"expected {num_recv_tokens * 3}, but got {token_src_info.numel()}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                ep_recv_counts.numel() == moe_expert_num,
                lambda: (
                    f"ep_recv_counts numel must be moe_expert_num, "
                    f"expected {moe_expert_num}, but got {ep_recv_counts.numel()}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                k > 0 and k <= min(16, moe_expert_num),
                lambda: (
                    f"token_idx K must be in range (0, min(16, moe_expert_num)], "
                    f"but got {k=}, {moe_expert_num=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                bs > 0 and bs <= 8000,
                lambda: (
                    f"token_idx BS must be in range [1, 8000], "
                    f"but got {bs=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                h >= 1024 and h <= 7168,
                lambda: (
                    f"recv_x H must be in range [1024, 7168], "
                    f"but got {h=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            if tp_recv_counts is not None:
                torch._check(
                    tp_recv_counts.dtype == torch.int32,
                    lambda: f"tp_recv_counts dtype must be int32.{ops_error(ErrCode.VALUE)}",
                )
                torch._check(
                    tp_recv_counts.dim() == 1,
                    lambda: f"tp_recv_counts must be 1D tensor.{ops_error(ErrCode.VALUE)}",
                )
            if combine_send_cost_stats is not None:
                torch._check(
                    combine_send_cost_stats.dtype == torch.int32,
                    lambda: f"combine_send_cost_stats dtype must be int32.{ops_error(ErrCode.VALUE)}",
                )
                torch._check(
                    combine_send_cost_stats.dim() == 1,
                    lambda: f"combine_send_cost_stats must be 1D tensor.{ops_error(ErrCode.VALUE)}",
                )
                torch._check(
                    combine_send_cost_stats.size(0) == ep_world_size,
                    lambda: (
                        "combine_send_cost_stats shape[0] must equal ep_world_size."
                        f"{ops_error(ErrCode.VALUE)}"
                    ),
                )

            x = recv_x.new_empty((bs, h), dtype=recv_x.dtype)
            return x


moe_combine_prefill_op_builder = MoeCombinePrefillOpBuilder()
op_module = moe_combine_prefill_op_builder.load()


@impl(AS_LIBRARY, moe_combine_prefill_op_builder.name, "PrivateUse1")
def npu_moe_combine_prefill(
    # recv_x, token_src_info, ep_recv_counts, recv_topk_weights, token_idx,
    # ep_group_name, ep_world_size, ep_rank_id, moe_expert_num,
    # tp_recv_counts=None,
    # tp_group_name="", tp_world_size=0, tp_rank_id=0, real_max_bs=0,
    # round=4, per_round_tokens=1024,
    # combine_send_cost_stats=None,
    context, recv_x, token_src_info, ep_recv_counts, topk_weights, token_idx, ep_world_size,
    ep_rank_id, moe_expert_num, ccl_buffer_size, tp_recv_counts=None, combine_send_cost_stats=None,
    tp_world_size=0, tp_rank_id=0, real_max_bs=0, round=4, per_round_tokens=1024,
):
    """
    MoE EP combine prefill with Mc2 communication context.

    Args:
        context: Mc2 context tensor from update_context()
        recv_x: Received expanded tensor from dispatch
        token_src_info: Token source info tensor (num_recv_tokens * 3)
        ep_recv_counts: EP receive counts tensor (moe_expert_num)
        topk_weights: Topk weights tensor (BS, K)
        token_idx: Token indices tensor
        ep_world_size: Expert parallel world size
        ep_rank_id: Current rank id in EP group
        moe_expert_num: Total number of experts
        ccl_buffer_size: CCL buffer size
        combine_send_cost_stats: Optional cost statistics tensor
        real_max_bs: Real max batch size (default 0)
        round: Round count (default 4)
        per_round_tokens: Tokens per round (default 1024)

    Returns:
        x: Combined output tensor (BS, H)
    """
    return op_module.npu_moe_combine_prefill(
        # recv_x, token_src_info, ep_recv_counts, recv_topk_weights, token_idx, tp_recv_counts,
        # ep_group_name, ep_world_size, ep_rank_id, moe_expert_num,
        # tp_group_name, tp_world_size, tp_rank_id, real_max_bs, round, per_round_tokens,
        # combine_send_cost_stats,
        context, recv_x, token_src_info, ep_recv_counts, topk_weights, token_idx, ep_world_size,
        ep_rank_id, moe_expert_num, ccl_buffer_size, tp_recv_counts, combine_send_cost_stats,
        tp_world_size, tp_rank_id, real_max_bs, round, per_round_tokens
    )