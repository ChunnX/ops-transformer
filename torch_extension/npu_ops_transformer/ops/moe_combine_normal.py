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


 class CombineNormalOpBuilder(OpBuilder):
    def __init__(self):
        super(CombineNormalOpBuilder, self).__init__("npu_moe_combine_normal")

    def sources(self):
        return ['ops/csrc/moe_combine_normal.cpp']

    def schema(self) -> str:
        return (
            "npu_moe_combine_normal(Tensor recv_x, Tensor token_src_info, Tensor ep_recv_counts, "
            "Tensor recv_topk_weights, Tensor token_idx, str ep_group_name, int ep_world_size, int ep_rank_id, "
            "int moe_expert_num, Tensor? tp_recv_counts=None, *, str tp_group_name='', int tp_world_size=0, "
            "int tp_rank_id=0, int real_max_bs=0, int round=4, int per_round_tokens=1024, "
            "Tensor? combine_send_cost_stats=None) -> Tensor"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_moe_combine_normal_meta(
            recv_x, token_src_info, ep_recv_counts, recv_topk_weights, token_idx,
            ep_group_name, ep_world_size, ep_rank_id, moe_expert_num,
            tp_recv_counts=None,
            tp_group_name="", tp_world_size=0, tp_rank_id=0, real_max_bs=0,
            round=4, per_round_tokens=1024,
            combine_send_cost_stats=None,
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
                recv_topk_weights.dim() == 2,
                lambda: f"recv_topk_weights must be 2D tensor (BS, K).{ops_error(ErrCode.VALUE)}",
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
            num_recv_tokens = recv_x.size(0)
            h = recv_x.size(1)
            bs = recv_topk_weights.size(0)
            k = recv_topk_weights.size(1)
            
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
                    f"recv_topk_weights K must be in range (0, min(16, moe_expert_num)], "
                    f"but got {k=}, {moe_expert_num=}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                bs > 0 and bs <= 8000,
                lambda: (
                    f"recv_topk_weights BS must be in range [1, 8000], "
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
            torch._check(
                    round > 0,
                    lambda: f"round must be greater than 0, but got {round=}.{ops_error(ErrCode.VALUE)}",
                )
                torch._check(
                    per_round_tokens > 0,
                    lambda: f"per_round_tokens must be greater than 0, but got {per_round_tokens=}.{ops_error(ErrCode.VALUE)}",
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


combine_normal_op_builder = CombineNormalOpBuilder()
op_module = combine_normal_op_builder.load()


@impl(AS_LIBRARY, combine_normal_op_builder.name, "PrivateUse1")
def npu_moe_combine_normal(
    recv_x, token_src_info, ep_recv_counts, recv_topk_weights, token_idx,
    ep_group_name, ep_world_size, ep_rank_id, moe_expert_num,
    tp_recv_counts=None,
    tp_group_name="", tp_world_size=0, tp_rank_id=0, real_max_bs=0,
    round=4, per_round_tokens=1024,
    combine_send_cost_stats=None,
):
"""
MoE EP combine (normal path) with Hccl communication group.

This function combines the expert outputs back to original token positions.
Typical flow: dispatch -> expert compute -> npu_moe_combine_normal.
"""
return op_module.npu_moe_combine_normal(
    recv_x, token_src_info, ep_recv_counts, recv_topk_weights, token_idx, tp_recv_counts,
    ep_group_name, ep_world_size, ep_rank_id, moe_expert_num,
    tp_group_name, tp_world_size, tp_rank_id, real_max_bs, round, per_round_tokens,
    combine_send_cost_stats,
)