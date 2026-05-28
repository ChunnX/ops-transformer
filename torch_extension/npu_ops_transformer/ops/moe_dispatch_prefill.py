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

QUANT_MODE_NO_QUANT = 0
QUANT_MODE_DYNAMIC_SCALES = 2


class MoeDispatchPrefillOpBuilder(OpBuilder):
    def __init__(self):
        super(MoeDispatchPrefillOpBuilder, self).__init__("npu_moe_dispatch_prefill")

    def sources(self):
        return ['ops/csrc/moe_dispatch_prefill.cpp']

    def schema(self) -> str:
        return (
            "npu_moe_dispatch_prefill(Tensor context, Tensor x, Tensor topk_idx, Tensor[] ext_handle_data, "
            "int ep_world_size, int ep_rank_id, int local_ranksize, int moe_expert_num, int ccl_buffer_size, "
            "*, Tensor? x_scales=None, Tensor? topk_weights=None, Tensor? num_tokens_per_rank=None, "
            "Tensor? is_token_in_rank=None, Tensor? num_tokens_per_expert=None, "
            "Tensor? dispatch_wait_recv_cost_stats=None, int num_worst_tokens=0, int quant_mode=0, "
            "int round=4, int per_round_tokens=1024) -> (Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_moe_dispatch_prefill_meta(
            context, x, topk_idx, ext_handle_data, ep_world_size, ep_rank_id, local_ranksize, moe_expert_num,
            ccl_buffer_size, x_scales=None, topk_weights=None, num_tokens_per_rank=None, is_token_in_rank=None,
            num_tokens_per_expert=None, dispatch_wait_recv_cost_stats=None, num_worst_tokens=0, quant_mode=0,
            round=4, per_round_tokens=1024,
        ):
            torch._check(
                x.dim() == 2,
                lambda: f"x must be 2D tensor (num_tokens, H).{ops_error(ErrCode.VALUE)}",
            )
            torch._check(
                topk_idx.dim() == 2,
                lambda: f"topk_idx must be 2D tensor (num_tokens, K).{ops_error(ErrCode.VALUE)}",
            )
            torch._check(
                len(ext_handle_data) == 2,
                lambda: f"ext_handle_data must contain exactly 2 tensors.{ops_error(ErrCode.VALUE)}",
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
                quant_mode in (QUANT_MODE_NO_QUANT, QUANT_MODE_DYNAMIC_SCALES),
                lambda: (
                    f"quant_mode should be 0 or 2, but got {quant_mode=}."
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

            num_tokens = topk_idx.size(0)
            h = x.size(1)
            num_local_experts = moe_expert_num // ep_world_size
            num_recv_tokens = num_tokens * num_local_experts

            if num_tokens_per_rank is not None:
                torch._check(
                    num_tokens_per_rank.dim() == 1,
                    lambda: f"num_tokens_per_rank must be 1D tensor.{ops_error(ErrCode.VALUE)}",
                )
                torch._check(
                    num_tokens_per_rank.size(0) == ep_world_size,
                    lambda: f"num_tokens_per_rank size(0) must equal ep_world_size.{ops_error(ErrCode.VALUE)}",
                )

            if is_token_in_rank is not None:
                torch._check(
                    is_token_in_rank.dim() == 2,
                    lambda: f"is_token_in_rank must be 2D tensor.{ops_error(ErrCode.VALUE)}",
                )

            if num_tokens_per_expert is not None:
                torch._check(
                    num_tokens_per_expert.dim() == 1,
                    lambda: f"num_tokens_per_expert must be 1D tensor.{ops_error(ErrCode.VALUE)}",
                )
                torch._check(
                    num_tokens_per_expert.size(0) == moe_expert_num * round,
                    lambda: f"num_tokens_per_expert size(0) must equal moe_expert_num * round.{ops_error(ErrCode.VALUE)}",
                )

            if topk_weights is not None:
                torch._check(
                    topk_weights.dim() == 2,
                    lambda: f"topk_weights must be 2D tensor.{ops_error(ErrCode.VALUE)}",
                )

            if dispatch_wait_recv_cost_stats is not None:
                torch._check(
                    dispatch_wait_recv_cost_stats.dtype == torch.int32,
                    lambda: f"dispatch_wait_recv_cost_stats dtype must be int32.{ops_error(ErrCode.VALUE)}",
                )
                torch._check(
                    dispatch_wait_recv_cost_stats.dim() == 1,
                    lambda: f"dispatch_wait_recv_cost_stats must be 1D.{ops_error(ErrCode.VALUE)}",
                )
                torch._check(
                    dispatch_wait_recv_cost_stats.size(0) == ep_world_size,
                    lambda: (
                        "dispatch_wait_recv_cost_stats shape[0] must equal ep_world_size."
                        f"{ops_error(ErrCode.VALUE)}"
                    ),
                )

            recv_dtype = torch.int8 if quant_mode != QUANT_MODE_NO_QUANT else x.dtype
            recv_x = x.new_empty((num_recv_tokens, h), dtype=recv_dtype)
            dynamic_scales = x.new_empty((num_recv_tokens,), dtype=torch.float32)
            assist_info_for_combine = x.new_empty((num_recv_tokens * 3,), dtype=torch.int32)
            recv_count = x.new_empty((round, moe_expert_num), dtype=torch.int32)

            return recv_x, dynamic_scales, assist_info_for_combine, recv_count


moe_dispatch_prefill_op_builder = MoeDispatchPrefillOpBuilder()
op_module = moe_dispatch_prefill_op_builder.load()


@impl(AS_LIBRARY, moe_dispatch_prefill_op_builder.name, "PrivateUse1")
def npu_moe_dispatch_prefill(
    context, x, topk_idx, ext_handle_data, ep_world_size, ep_rank_id, local_ranksize, moe_expert_num,
    ccl_buffer_size, x_scales=None, topk_weights=None, num_tokens_per_rank=None, is_token_in_rank=None,
    num_tokens_per_expert=None, dispatch_wait_recv_cost_stats=None, num_worst_tokens=0, quant_mode=0,
    round=4, per_round_tokens=1024,
):
    """
    MoE EP dispatch prefill with Mc2 communication context.

    Args:
        context: Mc2 context tensor from update_context()
        x: Input tensor (num_tokens, H)
        topk_idx: Topk indices tensor (num_tokens, K)
        ext_handle_data: [notify_send_data, send_token_idx_small] from dispatch_layout
        ep_world_size: Expert parallel world size
        ep_rank_id: Current rank id in EP group
        local_ranksize: Local rank size (typically 8)
        moe_expert_num: Total number of experts
        ccl_buffer_size: CCL buffer size
        x_scales: Optional scales for quantization
        topk_weights: Optional topk weights
        num_tokens_per_rank: Per-rank token counts from dispatch_layout
        is_token_in_rank: Token-in-rank flags from dispatch_layout
        num_tokens_per_expert: Per-expert token counts from dispatch_layout
        dispatch_wait_recv_cost_stats: Optional cost statistics tensor
        num_worst_tokens: Number of worst tokens (default 0)
        quant_mode: Quantization mode (0=no quant, 2=dynamic int8)
        round: Round count (default 4)
        per_round_tokens: Tokens per round (default 1024)

    Returns:
        recv_x: Received expanded tensor
        dynamic_scales: Dynamic scales (if quantized)
        assist_info_for_combine: Assist info for combine
        recv_count: Receive count tensor
    """
    return op_module.npu_moe_dispatch_prefill(
        context, x, topk_idx, ext_handle_data, ep_world_size, ep_rank_id, local_ranksize, moe_expert_num,
        ccl_buffer_size, x_scales, topk_weights, num_tokens_per_rank, is_token_in_rank, num_tokens_per_expert,
        dispatch_wait_recv_cost_stats, num_worst_tokens, quant_mode, round, per_round_tokens,
    )
