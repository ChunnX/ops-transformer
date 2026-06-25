# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch
import torch_npu
from typing import Optional, Tuple
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY


class _MoeReRoutingOpBuilder(OpBuilder):
    def __init__(self):
        super(_MoeReRoutingOpBuilder, self).__init__("moe_re_routing")

    def sources(self):
        return ['ops/csrc/moe_re_routing.cpp']

    def schema(self) -> str:
        return "moe_re_routing(Tensor tokens, Tensor expert_token_num_per_rank, " \
            "*, Tensor? per_token_scales=None, Tensor? expert_topk_weight=None, " \
            "int expert_token_num_type=1, int idx_type=0) " \
            "-> (Tensor, Tensor, Tensor, Tensor, Tensor)"

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def moe_re_routing_meta(tokens, expert_token_num_per_rank,
                                     per_token_scales=None, expert_topk_weight=None,
                                     expert_token_num_type=1, idx_type=0):
            torch._check(tokens.dim() == 2,
                lambda: f"tokens must be 2D, but got {tokens.dim()}D.{ops_error(ErrCode.VALUE)}")
            torch._check(expert_token_num_per_rank.dim() == 2,
                lambda: f"expert_token_num_per_rank must be 2D, but got {expert_token_num_per_rank.dim()}D.{ops_error(ErrCode.VALUE)}")
            torch._check(expert_token_num_type in (0, 1),
                lambda: f"expert_token_num_type must be 0 or 1, got {expert_token_num_type}.{ops_error(ErrCode.VALUE)}")
            torch._check(idx_type in (0, 1),
                lambda: f"idx_type must be 0 or 1, got {idx_type}.{ops_error(ErrCode.VALUE)}")

            A = tokens.size(0)
            H = tokens.size(1)
            E = expert_token_num_per_rank.size(1)

            permute_tokens = tokens.new_empty((A, H), dtype=tokens.dtype)

            if per_token_scales is not None:
                permute_per_token_scales = per_token_scales.new_empty(per_token_scales.shape,
                                                                       dtype=per_token_scales.dtype)
            else:
                permute_per_token_scales = tokens.new_empty((A,), dtype=torch.float32)

            permute_token_idx = tokens.new_empty((A,), dtype=torch.int32)
            expert_token_num = expert_token_num_per_rank.new_empty((E,),
                                                                    dtype=expert_token_num_per_rank.dtype)

            if expert_topk_weight is not None:
                permute_topk_weight = expert_topk_weight.new_empty(expert_topk_weight.shape,
                                                                    dtype=expert_topk_weight.dtype)
            else:
                permute_topk_weight = tokens.new_empty((0,), dtype=torch.float32)

            return (permute_tokens, permute_per_token_scales, permute_token_idx,
                    expert_token_num, permute_topk_weight)


_moe_re_routing_op_builder = _MoeReRoutingOpBuilder()
_op_module = _moe_re_routing_op_builder.load()


@impl(AS_LIBRARY, _moe_re_routing_op_builder.name, "PrivateUse1")
def moe_re_routing_dispatch(tokens, expert_token_num_per_rank,
                                 per_token_scales=None, expert_topk_weight=None,
                                 expert_token_num_type=1, idx_type=0):
    return _op_module.moe_re_routing(tokens, expert_token_num_per_rank, per_token_scales,
                                          expert_topk_weight, expert_token_num_type, idx_type)


def moe_re_routing(tokens: torch.Tensor, expert_token_num_per_rank: torch.Tensor,
                   per_token_scales: Optional[torch.Tensor] = None,
                   expert_topk_weight: Optional[torch.Tensor] = None,
                   expert_token_num_type: int = 1, idx_type: int = 0
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """moe_re_routing operator.

    Rearranges tokens from rank-order to expert-order for MoE (Mixture of Experts)
    distributed inference.

    Args:
        tokens: Input tensor of shape (A, H), dtype supports float16/bfloat16/int8/
            float8_e5m2/float8_e4m3fn/hifloat8/float4_e2m1/float4_e1m2.
        expert_token_num_per_rank: Token counts per expert per rank of shape (N, E),
            dtype int32 or int64.
        per_token_scales: Optional scale tensor. Shape (A) for float32,
            (A, S) for float32, (A, K/64, 2) for float8_e8m0 with FP8 tokens.
        expert_topk_weight: Optional topk weight tensor of shape (n*k, 1), dtype float32.
            When provided, permute_topk_weight output will be produced.
        expert_token_num_type: Type of expert_token_num output (0=cumsum, 1=count).
            Default: 1 (count).
        idx_type: Index type (0=gather, 1=scatter). Default: 0 (gather).
            Scatter index only supported on Ascend 950.

    Returns:
        A tuple of (permute_tokens, permute_per_token_scales, permute_token_idx,
                    expert_token_num, permute_topk_weight).
    """
    return torch.ops.cann_ops_transformer.moe_re_routing(
        tokens, expert_token_num_per_rank, per_token_scales=per_token_scales,
        expert_topk_weight=expert_topk_weight, expert_token_num_type=expert_token_num_type,
        idx_type=idx_type)
