# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional, Tuple, List

import torch
import torch_npu
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error

from npu_ops_transformer.op_builder.builder import OpBuilder
from npu_ops_transformer.op_builder.builder import AS_LIBRARY


# ---------------------------------------------------------------------------
# Internal cpp op registration (private, not re-exported from this module)
# ---------------------------------------------------------------------------

class _MegaMoeOpBuilder(OpBuilder):
    def __init__(self):
        super(_MegaMoeOpBuilder, self).__init__("npu_mega_moe")

    def sources(self):
        return ['ops/csrc/mega_moe.cpp']

    def schema(self) -> str:
        return "npu_mega_moe(Tensor context, Tensor x, Tensor topk_ids, Tensor topk_weights, " \
            "Tensor[] weight1, Tensor[] weight2, int moe_expert_num, int ep_world_size, int ccl_buffer_size, *, " \
            "Tensor[]? weight_scales1=None, Tensor[]? weight_scales2=None, Tensor? x_active_mask=None, " \
            "Tensor? scales=None, int? max_recv_token_num=0, int? dispatch_quant_mode=0, " \
            "int? combine_quant_mode=0, str? comm_alg=\"\", int? global_bs=0, " \
            "int? dispatch_quant_out_type=28, int? weight1_type=28, int? weight2_type=28) -> (Tensor, Tensor)"

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_mega_moe_meta(context, x, topk_ids, topk_weights, weight1, weight2,
                                 moe_expert_num, ep_world_size, ccl_buffer_size,
                                 weight_scales1=None, weight_scales2=None, x_active_mask=None,
                                 scales=None, max_recv_token_num=0, dispatch_quant_mode=0,
                                 combine_quant_mode=0, comm_alg="", global_bs=0,
                                 dispatch_quant_out_type=28, weight1_type=28, weight2_type=28):
            torch._check(
                ep_world_size != 0,
                lambda: (
                    f"ep_rank_id should not be 0, "
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            bs = x.size(0)
            h = x.size(1)
            local_moe_expert_num = moe_expert_num // ep_world_size
            y = x.new_empty(tuple([bs, h]), dtype=x.dtype)
            expert_token_nums = x.new_empty((local_moe_expert_num), dtype=torch.int32)
            return (y, expert_token_nums)


_mega_moe_op_builder = _MegaMoeOpBuilder()
_op_module = _mega_moe_op_builder.load()


@impl(AS_LIBRARY, _mega_moe_op_builder.name, "PrivateUse1")
def _npu_mega_moe(context, x, topk_ids, topk_weights, weight1, weight2,
                     moe_expert_num, ep_world_size, ccl_buffer_size,
                     weight_scales1=None, weight_scales2=None,
                     x_active_mask=None, scales=None, max_recv_token_num=0,
                     dispatch_quant_mode=0, combine_quant_mode=0,
                     comm_alg="", global_bs=0,
                     dispatch_quant_out_type=28, weight1_type=28, weight2_type=28):
    return _op_module.npu_mega_moe(
        context, x, topk_ids, topk_weights, weight1, weight2, moe_expert_num, ep_world_size,
        ccl_buffer_size, weight_scales1, weight_scales2, x_active_mask, scales, max_recv_token_num,
        dispatch_quant_mode, combine_quant_mode, comm_alg, global_bs,
        dispatch_quant_out_type, weight1_type, weight2_type)


# ---------------------------------------------------------------------------
# Internal helpers (placeholders, to be filled with real layout/sizing logic)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SymmBuffer:
    def __init__(
        self,
        group,
        num_experts: int,
        num_max_tokens_per_rank: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        use_fp8_dispatch: bool = True,
        activation: str = "swiglu",
        *,
        context: Optional[torch.Tensor] = None,
        ccl_buffer_size: int = 0,
    ):
        # Metadata
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.use_fp8_dispatch = use_fp8_dispatch
        self.activation = activation

        # NPU-specific: HCCL context + ccl buffer size used by the fused op
        self.context = context
        self.ccl_buffer_size = ccl_buffer_size

        # Buffer / handle placeholders (kept for API parity with DeepGEMM).
        # TODO: allocate a real symmetric buffer / rendezvous on NPU.
        self.buffer = None
        self.handle = None

        # Eight tensor views (DeepGEMM ordering). Filled by the caller
        # or by a future slice routine.
        self.x: Optional[torch.Tensor] = None
        self.x_sf: Optional[torch.Tensor] = None
        self.topk_idx: Optional[torch.Tensor] = None
        self.topk_weights: Optional[torch.Tensor] = None
        self.l1_acts: Optional[torch.Tensor] = None
        self.l1_acts_sf: Optional[torch.Tensor] = None
        self.l2_acts: Optional[torch.Tensor] = None
        self.l2_acts_sf: Optional[torch.Tensor] = None

    def destroy(self):
        self.handle = None
        self.buffer = None
        self.group = None
        self.context = None
        self.x = None
        self.x_sf = None
        self.topk_idx = None
        self.topk_weights = None
        self.l1_acts = None
        self.l1_acts_sf = None
        self.l2_acts = None
        self.l2_acts_sf = None


def get_symm_buffer_for_mega_moe(
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    use_fp8_dispatch: bool = True,
    activation: str = "swiglu",
    *,
    context: Optional[torch.Tensor] = None,
    ccl_buffer_size: int = 0,
) -> SymmBuffer:
    """Public entry #1: build a `SymmBuffer` for the mega-MoE op.

    Mirrors `deep_gemm.mega.get_symm_buffer_for_mega_moe`.
    """
    # Token count must be aligned to block sizes
    alignment = _get_token_alignment_for_mega_moe()
    aligned_max = ((num_max_tokens_per_rank + alignment - 1) // alignment) * alignment

    return SymmBuffer(
        group,
        num_experts,
        aligned_max,
        num_topk,
        hidden,
        intermediate_hidden,
        use_fp8_dispatch,
        activation,
        context=context,
        ccl_buffer_size=ccl_buffer_size,
    )


def transform_weights_for_mega_moe(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
    """Public entry #2: re-layout the weights for the mega-MoE op.

    NOTE: NPU-side layout transforms are placeholders and currently
    return the inputs unchanged.
    """
    l1_interleaved = _interleave_l1_weights(l1_weights)
    l1_weights = (l1_interleaved[0], _transpose_sf_for_utccp(l1_interleaved[1]))
    l2_weights = (l2_weights[0], _transpose_sf_for_utccp(l2_weights[1]))
    return l1_weights, l2_weights


def fp8_fp4_mega_moe(
    y: Optional[torch.Tensor],
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer: SymmBuffer,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    recipe: Tuple[int, int, int] = (1, 1, 32),
    activation: str = "swiglu",
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    # NPU-specific passthrough parameters (mirrors `npu_mega_moe_v2` schema).
    *,
    x: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    weight_scales1: Optional[List[torch.Tensor]] = None,
    weight_scales2: Optional[List[torch.Tensor]] = None,
    x_active_mask: Optional[torch.Tensor] = None,
    scales: Optional[torch.Tensor] = None,
    max_recv_token_num: int = 0,
    dispatch_quant_mode: int = 0,
    combine_quant_mode: int = 0,
    comm_alg: str = "",
    global_bs: int = 0,
    dispatch_quant_out_type: int = 28,
    weight1_type: int = 28,
    weight2_type: int = 28,
):
    """Public entry #3: invoke the fused NPU mega-MoE operator.

    Internally delegates to the v2 cpp op registered in this module.
    """
    # Resolve input tensors: prefer explicit kwargs, fall back to SymmBuffer fields.
    x_in = x if x is not None else sym_buffer.x
    topk_ids_in = topk_ids if topk_ids is not None else sym_buffer.topk_idx
    topk_weights_in = topk_weights if topk_weights is not None else sym_buffer.topk_weights

    # Materialize weight lists (the cpp op expects TensorList)
    w1 = list(l1_weights) if not isinstance(l1_weights, list) else l1_weights
    w2 = list(l2_weights) if not isinstance(l2_weights, list) else l2_weights

    # Resolve ep world size
    if hasattr(sym_buffer.group, "size"):
        ep_world_size = sym_buffer.group.size()
    else:
        ep_world_size = int(sym_buffer.group)

    y_out, expert_token_nums = _npu_mega_moe_v2(
        sym_buffer.context,
        x_in,
        topk_ids_in,
        topk_weights_in,
        w1,
        w2,
        sym_buffer.num_experts,
        ep_world_size,
        sym_buffer.ccl_buffer_size,
        weight_scales1,
        weight_scales2,
        x_active_mask,
        scales,
        max_recv_token_num,
        dispatch_quant_mode,
        combine_quant_mode,
        comm_alg,
        global_bs,
        dispatch_quant_out_type,
        weight1_type,
        weight2_type,
    )

    if y is not None:
        y.copy_(y_out)
        return expert_token_nums
    return y_out, expert_token_nums


__all__ = [
    "SymmBuffer",
    "get_symm_buffer_for_mega_moe",
    "transform_weights_for_mega_moe",
    "fp8_fp4_mega_moe",
]
