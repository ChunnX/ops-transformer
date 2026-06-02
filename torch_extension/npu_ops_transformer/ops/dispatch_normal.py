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


class DispatchNormalOpBuilder(OpBuilder):
    def __init__(self):
        super(DispatchNormalOpBuilder, self).__init__("npu_dispatch_normal")

    def sources(self):
        return ['ops/csrc/dispatch_normal.cpp']

    def schema(self) -> str:
        return (
            "npu_dispatch_normal(Tensor context, Tensor x, Tensor topk_idx, Tensor send_offset, Tensor send_token_idx, "
            "Tensor[] recv_offset, Tensor[] expert_output_offset, int ep_world_size, int ep_rank_id, "
            "int moe_expert_num, int ccl_buffer_size, int num_recv_tokens, *, int quant_mode=0, int real_max_bs=0, "
            "int global_bs=0, int round=4, int per_round_tokens=1024, "
            "Tensor? dispatch_wait_recv_cost_stats=None, Tensor? ext_handle_data=None) -> (Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_dispatch_normal_meta(
            context, x, topk_idx, send_offset, send_token_idx, recv_offset, expert_output_offset,
            ep_world_size, ep_rank_id, moe_expert_num,
            ccl_buffer_size, num_recv_tokens, quant_mode=0, real_max_bs=0, global_bs=0, round=4,
            per_round_tokens=1024, dispatch_wait_recv_cost_stats=None, ext_handle_data=None,
        ):
            torch._check(
                len(recv_offset) == 2,
                lambda: f"recv_offset must contain exactly 2 tensors [recv_offset, recv_count].{ops_error(ErrCode.VALUE)}",
            )
            torch._check(
                len(expert_output_offset) == 3,
                lambda: (
                    "expert_output_offset must contain exactly 3 tensors "
                    "[expert_global_offset, srcrank_in_expert_offset, r_in_srcrank_offset]."
                    f"{ops_error(ErrCode.VALUE)}"
                ),
            )
            recv_offset_tensor, recv_count_tensor = recv_offset
            expert_global_offset_tensor, srcrank_in_expert_offset_tensor, r_in_srcrank_offset_tensor = (
                expert_output_offset
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
                recv_offset_tensor.dim() == 1
                and recv_count_tensor.dim() == 1
                and expert_global_offset_tensor.dim() == 1
                and srcrank_in_expert_offset_tensor.dim() == 1
                and r_in_srcrank_offset_tensor.dim() == 1,
                lambda: (
                    f"all tensors in recv_offset and expert_output_offset must be 1D.{ops_error(ErrCode.VALUE)}"
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
            recv_offset_expected_numel = 2 * round * moe_expert_num
            expert_output_expected_numel = (
                moe_expert_num // ep_world_size + moe_expert_num + round * moe_expert_num
            )
            recv_offset_numel = recv_offset_tensor.numel() + recv_count_tensor.numel()
            expert_output_numel = (
                expert_global_offset_tensor.numel()
                + srcrank_in_expert_offset_tensor.numel()
                + r_in_srcrank_offset_tensor.numel()
            )
            torch._check(
                recv_offset_numel == recv_offset_expected_numel,
                lambda: (
                    f"recv_offset numel mismatch, expected {recv_offset_expected_numel} "
                    f"(=2*round*moe_expert_num), but got {recv_offset_numel}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                expert_output_numel == expert_output_expected_numel,
                lambda: (
                    f"expert_output_offset numel mismatch, expected {expert_output_expected_numel} "
                    f"(=moe_expert_num/ep_world_size + moe_expert_num + round*moe_expert_num), "
                    f"but got {expert_output_numel}."
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            torch._check(
                num_recv_tokens > 0,
                lambda: f"num_recv_tokens must be positive.{ops_error(ErrCode.VALUE)}",
            )
            torch._check(
                global_bs > 0,
                lambda: f"global_bs must be positive.{ops_error(ErrCode.VALUE)}",
            )
            torch._check(
                ccl_buffer_size > 0,
                lambda: f"ccl_buffer_size must be positive.{ops_error(ErrCode.VALUE)}",
            )
            if dispatch_wait_recv_cost_stats is not None:
                torch._check(
                    dispatch_wait_recv_cost_stats.dtype == torch.int32,
                    lambda: (
                        f"dispatch_wait_recv_cost_stats dtype must be int32.{ops_error(ErrCode.VALUE)}"
                    ),
                )
                torch._check(
                    dispatch_wait_recv_cost_stats.dim() == 1,
                    lambda: (
                        f"dispatch_wait_recv_cost_stats must be 1D.{ops_error(ErrCode.VALUE)}"
                    ),
                )
                torch._check(
                    dispatch_wait_recv_cost_stats.size(0) == ep_world_size,
                    lambda: (
                        "dispatch_wait_recv_cost_stats shape[0] must equal ep_world_size."
                        f"{ops_error(ErrCode.VALUE)}"
                    ),
                )
            if ext_handle_data is not None:
                torch._check(
                    ext_handle_data.dtype == torch.int32,
                    lambda: (
                        f"ext_handle_data dtype must be int32.{ops_error(ErrCode.VALUE)}"
                    ),
                )
                torch._check(
                    ext_handle_data.dim() == 1,
                    lambda: (
                        f"ext_handle_data must be 1D.{ops_error(ErrCode.VALUE)}"
                    ),
                )
                torch._check(
                    ext_handle_data.size(0) == 0,
                    lambda: (
                        f"ext_handle_data shape[0] must be 0.{ops_error(ErrCode.VALUE)}"
                    ),
                )

            h = x.size(1)
            recv_dtype = torch.int8 if quant_mode != QUANT_MODE_NO_QUANT else x.dtype
            recv_x = x.new_empty((num_recv_tokens, h), dtype=recv_dtype)
            x_scales = x.new_empty((num_recv_tokens,), dtype=torch.float32)
            assist_info_for_combine = x.new_empty((num_recv_tokens * 3,), dtype=torch.int32)
            ext_handle_data_out = (
                ext_handle_data if ext_handle_data is not None else x.new_empty((0,), dtype=torch.int32)
            )
            return recv_x, x_scales, assist_info_for_combine, ext_handle_data_out


dispatch_normal_op_builder = DispatchNormalOpBuilder()
op_module = dispatch_normal_op_builder.load()


@impl(AS_LIBRARY, dispatch_normal_op_builder.name, "PrivateUse1")
def npu_dispatch_normal(
    context, x, topk_idx, send_offset, send_token_idx, recv_offset, expert_output_offset, ep_world_size, ep_rank_id,
    moe_expert_num, ccl_buffer_size,
    num_recv_tokens, quant_mode=0, real_max_bs=0, global_bs=0, round=4, per_round_tokens=1024,
    dispatch_wait_recv_cost_stats=None, ext_handle_data=None,
):
    """
    MoE EP dispatch (normal path) with Mc2 communication context.

    Prepare context via update_context(group_name, ep_world_size, ccl_buffer_size, context_tensor) first.
    Typical flow: notify_dispatch -> npu_dispatch_normal -> expert compute -> combine.

    Returns:
        recv_x: expanded tokens (num_recv_tokens, H)
        x_scales: per-token scales (num_recv_tokens,)
        assist_info_for_combine: 1D int32, numel = num_recv_tokens * 3 (combine token_src_info)
        ext_handle_data: placeholder 1D int32 with numel 0 (not recv_count; use recv_offset[1] from notify)
    """
    return op_module.npu_dispatch_normal(
        context, x, topk_idx, send_offset, send_token_idx, recv_offset, expert_output_offset, ep_world_size,
        ep_rank_id, moe_expert_num, ccl_buffer_size,
        num_recv_tokens, quant_mode, real_max_bs, global_bs, round, per_round_tokens, dispatch_wait_recv_cost_stats,
        ext_handle_data,
    )
