# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import torch
import torch_npu
from typing import Optional, Tuple, List
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY


class _MoeInitRoutingOpBuilder(OpBuilder):
    def __init__(self):
        super(_MoeInitRoutingOpBuilder, self).__init__("npu_moe_init_routing")

    def sources(self):
        return ['ops/csrc/moe_init_routing.cpp']

    def schema(self) -> str:
        return "npu_moe_init_routing(Tensor x, Tensor expert_idx, " \
            "*, Tensor? scale=None, Tensor? offset=None, Tensor? topk_weight=None, " \
            "int active_num=-1, int expert_capacity=-1, int expert_num=-1, " \
            "int drop_pad_mode=0, int expert_tokens_num_type=0, bool expert_tokens_num_flag=False, " \
            "int quant_mode=-1, int[]? active_expert_range=None, int row_idx_type=0) " \
            "-> (Tensor, Tensor, Tensor, Tensor, Tensor)"

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_moe_init_routing_meta(x, expert_idx, scale=None, offset=None, topk_weight=None,
                                       active_num=-1, expert_capacity=-1, expert_num=-1,
                                       drop_pad_mode=0, expert_tokens_num_type=0, expert_tokens_num_flag=False,
                                       quant_mode=-1, active_expert_range=None, row_idx_type=0):
            torch._check(x.dim() == 2,
                lambda: f"x must be 2D, but got {x.dim()}D.{ops_error(ErrCode.VALUE)}")
            torch._check(expert_idx.dim() == 2,
                lambda: f"expert_idx must be 2D, but got {expert_idx.dim()}D.{ops_error(ErrCode.VALUE)}")

            n = x.size(0)
            cols = x.size(1)
            k = expert_idx.size(1)

            active_num_real = n * k if active_num < 0 else active_num
            out_num = active_num_real

            expanded_x_rows = out_num
            if drop_pad_mode == 1:
                expanded_x_rows = expert_num * expert_capacity

            out_dtype = torch.int8
            if quant_mode < 0 or quant_mode == 0:
                out_dtype = x.dtype
            elif quant_mode == 4:
                out_dtype = torch.float8_e5m2

            expanded_x = x.new_empty((expanded_x_rows, cols), dtype=out_dtype)
            expanded_row_idx = x.new_empty((expanded_x_rows), dtype=torch.int32)

            expert_tokens_count_size = expert_num
            if expert_tokens_num_type == 1 and expert_tokens_num_flag:
                expert_tokens_count_size = expert_num + 1
            expert_tokens_count_or_cumsum = x.new_empty((expert_tokens_count_size), dtype=torch.int64)

            if quant_mode < 0 or quant_mode == 0 or quant_mode == 1:
                expanded_scale = x.new_empty((out_num), dtype=torch.float32)
            elif quant_mode == 4:
                cols_aligned = (cols + 255) // 256
                expanded_scale = x.new_empty((out_num, cols_aligned, 2), dtype=torch.float8_e5m2)
            else:
                expanded_scale = x.new_empty((0), dtype=torch.float32)

            if topk_weight is not None:
                topk_weight_rows = out_num
                if drop_pad_mode == 1:
                    topk_weight_rows = expert_num * expert_capacity
                expanded_topk_weight = x.new_empty((topk_weight_rows, 1), dtype=torch.float32)
            else:
                expanded_topk_weight = x.new_empty((0), dtype=torch.float32)

            return (expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum,
                    expanded_scale, expanded_topk_weight)


_moe_init_routing_op_builder = _MoeInitRoutingOpBuilder()
_op_module = _moe_init_routing_op_builder.load()


@impl(AS_LIBRARY, _moe_init_routing_op_builder.name, "PrivateUse1")
def npu_moe_init_routing_dispatch(x, expert_idx, scale=None, offset=None, topk_weight=None,
                                    active_num=-1, expert_capacity=-1, expert_num=-1,
                                    drop_pad_mode=0, expert_tokens_num_type=0, expert_tokens_num_flag=False,
                                    quant_mode=-1, active_expert_range=None, row_idx_type=0):
    return _op_module.npu_moe_init_routing(x, expert_idx, scale, offset, topk_weight,
                                            active_num, expert_capacity, expert_num, drop_pad_mode,
                                            expert_tokens_num_type, expert_tokens_num_flag, quant_mode,
                                            active_expert_range, row_idx_type)


def moe_init_routing(x: torch.Tensor, expert_idx: torch.Tensor,
                     scale: Optional[torch.Tensor] = None, offset: Optional[torch.Tensor] = None,
                     topk_weight: Optional[torch.Tensor] = None,
                     active_num: int = -1, expert_capacity: int = -1, expert_num: int = -1,
                     drop_pad_mode: int = 0, expert_tokens_num_type: int = 0,
                     expert_tokens_num_flag: bool = False, quant_mode: int = -1,
                     active_expert_range: Optional[List[int]] = None, row_idx_type: int = 0
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """moe_init_routing operator.

    Performs MoE (Mixture of Experts) init routing, expanding tokens to their
    assigned experts and producing routing indices.

    Args:
        x: Input tensor of shape (n, cols), dtype supports float16/bfloat16/float32/int8.
        expert_idx: Expert indices of shape (n, k), dtype int32.
        scale: Optional scale tensor for quantization.
        offset: Optional offset tensor for quantization.
        topk_weight: Optional topk weight tensor of shape (n, k), dtype float32.
            When provided, expanded_topk_weight output will also be produced.
        active_num: Number of active tokens. -1 means n*k (default).
        expert_capacity: Capacity per expert (used in drop_pad_mode).
        expert_num: Total number of experts.
        drop_pad_mode: 0 for no drop/pad, 1 for drop/pad mode.
        expert_tokens_num_type: Type of expert token count output (0=count, 1=cumsum).
        expert_tokens_num_flag: Whether to include extra token in count output.
        quant_mode: Quantization mode (-1=none, 0=none, 1=static, 2=dynamic, etc).
        active_expert_range: Optional range of active experts [start, end].
        row_idx_type: Row index type (0=gather, 1=scatter).

    Returns:
        A tuple of (expanded_x, expanded_row_idx, expert_tokens_count_or_cumsum,
                    expanded_scale, expanded_topk_weight).
    """
    return torch.ops.cann_ops_transformer.npu_moe_init_routing(
        x, expert_idx, scale=scale, offset=offset, topk_weight=topk_weight,
        active_num=active_num, expert_capacity=expert_capacity, expert_num=expert_num,
        drop_pad_mode=drop_pad_mode, expert_tokens_num_type=expert_tokens_num_type,
        expert_tokens_num_flag=expert_tokens_num_flag, quant_mode=quant_mode,
        active_expert_range=active_expert_range, row_idx_type=row_idx_type)
