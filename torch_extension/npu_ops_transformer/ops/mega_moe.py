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
from .update_context_by_hccl_channel import update_context_by_hccl_channel


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


class IntWrapper:
    def __init__(self, value=0):
        self.value = value


class SymmBuffer:
    def __init__(
        self,
        group,
        num_experts: int,
        num_max_tokens_per_rank: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        max_recv_token_num: int = 0,
        dispatch_quant_mode: int = 0,
        dispatch_quant_out_type: int = 28,
        combine_quant_mode: int = 0,
        comm_alg: str = ""
    ):
        # Metadata
        self.group = group
        self.rank_id = torch.distributed.get_rank(group)
        self.group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank_id, init_comm=False)
        self.ccl_buffer_size = IntWrapper()
        self.ep_world_size = IntWrapper()
        context_struct_size = (2 + 1024)
        context_tensor_size = ((context_struct_size * 8) + 3) // 4
        self.context = torch.zeros(context_tensor_size, dtype=torch.int32).npu()
        update_context_by_hccl_channel(self.group_name, self.ep_world_size, self.ccl_buffer_size, self.context)
        self.num_experts = num_experts
        self.max_recv_token_num = max_recv_token_num
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.dispatch_quant_mode = dispatch_quant_mode
        self.dispatch_quant_out_type = dispatch_quant_out_type
        self.combine_quant_mode = combine_quant_mode
        self.comm_alg = comm_alg


def get_symm_buffer_for_mega_moe(
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    max_recv_token_num: int = 0,
    dispatch_quant_mode: int = 0,
    dispatch_quant_out_type: int = 28,
    combine_quant_mode: int = 0,
    comm_alg: str = ""
) -> SymmBuffer:

    return SymmBuffer(
        group,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        max_recv_token_num,
        dispatch_quant_mode,
        dispatch_quant_out_type,
        combine_quant_mode,
        comm_alg
    )


def mega_moe(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    l1_weights: List[torch.Tensor],
    l2_weights: List[torch.Tensor],
    sym_buffer: SymmBuffer,
    scales: Optional[torch.Tensor] = None,
    l1_weights_sf: Optional[List[torch.Tensor]] = None,
    l2_weights_sf: Optional[List[torch.Tensor]] = None,
    x_active_mask: Optional[torch.Tensor] = None,
    weight1_type: int = 28,
    weight2_type: int = 28
):

    return _npu_mega_moe(
        sym_buffer.context,
        x,
        topk_ids,
        topk_weights,
        l1_weights,
        l2_weights,
        sym_buffer.num_experts,
        sym_buffer.ep_world_size.value,
        sym_buffer.ccl_buffer_size.value,
        l1_weights_sf,
        l2_weights_sf,
        x_active_mask,
        scales,
        sym_buffer.max_recv_token_num,
        sym_buffer.dispatch_quant_mode,
        sym_buffer.combine_quant_mode,
        sym_buffer.comm_alg,
        sym_buffer.num_max_tokens_per_rank,
        sym_buffer.dispatch_quant_out_type,
        weight1_type,
        weight2_type
    )


