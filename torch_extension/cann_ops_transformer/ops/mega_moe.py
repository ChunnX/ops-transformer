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
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY
from .comm_context import CommContextManager


class _MegaMoeOpBuilder(OpBuilder):
    def __init__(self):
        super(_MegaMoeOpBuilder, self).__init__("npu_mega_moe")

    def sources(self):
        return ['ops/csrc/mega_moe.cpp']

    def schema(self) -> str:
        return "npu_mega_moe(Tensor context, Tensor x, Tensor topk_ids, Tensor topk_weights, " \
            "Tensor[] weight1, Tensor[] weight2, int moe_expert_num, int ep_world_size, int ccl_buffer_size, *, " \
            "Tensor[]? weight_scales1=None, Tensor[]? weight_scales2=None, " \
            "Tensor[]? bias1=None, Tensor[]? bias2=None, " \
            "Tensor? x_active_mask=None, int max_recv_token_num=0, " \
            "int dispatch_quant_mode=0, int combine_quant_mode=0, " \
            "str comm_alg=\"\", int num_max_tokens_per_rank=0, str activation=\"swiglu\", " \
            "float? activation_clamp=None, " \
            "int? dispatch_quant_out_dtype=None,  " \
            "int? weight1_type=None, int? weight2_type=None, int? topo_type=None, " \
            "int? rank_num_per_server=None) -> (Tensor, Tensor)"    

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_mega_moe_meta(context, x, topk_ids, topk_weights, weight1, weight2,
                              moe_expert_num, ep_world_size, ccl_buffer_size,
                              weight_scales1=None, weight_scales2=None, bias1=None, bias2=None,
                              x_active_mask=None, max_recv_token_num=0, dispatch_quant_mode=0,
                              combine_quant_mode=0, comm_alg="",
                              num_max_tokens_per_rank=0, activation="swiglu", activation_clamp=None,
                              dispatch_quant_out_dtype=None,
                              weight1_type=None, weight2_type=None, topo_type=None, rank_num_per_server=None):
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


@impl(AS_LIBRARY, _mega_moe_op_builder.name, "PrivateUse1")
def _npu_mega_moe(context, x, topk_ids, topk_weights, weight1, weight2,
                  moe_expert_num, ep_world_size, ccl_buffer_size,
                  weight_scales1=None, weight_scales2=None, bias1=None, bias2=None,
                  x_active_mask=None, max_recv_token_num=0, dispatch_quant_mode=0,
                  combine_quant_mode=0, comm_alg="",
                  num_max_tokens_per_rank=0, activation="swiglu", activation_clamp=None,
                  dispatch_quant_out_dtype=None,
                   weight1_type=None, weight2_type=None, topo_type=None, rank_num_per_server=None):
    _op_module = _mega_moe_op_builder.load()
    return _op_module.npu_mega_moe(
        context, x, topk_ids, topk_weights, weight1, weight2, moe_expert_num, ep_world_size,
        ccl_buffer_size, weight_scales1, weight_scales2, bias1, bias2, x_active_mask,
        max_recv_token_num, dispatch_quant_mode, combine_quant_mode, comm_alg,
        num_max_tokens_per_rank, activation, activation_clamp, dispatch_quant_out_dtype,
        weight1_type, weight2_type, topo_type, rank_num_per_server)


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
        dispatch_quant_out_dtype: Optional[int] = None,
        combine_quant_mode: int = 0,
        comm_alg: str = ""
    ):
        # Metadata
        self.group = group
        self.rank_id = torch.distributed.get_rank(group)
        self.group_name = group._get_backend(torch.device("npu")).get_hccl_comm_name(self.rank_id, init_comm=False)
        self.ep_world_size = torch.distributed.get_world_size(group)
        self._ctx_manager = CommContextManager(self.group_name, self.ep_world_size, backend={
            "Ascend910B": "kfc",
            "Ascend910_93": "kfc",
            "Ascend950": "channel"
        })
        self.context = self._ctx_manager.create_context()
        self.ccl_buffer_size = self._ctx_manager.ccl_buffer_size
        self.num_experts = num_experts
        self.max_recv_token_num = max_recv_token_num
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.dispatch_quant_mode = dispatch_quant_mode
        self.dispatch_quant_out_dtype = dispatch_quant_out_dtype
        self.combine_quant_mode = combine_quant_mode
        self.comm_alg = comm_alg
        self.topo_type = self._ctx_manager.topo_type
        self.rank_num_per_server = self._ctx_manager.rank_num_per_server


def get_mega_moe_ccl_buffer_size(
    ep_world_size: int, moe_expert_num: int, num_max_tokens_per_rank: int,
    num_topk: int, hidden: int,
    dispatch_quant_mode: int = 0, dispatch_quant_out_dtype: int = 28,
    combine_quant_mode: int = 0, comm_alg: str = ""
) -> int:
    def inline_align(val, align):
        return (val + align - 1) // align * align
    torch._check(((ep_world_size >= 2) and (ep_world_size <= 1024)),
        lambda: (f"ep_world_size only support in [2, 1024], but got {ep_world_size=}."))
    torch._check(((hidden >= 1024) and (hidden <= 8192)),
        lambda: (f"hidden only support in [1024, 8192], but got {hidden=}."))
    total_ub_size = 256 * 1024 # 950系列ubSize总大小为256K
    tokens_per_rank_limit = (8 * (total_ub_size - 48 * 1024)) // (num_topk * (64 + ep_world_size))
    torch._check(((num_max_tokens_per_rank >= 1) and (num_max_tokens_per_rank <= tokens_per_rank_limit)),
        lambda: (f"num_max_tokens_per_rank only support in [1, {tokens_per_rank_limit=}], "
                    f"tokens_per_rank_limit = (8 * (total_ub_size - 48 * 1024)) / (num_topk * (64 + ep_world_size)), "
                    f"but got {num_max_tokens_per_rank=}."))
    torch._check(((moe_expert_num >= 1) and (moe_expert_num <= 2048)),
        lambda: (f"moe_expert_num only support in [1, 2048], but got {moe_expert_num=}."))
    torch._check(((num_topk >= 1) and (num_topk <= 16)),
        lambda: (f"num_topk only support in [1, 16], but got {num_topk=}."))
    local_moe_expert_num = moe_expert_num // ep_world_size
    align_32 = 32
    align_256 = 256
    align_512 = 512
    y_out_dtype_size = 2
    mb_conversion = 1024 * 1024
    # 全卡软同步使用 60M
    peermem_data_offset = 60 * 1024
    # mask_recv_size 
    compare_count = inline_align(num_max_tokens_per_rank * num_topk * 4, align_256) // 4 # 4 = sizeof(int32)
    mask_align_size = inline_align(compare_count // 8, align_32) # 8 = align_32 / sizeof(int32)
    mask_slot_size = mask_align_size + align_32 # 后32字节存count数
    mask_recv_size = inline_align(local_moe_expert_num * ep_world_size * mask_slot_size, align_512)
    # quant_token_scale_size
    mx_scale_num = (hidden + align_32 - 1) // align_32
    data_bytes = inline_align(hidden, align_256) # 单token 256字节对齐，当前量化为mxfp8
    token_bytes = inline_align(data_bytes + mx_scale_num, align_32) # token拼接scale长度
    quant_token_scale_size = inline_align(num_max_tokens_per_rank * token_bytes, align_512)
    # combine_send_size
    combine_out = inline_align(num_max_tokens_per_rank * hidden * num_topk * y_out_dtype_size, align_512)
    # 所需总大小
    ccl_buffer_size = peermem_data_offset + mask_recv_size + quant_token_scale_size + combine_out
    ccl_buffer_size = inline_align(inline_align(ccl_buffer_size, mb_conversion) // mb_conversion, 2) // 2

    return ccl_buffer_size


def get_symm_buffer_for_mega_moe(
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    *,
    max_recv_token_num: int = 0,
    dispatch_quant_mode: int = 0,
    dispatch_quant_out_dtype: Optional[int] = None,
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
        dispatch_quant_out_dtype,
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
    *,
    l1_weights_sf: Optional[List[torch.Tensor]] = None,
    l2_weights_sf: Optional[List[torch.Tensor]] = None,
    l1_bias: Optional[List[torch.Tensor]] = None,
    l2_bias: Optional[List[torch.Tensor]] = None,
    x_active_mask: Optional[torch.Tensor] = None,
    activation: str = "swiglu",
    activation_clamp: Optional[float] = None,
    weight1_type: Optional[int] = None,
    weight2_type: Optional[int] = None,
):

    return torch.ops.cann_ops_transformer.npu_mega_moe(
        sym_buffer.context,
        x,
        topk_ids,
        topk_weights,
        l1_weights,
        l2_weights,
        sym_buffer.num_experts,
        sym_buffer.ep_world_size,
        sym_buffer.ccl_buffer_size,
        weight_scales1=l1_weights_sf,
        weight_scales2=l2_weights_sf,
        bias1=l1_bias,
        bias2=l2_bias,
        x_active_mask=x_active_mask,
        max_recv_token_num=sym_buffer.max_recv_token_num,
        dispatch_quant_mode=sym_buffer.dispatch_quant_mode,
        combine_quant_mode=sym_buffer.combine_quant_mode,
        comm_alg=sym_buffer.comm_alg,
        num_max_tokens_per_rank=sym_buffer.num_max_tokens_per_rank,
        activation=activation,
        activation_clamp=activation_clamp,
        dispatch_quant_out_dtype=sym_buffer.dispatch_quant_out_dtype,
        weight1_type=weight1_type,
        weight2_type=weight2_type,
        topo_type=sym_buffer.topo_type,
        rank_num_per_server=sym_buffer.rank_num_per_server
    )