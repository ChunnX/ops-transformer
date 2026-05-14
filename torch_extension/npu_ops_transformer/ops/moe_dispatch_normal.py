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
# from .update_context_by_hccl_channel import update_context_by_hccl_channel


class _MegaMoeOpBuilder(OpBuilder):
    def __init__(self):
        super(MoeDispatchNormalOpBuilder, self).__init__("npu_moe_dispatch_normal")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/moe_dispatch_normal.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return "npu_moe_dispatch_normal(Tensor x, Tensor topk_idx, Tensor send_offset, Tensor send_token_idx, " \
               "Tensor recv_offset, Tensor recv_count, str group_ep, int ep_world_size, int ep_rank_id, " \
               "str group_tp, int tp_world_size, int tp_rank_id, int moe_expert_num, int quant_mode, int global_bs) " \
               "-> (Tensor, Tensor, Tensor, Tensor)"

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_moe_dispatch_normal_meta(x, topk_idx, send_offset, send_token_idx, recv_offset, recv_count,
                                         group_ep, ep_world_size, ep_rank_id, group_tp, tp_world_size, 
                                         tp_rank_id, moe_expert_num, quant_mode, global_bs):
            torch._check(
                (ep_rank_id >= 0) and (ep_rank_id < ep_world_size),
                lambda: (
                    f"ep_rank_id should be in [0, ep_world_size) and ep_world_size should not be 0, "
                    f"{ops_error(ErrCode.VALUE)}."
                ),
            )
            bs = x.size(0)
            h = x.size(1)
            k = topk_idx.size(1)

            global_bs_real = global_bs if global_bs > 0 else bs * ep_world_size
            local_moe_expert_num = moe_expert_num // ep_world_size
            a = global_bs_real * min(local_moe_expert_num, k)

            real_a = a if tp_world_size == 0 else (a * tp_world_size)   #TODO
            
            output_dtype = torch.int8 if quant_mode != 0 else x.dtype   #TODO
            
            recv_x = x.new_empty((real_a, h), dtype=output_dtype)
            recv_x_scales = x.new_empty((real_a), dtype=torch.float32)
            assist_info_for_combine = x.new_empty((a * 128), dtype=torch.int32)
            wait_recv_cost_stats = x.new_empty((ep_world_size), dtype=torch.int32)

            return (recv_x, recv_x_scales, assist_info_for_combine, wait_recv_cost_stats)


moe_dispatch_normal_op_builder = MoeDispatchNormalOpBuilder()
op_module = moe_dispatch_normal_op_builder.load()


@impl(AS_LIBRARY, moe_dispatch_normal_op_builder.name, "PrivateUse1")
def npu_moe_dispatch_normal(x, topk_idx, send_offset, send_token_idx, recv_offset, recv_count,
                            group_ep, ep_world_size, ep_rank_id, group_tp, tp_world_size, 
                            tp_rank_id, moe_expert_num, quant_mode, global_bs):
    """
    Dispatcher implementation for NPU.
    'PrivateUse1' is the dispatch key for custom NPU backends.
    """
    return op_module.npu_moe_dispatch_normal(x, topk_idx, send_offset, send_token_idx, recv_offset, recv_count,
                                             group_ep, ep_world_size, ep_rank_id, group_tp, tp_world_size,
                                             tp_rank_id, moe_expert_num, quant_mode, global_bs)


try:
    import torchair
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
    from torchair._ge_concrete_graph.supported_declaration import Support
    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:
    @declare_supported([
        Support(torch.bfloat16, (8, 128)),
        Support(torch.float16, (8, 128)),
    ])
    @register_fx_node_ge_converter(torch.ops.npu_ops_transformer.npu_moe_dispatch_normal.default)
    def converter_moe_dispatch_normal(
        x: Tensor,
        topk_idx: Tensor,
        send_offset: Tensor,
        send_token_idx: Tensor,
        recv_offset: Tensor,
        recv_count: Tensor,
        group_ep: str,
        ep_world_size: int,
        ep_rank_id: int,
        group_tp: str,
        tp_world_size: int,
        tp_rank_id: int,
        moe_expert_num: int,
        quant_mode: int,
        global_bs: int,
        meta_outputs: TensorSpec = None):
        
        return ge.MoeDispatchNormal(x,
                                   topk_idx,
                                   send_offset,
                                   send_token_idx,
                                   recv_offset,
                                   recv_count,
                                   group_ep=group_ep,
                                   ep_world_size=ep_world_size,
                                   ep_rank_id=ep_rank_id,
                                   group_tp=group_tp,
                                   tp_world_size=tp_world_size,
                                   tp_rank_id=tp_rank_id,
                                   moe_expert_num=moe_expert_num,
                                   quant_mode=quant_mode,
                                   global_bs=global_bs)

