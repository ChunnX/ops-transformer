# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
try:
    import torch
    import torch_npu
    import torchair
    from torch.library import impl
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
    from torchair._ge_concrete_graph.supported_declaration import Support
    from typing import Any, Dict, List, Tuple, Union, Callable, Optional
    from torchair._ge_concrete_graph.ge_ir_pb2 import GraphDef, OpDef, TensorDescriptor, TensorDef
    from torchair.ge._ge_graph import get_default_ge_graph, next_unique_name
    from torchair.ge._ge_graph import auto_convert_to_tensor
    from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, TensorType
    from torchair.ge._ge_graph import compat_as_bytes, compat_as_bytes_list
    from torchair.ge._ge_graph import trans_to_list_list_int, trans_to_list_list_float
    from torchair.ge._ge_graph import get_invalid_desc
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair.ge import attr
    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:
    @auto_convert_to_tensor([False, False, False, False, False, False],
        [False, False, False, False, False, False])
    def MoeDispatchNormal(x: Tensor, topk_idx: Tensor, send_offset: Tensor, send_token_idx: Tensor,
                          recv_offset: Tensor, recv_count: Tensor, *, 
                          group_ep: str, ep_world_size: int, ep_rank_id: int,
                          group_tp: str, tp_world_size: int, tp_rank_id: int, 
                          moe_expert_num: int, quant_mode: int, global_bs: int,
                          dependencies=[], node_name=None):
        """REG_OP(MoeDispatchNormal)\n
        .INPUT(x, TensorType({DT_BF16, DT_FLOAT16}))\n
        .INPUT(topk_idx, TensorType({DT_INT32}))\n
        .INPUT(send_offset, TensorType({DT_INT32}))\n
        .INPUT(send_token_idx, TensorType({DT_INT32}))\n
        .INPUT(recv_offset, TensorType({DT_INT32}))\n
        .INPUT(recv_count, TensorType({DT_INT32}))\n
        .OUTPUT(recv_x, TensorType({DT_BF16, DT_INT8, DT_FLOAT16}))\n
        .OUTPUT(recv_x_scales, TensorType({DT_FLOAT}))\n
        .OUTPUT(assist_info_for_combine, TensorType({DT_INT32}))\n
        .OUTPUT(wait_recv_cost_stats, TensorType({DT_INT32}))\n
        .REQUIRED_ATTR(group_ep, String)\n
        .REQUIRED_ATTR(ep_world_size, Int)\n
        .REQUIRED_ATTR(ep_rank_id, Int)\n
        .ATTR(group_tp, String, "")\n
        .ATTR(tp_world_size, Int, 0)\n
        .ATTR(tp_rank_id, Int, 0)\n
        .REQUIRED_ATTR(moe_expert_num, Int)\n
        .ATTR(quant_mode, Int, 0)\n
        .ATTR(global_bs, Int, 0)\n
        """
        inputs = {
            "x": x,
            "topk_idx": topk_idx,
            "send_offset": send_offset,
            "send_token_idx": send_token_idx,
            "recv_offset": recv_offset,
            "recv_count": recv_count,
        }

        attrs = {
            "group_ep": attr.Str(group_ep),
            "ep_world_size": attr.Int(ep_world_size),
            "ep_rank_id": attr.Int(ep_rank_id),
            "group_tp": attr.Str(group_tp),
            "tp_world_size": attr.Int(tp_world_size),
            "tp_rank_id": attr.Int(tp_rank_id),
            "moe_expert_num": attr.Int(moe_expert_num),
            "quant_mode": attr.Int(quant_mode),
            "global_bs": attr.Int(global_bs),
        }

        outputs = [
            "recv_x",
            "recv_x_scales",
            "assist_info_for_combine",
            "wait_recv_cost_stats",
        ]

        return ge_op(
            op_type="MoeDispatchNormal",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("MoeDispatchNormal") \
            .input("x", "DT_BF16, DT_FLOAT16") \
            .input("topk_idx", "DT_INT32") \
            .input("send_offset", "DT_INT32") \
            .input("send_token_idx", "DT_INT32") \
            .input("recv_offset", "DT_INT32") \
            .input("recv_count", "DT_INT32") \
            .required_attr("group_ep", attr.Str) \
            .required_attr("ep_world_size", attr.Int) \
            .required_attr("ep_rank_id", attr.Int) \
            .attr("group_tp", attr.Str("")) \
            .attr("tp_world_size", attr.Int(0)) \
            .attr("tp_rank_id", attr.Int(0)) \
            .required_attr("moe_expert_num", attr.Int) \
            .attr("quant_mode", attr.Int(0)) \
            .attr("global_bs", attr.Int(0)) \
            .output("recv_x", "DT_BF16, DT_INT8, DT_FLOAT16") \
            .output("recv_x_scales", "DT_FLOAT") \
            .output("assist_info_for_combine", "DT_INT32") \
            .output("wait_recv_cost_stats", "DT_INT32")
        )

    @register_fx_node_ge_converter(torch.ops.npu_ops_transformer.npu_moe_dispatch_normal.default)
    def convert_npu_moe_dispatch_normal(
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
        
        return MoeDispatchNormal(x=x,
                                 topk_idx=topk_idx,
                                 send_offset=send_offset,
                                 send_token_idx=send_token_idx,
                                 recv_offset=recv_offset,
                                 recv_count=recv_count,
                                 group_ep=group_ep,
                                 ep_world_size=ep_world_size,
                                 ep_rank_id=ep_rank_id,
                                 group_tp=group_tp,
                                 tp_world_size=tp_world_size,
                                 tp_rank_id=tp_rank_id,
                                 moe_expert_num=moe_expert_num,
                                 quant_mode=quant_mode,
                                 global_bs=global_bs)