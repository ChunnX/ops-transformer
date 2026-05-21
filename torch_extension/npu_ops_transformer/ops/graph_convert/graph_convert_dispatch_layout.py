# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for Graph Mode

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
    @auto_convert_to_tensor([False], [False])
    def DispatchLayout(topkIdx: Tensor, *, num_tokens: int, num_ranks: int, num_experts: int,
                       num_topk: int, local_ranksize: int, per_round_tokens: int, rank_id: int,
                       dependencies=[], node_name=None):
        """REG_OP(DispatchLayout)\n
        .INPUT(topkIdx, TensorType({DT_INT64}))\n
        .OUTPUT(numTokensPerRank, TensorType({DT_INT32}))\n
        .OUTPUT(numTokensPerExpert, TensorType({DT_INT32}))\n
        .OUTPUT(isTokenInRank, TensorType({DT_INT32}))\n
        .OUTPUT(notifySendData, TensorType({DT_INT32}))\n
        .OUTPUT(sendTokenIdxSmall, TensorType({DT_INT32}))\n
        .REQUIRED_ATTR(num_tokens, Int)\n
        .REQUIRED_ATTR(num_ranks, Int)\n
        .REQUIRED_ATTR(num_experts, Int)\n
        .REQUIRED_ATTR(num_topk, Int)\n
        .REQUIRED_ATTR(local_ranksize, Int)\n
        .REQUIRED_ATTR(per_round_tokens, Int)\n
        .REQUIRED_ATTR(rank_id, Int)\n
        """
        # process inputs
        inputs = {
            "topkIdx": topkIdx,
        }

        # process attrs
        attrs = {
            "num_tokens": attr.Int(num_tokens),
            "num_ranks": attr.Int(num_ranks),
            "num_experts": attr.Int(num_experts),
            "num_topk": attr.Int(num_topk),
            "local_ranksize": attr.Int(local_ranksize),
            "per_round_tokens": attr.Int(per_round_tokens),
            "rank_id": attr.Int(rank_id),
        }

        # process outputs
        outputs = [
            "numTokensPerRank",
            "numTokensPerExpert",
            "isTokenInRank",
            "notifySendData",
            "sendTokenIdxSmall",
        ]

        return ge_op(
            op_type="DispatchLayout",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("DispatchLayout") \
            .input("topkIdx", "DT_INT64") \
            .required_attr("num_tokens", attr.Int) \
            .required_attr("num_ranks", attr.Int) \
            .required_attr("num_experts", attr.Int) \
            .required_attr("num_topk", attr.Int) \
            .required_attr("local_ranksize", attr.Int) \
            .required_attr("per_round_tokens", attr.Int) \
            .required_attr("rank_id", attr.Int) \
            .output("numTokensPerRank", "DT_INT32") \
            .output("numTokensPerExpert", "DT_INT32") \
            .output("isTokenInRank", "DT_INT32") \
            .output("notifySendData", "DT_INT32") \
            .output("sendTokenIdxSmall", "DT_INT32")
        )

    @register_fx_node_ge_converter(torch.ops.npu_ops_transformer.npu_dispatch_layout.default)
    def convert_npu_dispatch_layout(
        topk_idx: Tensor,
        num_tokens: int,
        num_ranks: int,
        num_experts: int,
        num_topk: int,
        local_ranksize: int,
        per_round_tokens: int,
        rank_id: int,
        *,
        meta_outputs: TensorSpec = None):
        return DispatchLayout(topkIdx=topk_idx,
                              num_tokens=num_tokens,
                              num_ranks=num_ranks,
                              num_experts=num_experts,
                              num_topk=num_topk,
                              local_ranksize=local_ranksize,
                              per_round_tokens=per_round_tokens,
                              rank_id=rank_id)