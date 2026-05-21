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
    @auto_convert_to_tensor([False, False], [False, False])
    def NotifyDispatch(sendData: Tensor, tokenPerExpertData: Tensor, *, send_count: int, num_tokens: int,
                       comm_group: str, rank_size: int, rank_id: int, local_rank_size: int,
                       local_rank_id: int, round: int, per_round_tokens: int,
                       dependencies=[], node_name=None):
        """REG_OP(NotifyDispatch)\n
        .INPUT(sendData, TensorType({DT_INT32}))\n
        .INPUT(tokenPerExpertData, TensorType({DT_INT32}))\n
        .OUTPUT(sendDataOffset, TensorType({DT_INT32}))\n
        .OUTPUT(recvData, TensorType({DT_INT32}))\n
        .OUTPUT(recvCount, TensorType({DT_INT32}))\n
        .OUTPUT(recvOffset, TensorType({DT_INT32}))\n
        .OUTPUT(expertGlobalOffset, TensorType({DT_INT32}))\n
        .OUTPUT(srcrankInExpertOffset, TensorType({DT_INT32}))\n
        .OUTPUT(rInSrcrankOffset, TensorType({DT_INT32}))\n
        .OUTPUT(totalRecvTokens, TensorType({DT_INT32}))\n
        .OUTPUT(maxBs, TensorType({DT_INT32}))\n
        .OUTPUT(recvTokensPerExpert, TensorType({DT_INT32}))\n
        .REQUIRED_ATTR(send_count, Int)\n
        .REQUIRED_ATTR(num_tokens, Int)\n
        .REQUIRED_ATTR(comm_group, String)\n
        .REQUIRED_ATTR(rank_size, Int)\n
        .REQUIRED_ATTR(rank_id, Int)\n
        .REQUIRED_ATTR(local_rank_size, Int)\n
        .REQUIRED_ATTR(local_rank_id, Int)\n
        .REQUIRED_ATTR(round, Int)\n
        .REQUIRED_ATTR(per_round_tokens, Int)\n
        """
        # process inputs
        inputs = {
            "sendData": sendData,
            "tokenPerExpertData": tokenPerExpertData,
        }

        # process attrs
        attrs = {
            "send_count": attr.Int(send_count),
            "num_tokens": attr.Int(num_tokens),
            "comm_group": attr.Str(comm_group),
            "rank_size": attr.Int(rank_size),
            "rank_id": attr.Int(rank_id),
            "local_rank_size": attr.Int(local_rank_size),
            "local_rank_id": attr.Int(local_rank_id),
            "round": attr.Int(round),
            "per_round_tokens": attr.Int(per_round_tokens),
        }

        # process outputs
        outputs = [
            "sendDataOffset",
            "recvData",
            "recvCount",
            "recvOffset",
            "expertGlobalOffset",
            "srcrankInExpertOffset",
            "rInSrcrankOffset",
            "totalRecvTokens",
            "maxBs",
            "recvTokensPerExpert",
        ]

        return ge_op(
            op_type="NotifyDispatch",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("NotifyDispatch") \
            .input("sendData", "DT_INT32") \
            .input("tokenPerExpertData", "DT_INT32") \
            .required_attr("send_count", attr.Int) \
            .required_attr("num_tokens", attr.Int) \
            .required_attr("comm_group", attr.Str) \
            .required_attr("rank_size", attr.Int) \
            .required_attr("rank_id", attr.Int) \
            .required_attr("local_rank_size", attr.Int) \
            .required_attr("local_rank_id", attr.Int) \
            .required_attr("round", attr.Int) \
            .required_attr("per_round_tokens", attr.Int) \
            .output("sendDataOffset", "DT_INT32") \
            .output("recvData", "DT_INT32") \
            .output("recvCount", "DT_INT32") \
            .output("recvOffset", "DT_INT32") \
            .output("expertGlobalOffset", "DT_INT32") \
            .output("srcrankInExpertOffset", "DT_INT32") \
            .output("rInSrcrankOffset", "DT_INT32") \
            .output("totalRecvTokens", "DT_INT32") \
            .output("maxBs", "DT_INT32") \
            .output("recvTokensPerExpert", "DT_INT32")
        )

    @register_fx_node_ge_converter(torch.ops.npu_ops_transformer.npu_notify_dispatch.default)
    def convert_npu_notify_dispatch(
        send_data: Tensor,
        token_per_expert_data: Tensor,
        send_count: int,
        num_tokens: int,
        group_ep: str,
        rank_size: int,
        rank_id: int,
        local_rank_size: int,
        local_rank_id: int,
        round: int,
        per_round_tokens: int,
        *,
        meta_outputs: TensorSpec = None):
        return NotifyDispatch(sendData=send_data,
                              tokenPerExpertData=token_per_expert_data,
                              send_count=send_count,
                              num_tokens=num_tokens,
                              group_ep=group_ep,
                              rank_size=rank_size,
                              rank_id=rank_id,
                              local_rank_size=local_rank_size,
                              local_rank_id=local_rank_id,
                              round=round,
                              per_round_tokens=per_round_tokens)