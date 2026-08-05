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
    from torchair._ge_concrete_graph.fx2ge_converter import (
        declare_supported,
        register_fx_node_ge_converter,
    )
    from torchair._ge_concrete_graph.supported_declaration import Support
    from typing import Any, Dict, List, Tuple, Union, Callable, Optional
    from torchair._ge_concrete_graph.ge_ir_pb2 import (
        GraphDef,
        OpDef,
        TensorDescriptor,
        TensorDef,
    )
    from torchair.ge._ge_graph import get_default_ge_graph, next_unique_name
    from torchair.ge._ge_graph import auto_convert_to_tensor
    from torchair.ge._ge_graph import DataType, TensorType
    from torchair.ge._ge_graph import compat_as_bytes, compat_as_bytes_list
    from torchair.ge._ge_graph import trans_to_list_list_int, trans_to_list_list_float
    from torchair.ge._ge_graph import get_invalid_desc
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair.ge import attr

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    @auto_convert_to_tensor(
        [False, False, False, False, False, False, True],
        [False, False, False, False, False, False, False],
    )
    def FFNToAttentionV2(
        x: Tensor,
        session_ids: Tensor,
        micro_batch_ids: Tensor,
        token_ids: Tensor,
        expert_offsets: Tensor,
        actual_token_num: Tensor,
        attn_rank_table: Optional[Tensor],
        *,
        group: str,
        world_size: int,
        token_info_table_shape: List[int],
        token_data_shape: List[int],
        dependencies=[],
        node_name=None,
    ):
        """REG_OP(FFNToAttentionV2)\n
        .INPUT(x, TensorType({DT_BF16, DT_FLOAT16}))\n
        .INPUT(session_ids, TensorType({DT_INT32}))\n
        .INPUT(micro_batch_ids, TensorType({DT_INT32}))\n
        .INPUT(token_ids, TensorType({DT_INT32}))\n
        .INPUT(expert_offsets, TensorType({DT_INT32}))\n
        .INPUT(actual_token_num, TensorType({DT_INT64}))\n
        .OPTIONAL_INPUT(attn_rank_table, TensorType({DT_INT32}))\n
        .REQUIRED_ATTR(group, String)\n
        .REQUIRED_ATTR(world_size, Int)\n
        .REQUIRED_ATTR(token_info_table_shape, ListInt)\n
        .REQUIRED_ATTR(token_data_shape, ListInt)\n
        """
        inputs = {
            "x": x,
            "session_ids": session_ids,
            "micro_batch_ids": micro_batch_ids,
            "token_ids": token_ids,
            "expert_offsets": expert_offsets,
            "actual_token_num": actual_token_num,
            "attn_rank_table": attn_rank_table,
        }

        attrs = {
            "group": attr.Str(group),
            "world_size": attr.Int(world_size),
            "token_info_table_shape": attr.ListInt(token_info_table_shape),
            "token_data_shape": attr.ListInt(token_data_shape),
        }

        outputs = []

        return ge_op(
            op_type="FFNToAttentionV2",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("FFNToAttentionV2")
            .input("x", "DT_BF16, DT_FLOAT16")
            .input("session_ids", "DT_INT32")
            .input("micro_batch_ids", "DT_INT32")
            .input("token_ids", "DT_INT32")
            .input("expert_offsets", "DT_INT32")
            .input("actual_token_num", "DT_INT64")
            .optional_input("attn_rank_table", "DT_INT32")
            .required_attr("group", attr.Str)
            .required_attr("world_size", attr.Int)
            .required_attr("token_info_table_shape", attr.ListInt)
            .required_attr("token_data_shape", attr.ListInt),
        )

    @register_fx_node_ge_converter(torch.ops.cann_ops_transformer.npu_ffn_to_attention_v2.default)
    def convert_npu_ffn_to_attention_v2(
        x: Tensor,
        session_ids: Tensor,
        micro_batch_ids: Tensor,
        token_ids: Tensor,
        expert_offsets: Tensor,
        actual_token_num: Tensor,
        group: str,
        world_size: int,
        token_info_table_shape: List[int],
        token_data_shape: List[int],
        *,
        attn_rank_table: Optional[Tensor] = None,
        meta_outputs: TensorSpec = None,
    ):
        return FFNToAttentionV2(
            x=x,
            session_ids=session_ids,
            micro_batch_ids=micro_batch_ids,
            token_ids=token_ids,
            expert_offsets=expert_offsets,
            actual_token_num=actual_token_num,
            attn_rank_table=attn_rank_table,
            group=group,
            world_size=world_size,
            token_info_table_shape=token_info_table_shape,
            token_data_shape=token_data_shape,
        )
