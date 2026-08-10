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
    from typing import Tuple
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair.ge._ge_graph import auto_convert_to_tensor
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from torchair.ge import attr

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False


if _TORCHAIR_AVAILABLE:

    @auto_convert_to_tensor(
        [False, False],
        [False, False],
    )
    def MoeInitRouting(
        x: Tensor,
        expert_idx: Tensor,
        *,
        active_num: int = 0,
        expert_capacity: int = 0,
        expert_num: int = 0,
        drop_pad_mode: int = 0,
        expert_tokens_count_or_cumsum_flag: int = 0,
        expert_tokens_before_capacity_flag: bool = False,
        dependencies=None,
        node_name=None,
    ):
        """REG_OP(MoeInitRoutingV2)\n
        .INPUT(x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
        .INPUT(expert_idx, TensorType({DT_INT32, DT_INT64}))\n
        .OUTPUT(expanded_x, TensorType({DT_FLOAT, DT_FLOAT16, DT_BF16}))\n
        .OUTPUT(expanded_row_idx, TensorType({DT_INT32}))\n
        .OUTPUT(expert_tokens_count_or_cumsum, TensorType({DT_INT32}))\n
        .OUTPUT(expert_tokens_before_capacity, TensorType({DT_INT32}))\n
        .ATTR(active_num, Int, 0)\n
        .ATTR(expert_capacity, Int, 0)\n
        .ATTR(expert_num, Int, 0)\n
        .ATTR(drop_pad_mode, Int, 0)\n
        .ATTR(expert_tokens_count_or_cumsum_flag, Int, 0)\n
        .ATTR(expert_tokens_before_capacity_flag, Bool, false)\n
        """
        if dependencies is None:
            dependencies = []

        inputs = {
            "x": x,
            "expert_idx": expert_idx,
        }

        attrs = {
            "active_num": attr.Int(active_num),
            "expert_capacity": attr.Int(expert_capacity),
            "expert_num": attr.Int(expert_num),
            "drop_pad_mode": attr.Int(drop_pad_mode),
            "expert_tokens_count_or_cumsum_flag": attr.Int(
                expert_tokens_count_or_cumsum_flag
            ),
            "expert_tokens_before_capacity_flag": attr.Bool(
                expert_tokens_before_capacity_flag
            ),
        }

        outputs = [
            "expanded_x",
            "expanded_row_idx",
            "expert_tokens_count_or_cumsum",
            "expert_tokens_before_capacity",
        ]

        return ge_op(
            op_type="MoeInitRoutingV2",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            node_name=node_name,
            ir=IrDef("MoeInitRoutingV2")
            .input("x", "DT_FLOAT, DT_FLOAT16, DT_BF16")
            .input("expert_idx", "DT_INT32, DT_INT64")
            .output("expanded_x", "DT_FLOAT, DT_FLOAT16, DT_BF16")
            .output("expanded_row_idx", "DT_INT32")
            .output("expert_tokens_count_or_cumsum", "DT_INT32")
            .output("expert_tokens_before_capacity", "DT_INT32")
            .attr("active_num", attr.Int(0))
            .attr("expert_capacity", attr.Int(0))
            .attr("expert_num", attr.Int(0))
            .attr("drop_pad_mode", attr.Int(0))
            .attr("expert_tokens_count_or_cumsum_flag", attr.Int(0))
            .attr("expert_tokens_before_capacity_flag", attr.Bool(False)),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.moe_init_routing.default
    )
    def convert_moe_init_routing(
        x: Tensor,
        expert_idx: Tensor,
        active_num: int = 0,
        expert_capacity: int = 0,
        expert_num: int = 0,
        drop_pad_mode: int = 0,
        expert_tokens_count_or_cumsum_flag: int = 0,
        expert_tokens_before_capacity_flag: bool = False,
        meta_outputs: TensorSpec = None,
    ):
        return MoeInitRouting(
            x=x,
            expert_idx=expert_idx,
            active_num=active_num,
            expert_capacity=expert_capacity,
            expert_num=expert_num,
            drop_pad_mode=drop_pad_mode,
            expert_tokens_count_or_cumsum_flag=expert_tokens_count_or_cumsum_flag,
            expert_tokens_before_capacity_flag=expert_tokens_before_capacity_flag,
        )
