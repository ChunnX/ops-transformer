# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
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

    @auto_convert_to_tensor(
        [
            False,
            False,
            False,
            False,
        ],
        [
            False,
            False,
            False,
            False,
        ],
    )
    def MhcPreSplitPostRes(
        x: Tensor,
        phi: Tensor,
        alpha: Tensor,
        bias: Tensor,
        *,
        norm_eps: float = 1e-6,
        dependencies=None,
        node_name=None,
    ):
        """REG_OP(MhcPreSplitPostRes)\n
        .INPUT(x, TensorType({DT_BF16, DT_FLOAT16}))\n
        .INPUT(phi, TensorType({DT_FLOAT}))\n
        .INPUT(alpha, TensorType({DT_FLOAT}))\n
        .INPUT(bias, TensorType({DT_FLOAT}))\n
        .OUTPUT(h_post, TensorType({DT_FLOAT}))\n
        .OUTPUT(h_res, TensorType({DT_FLOAT}))\n
        .ATTR(norm_eps, Float, 1e-6)\n
        """
        if dependencies is None:
            dependencies = []

        inputs = {
            "x": x,
            "phi": phi,
            "alpha": alpha,
            "bias": bias,
        }

        attrs = {
            "norm_eps": attr.Float(norm_eps),
        }

        outputs = [
            "h_post",
            "h_res",
        ]

        return ge_op(
            op_type="MhcPreSplitPostRes",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("MhcPreSplitPostRes")
            .input("x", "DT_BF16, DT_FLOAT16")
            .input("phi", "DT_FLOAT")
            .input("alpha", "DT_FLOAT")
            .input("bias", "DT_FLOAT")
            .attr("norm_eps", attr.Float(1e-6))
            .output("h_post", "DT_FLOAT")
            .output("h_res", "DT_FLOAT"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.mhc_pre_split_post_res.default
    )
    def convert_mhc_pre_split_post_res(
        x: Tensor,
        phi: Tensor,
        alpha: Tensor,
        bias: Tensor,
        norm_eps: float = 1e-6,  # 带默认值与 schema 对齐：dynamo 会折叠等于 schema 默认值的常量标量，
        meta_outputs: TensorSpec = None,  # fx 图 args 将不含 norm_eps，签名缺默认值时 GE 转换报缺参
    ):
        return MhcPreSplitPostRes(
            x=x,
            phi=phi,
            alpha=alpha,
            bias=bias,
            norm_eps=norm_eps,
        )
