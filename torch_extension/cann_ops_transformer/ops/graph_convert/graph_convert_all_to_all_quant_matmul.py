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
        [False, False, False, True, True, True, True, True, True, True],
        [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
    )
    def AlltoAllMatmul(
        context: Tensor,
        x1: Tensor,
        x2: Tensor,
        bias: Optional[Tensor],
        x1_scale: Optional[Tensor],
        x2_scale: Optional[Tensor],
        comm_scale: Optional[Tensor],
        x1_offset: Optional[Tensor],
        x2_offset: Optional[Tensor],
        scales: Optional[Tensor],
        *,
        hccl_buffer_size: int,
        world_size: int,
        all2all_axes: List[int] = [-2, -1],
        y_dtype: int = -1,
        x1_quant_mode: int = 0,
        x2_quant_mode: int = 0,
        comm_quant_mode: int = 0,
        x1_quant_dtype: int = -1,
        comm_quant_dtype: int = -1,
        transpose_x1: bool = False,
        transpose_x2: bool = False,
        group_size: int = 0,
        comm_mode: str = "ai_cpu",
        alltoall_out_flag: bool = True,
        dependencies=[],
        node_name=None,
    ):
        """REG_OP(AlltoAllMatmul)\n
        .INPUT(context, TensorType({DT_INT32}))\n
        .INPUT(x1, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1}))\n
        .INPUT(x2, TensorType({DT_FLOAT16, DT_BF16, DT_INT8, DT_INT4, DT_FLOAT8_E5M2,\n
            DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1}))\n
        .OPTIONAL_INPUT(bias, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))\n
        .OPTIONAL_INPUT(x1_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))\n
        .OPTIONAL_INPUT(x2_scale, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))\n
        .OPTIONAL_INPUT(comm_scale, TensorType({DT_FLOAT}))\n
        .OPTIONAL_INPUT(x1_offset, TensorType({DT_FLOAT}))\n
        .OPTIONAL_INPUT(x2_offset, TensorType({DT_FLOAT}))\n
        .OPTIONAL_INPUT(scales, TensorType({DT_FLOAT, DT_FLOAT8_E8M0}))\n
        .OUTPUT(y, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT}))\n
        .OUTPUT(all2all_out, TensorType({DT_FLOAT16, DT_BF16, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN,\n
            DT_FLOAT4_E2M1}))\n
        .REQUIRED_ATTR(hccl_buffer_size, Int)\n
        .REQUIRED_ATTR(world_size, Int)\n
        .ATTR(all2all_axes, ListInt, {-2, -1})\n
        .ATTR(y_dtype, Int, -1)\n
        .ATTR(x1_quant_mode, Int, 0)\n
        .ATTR(x2_quant_mode, Int, 0)\n
        .ATTR(comm_quant_mode, Int, 0)\n
        .ATTR(x1_quant_dtype, Int, -1)\n
        .ATTR(comm_quant_dtype, Int, -1)\n
        .ATTR(transpose_x1, Bool, false)\n
        .ATTR(transpose_x2, Bool, false)\n
        .ATTR(group_size, Int, 0)\n
        .ATTR(comm_mode, String, "ai_cpu")\n
        .ATTR(alltoall_out_flag, Bool, true)\n
        """
        inputs = {
            "context": context,
            "x1": x1,
            "x2": x2,
            "bias": bias,
            "x1_scale": x1_scale,
            "x2_scale": x2_scale,
            "comm_scale": comm_scale,
            "x1_offset": x1_offset,
            "x2_offset": x2_offset,
            "scales": scales,
        }

        attrs = {
            "hccl_buffer_size": attr.Int(hccl_buffer_size),
            "world_size": attr.Int(world_size),
            "all2all_axes": attr.ListInt(all2all_axes),
            "y_dtype": attr.Int(y_dtype),
            "x1_quant_mode": attr.Int(x1_quant_mode),
            "x2_quant_mode": attr.Int(x2_quant_mode),
            "comm_quant_mode": attr.Int(comm_quant_mode),
            "x1_quant_dtype": attr.Int(x1_quant_dtype),
            "comm_quant_dtype": attr.Int(comm_quant_dtype),
            "transpose_x1": attr.Bool(transpose_x1),
            "transpose_x2": attr.Bool(transpose_x2),
            "group_size": attr.Int(group_size),
            "comm_mode": attr.Str(comm_mode),
            "alltoall_out_flag": attr.Bool(alltoall_out_flag),
        }

        outputs = [
            "y",
            "all2all_out",
        ]

        return ge_op(
            op_type="AlltoAllMatmul",
            inputs=inputs,
            attrs=attrs,
            outputs=outputs,
            dependencies=dependencies,
            ir=IrDef("AlltoAllMatmul")
            .input("context", "DT_INT32")
            .input(
                "x1",
                "DT_FLOAT16, DT_BF16, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1",
            )
            .input(
                "x2",
                "DT_FLOAT16, DT_BF16, DT_INT8, DT_INT4, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, "
                "DT_FLOAT4_E2M1",
            )
            .optional_input("bias", "DT_FLOAT16, DT_BF16, DT_FLOAT")
            .optional_input("x1_scale", "DT_FLOAT, DT_FLOAT8_E8M0")
            .optional_input("x2_scale", "DT_FLOAT, DT_FLOAT8_E8M0")
            .optional_input("comm_scale", "DT_FLOAT")
            .optional_input("x1_offset", "DT_FLOAT")
            .optional_input("x2_offset", "DT_FLOAT")
            .optional_input("scales", "DT_FLOAT, DT_FLOAT8_E8M0")
            .required_attr("hccl_buffer_size", attr.Int)
            .required_attr("world_size", attr.Int)
            .attr("all2all_axes", attr.ListInt([-2, -1]))
            .attr("y_dtype", attr.Int(-1))
            .attr("x1_quant_mode", attr.Int(0))
            .attr("x2_quant_mode", attr.Int(0))
            .attr("comm_quant_mode", attr.Int(0))
            .attr("x1_quant_dtype", attr.Int(-1))
            .attr("comm_quant_dtype", attr.Int(-1))
            .attr("transpose_x1", attr.Bool(False))
            .attr("transpose_x2", attr.Bool(False))
            .attr("group_size", attr.Int(0))
            .attr("comm_mode", attr.Str("ai_cpu"))
            .attr("alltoall_out_flag", attr.Bool(True))
            .output("y", "DT_FLOAT16, DT_BF16, DT_FLOAT")
            .output(
                "all2all_out",
                "DT_FLOAT16, DT_BF16, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN, DT_FLOAT4_E2M1",
            ),
        )

    @register_fx_node_ge_converter(torch.ops.cann_ops_transformer.npu_all_to_all_quant_matmul.default)
    def convert_npu_all_to_all_quant_matmul(
        context: Tensor,
        x1: Tensor,
        x2: Tensor,
        hccl_buffer_size: int,
        world_size: int,
        *,
        bias: Optional[Tensor] = None,
        x1_scale: Optional[Tensor] = None,
        x2_scale: Optional[Tensor] = None,
        comm_scale: Optional[Tensor] = None,
        x1_offset: Optional[Tensor] = None,
        x2_offset: Optional[Tensor] = None,
        all2all_axes: Optional[List[int]] = None,
        y_dtype: int = -1,
        x1_quant_mode: int = 0,
        x2_quant_mode: int = 0,
        comm_quant_mode: int = 0,
        x1_quant_dtype: int = -1,
        comm_quant_dtype: int = -1,
        transpose_x1: bool = False,
        transpose_x2: bool = False,
        group_size: int = 0,
        comm_mode: str = "ai_cpu",
        alltoall_out_flag: bool = True,
        meta_outputs: TensorSpec = None,
    ):
        return AlltoAllMatmul(
            context=context,
            x1=x1,
            x2=x2,
            bias=bias,
            x1_scale=x1_scale,
            x2_scale=x2_scale,
            comm_scale=comm_scale,
            x1_offset=x1_offset,
            x2_offset=x2_offset,
            scales=None,
            hccl_buffer_size=hccl_buffer_size,
            world_size=world_size,
            all2all_axes=all2all_axes if all2all_axes is not None else [-2, -1],
            y_dtype=y_dtype,
            x1_quant_mode=x1_quant_mode,
            x2_quant_mode=x2_quant_mode,
            comm_quant_mode=comm_quant_mode,
            x1_quant_dtype=x1_quant_dtype,
            comm_quant_dtype=comm_quant_dtype,
            transpose_x1=transpose_x1,
            transpose_x2=transpose_x2,
            group_size=group_size,
            comm_mode=comm_mode,
            alltoall_out_flag=alltoall_out_flag,
        )
