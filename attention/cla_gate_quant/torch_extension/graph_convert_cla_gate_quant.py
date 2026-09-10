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
    from torchair.ge import attr
    from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False


if _TORCHAIR_AVAILABLE:

    def _pack_fp4_to_uint8(tensor: Tensor) -> Tensor:
        """Pack a logical FP4 GE tensor [..., K] into uint8 [..., K/2]."""
        bit_shape = [1] * (tensor.rank - 1) + [2]
        div_x2 = ge.Cast(ge.Const(bit_shape), dst_type=DataType.DT_INT32)
        logical_shape = ge.Shape(tensor)
        packed_shape = ge.Div(logical_shape, div_x2)
        pair_shape = ge.ConcatV2(
            [packed_shape, ge.Cast(ge.Const([2]), dst_type=DataType.DT_INT32)],
            concat_dim=0,
            N=2,
        )
        packed = ge.Bitcast(ge.Reshape(tensor, pair_shape), type=DataType.DT_UINT8)
        return ge.Reshape(packed, packed_shape)

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.cla_gate_quant.default
    )
    def convert_cla_gate_quant(
        global_attn: Tensor,
        local_attn: Tensor,
        global_gate_logits: Tensor,
        local_gate_logits: Tensor,
        *,
        round_mode: str = "rint",
        scale_alg: int = 1,
        dst_type: int = 36,
        input_attn_layout: str = "TND",
        dual_axis_flag: bool = True,
        meta_outputs: TensorSpec = None,
    ):
        row_data, row_scale, col_data, col_scale = ge_op(
            op_type="ClaGateQuant",
            inputs={
                "global_attn": global_attn,
                "local_attn": local_attn,
                "global_gate_logits": global_gate_logits,
                "local_gate_logits": local_gate_logits,
            },
            attrs={
                "round_mode": attr.String(round_mode),
                "scale_alg": attr.Int(scale_alg),
                "dst_type": attr.Int(dst_type),
                "inputAttnLayout": attr.String(input_attn_layout),
                "dual_axis_flag": attr.Bool(dual_axis_flag),
            },
            outputs=[
                "row_data",
                "row_scale",
                "col_data",
                "col_scale",
            ],
            ir=IrDef("ClaGateQuant")
            .input("global_attn", "DT_FLOAT16, DT_BF16")
            .input("local_attn", "DT_FLOAT16, DT_BF16")
            .input("global_gate_logits", "DT_FLOAT16, DT_BF16")
            .input("local_gate_logits", "DT_FLOAT16, DT_BF16")
            .attr("round_mode", attr.String("rint"))
            .attr("scale_alg", attr.Int(1))
            .attr("dst_type", attr.Int(36))
            .attr("inputAttnLayout", attr.String("TND"))
            .attr("dual_axis_flag", attr.Bool(True))
            .output("row_data", "DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2")
            .output("row_scale", "DT_FLOAT8_E8M0")
            .output("col_data", "DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2")
            .output("col_scale", "DT_FLOAT8_E8M0"),
        )

        if dst_type in (40, 41):
            row_data = _pack_fp4_to_uint8(row_data)
            col_data = _pack_fp4_to_uint8(col_data)

        return row_data, row_scale, col_data, col_scale

else:

    def convert_cla_gate_quant(*args, **kwargs):
        raise RuntimeError(
            "ClaGateQuant graph converter: torchair is not available."
        )
