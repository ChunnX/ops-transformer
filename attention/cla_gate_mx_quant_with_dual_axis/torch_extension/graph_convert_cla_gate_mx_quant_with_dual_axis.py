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
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False


if _TORCHAIR_AVAILABLE:

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.cla_gate_mx_quant_with_dual_axis.default
    )
    def convert_cla_gate_mx_quant_with_dual_axis(
        global_out: Tensor,
        local_out: Tensor,
        global_gate_logits: Tensor,
        local_gate_logits: Tensor,
        *,
        round_mode: str = "rint",
        scale_alg: int = 1,
        dst_type: int = 36,
        meta_outputs: TensorSpec = None,
    ):
        return ge_op(
            op_type="ClaGateMxQuantWithDualAxis",
            inputs={
                "global_out": global_out,
                "local_out": local_out,
                "global_gate_logits": global_gate_logits,
                "local_gate_logits": local_gate_logits,
            },
            attrs={
                "round_mode": attr.String(round_mode),
                "scale_alg": attr.Int(scale_alg),
                "dst_type": attr.Int(dst_type),
            },
            outputs=[
                "row_data",
                "row_scale",
                "col_data",
                "col_scale",
            ],
            ir=IrDef("ClaGateMxQuantWithDualAxis")
            .input("global_out", "DT_FLOAT16, DT_BF16")
            .input("local_out", "DT_FLOAT16, DT_BF16")
            .input("global_gate_logits", "DT_FLOAT16, DT_BF16")
            .input("local_gate_logits", "DT_FLOAT16, DT_BF16")
            .attr("round_mode", attr.String("rint"))
            .attr("scale_alg", attr.Int(1))
            .attr("dst_type", attr.Int(36))
            .output("row_data", "DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2")
            .output("row_scale", "DT_FLOAT8_E8M0")
            .output("col_data", "DT_FLOAT4_E2M1, DT_FLOAT4_E1M2, DT_FLOAT8_E4M3FN, DT_FLOAT8_E5M2")
            .output("col_scale", "DT_FLOAT8_E8M0"),
        )

else:

    def convert_cla_gate_mx_quant_with_dual_axis(*args, **kwargs):
        raise RuntimeError(
            "ClaGateMxQuantWithDualAxis graph converter: torchair is not available."
        )
