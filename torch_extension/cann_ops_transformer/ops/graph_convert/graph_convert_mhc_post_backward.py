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
    from torchair.ge._ge_graph import Tensor, TensorSpec, auto_convert_to_tensor
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False


if _TORCHAIR_AVAILABLE:

    @auto_convert_to_tensor(
        [False, False, False, False, False], [False, False, False, False, False]
    )
    def mhc_post_backward_ge(
        grad_output: Tensor,
        x: Tensor,
        h_res: Tensor,
        h_out: Tensor,
        h_post: Tensor,
        *,
        dependencies=None,
        node_name=None,
    ):
        if dependencies is None:
            dependencies = []

        return ge_op(
            op_type="MhcPostBackward",
            inputs={
                "grad_y": grad_output,
                "x": x,
                "h_res": h_res,
                "h_out": h_out,
                "h_post": h_post,
            },
            attrs={},
            outputs=["grad_x", "grad_h_res", "grad_h_out", "grad_h_post"],
            dependencies=dependencies,
            node_name=node_name,
            ir=IrDef("MhcPostBackward")
            .input("grad_y", "DT_BF16, DT_FLOAT16")
            .input("x", "DT_BF16, DT_FLOAT16")
            .input("h_res", "DT_FLOAT")
            .input("h_out", "DT_BF16, DT_FLOAT16")
            .input("h_post", "DT_FLOAT")
            .output("grad_x", "DT_BF16, DT_FLOAT16")
            .output("grad_h_res", "DT_FLOAT")
            .output("grad_h_out", "DT_BF16, DT_FLOAT16")
            .output("grad_h_post", "DT_FLOAT"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.mhc_post_backward.default
    )
    def convert_mhc_post_backward(
        grad_output: Tensor,
        x: Tensor,
        h_res: Tensor,
        h_out: Tensor,
        h_post: Tensor,
        *,
        meta_outputs: TensorSpec = None,
    ):
        return mhc_post_backward_ge(
            grad_output=grad_output,
            x=x,
            h_res=h_res,
            h_out=h_out,
            h_post=h_post,
        )

else:

    def convert_mhc_post_backward(*args, **kwargs):
        raise RuntimeError(
            "GE converter requires torchair, but torchair is not available."
        )
