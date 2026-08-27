# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE converter for the decomposed MhcPreSinkhorn backward graph.

try:
    import torch
    import torch_npu
    import torchair
    from typing import Optional
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair._ge_concrete_graph.compat_ir import IrDef, ge_op
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from torchair.ge import attr
    from torchair.ge._ge_graph import DataType, Tensor, TensorSpec

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False


if _TORCHAIR_AVAILABLE:

    def _mhc_sinkhorn_backward(grad_h_res: Tensor, norm_out: Tensor, sum_out: Tensor):
        return ge_op(
            op_type="MhcSinkhornBackward",
            inputs={"grad_y": grad_h_res, "norm": norm_out, "sum": sum_out},
            outputs=["grad_input"],
            ir=IrDef("MhcSinkhornBackward")
            .input("grad_y", "DT_FLOAT")
            .input("norm", "DT_FLOAT")
            .input("sum", "DT_FLOAT")
            .output("grad_input", "DT_FLOAT"),
        )

    def _mhc_pre_backward(
        x: Tensor,
        phi: Tensor,
        alpha: Tensor,
        grad_hin: Tensor,
        grad_h_post: Tensor,
        grad_h_res: Tensor,
        inv_rms: Tensor,
        h_mix: Tensor,
        h_pre: Tensor,
        h_post: Tensor,
        hc_eps: float,
    ):
        return ge_op(
            op_type="MhcPreBackward",
            inputs={
                "x": x,
                "phi": phi,
                "alpha": alpha,
                "grad_h_in": grad_hin,
                "grad_h_post": grad_h_post,
                "grad_h_res": grad_h_res,
                "inv_rms": inv_rms,
                "h_mix": h_mix,
                "h_pre": h_pre,
                "h_post": h_post,
                "gamma": None,
                "grad_x_post": None,
            },
            attrs={"hc_eps": attr.Float(hc_eps)},
            outputs=["grad_x", "grad_phi", "grad_alpha", "grad_bias"],
            ir=IrDef("MhcPreBackward")
            .input("x", "DT_BF16, DT_FLOAT16")
            .input("phi", "DT_FLOAT")
            .input("alpha", "DT_FLOAT")
            .input("grad_h_in", "DT_BF16, DT_FLOAT16")
            .input("grad_h_post", "DT_FLOAT")
            .input("grad_h_res", "DT_FLOAT")
            .input("inv_rms", "DT_FLOAT")
            .input("h_mix", "DT_FLOAT")
            .input("h_pre", "DT_FLOAT")
            .input("h_post", "DT_FLOAT")
            .optional_input("gamma", "DT_FLOAT")
            .optional_input("grad_x_post", "DT_BF16, DT_FLOAT16")
            .attr("hc_eps", attr.Float(1e-6))
            .output("grad_x", "DT_BF16, DT_FLOAT16")
            .output("grad_phi", "DT_FLOAT")
            .output("grad_alpha", "DT_FLOAT")
            .output("grad_bias", "DT_FLOAT"),
        )

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.mhc_pre_sinkhorn_backward.default
    )
    def convert_mhc_pre_sinkhorn_backward(
        grad_hin: Tensor,
        grad_h_post: Tensor,
        grad_h_res: Tensor,
        x: Tensor,
        phi: Tensor,
        alpha: Tensor,
        bias: Tensor,
        h_pre: Tensor,
        h_post: Tensor,
        hc_before_norm: Tensor,
        inv_rms: Tensor,
        sum_out: Tensor,
        norm_out: Tensor,
        hc_eps: float,
        meta_outputs: Optional[TensorSpec] = None,
    ):
        del meta_outputs
        int64_zero = ge.Const(0, dtype=DataType.DT_INT64)
        h_post_shape = ge.Shape(h_post)
        n = ge.GatherV2(
            h_post_shape,
            ge.Const([h_post.rank - 1], dtype=DataType.DT_INT64),
            int64_zero,
        )
        grad_h_res_shape = ge.ConcatV2([h_post_shape, n], 0, N=2)
        grad_h_res_matrix = ge.Reshape(grad_h_res, grad_h_res_shape)

        grad_h_res_before_sinkhorn = _mhc_sinkhorn_backward(
            grad_h_res_matrix, norm_out, sum_out
        )
        inv_rms_squeezed = ge.Squeeze(inv_rms, axis=[-1])
        return _mhc_pre_backward(
            x,
            phi,
            alpha,
            grad_hin,
            grad_h_post,
            grad_h_res_before_sinkhorn,
            inv_rms_squeezed,
            hc_before_norm,
            h_pre,
            h_post,
            hc_eps,
        )
