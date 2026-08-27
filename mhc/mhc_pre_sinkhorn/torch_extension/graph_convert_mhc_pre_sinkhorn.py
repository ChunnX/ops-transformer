# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE converter for the decomposed MhcPreSinkhorn forward graph.

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

    def _mhc_pre_sinkhorn_fused(
        x: Tensor,
        phi: Tensor,
        alpha: Tensor,
        bias: Tensor,
        hc_mult: int,
        num_iters: int,
        hc_eps: float,
        norm_eps: float,
    ):
        return ge_op(
            op_type="MhcPreSinkhorn",
            inputs={"x": x, "phi": phi, "alpha": alpha, "bias": bias},
            attrs={
                "hc_mult": attr.Int(hc_mult),
                "num_iters": attr.Int(num_iters),
                "hc_eps": attr.Float(hc_eps),
                "norm_eps": attr.Float(norm_eps),
                "need_backward": attr.Bool(False),
            },
            outputs=[
                "hin",
                "hPost",
                "hRes",
                "hPre",
                "hcBeforeNorm",
                "invRms",
                "sumOut",
                "normOut",
            ],
            ir=IrDef("MhcPreSinkhorn")
            .input("x", "DT_BF16, DT_FLOAT16")
            .input("phi", "DT_FLOAT")
            .input("alpha", "DT_FLOAT")
            .input("bias", "DT_FLOAT")
            .attr("hc_mult", attr.Int(4))
            .attr("num_iters", attr.Int(20))
            .attr("hc_eps", attr.Float(1e-6))
            .attr("norm_eps", attr.Float(1e-6))
            .attr("need_backward", attr.Bool(False))
            .output("hin", "DT_BF16, DT_FLOAT16")
            .output("hPost", "DT_FLOAT")
            .output("hRes", "DT_FLOAT")
            .output("hPre", "DT_FLOAT")
            .output("hcBeforeNorm", "DT_FLOAT")
            .output("invRms", "DT_FLOAT")
            .output("sumOut", "DT_FLOAT")
            .output("normOut", "DT_FLOAT"),
        )

    def _mhc_pre(
        x: Tensor,
        phi: Tensor,
        alpha: Tensor,
        bias: Tensor,
        norm_eps: float,
        hc_eps: float,
        out_flag: bool,
    ):
        outputs = ["hin", "h_post", "h_res"]
        ir = (
            IrDef("MhcPre")
            .input("x", "DT_BF16, DT_FLOAT16")
            .input("phi", "DT_FLOAT")
            .input("alpha", "DT_FLOAT")
            .input("bias", "DT_FLOAT")
            .optional_input("gamma", "DT_FLOAT")
            .attr("out_flag", attr.Int(0))
            .attr("norm_eps", attr.Float(1e-6))
            .attr("hc_eps", attr.Float(1e-6))
            .attr("op_impl_mode", attr.Int(0))
            .output("hin", "DT_BF16, DT_FLOAT16")
            .output("h_post", "DT_FLOAT")
            .output("h_res", "DT_FLOAT")
        )
        if out_flag:
            outputs.extend(["inv_rms", "h_mix", "h_pre"])
            ir = (
                ir.output("inv_rms", "DT_FLOAT")
                .output("h_mix", "DT_FLOAT")
                .output("h_pre", "DT_FLOAT")
            )
        return ge_op(
            op_type="MhcPre",
            inputs={
                "x": x,
                "phi": phi,
                "alpha": alpha,
                "bias": bias,
                "gamma": None,
            },
            attrs={
                "out_flag": attr.Int(int(out_flag)),
                "norm_eps": attr.Float(norm_eps),
                "hc_eps": attr.Float(hc_eps),
                "op_impl_mode": attr.Int(0),
            },
            outputs=outputs,
            ir=ir,
        )

    def _mhc_sinkhorn(
        h_res: Tensor,
        hc_eps: float,
        num_iters: int,
        out_flag: bool,
    ):
        return ge_op(
            op_type="MhcSinkhorn",
            inputs={"h_res": h_res},
            attrs={
                "eps": attr.Float(hc_eps),
                "num_iters": attr.Int(num_iters),
                "out_flag": attr.Int(int(out_flag)),
            },
            outputs=["y", "norm_out", "sum_out"],
            ir=IrDef("MhcSinkhorn")
            .input("h_res", "DT_FLOAT")
            .attr("eps", attr.Float(1e-6))
            .attr("num_iters", attr.Int(20))
            .attr("out_flag", attr.Int(0))
            .output("y", "DT_FLOAT")
            .output("norm_out", "DT_FLOAT")
            .output("sum_out", "DT_FLOAT"),
        )

    def _flatten_residual(h_res: Tensor, rank: int):
        shape = ge.Shape(h_res)
        axis = ge.Const(0, dtype=DataType.DT_INT64)
        prefix_indices = ge.Const(list(range(rank - 2)), dtype=DataType.DT_INT64)
        prefix = ge.GatherV2(shape, prefix_indices, axis)
        n = ge.GatherV2(
            shape,
            ge.Const([rank - 1], dtype=DataType.DT_INT64),
            axis,
        )
        flattened_shape = ge.ConcatV2([prefix, ge.Mul(n, n)], 0, N=2)
        return ge.Reshape(h_res, flattened_shape)

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.mhc_pre_sinkhorn.default
    )
    def convert_mhc_pre_sinkhorn(
        x: Tensor,
        phi: Tensor,
        alpha: Tensor,
        bias: Tensor,
        hc_mult: int,
        num_iters: int,
        hc_eps: float,
        norm_eps: float,
        out_flag: bool,
        meta_outputs: Optional[TensorSpec] = None,
    ):
        del meta_outputs
        if not out_flag:
            return _mhc_pre_sinkhorn_fused(
                x, phi, alpha, bias, hc_mult, num_iters, hc_eps, norm_eps
            )

        pre_outputs = _mhc_pre(x, phi, alpha, bias, norm_eps, hc_eps, True)
        hin, h_post, h_res_before_sinkhorn = pre_outputs[:3]
        h_res, norm_out, sum_out = _mhc_sinkhorn(
            h_res_before_sinkhorn,
            hc_eps,
            num_iters,
            out_flag,
        )
        h_res = _flatten_residual(h_res, x.rank)

        inv_rms, h_mix, h_pre = pre_outputs[3:]
        inv_rms = ge.Unsqueeze(inv_rms, axes=[-1])

        return hin, h_post, h_res, h_pre, h_mix, inv_rms, sum_out, norm_out
