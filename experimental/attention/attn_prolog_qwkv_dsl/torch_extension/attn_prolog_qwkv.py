# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import torch
import torch_npu
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library

ATTN_PROLOG_QWKV_OP_NAME = "ds41.attn_prolog_qwkv"


class AttnPrologQwkvOpBuilder(OpBuilder):
    def __init__(self):
        super(AttnPrologQwkvOpBuilder, self).__init__(
            ATTN_PROLOG_QWKV_OP_NAME, category="attention"
        )

    def sources(self):
        """Python DSL implementation; no C++ extension is compiled."""
        return []

    def schema(self):
        """PyTorch operator signature."""
        return [
            (
                "ds41.attn_prolog_qwkv(Tensor x, Tensor qr, Tensor wqb, Tensor ww, "
                "Tensor descale_qr, Tensor descale_wqb, Tensor rope_sin, Tensor rope_cos, "
                "*, float softmax_scale, Tensor? q=None, Tensor? descale_q=None, "
                "Tensor? w=None) -> (Tensor, Tensor, Tensor)"
            ),
        ]

    def register_meta(self):
        """Register the FakeTensor/Meta output implementation."""
        @impl(get_as_library(), self.name, "Meta")
        def attn_prolog_qwkv_meta(
            x,
            qr,
            wqb,
            ww,
            descale_qr,
            descale_wqb,
            rope_sin,
            rope_cos,
            *,
            softmax_scale,
            q=None,
            descale_q=None,
            w=None,
        ):
            t = x.shape[0]
            n_heads = ww.shape[0]
            d = wqb.shape[0] // n_heads
            groups = (d + 63) // 64
            q_out = torch.empty((t, n_heads, d // 2), dtype=torch.uint8, device="meta")
            descale_out = torch.empty(
                (t, n_heads, groups, 2), dtype=torch.uint8, device="meta"
            )
            w_out = torch.empty((t, n_heads), dtype=torch.float32, device="meta")
            return q_out, descale_out, w_out


attn_prolog_qwkv_op_builder = AttnPrologQwkvOpBuilder()
attn_prolog_qwkv_op_builder._ensure_initialized()


def _run_dsl(
    x,
    qr,
    wqb,
    ww,
    descale_qr,
    descale_wqb,
    rope_sin,
    rope_cos,
    *,
    softmax_scale,
    q=None,
    descale_q=None,
    w=None,
):
    from .op import attn_prolog_qwkv as dsl_attn_prolog_qwkv

    return dsl_attn_prolog_qwkv(
        x,
        qr,
        wqb,
        ww,
        descale_qr,
        descale_wqb,
        rope_sin,
        rope_cos,
        softmax_scale=softmax_scale,
        q=q,
        descale_q=descale_q,
        w=w,
    )


@impl(get_as_library(), attn_prolog_qwkv_op_builder.name, "PrivateUse1")
def attn_prolog_qwkv(
    x,
    qr,
    wqb,
    ww,
    descale_qr,
    descale_wqb,
    rope_sin,
    rope_cos,
    *,
    softmax_scale,
    q=None,
    descale_q=None,
    w=None,
):
    """Dispatch to the in-tree CANNBotDSL host entry."""
    return _run_dsl(
        x,
        qr,
        wqb,
        ww,
        descale_qr,
        descale_wqb,
        rope_sin,
        rope_cos,
        softmax_scale=softmax_scale,
        q=q,
        descale_q=descale_q,
        w=w,
    )


@torch.library.register_kernel(
    "cann_ops_transformer::" + ATTN_PROLOG_QWKV_OP_NAME, None
)
def attn_prolog_qwkv_fallback(
    x,
    qr,
    wqb,
    ww,
    descale_qr,
    descale_wqb,
    rope_sin,
    rope_cos,
    *,
    softmax_scale,
    q=None,
    descale_q=None,
    w=None,
):
    return attn_prolog_qwkv(
        x,
        qr,
        wqb,
        ww,
        descale_qr,
        descale_wqb,
        rope_sin,
        rope_cos,
        softmax_scale=softmax_scale,
        q=q,
        descale_q=descale_q,
        w=w,
    )


torch.compiler.allow_in_graph(attn_prolog_qwkv)
