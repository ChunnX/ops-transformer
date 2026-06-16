# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Optional

import torch
import torch_npu
from torch.library import impl

from npu_ops_transformer.op_builder.builder import AS_LIBRARY
from npu_ops_transformer.op_builder.builder import OpBuilder


class SparseLightningIndexerKLLossGradOpBuilder(OpBuilder):
    def __init__(self):
        super(SparseLightningIndexerKLLossGradOpBuilder, self).__init__(
            "npu_sparse_lightning_indexer_kl_loss_grad"
        )

    def sources(self):
        return ["ops/csrc/npu_sparse_lightning_indexer_kl_loss_grad.cpp"]

    def schema(self) -> str:
        return (
            "npu_sparse_lightning_indexer_kl_loss_grad("
            "Tensor q, Tensor k, Tensor w, Tensor sparse_indices, Tensor attn_softmax_l1_norm, "
            "*, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, Tensor? seqused_q=None, "
            "Tensor? seqused_k=None, Tensor? cmp_residual_k=None, Tensor? metadata=None, "
            "str layout_q=\"TND\", str layout_k=\"TND\", int mask_mode=3, "
            "int cmp_ratio=1) -> (Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_sparse_lightning_indexer_kl_loss_grad_meta(
            q,
            k,
            w,
            sparse_indices,
            attn_softmax_l1_norm,
            *,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            seqused_q=None,
            seqused_k=None,
            cmp_residual_k=None,
            metadata=None,
            layout_q="TND",
            layout_k="TND",
            mask_mode=3,
            cmp_ratio=1,
        ):
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dw = torch.empty_like(w)
            softmax_out = torch.empty_like(attn_softmax_l1_norm)
            return dq, dk, dw, softmax_out


sparse_lightning_indexer_kl_loss_grad_op_builder = SparseLightningIndexerKLLossGradOpBuilder()
op_module = sparse_lightning_indexer_kl_loss_grad_op_builder.load()


@impl(AS_LIBRARY, sparse_lightning_indexer_kl_loss_grad_op_builder.name, "PrivateUse1")
def npu_sparse_lightning_indexer_kl_loss_grad(
    q,
    k,
    w,
    sparse_indices,
    attn_softmax_l1_norm,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    metadata=None,
    layout_q="TND",
    layout_k="TND",
    mask_mode=3,
    cmp_ratio=1,
):
    return op_module.npu_sparse_lightning_indexer_kl_loss_grad(
        q,
        k,
        w,
        sparse_indices,
        attn_softmax_l1_norm,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        metadata,
        layout_q,
        layout_k,
        mask_mode,
        cmp_ratio,
    )


try:
    import torchair
    from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair.ge import attr

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False


if _TORCHAIR_AVAILABLE:
    @register_fx_node_ge_converter(
        torch.ops.npu_ops_transformer.npu_sparse_lightning_indexer_kl_loss_grad.default
    )
    def convert_npu_sparse_lightning_indexer_kl_loss_grad(
        q: Tensor,
        k: Tensor,
        w: Tensor,
        sparse_indices: Tensor,
        attn_softmax_l1_norm: Tensor,
        *,
        cu_seqlens_q: Optional[Tensor] = None,
        cu_seqlens_k: Optional[Tensor] = None,
        seqused_q: Optional[Tensor] = None,
        seqused_k: Optional[Tensor] = None,
        cmp_residual_k: Optional[Tensor] = None,
        metadata: Optional[Tensor] = None,
        layout_q: str = "TND",
        layout_k: str = "TND",
        mask_mode: int = 3,
        cmp_ratio: int = 1,
        meta_outputs: TensorSpec = None,
    ):
        return torchair.ge.custom_op(
            "SparseLightningIndexerKLLossGrad",
            inputs={
                "q": q,
                "k": k,
                "w": w,
                "sparse_indices": sparse_indices,
                "attn_softmax_l1_norm": attn_softmax_l1_norm,
                "cu_seqlens_q": cu_seqlens_q,
                "cu_seqlens_k": cu_seqlens_k,
                "seqused_q": seqused_q,
                "seqused_k": seqused_k,
                "cmp_residual_k": cmp_residual_k,
                "metadata": metadata,
            },
            attrs={
                "layout_q": attr.Str(layout_q),
                "layout_k": attr.Str(layout_k),
                "mask_mode": attr.Int(mask_mode),
                "cmp_ratio": attr.Int(cmp_ratio),
            },
            outputs=["dq", "dk", "dw", "softmax_out"],
        )
