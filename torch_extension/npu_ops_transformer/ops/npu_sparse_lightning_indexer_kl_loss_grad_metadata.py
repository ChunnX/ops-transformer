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


SLI_KL_LOSS_GRAD_METADATA_SIZE = 64


class SparseLightningIndexerKLLossGradMetadataOpBuilder(OpBuilder):
    def __init__(self):
        super(SparseLightningIndexerKLLossGradMetadataOpBuilder, self).__init__(
            "npu_sparse_lightning_indexer_kl_loss_grad_metadata"
        )

    def sources(self):
        return ["ops/csrc/npu_sparse_lightning_indexer_kl_loss_grad_metadata.cpp"]

    def schema(self) -> str:
        return (
            "npu_sparse_lightning_indexer_kl_loss_grad_metadata("
            "*, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, Tensor? seqused_q=None, "
            "Tensor? seqused_k=None, Tensor? cmp_residual_k=None, int b_size=0, int s1_size=0, "
            "int s2_size=0, int k_size=0, int mask_mode=3, int cmp_ratio=1, "
            "str layout_q=\"TND\", str layout_k=\"TND\") -> Tensor"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_sparse_lightning_indexer_kl_loss_grad_metadata_meta(
            *,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            seqused_q=None,
            seqused_k=None,
            cmp_residual_k=None,
            b_size=0,
            s1_size=0,
            s2_size=0,
            k_size=0,
            mask_mode=3,
            cmp_ratio=1,
            layout_q="TND",
            layout_k="TND",
        ):
            return torch.empty((SLI_KL_LOSS_GRAD_METADATA_SIZE,), dtype=torch.int32, device="meta")


sparse_lightning_indexer_kl_loss_grad_metadata_op_builder = SparseLightningIndexerKLLossGradMetadataOpBuilder()
op_module = sparse_lightning_indexer_kl_loss_grad_metadata_op_builder.load()


@impl(AS_LIBRARY, sparse_lightning_indexer_kl_loss_grad_metadata_op_builder.name, "PrivateUse1")
def npu_sparse_lightning_indexer_kl_loss_grad_metadata(
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    b_size=0,
    s1_size=0,
    s2_size=0,
    k_size=0,
    mask_mode=3,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="TND",
):
    return op_module.npu_sparse_lightning_indexer_kl_loss_grad_metadata(
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        b_size,
        s1_size,
        s2_size,
        k_size,
        mask_mode,
        cmp_ratio,
        layout_q,
        layout_k,
    )


@torch.library.register_kernel(
    "npu_ops_transformer::npu_sparse_lightning_indexer_kl_loss_grad_metadata", None
)
def npu_sparse_lightning_indexer_kl_loss_grad_metadata_fallback(
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    b_size=0,
    s1_size=0,
    s2_size=0,
    k_size=0,
    mask_mode=3,
    cmp_ratio=1,
    layout_q="TND",
    layout_k="TND",
):
    return npu_sparse_lightning_indexer_kl_loss_grad_metadata(
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        b_size=b_size,
        s1_size=s1_size,
        s2_size=s2_size,
        k_size=k_size,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        layout_q=layout_q,
        layout_k=layout_k,
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
        torch.ops.npu_ops_transformer.npu_sparse_lightning_indexer_kl_loss_grad_metadata.default
    )
    def convert_npu_sparse_lightning_indexer_kl_loss_grad_metadata(
        *,
        cu_seqlens_q: Optional[Tensor] = None,
        cu_seqlens_k: Optional[Tensor] = None,
        seqused_q: Optional[Tensor] = None,
        seqused_k: Optional[Tensor] = None,
        cmp_residual_k: Optional[Tensor] = None,
        b_size: int = 0,
        s1_size: int = 0,
        s2_size: int = 0,
        k_size: int = 0,
        mask_mode: int = 3,
        cmp_ratio: int = 1,
        layout_q: str = "TND",
        layout_k: str = "TND",
        meta_outputs: TensorSpec = None,
    ):
        stream_info = torch.npu.get_stream_limit(torch.npu.current_stream())
        aic_core_num = stream_info.get("cube_core_num")
        return torchair.ge.custom_op(
            "SparseLightningIndexerKLLossGradMetadata",
            inputs={
                "cu_seqlens_q": cu_seqlens_q,
                "cu_seqlens_k": cu_seqlens_k,
                "seqused_q": seqused_q,
                "seqused_k": seqused_k,
                "cmp_residual_k": cmp_residual_k,
            },
            attrs={
                "b_size": attr.Int(b_size),
                "s1_size": attr.Int(s1_size),
                "s2_size": attr.Int(s2_size),
                "k_size": attr.Int(k_size),
                "mask_mode": attr.Int(mask_mode),
                "cmp_ratio": attr.Int(cmp_ratio),
                "layout_q": attr.Str(layout_q),
                "layout_k": attr.Str(layout_k),
                "aic_core_num": attr.Int(aic_core_num),
            },
            outputs=["metadata"],
        )
