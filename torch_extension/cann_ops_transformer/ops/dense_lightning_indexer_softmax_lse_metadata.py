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

from cann_ops_transformer.op_builder.builder import AS_LIBRARY
from cann_ops_transformer.op_builder.builder import OpBuilder


DLI_METADATA_SIZE = 64
DLI_METADATA_OP_NAME = "dense_lightning_indexer_softmax_lse_metadata"


class DenseLightningIndexerSoftmaxLseMetadataOpBuilder(OpBuilder):
    def __init__(self):
        super(DenseLightningIndexerSoftmaxLseMetadataOpBuilder, self).__init__(
            "dense_lightning_indexer_softmax_lse_metadata"
        )

    def sources(self):
        return ["ops/csrc/dense_lightning_indexer_softmax_lse_metadata.cpp"]

    def schema(self) -> str:
        return (
            "dense_lightning_indexer_softmax_lse_metadata("
            "int num_heads_q, int num_heads_k, int head_dim, *, "
            "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, Tensor? seqused_q=None, "
            "Tensor? seqused_k=None, Tensor? cmp_residual_k=None, int? batch_size=None, "
            "int? max_seqlen_q=None, int? max_seqlen_k=None, "
            "str? layout_q=None, str? layout_k=None, int? mask_mode=None, "
            "int? cmp_ratio=None) -> Tensor"
        )

    def register_meta(self):
        @torch.library.register_fake("cann_ops_transformer::" + DLI_METADATA_OP_NAME)
        def dense_lightning_indexer_softmax_lse_metadata_meta(
            num_heads_q,
            num_heads_k,
            head_dim,
            *,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            seqused_q=None,
            seqused_k=None,
            cmp_residual_k=None,
            batch_size=None,
            max_seqlen_q=None,
            max_seqlen_k=None,
            layout_q=None,
            layout_k=None,
            mask_mode=None,
            cmp_ratio=None,
        ):
            return torch.empty((DLI_METADATA_SIZE,), dtype=torch.int32, device="meta")


dense_lightning_indexer_softmax_lse_metadata_op_builder = DenseLightningIndexerSoftmaxLseMetadataOpBuilder()
op_module = dense_lightning_indexer_softmax_lse_metadata_op_builder.load()


@impl(AS_LIBRARY, DLI_METADATA_OP_NAME, "PrivateUse1")
def dense_lightning_indexer_softmax_lse_metadata(
    num_heads_q,
    num_heads_k,
    head_dim,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    layout_q=None,
    layout_k=None,
    mask_mode=None,
    cmp_ratio=None,
):
    batch_size = 0 if batch_size is None else batch_size
    max_seqlen_q = 0 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_k = 0 if max_seqlen_k is None else max_seqlen_k
    layout_q = "BSND" if layout_q is None else layout_q
    layout_k = "BSND" if layout_k is None else layout_k
    mask_mode = 0 if mask_mode is None else mask_mode
    cmp_ratio = 1 if cmp_ratio is None else cmp_ratio

    return op_module.dense_lightning_indexer_softmax_lse_metadata(
        num_heads_q,
        num_heads_k,
        head_dim,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        layout_q,
        layout_k,
        mask_mode,
        cmp_ratio,
    )


@torch.library.register_kernel("cann_ops_transformer::" + DLI_METADATA_OP_NAME, None)
def dense_lightning_indexer_softmax_lse_metadata_fallback(
    num_heads_q,
    num_heads_k,
    head_dim,
    *,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    batch_size=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    layout_q=None,
    layout_k=None,
    mask_mode=None,
    cmp_ratio=None,
):
    return dense_lightning_indexer_softmax_lse_metadata(
        num_heads_q,
        num_heads_k,
        head_dim,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        batch_size=batch_size,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        layout_q=layout_q,
        layout_k=layout_k,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
    )


torch.compiler.allow_in_graph(dense_lightning_indexer_softmax_lse_metadata)
