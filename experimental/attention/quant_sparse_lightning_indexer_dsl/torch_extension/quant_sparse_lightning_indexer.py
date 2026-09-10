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


class QuantSparseLightningIndexerOpBuilder(OpBuilder):
    def __init__(self):
        super(QuantSparseLightningIndexerOpBuilder, self).__init__(
            "ds41.quant_sparse_lightning_indexer", category="attention"
        )

    def sources(self):
        """Python DSL implementation; no C++ extension is compiled."""
        return []

    def schema(self):
        """PyTorch operator signature."""
        return [
            (
                "ds41.quant_sparse_lightning_indexer(Tensor q, Tensor k, Tensor w, "
                "Tensor descale_q, Tensor descale_k, Tensor candidate_block_indices, "
                "Tensor candidate_block_length, int topk, int quant_mode, "
                "int candidate_block_size, *, Tensor? cu_seqlens_q=None, "
                "Tensor? seqused_q=None, Tensor? seqused_k=None, "
                "Tensor? cmp_residual_k=None, Tensor? block_table=None, "
                "Tensor? output_idx_offset=None, Tensor? metadata=None, "
                "int max_seqlen_q=-1, "
                "int mask_mode=0, int cmp_ratio=1, "
                'str layout_q="TND", str layout_kv="PA_BBND", '
                "bool return_value=False) -> (Tensor, Tensor)"
            ),
        ]

    def register_meta(self):
        """Register the FakeTensor/Meta output implementation."""
        @impl(get_as_library(), self.name, "Meta")
        def quant_sparse_lightning_indexer_meta(
            q,
            k,
            w,
            descale_q,
            descale_k,
            candidate_block_indices,
            candidate_block_length,
            topk,
            quant_mode,
            candidate_block_size,
            *,
            cu_seqlens_q=None,
            seqused_q=None,
            seqused_k=None,
            cmp_residual_k=None,
            block_table=None,
            output_idx_offset=None,
            metadata=None,
            max_seqlen_q=-1,
            mask_mode=0,
            cmp_ratio=1,
            layout_q="TND",
            layout_kv="PA_BBND",
            return_value=False,
        ):
            output_shape = (q.shape[0], k.shape[2], topk)
            sparse_indices = torch.empty(
                output_shape, dtype=torch.int32, device="meta"
            )
            sparse_values_shape = output_shape if return_value else (0,)
            sparse_values = torch.empty(
                sparse_values_shape, dtype=torch.bfloat16, device="meta"
            )
            return sparse_indices, sparse_values


quant_sparse_lightning_indexer_op_builder = QuantSparseLightningIndexerOpBuilder()
quant_sparse_lightning_indexer_op_builder._ensure_initialized()


@impl(get_as_library(), quant_sparse_lightning_indexer_op_builder.name, "PrivateUse1")
def quant_sparse_lightning_indexer(
    q,
    k,
    w,
    descale_q,
    descale_k,
    candidate_block_indices,
    candidate_block_length,
    topk,
    quant_mode,
    candidate_block_size,
    *,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    output_idx_offset=None,
    metadata=None,
    max_seqlen_q=-1,
    mask_mode=0,
    cmp_ratio=1,
    layout_q="TND",
    layout_kv="PA_BBND",
    return_value=False,
):
    """Run the CANNBotDSL QSLI kernel through the Torch NPU dispatcher."""
    from ops.quant_sparse_lightning_indexer_dsl import (
        quant_sparse_lightning_indexer as dsl_quant_sparse_lightning_indexer,
    )

    return dsl_quant_sparse_lightning_indexer(
        q,
        k,
        w,
        descale_q,
        descale_k,
        candidate_block_indices,
        candidate_block_length,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        block_table=block_table,
        output_idx_offset=output_idx_offset,
        metadata=metadata,
        topk=topk,
        quant_mode=quant_mode,
        candidate_block_size=candidate_block_size,
        max_seqlen_q=max_seqlen_q,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        layout_q=layout_q,
        layout_kv=layout_kv,
        return_value=return_value,
    )
