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
from .quant_lightning_indexer_dsl import (
    quant_lightning_indexer as _dsl_quant_lightning_indexer,
)


class QuantLightningIndexerOpBuilder(OpBuilder):
    def __init__(self):
        super(QuantLightningIndexerOpBuilder, self).__init__(
            "quant_lightning_indexer", category="attention"
        )

    def sources(self):
        """Allocation-only Python placeholder; DSL integration is pending."""
        return []

    def schema(self):
        """PyTorch operator signature."""
        return [
            (
                "quant_lightning_indexer(Tensor q, Tensor k, Tensor w, "
                "Tensor q_descale, Tensor k_descale, int topk, int quant_mode, *, "
                "Tensor? cu_seqlens_q=None, Tensor? seqused_q=None, "
                "Tensor? seqused_k=None, Tensor? cmp_residual_k=None, "
                "Tensor? block_table=None, Tensor? metadata=None, "
                "int max_seqlen_q=-1, "
                "int mask_mode=0, int cmp_ratio=1, "
                'str layout_q="TND", str layout_kv="PA_BBND", '
                "bool return_value=False, int candidate_topk_blocks=-1, "
                "int candidate_block_size=-1) -> (Tensor, Tensor, Tensor, Tensor)"
            ),
        ]

    def register_meta(self):
        """Register the allocation-only Meta implementation."""
        @impl(get_as_library(), self.name, "Meta")
        def quant_lightning_indexer_meta(
            q,
            k,
            w,
            q_descale,
            k_descale,
            topk,
            quant_mode,
            *,
            cu_seqlens_q=None,
            seqused_q=None,
            seqused_k=None,
            cmp_residual_k=None,
            block_table=None,
            metadata=None,
            max_seqlen_q=-1,
            mask_mode=0,
            cmp_ratio=1,
            layout_q="TND",
            layout_kv="PA_BBND",
            return_value=False,
            candidate_topk_blocks=-1,
            candidate_block_size=-1,
        ):
            output_shape = (q.shape[0], k.shape[2], topk)
            sparse_indices = torch.empty(
                output_shape, dtype=torch.int32, device="meta"
            )
            sparse_values_shape = output_shape if return_value else (0,)
            sparse_values = torch.empty(
                sparse_values_shape, dtype=torch.bfloat16, device="meta"
            )
            if candidate_topk_blocks > 0:
                candidate_indices_shape = (
                    q.shape[0],
                    k.shape[2],
                    candidate_topk_blocks,
                )
                candidate_length_shape = (q.shape[0], k.shape[2])
            else:
                candidate_indices_shape = (0,)
                candidate_length_shape = (0,)
            candidate_block_indices = torch.empty(
                candidate_indices_shape, dtype=torch.int32, device="meta"
            )
            candidate_block_length = torch.empty(
                candidate_length_shape, dtype=torch.int32, device="meta"
            )
            return (
                sparse_indices,
                sparse_values,
                candidate_block_indices,
                candidate_block_length,
            )


quant_lightning_indexer_op_builder = QuantLightningIndexerOpBuilder()
quant_lightning_indexer_op_builder._ensure_initialized()


@impl(get_as_library(), quant_lightning_indexer_op_builder.name, "PrivateUse1")
def quant_lightning_indexer(
    q,
    k,
    w,
    q_descale,
    k_descale,
    topk,
    quant_mode,
    *,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    metadata=None,
    max_seqlen_q=-1,
    mask_mode=0,
    cmp_ratio=1,
    layout_q="TND",
    layout_kv="PA_BBND",
    return_value=False,
    candidate_topk_blocks=-1,
    candidate_block_size=-1,
):
    """Run the CANNBotDSL QLI kernel through the Torch NPU dispatcher."""
    outputs = _dsl_quant_lightning_indexer(
        q,
        k,
        w,
        q_descale,
        k_descale,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        block_table=block_table,
        metadata=metadata,
        topk=topk,
        quant_mode=quant_mode,
        max_seqlen_q=max_seqlen_q,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        layout_q=layout_q,
        layout_kv=layout_kv,
        return_value=return_value,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
    )
    return (
        outputs["sparse_indices"],
        outputs["sparse_values"],
        outputs["candidate_block_indices"],
        outputs["candidate_block_length"],
    )
