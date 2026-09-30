# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional
import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


def _indices_last_dim(
    topk: int, cmp_ratio: int, expand_indices: bool, append_tail: bool
) -> int:
    if not expand_indices:
        return topk
    expanded = topk * cmp_ratio
    return expanded + cmp_ratio - 1 if append_tail else expanded


class GenericLightningIndexerOpBuilder(OpBuilder):
    def __init__(self):
        super(GenericLightningIndexerOpBuilder, self).__init__(
            "generic_lightning_indexer", category="attention"
        )

    def sources(self):
        return ["csrc/attention/generic_lightning_indexer.cpp"]

    def schema(self) -> str:
        return (
            "generic_lightning_indexer(Tensor query, Tensor key, *, "
            "Tensor? weights=None, Tensor? actual_seq_q=None, Tensor? actual_seq_k=None, "
            "Tensor? seqused_q=None, Tensor? seqused_k=None, Tensor? cmp_residual_k=None, "
            "Tensor? block_table=None, Tensor? q_descale=None, Tensor? k_descale=None, "
            "Tensor? output_idx_offset=None, Tensor? metadata=None, Tensor? start_loc=None, "
            'int topk=2048, int cmp_ratio=1, str layout_q="BSND", str layout_k="BSND", '
            "int sparse_mode=0, int act_type=0, bool has_weighted_sum=True, "
            "bool expand_indices=False, bool append_tail=False, int quant_mode=-1, "
            "bool return_value=False, int select_mode=0, int key_reduce_mode=0, "
            "float score_scale=1.0, int key_stride0=-1, int k_descale_stride0=-1, "
            "int max_seqlen_q=-1, int init_blocks=0, int local_blocks=0) -> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def generic_lightning_indexer_meta(
            query,
            key,
            *,
            weights=None,
            actual_seq_q=None,
            actual_seq_k=None,
            seqused_q=None,
            seqused_k=None,
            cmp_residual_k=None,
            block_table=None,
            q_descale=None,
            k_descale=None,
            output_idx_offset=None,
            metadata=None,
            start_loc=None,
            topk=2048,
            cmp_ratio=1,
            layout_q="BSND",
            layout_k="BSND",
            sparse_mode=0,
            act_type=0,
            has_weighted_sum=True,
            expand_indices=False,
            append_tail=False,
            quant_mode=-1,
            return_value=False,
            select_mode=0,
            key_reduce_mode=0,
            score_scale=1.0,
            key_stride0=-1,
            k_descale_stride0=-1,
            max_seqlen_q=-1,
            init_blocks=0,
            local_blocks=0,
        ):
            if select_mode == 1:
                pa_layouts = ("PA_BBND", "PA_BSND", "BBND", "BNBD")
                if layout_k in pa_layouts and block_table is not None:
                    n_raw = block_table.shape[1]
                elif layout_k == "TND":
                    n_raw = (key.shape[0] + cmp_ratio - 1) // cmp_ratio
                else:
                    n_raw = (key.shape[1] + cmp_ratio - 1) // cmp_ratio
                n_blocks = ((n_raw + 15) // 16) * 16
                if layout_q == "BSND":
                    val_shape = [
                        query.shape[0],
                        query.shape[1],
                        query.shape[2],
                        n_blocks,
                    ]
                else:
                    val_shape = [query.shape[1], query.shape[0], n_blocks]
                return (
                    torch.empty([0], dtype=torch.int32, device="meta"),
                    torch.empty(val_shape, dtype=torch.float32, device="meta"),
                )
            last = _indices_last_dim(topk, cmp_ratio, expand_indices, append_tail)
            if layout_q == "BSND":
                idx_shape = [query.shape[0], query.shape[1], last]
                val_shape = [query.shape[0], query.shape[1], topk]
            else:
                idx_shape = [query.shape[0], last]
                val_shape = [query.shape[0], topk]
            sparse_indices = torch.empty(idx_shape, dtype=torch.int32, device="meta")
            if return_value:
                sparse_values = torch.empty(
                    val_shape, dtype=torch.float32, device="meta"
                )
            else:
                sparse_values = torch.empty([0], dtype=torch.float32, device="meta")
            return (sparse_indices, sparse_values)


generic_lightning_indexer_op_builder = GenericLightningIndexerOpBuilder()
generic_lightning_indexer_op_builder._ensure_initialized()


@impl(get_as_library(), generic_lightning_indexer_op_builder.name, "PrivateUse1")
def generic_lightning_indexer(
    query,
    key,
    *,
    weights=None,
    actual_seq_q=None,
    actual_seq_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    block_table=None,
    q_descale=None,
    k_descale=None,
    output_idx_offset=None,
    metadata=None,
    start_loc=None,
    topk=2048,
    cmp_ratio=1,
    layout_q="BSND",
    layout_k="BSND",
    sparse_mode=0,
    act_type=0,
    has_weighted_sum=True,
    expand_indices=False,
    append_tail=False,
    quant_mode=-1,
    return_value=False,
    select_mode=0,
    key_reduce_mode=0,
    score_scale=1.0,
    key_stride0=-1,
    k_descale_stride0=-1,
    max_seqlen_q=-1,
    init_blocks=0,
    local_blocks=0,
):
    op_module = generic_lightning_indexer_op_builder.load()
    return op_module.generic_lightning_indexer(
        query,
        key,
        weights,
        actual_seq_q,
        actual_seq_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        block_table,
        q_descale,
        k_descale,
        output_idx_offset,
        metadata,
        start_loc,
        topk,
        cmp_ratio,
        layout_q,
        layout_k,
        sparse_mode,
        act_type,
        has_weighted_sum,
        expand_indices,
        append_tail,
        quant_mode,
        return_value,
        select_mode,
        key_reduce_mode,
        score_scale,
        key_stride0,
        k_descale_stride0,
        max_seqlen_q,
        init_blocks,
        local_blocks,
    )
