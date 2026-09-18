# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class FusedQuantLightningIndexerManageOpBuilder(OpBuilder):
    def __init__(self):
        super(FusedQuantLightningIndexerManageOpBuilder, self).__init__(
            "fused_quant_lightning_indexer_manage", category="attention"
        )

    def sources(self):
        return ["csrc/attention/fused_quant_lightning_indexer_manage.cpp"]

    def schema(self) -> str:
        return (
            "fused_quant_lightning_indexer_manage("
            "Tensor index_weights, Tensor query_dequant_scale, Tensor query, "
            "Tensor index_key_dequant_scale, Tensor index_key_cache, "
            "Tensor index_block_table, Tensor actual_seq_lengths_query, "
            "Tensor actual_seq_lengths_key, Tensor offload_seq_lengths_key, "
            "Tensor num_cache_tokens, Tensor request_state, "
            "Tensor req_pool_entries, Tensor(a!) cache_slots_pool, "
            "Tensor(b!) topk_src_ids, Tensor(c!) topk_dst_slots, "
            "Tensor(d!) topk_miss_counts, Tensor(e!) miss_src_ids, "
            "Tensor(f!) miss_dst_slots, Tensor(g!) miss_counts"
            ") -> ()"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def fused_quant_lightning_indexer_manage_meta(
            index_weights: torch.Tensor,
            query_dequant_scale: torch.Tensor,
            query: torch.Tensor,
            index_key_dequant_scale: torch.Tensor,
            index_key_cache: torch.Tensor,
            index_block_table: torch.Tensor,
            actual_seq_lengths_query: torch.Tensor,
            actual_seq_lengths_key: torch.Tensor,
            offload_seq_lengths_key: torch.Tensor,
            num_cache_tokens: torch.Tensor,
            request_state: torch.Tensor,
            req_pool_entries: torch.Tensor,
            cache_slots_pool: torch.Tensor,
            topk_src_ids: torch.Tensor,
            topk_dst_slots: torch.Tensor,
            topk_miss_counts: torch.Tensor,
            miss_src_ids: torch.Tensor,
            miss_dst_slots: torch.Tensor,
            miss_counts: torch.Tensor,
        ) -> None:
            return None


_op_builder = FusedQuantLightningIndexerManageOpBuilder()
_op_builder._ensure_initialized()


@impl(get_as_library(), _op_builder.name, "PrivateUse1")
def _fused_quant_lightning_indexer_manage(
    index_weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    query: torch.Tensor,
    index_key_dequant_scale: torch.Tensor,
    index_key_cache: torch.Tensor,
    index_block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    offload_seq_lengths_key: torch.Tensor,
    num_cache_tokens: torch.Tensor,
    request_state: torch.Tensor,
    req_pool_entries: torch.Tensor,
    cache_slots_pool: torch.Tensor,
    topk_src_ids: torch.Tensor,
    topk_dst_slots: torch.Tensor,
    topk_miss_counts: torch.Tensor,
    miss_src_ids: torch.Tensor,
    miss_dst_slots: torch.Tensor,
    miss_counts: torch.Tensor,
) -> None:
    op_module = _op_builder.load()
    return op_module.fused_quant_lightning_indexer_manage(
        index_weights,
        query_dequant_scale,
        query,
        index_key_dequant_scale,
        index_key_cache,
        index_block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        offload_seq_lengths_key,
        num_cache_tokens,
        request_state,
        req_pool_entries,
        cache_slots_pool,
        topk_src_ids,
        topk_dst_slots,
        topk_miss_counts,
        miss_src_ids,
        miss_dst_slots,
        miss_counts,
    )


def fused_quant_lightning_indexer_manage(
    index_weights: torch.Tensor,
    query_dequant_scale: torch.Tensor,
    query: torch.Tensor,
    index_key_dequant_scale: torch.Tensor,
    index_key_cache: torch.Tensor,
    index_block_table: torch.Tensor,
    actual_seq_lengths_query: torch.Tensor,
    actual_seq_lengths_key: torch.Tensor,
    offload_seq_lengths_key: torch.Tensor,
    num_cache_tokens: torch.Tensor,
    request_state: torch.Tensor,
    req_pool_entries: torch.Tensor,
    cache_slots_pool: torch.Tensor,
    topk_src_ids: torch.Tensor,
    topk_dst_slots: torch.Tensor,
    topk_miss_counts: torch.Tensor,
    miss_src_ids: torch.Tensor,
    miss_dst_slots: torch.Tensor,
    miss_counts: torch.Tensor,
) -> None:
    """Run quantized Lightning Indexer and write results to caller-owned output tensors."""
    torch.ops.cann_ops_transformer.fused_quant_lightning_indexer_manage(
        index_weights,
        query_dequant_scale,
        query,
        index_key_dequant_scale,
        index_key_cache,
        index_block_table,
        actual_seq_lengths_query,
        actual_seq_lengths_key,
        offload_seq_lengths_key,
        num_cache_tokens,
        request_state,
        req_pool_entries,
        cache_slots_pool,
        topk_src_ids,
        topk_dst_slots,
        topk_miss_counts,
        miss_src_ids,
        miss_dst_slots,
        miss_counts,
    )
