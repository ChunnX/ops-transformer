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

from .builder import get_as_library


def lightning_indexer_schema(op_name):
    return (
        f"{op_name}(Tensor q, Tensor k, Tensor w, int topk, *, "
        "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, Tensor? seqused_q=None, "
        "Tensor? seqused_k=None, Tensor? cmp_residual_k=None, Tensor? block_table=None, "
        "Tensor? output_idx_offset=None, Tensor? metadata=None, int max_seqlen_q=-1, "
        'str layout_q="BSND", str layout_k="BSND", int mask_mode=0, int cmp_ratio=1, '
        "int return_value=0) -> (Tensor, Tensor)"
    )


def register_lightning_indexer(builder):
    @impl(get_as_library(), builder.name, "Meta")
    def lightning_indexer_meta(
        q,
        k,
        w,
        topk,
        *,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=None,
        cmp_residual_k=None,
        block_table=None,
        output_idx_offset=None,
        metadata=None,
        max_seqlen_q=-1,
        layout_q="BSND",
        layout_k="BSND",
        mask_mode=0,
        cmp_ratio=1,
        return_value=0,
    ):
        key_head_num = k.shape[1] if layout_k == "TND" else k.shape[2]
        output_shape = (
            [q.shape[0], q.shape[1], key_head_num, topk]
            if layout_q == "BSND"
            else [q.shape[0], key_head_num, topk]
        )
        sparse_indices = torch.empty(output_shape, dtype=torch.int32, device="meta")
        sparse_values = torch.empty(
            output_shape if return_value else [0], dtype=torch.float, device="meta"
        )
        return sparse_indices, sparse_values

    @impl(get_as_library(), builder.name, "PrivateUse1")
    def lightning_indexer(
        q,
        k,
        w,
        topk,
        *,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=None,
        cmp_residual_k=None,
        block_table=None,
        output_idx_offset=None,
        metadata=None,
        max_seqlen_q=-1,
        layout_q="BSND",
        layout_k="BSND",
        mask_mode=0,
        cmp_ratio=1,
        return_value=0,
    ):
        op = getattr(builder.load(), builder.name)
        return op(
            q,
            k,
            w,
            topk,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            cmp_residual_k,
            block_table,
            output_idx_offset,
            metadata,
            max_seqlen_q,
            layout_q,
            layout_k,
            mask_mode,
            cmp_ratio,
            return_value,
        )

    return lightning_indexer
