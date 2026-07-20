#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Thin TTK wrapper for LightningIndexer with precomputed metadata."""

from typing import Optional

import torch

import cann_ops_transformer  # noqa: F401 - registers torch.ops namespace


def lightning_indexer_v2(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    *,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cmp_residual_k: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    output_idx_offset: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    topk: int = 0,
    max_seqlen_q: int = -1,
    layout_q: str = "BSND",
    layout_k: str = "BSND",
    mask_mode: int = 0,
    cmp_ratio: int = 1,
    return_value: int = 0,
):
    return torch.ops.cann_ops_transformer.lightning_indexer(
        q,
        k,
        w,
        int(topk),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        block_table=block_table,
        output_idx_offset=output_idx_offset,
        metadata=metadata,
        max_seqlen_q=int(max_seqlen_q),
        layout_q=layout_q,
        layout_k=layout_k,
        mask_mode=int(mask_mode),
        cmp_ratio=int(cmp_ratio),
        return_value=int(return_value),
    )
