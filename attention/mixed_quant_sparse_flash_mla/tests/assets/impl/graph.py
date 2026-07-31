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

"""Mainline TTK graph adapter with metadata generated before capture."""

import torch


class MixedQuantSparseFlashMlaAclGraph(torch.nn.Module):
    def __init__(
        self,
        q,
        *,
        ori_kv=None,
        cmp_kv=None,
        ori_sparse_indices=None,
        cmp_sparse_indices=None,
        ori_block_table=None,
        cmp_block_table=None,
        cu_seqlens_q=None,
        cu_seqlens_ori_kv=None,
        cu_seqlens_cmp_kv=None,
        seqused_q=None,
        seqused_ori_kv=None,
        seqused_cmp_kv=None,
        cmp_residual_kv=None,
        ori_topk_length=None,
        cmp_topk_length=None,
        sinks=None,
        quant_mode=1,
        rope_head_dim=64,
        softmax_scale=1.0,
        cmp_ratio=1,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="BSND",
        layout_kv="BSND",
        topk_value_mode=1,
        return_softmax_lse=False,
        has_ori_kv=None,
        has_cmp_kv=None,
        metadata_cmp_topk=None,
        key_dtype=None,
        value_dtype=None,
    ):
        super().__init__()
        from mqsmla_ttk_ops import build_mixed_quant_sparse_flash_mla_metadata

        self.q = q
        self.ori_kv = ori_kv
        self.cmp_kv = cmp_kv
        self.ori_sparse_indices = ori_sparse_indices
        self.cmp_sparse_indices = cmp_sparse_indices
        self.ori_block_table = ori_block_table
        self.cmp_block_table = cmp_block_table
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_ori_kv = cu_seqlens_ori_kv
        self.cu_seqlens_cmp_kv = cu_seqlens_cmp_kv
        self.seqused_q = seqused_q
        self.seqused_ori_kv = seqused_ori_kv
        self.seqused_cmp_kv = seqused_cmp_kv
        self.cmp_residual_kv = cmp_residual_kv
        self.ori_topk_length = ori_topk_length
        self.cmp_topk_length = cmp_topk_length
        self.sinks = sinks
        self.quant_mode = int(quant_mode)
        self.rope_head_dim = int(rope_head_dim)
        self.softmax_scale = float(softmax_scale)
        self.cmp_ratio = int(cmp_ratio)
        self.ori_mask_mode = int(ori_mask_mode)
        self.cmp_mask_mode = int(cmp_mask_mode)
        self.ori_win_left = int(ori_win_left)
        self.ori_win_right = int(ori_win_right)
        self.layout_q = layout_q
        self.layout_kv = layout_kv
        self.topk_value_mode = int(topk_value_mode)
        self.return_softmax_lse = bool(return_softmax_lse)
        self.key_dtype = key_dtype
        self.value_dtype = value_dtype
        self.metadata = build_mixed_quant_sparse_flash_mla_metadata(
            q,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
            ori_sparse_indices=ori_sparse_indices,
            cmp_sparse_indices=cmp_sparse_indices,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
            seqused_q=seqused_q,
            seqused_ori_kv=seqused_ori_kv,
            seqused_cmp_kv=seqused_cmp_kv,
            cmp_residual_kv=cmp_residual_kv,
            ori_topk_length=ori_topk_length,
            cmp_topk_length=cmp_topk_length,
            quant_mode=quant_mode,
            rope_head_dim=rope_head_dim,
            cmp_ratio=cmp_ratio,
            ori_mask_mode=ori_mask_mode,
            cmp_mask_mode=cmp_mask_mode,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            layout_q=layout_q,
            layout_kv=layout_kv,
            has_ori_kv=has_ori_kv,
            has_cmp_kv=has_cmp_kv,
            metadata_cmp_topk=metadata_cmp_topk,
        )

    def forward(self):
        return torch.ops.cann_ops_transformer.mixed_quant_sparse_flash_mla(
            self.q,
            ori_kv=self.ori_kv,
            cmp_kv=self.cmp_kv,
            ori_sparse_indices=self.ori_sparse_indices,
            cmp_sparse_indices=self.cmp_sparse_indices,
            ori_block_table=self.ori_block_table,
            cmp_block_table=self.cmp_block_table,
            cu_seqlens_q=self.cu_seqlens_q,
            cu_seqlens_ori_kv=self.cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=self.cu_seqlens_cmp_kv,
            seqused_q=self.seqused_q,
            seqused_ori_kv=self.seqused_ori_kv,
            seqused_cmp_kv=self.seqused_cmp_kv,
            cmp_residual_kv=self.cmp_residual_kv,
            ori_topk_length=self.ori_topk_length,
            cmp_topk_length=self.cmp_topk_length,
            sinks=self.sinks,
            metadata=self.metadata,
            quant_mode=self.quant_mode,
            rope_head_dim=self.rope_head_dim,
            softmax_scale=self.softmax_scale,
            cmp_ratio=self.cmp_ratio,
            ori_mask_mode=self.ori_mask_mode,
            cmp_mask_mode=self.cmp_mask_mode,
            ori_win_left=self.ori_win_left,
            ori_win_right=self.ori_win_right,
            layout_q=self.layout_q,
            layout_kv=self.layout_kv,
            topk_value_mode=self.topk_value_mode,
            return_softmax_lse=self.return_softmax_lse,
            key_dtype=self.key_dtype,
            value_dtype=self.value_dtype,
        )
