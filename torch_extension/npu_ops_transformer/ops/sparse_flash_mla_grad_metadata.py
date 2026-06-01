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
from npu_ops_transformer.op_builder.builder import AS_LIBRARY
from npu_ops_transformer.op_builder.builder import OpBuilder

class SparseFlashMlaGradMetadataOpBuilder(OpBuilder):
    def __init__(self):
        super(SparseFlashMlaGradMetadataOpBuilder, self).__init__(
            "npu_sparse_flash_mla_grad_metadata"
        )

    def sources(self):
        return ["ops/csrc/sparse_flash_mla_grad_metadata.cpp"]

    def schema(self) -> str:
        return (
            "npu_sparse_flash_mla_grad_metadata(" \
            "int num_heads_q, int num_heads_kv, int head_dim, " \
            "*, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_ori_kv=None, Tensor? cu_seqlens_cmp_kv=None, " \
            "Tensor? seqused_q=None, Tensor? seqused_ori_kv=None, Tensor? seqused_cmp_kv=None, " \
            "Tensor? cmp_residual_kv=None, Tensor? ori_topk_length=None, Tensor? cmp_topk_length=None, " \
            "int batch_size=None, int max_seqlen_q=None, int max_seqlen_ori_kv=None, int max_seqlen_cmp_kv=None, " \
            "int ori_topk=None, int cmp_topk=None, int cmp_ratio=None, int ori_mask_mode=0, int cmp_mask_mode=0," \
            "int ori_win_left=-1, int ori_win_right=-1, " \
            "str layout_q=\"BSND\", str layout_kv=\"BSND\", bool has_ori_kv=True, bool has_cmp_kv=True) -> Tensor"
        )

    def register_meta(self):
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_sparse_flash_mla_grad_metadata_meta(
            num_heads_q, num_heads_kv, head_dim,
            *, cu_seqlens_q=None, cu_seqlens_ori_kv=None, cu_seqlens_cmp_kv=None,
            seqused_q=None, seqused_ori_kv=None, seqused_cmp_kv=None,
            cmp_residual_kv=None, ori_topk_length=None, cmp_topk_length=None,
            batch_size=None, max_seqlen_q=None, max_seqlen_ori_kv=None, max_seqlen_cmp_kv=None,
            ori_topk=None, cmp_topk=None, cmp_ratio=None, ori_mask_mode=0, cmp_mask_mode=0,
            ori_win_left=-1, ori_win_right=-1,
            layout_q="BSND", layout_kv="BSND", has_ori_kv=True, has_cmp_kv=True
        ):
            return None


smlag_meta_op_builder = SparseFlashMlaGradMetadataOpBuilder()
op_module = smlag_meta_op_builder.load()

@impl(AS_LIBRARY, smlag_meta_op_builder.name, "PrivateUse1")
def npu_sparse_flash_mla_grad_metadata(num_heads_q, num_heads_kv, head_dim,
                                        *, cu_seqlens_q=None, cu_seqlens_ori_kv=None, cu_seqlens_cmp_kv=None,
                                        seqused_q=None, seqused_ori_kv=None, seqused_cmp_kv=None,
                                        cmp_residual_kv=None, ori_topk_length=None, cmp_topk_length=None,
                                        batch_size=None, max_seqlen_q=None, max_seqlen_ori_kv=None, max_seqlen_cmp_kv=None,
                                        ori_topk=None, cmp_topk=None, cmp_ratio=None, ori_mask_mode=0, cmp_mask_mode=0,
                                        ori_win_left=-1, ori_win_right=-1,
                                        layout_q="BSND", layout_kv="BSND", has_ori_kv=True, has_cmp_kv=True
                                    ):
    """
    dispatcher implementation for NPU.
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    return op_module.npu_sparse_flash_mla_grad_metadata(num_heads_q, num_heads_kv, head_dim,
                                        cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv,
                                        seqused_q, seqused_ori_kv, seqused_cmp_kv,
                                        cmp_residual_kv, ori_topk_length, cmp_topk_length,
                                        batch_size, max_seqlen_q, max_seqlen_ori_kv, max_seqlen_cmp_kv,
                                        ori_topk, cmp_topk, cmp_ratio, ori_mask_mode, cmp_mask_mode,
                                        ori_win_left, ori_win_right,
                                        layout_q, layout_kv, has_ori_kv, has_cmp_kv
                                    )