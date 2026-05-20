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
import torch_npu
from torch.library import impl
from npu_ops_transformer.op_builder.builder import OpBuilder
from npu_ops_transformer.op_builder.builder import AS_LIBRARY


class SparseFlashMlaOpBuilder(OpBuilder):
    def __init__(self):
        super(SparseFlashMlaOpBuilder, self).__init__("npu_sparse_flash_mla")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/sparse_flash_mla.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return "npu_sparse_flash_mla(Tensor q, *," \
                                            "Tensor? ori_kv=None, Tensor? cmp_kv=None, " \
                                            "Tensor? ori_sparse_indices=None, Tensor? cmp_sparse_indices=None, " \
                                            "Tensor? ori_block_table=None, Tensor? cmp_block_table=None, " \
                                            "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_ori_kv=None, " \
                                            "Tensor? cu_seqlens_cmp_kv=None, Tensor? seqused_q=None, " \
                                            "Tensor? seqused_ori_kv=None, Tensor? seqused_cmp_kv=None, " \
                                            "Tensor? cmp_residual_kv=None, " \
                                            "Tensor? ori_topk_length=None, Tensor? cmp_topk_length=None, " \
                                            "Tensor? sinks=None, Tensor? metadata=None, " \
                                            "int quant_mode = 1, int tile_size=64, int rope_head_dim=64, " \
                                            "float softmax_scale= 1.0, int cmp_ratio=1, " \
                                            "int ori_mask_mode=4, int cmp_mask_mode=3, " \
                                            "int ori_win_left=127, int ori_win_right=0, " \
                                            "str layout_q=\"BSND\", str layout_kv=\"PA_ND\", " \
                                            "int topk_value_mode=1, bool return_softmax_lse=False, " \
                                            "int? key_dtype=None, int? value_dtype=None) -> (Tensor, Tensor)"
 
    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_sparse_flash_mla_meta(q,
                                                ori_kv=None, cmp_kv=None,
                                                ori_sparse_indices=None, cmp_sparse_indices=None,
                                                ori_block_table=None, cmp_block_table=None,
                                                cu_seqlens_q=None, cu_seqlens_ori_kv=None,
                                                cu_seqlens_cmp_kv=None, seqused_q=None,
                                                seqused_ori_kv=None, seqused_cmp_kv=None,
                                                cmp_residual_kv=None,
                                                ori_topk_length=None, cmp_topk_length=None,
                                                sinks=None, metadata=None,
                                                quant_mode=1, tile_size=128, rope_head_dim=64,
                                                softmax_scale=1.0, cmp_ratio=1,
                                                ori_mask_mode=4, cmp_mask_mode=3,
                                                ori_win_left=127, ori_win_right=0,
                                                layout_q='BSND', layout_kv='PA_ND',
                                                topk_value_mode=1, return_softmax_lse=False,
                                                key_dtype=None, value_dtype=None):
            key_headnum = ori_kv.shape[1] if layout_kv == "TND" else ori_kv.shape[2]
            if layout_q == "BSND":
                ## 添加softmax_lse
                attn_out = torch.empty(q.shape, dtype=q.dtype, device="meta")
                if return_softmax_lse:
                    softmax_lse = torch.empty([q.shape[0], ori_kv.shape[2], q.shape[1], q.shape[2] / ori_kv.shape[2]],
                                              dtype=torch.float32, device="meta")
                else:
                    # 给一个空的合法张量，不能是 nullptr
                    softmax_lse = torch.empty([], dtype=torch.float32, device="meta")
            else:
                attn_out = torch.empty(q.shape, dtype=q.dtype, device="meta")
                if return_softmax_lse:
                    softmax_lse = torch.empty([ori_kv.shape[1], q.shape[0], q.shape[2] / ori_kv.shape[2]],
                                              dtype=torch.float32, device="meta")
                else:
                    # 给一个空的合法张量，不能是 nullptr
                    softmax_lse = torch.empty([], dtype=torch.float32, device="meta")
            return (attn_out, softmax_lse)
 
# Instantiate the builder
npu_sparse_flash_mla_op_builder = SparseFlashMlaOpBuilder()
op_module = npu_sparse_flash_mla_op_builder.load()  # Compiles/loads the .so file


@impl(AS_LIBRARY, npu_sparse_flash_mla_op_builder.name, "PrivateUse1")
def npu_sparse_flash_mla(q,
                                    ori_kv=None, cmp_kv=None,
                                    ori_sparse_indices=None, cmp_sparse_indices=None,
                                    ori_block_table=None, cmp_block_table=None,
                                    cu_seqlens_q=None, cu_seqlens_ori_kv=None,
                                    cu_seqlens_cmp_kv=None, seqused_q=None,
                                    seqused_ori_kv=None, seqused_cmp_kv=None,
                                    cmp_residual_kv=None,
                                    ori_topk_length=None, cmp_topk_length=None,
                                    sinks=None, metadata=None,
                                    quant_mode=1, tile_size=128, rope_head_dim=64,
                                    softmax_scale=1.0, cmp_ratio=1,
                                    ori_mask_mode=4, cmp_mask_mode=3,
                                    ori_win_left=127, ori_win_right=0,
                                    layout_q='BSND', layout_kv='PA_ND',
                                    topk_value_mode=1, return_softmax_lse=False,
                                    key_dtype=None, value_dtype=None):
    """
    dispatcher implementation for NPU.zhe
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    return op_module.npu_sparse_flash_mla(q,
                                                ori_kv, cmp_kv,
                                                ori_sparse_indices, cmp_sparse_indices,
                                                ori_block_table, cmp_block_table,
                                                cu_seqlens_q, cu_seqlens_ori_kv,
                                                cu_seqlens_cmp_kv, seqused_q,
                                                seqused_ori_kv, seqused_cmp_kv,
                                                cmp_residual_kv,
                                                ori_topk_length, cmp_topk_length,
                                                sinks, metadata,
                                                quant_mode, tile_size, rope_head_dim,
                                                softmax_scale, cmp_ratio,
                                                ori_mask_mode, cmp_mask_mode,
                                                ori_win_left, ori_win_right,
                                                layout_q, layout_kv,
                                                topk_value_mode, return_softmax_lse,
                                                key_dtype, value_dtype)