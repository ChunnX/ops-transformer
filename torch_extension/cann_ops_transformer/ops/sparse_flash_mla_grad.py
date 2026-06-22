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
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY


class SparseFlashMlaGradOpBuilder(OpBuilder):
    def __init__(self):
        super(SparseFlashMlaGradOpBuilder, self).__init__("sparse_flash_mla_grad")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/sparse_flash_mla_grad.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return [
            "sparse_flash_mla_grad_metadata(" \
            "int num_heads_q, int num_heads_kv, int head_dim, " \
            "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_ori_kv=None, Tensor? cu_seqlens_cmp_kv=None, " \
            "Tensor? seqused_q=None, Tensor? seqused_ori_kv=None, Tensor? seqused_cmp_kv=None, " \
            "Tensor? cmp_residual_kv=None, Tensor? ori_topk_length=None, Tensor? cmp_topk_length=None, " \
            "int batch_size=None, int max_seqlen_q=None, int max_seqlen_ori_kv=None, int max_seqlen_cmp_kv=None, " \
            "int ori_topk=None, int cmp_topk=None, int cmp_ratio=None, int ori_mask_mode=0, int cmp_mask_mode=0," \
            "int ori_win_left=-1, int ori_win_right=-1, " \
            "str layout_q=\"BSND\", str layout_kv=\"BSND\", bool has_ori_kv=True, bool has_cmp_kv=True) -> Tensor",

            "sparse_flash_mla_grad(Tensor q, Tensor dout, Tensor attn_out, Tensor softmax_lse," \
            "Tensor? ori_kv=None, Tensor? cmp_kv=None, Tensor? ori_sparse_indices=None, " \
            "Tensor? cmp_sparse_indices=None, Tensor?cu_seqlens_q=None,"\
            "Tensor? cu_seqlens_ori_kv=None, Tensor? cu_seqlens_cmp_kv=None, Tensor? seqused_q=None," \
            "Tensor? seqused_ori_kv=None, Tensor? seqused_cmp_kv=None, Tensor? cmp_residual_kv=None," \
            "Tensor? ori_topk_length=None, Tensor? cmp_topk_length=None, Tensor? sinks=None," \
            "Tensor? metadata=None," \
            "float softmax_scale=None, int cmp_ratio=None, int ori_mask_mode=0, int cmp_mask_mode=0," \
            "int ori_win_left=-1, int ori_win_right=-1," \
            "str layout_q=\"BSND\", str layout_kv=\"BSND\")->" \
            "(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
        ]

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @torch.library.register_fake("cann_ops_transformer::sparse_flash_mla_grad_metadata")
        def sparse_flash_mla_grad_metadata_meta(
            num_heads_q, num_heads_kv, head_dim,
            cu_seqlens_q=None, cu_seqlens_ori_kv=None, cu_seqlens_cmp_kv=None,
            seqused_q=None, seqused_ori_kv=None, seqused_cmp_kv=None,
            cmp_residual_kv=None, ori_topk_length=None, cmp_topk_length=None,
            batch_size=None, max_seqlen_q=None, max_seqlen_ori_kv=None, max_seqlen_cmp_kv=None,
            ori_topk=None, cmp_topk=None, cmp_ratio=None, ori_mask_mode=0, cmp_mask_mode=0,
            ori_win_left=-1, ori_win_right=-1,
            layout_q="BSND", layout_kv="BSND", has_ori_kv=True, has_cmp_kv=True
        ):
            return None

        @impl(AS_LIBRARY, self.name, "Meta")
        def sparse_flash_mla_grad_meta(q, dout, attn_out, softmax_lse, ori_kv=None, cmp_kv=None,
                                ori_sparse_indices=None, cmp_sparse_indices=None, cu_seqlens_q=None,
                                cu_seqlens_ori_kv=None, cu_seqlens_cmp_kv=None, seqused_q=None,
                                seqused_ori_kv=None, seqused_cmp_kv=None, cmp_residual_kv=None, 
                                ori_topk_length=None, cmp_topk_length=None, 
                                sinks=None, metadata=None,
                                softmax_scale=None, cmp_ratio=None, ori_mask_mode=0, cmp_mask_mode=0,
                                ori_win_left=-1, ori_win_right=-1,
                                layout_q="BSND", layout_kv="BSND"):
            dq = q.new_empty(q.shape, dtype=q.dtype, device='meta')
            dori_kv = ori_kv.new_empty(ori_kv.shape, dtype=ori_kv.dtype, device='meta') if ori_kv is not None else None
            dcmp_kv = cmp_kv.new_empty(cmp_kv.shape, dtype=cmp_kv.dtype, device='meta') if cmp_kv is not None else None
            dsinks = sinks.new_empty(sinks.shape, dtype=sinks.dtype, device='meta') if sinks is not None else None
            ori_softmax_l1_norm = \
                ori_sparse_indices.new_empty(ori_sparse_indices.shape, dtype=ori_sparse_indices.dtype, device='meta') \
                if ori_sparse_indices is not None else None
            cmp_softmax_l1_norm = \
                cmp_sparse_indices.new_empty(cmp_sparse_indices.shape, dtype=cmp_sparse_indices.dtype, device='meta') \
                if cmp_sparse_indices is not None else None

            return (dq, dori_kv, dcmp_kv, dsinks, ori_softmax_l1_norm, cmp_softmax_l1_norm)


# Instantiate the builder
smlag_op_builder = SparseFlashMlaGradOpBuilder()
op_module = smlag_op_builder.load()  # Compiles/loads the .so file


@impl(AS_LIBRARY, "sparse_flash_mla_grad_metadata", "PrivateUse1")
def sparse_flash_mla_grad_metadata(num_heads_q, num_heads_kv, head_dim,
                                        cu_seqlens_q=None, cu_seqlens_ori_kv=None, cu_seqlens_cmp_kv=None,
                                        seqused_q=None, seqused_ori_kv=None, seqused_cmp_kv=None,
                                        cmp_residual_kv=None, ori_topk_length=None, cmp_topk_length=None,
                                        batch_size=None, max_seqlen_q=None, 
                                        max_seqlen_ori_kv=None, max_seqlen_cmp_kv=None,
                                        ori_topk=None, cmp_topk=None, cmp_ratio=None, 
                                        ori_mask_mode=0, cmp_mask_mode=0,
                                        ori_win_left=-1, ori_win_right=-1,
                                        layout_q="BSND", layout_kv="BSND", has_ori_kv=True, has_cmp_kv=True
                                    ):
    """
    dispatcher implementation for NPU.
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    return op_module.sparse_flash_mla_grad_metadata(num_heads_q, num_heads_kv, head_dim,
                                        cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv,
                                        seqused_q, seqused_ori_kv, seqused_cmp_kv,
                                        cmp_residual_kv, ori_topk_length, cmp_topk_length,
                                        batch_size, max_seqlen_q, max_seqlen_ori_kv, max_seqlen_cmp_kv,
                                        ori_topk, cmp_topk, cmp_ratio, ori_mask_mode, cmp_mask_mode,
                                        ori_win_left, ori_win_right,
                                        layout_q, layout_kv, has_ori_kv, has_cmp_kv
                                    )


@torch.library.register_kernel("cann_ops_transformer::sparse_flash_mla_grad_metadata", None)
def sparse_flash_mla_grad_metadata_fallback(num_heads_q, num_heads_kv, head_dim,
                                        cu_seqlens_q=None, cu_seqlens_ori_kv=None, cu_seqlens_cmp_kv=None,
                                        seqused_q=None, seqused_ori_kv=None, seqused_cmp_kv=None,
                                        cmp_residual_kv=None, ori_topk_length=None, cmp_topk_length=None,
                                        batch_size=None, max_seqlen_q=None, 
                                        max_seqlen_ori_kv=None, max_seqlen_cmp_kv=None,
                                        ori_topk=None, cmp_topk=None, cmp_ratio=None, 
                                        ori_mask_mode=0, cmp_mask_mode=0,
                                        ori_win_left=-1, ori_win_right=-1,
                                        layout_q="BSND", layout_kv="BSND", 
                                        has_ori_kv=True, has_cmp_kv=True
                                    ):
    # 处理所有 tensor 都为 None 的情况
    # 调用 NPU 实现
    return sparse_flash_mla_grad_metadata(num_heads_q, num_heads_kv, head_dim,
                                        cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv,
                                        seqused_q, seqused_ori_kv, seqused_cmp_kv,
                                        cmp_residual_kv, ori_topk_length, cmp_topk_length,
                                        batch_size, max_seqlen_q, max_seqlen_ori_kv, max_seqlen_cmp_kv,
                                        ori_topk, cmp_topk, cmp_ratio, ori_mask_mode, cmp_mask_mode,
                                        ori_win_left, ori_win_right,
                                        layout_q, layout_kv, has_ori_kv, has_cmp_kv
                                    )

torch.compiler.allow_in_graph(sparse_flash_mla_grad_metadata)


@impl(AS_LIBRARY, smlag_op_builder.name, "PrivateUse1")
def sparse_flash_mla_grad(q, dout, attn_out, softmax_lse, ori_kv=None, cmp_kv=None,
                                ori_sparse_indices=None, cmp_sparse_indices=None, cu_seqlens_q=None,
                                cu_seqlens_ori_kv=None, cu_seqlens_cmp_kv=None, seqused_q=None,
                                seqused_ori_kv=None, seqused_cmp_kv=None, cmp_residual_kv=None, 
                                ori_topk_length=None, cmp_topk_length=None, 
                                sinks=None, metadata=None,
                                softmax_scale=None, cmp_ratio=None, ori_mask_mode=0, cmp_mask_mode=0,
                                ori_win_left=-1, ori_win_right=-1,
                                layout_q="BSND", layout_kv="BSND"):
    """
    dispatcher implementation for NPU.
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    return op_module.sparse_flash_mla_grad(q, dout, attn_out, softmax_lse, ori_kv, cmp_kv,
                                ori_sparse_indices, cmp_sparse_indices, cu_seqlens_q,
                                cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q,
                                seqused_ori_kv, seqused_cmp_kv, cmp_residual_kv, 
                                ori_topk_length, cmp_topk_length, 
                                sinks, metadata,
                                softmax_scale, cmp_ratio, ori_mask_mode, cmp_mask_mode,
                                ori_win_left, ori_win_right,
                                layout_q, layout_kv)