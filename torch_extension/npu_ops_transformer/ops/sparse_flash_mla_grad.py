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


class SparseFlashMlaGradOpBuilder(OpBuilder):
    def __init__(self):
        super(SparseFlashMlaGradOpBuilder, self).__init__("npu_sparse_flash_mla_grad")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/sparse_flash_mla_grad.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return "npu_sparse_flash_mla_grad(Tensor q, Tensor dout, Tensor attn_out, Tensor softmax_lse," \
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

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_sparse_flash_mla_grad_meta(q, dout, attn_out, softmax_lse, ori_kv=None, cmp_kv=None,
                                ori_sparse_indices=None, cmp_sparse_indices=None, cu_seqlens_q=None,
                                cu_seqlens_ori_kv=None, cu_seqlens_cmp_kv=None, seqused_q=None,
                                seqused_ori_kv=None, seqused_cmp_kv=None, cmp_residual_kv=None, 
                                ori_topk_length=None, cmp_topk_length=None, 
                                sinks=None, metadata=None,
                                softmax_scale=None, cmp_ratio=None, ori_mask_mode=0, cmp_mask_mode=0,
                                ori_win_left=-1, ori_win_right=-1,
                                layout_q="BSND", layout_kv="BSND"):
            dq = q.new_empty(q.shape, dtype=q.dtype, device='meta')
            dOriKv = ori_kv.new_empty(ori_kv.shape, dtype=ori_kv.dtype, device='meta') if ori_kv != None else None
            dCmpKv = cmp_kv.new_empty(cmp_kv.shape, dtype=cmp_kv.dtype, device='meta') if cmp_kv != None else None
            dSinks = sinks.new_empty(sinks.shape, dtype=sinks.dtype, device='meta') if sinks != None else None
            oriSoftmaxL1Norm = ori_sparse_indices.new_empty(ori_sparse_indices.shape, dtype=ori_sparse_indices.dtype, device='meta') if ori_sparse_indices != None else None
            cmpSoftmaxL1Norm = cmp_sparse_indices.new_empty(cmp_sparse_indices.shape, dtype=cmp_sparse_indices.dtype, device='meta') if cmp_sparse_indices != None else None

            return (dq, dOriKv, dCmpKv, dSinks, oriSoftmaxL1Norm, cmpSoftmaxL1Norm)


# Instantiate the builder
smlag_op_builder = SparseFlashMlaGradOpBuilder()
op_module = smlag_op_builder.load()  # Compiles/loads the .so file


@impl(AS_LIBRARY, smlag_op_builder.name, "PrivateUse1")
def npu_sparse_flash_mla_grad(q, dout, attn_out, softmax_lse, ori_kv=None, cmp_kv=None,
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
    return op_module.npu_sparse_flash_mla_grad(q, dout, attn_out, softmax_lse, ori_kv, cmp_kv,
                                ori_sparse_indices, cmp_sparse_indices, cu_seqlens_q,
                                cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q,
                                seqused_ori_kv, seqused_cmp_kv, cmp_residual_kv, 
                                ori_topk_length, cmp_topk_length, 
                                sinks, metadata,
                                softmax_scale, cmp_ratio, ori_mask_mode, cmp_mask_mode,
                                ori_win_left, ori_win_right,
                                layout_q, layout_kv)