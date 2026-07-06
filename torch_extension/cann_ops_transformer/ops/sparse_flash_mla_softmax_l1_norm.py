# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Optional

import torch
import torch_npu
from torch.library import impl

from cann_ops_transformer.op_builder.builder import AS_LIBRARY
from cann_ops_transformer.op_builder.builder import OpBuilder


SMLA_OP_NAME = "sparse_flash_mla_softmax_l1_norm"


class SparseFlashMlaSoftmaxL1NormOpBuilder(OpBuilder):
    def __init__(self):
        super(SparseFlashMlaSoftmaxL1NormOpBuilder, self).__init__(
            "sparse_flash_mla_softmax_l1_norm"
        )

    def sources(self):
        return ["ops/csrc/sparse_flash_mla_softmax_l1_norm.cpp"]

    def schema(self) -> str:
        return (
            "sparse_flash_mla_softmax_l1_norm("
            "Tensor query, Tensor key, Tensor lse, *, "
            "Tensor? sparse_indices=None, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, "
            "Tensor? seqused_q=None, Tensor? seqused_k=None, Tensor? cmp_residual_k=None, "
            "Tensor? topk_length=None, Tensor? metadata=None, "
            "float scale_value=1.0, int cmp_ratio=1, int mask_mode=0, "
            "str layout_q=\"BSND\", str layout_k=\"BSND\") -> Tensor"
        )

    def register_meta(self):
        @torch.library.register_fake("cann_ops_transformer::" + SMLA_OP_NAME)
        def sparse_flash_mla_softmax_l1_norm_meta(
            query,
            key,
            lse,
            *,
            sparse_indices=None,
            cu_seqlens_q=None,
            cu_seqlens_k=None,
            seqused_q=None,
            seqused_k=None,
            cmp_residual_k=None,
            topk_length=None,
            metadata=None,
            scale_value=1.0,
            cmp_ratio=1,
            mask_mode=0,
            layout_q="BSND",
            layout_k="BSND",
        ):
            if layout_q == "BSND":
                b = query.size(0)
                s1 = query.size(1)
                kv_head_num = key.size(2)
                s2 = key.size(1)
                out_shape = (b, s1, kv_head_num, s2)
            else:
                s1 = query.size(0)
                kv_head_num = key.size(1)
                s2 = key.size(0)
                out_shape = (s1, kv_head_num, s2)
            return torch.empty(out_shape, dtype=torch.float32, device="meta")


sparse_flash_mla_softmax_l1_norm_op_builder = SparseFlashMlaSoftmaxL1NormOpBuilder()
op_module = sparse_flash_mla_softmax_l1_norm_op_builder.load()


@impl(AS_LIBRARY, SMLA_OP_NAME, "PrivateUse1")
def sparse_flash_mla_softmax_l1_norm(
    query,
    key,
    lse,
    *,
    sparse_indices=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    topk_length=None,
    metadata=None,
    scale_value=1.0,
    cmp_ratio=1,
    mask_mode=0,
    layout_q="BSND",
    layout_k="BSND",
):
    """SparseFlashMlaSoftmaxL1Norm 前向计算，封装 aclnnSparseFlashMlaSoftmaxL1Norm。

    计算 SparseFlashMla 注意力的 Softmax L1Norm 结果，支持 Sliding Window Attention、
    Compressed Attention 以及 Sparse Compressed Attention 场景。

    Args:
        query (Tensor): 查询张量，shape (B,S1,N1,D) 或 (T1,N1,D)，dtype float16/bfloat16。
        key (Tensor): 键张量，shape (B,S2,N2,D) 或 (T2,N2,D)，dtype 与 query 一致，N2=1，D=512。
        lse (Tensor): 注意力正向计算的 lse 输出，shape (B,N2,S1,G) 或 (N2,T1,G)，dtype float32。
        sparse_indices (Tensor, optional): 稀疏场景下选择的注意力索引，dtype int32。
        cu_seqlens_q (Tensor, optional): TND 场景下 query 的累积序列长度，shape (B+1,)，dtype int32。
        cu_seqlens_k (Tensor, optional): TND 场景下 key 的累积序列长度，shape (B+1,)，dtype int32。
        seqused_q (Tensor, optional): 每个 batch 中 query 实际参与运算的 token 数，shape (B,)，dtype int32。
        seqused_k (Tensor, optional): 每个 batch 中 key 实际参与运算的 token 数，shape (B,)，dtype int32。
        cmp_residual_k (Tensor, optional): 每个 batch S2//cmpRatio 后的余数，shape (B,)，dtype int32。
        topk_length (Tensor, optional): 每行 query 对应的 key 实际可选的 topk 长度，dtype int32。
        metadata (Tensor, optional): 前置 AICPU 算子输出的 tiling metadata，shape (64,)，dtype int32。
        scale_value (float): 缩放系数，默认 1.0。
        cmp_ratio (int): key 的压缩率，取值 [1, 128]，默认 1。
        mask_mode (int): mask 模式，0=No mask，3=rightDownCausal，默认 0。
        layout_q (str): query 排列格式，支持 "BSND"/"TND"，默认 "BSND"。
        layout_k (str): key 排列格式，支持 "BSND"/"TND"，默认 "BSND"。

    Returns:
        Tensor: softmax L1Norm 结果，shape (B,S1,N2,S2) 或 (T1,N2,T2)，dtype float32。
    """
    return op_module.sparse_flash_mla_softmax_l1_norm(
        query,
        key,
        lse,
        sparse_indices,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        cmp_residual_k,
        topk_length,
        metadata,
        scale_value,
        cmp_ratio,
        mask_mode,
        layout_q,
        layout_k,
    )


@torch.library.register_kernel("cann_ops_transformer::" + SMLA_OP_NAME, None)
def sparse_flash_mla_softmax_l1_norm_fallback(
    query,
    key,
    lse,
    *,
    sparse_indices=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    cmp_residual_k=None,
    topk_length=None,
    metadata=None,
    scale_value=1.0,
    cmp_ratio=1,
    mask_mode=0,
    layout_q="BSND",
    layout_k="BSND",
):
    return sparse_flash_mla_softmax_l1_norm(
        query,
        key,
        lse,
        sparse_indices=sparse_indices,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        cmp_residual_k=cmp_residual_k,
        topk_length=topk_length,
        metadata=metadata,
        scale_value=scale_value,
        cmp_ratio=cmp_ratio,
        mask_mode=mask_mode,
        layout_q=layout_q,
        layout_k=layout_k,
    )


torch.compiler.allow_in_graph(sparse_flash_mla_softmax_l1_norm)
