# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Optional
import torch
import torchair
from torch.library import impl
from npu_ops_transformer.op_builder.builder import OpBuilder
from npu_ops_transformer.op_builder.builder import AS_LIBRARY
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType


class SparseFlashMlaMetadataOpBuilder(OpBuilder):
    def __init__(self):
        super(SparseFlashMlaMetadataOpBuilder, self).__init__("npu_sparse_flash_mla_metadata")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/sparse_flash_mla_metadata.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return "npu_sparse_flash_mla_metadata(int num_heads_q, int num_heads_kv, int head_dim, *, " \
            "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_ori_kv=None, Tensor? cu_seqlens_cmp_kv=None, " \
            "Tensor? seqused_q=None, Tensor? seqused_ori_kv=None, Tensor? seqused_cmp_kv=None, " \
            "Tensor? cmp_residual_kv=None, Tensor? ori_topk_length=None, Tensor? cmp_topk_length=None, " \
            "int? batch_size=None, int? max_seqlen_q=None, int? max_seqlen_ori_kv=None, int? max_seqlen_cmp_kv=None," \
            "int? ori_topk=None, int? cmp_topk=None, int? cmp_ratio=None, int? ori_mask_mode=None," \
            "int? cmp_mask_mode=None, int? ori_win_left=None, int? ori_win_right=None, str? layout_q=None," \
            "str? layout_kv=None, bool? has_ori_kv=None, bool? has_cmp_kv=None, int? batch_invariant=None) -> Tensor"

    def register_meta(self):
        """
        Registers Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @torch.library.register_fake("npu_ops_transformer::" + self.name)
        def npu_sparse_flash_mla_metadata_meta(
            num_heads_q: int, num_heads_kv: int, head_dim: int, cu_seqlens_q: Optional[Tensor] = None,
            cu_seqlens_ori_kv: Optional[Tensor] = None, cu_seqlens_cmp_kv: Optional[Tensor] = None,
            seqused_q: Optional[Tensor] = None, seqused_ori_kv: Optional[Tensor] = None,
            seqused_cmp_kv: Optional[Tensor] = None, cmp_residual_kv: Optional[Tensor] = None,
            ori_topk_length: Optional[Tensor] = None, cmp_topk_length: Optional[Tensor] = None,
            batch_size: Optional[int] = None, max_seqlen_q: Optional[int] = None,
            max_seqlen_ori_kv: Optional[int] = None, max_seqlen_cmp_kv: Optional[int] = None,
            ori_topk: Optional[int] = None, cmp_topk: Optional[int] = None, cmp_ratio: Optional[int] = None,
            ori_mask_mode: Optional[int] = None, cmp_mask_mode: Optional[int] = None,
            ori_win_left: Optional[int] = None, ori_win_right: Optional[int] = None, layout_q: Optional[str] = None,
            layout_kv: Optional[str] = None, has_ori_kv: Optional[bool] = None, has_cmp_kv: Optional[bool] = None,
            batch_invariant: Optional[int] = None):
            return torch.empty((1024), dtype=torch.int32, device="npu")

# Instantiate the builder
sparse_flash_mla_metadata_op_builder = SparseFlashMlaMetadataOpBuilder()
op_module = sparse_flash_mla_metadata_op_builder.load()


@impl(AS_LIBRARY, sparse_flash_mla_metadata_op_builder.name, "PrivateUse1")
def npu_sparse_flash_mla_metadata(
    num_heads_q: int, num_heads_kv: int, head_dim: int, cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_ori_kv: Optional[Tensor] = None, cu_seqlens_cmp_kv: Optional[Tensor] = None,
    seqused_q: Optional[Tensor] = None, seqused_ori_kv: Optional[Tensor] = None,
    seqused_cmp_kv: Optional[Tensor] = None, cmp_residual_kv: Optional[Tensor] = None,
    ori_topk_length: Optional[Tensor] = None, cmp_topk_length: Optional[Tensor] = None,
    batch_size: Optional[int] = None, max_seqlen_q: Optional[int] = None, max_seqlen_ori_kv: Optional[int] = None,
    max_seqlen_cmp_kv: Optional[int] = None, ori_topk: Optional[int] = None, cmp_topk: Optional[int] = None,
    cmp_ratio: Optional[int] = None, ori_mask_mode: Optional[int] = None, cmp_mask_mode: Optional[int] = None,
    ori_win_left: Optional[int] = None, ori_win_right: Optional[int] = None, layout_q: Optional[str] = None,
    layout_kv: Optional[str] = None, has_ori_kv: Optional[bool] = None, has_cmp_kv: Optional[bool] = None,
    batch_invariant: Optional[int] = None):
    """
    Dispatcher implementation: NPU.
    'PrivateUse1' is dispatch key for custom NPU backends.
    """
    batch_size = 0 if batch_size is None else batch_size
    max_seqlen_q = 0 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_ori_kv = 0 if max_seqlen_ori_kv is None else max_seqlen_ori_kv
    max_seqlen_cmp_kv = 0 if max_seqlen_cmp_kv is None else max_seqlen_cmp_kv
    ori_topk = 0 if ori_topk is None else ori_topk
    cmp_topk = 0 if cmp_topk is None else cmp_topk
    cmp_ratio = 1 if cmp_ratio is None else cmp_ratio
    ori_mask_mode = 0 if ori_mask_mode is None else ori_mask_mode
    cmp_mask_mode = 0 if cmp_mask_mode is None else cmp_mask_mode
    ori_win_left = -1 if ori_win_left is None else ori_win_left
    ori_win_right = -1 if ori_win_right is None else ori_win_right
    layout_q = "BSND" if layout_q is None else layout_q
    layout_kv = "BSND" if layout_kv is None else layout_kv
    has_ori_kv = True if has_ori_kv is None else has_ori_kv
    has_cmp_kv = True if has_cmp_kv is None else has_cmp_kv
    batch_invariant = 0 if batch_invariant is None else batch_invariant

    return op_module.npu_sparse_flash_mla_metadata(
        num_heads_q, num_heads_kv, head_dim, cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q,
        seqused_ori_kv, seqused_cmp_kv, cmp_residual_kv, ori_topk_length, cmp_topk_length, batch_size, max_seqlen_q,
        max_seqlen_ori_kv, max_seqlen_cmp_kv, ori_topk, cmp_topk, cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left,
        ori_win_right, layout_q, layout_kv, has_ori_kv, has_cmp_kv, batch_invariant)


@torch.library.register_kernel("npu_ops_transformer::npu_sparse_flash_mla_metadata", None)
def npu_sparse_flash_mla_metadata_fallback(
    num_heads_q: int, num_heads_kv: int, head_dim: int, cu_seqlens_q: Optional[Tensor] = None,
    cu_seqlens_ori_kv: Optional[Tensor] = None, cu_seqlens_cmp_kv: Optional[Tensor] = None,
    seqused_q: Optional[Tensor] = None, seqused_ori_kv: Optional[Tensor] = None,
    seqused_cmp_kv: Optional[Tensor] = None, cmp_residual_kv: Optional[Tensor] = None,
    ori_topk_length: Optional[Tensor] = None, cmp_topk_length: Optional[Tensor] = None,
    batch_size: Optional[int] = None, max_seqlen_q: Optional[int] = None, max_seqlen_ori_kv: Optional[int] = None,
    max_seqlen_cmp_kv: Optional[int] = None, ori_topk: Optional[int] = None, cmp_topk: Optional[int] = None,
    cmp_ratio: Optional[int] = None, ori_mask_mode: Optional[int] = None, cmp_mask_mode: Optional[int] = None,
    ori_win_left: Optional[int] = None, ori_win_right: Optional[int] = None, layout_q: Optional[str] = None,
    layout_kv: Optional[str] = None, has_ori_kv: Optional[bool] = None, has_cmp_kv: Optional[bool] = None,
    batch_invariant: Optional[int] = None):
    # 处理所有 tensor 都为 None 的情况
    # 调用 NPU 实现
    return npu_sparse_flash_mla_metadata(
        num_heads_q, num_heads_kv, head_dim, cu_seqlens_q, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q,
        seqused_ori_kv, seqused_cmp_kv, cmp_residual_kv, ori_topk_length, cmp_topk_length, batch_size, max_seqlen_q,
        max_seqlen_ori_kv, max_seqlen_cmp_kv, ori_topk, cmp_topk, cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left,
        ori_win_right, layout_q, layout_kv, has_ori_kv, has_cmp_kv, batch_invariant)

torch.compiler.allow_in_graph(npu_sparse_flash_mla_metadata)
