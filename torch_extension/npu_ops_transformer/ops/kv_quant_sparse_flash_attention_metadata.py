# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
import torchair
from torch.library import impl
from npu_ops_transformer.op_builder.builder import OpBuilder
from npu_ops_transformer.op_builder.builder import AS_LIBRARY
from typing import Optional
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType


class KvQuantSparseFlashAttentionMetadataOpBuilder(OpBuilder):
    def __init__(self):
        super(KvQuantSparseFlashAttentionMetadataOpBuilder, self).__init__("npu_kv_quant_sparse_flash_attention_metadata")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/kv_quant_sparse_flash_attention_metadata.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return "npu_kv_quant_sparse_flash_attention_metadata(int num_heads_q, int num_heads_kv, int head_dim, int kv_quant_mode," \
        " *, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_kv=None, Tensor? seqused_q=None, Tensor? seqused_kv=None, Tensor? topk_length=None," \
        "int? batch_size=None, int? max_seqlen_q=None, int? max_seqlen_kv=None, int? topk=None, int? tile_size=None, int? rope_head_dim=None, int? mask_mode=None, int? win_left=None, int? win_right=None, " \
        "str? layout_q=None, str? layout_kv=None) -> Tensor"

    def register_meta(self):
        """
        Registers Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_kv_quant_sparse_flash_attention_metadata_meta(
            num_heads_q: int, num_heads_kv: int, head_dim: int, kv_quant_mode: int, 
            cu_seqlens_q: Optional[Tensor] = None, cu_seqlens_kv: Optional[Tensor] = None,
            seqused_q: Optional[Tensor] = None, seqused_kv: Optional[Tensor] = None, 
            topk_length: Optional[Tensor] = None, batch_size: Optional[int] = None, max_seqlen_q: Optional[int] = None, 
            max_seqlen_kv: Optional[int] = None, topk: Optional[int] = None, tile_size: Optional[int] = None, 
            rope_head_dim: Optional[int] = None, mask_mode: Optional[int] = None, win_left: Optional[int] = None, 
            win_right: Optional[int] = None, layout_q: Optional[str] = None, layout_kv: Optional[str] = None):
            return torch.empty((1024), dtype=torch.int32, device="meta")

# Instantiate the builder
kv_quant_sparse_flash_attention_metadata_op_builder = KvQuantSparseFlashAttentionMetadataOpBuilder()
op_module = kv_quant_sparse_flash_attention_metadata_op_builder.load()


@impl(AS_LIBRARY, kv_quant_sparse_flash_attention_metadata_op_builder.name, "PrivateUse1")
def npu_kv_quant_sparse_flash_attention_metadata(
    num_heads_q: int, num_heads_kv: int, head_dim: int, kv_quant_mode: int, *, 
    cu_seqlens_q: Optional[Tensor] = None, cu_seqlens_kv: Optional[Tensor] = None, seqused_q: Optional[Tensor] = None, 
    seqused_kv: Optional[Tensor] = None, topk_length: Optional[Tensor] = None, batch_size: Optional[int] = None, 
    max_seqlen_q: Optional[int] = None, max_seqlen_kv: Optional[int] = None, topk: Optional[int] = None, 
    tile_size: Optional[int] = None, rope_head_dim: Optional[int] = None, mask_mode: Optional[int] = None, 
    win_left: Optional[int] = None, win_right: Optional[int] = None, layout_q: Optional[str] = None, 
    layout_kv: Optional[str] = None):
    """
    Dispatcher implementation: NPU.
    'PrivateUse1' is dispatch key for custom NPU backends.
    """
    batch_size = 0 if batch_size is None else batch_size
    max_seqlen_q = -1 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_kv = -1 if max_seqlen_kv is None else max_seqlen_kv
    topk = 0 if topk is None else topk
    tile_size = 0 if tile_size is None else tile_size
    rope_head_dim = 0 if rope_head_dim is None else rope_head_dim
    mask_mode = 0 if mask_mode is None else mask_mode
    win_left = -1 if win_left is None else win_left
    win_right = -1 if win_right is None else win_right
    layout_q = "BSND" if layout_q is None else layout_q
    layout_kv = "PA_BNBD" if layout_kv is None else layout_kv
 
    return op_module.npu_kv_quant_sparse_flash_attention_metadata(
        num_heads_q, num_heads_kv, head_dim, kv_quant_mode, cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, 
        topk_length, batch_size, max_seqlen_q, max_seqlen_kv, topk, tile_size, rope_head_dim, mask_mode, win_left, 
        win_right, layout_q, layout_kv)


@torch.library.register_kernel("npu_ops_transformer::npu_kv_quant_sparse_flash_attention_metadata", None)
def npu_kv_quant_sparse_flash_attention_metadata_fallback(
    num_heads_q: int, num_heads_kv: int, head_dim: int, kv_quant_mode: int, *, 
    cu_seqlens_q: Optional[Tensor] = None, cu_seqlens_kv: Optional[Tensor] = None, seqused_q: Optional[Tensor] = None, 
    seqused_kv: Optional[Tensor] = None, topk_length: Optional[Tensor] = None, batch_size: Optional[int] = None, 
    max_seqlen_q: Optional[int] = None, max_seqlen_kv: Optional[int] = None, topk: Optional[int] = None, 
    tile_size: Optional[int] = None, rope_head_dim: Optional[int] = None, mask_mode: Optional[int] = None, 
    win_left: Optional[int] = None, win_right: Optional[int] = None, layout_q: Optional[str] = None, 
    layout_kv: Optional[str] = None):
    # 处理所有 tensor 都为 None 的情况
    # 可以在这里创建一个 NPU tensor 来触发 PrivateUse1 backend
    if all(t is None for t in [cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv]):
             _ = torch.empty(1, dtype=torch.int32, device="npu")
    # 调用 NPU 实现
    batch_size = 0 if batch_size is None else batch_size
    max_seqlen_q = -1 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_kv = -1 if max_seqlen_kv is None else max_seqlen_kv
    topk = 0 if topk is None else topk
    tile_size = 0 if tile_size is None else tile_size
    rope_head_dim = 0 if rope_head_dim is None else rope_head_dim
    mask_mode = 0 if mask_mode is None else mask_mode
    win_left = -1 if win_left is None else win_left
    win_right = -1 if win_right is None else win_right
    layout_q = "BSND" if layout_q is None else layout_q
    layout_kv = "PA_BNBD" if layout_kv is None else layout_kv
 
    return op_module.npu_kv_quant_sparse_flash_attention_metadata(
        num_heads_q, num_heads_kv, head_dim, kv_quant_mode, cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, 
        topk_length, batch_size, max_seqlen_q, max_seqlen_kv, topk, tile_size, rope_head_dim, mask_mode, win_left, 
        win_right, layout_q, layout_kv)
        
torch.compiler.allow_in_graph(npu_kv_quant_sparse_flash_attention_metadata)