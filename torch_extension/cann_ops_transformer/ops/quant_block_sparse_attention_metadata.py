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
from npu_ops_transformer.op_builder.builder import OpBuilder
from npu_ops_transformer.op_builder.builder import AS_LIBRARY
from typing import Optional

class QuantBlockSparseAttentionMetadataOpBuilder(OpBuilder):
    def __init__(self):
        super(QuantBlockSparseAttentionMetadataOpBuilder, self).__init__(
            "npu_quant_block_sparse_attention_metadata")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/quant_block_sparse_attention_metadata.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return "npu_quant_block_sparse_attention_metadata(Tensor sparse_seq_len, int num_heads_q, int num_heads_kv, int head_dim, *, "                         \
            "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_kv=None, Tensor? seqused_q=None, "                \
            "Tensor? seqused_kv=None, int batch_size=0, int sparse_block_size_q=128, int sparse_block_size_kv=128, int quant_mode=0, "  \
            "int mask_mode=3, int max_seqlen_q=0, int max_seqlen_kv=0, str layout_q='BSND', "               \
            "str layout_kv='PA_BNSD', str layout_sparse_indices='B_N_Qb_Kb') -> Tensor"

    def register_meta(self):
        """
        Registers Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @torch.library.register_fake("npu_ops_transformer::" + self.name)
        def npu_quant_block_sparse_attention_metadata_meta(
            sparse_seq_len: torch.Tensor, num_heads_q: int, num_heads_kv: int, head_dim: int, 
            cu_seqlens_q: Optional[torch.Tensor] = None,
            cu_seqlens_kv: Optional[torch.Tensor] = None, seqused_q: Optional[torch.Tensor] = None,
            seqused_kv: Optional[torch.Tensor] = None, batch_size: Optional[int] = None,
             sparse_block_size_q: Optional[int] = None,
            sparse_block_size_kv: Optional[int] = None, quant_mode: Optional[int] = None,
            mask_mode: Optional[int] = None, max_seqlen_q: Optional[int] = None,
            max_seqlen_kv: Optional[int] = None, layout_q: Optional[str] = None,
            layout_kv: Optional[str] = None, layout_sparse_indices: Optional[str] = None):
            return torch.empty((2048), dtype=torch.int32, device="npu")


# Instantiate the builder
quant_block_sparse_attention_metadata_op_builder = QuantBlockSparseAttentionMetadataOpBuilder()
op_module = quant_block_sparse_attention_metadata_op_builder.load()


@impl(AS_LIBRARY, quant_block_sparse_attention_metadata_op_builder.name, "PrivateUse1")
def npu_quant_block_sparse_attention_metadata(
    sparse_seq_len: torch.Tensor, num_heads_q: int, num_heads_kv: int, head_dim: int,
    cu_seqlens_q: Optional[torch.Tensor] = None, cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None, seqused_kv: Optional[torch.Tensor] = None, batch_size: Optional[int] = None,
    sparse_block_size_q: Optional[int] = None, sparse_block_size_kv: Optional[int] = None,
    quant_mode: Optional[int] = None, mask_mode: Optional[int] = None, max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None, layout_q: Optional[str] = None, layout_kv: Optional[str] = None,
    layout_sparse_indices: Optional[str] = None):
    """
    Dispatcher implementation: NPU.
    'PrivateUse1' is dispatch key for custom NPU backends.
    """
    batch_size = 0 if batch_size is None else batch_size
    sparse_block_size_q = 128 if sparse_block_size_q is None else sparse_block_size_q
    sparse_block_size_kv = 128 if sparse_block_size_kv is None else sparse_block_size_kv
    quant_mode = 0 if quant_mode is None else quant_mode
    mask_mode = 3 if mask_mode is None else mask_mode
    max_seqlen_q = 0 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_kv = 0 if max_seqlen_kv is None else max_seqlen_kv
    layout_q = "BSND" if layout_q is None else layout_q
    layout_kv = "PA_BNSD" if layout_kv is None else layout_kv
    layout_sparse_indices = "B_N_Qb_Kb" if layout_sparse_indices is None else layout_sparse_indices

    return op_module.npu_quant_block_sparse_attention_metadata(
        sparse_seq_len, num_heads_q, num_heads_kv, head_dim, 
        cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, batch_size,  sparse_block_size_q, sparse_block_size_kv, quant_mode, mask_mode, max_seqlen_q, max_seqlen_kv,
        layout_q, layout_kv, layout_sparse_indices)


@torch.library.register_kernel("npu_ops_transformer::npu_quant_block_sparse_attention_metadata", None)
def npu_quant_block_sparse_attention_metadata_fallback(
    sparse_seq_len: torch.Tensor, num_heads_q: int, num_heads_kv: int, head_dim: int, 
    cu_seqlens_q: Optional[torch.Tensor] = None, cu_seqlens_kv: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None, seqused_kv: Optional[torch.Tensor] = None, batch_size: Optional[int] = None,
    sparse_block_size_q: Optional[int] = None, sparse_block_size_kv: Optional[int] = None,
    quant_mode: Optional[int] = None, mask_mode: Optional[int] = None, max_seqlen_q: Optional[int] = None,
    max_seqlen_kv: Optional[int] = None, layout_q: Optional[str] = None, layout_kv: Optional[str] = None,
    layout_sparse_indices: Optional[str] = None):
    return npu_quant_block_sparse_attention_metadata(
        sparse_seq_len, num_heads_q, num_heads_kv, head_dim, 
        cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, batch_size,  sparse_block_size_q, sparse_block_size_kv, quant_mode, mask_mode, max_seqlen_q, max_seqlen_kv,
        layout_q, layout_kv, layout_sparse_indices)


torch.compiler.allow_in_graph(npu_quant_block_sparse_attention_metadata)