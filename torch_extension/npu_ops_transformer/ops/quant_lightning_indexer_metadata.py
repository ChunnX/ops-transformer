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


class QuantLightningIndexerMetadataOpBuilder(OpBuilder):
    def __init__(self):
        super(QuantLightningIndexerMetadataOpBuilder, self).__init__("npu_quant_lightning_indexer_metadata")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/quant_lightning_indexer_metadata.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return "npu_quant_lightning_indexer_metadata(int num_heads_q, int num_heads_k, int head_dim, " \
        "int query_quant_mode, int key_quant_mode, *, Tensor? actual_seq_lengths_query=None, " \
        "Tensor? actual_seq_lengths_key=None, int? batch_size=None, int? max_seqlen_q=None, int? max_seqlen_k=None, " \
        "str? layout_query=None, str? layout_key=None, int? sparse_count=None, int? sparse_mode=None, " \
        "int? pre_tokens=None, int? next_tokens=None, int? cmp_ratio=None) -> Tensor"

    def register_meta(self):
        """
        Registers Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @torch.library.register_fake("npu_ops_transformer::" + self.name)
        def npu_quant_lightning_indexer_metadata_meta(
            num_heads_q: int, num_heads_k: int, head_dim: int, query_quant_mode: int, key_quant_mode: int, 
            actual_seq_lengths_query: Optional[Tensor] = None, actual_seq_lengths_key: Optional[Tensor] = None,
            batch_size: Optional[int] = None, max_seqlen_q: Optional[int] = None, max_seqlen_k: Optional[int] = None, 
            layout_query: Optional[str] = None, layout_key: Optional[str] = None, sparse_count: Optional[int] = None, 
            sparse_mode: Optional[int] = None, pre_tokens: Optional[int] = None, next_tokens: Optional[int] = None, 
            cmp_ratio: Optional[int] = None):
            return torch.empty((1024), dtype=torch.int32, device="meta")

# Instantiate the builder
quant_lightning_indexer_metadata_op_builder = QuantLightningIndexerMetadataOpBuilder()
op_module = quant_lightning_indexer_metadata_op_builder.load()

@impl(AS_LIBRARY, quant_lightning_indexer_metadata_op_builder.name, "PrivateUse1")
def npu_quant_lightning_indexer_metadata(
    num_heads_q: int, num_heads_k: int, head_dim: int, query_quant_mode: int, key_quant_mode: int, 
    actual_seq_lengths_query: Optional[Tensor] = None, actual_seq_lengths_key: Optional[Tensor] = None,
    batch_size: Optional[int] = None, max_seqlen_q: Optional[int] = None, max_seqlen_k: Optional[int] = None, 
    layout_query: Optional[str] = None, layout_key: Optional[str] = None, sparse_count: Optional[int] = None, 
    sparse_mode: Optional[int] = None, pre_tokens: Optional[int] = None, next_tokens: Optional[int] = None, 
    cmp_ratio: Optional[int] = None):
    """
    Dispatcher implementation: NPU.
    'PrivateUse1' is dispatch key for custom NPU backends.
    """
    batch_size = 0 if batch_size is None else batch_size
    max_seqlen_q = -1 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_k = -1 if max_seqlen_k is None else max_seqlen_k
    layout_query = "BSND" if layout_query is None else layout_query
    layout_key = "BSND" if layout_key is None else layout_key
    sparse_count = 2048 if sparse_count is None else sparse_count
    sparse_mode = 3 if sparse_mode is None else sparse_mode
    pre_tokens = 9223372036854775807 if pre_tokens is None else pre_tokens
    next_tokens = 9223372036854775807 if next_tokens is None else next_tokens
    cmp_ratio = 1 if cmp_ratio is None else cmp_ratio
 
    return op_module.npu_quant_lightning_indexer_metadata(
        num_heads_q, num_heads_k, head_dim, query_quant_mode, key_quant_mode, actual_seq_lengths_query, 
        actual_seq_lengths_key, batch_size, max_seqlen_q, max_seqlen_k, layout_query, layout_key, sparse_count, 
        sparse_mode, pre_tokens, next_tokens, cmp_ratio)


@torch.library.register_kernel("npu_ops_transformer::npu_quant_lightning_indexer_metadata", None)
def npu_quant_lightning_indexer_metadata_fallback(
    num_heads_q: int, num_heads_k: int, head_dim: int, query_quant_mode: int, key_quant_mode: int, 
    actual_seq_lengths_query: Optional[Tensor] = None, actual_seq_lengths_key: Optional[Tensor] = None,
    batch_size: Optional[int] = None, max_seqlen_q: Optional[int] = None, max_seqlen_k: Optional[int] = None, 
    layout_query: Optional[str] = None, layout_key: Optional[str] = None, sparse_count: Optional[int] = None, 
    sparse_mode: Optional[int] = None, pre_tokens: Optional[int] = None, next_tokens: Optional[int] = None, 
    cmp_ratio: Optional[int] = None):
    # 处理所有 tensor 都为 None 的情况
    # 可以在这里创建一个 NPU tensor 来触发 PrivateUse1 backend
    if all(t is None for t in [actual_seq_lengths_query, actual_seq_lengths_key]):
             _ = torch.empty(1, dtype=torch.int32, device="npu")
    # 调用 NPU 实现
    return npu_quant_lightning_indexer_metadata(
        num_heads_q, num_heads_k, head_dim, query_quant_mode, key_quant_mode, actual_seq_lengths_query, 
        actual_seq_lengths_key, batch_size, max_seqlen_q, max_seqlen_k, layout_query, layout_key, sparse_count, 
        sparse_mode, pre_tokens, next_tokens, cmp_ratio)
        
torch.compiler.allow_in_graph(npu_quant_lightning_indexer_metadata)