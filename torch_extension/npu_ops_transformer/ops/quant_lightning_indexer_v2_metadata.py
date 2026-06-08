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


class QuantLightningIndexerV2MetadataOpBuilder(OpBuilder):
    def __init__(self):
        super(QuantLightningIndexerV2MetadataOpBuilder, self).__init__("npu_quant_lightning_indexer_v2_metadata")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/quant_lightning_indexer_v2_metadata.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return "npu_quant_lightning_indexer_v2_metadata(int num_heads_q, int num_heads_k, int head_dim, int topk, " \
            "int q_quant_mode, int k_quant_mode, *, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None, "         \
            "Tensor? seqused_q=None, Tensor? seqused_k=None, Tensor? cmp_residual_k=None, int batch_size=0, "       \
            "int max_seqlen_q=0, int max_seqlen_k=0, str layout_q='BSND', str layout_k='BSND', int mask_mode=0, "   \
            "int cmp_ratio=1) -> Tensor"

    def register_meta(self):
        """
        Registers Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @torch.library.register_fake("npu_ops_transformer::" + self.name)
        def npu_quant_lightning_indexer_v2_metadata_meta(
            num_heads_q: int, num_heads_k: int, head_dim: int, topk: int, q_quant_mode: int, k_quant_mode: int,
            cu_seqlens_q: Optional[Tensor] = None, cu_seqlens_k: Optional[Tensor] = None,
            seqused_q: Optional[Tensor] = None, seqused_k: Optional[Tensor] = None,
            cmp_residual_k: Optional[Tensor] = None,
            batch_size: Optional[int] = None, max_seqlen_q: Optional[int] = None, max_seqlen_k: Optional[int] = None,
            layout_q: Optional[str] = None, layout_k: Optional[str] = None, mask_mode: Optional[int] = None,
            cmp_ratio: Optional[int] = None):
            return torch.empty((1024), dtype=torch.int32, device="npu")

# Instantiate the builder
quant_lightning_indexer_v2_metadata_op_builder = QuantLightningIndexerV2MetadataOpBuilder()
op_module = quant_lightning_indexer_v2_metadata_op_builder.load()


@impl(AS_LIBRARY, quant_lightning_indexer_v2_metadata_op_builder.name, "PrivateUse1")
def npu_quant_lightning_indexer_v2_metadata(
    num_heads_q: int, num_heads_k: int, head_dim: int, topk: int, q_quant_mode: int, k_quant_mode: int,
    cu_seqlens_q: Optional[Tensor] = None, cu_seqlens_k: Optional[Tensor] = None, seqused_q: Optional[Tensor] = None,
    seqused_k: Optional[Tensor] = None, cmp_residual_k: Optional[Tensor] = None,
    batch_size: Optional[int] = None, max_seqlen_q: Optional[int] = None, max_seqlen_k: Optional[int] = None,
    layout_q: Optional[str] = None, layout_k: Optional[str] = None, mask_mode: Optional[int] = None,
    cmp_ratio: Optional[int] = None):
    """
    Dispatcher implementation: NPU.
    'PrivateUse1' is dispatch key for custom NPU backends.
    """
    batch_size = 0 if batch_size is None else batch_size
    max_seqlen_q = 0 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_k = 0 if max_seqlen_k is None else max_seqlen_k
    layout_q = "BSND" if layout_q is None else layout_q
    layout_k = "BSND" if layout_k is None else layout_k
    mask_mode = 0 if mask_mode is None else mask_mode
    cmp_ratio = 1 if cmp_ratio is None else cmp_ratio
 
    return op_module.npu_quant_lightning_indexer_v2_metadata(
        num_heads_q, num_heads_k, head_dim, topk, q_quant_mode, k_quant_mode,
        cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k,
        batch_size, max_seqlen_q, max_seqlen_k, layout_q, layout_k, mask_mode, cmp_ratio)


@torch.library.register_kernel("npu_ops_transformer::npu_quant_lightning_indexer_v2_metadata", None)
def npu_quant_lightning_indexer_v2_metadata_fallback(
    num_heads_q: int, num_heads_k: int, head_dim: int, topk: int, q_quant_mode: int, k_quant_mode: int,
    cu_seqlens_q: Optional[Tensor] = None, cu_seqlens_k: Optional[Tensor] = None, seqused_q: Optional[Tensor] = None,
    seqused_k: Optional[Tensor] = None, cmp_residual_k: Optional[Tensor] = None,
    batch_size: Optional[int] = None, max_seqlen_q: Optional[int] = None, max_seqlen_k: Optional[int] = None,
    layout_q: Optional[str] = None, layout_k: Optional[str] = None, mask_mode: Optional[int] = None,
    cmp_ratio: Optional[int] = None):
    # 处理所有 tensor 都为 None 的情况
    # 可以在这里创建一个 NPU tensor 来触发 PrivateUse1 backend
    if all(t is None for t in [cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k]):
        _ = torch.empty(1, dtype=torch.int32, device="npu")
    # 调用 NPU 实现
    return npu_quant_lightning_indexer_v2_metadata(
        num_heads_q, num_heads_k, head_dim, topk, q_quant_mode, k_quant_mode,
        cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, cmp_residual_k,
        batch_size, max_seqlen_q, max_seqlen_k, layout_q, layout_k, mask_mode, cmp_ratio)
        
torch.compiler.allow_in_graph(npu_quant_lightning_indexer_v2_metadata)