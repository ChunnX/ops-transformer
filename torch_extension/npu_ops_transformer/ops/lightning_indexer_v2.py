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


class LightningIndexerV2OpBuilder(OpBuilder):
    def __init__(self):
        super(LightningIndexerV2OpBuilder, self).__init__("npu_lightning_indexer_v2")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/lightning_indexer_v2.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return "npu_lightning_indexer_v2(Tensor q, Tensor k, Tensor w, " \
            "int topk, *, Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_k=None," \
            "Tensor? seqused_q=None, Tensor? seqused_k=None, " \
            "Tensor? cmp_residual_k=None, Tensor? block_table=None, " \
            "Tensor? output_idx_offset=None, Tensor? metadata=None, int max_seqlen_q=-1," \
            "str layout_q=\"BSND\", str layout_k=\"BSND\", int mask_mode=0, " \
            "int cmp_ratio=1," \
            "int return_value=0) -> (Tensor, Tensor)"

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_lightning_indexer_v2_meta(q, k, w, topk, *, cu_seqlens_q=None, cu_seqlens_k=None,
                                          seqused_q=None, seqused_k=None, cmp_residual_k=None, block_table=None,
                                          output_idx_offset=None, metadata=None, max_seqlen_q=-1,
                                          layout_q="BSND", layout_k="BSND", mask_mode=0,
                                          cmp_ratio=1, return_value=0):
            key_head_num = k.shape[1] if layout_k == "TND" else k.shape[2]

            if layout_q == "BSND":
                sparse_indices_out = torch.empty([q.shape[0], q.shape[1], key_head_num, topk],
                    dtype=torch.int32, device="meta")
            else:
                sparse_indices_out = torch.empty([q.shape[0], key_head_num, topk], dtype=torch.int32, device="meta")
            if return_value:
                if layout_q == "BSND":
                    sparse_values_out = torch.empty([q.shape[0], q.shape[1], key_head_num, topk],
                        dtype=torch.float, device="meta")
                else:
                    sparse_values_out = torch.empty([q.shape[0], key_head_num, topk],
                        dtype=torch.float, device="meta")
            else:
                sparse_values_out = torch.empty([0], dtype=torch.float, device="meta")
            return (sparse_indices_out, sparse_values_out)

# Instantiate the builder
npu_lightning_indexer_v2_op_builder = LightningIndexerV2OpBuilder()
op_module = npu_lightning_indexer_v2_op_builder.load()  # Compiles/loads the .so file


@impl(AS_LIBRARY, npu_lightning_indexer_v2_op_builder.name, "PrivateUse1")
def npu_lightning_indexer_v2(q, k, w, topk, *, cu_seqlens_q=None, cu_seqlens_k=None,
                            seqused_q=None, seqused_k=None, cmp_residual_k=None, block_table=None, 
                            output_idx_offset=None, metadata=None, max_seqlen_q=-1,
                            layout_q="BSND", layout_k="BSND", mask_mode=0, 
                            cmp_ratio=1, return_value=0):
    """
    dispatcher implementation for NPU.zhe
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    return op_module.npu_lightning_indexer_v2(q, k, w, topk, cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, 
                                              cmp_residual_k, block_table, output_idx_offset, metadata, max_seqlen_q, 
                                              layout_q, layout_k, mask_mode, cmp_ratio, return_value)