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


class QuantLightningIndexerV2OpBuilder(OpBuilder):
    def __init__(self):
        super(QuantLightningIndexerV2OpBuilder, self).__init__("npu_quant_lightning_indexer_v2")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/quant_lightning_indexer_v2.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return "npu_quant_lightning_indexer_v2(Tensor query, Tensor key, Tensor weights, Tensor query_dequant_scale, " \
            "Tensor key_dequant_scale, int query_quant_mode, int key_quant_mode, *, Tensor? actual_seq_lengths_query=None, " \
            "Tensor? actual_seq_lengths_key=None, Tensor? block_table=None, Tensor? metadata=None, str layout_query=\"BSND\", " \
            "str layout_key=\"BSND\", int sparse_count=2048, int sparse_mode=3, int pre_tokens=9223372036854775807, int next_tokens=9223372036854775807, " \
            "int? query_dtype=None, int? key_dtype=None) -> Tensor"

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_quant_lightning_indexer_v2_meta(query, key, weights, query_dequant_scale, key_dequant_scale, 
                                                query_quant_mode, key_quant_mode, *, actual_seq_lengths_query=None,
                                                actual_seq_lengths_key=None, block_table=None, metadata=None, 
                                                layout_query="BSND", layout_key="BSND", sparse_count=2048, sparse_mode=3,
                                                pre_tokens=9223372036854775807, next_tokens=9223372036854775807, query_dtype=None, key_dtype=None):
            keyHeadNum = key.shape[1] if layout_key == "TND" else key.shape[2]

            if layout_query == "BSND":
                output_size = (query.shape[0], query.shape[1], keyHeadNum, sparse_count)
            else:
                output_size = (query.shape[0], keyHeadNum, sparse_count)
            return torch.empty((output_size), dtype=torch.int32, device="meta")

# Instantiate the builder
npu_quant_lightning_indexer_v2_op_builder = QuantLightningIndexerV2OpBuilder()
op_module = npu_quant_lightning_indexer_v2_op_builder.load()  # Compiles/loads the .so file


@impl(AS_LIBRARY, npu_quant_lightning_indexer_v2_op_builder.name, "PrivateUse1")
def npu_quant_lightning_indexer_v2(query, key, weights, query_dequant_scale, key_dequant_scale, 
                                   query_quant_mode, key_quant_mode, *, actual_seq_lengths_query=None,
                                   actual_seq_lengths_key=None, block_table=None, metadata=None, 
                                   layout_query="BSND", layout_key="BSND", sparse_count=2048, sparse_mode=3,
                                   pre_tokens=9223372036854775807, next_tokens=9223372036854775807, query_dtype=None, key_dtype=None):
    """
    dispatcher implementation for NPU.zhe
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    return op_module.npu_quant_lightning_indexer_v2(query, key, weights, query_dequant_scale, key_dequant_scale, 
                                                    query_quant_mode, key_quant_mode, actual_seq_lengths_query,
                                                    actual_seq_lengths_key, block_table, metadata, 
                                                    layout_query, layout_key, sparse_count, sparse_mode,
                                                    pre_tokens, next_tokens, query_dtype, key_dtype)