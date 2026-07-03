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


class SparseFlashAttentionOpBuilder(OpBuilder):
    def __init__(self):
        super(SparseFlashAttentionOpBuilder, self).__init__("sparse_flash_attention")

    def sources(self):
        """Path to C++ source code."""
        return ["ops/csrc/sparse_flash_attention.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return "sparse_flash_attention(Tensor query, Tensor key, Tensor value, Tensor sparse_indices, " \
            "float scale_value, *, Tensor? block_table=None, Tensor? actual_seq_lengths_query=None, " \
            "Tensor? actual_seq_lengths_kv=None, Tensor? query_rope=None, Tensor? key_rope=None, " \
            "int sparse_block_size=1, str layout_query=\"BSND\", str layout_kv=\"BSND\", " \
            "int sparse_mode=3, int pre_tokens=9223372036854775807, " \
            "int next_tokens=9223372036854775807, int attention_mode=0, " \
            "bool return_softmax_lse=False, Tensor? sinks=None) -> (Tensor, Tensor, Tensor)"

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @impl(AS_LIBRARY, self.name, "Meta")
        def sparse_flash_attention_meta(query, key, value, sparse_indices, scale_value,
                                               block_table=None, actual_seq_lengths_query=None,
                                               actual_seq_lengths_kv=None, query_rope=None, key_rope=None,
                                               sparse_block_size=1, layout_query="BSND", layout_kv="BSND",
                                               sparse_mode=3,
                                               pre_tokens=9223372036854775807,
                                               next_tokens=9223372036854775807,
                                               attention_mode=0, return_softmax_lse=False, sinks=None):
            torch._check(
                layout_query in ("BSND", "TND"),
                lambda: "The layout of query only support BSND and TND, but got " + str(layout_query),
            )
            torch._check(query.numel() > 0, lambda: "Tensor query is empty.")
            torch._check(key.numel() > 0, lambda: "Tensor key is empty.")
            torch._check(sparse_indices.numel() > 0, lambda: "Tensor sparse_indices is empty.")

            if layout_query == "TND":
                torch._check(
                    query.dim() == 3,
                    lambda: "When the layout of query is TND, the query dimension must be 3, but got "
                    + str(query.dim()),
                )
                attention_out_size = (query.shape[0], query.shape[1], query.shape[2])
                if return_softmax_lse:
                    softmax_size = (key.shape[1], query.shape[0], query.shape[1] // key.shape[1])
                else:
                    softmax_size = (0,)
            else:
                torch._check(
                    query.dim() == 4,
                    lambda: "When the layout of query is BSND, the query dimension must be 4, but got "
                    + str(query.dim()),
                )
                attention_out_size = (query.shape[0], query.shape[1], query.shape[2], query.shape[3])
                if return_softmax_lse:
                    softmax_size = (query.shape[0], key.shape[2], query.shape[1], query.shape[2] // key.shape[2])
                else:
                    softmax_size = (0,)

            return (
                torch.empty(attention_out_size, dtype=query.dtype, device="meta"),
                torch.empty(softmax_size, dtype=torch.float32, device="meta"),
                torch.empty(softmax_size, dtype=torch.float32, device="meta"),
            )


sparse_flash_attention_op_builder = SparseFlashAttentionOpBuilder()
op_module = sparse_flash_attention_op_builder.load()


@impl(AS_LIBRARY, sparse_flash_attention_op_builder.name, "PrivateUse1")
def sparse_flash_attention(query, key, value, sparse_indices, scale_value,
                                  block_table=None, actual_seq_lengths_query=None,
                                  actual_seq_lengths_kv=None, query_rope=None, key_rope=None,
                                  sparse_block_size=1, layout_query="BSND", layout_kv="BSND",
                                  sparse_mode=3,
                                  pre_tokens=9223372036854775807,
                                  next_tokens=9223372036854775807,
                                  attention_mode=0, return_softmax_lse=False, sinks=None):
    """
    Dispatcher implementation for NPU.
    'PrivateUse1' is dispatch key for custom NPU backends.
    """
    return op_module.sparse_flash_attention(
        query, key, value, sparse_indices, scale_value,
        block_table, actual_seq_lengths_query, actual_seq_lengths_kv,
        query_rope, key_rope, sparse_block_size, layout_query,
        layout_kv, sparse_mode, pre_tokens, next_tokens, attention_mode,
        return_softmax_lse, sinks,
    )
