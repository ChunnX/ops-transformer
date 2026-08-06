# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional
import torch
from torch.library import impl
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY


SLIG__LOSS_SIZE = 1

class SparseLightningIndexersGradKLLosOpBuilder(OpBuilder):
    def __init__(self):
        super(SparseLightningIndexersGradKLLosOpBuilder, self).__init__(
            "sparse_lightning_indexer_grad_kl_loss"
        )

    def sources(self):
        return ["ops/csrc/sparse_lightning_indexer_grad_kl_loss.cpp"]

    def schema(self) -> str:
        return [
            "npu_sparse_lightning_indexer_grad_kl_loss(Tensor query, Tensor key, Tensor query_index, Tensor key_index,"
            "Tensor weights, Tensor sparse_indices, Tensor softmax_max, Tensor softmax_sum, float scale_value, *,"
            "Tensor? query_rope=None, Tensor? key_rope=None, SymInt[]? actual_seq_qlen=None,"
            "SymInt[]? actual_seq_klen=None str? layout='BSND', int? sparse_mode=3,"
            "int? pre_tokens=9223372036854775807, int? next_tokens=9223372036854775807, Tensor? sinks=None,)"
            "-> (Tensor, Tensor, Tensor, Tensor)",
        ]

    def register_meta(self):

        @impl(AS_LIBRARY, self.name, "Meta")
        def sparse_lightning_indexer_grad_kl_loss_meta(
            query,
            key,
            query_index,
            key_index,
            weights,
            sparse_indices,
            softmax_max,
            softmax_sum,
            scale_value,
            *,
            query_rope=None,
            key_rope=None,
            actual_seq_qlen=None,
            actual_seq_klen=None,
            layout='BSND',
            sparse_mode=3,
            pre_tokens=9223372036854775807,
            next_tokens=9223372036854775807,
            sinks=None,
        ):
        , , d_weights, 
            d_query_index = query_index.new_empty(query_index.shape, dtype=query_index.dtype, device="meta")
            d_key_index = key_index.new_empty(key_index.shape, dtype=key_index.dtype, device="meta")
            d_weights = weights.new_empty(weights.shape, dtype=weights.dtype, device="meta")
            loss = torch.empty({SLIG__LOSS_SIZE},  dtype=torch.float, device="meta")
            return d_query_index, d_key_index, d_weights, loss


sparse_lightning_indexer_grad_kl_loss_op_builder = (
    SparseLightningIndexersGradKLLosOpBuilder()
)

@impl(AS_LIBRARY, sparse_lightning_indexer_grad_kl_loss_op_builder.name, "PrivateUse1")
def sparse_lightning_indexer_grad_kl_loss(
    query,
    key,
    query_index,
    key_index,
    weights,
    sparse_indices,
    softmax_max,
    softmax_sum,
    scale_value,
    *,
    query_rope=None,
    key_rope=None,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout='BSND',
    sparse_mode=3,
    pre_tokens=9223372036854775807,
    next_tokens=9223372036854775807,
    sinks=None,
):
    op_module = sparse_lightning_indexer_grad_kl_loss_op_builder.load()
    return op_module.sparse_lightning_indexer_grad_kl_loss(
            query,
            key,
            query_index,
            key_index,
            weights,
            sparse_indices,
            softmax_max,
            softmax_sum,
            scale_value,
            query_rope,
            key_rope,
            actual_seq_qlen,
            actual_seq_klen,
            layout,
            sparse_mode,
            pre_tokens,
            next_tokens,
            sinks,
    )

sparse_lightning_indexer_grad_kl_loss = torch.ops.cann_ops_transformer.sparse_lightning_indexer_grad_kl_loss
