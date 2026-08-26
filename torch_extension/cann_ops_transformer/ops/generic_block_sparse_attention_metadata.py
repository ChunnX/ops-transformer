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

from ..op_builder import OpBuilder, get_as_library


_METADATA_SIZE = 1024
_OP_NAME = "npu_generic_block_sparse_attention_metadata"


class GenericBlockSparseAttentionMetadataOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__(_OP_NAME)

    def sources(self):
        return ["ops/csrc/generic_block_sparse_attention_metadata.cpp"]

    def schema(self):
        return (
            "npu_generic_block_sparse_attention_metadata("
            "Tensor sparse_block_idx, Tensor sparse_block_count, "
            "int max_q_seq_len, int max_kv_seq_len, int num_q_heads, int num_kv_heads, int head_dim, "
            "int[] block_shape, *, Tensor? cu_seq_lengths=None, Tensor? cu_seq_lengths_kv=None, "
            "Tensor? seq_used_q=None, Tensor? seq_used_kv=None, int is_packed_gqa=1, "
            "str q_input_layout=\"BSND\", str kv_input_layout=\"BSND\", "
            "int mask_type=0, int quant_type=0, int softmax_precision=0, "
            "int window_size_left=-1, int window_size_right=-1) -> Tensor"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def npu_generic_block_sparse_attention_metadata_meta(
            sparse_block_idx,
            sparse_block_count,
            max_q_seq_len,
            max_kv_seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            block_shape,
            *,
            cu_seq_lengths=None,
            cu_seq_lengths_kv=None,
            seq_used_q=None,
            seq_used_kv=None,
            is_packed_gqa=1,
            q_input_layout="BSND",
            kv_input_layout="BSND",
            mask_type=0,
            quant_type=0,
            softmax_precision=0,
            window_size_left=-1,
            window_size_right=-1,
        ):
            return sparse_block_idx.new_empty((_METADATA_SIZE,), dtype=torch.int32)


if getattr(torch_npu, "_ops_transformer_generic_block_sparse_attention_metadata_registered", False):
    npu_generic_block_sparse_attention_metadata = torch_npu.npu_generic_block_sparse_attention_metadata
else:
    generic_block_sparse_attention_metadata_op_builder = GenericBlockSparseAttentionMetadataOpBuilder()
    generic_block_sparse_attention_metadata_op_builder._ensure_initialized()

    @impl(get_as_library(), _OP_NAME, "PrivateUse1")
    def npu_generic_block_sparse_attention_metadata(
        sparse_block_idx,
        sparse_block_count,
        max_q_seq_len,
        max_kv_seq_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        block_shape,
        *,
        cu_seq_lengths=None,
        cu_seq_lengths_kv=None,
        seq_used_q=None,
        seq_used_kv=None,
        is_packed_gqa=1,
        q_input_layout="BSND",
        kv_input_layout="BSND",
        mask_type=0,
        quant_type=0,
        softmax_precision=0,
        window_size_left=-1,
        window_size_right=-1,
    ):
        op_module = generic_block_sparse_attention_metadata_op_builder.load()
        return op_module.npu_generic_block_sparse_attention_metadata(
            sparse_block_idx,
            sparse_block_count,
            cu_seq_lengths,
            cu_seq_lengths_kv,
            seq_used_q,
            seq_used_kv,
            max_q_seq_len,
            max_kv_seq_len,
            num_q_heads,
            num_kv_heads,
            head_dim,
            block_shape,
            is_packed_gqa,
            q_input_layout,
            kv_input_layout,
            mask_type,
            quant_type,
            softmax_precision,
            window_size_left,
            window_size_right,
        )

    torch_npu.npu_generic_block_sparse_attention_metadata = npu_generic_block_sparse_attention_metadata
    torch_npu._ops_transformer_generic_block_sparse_attention_metadata_registered = True


generic_block_sparse_attention_metadata = npu_generic_block_sparse_attention_metadata
torch.compiler.allow_in_graph(npu_generic_block_sparse_attention_metadata)
