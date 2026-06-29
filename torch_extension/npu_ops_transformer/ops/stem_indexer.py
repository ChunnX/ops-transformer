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
from typing import Optional

from npu_ops_transformer.op_builder.builder import AS_LIBRARY
from npu_ops_transformer.op_builder.builder import OpBuilder


STEM_INDEXER_METADATA_OP_NAME = "npu_stem_indexer_metadata"


class StemIndexerOpBuilder(OpBuilder):
    def __init__(self):
        super(StemIndexerOpBuilder, self).__init__("npu_stem_indexer")

    def sources(self):
        """Path to C++ source code."""
        return ["ops/csrc/stem_indexer.cpp"]

    def schema(self):
        """PyTorch operator signature."""
        return [
            (
                "npu_stem_indexer_metadata("
                "Tensor q_seq_lens, Tensor kv_seq_lens, int q_heads, int kv_heads, *, "
                "bool causal=True, int stem_block_size=128, int dim_qkflat=128, "
                "int window_size=4) -> Tensor"
            ),
            (
                "npu_stem_indexer("
                "Tensor qflat, Tensor kflat, Tensor vbias, Tensor q_seq_lens, Tensor kv_seq_lens, "
                "Tensor num_prompt_tokens, Tensor metadata, *, "
                "bool causal=True, int stem_block_size=128, int stem_stride=16, float alpha=1.0, "
                "int initial_blocks=4, int window_size=4, float k_block_num_rate_medium=0.2, "
                "int k_block_num_bias_medium=30, float k_block_num_rate_large=0.1, "
                "int k_block_num_bias_large=30) -> (Tensor, Tensor)"
            ),
        ]

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @impl(AS_LIBRARY, STEM_INDEXER_METADATA_OP_NAME, "Meta")
        def npu_stem_indexer_metadata_meta(
            q_seq_lens,
            kv_seq_lens,
            q_heads,
            kv_heads,
            *,
            causal=True,
            stem_block_size=128,
            dim_qkflat=128,
            window_size=4,
        ):
            return torch.empty((2048,), dtype=torch.int32, device="meta")

        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_stem_indexer_meta(
            qflat,
            kflat,
            vbias,
            q_seq_lens,
            kv_seq_lens,
            num_prompt_tokens,
            metadata,
            *,
            causal=True,
            stem_block_size=128,
            stem_stride=16,
            alpha=1.0,
            initial_blocks=4,
            window_size=4,
            k_block_num_rate_medium=0.2,
            k_block_num_bias_medium=30,
            k_block_num_rate_large=0.1,
            k_block_num_bias_large=30,
        ):
            batch_size = qflat.size(0)
            num_q_heads = qflat.size(1)
            q_block_num = qflat.size(2)
            k_block_num = kflat.size(2)
            sparse_indices = torch.empty(
                (batch_size, num_q_heads, q_block_num, k_block_num),
                dtype=torch.int32,
                device="meta",
            )
            sparse_seq_len = torch.empty(
                (batch_size, num_q_heads, q_block_num),
                dtype=torch.int32,
                device="meta",
            )
            return sparse_indices, sparse_seq_len


stem_indexer_op_builder = StemIndexerOpBuilder()
op_module = stem_indexer_op_builder.load()


@impl(AS_LIBRARY, STEM_INDEXER_METADATA_OP_NAME, "PrivateUse1")
def npu_stem_indexer_metadata(
    q_seq_lens,
    kv_seq_lens,
    q_heads,
    kv_heads,
    *,
    causal: Optional[bool] = True,
    stem_block_size: Optional[int] = 128,
    dim_qkflat: Optional[int] = 128,
    window_size: Optional[int] = 4,
):
    """
    Dispatcher implementation for NPU.
    'PrivateUse1' is the dispatch key for custom NPU backends.
    """
    output = torch.empty((2048,), dtype=torch.int32, device="npu")
    return op_module.npu_stem_indexer_metadata(
        q_seq_lens,
        kv_seq_lens,
        q_heads,
        kv_heads,
        causal,
        stem_block_size,
        dim_qkflat,
        window_size,
        output,
    )


@impl(AS_LIBRARY, stem_indexer_op_builder.name, "PrivateUse1")
def npu_stem_indexer(
    qflat,
    kflat,
    vbias,
    q_seq_lens,
    kv_seq_lens,
    num_prompt_tokens,
    metadata,
    *,
    causal=True,
    stem_block_size=128,
    stem_stride=16,
    alpha=1.0,
    initial_blocks=4,
    window_size=4,
    k_block_num_rate_medium=0.2,
    k_block_num_bias_medium=30,
    k_block_num_rate_large=0.1,
    k_block_num_bias_large=30,
):
    """
    Dispatcher implementation for NPU.
    'PrivateUse1' is the dispatch key for custom NPU backends.
    """
    return op_module.npu_stem_indexer(
        qflat,
        kflat,
        vbias,
        q_seq_lens,
        kv_seq_lens,
        num_prompt_tokens,
        metadata,
        causal,
        stem_block_size,
        stem_stride,
        alpha,
        initial_blocks,
        window_size,
        k_block_num_rate_medium,
        k_block_num_bias_medium,
        k_block_num_rate_large,
        k_block_num_bias_large,
    )
