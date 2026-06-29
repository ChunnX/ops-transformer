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


class StemIndexerMetadataOpBuilder(OpBuilder):
    def __init__(self):
        super(StemIndexerMetadataOpBuilder, self).__init__("npu_stem_indexer_metadata")

    def sources(self):
        """Path to C++ source code."""
        return ["ops/csrc/stem_indexer_metadata.cpp"]

    def schema(self) -> str:
        """PyTorch operator signature."""
        return (
            "npu_stem_indexer_metadata("                                                                    \
            "Tensor q_seq_lens, Tensor kv_seq_lens, int q_heads, int kv_heads, *,"                          \
            "bool causal=True, int stem_block_size=128, int dim_qkflat=128, int window_size=4) -> Tensor"
        )

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @torch.library.register_fake("npu_ops_transformer::" + self.name)
        def npu_stem_indexer_meta(
            q_seq_lens, kv_seq_lens, q_heads, kv_heads, *,
            causal: Optional[bool] = True, stem_block_size: Optional[int] = 128, dim_qkflat: Optional[int] = 128,
            window_size: Optional[int] = 4
        ):
            return torch.empty((2048,), dtype=torch.int32, device="npu")

stem_indexer_metadata_op_builder = StemIndexerMetadataOpBuilder()
op_module = stem_indexer_metadata_op_builder.load()


@impl(AS_LIBRARY, stem_indexer_metadata_op_builder.name, "PrivateUse1")
def npu_stem_indexer_metadata(
    q_seq_lens, kv_seq_lens, q_heads, kv_heads,
    causal: Optional[bool] = True, stem_block_size: Optional[int] = 128, dim_qkflat: Optional[int] = 128,
    window_size: Optional[int] = 4
):
    """
    Dispatcher implementation for NPU.
    'PrivateUse1' is the dispatch key for custom NPU backends.
    """
    output = torch.empty((2048,), dtype=torch.int32, device="npu")
    return op_module.npu_stem_indexer_metadata(
        q_seq_lens, kv_seq_lens, q_heads, kv_heads,
        causal, stem_block_size, dim_qkflat, window_size, output)