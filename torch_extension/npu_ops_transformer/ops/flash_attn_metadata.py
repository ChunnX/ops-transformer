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
from typing import Optional


class FlashAttnMetadataOpBuilder(OpBuilder):
    def __init__(self):
        super(FlashAttnMetadataOpBuilder, self).__init__("npu_flash_attn_metadata")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/flash_attn_metadata.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return "npu_flash_attn_metadata(Tensor? actual_seq_lengths_q, Tensor? actual_seq_lengths_kv, " \
            "int num_heads_q, int num_heads_kv, int head_dim, int sparse_mode, " \
            "*, int? batch_size=None, int? max_seqlen_q=None, int? max_seqlen_kv=None) -> Tensor"

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @impl(AS_LIBRARY, self.name, "Meta")
        def npu_flash_attn_metadata_meta(actual_seq_lengths_q, actual_seq_lengths_kv, 
                                         num_heads_q, num_heads_kv, head_dim, sparse_mode, 
                                         batch_size: Optional[int] = None, 
                                         max_seqlen_q: Optional[int] = None, 
                                         max_seqlen_kv: Optional[int] = None):
            return torch.empty((4096,), dtype=torch.int32, device="npu")

# Instantiate the builder
flash_attn_metadata_op_builder = FlashAttnMetadataOpBuilder()
op_module = flash_attn_metadata_op_builder.load()


@impl(AS_LIBRARY, flash_attn_metadata_op_builder.name, "PrivateUse1")
def npu_flash_attn_metadata(actual_seq_lengths_q, actual_seq_lengths_kv, 
                            num_heads_q, num_heads_kv, head_dim, sparse_mode, *, 
                            batch_size: Optional[int] = None, 
                            max_seqlen_q: Optional[int] = None, 
                            max_seqlen_kv: Optional[int] = None):
    """
    Dispatcher implementation for NPU.
    'PrivateUse1' is the dispatch key for custom NPU backends.
    """
    batch_size_val = 0 if batch_size is None else batch_size
    max_seqlen_q_val = 0 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_kv_val = 0 if max_seqlen_kv is None else max_seqlen_kv
    
    return op_module.npu_flash_attn_metadata(actual_seq_lengths_q, actual_seq_lengths_kv, 
                                              num_heads_q, num_heads_kv, head_dim, sparse_mode, 
                                              batch_size_val, max_seqlen_q_val, max_seqlen_kv_val)
