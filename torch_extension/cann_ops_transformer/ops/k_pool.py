# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from enum import IntEnum
from typing import Optional
import torch
from torch.library import impl
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY


class CacheMode(IntEnum):
    LINEAR_BUFFER = 1
    RING_BUFFER = 2


class KPoolOpBuilder(OpBuilder):
    def __init__(self):
        super(KPoolOpBuilder, self).__init__("k_pool")

    def sources(self):
        """Path to C++ source code."""
        return ["ops/csrc/k_pool.cpp"]

    def schema(self):
        """PyTorch operator signature."""
        return (
            "k_pool(Tensor hidden_states, Tensor wk, Tensor gate_weight, Tensor(a!) state_cache, "
            "Tensor ape, "
            "int cmp_ratio=4, *, "
            "Tensor? cache_block_table=None, Tensor? seq_lens=None, "
            "Tensor? seq_used=None, Tensor? start_pos=None, "
            "int coff=1, int cache_mode=1) -> Tensor"
        )

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @impl(AS_LIBRARY, self.name, "Meta")
        def k_pool_meta(
            hidden_states,
            wk,
            gate_weight,
            state_cache,
            ape,
            cmp_ratio=4,
            *,
            cache_block_table=None,
            seq_lens=None,
            seq_used=None,
            start_pos=None,
            coff=1,
            cache_mode=CacheMode.LINEAR_BUFFER,
        ):
            d = wk.size(0) // coff
            if hidden_states.dim() == 3:
                b = hidden_states.size(0)
                s = hidden_states.size(1)
                sr = (s + cmp_ratio - 1) // cmp_ratio
                pooled_key_size = (b, sr, d)
            else:
                t = hidden_states.size(0)
                sr = (t + cmp_ratio - 1) // cmp_ratio
                pooled_key_size = (sr, d)

            return torch.empty(pooled_key_size, dtype=hidden_states.dtype, device="meta")


k_pool_op_builder = KPoolOpBuilder()


@impl(AS_LIBRARY, k_pool_op_builder.name, "PrivateUse1")
def k_pool(
    hidden_states: torch.Tensor,
    wk: torch.Tensor,
    gate_weight: torch.Tensor,
    state_cache: torch.Tensor,
    ape: torch.Tensor,
    cmp_ratio: int = 4,
    *,
    cache_block_table: Optional[torch.Tensor] = None,
    seq_lens: Optional[torch.Tensor] = None,
    seq_used: Optional[torch.Tensor] = None,
    start_pos: Optional[torch.Tensor] = None,
    coff: Optional[int] = 1,
    cache_mode: Optional[CacheMode] = CacheMode.LINEAR_BUFFER,
) -> torch.tensor:
    """
    dispatcher implementation for NPU.
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    op_module = k_pool_op_builder.load()
    return op_module.k_pool(
        hidden_states,
        wk,
        gate_weight,
        state_cache,
        ape,
        cmp_ratio,
        cache_block_table,
        seq_lens,
        seq_used,
        start_pos,
        coff,
        cache_mode,
    )


k_pool.CacheMode = CacheMode
