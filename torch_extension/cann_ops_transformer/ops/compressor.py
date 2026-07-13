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


class CompressorOpBuilder(OpBuilder):
    def __init__(self):
        super(CompressorOpBuilder, self).__init__("compressor")

    def sources(self):
        """Path to C++ source code."""
        return ["ops/csrc/compressor.cpp"]

    def schema(self):
        """PyTorch operator signature."""
        return (
            "compressor(Tensor x, Tensor wkv, Tensor wgate, Tensor(a!) state_cache, "
            "Tensor ape, "
            "int cmp_ratio=4, *, "
            "Tensor? state_block_table=None, Tensor? cu_seqlens=None, "
            "Tensor? seqused=None, Tensor? start_pos=None, "
            "int coff=1, int cache_mode=1) -> Tensor"
        )

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @impl(AS_LIBRARY, self.name, "Meta")
        def compressor_meta(
            x,
            wkv,
            wgate,
            state_cache,
            ape,
            cmp_ratio=4,
            *,
            state_block_table=None,
            cu_seqlens=None,
            seqused=None,
            start_pos=None,
            coff=1,
            cache_mode=1,
        ):
            if x.dim() == 3:
                b = x.size(0)
                s = x.size(1)
                h = x.size(2)
                sr = (s + cmp_ratio - 1) // cmp_ratio
                cmp_kv_size = (b, sr, h)
            else:
                t = x.size(0)
                h = x.size(1)
                sr = (t + cmp_ratio - 1) // cmp_ratio
                cmp_kv_size = (sr, h)

            return torch.empty(cmp_kv_size, dtype=x.dtype, device="meta")


compressor_op_builder = CompressorOpBuilder()


@impl(AS_LIBRARY, compressor_op_builder.name, "PrivateUse1")
def compressor(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    state_cache: torch.Tensor,
    ape: torch.Tensor,
    cmp_ratio: int = 4,
    *,
    state_block_table: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    start_pos: Optional[torch.Tensor] = None,
    coff: Optional[int] = 1,
    cache_mode: Optional[int] = 1,
) -> torch.tensor:
    """
    dispatcher implementation for NPU.
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    op_module = compressor_op_builder.load()
    return op_module.compressor(
        x,
        wkv,
        wgate,
        state_cache,
        ape,
        cmp_ratio,
        state_block_table,
        cu_seqlens,
        seqused,
        start_pos,
        coff,
        cache_mode,
    )
