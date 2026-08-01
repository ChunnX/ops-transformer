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


class CompressorGradOpBuilder(OpBuilder):
    def __init__(self):
        super(CompressorGradOpBuilder, self).__init__("compressor_grad")

    def sources(self):
        """Path to C++ source code."""
        return ["ops/csrc/compressor_grad.cpp"]

    def schema(self):
        """PyTorch operator signature."""
        return (
            "compressor_grad(Tensor x, Tensor wkv, Tensor wgate, Tensor d_cmp_kv, "
            "Tensor softmax_score, Tensor kv, int cmp_ratio=1, *, "
            "int coff=1, "
            "Tensor? cu_seqlens=None, "
            "Tensor? seqused=None, Tensor? start_pos=None"
            ") -> (Tensor, Tensor, Tensor, Tensor)",
        )

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @impl(AS_LIBRARY, self.name, "Meta")
        def compressor_grad_meta(
            x,
            wkv,
            wgate,
            d_cmp_kv,
            softmax_score,
            kv,
            cmp_ratio,
            *,
            coff=1,
            cu_seqlens=None,
            seqused=None,
            start_pos=None,
            
        ):
            if x.dim() == 3:
                b = x.size(0)
                s = x.size(1)
                h = x.size(2)
                coff_d = wkv.size(0)
                d_x_size = (b, s, h)
                d_wkv_size = (coff_d, h)
                d_wgate_size = (coff_d, h)
                d_ape_size = (cmp_ratio, coff_d)
            else:
                t = x.size(0)
                h = x.size(1)
                coff_d = wkv.size(0)
                d_x_size = (t, h)
                d_wkv_size = (coff_d, h)
                d_wgate_size = (coff_d, h)
                d_ape_size = (cmp_ratio, coff_d)
            
            d_x = torch.empty(d_x_size, dtype=x.dtype, device="meta")
            d_wkv = torch.empty(d_wkv_size, dtype=x.dtype, device="meta")
            d_wgate = torch.empty(d_wgate_size, dtype=x.dtype, device="meta")
            d_ape = torch.empty(d_ape_size, dtype=softmax_score.dtype, device="meta")

            return (d_x, d_wkv, d_wgate, d_ape)

compressor_grad_op_builder = CompressorGradOpBuilder()


@impl(AS_LIBRARY, compressor_grad_op_builder.name, "PrivateUse1")
def compressor_grad(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    d_cmp_kv: torch.Tensor,
    softmax_score: torch.Tensor,
    kv: torch.Tensor,
    cmp_ratio: int = 4,
    *,
    coff: Optional[int] = 1,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    start_pos: Optional[torch.Tensor] = None,
) -> torch.tensor:
    """
    dispatcher implementation for NPU.
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    op_module = compressor_grad_op_builder.load()
    return op_module.compressor_grad(
        x,
        wkv,
        wgate,
        d_cmp_kv,
        softmax_score,
        kv,
        cmp_ratio,
        coff,
        cu_seqlens,
        seqused,
        start_pos,
    )
