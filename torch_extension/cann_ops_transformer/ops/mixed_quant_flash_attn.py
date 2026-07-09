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
import torch_npu
from enum import IntEnum
from torch.library import impl
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY

class _MaskMode(IntEnum):
    ALL = 0
    CAUSAL = 3
    WINDOW = 4

class _QuantMode(IntEnum):
    A16C4_MXFP4 = 1

class MixedQuantFlashAttenOpBuilder(OpBuilder):
    def __init__(self):
        super(MixedQuantFlashAttenOpBuilder, self).__init__("mixed_quant_flash_attn")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/mixed_quant_flash_attn.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return "mixed_quant_flash_attn(Tensor q, Tensor k, Tensor v, Tensor k_descale, Tensor v_descale," \
            "Tensor?block_table=None, Tensor?cu_seqlens_q=None," \
            "Tensor?seqused_q=None," \
            "Tensor?seqused_kv=None, Tensor?sinks=None, Tensor?attn_mask=None, Tensor?metadata=None," \
            "int quant_mode=1, float softmax_scale=1.0, int mask_mode=0, int win_left=-1, int win_right=-1," \
            "int max_seqlen_q=-1, int max_seqlen_kv=-1," \
            "str layout_q=\"BSND\", str layout_kv=\"PA_BBND\", str layout_attn_out=\"BSND\", " \
            "bool return_softmax_lse=False)->(Tensor, Tensor)"

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @impl(AS_LIBRARY, self.name, "Meta")
        def mixed_quant_flash_attn_meta(
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, k_descale: torch.Tensor, v_descale: torch.Tensor, 
            block_table: Optional[torch.Tensor] = None, 
            cu_seqlens_q: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None, seqused_kv: Optional[torch.Tensor] = None, 
            sinks: Optional[torch.Tensor] = None, 
            attn_mask: Optional[torch.Tensor] = None, 
            metadata: Optional[torch.Tensor] = None,
            quant_mode: _QuantMode = _QuantMode.A16C4_MXFP4, 
            softmax_scale: Optional[float] = 1.0, 
            mask_mode: Optional[_MaskMode] = _MaskMode.ALL, win_left: Optional[int] = -1, win_right: Optional[int] = -1,
            max_seqlen_q: Optional[int] = -1, max_seqlen_kv: Optional[int] = -1,
            layout_q: Optional[str] = "BSND", layout_kv: Optional[str] = "PA_BBND", layout_attn_out: Optional[str] = "BSND", 
            return_softmax_lse: Optional[bool] = False
        ):
            if layout_q == "TND":
                tSize = q.size(0)
                nSize = q.size(1)
                dSize = v.size(2)
                softmaxOutSize = (nSize, tSize)
            elif layout_q == "BSND":
                bSize = q.size(0)
                sSize = q.size(1)
                nSize = q.size(2)
                dSize = v.size(3)
                softmaxOutSize = (bSize, nSize, sSize)
            else:
                bSize = q.size(0)
                nSize = q.size(1)
                sSize = q.size(2)
                dSize = v.size(3)
                softmaxOutSize = (bSize, nSize, sSize)

            if layout_attn_out == "TND":
                torch._check(
                    layout_q == "TND",
                    lambda: "When the layout of output is TND, the layout of query  must be TND, but got " + str(layout_q),
                )
                attentionOutSize = (tSize, nSize, dSize)
            elif layout_attn_out == "BNSD":
                torch._check(
                    layout_q == "BNSD",
                    lambda: "When the layout of output is BNSD, the layout of query  must be BNSD, but got " + str(layout_q),
                )
                attentionOutSize = (bSize, nSize, sSize, dSize)
            else:
                torch._check(
                    layout_q != "TND",
                    lambda: "When the layout of output is BSND, the layout of query  must be BNSD or BSND, but got " + str(layout_q),
                )
                attentionOutSize = (bSize, sSize, nSize, dSize)

            
            return (
 	                 torch.empty(attentionOutSize, dtype=q.dtype, device='meta'),
 	                 torch.empty(softmaxOutSize, dtype=q.dtype, device='meta')
 	             )


# Instantiate the builder
mixed_quant_flash_attn_op_builder = MixedQuantFlashAttenOpBuilder()
op_module = mixed_quant_flash_attn_op_builder.load()  # Compiles/loads the .so file


@impl(AS_LIBRARY, mixed_quant_flash_attn_op_builder.name, "PrivateUse1")
def mixed_quant_flash_attn(
            q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, k_descale: torch.Tensor, v_descale: torch.Tensor, 
            block_table: Optional[torch.Tensor] = None, 
            cu_seqlens_q: Optional[torch.Tensor] = None,
            seqused_q: Optional[torch.Tensor] = None, seqused_kv: Optional[torch.Tensor] = None, 
            sinks: Optional[torch.Tensor] = None, 
            attn_mask: Optional[torch.Tensor] = None, 
            metadata: Optional[torch.Tensor] = None,
            quant_mode: _QuantMode = _QuantMode.A16C4_MXFP4, 
            softmax_scale: Optional[float] = 1.0, 
            mask_mode: Optional[_MaskMode] = _MaskMode.ALL, win_left: Optional[int] = -1, win_right: Optional[int] = -1,
            max_seqlen_q: Optional[int] = -1, max_seqlen_kv: Optional[int] = -1,
            layout_q: Optional[str] = "BSND", layout_kv: Optional[str] = "PA_BBND", layout_attn_out: Optional[str] = "BSND", 
            return_softmax_lse: Optional[bool] = False
):
    """
    dispatcher implementation for NPU.
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    quant_mode = quant_mode.value if isinstance(quant_mode, IntEnum) else quant_mode
    mask_mode = mask_mode.value if isinstance(mask_mode, IntEnum) else mask_mode
    print(f"   - Quant Mode: {quant_mode}")
    print(f"   - Mask Mode:  {mask_mode}")

    return op_module.mixed_quant_flash_attn(q, k, v, k_descale, v_descale, block_table, cu_seqlens_q,
                                    seqused_q,
                                    seqused_kv, sinks, attn_mask, metadata,
                                    quant_mode, softmax_scale, mask_mode, win_left, win_right,
                                    max_seqlen_q, max_seqlen_kv,
                                    layout_q, layout_kv, layout_attn_out,
                                    return_softmax_lse)

mixed_quant_flash_attn.MaskMode = _MaskMode
mixed_quant_flash_attn.QuantMode = _QuantMode