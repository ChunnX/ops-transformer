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


class QuantFlashAttenOpBuilder(OpBuilder):
    def __init__(self):
        super(QuantFlashAttenOpBuilder, self).__init__("npu_quant_flash_attn")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/quant_flash_attn.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        # TODO: 接口替换，基于新接口表格
        print("schema entrance ========================\n")
        return "npu_quant_flash_attn(Tensor q, Tensor k, Tensor v," \
            "Tensor q_descale, Tensor k_descale, Tensor v_descale," \
            "int q_quant_mode, int k_quant_mode, int v_quant_mode," \
            "Tensor?block_table=None, Tensor?cu_seqlens_q=None," \
            "Tensor?cu_seqlens_kv=None, Tensor?seqused_q=None," \
            "Tensor?seqused_kv=None, Tensor?sinks=None, Tensor?metadata=None," \
            "int q_dtype=None, int k_dtype=None, int v_dtype=None," \
            "int q_descale_dtype=None, int k_descale_dtype=None, int v_descale_dtype=None," \
            "int quant_block_size_qs=128, int quant_block_size_ks=256, int quant_block_size_vs=256," \
            "float softmax_scale=1.0, int mask_mode=1, int win_left=-1, int win_right=-1," \
            "int max_seqlen_q=-1, int max_seqlen_kv=-1," \
            "str layout_q=\"BSND\", str layout_kv=\"BSND\", str layout_out=\"BSND\"," \
            "int softmax_precision=0, int return_softmax_lse=0)->(Tensor, Tensor)"

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @impl(AS_LIBRARY, self.name, "Meta") #TODO：同步接口
        def npu_quant_flash_attn_meta(q, k, v, block_table=None, cu_seqlens_q=None,
                                cu_seqlens_kv=None, seqused_q=None,
                                seqused_kv=None, sinks=None, attn_mask=None, metadata=None,
                                softmax_scale=1.0, mask_mode=1, win_left=-1, win_right=-1,
                                max_seqlen_q=-1, max_seqlen_kv=-1,
                                layout_q="BSND", layout_kv="BSND", layout_out="BSND",
                                return_softmax_lse=0, deterministic=0):
            print("meta entrance ========================\n")
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

            if layout_out == "TND":
                torch._check(
                    layout_q == "TND",
                    lambda: "When the layout of output is TND, the layout of query  must be TND, but got " + str(layout_q),
                )
                attentionOutSize = (tSize, nSize, dSize)
            elif layout_out == "BNSD":
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

            
            return attentionOutSize, softmaxOutSize


# Instantiate the builder
quant_flash_attn_op_builder = QuantFlashAttenOpBuilder()
op_module = quant_flash_attn_op_builder.load()  # Compiles/loads the .so file


@impl(AS_LIBRARY, quant_flash_attn_op_builder.name, "PrivateUse1")# TODO 接口
def npu_quant_flash_attn(q, k, v,
                        q_descale, k_descale, v_descale,
                        q_quant_mode, k_quant_mode, v_quant_mode,
                        block_table=None,
                        cu_seqlens_q=None, cu_seqlens_kv=None,
                        seqused_q=None, seqused_kv=None,
                        sinks=None, attn_mask=None, metadata=None,
                        q_dtype=None, k_dtype=None, v_dtype=None,
                        q_descale_dtype=None, k_descale_dtype=None, v_descale_dtype=None,
                        quant_block_size_qs = 128,
                        quant_block_size_ks = 256,
                        quant_block_size_vs = 256,
                        softmax_scale=1.0, mask_mode=1, win_left=-1, win_right=-1,
                        max_seqlen_q=-1, max_seqlen_kv=-1,
                        layout_q="BSND", layout_kv="BSND", layout_out="BSND",
                        softmax_precision=0, return_softmax_lse=0):                        
    """
    dispatcher implementation for NPU.
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    print("quant_flash_attn entrance ========================\n")
    if (q.dtype == torch.uint8):
        # q用uint8传入时，必须传入相应的伪装类型
        if (q_dtype is None):
            raise ValueError("q_dtype can not be None when q is torch.uint8.")
        if (q_dtype not in {torch_npu.float4_e2m1fn_x2, torch_npu.hifloat8}):
            raise ValueError("q_dtype must be torch_npu.float4_e2m1fn_x2 or torch_npu.hifloat8.")

    if (k.dtype == torch.uint8):
        if (k_dtype is None):
            raise ValueError("k_dtype can not be None when k is torch.uint8.")
        if (k_dtype not in {torch_npu.float4_e2m1fn_x2, torch_npu.hifloat8}):
            raise ValueError("k_dtype must be torch_npu.float4_e2m1fn_x2 or torch_npu.hifloat8.")

    if (v.dtype == torch.uint8):
        if (v_dtype is None):
            raise ValueError("v_dtype can not be None when q is torch.uint8.")
        if (v_dtype not in {torch_npu.float4_e2m1fn_x2, torch_npu.hifloat8}):
            raise ValueError("v_dtype must be torch_npu.float4_e2m1fn_x2 or torch_npu.hifloat8.")

    if (q_descale_dtype is None or k_descale_dtype is None or k_descale_dtype is None):
        raise ValueError("q_descale_dtype/k_descale_dtype/k_descale_dtype can not be None.")
    if (q_descale_dtype is not torch_npu.float8_e8m0fnu):
        raise ValueError("q_descale_dtype must be torch_npu.float8_e8m0fnu.")
    if (k_descale_dtype is not torch_npu.float8_e8m0fnu):
        raise ValueError("k_descale_dtype must be torch_npu.float8_e8m0fnu.")
    if (v_descale_dtype is not torch_npu.float8_e8m0fnu):
        raise ValueError("v_descale_dtype must be torch_npu.float8_e8m0fnu.")
    return op_module.npu_quant_flash_attn(q, k, v,
                                    q_descale, k_descale, v_descale,
                                    q_quant_mode, k_quant_mode, v_quant_mode,
                                    block_table,
                                    cu_seqlens_q, cu_seqlens_kv,
                                    seqused_q, seqused_kv,
                                    sinks, attn_mask, metadata,
                                    q_dtype, k_dtype, v_dtype,
                                    q_descale_dtype, k_descale_dtype, v_descale_dtype,
                                    quant_block_size_qs,
                                    quant_block_size_ks,
                                    quant_block_size_vs,
                                    softmax_scale, mask_mode, win_left, win_right,
                                    max_seqlen_q, max_seqlen_kv,
                                    layout_q, layout_kv, layout_out,
                                    softmax_precision, return_softmax_lse
                                    )