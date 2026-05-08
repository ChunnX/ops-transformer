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

def _calculate_batch_size(batch_size, cu_seqlens_q, seqused_q):
    batchSize = 0                                                                                                                                                                                                                  
    if batch_size is not None:                                                                                                                                                                                                     
        batchSize = batch_size                                                                                                                                                                                                     
    elif cu_seqlens_q is not None and cu_seqlens_q.size(0) > 0:                                                                                                                                                                    
        batchSize = cu_seqlens_q.size(0) - 1                                                                                                                                                                                       
    elif seqused_q is not None:                                                                                                                                                                                                    
        batchSize = seqused_q.size(0)
    return batchSize

def _calculate_metadata_size(batch_size, num_heads_kv):                                                                                                                                                   
    """计算 metadata tensor 的对齐后大小"""                                                                                                                                                                                                                                                                                                                                                                               
    metadataSize = ((36 + 72) * batch_size * num_heads_kv + 1) * 16                                                                                                                                                                 
    alignedSize = ((metadataSize + 4095) // 4096) * 4096                                                                                                                                                                        
    return alignedSize

class FlashAttnMetadataOpBuilder(OpBuilder):
    def __init__(self):
        super(FlashAttnMetadataOpBuilder, self).__init__("npu_flash_attn_metadata")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/flash_attn_metadata.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return "npu_flash_attn_metadata( int num_heads_q, int num_heads_kv, int head_dim, *, " \
            "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_kv=None, Tensor? seqused_q=None, Tensor? seqused_kv=None," \
            "int? batch_size=None, int? max_seqlen_q=None, int? max_seqlen_kv=None, " \
            "int? mask_mode=None, int? win_left=None, int? win_right=None, " \
            "str? layout_q=None, str? layout_kv=None, str? layout_out=None) -> Tensor"

    def register_meta(self):
        """
        Registers Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @torch.library.register_fake("npu_ops_transformer::" + self.name)
        def npu_flash_attn_metadata_meta( num_heads_q, num_heads_kv, head_dim,
                                          cu_seqlens_q = None, cu_seqlens_kv = None, seqused_q = None, seqused_kv = None,
                                          batch_size: Optional[int] = None, max_seqlen_q: Optional[int] = None,
                                          max_seqlen_kv: Optional[int] = None,
                                          mask_mode: Optional[int] = None, win_left: Optional[int] = None,
                                          win_right: Optional[int] = None, layout_q: Optional[str] = None,
                                          layout_kv: Optional[str] = None, layout_out: Optional[str] = None):
            metadataSize = _calculate_metadata_size(batch_size, num_heads_kv)
            return torch.empty((metadataSize,), dtype=torch.int32, device="meta")

# Instantiate the builder
flash_attn_metadata_op_builder = FlashAttnMetadataOpBuilder()
op_module = flash_attn_metadata_op_builder.load()


@impl(AS_LIBRARY, flash_attn_metadata_op_builder.name, "PrivateUse1")
def npu_flash_attn_metadata(num_heads_q, num_heads_kv, head_dim,
                            cu_seqlens_q = None, cu_seqlens_kv = None, seqused_q = None, seqused_kv = None,
                            batch_size: Optional[int] = None, max_seqlen_q: Optional[int] = None,
                            max_seqlen_kv: Optional[int] = None,
                            mask_mode: Optional[int] = None, win_left: Optional[int] = None,
                            win_right: Optional[int] = None, layout_q: Optional[str] = None,
                            layout_kv: Optional[str] = None, layout_out: Optional[str] = None):
    """
    Dispatcher implementation: NPU.
    'PrivateUse1' is dispatch key for custom NPU backends.
    """
    batch_size = _calculate_batch_size(batch_size, cu_seqlens_q, seqused_q) if batch_size is None else batch_size
    max_seqlen_q = -1 if max_seqlen_q is None else max_seqlen_q
    max_seqlen_kv = -1 if max_seqlen_kv is None else max_seqlen_kv
    mask_mode = 1 if mask_mode is None else mask_mode
    win_left = -1 if win_left is None else win_left
    win_right = -1 if win_right is None else win_right
    layout_q = "BSND" if layout_q is None else layout_q
    layout_kv = "BSND" if layout_kv is None else layout_kv
    layout_out = "BSND" if layout_out is None else layout_out

    metadataSize = _calculate_metadata_size(batch_size, num_heads_kv)
    output = torch.empty((metadataSize,), dtype=torch.int32, device="npu")
 
    return op_module.npu_flash_attn_metadata(cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv,
                                              num_heads_q, num_heads_kv, head_dim,
                                              batch_size, max_seqlen_q, max_seqlen_kv,
                                              mask_mode, win_left, win_right, layout_q, layout_kv, layout_out, output)


@torch.library.register_kernel("npu_ops_transformer::npu_flash_attn_metadata", None)                                                                                          
def npu_flash_attn_metadata_fallback(num_heads_q, num_heads_kv, head_dim,
                                    cu_seqlens_q = None, cu_seqlens_kv = None, seqused_q = None, seqused_kv = None,
                                    batch_size: Optional[int] = None, max_seqlen_q: Optional[int] = None,
                                    max_seqlen_kv: Optional[int] = None,
                                    mask_mode: Optional[int] = None, win_left: Optional[int] = None,
                                    win_right: Optional[int] = None, layout_q: Optional[str] = None,
                                    layout_kv: Optional[str] = None, layout_out: Optional[str] = None):                                                                                                                                    
    # 处理所有 tensor 都为 None 的情况                                                                                                                                         
    return npu_flash_attn_metadata(num_heads_q, num_heads_kv, head_dim,
                                    cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv,
                                    batch_size, max_seqlen_q,
                                    max_seqlen_kv,
                                    mask_mode, win_left,
                                    win_right, layout_q,
                                    layout_kv, layout_out)                                                                                                          
