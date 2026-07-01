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
from torch.library import impl
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY

_npu_core_cache = None
def get_npu_core_count():
    global _npu_core_cache
    if _npu_core_cache is not None:
        return _npu_core_cache
    try:
        device_id = torch.npu.current_device()
        limits = torch.npu.get_device_limit(device_id)
        _npu_core_cache = (limits["cube_core_num"], limits["vector_core_num"])
    except RuntimeError:
        props = torch.npu.get_device_properties()
        _npu_core_cache = (props.cube_core_num, props.vector_core_num)
    return _npu_core_cache

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
    aicNum, aivNum = get_npu_core_count()                                                                                                                                                                                                                                                                                                                                                                    
    metadataSize = ((aicNum + aivNum) * batch_size * num_heads_kv + 1) * 16                                                                                                                                                                 
    alignedSize = ((metadataSize + 4095) // 4096) * 4096                                                                                                                                                                        
    return alignedSize

class MixedQuantFlashAttnMetadataOpBuilder(OpBuilder):
    def __init__(self):
        super(MixedQuantFlashAttnMetadataOpBuilder, self).__init__("mixed_quant_flash_attn_metadata")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/mixed_quant_flash_attn_metadata.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return "mixed_quant_flash_attn_metadata( int num_heads_q, int num_heads_kv, int head_dim, int quant_mode, *, " \
            "Tensor? cu_seqlens_q=None, Tensor? cu_seqlens_kv=None, Tensor? seqused_q=None, Tensor? seqused_kv=None," \
            "int? batch_size=None, int? max_seqlen_q=-1, int? max_seqlen_kv=-1, " \
            "int? mask_mode=0, int? win_left=-1, int? win_right=-1, " \
            "str? layout_q=\"BSND\", str? layout_kv=\"BSND\", str? layout_out=\"BSND\") -> Tensor"

    def register_meta(self):
        """
        Registers Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """
        @torch.library.register_fake("cann_ops_transformer::" + self.name)
        def mixed_quant_flash_attn_metadata_meta(num_heads_q: int, num_heads_kv: int, head_dim: int, quant_mode: int,
                                    cu_seqlens_q: Optional[torch.Tensor] = None, cu_seqlens_kv: Optional[torch.Tensor] = None,
                                    seqused_q: Optional[torch.Tensor] = None, seqused_kv: Optional[torch.Tensor] = None,
                                    batch_size: Optional[int] = None, max_seqlen_q: Optional[int] = -1,
                                    max_seqlen_kv: Optional[int] = -1,
                                    mask_mode: int = 0, win_left: Optional[int] = -1,
                                    win_right: Optional[int] = -1, layout_q: Optional[str] = "BSND",
                                    layout_kv: Optional[str] = "BSND", layout_out: Optional[str] = "BSND") -> torch.Tensor:
            batch_size = _calculate_batch_size(batch_size, cu_seqlens_q, seqused_q) if batch_size is None else batch_size
            metadataSize = _calculate_metadata_size(batch_size, num_heads_kv)
            return torch.empty((metadataSize,), dtype=torch.int32, device="npu")

# Instantiate the builder
mixed_quant_flash_attn_metadata_op_builder = MixedQuantFlashAttnMetadataOpBuilder()
op_module = mixed_quant_flash_attn_metadata_op_builder.load()


@impl(AS_LIBRARY, mixed_quant_flash_attn_metadata_op_builder.name, "PrivateUse1")
def mixed_quant_flash_attn_metadata(num_heads_q: int, num_heads_kv: int, head_dim: int, quant_mode: int,
                            cu_seqlens_q: Optional[torch.Tensor] = None, cu_seqlens_kv: Optional[torch.Tensor] = None,
                            seqused_q: Optional[torch.Tensor] = None, seqused_kv: Optional[torch.Tensor] = None,
                            batch_size: Optional[int] = None, max_seqlen_q: Optional[int] = -1,
                            max_seqlen_kv: Optional[int] = -1,
                            mask_mode: int = 0, win_left: Optional[int] = -1,
                            win_right: Optional[int] = -1, layout_q: Optional[str] = "BSND",
                            layout_kv: Optional[str] = "BSND", layout_out: Optional[str] = "BSND") -> torch.Tensor:
    """
    Dispatcher implementation: NPU.
    'PrivateUse1' is dispatch key for custom NPU backends.
    """
    batch_size = _calculate_batch_size(batch_size, cu_seqlens_q, seqused_q) if batch_size is None else batch_size
    metadataSize = _calculate_metadata_size(batch_size, num_heads_kv)
    output = torch.empty((metadataSize,), dtype=torch.int32, device="npu")
 
    return op_module.mixed_quant_flash_attn_metadata(cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv,
                                              num_heads_q, num_heads_kv, head_dim, quant_mode,
                                              batch_size, max_seqlen_q, max_seqlen_kv,
                                              mask_mode, win_left, win_right, layout_q, layout_kv, layout_out, output)

                                                                                                        
