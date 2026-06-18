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
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY


class ScatterPaKvCacheWithKScaleOpBuilder(OpBuilder):
    """
    torch_npu适配：scatter_pa_kv_cache_with_k_scale算子
    
    功能：将FP8格式的key/value和对应的key_scale更新到KV cache中
    
    输入：
        key: [num_tokens, num_head, k_head_size] FP8格式
        value: [num_tokens, num_head, v_head_size] FP8格式
        key_cache: [num_blocks, num_head, block_size, k_head_size] FP8格式
        value_cache: [num_blocks, num_head, block_size, v_head_size] FP8格式
        slot_mapping: [num_tokens] int32/int64
        key_scale: [num_tokens, num_head] float32
        key_scale_cache: [num_blocks, num_head, block_size, 1] float32
    
    输出（inplace更新）：
        key_cache, value_cache, key_scale_cache
    """
    
    def __init__(self):
        super(ScatterPaKvCacheWithKScaleOpBuilder, self).__init__("scatter_pa_kv_cache_with_k_scale")

    def sources(self):
        """C++源代码路径"""
        return ['ops/csrc/scatter_pa_kv_cache_with_k_scale.cpp']

    def schema(self) -> str:
        """PyTorch算子签名"""
        return "scatter_pa_kv_cache_with_k_scale(" \
               "Tensor key, Tensor value, Tensor(a!) key_cache, Tensor(b!) value_cache, " \
               "Tensor slot_mapping, Tensor key_scale, Tensor(c!) key_scale_cache, *," \
               "str cache_layout='BNBD') -> (Tensor(a!), Tensor(b!), Tensor(c!))"

    def register_meta(self):
        """
        Meta实现：Shape/Dtype推导
        
        输出shape与输入cache相同（inplace更新）
        输出dtype与输入dtype相同
        """
        @impl(AS_LIBRARY, self.name, "Meta")
        def scatter_pa_kv_cache_with_k_scale_meta(
            key, value, key_cache, value_cache,
            slot_mapping, key_scale, key_scale_cache,
            *, cache_layout='BNBD'
        ):
            # 输出shape与输入cache相同
            key_cache_out = torch.empty(
                key_cache.size(),
                dtype=key_cache.dtype,
                device='meta'
            )
            
            value_cache_out = torch.empty(
                value_cache.size(),
                dtype=value_cache.dtype,
                device='meta'
            )
            
            key_scale_cache_out = torch.empty(
                key_scale_cache.size(),
                dtype=key_scale_cache.dtype,
                device='meta'
            )
            
            return (key_cache_out, value_cache_out, key_scale_cache_out)


# 实例化OpBuilder
scatter_pa_kv_cache_with_k_scale_op_builder = ScatterPaKvCacheWithKScaleOpBuilder()
op_module = scatter_pa_kv_cache_with_k_scale_op_builder.load()


@impl(AS_LIBRARY, scatter_pa_kv_cache_with_k_scale_op_builder.name, "PrivateUse1")
def scatter_pa_kv_cache_with_k_scale(
    key, value, key_cache, value_cache,
    slot_mapping, key_scale, key_scale_cache,
    *, cache_layout='BNBD'
):
    """
    torch_npu实现：scatter_pa_kv_cache_with_k_scale
    
    Args:
        key: [num_tokens, num_head, k_head_size] FP8格式，需要更新的key
        value: [num_tokens, num_head, v_head_size] FP8格式，需要更新的value
        key_cache: [num_blocks, num_head, block_size, k_head_size] FP8格式，被更新的cache
        value_cache: [num_blocks, num_head, block_size, v_head_size] FP8格式，被更新的cache
        slot_mapping: [num_tokens] int32/int64，每个token在cache中的偏移
        key_scale: [num_tokens, num_head] float32，需要更新的scale
        key_scale_cache: [num_blocks, num_head, block_size, 1] float32，被更新的scale cache
        cache_layout: str，cache布局格式（默认'BNBD'）
    
    Returns:
        tuple: (key_cache_out, value_cache_out, key_scale_cache_out)
        
    注意：
        1. 算子支持FP8_E5M2和FP8_E4M3FN两种dtype
        2. slot_mapping值必须在[0, num_blocks*block_size-1]范围内
        3. 超出范围的slot会被跳过（不更新对应cache）
    """
    return op_module.scatter_pa_kv_cache_with_k_scale(
        key, value, key_cache, value_cache,
        slot_mapping, key_scale, key_scale_cache,
        cache_layout
    )


# GE Converter for Graph Mode (torchair适配)
try:
    import torchair
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
    from torchair._ge_concrete_graph.supported_declaration import Support
    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:
    @declare_supported([
        Support(torch.float8_e5m2, (2, 2, 64), (2, 2, 64)),
        Support(torch.float8_e4m3fn, (2, 2, 64), (2, 2, 64)),
    ])
    @register_fx_node_ge_converter(torch.ops.cann_ops_transformer.scatter_pa_kv_cache_with_k_scale.default)
    def converter_scatter_pa_kv_cache_with_k_scale(
        key: Tensor,
        value: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        slot_mapping: Tensor,
        key_scale: Tensor,
        key_scale_cache: Tensor,
        *,
        cache_layout: str = 'BNBD',
        meta_outputs: TensorSpec = None
    ):
        """
        torchair GE转换器：scatter_pa_kv_cache_with_k_scale
        
        将PyTorch算子转换为GE图引擎算子，用于图模式执行
        
        Args:
            key: GE Tensor，FP8格式的key输入
            value: GE Tensor，FP8格式的value输入
            key_cache: GE Tensor，FP8格式的key cache
            value_cache: GE Tensor，FP8格式的value cache
            slot_mapping: GE Tensor，slot偏移索引
            key_scale: GE Tensor，float32格式的key scale
            key_scale_cache: GE Tensor，float32格式的key scale cache
            cache_layout: str，cache布局格式
            meta_outputs: TensorSpec，输出的shape/dtype信息
            
        Returns:
            tuple: (key_cache_out, value_cache_out, key_scale_cache_out)
        """
        return ge.ScatterPaKvCacheWithKScale(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            key_scale,
            key_scale_cache,
            cache_layout=cache_layout
        )