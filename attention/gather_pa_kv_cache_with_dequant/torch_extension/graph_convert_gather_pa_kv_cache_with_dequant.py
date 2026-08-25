# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file in compliance with the License.
# THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for Graph Mode

# GE Converter for Graph Mode (torchair适配)
try:
    import torch
    import torch_npu
    import torchair
    from torch.library import impl
    from torchair._ge_concrete_graph import ge_apis as ge
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair._ge_concrete_graph.fx2ge_converter import (
        declare_supported,
        register_fx_node_ge_converter,
    )
    from torchair._ge_concrete_graph.supported_declaration import Support
    from typing import Any, Dict, List, Tuple, Union, Callable, Optional
    from torchair._ge_concrete_graph.ge_ir_pb2 import (
        GraphDef,
        OpDef,
        TensorDescriptor,
        TensorDef,
    )
    from torchair.ge._ge_graph import get_default_ge_graph, next_unique_name
    from torchair.ge._ge_graph import auto_convert_to_tensor
    from torchair.ge._ge_graph import DataType, TensorType
    from torchair.ge._ge_graph import compat_as_bytes, compat_as_bytes_list
    from torchair.ge._ge_graph import trans_to_list_list_int, trans_to_list_list_float
    from torchair.ge._ge_graph import get_invalid_desc
    from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef
    from torchair.ge import attr

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False

if _TORCHAIR_AVAILABLE:

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.gather_pa_kv_cache_with_dequant.default
    )
    def converter_gather_pa_kv_cache_with_dequant(
        key_cache: Tensor,
        value_cache: Tensor,
        key_scale_cache: Tensor,
        value_scale: Tensor,
        block_tables: Tensor,
        seq_lens: Tensor,
        key_ref: Tensor,
        value_ref: Tensor,
        *,
        seq_offset: Optional[Tensor] = None,
        cache_mode: str = "Norm",
        is_seq_lens_cumsum: bool = False,
        key_dst_type: int = 27,
        value_dst_type: int = 27,
        meta_outputs: TensorSpec = None,
    ):
        """
        torchair GE转换器：gather_pa_kv_cache_with_dequant

        将PyTorch算子转换为GE图引擎算子，用于图模式执行。
        原地更新 key_ref/value_ref，无返回值。

        Args:
            key_cache: GE Tensor，INT8 key cache
            value_cache: GE Tensor，FP8 value cache
            key_scale_cache: GE Tensor，FLOAT32 key scale cache
            value_scale: GE Tensor，FLOAT32 value scale
            block_tables: GE Tensor，INT32 分页 block 索引
            seq_lens: GE Tensor，INT32 序列长度
            key_ref: GE Tensor，OUT BF16 key 输出（原地写）
            value_ref: GE Tensor，OUT BF16 value 输出（原地写）
            seq_offset: Optional GE Tensor，token 起始偏移
            cache_mode: str，scale 布局模式 "Norm"/"C8"
            is_seq_lens_cumsum: bool，seq_lens 是否前缀和
            key_dst_type: int，key 目标 dtype（aclDataType，BF16=27）
            value_dst_type: int，value 目标 dtype（aclDataType，BF16=27）
            meta_outputs: TensorSpec，输出 shape/dtype 信息

        """
        return ge.GatherPaKvCacheWithDequant(
            key_cache,
            value_cache,
            key_scale_cache,
            value_scale,
            block_tables,
            seq_lens,
            key_ref,
            value_ref,
            seq_offset,
            cache_mode=cache_mode,
            is_seq_lens_cumsum=is_seq_lens_cumsum,
            key_dst_type=key_dst_type,
            value_dst_type=value_dst_type,
        )
