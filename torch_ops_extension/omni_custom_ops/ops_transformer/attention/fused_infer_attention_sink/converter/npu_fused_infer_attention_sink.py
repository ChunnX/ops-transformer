# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import (
    Any, Callable, ContextManager, Iterator, List, Literal, NamedTuple, Optional, Sequence, Tuple, TypeVar,
    Union, overload,
)
import torch
import torch_npu
import torchair
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType, TensorType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, F64, I32, I16, I64, I8, U8, \
    BOOL, Support
from torchair._ge_concrete_graph.utils import dtype_promote
from torchair.ge import attr
from torch.library import Library, impl
from typing import Any, List, Tuple, Union, Callable, Optional
from torchair.ge._ge_graph import get_default_ge_graph, next_unique_name, auto_convert_to_tensor, compat_as_bytes, \
    get_invalid_desc


@register_fx_node_ge_converter(torch.ops.custom.npu_fused_infer_attention_sink.default)
def convert_npu_fused_infer_attention_sink(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    query_rope: Optional[Tensor] = None,
    key_rope: Optional[Tensor] = None,
    pse_shift: Optional[Tensor] = None,
    atten_mask: Optional[Tensor] = None,
    actual_seq_qlen: Optional[Tensor] = None,
    actual_seq_kvlen: Optional[Tensor] = None,
    block_table: Optional[Tensor] = None,
    dequant_scale_query: Optional[Tensor] = None,
    dequant_scale_key: Optional[Tensor] = None,
    dequant_offset_key: Optional[Tensor] = None,
    dequant_scale_value: Optional[Tensor] = None,
    dequant_offset_value: Optional[Tensor] = None,
    dequant_scale_key_rope: Optional[Tensor] = None,
    key_sink: Optional[Tensor] = None,
    key_rope_sink: Optional[Tensor] = None,
    value_sink: Optional[Tensor] = None,
    quant_scale_out: Optional[Tensor] = None,
    quant_offset_out: Optional[Tensor] = None,
    meta_data: Optional[Tensor] = None,
    num_query_heads: int = 1,
    num_key_value_heads: int = 0,
    softmax_scale: float = 1.0,
    pre_tokens: int = 2147483647,
    next_tokens: int = 2147483647,
    input_layout: str = "TND",
    sparse_mode: int = 0,
    block_size: int = 0,
    query_quant_mode: int = 0,
    key_quant_mode: int = 0,
    value_quant_mode: int = 0,
    inner_precise: int = 0,
    return_softmax_lse: bool = False,
    sink_number: int = 0,
    query_dtype: Optional[int] = None,
    key_dtype: Optional[int] = None,
    value_dtype: Optional[int] = None,
    query_rope_dtype: Optional[int] = None,
    key_rope_dtype: Optional[int] = None,
    key_shared_prefix_dtype: Optional[int] = None,
    value_shared_prefix_dtype: Optional[int] = None,
    dequant_scale_query_dtype: Optional[int] = None,
    dequant_scale_key_dtype: Optional[int] = None,
    dequant_scale_value_dtype: Optional[int] = None,
    dequant_scale_key_rope_dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None,
):
    if input_layout == 'BSH':
        const = ge.Const([1, 1, 8])
    else:
        const = ge.Const([1, 1, 1, 8])
    if key is not None and key.dtype == DataType.DT_INT32:
        shape = ge.Shape(key)
        key_shape = ge.Mul(shape, const)
        key = ge.Bitcast(key, type=DataType.DT_INT4)
        key = ge.Reshape(key, key_shape)

    if value is not None and value.dtype == DataType.DT_INT32:
        shape = ge.Shape(value)
        value_shape = ge.Mul(shape, const)
        value = ge.Bitcast(value, type=DataType.DT_INT4)
        value = ge.Reshape(value, value_shape)

    key_list = [key]
    value_list = [value]
    # dropped params
    quant_scale1 = None
    dequant_scale2 = None
    dequant_scale1 = None
    antiquant_scale = None
    antiquant_offset = None
    query_padding_size = None
    kv_padding_size = None
    learnable_sink = None
    antiquant_mode = 0
    pse_type = 0
    out_dtype = 0
    return torchair.ge.custom_op(
        "AiInfraFusedInferAttentionSink",
        inputs={"query": query,
                "key": key_list,
                "value": value_list,
                "pse_shift": pse_shift,
                "atten_mask": atten_mask,
                "actual_seq_lengths": actual_seq_qlen,
                "actual_seq_lengths_kv": actual_seq_kvlen,
                "dequant_scale1": dequant_scale1,
                "quant_scale1": quant_scale1,
                "dequant_scale2": dequant_scale2,
                "quant_scale2": quant_scale_out,
                "quant_offset2": quant_offset_out,
                "antiquant_scale": antiquant_scale,
                "antiquant_offset": antiquant_offset,
                "block_table": block_table,
                "query_padding_size": query_padding_size,
                "kv_padding_size": kv_padding_size,
                "key_antiquant_scale": dequant_scale_key,
                "key_antiquant_offset": dequant_offset_key,
                "value_antiquant_scale": dequant_scale_value,
                "value_antiquant_offset": dequant_offset_value,
                "query_rope": query_rope,
                "key_rope": key_rope,
                "key_rope_antiquant_scale": dequant_scale_key_rope,
                "dequant_scale_query": dequant_scale_query,
                "meta_data":meta_data,
                "learnable_sink": learnable_sink,
                "key_sink": key_sink,
                "key_rope_sink": key_rope_sink,
                "value_sink": value_sink,
                },
        attrs={"num_heads": attr.Int(num_query_heads),
               "scale": attr.Float(softmax_scale),
               "pre_tokens": attr.Int(pre_tokens),
               "next_tokens": attr.Int(next_tokens),
               "input_layout": attr.Str(input_layout),
               "num_key_value_heads": attr.Int(num_key_value_heads),
               "sparse_mode": attr.Int(sparse_mode),
               "inner_precise": attr.Int(inner_precise),
               "block_size": attr.Int(block_size),
               "antiquant_mode": attr.Int(antiquant_mode),
               "softmax_lse_flag": attr.Bool(return_softmax_lse),
               "key_antiquant_mode": attr.Int(key_quant_mode),
               "value_antiquant_mode": attr.Int(value_quant_mode),
               "query_quant_mode": attr.Int(query_quant_mode),
               "pse_type": attr.Int(pse_type),
               "sink_number": attr.Int(sink_number),
               "out_dtype": attr.Int(out_dtype),
               },
        outputs=['attention_out', 'softmax_lse']
    )