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


@register_fx_node_ge_converter(torch.ops.custom.npu_sparse_flash_attention_gqa.default)
def convert_npu_sparse_flash_attention_gqa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    sparse_indices: Tensor,
    scale_value: float,
    sparse_block_size: int,
    *,
    actual_seq_lengths_query: Optional[Union[List[int], Tensor]] = None,
    actual_seq_lengths_kv: Optional[Union[List[int], Tensor]] = None,
    block_table: Optional[Tensor] = None,
    num_query_heads: int = 1,
    num_key_value_heads: int = 0,
    input_layout: str = "BSH",
    input_layout_kv: str = "BSND",
    sparse_mode: int = 0,
    block_size: int = 0,
    return_softmax_lse: bool = False,
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
    if actual_seq_lengths_query is not None:
        actual_seq_lengths_query = dtype_promote(actual_seq_lengths_query, target_dtype=DataType.DT_INT64)
    if actual_seq_lengths_kv is not None:
        actual_seq_lengths_kv = dtype_promote(actual_seq_lengths_kv, target_dtype=DataType.DT_INT64)
    # dropped params
    return torchair.ge.custom_op(
        "AiInfraSparseFlashAttentionGqa",
        inputs={"query": query,
                "key": key_list,
                "value": value_list,
                "sparse_indices": sparse_indices,
                "actual_seq_lengths": actual_seq_lengths_query,
                "actual_seq_lengths_kv": actual_seq_lengths_kv,
                "block_table": block_table,
                },
        attrs={"num_heads": attr.Int(num_query_heads),
               "sparse_block_size": attr.Int(sparse_block_size),
               "scale": attr.Float(scale_value),
               "input_layout": attr.Str(input_layout),
               "input_layout_kv": attr.Str(input_layout_kv),
               "num_key_value_heads": attr.Int(num_key_value_heads),
               "sparse_mode": attr.Int(sparse_mode),
               "block_size": attr.Int(block_size),
               "softmax_lse_flag": attr.Bool(return_softmax_lse),
               },
        outputs=['attention_out', 'softmax_lse']
    )

@register_fx_node_ge_converter(torch.ops.custom.npu_sparse_flash_attention_gqa.tensor)
def convert_npu_sparse_flash_attention_gqa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    sparse_indices: Tensor,
    scale_value: float,
    sparse_block_size: int,
    *,
    actual_seq_lengths_query: Optional[Tensor] = None,
    actual_seq_lengths_kv: Optional[Tensor] = None,
    block_table: Optional[Tensor] = None,
    num_query_heads: int = 1,
    num_key_value_heads: int = 0,
    input_layout: str = "BSH",
    input_layout_kv: str = "BSND",
    sparse_mode: int = 0,
    block_size: int = 0,
    return_softmax_lse: bool = False,
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

    if actual_seq_lengths_query is not None:
        actual_seq_lengths_query = dtype_promote(actual_seq_lengths_query, target_dtype=DataType.DT_INT64)
    if actual_seq_lengths_kv is not None:
        actual_seq_lengths_kv = dtype_promote(actual_seq_lengths_kv, target_dtype=DataType.DT_INT64)
    # dropped params
    return torchair.ge.custom_op(
        "AiInfraSparseFlashAttentionGqa",
        inputs={"query": query,
                "key": key_list,
                "value": value_list,
                "sparse_indices": sparse_indices,
                "actual_seq_lengths": actual_seq_lengths_query,
                "actual_seq_lengths_kv": actual_seq_lengths_kv,
                "block_table": block_table,
                },
        attrs={"num_heads": attr.Int(num_query_heads),
               "sparse_block_size": attr.Int(sparse_block_size),
               "scale": attr.Float(scale_value),
               "input_layout": attr.Str(input_layout),
               "input_layout_kv": attr.Str(input_layout_kv),
               "num_key_value_heads": attr.Int(num_key_value_heads),
               "sparse_mode": attr.Int(sparse_mode),
               "block_size": attr.Int(block_size),
               "softmax_lse_flag": attr.Bool(return_softmax_lse),
               },
        outputs=['attention_out', 'softmax_lse']
    )

