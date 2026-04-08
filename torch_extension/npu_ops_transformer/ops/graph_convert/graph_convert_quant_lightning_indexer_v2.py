# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# GE Converter for Graph Mode
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import torch
from torch import Generator, contiguous_format, inf, strided, SymInt
from torch.types import Device, Number, _bool, _complex, _device, _dtype, _float, _int, _layout, _qscheme, _size
import torchair
from torchair._ge_concrete_graph import ge_apis as ge
from torchair._ge_concrete_graph.fx2ge_converter import declare_supported, register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec, DataType
from torchair._ge_concrete_graph.supported_declaration import _TypedTensor, F32, F16, BF16, F64, I32, I16, I64, \
    I8, U8, BOOL, Support
from torchair.ge import attr


@register_fx_node_ge_converter(torch.ops.npu_ops_transformer.npu_quant_lightning_indexer_v2.default)
def convert_npu_quant_lightning_indexer_v2(
    query: Tensor,
    key: Tensor,
    weights: Tensor,
    query_dequant_scale: Tensor,
    key_dequant_scale: Tensor,
    query_quant_mode: int,
    key_quant_mode: int,
    *,
    actual_seq_lengths_query: Optional[Tensor] = None,
    actual_seq_lengths_key: Optional[Tensor] = None,
    block_table: Optional[Tensor] = None,
    metadata: Optional[Tensor] = None,
    layout_query: str = "BSND",
    layout_key: str = "BSND",
    sparse_count: int = 2048,
    sparse_mode: int = 3,
    pre_tokens: int = 9223372036854775807,
    next_tokens: int = 9223372036854775807,
    query_dtype: Optional[int] = None,
    key_dtype: Optional[int] = None,
    meta_outputs: TensorSpec = None,
):
    import torch_npu
    if query is not None and query_dtype is not None and query_dtype != torch_npu.hifloat8:
        raise RuntimeError("The query_dtype is only used to support hifloat8, and should be default when the dtype of query tensor is not hifloat8")
    if key is not None and key_dtype is not None and key_dtype != torch_npu.hifloat8:
        raise RuntimeError("The key_dtype is only used to support hifloat8, and should be default when the dtype of key tensor is not hifloat8")

    if query is not None and query_dtype == torch_npu.hifloat8:
        query = ge.Bitcast(query, type=DataType.DT_HIFLOAT8)
    if key is not None and key_dtype == torch_npu.hifloat8:
        key = ge.Bitcast(key, type=DataType.DT_HIFLOAT8)
    return torchair.ge.custom_op(
        "QuantLightningIndexer",
        inputs={"query": query,
                "key": key,
                "weights": weights,
                "query_dequant_scale": query_dequant_scale,
                "key_dequant_scale": key_dequant_scale,
                "actual_seq_lengths_query": actual_seq_lengths_query,
                "actual_seq_lengths_key": actual_seq_lengths_key,
                "block_table": block_table,
                "metadata": metadata,
        },
        attrs={"query_quant_mode": attr.Int(query_quant_mode),
                "key_quant_mode": attr.Int(key_quant_mode),
                "layout_query": attr.Str(layout_query),
                "layout_key": attr.Str(layout_key),
                "sparse_count": attr.Int(sparse_count),
                "sparse_mode": attr.Int(sparse_mode),
                "pre_tokens": attr.Int(pre_tokens),
                "next_tokens": attr.Int(next_tokens),
        },
        outputs=['sparse_indices']
    )