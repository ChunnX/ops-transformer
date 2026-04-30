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


@register_fx_node_ge_converter(torch.ops.custom.npu_esa_select_topk.default)
def convert_npu_esa_select_topk(
    query: Tensor,
    key: Tensor,
    blk_size: int,
    init_blk_num: int,
    local_blk_num: int,
    topk: int,
    input_layout: str,
    *,
    actual_seq_q_len_optional: Optional[List[int]] = None,
    actual_seq_k_len_optional: Optional[List[int]] = None,
    actual_cmp_seq_k_len_optional: Optional[List[int]] = None,
    compress_blk_size: int = 16,
    meta_outputs: TensorSpec = None,
):
    if actual_seq_q_len_optional is not None:
        actual_seq_q_len_optional = dtype_promote(actual_seq_q_len_optional, target_dtype=DataType.DT_INT64)
    if actual_seq_k_len_optional is not None:
        actual_seq_k_len_optional = dtype_promote(actual_seq_k_len_optional, target_dtype=DataType.DT_INT64)
    if actual_cmp_seq_k_len_optional is not None:
        actual_cmp_seq_k_len_optional = dtype_promote(actual_cmp_seq_k_len_optional, target_dtype=DataType.DT_INT64)
    return torchair.ge.custom_op (
        "EsaSelectTopk",
        inputs={"query": query,
                "key": key,
                "actual_seq_qlen": actual_seq_q_len_optional,
                "actual_seq_klen": actual_seq_k_len_optional,
                "actual_cmp_seq_klen": actual_cmp_seq_k_len_optional,
                },
        attrs={"blk_size": attr.Int(blk_size),
               "init_blk_num": attr.Int(init_blk_num),
               "local_blk_num": attr.Int(local_blk_num),
               "topk": attr.Int(topk),
               "input_layout": attr.Str(input_layout),
               "compress_blk_size": attr.Int(compress_blk_size),
               },
        outputs=['topk_indices']
    )
