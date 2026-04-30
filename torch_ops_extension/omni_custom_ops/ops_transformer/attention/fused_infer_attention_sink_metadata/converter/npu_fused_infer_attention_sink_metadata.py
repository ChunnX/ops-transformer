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


@register_fx_node_ge_converter(torch.ops.custom._npu_fused_infer_attention_sink_metadata.default)
def convert_npu_fused_infer_attention_sink(
    num_heads_q: int,
    num_heads_kv: int,
    head_dim_qk: int,
    head_dim_v: int,
    *,
    actual_seq_lengths: Optional[Tensor] = None,
    actual_seq_lengths_kv: Optional[Tensor] = None,
    batch_size: int = 0,
    sparse_mode: int = 0,
    pre_tokens: int = 2147483647,
    next_tokens: int = 2147483647,
    input_layout: str = "TND",
    input_layout_kv: str = "TND",
    sink_num: int = 0,
    k_sink_num: int = 0,
    rope_head_dim: int = 0,
    block_size: int = 0,
    aic_core_num: int = 0,
    aiv_core_num: int = 0,
):
    stream_info = torch.npu.get_stream_limit(torch.npu.current_stream())
    soc_version = torch.npu.get_device_properties().name

    return torchair.ge.custom_op(
        "AiInfraFusedInferAttentionSinkMetadata",
        inputs={"actual_seq_lengths": actual_seq_lengths,
                "actual_seq_lengths_kv": actual_seq_lengths_kv,
                },
        attrs={"num_heads_q": attr.Int(num_heads_q),
               "num_heads_kv": attr.Int(num_heads_kv),
               "head_dim_qk": attr.Int(head_dim_qk),
               "head_dim_v": attr.Int(head_dim_v),
               "batch_size": attr.Int(batch_size),
               "sparse_mode": attr.Int(sparse_mode),
               "pre_tokens": attr.Int(pre_tokens),
               "next_tokens": attr.Int(next_tokens),
               "input_layout": attr.Str(input_layout),
               "input_layout_kv": attr.Str(input_layout_kv),
               "sink_num": attr.Int(sink_num),
               "k_sink_num": attr.Int(k_sink_num),
               "rope_head_dim": attr.Int(rope_head_dim),
               "block_size": attr.Int(block_size),
               "soc_version": attr.Str(soc_version),
               "aic_core_num": attr.Int(aic_core_num),
               "aiv_core_num": attr.Int(aiv_core_num),
               },
        outputs=['metaData']
    )