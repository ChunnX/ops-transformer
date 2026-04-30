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


# 为自定义算子注册converter，用于torch.compile 场景成图
@auto_convert_to_tensor(
    [False, False, False, False, False, False, False],
    [False, False, False, False, False, False, False])
def AiInfraChunkGatedDeltaRuleRecurrence(initial_state: Tensor, kgexp: Tensor, value: Tensor, k_cumdecay: Tensor,
    qgexp: Tensor, gexp: Tensor, cu_seqlens: Tensor, *, dependencies=[], node_name=None):
    inputs = {
        "initial_state": initial_state,
        "kgexp": kgexp,
        "value": value,
        "k_cumdecay": k_cumdecay,
        "qgexp": qgexp,
        "gexp": gexp,
        "cu_seqlens": cu_seqlens
    }
    outputs = [
        "initial_state",
        "attn_inter_out",
        "v_new_out"
    ]
    return torchair.ge.custom_op("AiInfraChunkGatedDeltaRuleRecurrence", inputs=inputs, attrs={}, outputs=outputs)


# 为自定义算子注册converter，用于torch.compile 场景成图

# 注意： meta_outputs形参名为固定写法，若写错会影响ge节点的输出dtype与shape推导
@register_fx_node_ge_converter(torch.ops.custom.npu_chunk_gated_delta_rule_recurrence.default)
def convert_npu_npu_chunk_gated_delta_rule_recurrence(initial_state: Tensor,
    kgexp: Tensor, value: Tensor, k_cumdecay: Tensor, qgexp: Tensor, gexp: Tensor, actual_seqlens: Tensor, *,
    meta_outputs: Any = None):
    (initial_state_out, attn_inter_out, v_new_out) = AiInfraChunkGatedDeltaRuleRecurrence(
        initial_state, kgexp, value, k_cumdecay, qgexp, gexp, actual_seqlens)
    return (attn_inter_out, v_new_out)
