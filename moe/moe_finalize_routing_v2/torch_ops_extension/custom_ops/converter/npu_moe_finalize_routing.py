# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import List, Optional

import torch
import torchair
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair.ge import attr


@register_fx_node_ge_converter(torch.ops.custom.npu_moe_finalize_routing.default)
def convert_npu_moe_finalize_routing(
    expanded_x: Tensor,
    expanded_row_idx: Tensor,
    *,
    x1: Optional[Tensor] = None,
    x2: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    scales: Optional[Tensor] = None,
    expert_idx: Optional[Tensor] = None,
    x: Optional[Tensor] = None,
    alpha1: Optional[Tensor] = None,
    alpha2: Optional[Tensor] = None,
    v: Optional[Tensor] = None,
    drop_pad_mode: Optional[int] = 0,
    zero_expert_range: Optional[List[int]] = None,
    copy_expert_range: Optional[List[int]] = None,
    constant_expert_range: Optional[List[int]] = None,
    k: Optional[int] = 1,
    meta_outputs: List[TensorSpec] = None,
):
    return torchair.ge.custom_op(
        "MoeFinalizeRoutingV4",
        inputs={
            "expanded_x": expanded_x,
            "expanded_row_idx": expanded_row_idx,
            "x1": x1,
            "x2": x2,
            "bias": bias,
            "scales": scales,
            "expert_idx": expert_idx,
            "x": x,
            "alpha1": alpha1,
            "alpha2": alpha2,
            "v": v,
        },
        attrs={
            "drop_pad_mode": attr.Int(drop_pad_mode if drop_pad_mode is not None else 0),
            "zero_expert_range": attr.ListInt(zero_expert_range if zero_expert_range is not None else []),
            "copy_expert_range": attr.ListInt(copy_expert_range if copy_expert_range is not None else []),
            "constant_expert_range": attr.ListInt(constant_expert_range if constant_expert_range is not None else []),
            "k": attr.Int(k if k is not None else 1),
        },
        outputs=["y"],
    )
