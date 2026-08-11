# copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import List, Optional, Tuple

import torch
import torchair
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec
from torchair.ge import attr


@register_fx_node_ge_converter(torch.ops.custom.npu_moe_token_permute.default)
def convert_npu_moe_token_permute(
    tokens: Tensor,
    indices: Tensor,
    *,
    num_out_tokens: Optional[int] = None,
    padded_mode: bool = False,
    quant_mode: int = -1,
    meta_outputs: List[TensorSpec] = None,
):
    return torchair.ge.custom_op(
        "MoeTokenPermuteV2",
        inputs={
            "tokens": tokens,
            "indices": indices,
        },
        attrs={
            "num_out_tokens": attr.Int(num_out_tokens if num_out_tokens is not None else 0),
            "padded_mode": attr.Bool(padded_mode),
            "quant_mode": attr.Int(quant_mode),
        },
        outputs=["permute_tokens", "sorted_indices", "expanded_scale"],
    )
