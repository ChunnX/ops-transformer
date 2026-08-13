# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ---------------------------------------------------------------------------
# GE Converter for Graph Mode
try:
    import torch
    import torch_npu
    import torchair
    from typing import List, Optional
    from torchair.ge._ge_graph import Tensor, TensorSpec
    from torchair._ge_concrete_graph.fx2ge_converter import (
        register_fx_node_ge_converter,
    )
    from torchair.ge import attr

    _TORCHAIR_AVAILABLE = True
except ImportError:
    _TORCHAIR_AVAILABLE = False


_ROTARY_MODE_MAP = {
    "interleave": 1,
}


if _TORCHAIR_AVAILABLE:

    @register_fx_node_ge_converter(
        torch.ops.cann_ops_transformer.inplace_partial_rotary_mul.default
    )
    def convert_inplace_partial_rotary_mul(
        x: Tensor,
        r1: Tensor,
        r2: Tensor,
        *,
        rotary_mode: str = "interleave",
        partial_slice: Optional[List[int]] = None,
        meta_outputs: TensorSpec = None,
    ):
        partial_slice = [0, 0] if partial_slice is None else partial_slice
        if rotary_mode not in _ROTARY_MODE_MAP:
            raise ValueError(
                f"rotary_mode only supports 'interleave', got '{rotary_mode}'."
            )
        # Use a custom GE node so TorchAir preserves the backend's in-place
        # aliasing instead of functionalizing the Tensor(a!) -> () op and
        # inserting a copy-back for view/unsqueeze inputs.
        return torchair.ge.custom_op(
            "InplacePartialRotaryMul",
            inputs={"x": x, "cos": r1, "sin": r2},
            attrs={
                "rotary_mode": attr.Int(_ROTARY_MODE_MAP[rotary_mode]),
                "partial_slice": attr.ListInt(partial_slice),
            },
            outputs=["x"],
        )
else:

    def convert_inplace_partial_rotary_mul(*args, **kwargs):
        raise RuntimeError(
            "GE converter requires torchair, but torchair is not available."
        )
