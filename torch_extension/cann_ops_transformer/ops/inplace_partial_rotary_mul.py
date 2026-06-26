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
import torch_npu
from torch.library import impl
from cann_ops_transformer.op_builder.builder import OpBuilder
from cann_ops_transformer.op_builder.builder import AS_LIBRARY


class InplacePartialRotaryMulOpBuilder(OpBuilder):
    def __init__(self):
        super(InplacePartialRotaryMulOpBuilder, self).__init__("inplace_partial_rotary_mul")

    def sources(self):
        """Path to C++ source code."""
        return ['ops/csrc/inplace_partial_rotary_mul.cpp']

    def schema(self) -> str:
        """PyTorch operator signature."""
        return (
            "inplace_partial_rotary_mul(Tensor(a!) x, Tensor r1, Tensor r2, *, "
            "str rotary_mode=\"interleave\", int[2] partial_slice=[0, 0]) -> ()"
        )

    def register_meta(self):
        """
        Registers the Meta implementation.
        """
        @impl(AS_LIBRARY, self.name, "Meta")
        def inplace_partial_rotary_mul_meta(
            x: torch.Tensor,
            r1: torch.Tensor,
            r2: torch.Tensor,
            *,
            rotary_mode: str = "interleave",
            partial_slice: Optional[List[int]] = None,
        ) -> None:
            partial_slice = [0, 0] if partial_slice is None else partial_slice
            if x.dim() != 4:
                raise ValueError(f"Input tensor x's dim num should be 4, actual {x.dim()}.")
            if rotary_mode != "interleave":
                raise ValueError("rotary_mode only supports 'interleave'.")
            if len(partial_slice) != 2:
                raise ValueError("partial_slice must contain exactly two integers.")
            return


# Instantiate the builder
inplace_partial_rotary_mul_op_builder = InplacePartialRotaryMulOpBuilder()
op_module = inplace_partial_rotary_mul_op_builder.load()


@impl(AS_LIBRARY, inplace_partial_rotary_mul_op_builder.name, "PrivateUse1")
def _inplace_partial_rotary_mul(
    x: torch.Tensor,
    r1: torch.Tensor,
    r2: torch.Tensor,
    *,
    rotary_mode: str = "interleave",
    partial_slice: Optional[List[int]] = None,
) -> None:
    """
    dispatcher implementation for NPU.
    'PrivateUse1' is the combine key for custom NPU backends.
    """
    partial_slice = [0, 0] if partial_slice is None else partial_slice
    op_module.inplace_partial_rotary_mul(x, r1, r2, rotary_mode, partial_slice)


def inplace_partial_rotary_mul(
    x: torch.Tensor,
    r1: torch.Tensor,
    r2: torch.Tensor,
    *,
    rotary_mode: str = "interleave",
    partial_slice: Optional[List[int]] = None,
) -> None:
    """
    Applies partial rotary position embedding to ``x`` in-place on NPU.

    Args:
        x (Tensor): Input tensor to be modified in-place.
        r1 (Tensor): Cosine component tensor.
        r2 (Tensor): Sine component tensor.
        rotary_mode (str): Only ``"interleave"`` is currently supported.
            Defaults to ``"interleave"``.
        partial_slice (List[int]): Two-element slice range along the head
            dimension. Defaults to ``[0, 0]``.
    """
    partial_slice = [0, 0] if partial_slice is None else partial_slice
    return torch.ops.cann_ops_transformer.inplace_partial_rotary_mul(
        x, r1, r2, rotary_mode=rotary_mode, partial_slice=partial_slice)
