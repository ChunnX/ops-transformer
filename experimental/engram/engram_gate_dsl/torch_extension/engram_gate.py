# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
from torch.library import custom_op

from ops.engram_gate import engram_gate as _engram_gate_impl


@custom_op("engram_gate_dsl::engram_gate", mutates_args=())
def engram_gate_torch(
    x: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    weight: torch.Tensor,
    image_mask: torch.Tensor | None = None,
    *,
    eps: float = 1.0e-6,
    clamp_value: float = 1.0e-6,
) -> torch.Tensor:
    return _engram_gate_impl(
        x, key, value, weight, image_mask, eps=eps, clamp_value=clamp_value
    )


@engram_gate_torch.register_fake
def _engram_gate_fake(
    x,
    key,
    value,
    weight,
    image_mask=None,
    *,
    eps=1.0e-6,
    clamp_value=1.0e-6,
):
    return torch.empty_like(x)
