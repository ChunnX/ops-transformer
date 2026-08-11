#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Lazy E2E wrapper for ops-test-kit signature discovery.

Keeping the CANN package import inside the function lets ``ttk e2e --validate``
parse the API in environments that do not have torch_npu installed.  A real
execution still imports and calls the installed public Torch extension.
"""

from typing import Tuple

import torch


def block_attn_res_update_ttk(
    partial_block: torch.Tensor,
    partial_delta: torch.Tensor,
    inter_numerator: torch.Tensor,
    inter_max: torch.Tensor,
    inter_exp_sum: torch.Tensor,
    effective_query: torch.Tensor,
    epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Call the installed BlockAttnResUpdate Torch API."""
    import cann_ops_transformer

    return cann_ops_transformer.block_attn_res_update(
        partial_block,
        partial_delta,
        inter_numerator,
        inter_max,
        inter_exp_sum,
        effective_query,
        epsilon,
    )
