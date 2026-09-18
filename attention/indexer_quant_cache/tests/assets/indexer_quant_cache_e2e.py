#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""TTK adapter: expose both mutated tensors without changing the public API."""


def run(
    cache,
    cache_scale,
    x,
    slot_mapping,
    *,
    quant_mode="fp8",
    round_scale=True,
    x_scale=1.0,
):
    from cann_ops_transformer.ops import indexer_quant_cache

    indexer_quant_cache(
        cache,
        cache_scale,
        x,
        slot_mapping,
        quant_mode=quant_mode,
        round_scale=round_scale,
        x_scale=x_scale,
    )
    return cache, cache_scale
