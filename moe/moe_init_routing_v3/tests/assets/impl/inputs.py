#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Inputs adapter for MoeInitRoutingV3 TTK integration.

Ensures expertIdx values are within [0, expertNum) range. Supports optional
invalid_ratio (via kwargs) to randomly insert -1 (invalid expert) into expertIdx.
"""

import numpy as np


def aclnn_customize_inputs(x, expertIdx, scaleOptional, offsetOptional,
                           activeNum, expertCapacity, expertNum, dropPadMode,
                           expertTokensNumType, expertTokensNumFlag, quantMode,
                           activeExpertRangeOptional, rowIdxType,
                           expandedXOut, expandedRowIdxOut,
                           expertTokensCountOrCumsumOut, expandedScaleOut, **kwargs):
    """Post-process randomly generated inputs to satisfy MoeInitRoutingV3 constraints."""
    expertIdx_np = np.asarray(expertIdx)
    expert_num = int(expertNum)

    expertIdx_np = np.random.randint(0, expert_num, size=expertIdx_np.shape).astype(np.int32)

    invalid_ratio = kwargs.get("invalid_ratio", 0.0)
    if invalid_ratio > 0.0:
        mask = np.random.random(size=expertIdx_np.shape) < invalid_ratio
        expertIdx_np[mask] = -1

    expertIdx[...] = expertIdx_np

    if scaleOptional is not None and quantMode == -1:
        scale_np = np.random.uniform(-2, 2, size=scaleOptional.shape).astype(np.float32)
        scaleOptional[...] = scale_np

    if offsetOptional is not None and quantMode == 0:
        offsetOptional[...] = np.random.uniform(-2, 2, size=offsetOptional.shape).astype(np.float32)

    return (x, expertIdx, scaleOptional, offsetOptional,
            activeNum, expertCapacity, expertNum, dropPadMode,
            expertTokensNumType, expertTokensNumFlag, quantMode,
            activeExpertRangeOptional, rowIdxType,
            expandedXOut, expandedRowIdxOut,
            expertTokensCountOrCumsumOut, expandedScaleOut)
