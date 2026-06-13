/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file inplace_partial_rotary_mul_grad_apt.cpp
 * \brief InplacePartialRotaryMulGrad AscendC kernel
 */

#include "kernel_operator.h"
#include "arch35/inplace_partial_rotary_mul_grad_tiling_data.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void inplace_partial_rotary_mul_grad(
    GM_ADDR dy, GM_ADDR cos, GM_ADDR sin, GM_ADDR x,
    GM_ADDR dyOut, GM_ADDR dcos, GM_ADDR dsin,
    GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

    InplacePartialRotaryMulGradTilingData tilingData;
    auto copySize = sizeof(InplacePartialRotaryMulGradTilingData);
    DataCopyParams copyParams;
    DataCopy(tiling, (uint8_t*)&tilingData, copySize, copyParams);

    int64_t blockLen = tilingData.totalLength / tilingData.usedCoreNum;
    int64_t blockOffset = GetBlockIdx() * blockLen;
    int64_t tailLen = tilingData.totalLength - GetBlockIdx() * blockLen;
    if (GetBlockIdx() == tilingData.usedCoreNum - 1) {
        blockLen = tailLen;
    }

    if (blockLen <= 0) {
        return;
    }

    int64_t bufSize = blockLen * sizeof(uint8_t);
    LocalTensor<uint8_t> ubBuf = TBuf<TPosition::VECCALC>::GetInstance().AllocBuf<uint8_t>(bufSize);

    if (blockOffset < tilingData.totalLength) {
        DataCopyExtParams extParams;
        DataCopy(ubBuf, dy + blockOffset * sizeof(uint8_t), bufSize, extParams);
        DataCopy(dyOut + blockOffset * sizeof(uint8_t), ubBuf, bufSize, extParams);
    }
}
