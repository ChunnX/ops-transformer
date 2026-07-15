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
 * \file compressor_grad_tiling_data.h
 * \brief
 */

#ifndef COMPRESSOR_GRAD_TILING_DATA_H
#define COMPRESSOR_GRAD_TILING_DATA_H
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

const uint32_t CMP_GRAD_MAX_AIC_CORE_NUM = 36;

namespace optiling {
    struct CompressorGradBaseParams {
        uint32_t batchSize = 0;
        uint32_t seqSize = 0;
        uint32_t hiddenSize = 0;
        uint32_t tokenSize = 0;
        uint32_t headDim = 0;
        uint32_t cmpRatio = 4;
        uint32_t usedCoreNum = 0;
        uint64_t stateCacheStrideDim0 = 0;
    };

    struct CompressorGradInnerSplitParams {
        uint32_t mBaseSize;
        uint32_t dBaseSize;
        uint32_t dGroupNum;
        uint32_t mBlockNum;
        uint32_t hBaseNum;
    };

    struct CompressorGradWorkspaceParams {
        uint32_t dxPartialSize;
        uint32_t dwkvPartialSize;
        uint32_t dwgatePartialSize;
        uint32_t dapePartialSize;
        uint32_t dbWorkspaceRatio = 2;
    };

    struct CompressorGradTilingData {
        CompressorGradBaseParams baseParams;
        CompressorGradInnerSplitParams innerSplitParams;
        CompressorGradWorkspaceParams workspaceParams;
    };
} // optiling

#endif  // COMPRESSOR_GRAD_TILING_DATA_H
