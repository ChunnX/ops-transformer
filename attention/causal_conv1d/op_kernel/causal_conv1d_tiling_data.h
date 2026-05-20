/**
 * Copyright (c) 2025 Tianjin University, Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the BSD 3-Clause License (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 */

/*!
 * \file causal_conv1d_tiling_data.h
 * \brief
 */

#ifndef CAUSAL_CONV1D_TILING_DATA_H_
#define CAUSAL_CONV1D_TILING_DATA_H_

#include <cstdint>


struct CausalConv1dTilingData {
    // input shapes
    int64_t dim;
    int64_t cuSeqlen;
    int64_t seqLen;
    int64_t batch;
    int64_t kernelWidth;
    int64_t stateLen;
    int64_t numCacheLines;

    // attributes
    int64_t activationMode;
    int64_t padSlotId;
    int64_t hasBias;

    // dim tiling
    int64_t baseDim;

    // token tiling (FN only)
    int64_t tokenBlockSize;
    int64_t tokenBlockCnt;

    // optional input flags
    int64_t hasNumAcceptedTokens;
    int64_t hasCacheIndices;
    int64_t hasInitialStateMode;
    int64_t hasInitStateWorkspace;

};
#endif // CAUSAL_CONV1D_TILING_DATA_H_
