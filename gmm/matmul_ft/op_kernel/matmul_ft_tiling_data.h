/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file matmul_ft_tiling_data.h
 * \brief tiling data struct
 */

#ifndef __MATMUL_FT_TILLING_DATA_H__
#define __MATMUL_FT_TILLING_DATA_H__

#include <cstdint>
#include <type_traits>

struct MatmulFtGemmCoordTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t k;
};

struct MatmulFtGemvCoordTilingData {
    uint32_t m;
    uint32_t n;
};

struct MatmulFtComputeTilingData {
    MatmulFtGemmCoordTilingData problemGemmShape;
    MatmulFtGemmCoordTilingData problemGemmShapeFirst;
    MatmulFtGemmCoordTilingData problemGemmShapeRemain;
    MatmulFtGemvCoordTilingData problemShape;
    MatmulFtGemvCoordTilingData problemShapeCol;
    MatmulFtGemvCoordTilingData problemCompShape;
    MatmulFtGemvCoordTilingData problemSliceShape;

    uint32_t totalInputElements;
    uint32_t totalOutputElements;
    uint32_t rowOutputElements;
    uint32_t colOutputElements;
    uint32_t xLen;
    uint32_t layoutThreLen;

    float roundingAlpha;
    float eMax;
    float stdEstARowRatio;
    float aRowScaleRatio;
    float stdEstRatios[2];
    float knRatios[2];
    float knScaleRatios[2];
    float knSqrtRatios[2];
    float kSqrtNRatios[2];
    float nScaleRatios[2];
    float nRatios[2];
    float nSqrtRatios[2];
    float nSquareRatios[2];

    uint32_t splitNNum;
    uint32_t splitReduceM;
    uint32_t splitReduceN;
    uint32_t splitKNum;
    uint8_t outputThre;
    uint8_t outputCE;
    uint8_t outputWorkspace;
};

struct MatmulFtBf16TilingPolicy {
    static constexpr uint32_t L1TileShape_N = 256;
    static constexpr uint32_t L1TileShape_K = 256;
    static constexpr uint32_t L0TileShape_K = 64;
    static constexpr uint32_t L1TileShapeBE_M = 256;
    static constexpr uint32_t L0TileShapeBE_M = 256;
    static constexpr uint32_t L1TileShapeFirst_K = 256;
    static constexpr uint32_t L0TileShapeFirst_K = 64;
    static constexpr uint32_t UBTileShapeforB_M = 48;
    static constexpr uint32_t UBTileShapeforA_M = 48;
    static constexpr uint32_t UBTileShapeforBRed_M = 8;
    static constexpr uint32_t UBTileShapeforBRed_N = 256;
};

struct MatmulFtFp32TilingPolicy {
    static constexpr uint32_t L1TileShape_N = 256;
    static constexpr uint32_t L1TileShape_K = 128;
    static constexpr uint32_t L0TileShape_K = 32;
    static constexpr uint32_t L1TileShapeBE_M = 128;
    static constexpr uint32_t L0TileShapeBE_M = 128;
    static constexpr uint32_t L1TileShapeFirst_K = 128;
    static constexpr uint32_t L0TileShapeFirst_K = 32;
    static constexpr uint32_t UBTileShapeforB_M = 64;
    static constexpr uint32_t UBTileShapeforA_M = 64;
    static constexpr uint32_t UBTileShapeforBRed_M = 8;
    static constexpr uint32_t UBTileShapeforBRed_N = 256;
};

struct MatmulFtFp16TilingPolicy {
    static constexpr uint32_t L1TileShape_N = 256;
    static constexpr uint32_t L1TileShape_K = 256;
    static constexpr uint32_t L0TileShape_K = 64;
    static constexpr uint32_t L1TileShapeBE_M = 256;
    static constexpr uint32_t L0TileShapeBE_M = 256;
    static constexpr uint32_t L1TileShapeFirst_K = 256;
    static constexpr uint32_t L0TileShapeFirst_K = 64;
    static constexpr uint32_t UBTileShapeforB_M = 64;
    static constexpr uint32_t UBTileShapeforA_M = 64;
    static constexpr uint32_t UBTileShapeforBRed_M = 8;
    static constexpr uint32_t UBTileShapeforBRed_N = 256;
};

struct MatmulFtTilingData {
    MatmulFtComputeTilingData compute;
};

static_assert(std::is_trivially_copyable<MatmulFtComputeTilingData>::value,
    "MatmulFtComputeTilingData must remain a plain tiling ABI type.");
static_assert(std::is_trivially_copyable<MatmulFtTilingData>::value,
    "MatmulFtTilingData must remain a plain tiling ABI type.");
#endif
