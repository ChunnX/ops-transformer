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
 * \file upsample_nearest3d_tiling_common.h
 * \brief
 */

#ifndef UPSAMPLE_NEAREST3D_TILING_COMMON_H
#define UPSAMPLE_NEAREST3D_TILING_COMMON_H

#include "../op_kernel/upsample_nearest3d_struct.h"
#include "tiling/platform/platform_ascendc.h"

namespace UpsampleNearest3d {

class UpsampleNearest3dCommon {
public:
    constexpr static int64_t BEST_PERFORMANCE_SIZE_1 = 256;
    constexpr static int64_t BEST_PERFORMANCE_SIZE_2 = 768;
    constexpr static int64_t BEST_PERFORMANCE_SIZE_3 = 1536;
    constexpr static int64_t BEST_PERFORMANCE_SIZE_4 = 2048;

    constexpr static float BEST_PERFORMANCE_SCALE_1 = 100.0f;
    constexpr static float BEST_PERFORMANCE_SCALE_2 = 24.0f;
    constexpr static float BEST_PERFORMANCE_SCALE_3 = 10.0f;
    constexpr static float BEST_PERFORMANCE_SCALE_4 = 6.0f;

    constexpr static float ZERO_FLOAT = 0.0f;
    constexpr static float ONE_FLOAT = 1.0f;

    constexpr static int64_t RESERVED_LENGTH = 4;

    constexpr static uint8_t HALF_TYPE = 1;
    constexpr static uint8_t FLOAT_TYPE = 2;
    constexpr static uint8_t BFLOAT_TYPE = 3;

    constexpr static uint8_t BATCH_DIM = 2;
    constexpr static uint8_t DIM = 3;
    constexpr static uint8_t D_INDEX = 0;
    constexpr static uint8_t H_INDEX = 1;
    constexpr static uint8_t W_INDEX = 2;

    constexpr static size_t SHAPE_SIZE = 3;

    constexpr static int64_t SMALL_W = 64;

    template <typename T>
    static void UpsampleNearest3dCommonTiling(
        T x, float* scales, const int64_t* outputShape, UpsampleNearest3dTilingData& tilingData, uint32_t coreNum)
    {
        // getShape
        int64_t batches = 1;
        int64_t inputShapes[3] = {0};
        int64_t outputShapes[3] = {0};

        for (uint8_t i = 0; i < BATCH_DIM; i++) {
            batches *= x.GetDim(i);
        }
        for (uint8_t i = 0; i < DIM; i++) {
            inputShapes[i] = x.GetDim(i + BATCH_DIM);
            outputShapes[i] = outputShape[i];
        }
        // getScale
        float realScaleD = ComputeScaleValue(inputShapes[D_INDEX], outputShapes[D_INDEX], scales[D_INDEX]);
        float realScaleH = ComputeScaleValue(inputShapes[H_INDEX], outputShapes[H_INDEX], scales[H_INDEX]);
        float realScaleW = ComputeScaleValue(inputShapes[W_INDEX], outputShapes[W_INDEX], scales[W_INDEX]);

        // getSlideSizeW
        int64_t slideSizeW = BEST_PERFORMANCE_SIZE_1;
        if (realScaleW <= BEST_PERFORMANCE_SCALE_4) {
            slideSizeW = BEST_PERFORMANCE_SIZE_4;
        } else if (realScaleW <= BEST_PERFORMANCE_SCALE_3) {
            slideSizeW = BEST_PERFORMANCE_SIZE_3;
        } else if (realScaleW <= BEST_PERFORMANCE_SCALE_2) {
            slideSizeW = BEST_PERFORMANCE_SIZE_2;
        } else {
            slideSizeW = BEST_PERFORMANCE_SIZE_1;
        }

        tilingData.batches = batches;
        for (uint8_t i = 0; i < DIM; i++) {
            tilingData.inputShapes[i] = inputShapes[i];
            tilingData.outputShapes[i] = outputShapes[i];
        }
        tilingData.scaleD = realScaleD;
        tilingData.scaleH = realScaleH;
        tilingData.scaleW = realScaleW;
        tilingData.slideSizeW = slideSizeW;

        size_t inputShapeSize = sizeof(inputShapes) / sizeof(inputShapes[0]);
        size_t outputShapeSize = sizeof(outputShapes) / sizeof(outputShapes[0]);

        // GetNeedCoreNum
        GetTensorSize(inputShapes, inputShapeSize, outputShapes, outputShapeSize, tilingData);
        GetNeedCoreNum(static_cast<int64_t>(coreNum), outputShapes, tilingData);

        tilingData.isView1DAndSmallW = false;
        if (GetISView1DAndSmallW(tilingData)) {
            tilingData.isView1DAndSmallW = true;
            tilingData.needCoreNum = static_cast<int64_t>(coreNum);
            tilingData.tailAvergingRow = CeilA2B(tilingData.inputRow, static_cast<int64_t>(coreNum));
            tilingData.needCoreNum = CeilA2B(tilingData.inputRow, tilingData.tailAvergingRow);
        }
    }

    static float ComputeScaleValue(int64_t inSize, int64_t outSize, float scale)
    {
        if (scale > ZERO_FLOAT) {
            return scale;
        } else {
            return outSize != 0 ? (static_cast<float>(inSize) / outSize) : ZERO_FLOAT;
        }
    }

    static void GetTensorSize(
        int64_t* inputShapes, size_t inputShapeSize, const int64_t* outputShapes, size_t outputShapeSize,
        UpsampleNearest3dTilingData& tilingData)
    {
        if (inputShapeSize < SHAPE_SIZE || outputShapeSize < SHAPE_SIZE) {
            return;
        }

        float realScaleD = tilingData.scaleD;
        float realScaleH = tilingData.scaleH;

        int64_t slideNumH = outputShapes[H_INDEX];
        int64_t tensorSizeH = 1;
        if (realScaleH > ZERO_FLOAT && realScaleH < ONE_FLOAT) {
            slideNumH = inputShapes[H_INDEX];
            tensorSizeH = RESERVED_LENGTH;
        }

        int64_t slideNumD = outputShapes[D_INDEX];
        int64_t tensorSizeD = 1;
        if (realScaleD > ZERO_FLOAT && realScaleD < ONE_FLOAT) {
            slideNumD = inputShapes[D_INDEX];
            tensorSizeD = RESERVED_LENGTH;
        }

        tilingData.slideNumH = slideNumH;
        tilingData.slideNumD = slideNumD;
        tilingData.tensorSizeH = tensorSizeH;
        tilingData.tensorSizeD = tensorSizeD;
    }

    static void GetNeedCoreNum(int64_t coreNumPlatform, int64_t* outputShapes, UpsampleNearest3dTilingData& tilingData)
    {
        float realScaleW = tilingData.scaleW;
        int64_t batches = tilingData.batches;
        int64_t slideNumH = tilingData.slideNumH;
        int64_t slideNumD = tilingData.slideNumD;
        int64_t slideSizeW = tilingData.slideSizeW;

        int64_t slideNumW = CeilA2B(outputShapes[W_INDEX], slideSizeW);
        int64_t tensorSizeW = Ceil(slideSizeW * std::min(realScaleW, BEST_PERFORMANCE_SCALE_1)) + RESERVED_LENGTH;

        int64_t slideNum = slideNumW * slideNumH * slideNumD;
        int64_t eachCoreSlideNum = coreNumPlatform > 0 ? slideNum / coreNumPlatform : 0;
        int64_t remainder = coreNumPlatform > 0 ? slideNum % coreNumPlatform : 0;
        int64_t inputRow = batches;
        int64_t groupCoreNum = coreNumPlatform;
        int64_t tailAvergingRow = 1;
        if (remainder > 0) {
            groupCoreNum = coreNumPlatform / remainder;
            tailAvergingRow = CeilA2B(inputRow, groupCoreNum);
            groupCoreNum = std::min(groupCoreNum, CeilA2B(inputRow, tailAvergingRow));
        }

        int64_t needCoreNum = coreNumPlatform;
        if (eachCoreSlideNum == 0 && remainder > 0) {
            needCoreNum = remainder * groupCoreNum;
        }

        tilingData.tensorSizeW = tensorSizeW;
        tilingData.eachCoreSlideNum = eachCoreSlideNum;
        tilingData.remainder = remainder;
        tilingData.tailStartSlideNum = eachCoreSlideNum * coreNumPlatform;
        tilingData.groupCoreNum = groupCoreNum;
        tilingData.inputRow = inputRow;
        tilingData.tailAvergingRow = tailAvergingRow;
        tilingData.needCoreNum = needCoreNum;
    }

    static bool GetISView1DAndSmallW(UpsampleNearest3dTilingData& tilingData)
    {
        if (tilingData.inputShapes[D_INDEX] != 1 || tilingData.outputShapes[D_INDEX] != 1) {
            return false;
        }
        if (tilingData.inputShapes[H_INDEX] != 1 || tilingData.outputShapes[H_INDEX] != 1) {
            return false;
        }
        if (tilingData.scaleW > BEST_PERFORMANCE_SCALE_1) {
            return false;
        }
        if (tilingData.outputShapes[W_INDEX] > SMALL_W) {
            return false;
        }
        return true;
    }

    template <typename T1, typename T2>
    static auto CeilA2B(T1 a, T2 b) -> T1
    {
        if (b != 0) {
            return (a + b - 1) / b;
        } else {
            return a;
        }
    }

    template <typename T1>
    static int32_t Ceil(T1 x)
    {
        int32_t floorX = int32_t(x);
        if (FloatEqual(x, floorX)) {
            return floorX;
        }
        return floorX + 1;
    }

    static bool FloatEqual(float a, float b)
    {
        float closeTo0 = float(1e-6);
        if (a > b) {
            return a - b < closeTo0;
        } else {
            return b - a < closeTo0;
        }
    }
};

} // namespace UpsampleNearest3d

// Out-of-class definitions for constexpr static members (required for ODR-use in C++14)
constexpr int64_t UpsampleNearest3d::UpsampleNearest3dCommon::BEST_PERFORMANCE_SIZE_1;
constexpr int64_t UpsampleNearest3d::UpsampleNearest3dCommon::BEST_PERFORMANCE_SIZE_2;
constexpr int64_t UpsampleNearest3d::UpsampleNearest3dCommon::BEST_PERFORMANCE_SIZE_3;
constexpr int64_t UpsampleNearest3d::UpsampleNearest3dCommon::BEST_PERFORMANCE_SIZE_4;
constexpr float UpsampleNearest3d::UpsampleNearest3dCommon::BEST_PERFORMANCE_SCALE_1;
constexpr float UpsampleNearest3d::UpsampleNearest3dCommon::BEST_PERFORMANCE_SCALE_2;
constexpr float UpsampleNearest3d::UpsampleNearest3dCommon::BEST_PERFORMANCE_SCALE_3;
constexpr float UpsampleNearest3d::UpsampleNearest3dCommon::BEST_PERFORMANCE_SCALE_4;
constexpr float UpsampleNearest3d::UpsampleNearest3dCommon::ZERO_FLOAT;
constexpr float UpsampleNearest3d::UpsampleNearest3dCommon::ONE_FLOAT;
constexpr int64_t UpsampleNearest3d::UpsampleNearest3dCommon::RESERVED_LENGTH;
constexpr int64_t UpsampleNearest3d::UpsampleNearest3dCommon::SMALL_W;

#endif // UPSAMPLE_NEAREST3D_TILING_COMMON_H
