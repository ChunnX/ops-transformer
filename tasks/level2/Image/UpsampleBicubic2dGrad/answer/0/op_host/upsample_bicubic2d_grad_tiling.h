/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file upsample_bicubic2d_grad_tiling.h
 * \brief
 */
#ifndef OPS_BUILT_IN_OP_TILING_RUNTIME_UPSAMPLE_BICUBIC2D_GRAD_H
#define OPS_BUILT_IN_OP_TILING_RUNTIME_UPSAMPLE_BICUBIC2D_GRAD_H
#include "tiling/tiling_api.h"
#include "register/tilingdata_base.h"

namespace optiling {

constexpr uint16_t MAX_CORE_COUNT = 50;
constexpr uint32_t NUM_TWO = 2;
constexpr uint32_t NUM_THREE = 3;
constexpr uint32_t NUM_FOUR = 4;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t UB_CLEAR_SIZE = 128U * 1024U;
constexpr uint32_t NUM_FRACTAL = 16;
constexpr uint32_t NUM_PER_BLOCK_FLOAT16 = 16;
constexpr uint32_t NUM_PER_BLOCK_FLOAT32 = 8;
constexpr float THRESHOLD = 0.25;
constexpr float MAX_SCALE = 50;
constexpr uint8_t SCHEDULE_MODE = 1;

enum class UpsampleBicubic2dGradTilingKey : uint64_t {
    BASE_MODE = 10000001,
    DETERMINISTIC_MODE = 10000002,
    UNDFINED = 10000099
};

struct UpsampleBicubic2dGradParam {
    // platform
    uint64_t CoreNum;
    uint64_t UBSize;
    // attr
    bool alignCorners;
    float scalesH;
    float scalesW;

    bool deterministic;

    ge::DataType dataType;

    uint64_t batch;
    uint64_t inputN;
    uint64_t inputC;
    uint64_t inputH;
    uint64_t inputW;
    uint64_t outputH;
    uint64_t outputW;

    uint64_t tailH;
    uint64_t CoreNumH;
    uint64_t loopH;
    uint64_t loopTailCoreH;
    uint64_t innerCoreNumH;
    uint64_t innerBatchH;
    uint64_t innerBatchTailCoreH;

    uint64_t tailW;
    uint64_t CoreNumW;
    uint64_t loopW;
    uint64_t loopTailCoreW;
    uint64_t innerCoreNumW;
    uint64_t innerBatchW;
    uint64_t innerBatchTailCoreW;

    uint64_t clearBaseN;
    uint64_t clearInterLoop;
    uint64_t clearInterTailN;
    uint64_t clearInterTailCoreNum;
    uint64_t clearOutLoop;
    uint64_t clearOutTailN;
    uint64_t clearOutTailCoreNum;

    uint64_t baseNH;
    uint64_t baseNW;

    uint64_t slideSize;
    uint64_t singleCoreKW;
    uint64_t singleCoreKH;
    uint64_t radioMatrixSize;
    uint64_t intermediateMatrixSize;

    bool needExpandW;
    bool needExpandH;

    int64_t perCoreSlideNumW = 0;
    int64_t perCoreTailSlideNumW = 0;
    int64_t extraTailSlideCoreNumW = 0;
    int64_t perCoreSlideNumH = 0;
    int64_t perCoreTailSlideNumH = 0;
    int64_t extraTailSlideCoreNumH = 0;
};

BEGIN_TILING_DATA_DEF(UpsampleBicubic2dGradTilingData)

TILING_DATA_FIELD_DEF(uint32_t, dataType);
TILING_DATA_FIELD_DEF(uint32_t, CoreNum);
TILING_DATA_FIELD_DEF(uint32_t, ubSize);
TILING_DATA_FIELD_DEF(uint32_t, alignCorners);
TILING_DATA_FIELD_DEF(float, scalesH);
TILING_DATA_FIELD_DEF(float, scalesW);
TILING_DATA_FIELD_DEF(uint32_t, baseNH);
TILING_DATA_FIELD_DEF(uint32_t, baseNW);

TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, inputN);
TILING_DATA_FIELD_DEF(uint32_t, inputC);
TILING_DATA_FIELD_DEF(uint32_t, inputH);
TILING_DATA_FIELD_DEF(uint32_t, inputW);
TILING_DATA_FIELD_DEF(uint32_t, outputH);
TILING_DATA_FIELD_DEF(uint32_t, outputW);

TILING_DATA_FIELD_DEF(uint32_t, tailH);
TILING_DATA_FIELD_DEF(uint32_t, CoreNumH);
TILING_DATA_FIELD_DEF(uint32_t, loopH);
TILING_DATA_FIELD_DEF(uint32_t, loopTailCoreH);
TILING_DATA_FIELD_DEF(uint32_t, innerCoreNumH);
TILING_DATA_FIELD_DEF(uint32_t, innerBatchH);
TILING_DATA_FIELD_DEF(uint32_t, innerBatchTailCoreH);

TILING_DATA_FIELD_DEF(uint32_t, tailW);
TILING_DATA_FIELD_DEF(uint32_t, CoreNumW);
TILING_DATA_FIELD_DEF(uint32_t, loopW);
TILING_DATA_FIELD_DEF(uint32_t, loopTailCoreW);
TILING_DATA_FIELD_DEF(uint32_t, innerCoreNumW);
TILING_DATA_FIELD_DEF(uint32_t, innerBatchW);
TILING_DATA_FIELD_DEF(uint32_t, innerBatchTailCoreW);

TILING_DATA_FIELD_DEF(uint32_t, clearBaseN);
TILING_DATA_FIELD_DEF(uint32_t, clearInterLoop);
TILING_DATA_FIELD_DEF(uint32_t, clearInterTailN);
TILING_DATA_FIELD_DEF(uint32_t, clearInterTailCoreNum);
TILING_DATA_FIELD_DEF(uint32_t, clearOutLoop);
TILING_DATA_FIELD_DEF(uint32_t, clearOutTailN);
TILING_DATA_FIELD_DEF(uint32_t, clearOutTailCoreNum);

TILING_DATA_FIELD_DEF(uint32_t, slideSize);
TILING_DATA_FIELD_DEF(uint32_t, needExpandW);
TILING_DATA_FIELD_DEF(uint32_t, needExpandH);

TILING_DATA_FIELD_DEF(uint64_t, singleCoreKW);
TILING_DATA_FIELD_DEF(uint64_t, singleCoreKH);
TILING_DATA_FIELD_DEF(uint64_t, radioMatrixSize);
TILING_DATA_FIELD_DEF(uint64_t, intermediateMatrixSize);

TILING_DATA_FIELD_DEF(uint32_t, perCoreSlideNumW);
TILING_DATA_FIELD_DEF(uint32_t, perCoreTailSlideNumW);
TILING_DATA_FIELD_DEF(uint32_t, extraTailSlideCoreNumW);

TILING_DATA_FIELD_DEF(uint32_t, perCoreSlideNumH);
TILING_DATA_FIELD_DEF(uint32_t, perCoreTailSlideNumH);
TILING_DATA_FIELD_DEF(uint32_t, extraTailSlideCoreNumH);

TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, MMParamH);
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, MMParamW);
//

END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UpsampleBicubic2dGrad, UpsampleBicubic2dGradTilingData)

class UpsampleBicubic2dGradTiling {
public:
    ge::graphStatus runTiling(gert::TilingContext *context);

protected:
    uint32_t GetDtypeSize();
    uint32_t GetNumPerBlock();
    bool GetPlatformInfo(const gert::TilingContext *context);
    bool GetCheckAttr(const gert::TilingContext *context);
    bool CheckInOutShapes(const gert::TilingContext *context);
    bool GetMMTilingData(const gert::TilingContext *context);
    bool GetClearTilingData();
    bool GetTilingData(const gert::TilingContext *context);
    bool SetTilingData(gert::TilingContext *context);
    bool SetLaunchInfo(gert::TilingContext *context);
    bool IsDeterministicCalc(const gert::TilingContext * context);
    bool GetTilingDataDC(const gert::TilingContext *context);
    void CalcScales();
    void CalcNeedCoreNum();
    void CalcNeedCoreNumW();
    void CalcNeedCoreNumH();
    void CalcSingleCoreK();
    void CalcTCubeTiling(const gert::TilingContext *context);
    bool SetTilingDataDC(gert::TilingContext *context);
    bool SetLaunchInfoDC(gert::TilingContext *context);
    void TilingPrintParam(const gert::TilingContext* context);

private:
    UpsampleBicubic2dGradTilingData tilingData;
    UpsampleBicubic2dGradParam _Params;
    matmul_tiling::PlatformInfo platformInfo_;
};

}  // namespace optiling
#endif  // OPS_BUILT_IN_OP_TILING_RUNTIME_UPSAMPLE_BICUBIC2D_GRAD_H
