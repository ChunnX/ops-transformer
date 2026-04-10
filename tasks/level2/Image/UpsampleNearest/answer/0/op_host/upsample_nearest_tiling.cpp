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
 * \file upsample_nearest_tiling.cpp
 * \brief
 */

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "upsample_nearest_tiling.h"

namespace optiling {
constexpr int8_t NHWC_N_INDEX = 0;
constexpr int8_t NHWC_H_INDEX = 1;
constexpr int8_t NHWC_W_INDEX = 2;
constexpr int8_t NHWC_C_INDEX = 3;

constexpr int8_t NCHW_N_INDEX = 0;
constexpr int8_t NCHW_C_INDEX = 1;
constexpr int8_t NCHW_H_INDEX = 2;
constexpr int8_t NCHW_W_INDEX = 3;

constexpr int8_t OUT_H_INDEX = 0;
constexpr int8_t OUT_W_INDEX = 1;
constexpr int8_t OUT_L_INDEX = 0;

constexpr int8_t NLC_N_INDEX = 0;
constexpr int8_t NLC_L_INDEX = 1;
constexpr int8_t NLC_C_INDEX = 2;

constexpr uint32_t OUTPUT_SIZE_ATTR = 0;

constexpr uint32_t SCALE_H_ATTR = 1;
constexpr uint32_t SCALE_W_ATTR = 2;
constexpr uint32_t EXACT_ATTR = 3;
constexpr uint32_t SCALE_L_ATTR = 1;

constexpr uint32_t DATE_TYPE_FLOAT16 = 1;
constexpr uint32_t DATE_TYPE_FLOAT = 2;
constexpr uint32_t DATE_TYPE_HALF = 3;

constexpr uint64_t WORK_SPACE_SIZE = 32 * 1024 * 1024;
constexpr uint32_t BYTE_LEN_4 = 4;
constexpr uint32_t BYTE_LEN_2 = 2;

constexpr uint32_t NHWC_DIM_SIZE = 4;
constexpr uint32_t NLC_DIM_SIZE = 3;
constexpr uint32_t ADDR_ALIGN_SIZE = 512;
constexpr uint32_t COMMON_TILING_KEY = 1000;
constexpr uint32_t SMALL_CW_TILING_KEY = 1001;
constexpr uint32_t SMALL_C_TILING_KEY = 1002;
constexpr uint32_t SMALL_NCH_TILING_KEY = 1003;

constexpr uint32_t MAX_SMALL_SHPAE = 8192;
constexpr uint32_t MAX_SMALL_SACALE = 2;
constexpr uint8_t SCHEDULE_MODE = 1;

class UpsampleNearestTiling
{
public:
    explicit UpsampleNearestTiling(gert::TilingContext* context) : tilingContext(context){};
    ge::graphStatus RunBigKernelTiling();

private:
    ge::graphStatus ParseInputAttrs();
    inline float ComputeScaleValue(int64_t inputSize, int64_t outputSize, const float scale) const;
    void GetWorkSpace() const;
    uint32_t GetDataTypeVal() const;
    int64_t GetBestAvergingCols(uint32_t coreNumPlatform);
    uint32_t GetNeedCoreNum(uint32_t coreNumPlatform);
    void FillTilingData();
    void GetTilingKey();

    template <typename T1, typename T2>
    inline T1 CeilA2B(T1 a, T2 b) const;

    template <typename T1>
    inline int64_t Ceil(T1 x);

private:
    UpsampleNearestTilingData tilingData;
    gert::TilingContext* tilingContext = nullptr;
    ge::DataType dataType = ge::DT_UNDEFINED;
    ge::Format inputFormat = ge::Format::FORMAT_NHWC;
    uint8_t dim = 0;
    float realScaleH = 0.0f;
    float realScaleW = 0.0f;
    uint32_t tailColStartList[MAX_CORE_CONT] = {0};
    uint32_t tailColEndList[MAX_CORE_CONT] = {0};
    uint32_t tailRowStartList[MAX_CORE_CONT] = {0};
    uint32_t tailRowEndList[MAX_CORE_CONT] = {0};

    int64_t outputShapes[4] = {0};
    int64_t inputShapes[4] = {0};

    bool exactMode = true;
    uint32_t tilingKey = 1000;
    uint32_t needCoreNum = 1;
};

ge::graphStatus UpsampleNearestTiling::RunBigKernelTiling()
{
    if (ParseInputAttrs() == ge::GRAPH_FAILED) {
        return ge::GRAPH_FAILED;
    }

    GetTilingKey();

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    uint32_t totalCoreNum = ascendcPlatform.GetCoreNumAiv();
    needCoreNum = GetNeedCoreNum(totalCoreNum);
    GetWorkSpace();
    FillTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus UpsampleNearestTiling::ParseInputAttrs()
{
    const gert::RuntimeAttrs* attrs = tilingContext->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto srcShape = tilingContext->GetInputShape(0);
    dim = srcShape->GetStorageShape().GetDimNum();

    auto inputShape = srcShape->GetOriginShape();

    const gert::ContinuousVector* outputSizeAttr = attrs->GetAttrPointer<gert::ContinuousVector>(OUTPUT_SIZE_ATTR);
    const int64_t* outputSizeArray = reinterpret_cast<const int64_t*>(outputSizeAttr->GetData());

    exactMode = *(attrs->GetAttrPointer<bool>(EXACT_ATTR));

    for (int8_t i = 0; i < dim; i++) {
        inputShapes[i] = inputShape.GetDim(i);
        outputShapes[i] = inputShape.GetDim(i);
    }
    inputFormat = static_cast<ge::Format>(GetPrimaryFormat(tilingContext->GetInputDesc(0)->GetStorageFormat()));
    if (dim == NHWC_DIM_SIZE) {
        const float scaleH = *(attrs->GetAttrPointer<float>(SCALE_H_ATTR));
        const float scaleW = *(attrs->GetAttrPointer<float>(SCALE_W_ATTR));
        if (inputFormat == ge::Format::FORMAT_NCHW) {
            outputShapes[NCHW_H_INDEX] = outputSizeArray[OUT_H_INDEX];
            outputShapes[NCHW_W_INDEX] = outputSizeArray[OUT_W_INDEX];
            realScaleH = ComputeScaleValue(inputShapes[NCHW_H_INDEX], outputShapes[NCHW_H_INDEX], scaleH);
            realScaleW = ComputeScaleValue(inputShapes[NCHW_W_INDEX], outputShapes[NCHW_W_INDEX], scaleW);
        } else {
            outputShapes[NHWC_H_INDEX] = outputSizeArray[OUT_H_INDEX];
            outputShapes[NHWC_W_INDEX] = outputSizeArray[OUT_W_INDEX];
            realScaleH = ComputeScaleValue(inputShapes[NHWC_H_INDEX], outputShapes[NHWC_H_INDEX], scaleH);
            realScaleW = ComputeScaleValue(inputShapes[NHWC_W_INDEX], outputShapes[NHWC_W_INDEX], scaleW);
        }
    } else if (dim == NLC_DIM_SIZE) {
        inputShapes[NHWC_H_INDEX] = 1;
        inputShapes[NHWC_W_INDEX] = inputShape.GetDim(NLC_L_INDEX);
        inputShapes[NHWC_C_INDEX] = inputShape.GetDim(NLC_C_INDEX);
        outputShapes[NHWC_H_INDEX] = 1;
        outputShapes[NHWC_W_INDEX] = outputSizeArray[OUT_L_INDEX];
        outputShapes[NHWC_C_INDEX] = inputShapes[NLC_C_INDEX];
        const float scaleL = *(attrs->GetAttrPointer<float>(SCALE_H_ATTR));
        realScaleH = 1.0;
        realScaleW = ComputeScaleValue(inputShapes[NHWC_W_INDEX], outputShapes[NHWC_W_INDEX], scaleL);
    } else {
        return ge::GRAPH_FAILED;
    }

    auto srcDtype = tilingContext->GetInputDesc(0)->GetDataType();
    if (dataType == ge::DT_UNDEFINED) {
        dataType = srcDtype;
    } else if (srcDtype != dataType) {
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

void UpsampleNearestTiling::GetWorkSpace() const
{
    size_t* workspaces = tilingContext->GetWorkspaceSizes(1);
    workspaces[0] = WORK_SPACE_SIZE;
}

void UpsampleNearestTiling::GetTilingKey()
{
    if (inputFormat == ge::Format::FORMAT_NCHW) {
        tilingKey = SMALL_NCH_TILING_KEY;
    } else {
        int64_t inputC = inputShapes[NHWC_C_INDEX];
        int64_t outputH = inputShapes[NHWC_H_INDEX];
        int64_t outputW = inputShapes[NHWC_W_INDEX];
        int64_t inputW = outputShapes[NHWC_W_INDEX];
        if (inputC * inputW < MAX_SMALL_SHPAE && inputC * outputW < MAX_SMALL_SHPAE && outputH > 1) {
            tilingKey = SMALL_CW_TILING_KEY;
        } else if (inputC < MAX_SMALL_SHPAE && outputW < MAX_SMALL_SHPAE && realScaleW < MAX_SMALL_SACALE) {
            tilingKey = SMALL_C_TILING_KEY;
        } else {
            tilingKey = COMMON_TILING_KEY;
        }
    }
}

inline float UpsampleNearestTiling::ComputeScaleValue(int64_t inputSize, int64_t outputSize, const float scale) const
{
    if ((dim == NHWC_DIM_SIZE) && (inputSize == outputSize)) {
        return static_cast<float>(1);
    }
    if (scale > 0) {
        return scale;
    } else {
        return static_cast<float>(inputSize) / static_cast<float>(outputSize);
    }
}

template <typename T1, typename T2>
inline auto UpsampleNearestTiling::CeilA2B(T1 a, T2 b) const -> T1
{
    if (b != 0) {
        return (a + b - 1) / b;
    } else {
        return a;
    }
}

uint32_t UpsampleNearestTiling::GetDataTypeVal() const
{
    switch (dataType) {
        case ge::DT_FLOAT:
            return BYTE_LEN_4;
        case ge::DT_FLOAT16:
            return BYTE_LEN_2;
        case ge::DT_BF16:
            return BYTE_LEN_2;
        default:
            return 0;
    }
}

int64_t UpsampleNearestTiling::GetBestAvergingCols(uint32_t coreNumPlatform)
{
    int64_t outputH = outputShapes[NHWC_H_INDEX];
    int64_t outputW = outputShapes[NHWC_W_INDEX];
    if (inputFormat == ge::Format::FORMAT_NCHW) {
        outputH = outputShapes[NCHW_H_INDEX];
        outputW = outputShapes[NCHW_W_INDEX];
    }
    uint32_t dataTypeSize = GetDataTypeVal();
    int64_t minAvergingCols = dataTypeSize > 0 ? 32 / dataTypeSize : outputW;

    for (int64_t i = 1; i <= coreNumPlatform; i++) {
        if ((coreNumPlatform % i == 0) && (outputW % i == 0)) {
            int64_t j = coreNumPlatform / i;
            if (outputH % j == 0) {
                minAvergingCols = coreNumPlatform / i;
                break;
            }
        }
    }

    if (tilingKey == SMALL_CW_TILING_KEY) {
        minAvergingCols = outputW;
    }
    return minAvergingCols;
}

uint32_t UpsampleNearestTiling::GetNeedCoreNum(uint32_t coreNumPlatform)
{
    int64_t outputH = outputShapes[NHWC_H_INDEX];
    int64_t outputW = outputShapes[NHWC_W_INDEX];
    if (inputFormat == ge::Format::FORMAT_NCHW) {
        outputH = outputShapes[NCHW_H_INDEX];
        outputW = outputShapes[NCHW_W_INDEX];
    }
    int64_t realCoreNum = 0;
    int64_t slideSizeW = 0;
    int64_t slideSizeH = 0;
    int64_t colGroupNum = 0;
    int64_t groupRowCoreNum = 0;
    int64_t groupColCoreNum = 0;
    int64_t minAvergingCols = GetBestAvergingCols(coreNumPlatform);

    if (outputH < coreNumPlatform) {
        int64_t tailAvergingCols = std::max(CeilA2B(outputW, coreNumPlatform), minAvergingCols);
        colGroupNum = std::min(static_cast<int64_t>(coreNumPlatform), CeilA2B(outputW, tailAvergingCols));
        slideSizeW = tailAvergingCols;
    } else {
        colGroupNum = 1;
        slideSizeW = outputW;
    }
    groupColCoreNum = colGroupNum > 0 ? coreNumPlatform / colGroupNum : 0;
    int64_t row = groupColCoreNum > 0 ? outputH / groupColCoreNum : 0;
    int64_t tailAvergingRows = std::max(row, static_cast<int64_t>(1));
    groupRowCoreNum = std::min(groupColCoreNum, CeilA2B(outputH, tailAvergingRows));
    realCoreNum = colGroupNum * groupRowCoreNum;
    int64_t tailRowRemainder = outputH - groupRowCoreNum * tailAvergingRows;
    slideSizeH = tailAvergingRows;

    int64_t realNeedCoreNum = 0;
    int64_t tailRowOffset = 0;
    int64_t tempTailRowRemainder = tailRowRemainder;
    for (int64_t coreIndex = 0; coreIndex < realCoreNum; coreIndex++) {
        int64_t groupColIndex = groupRowCoreNum > 0 ? coreIndex / groupRowCoreNum : 0;
        tailColStartList[coreIndex] = groupColIndex * slideSizeW;
        tailColEndList[coreIndex] = std::min((groupColIndex + 1) * slideSizeW, outputW);
        int64_t groupRowIndex = groupRowCoreNum > 0 ? coreIndex % groupRowCoreNum : 0;
        if (groupRowIndex == 0) {
            tempTailRowRemainder = tailRowRemainder;
            tailRowOffset = 0;
        }
        tailRowStartList[coreIndex] = groupRowIndex * slideSizeH + tailRowOffset;
        if (tempTailRowRemainder > 0) {
            tempTailRowRemainder -= 1;
            tailRowOffset += 1;
        }
        tailRowEndList[coreIndex] = std::min((groupRowIndex + 1) * slideSizeH + tailRowOffset, outputH);
        realNeedCoreNum++;
    }
    realNeedCoreNum = realNeedCoreNum < 1 ? 1 : realNeedCoreNum;

    return realNeedCoreNum;
}

void UpsampleNearestTiling::FillTilingData()
{
    tilingData.set_dataType(GetDataTypeVal());
    tilingData.set_scaleH(realScaleH);
    tilingData.set_scaleW(realScaleW);
    tilingData.set_exactMode(exactMode);

    tilingData.set_needCoreNum(needCoreNum);

    tilingData.set_inputShapes(inputShapes);
    tilingData.set_outputShapes(outputShapes);

    tilingData.set_tailColStartList(tailColStartList);
    tilingData.set_tailColEndList(tailColEndList);
    tilingData.set_tailRowStartList(tailRowStartList);
    tilingData.set_tailRowEndList(tailRowEndList);

    tilingContext->SetBlockDim(needCoreNum);
    tilingContext->SetTilingKey(tilingKey);

    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
}

static ge::graphStatus tiling4UpsampleNearestTiling(gert::TilingContext* context)
{
    UpsampleNearestTiling tilingObject(context);
    return tilingObject.RunBigKernelTiling();
}

IMPL_OP_OPTILING(UpsampleNearest)
    .Tiling(tiling4UpsampleNearestTiling);

} // namespace optiling
