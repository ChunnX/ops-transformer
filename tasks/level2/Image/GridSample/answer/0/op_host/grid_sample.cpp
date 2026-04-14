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
 * \file grid_sample.cpp
 * \brief
 */
#include <cstring>
#include <limits>
#include <algorithm>
#include "register/op_def_registry.h"
#include "graph/utils/type_utils.h"
#include "tiling/platform/platform_ascendc.h"
#include "grid_sample_tiling.h"

#define OP_LOGD(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_LOGE(nodeName, fmt, ...) std::printf(fmt, ##__VA_ARGS__); std::printf("\n")
#define OP_CHECK_NULL_WITH_CONTEXT(context, ptr) \
    if ((ptr) == nullptr) { \
        std::printf("nullptr error!"); \
        return ge::GRAPH_FAILED; \
    }
#define unlikely(x) __builtin_expect((x), 0)
#define OP_CHECK_IF(condition, log, return_expr) \
    do { \
        if (unlikely(condition)) { \
            log; \
            return_expr; \
        } \
    } while (0)

namespace optiling {

constexpr uint64_t TILING_OFFSET = 1000000000000UL;

static const size_t DIM_NUM_4D = 4;
static const size_t DIM_NUM_5D = 5;
static const size_t DIM_2 = 2;
static const size_t DIM_3 = 3;
static const size_t DIM_4 = 4;
static const size_t INT_16 = 16;
static const size_t INT_22 = 22;
static const size_t INT_64 = 64;
static const size_t INT_88 = 88;
static const int64_t INTERPOLATION_MODE_BILINEAR = 0;
static const int64_t INTERPOLATION_MODE_NEAREST = 1;
static const int64_t INTERPOLATION_MODE_BICUBIC = 2;
static const int64_t PADDING_MODE_ZEROS = 0;
static const int64_t PADDING_MODE_BORDER = 1;
static const int64_t PADDING_MODE_REFLECTION = 2;
static const int64_t ALIGN_CORNERS_FALSE = 0;
static const int64_t ALIGN_CORNERS_TRUE = 1;
const static int64_t SIZE_16 = 16;
const static int64_t LENGTH_1024 = 1024;
const static int64_t FULL_LOAD_TYPE = 2;
const static int64_t X_MAX_HWC_FACTOR = 20480;
const static int64_t C1_X_COUNT = 4096;
const static int64_t NUM_C32 = 32;
const static int64_t MIN_HW_C32 = 8;
const static int64_t TEMPLATE_C32 = 2;
const static int64_t DOUBLE = 2;
constexpr uint8_t SCHEDULE_MODE = 1;
constexpr uint8_t BLOCK_NUM = 8;

enum class GridSampleDtypeKey : int32_t { FLOAT16 = 1, FLOAT32 = 2, BFLOAT16 = 3 };

static uint64_t ComputeTilingKey(int64_t interpolationMode, GridSampleDtypeKey dtypeKey,
    int64_t dimValue, int64_t schedulerMode, int64_t dimension, int64_t templateCNum, int64_t tempType)
{
    // Replicate GET_TILINGKEY logic: sum = a + 10*b + 100*c + 1000*d + 10000*e + 100000*f + 1000000*g
    uint64_t sum = static_cast<uint64_t>(interpolationMode)
        + 10ULL * static_cast<uint64_t>(dtypeKey)
        + 100ULL * static_cast<uint64_t>(dimValue)
        + 1000ULL * static_cast<uint64_t>(schedulerMode)
        + 10000ULL * static_cast<uint64_t>(dimension)
        + 100000ULL * static_cast<uint64_t>(templateCNum)
        + 1000000ULL * static_cast<uint64_t>(tempType);
    return sum % TILING_OFFSET;
}

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    GridSampleTilingData tilingData;

    // Get platform info
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto coreNumVar = ascendcPlatform.GetCoreNumAiv();
    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);

    // Get shape info
    auto inputX = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputX);
    auto inputXDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputXDesc);
    ge::DataType xDtype = inputXDesc->GetDataType();
    auto inputGrid = context->GetInputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputGrid);
    auto gridXDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, gridXDesc);
    ge::DataType gridDtype = gridXDesc->GetDataType();

    auto xShape = inputX->GetStorageShape();
    auto gridShape = inputGrid->GetStorageShape();

    OP_CHECK_IF((xShape.GetDimNum() != DIM_NUM_4D && xShape.GetDimNum() != DIM_NUM_5D) ||
                    (gridShape.GetDimNum() != DIM_NUM_4D && gridShape.GetDimNum() != DIM_NUM_5D),
        OP_LOGE(context->GetNodeName(), "x / grid shape length should be 4 or 5"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(xShape.GetDimNum() != gridShape.GetDimNum(),
        OP_LOGE(context->GetNodeName(), "x / grid shape length should be equal."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(gridShape.GetDim(0) != xShape.GetDim(0),
        OP_LOGE(context->GetNodeName(), "x / grid shape[0] should be same"),
        return ge::GRAPH_FAILED);

    int64_t dimension = 0;
    int64_t dimValue = 0;
    int64_t inN = 0, inC = 0, inD = 0, inH = 0, inW = 0;
    int64_t outD = 0, outH = 0, outW = 0;
    int64_t interpolationMode = 0;
    int64_t paddingMode = 0;
    int64_t alignCorners = 0;
    int64_t channelLast = 0;
    int64_t schedulerMode = 0;
    int64_t tempType = 1;
    int64_t templateCNum = 0;
    int64_t hwFactor = 512;

    if (xShape.GetDimNum() == DIM_NUM_5D) {
        dimension = 1;
        dimValue = gridShape.GetDim(DIM_4);
        OP_CHECK_IF(dimValue != DIM_3,
            OP_LOGE(context->GetNodeName(), "only support (N, D, H, W, 3) for grid"),
            return ge::GRAPH_FAILED);
    } else {
        dimension = 0;
        dimValue = gridShape.GetDim(DIM_3);
        OP_CHECK_IF(dimValue != DIM_2,
            OP_LOGE(context->GetNodeName(), "only support (N, H, W, 2) for grid"),
            return ge::GRAPH_FAILED);
    }

    OP_CHECK_IF((xDtype != ge::DT_FLOAT && xDtype != ge::DT_FLOAT16 && xDtype != ge::DT_BF16),
        OP_LOGE(context->GetNodeName(), "x datatype only support FLOAT32, FLOAT16, BFLOAT16"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF((gridDtype != ge::DT_FLOAT && gridDtype != ge::DT_FLOAT16 && gridDtype != ge::DT_BF16),
        OP_LOGE(context->GetNodeName(), "grid datatype only support FLOAT32, FLOAT16, BFLOAT16"),
        return ge::GRAPH_FAILED);

    // Get attrs
    auto *attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const char *pInterpolationMode = attrs->GetAttrPointer<char>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, pInterpolationMode);
    if (strcmp(pInterpolationMode, "bilinear") == 0) {
        interpolationMode = INTERPOLATION_MODE_BILINEAR;
    } else if (strcmp(pInterpolationMode, "bicubic") == 0) {
        interpolationMode = INTERPOLATION_MODE_BICUBIC;
    } else if (strcmp(pInterpolationMode, "nearest") == 0) {
        interpolationMode = INTERPOLATION_MODE_NEAREST;
    } else {
        OP_LOGE(context->GetNodeName(), "interpolation_mode only support bilinear or nearest or bicubic.");
        return ge::GRAPH_FAILED;
    }

    OP_CHECK_IF(dimension == 1 && interpolationMode == INTERPOLATION_MODE_BICUBIC,
        OP_LOGE(context->GetNodeName(), "GridSampler3D interpolation_mode only support bilinear or nearest"),
        return ge::GRAPH_FAILED);

    const char *pPaddingMode = attrs->GetAttrPointer<char>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, pPaddingMode);
    if (strcmp(pPaddingMode, "zeros") == 0) {
        paddingMode = PADDING_MODE_ZEROS;
    } else if (strcmp(pPaddingMode, "border") == 0) {
        paddingMode = PADDING_MODE_BORDER;
    } else if (strcmp(pPaddingMode, "reflection") == 0) {
        paddingMode = PADDING_MODE_REFLECTION;
    } else {
        OP_LOGE(context->GetNodeName(), "padding_mode only support zeros or border or reflection.");
        return ge::GRAPH_FAILED;
    }

    const bool *pAlignCorners = attrs->GetAttrPointer<bool>(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, pAlignCorners);
    alignCorners = ALIGN_CORNERS_FALSE;
    if (*pAlignCorners) {
        alignCorners = ALIGN_CORNERS_TRUE;
    }

    const bool *pChannelLast = attrs->GetAttrPointer<bool>(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, pChannelLast);
    channelLast = 0;
    if (*pChannelLast) {
        channelLast = 1;
    }

    inN = xShape.GetDim(0);
    if (dimension == 0) {
        if (channelLast == 0) {
            inC = xShape.GetDim(1);
            inH = xShape.GetDim(DIM_2);
            inW = xShape.GetDim(DIM_3);
        } else {
            inH = xShape.GetDim(1);
            inW = xShape.GetDim(DIM_2);
            inC = xShape.GetDim(DIM_3);
        }
        outH = gridShape.GetDim(1);
        outW = gridShape.GetDim(DIM_2);

        if ((channelLast == 1) && (strcmp(pInterpolationMode, "bilinear") == 0) &&
            (inC * inH * inW <= X_MAX_HWC_FACTOR)) {
            tempType = FULL_LOAD_TYPE;
            hwFactor = 1024;
            auto gridSampleSocVersion = ascendcPlatform.GetSocVersion();
            if ((outH * outW < BLOCK_NUM) && (inN < static_cast<int64_t>(coreNumVar) * BLOCK_NUM) &&
                (gridSampleSocVersion == platform_ascendc::SocVersion::ASCEND310P)) {
                context->SetScheduleMode(SCHEDULE_MODE);
            }
            if ((inC == 1) && (inH * inW < C1_X_COUNT)) {
                templateCNum = 1;
            } else if ((inC == NUM_C32) && (inH > MIN_HW_C32) && (inW > MIN_HW_C32)) {
                templateCNum = TEMPLATE_C32;
            } else {
                templateCNum = 0;
            }
        }

        OP_CHECK_IF(inN < 1 || inC < 1 || inH < 1 || inW < 1 || outW < 1 || outH < 1,
            OP_LOGE(context->GetNodeName(), "Invalid shape. Maybe empty tensor."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(inH * inW > static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
            OP_LOGE(context->GetNodeName(), "no support for H*W of x greater than int32 max value"),
            return ge::GRAPH_FAILED);

        const int64_t *pSchedulerMode = attrs->GetAttrPointer<int64_t>(4);
        OP_CHECK_NULL_WITH_CONTEXT(context, pSchedulerMode);
        OP_LOGD(context->GetNodeName(), "scheduler_mode is: %ld", *pSchedulerMode);
        schedulerMode = *pSchedulerMode;
        OP_CHECK_IF(schedulerMode != 0 && schedulerMode != 1,
            OP_LOGE(context->GetNodeName(), "scheduler_mode only support 0 or 1."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(!(*pChannelLast) && schedulerMode == 1,
            OP_LOGE(context->GetNodeName(), "scheduler_mode support 1 only in the channel last scenario."),
            return ge::GRAPH_FAILED);
    } else {
        if (channelLast == 0) {
            inC = xShape.GetDim(1);
            inD = xShape.GetDim(DIM_2);
            inH = xShape.GetDim(DIM_3);
            inW = xShape.GetDim(DIM_4);
        } else {
            inD = xShape.GetDim(1);
            inH = xShape.GetDim(DIM_2);
            inW = xShape.GetDim(DIM_3);
            inC = xShape.GetDim(DIM_4);
        }
        outD = gridShape.GetDim(1);
        outH = gridShape.GetDim(DIM_2);
        outW = gridShape.GetDim(DIM_3);

        OP_CHECK_IF(inN < 1 || inC < 1 || inD < 1 || inH < 1 || inW < 1 || outD < 1 || outW < 1 || outH < 1,
            OP_LOGE(context->GetNodeName(), "Invalid shape. Maybe empty tensor."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(inH * inW * inD > static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
            OP_LOGE(context->GetNodeName(), "no support for D*H*W of x greater than int32 max value"),
            return ge::GRAPH_FAILED);

        // Special case: portrait mode for 3D
        if (inN == gridShape.GetDim(0) && inD == outD && inH == outH && inW == outW && inD == INT_16 && inH == INT_64 &&
            inW == INT_64 && dimValue == DIM_3 && inC == DIM_4 && (inN == INT_22 || inN == INT_88)) {
            schedulerMode = 1;
        }
    }

    // Compute tiling data
    tilingData.set_coreNumVar(coreNumVar);
    tilingData.set_inN(inN);
    tilingData.set_inC(inC);
    tilingData.set_inD(inD);
    tilingData.set_inH(inH);
    tilingData.set_inW(inW);
    tilingData.set_outD(outD);
    tilingData.set_outH(outH);
    tilingData.set_outW(outW);
    tilingData.set_interpolationMode(interpolationMode);
    tilingData.set_paddingMode(paddingMode);
    tilingData.set_alignCorners(alignCorners);
    tilingData.set_channelLast(channelLast);

    int64_t outHW = outH * outW;
    int64_t needCoreNum = coreNumVar;
    if (inN < coreNumVar && outHW <= hwFactor) {
        needCoreNum = inN;
    }
    tilingData.set_needCoreNum(needCoreNum);

    // Compute workspace size
    uint64_t workspaceSize = SIZE_16 * LENGTH_1024 * LENGTH_1024;
    if (xDtype == ge::DT_FLOAT16 || xDtype == ge::DT_BF16) {
        size_t outputShapeSize =
            static_cast<size_t>(needCoreNum) * static_cast<size_t>(inC) * static_cast<size_t>(hwFactor) * sizeof(float);
        workspaceSize = workspaceSize + outputShapeSize;
    }
    if (tempType == FULL_LOAD_TYPE) {
        workspaceSize = workspaceSize * DOUBLE;
    }

    // Set tiling key
    GridSampleDtypeKey dtypeKey = GridSampleDtypeKey::FLOAT32;
    if (xDtype == ge::DT_FLOAT16) {
        dtypeKey = GridSampleDtypeKey::FLOAT16;
    } else if (xDtype == ge::DT_BF16) {
        dtypeKey = GridSampleDtypeKey::BFLOAT16;
    }

    uint64_t tilingKey = ComputeTilingKey(interpolationMode, dtypeKey, dimValue, schedulerMode, dimension, templateCNum, tempType);
    OP_LOGD(context->GetNodeName(), "schedulerMode:%ld,tilingKey:%zu.", schedulerMode, tilingKey);
    context->SetTilingKey(tilingKey);

    // Set block dim
    context->SetBlockDim(needCoreNum);

    // Set workspace
    size_t *workspaces = context->GetWorkspaceSizes(1);
    workspaces[0] = workspaceSize;

    // Save tiling data
    gert::TilingData *rawTilingData = context->GetRawTilingData();
    OP_CHECK_IF(rawTilingData == nullptr,
        OP_LOGE(context->GetNodeName(), "GetRawTilingData failed."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(tilingData.GetDataSize() > rawTilingData->GetCapacity(),
        OP_LOGE(context->GetNodeName(),
            "actual tiling data size %zu > context tiling data size %zu",
            tilingData.GetDataSize(), rawTilingData->GetCapacity()),
        return ge::GRAPH_FAILED);
    tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
    rawTilingData->SetDataSize(tilingData.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

} // namespace optiling


namespace ops {
class GridSample : public OpDef {
public:
    explicit GridSample(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("grid")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Attr("interpolation_mode").AttrType(OPTIONAL).String("bilinear");
        this->Attr("padding_mode").AttrType(OPTIONAL).String("zeros");
        this->Attr("align_corners").AttrType(OPTIONAL).Bool(false);
        this->Attr("channel_last").AttrType(OPTIONAL).Bool(false);
        this->Attr("scheduler_mode").AttrType(OPTIONAL).Int(1);

        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
        this->AICore().AddConfig("ascend910_93");
    }
};

OP_ADD(GridSample);
}  // namespace ops
