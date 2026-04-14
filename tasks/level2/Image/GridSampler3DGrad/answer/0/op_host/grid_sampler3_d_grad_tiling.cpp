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
 * \file grid_sampler3_d_grad_tiling.cpp
 * \brief
 */

#include <iostream>
#include <map>
#include <vector>
#include <string>
#include "grid_sampler3_d_grad_tiling.h"
#include "register/op_def_registry.h"

namespace optiling {
constexpr uint32_t RESERVED_UB = static_cast<uint32_t>(8 * 1024);
constexpr uint32_t DIM_LEN = 5;
constexpr uint32_t GRAD_INPUT_INDEX = 0;
constexpr uint32_t X_INPUT_INDEX = 1;
constexpr uint32_t GRID_INPUT_INDEX = 2;
constexpr uint32_t DIM_INDEX0 = 0;
constexpr uint32_t DIM_INDEX1 = 1;
constexpr uint32_t DIM_INDEX2 = 2;
constexpr uint32_t DIM_INDEX3 = 3;
constexpr uint32_t DIM_INDEX4 = 4;
constexpr uint32_t FP32_BLOCK_NUM = 8;
constexpr uint32_t FP32_GROUP_SIZE = 8;
constexpr uint32_t DTYPE_SIZE_FLOAT = 4;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t CHANNEL_256 = 256;
constexpr uint32_t CHANNEL_512 = 512;
constexpr uint32_t CHANNEL_1024 = 1024;
constexpr uint32_t CONST_TWO = 2;
constexpr uint32_t FP32_GROUP_SIZE_LT_256 = 32;
constexpr uint32_t FP32_GROUP_SIZE_GT_256_LT_512 = 16;
constexpr uint32_t FP32_GROUP_SIZE_GT_512_LT_1024 = 8;
constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t ALIGN_256_BYTES = 256;
constexpr uint32_t BILINEAR_DIVIDE_UB_NUM = 180;
constexpr uint32_t NEAREST_DIVIDE_UB_NUM = 30;
constexpr uint32_t CONST_SEVENTEEN = 17;
constexpr uint32_t WORKSPACE_SIZE = static_cast<uint32_t>(16 * 1024 * 1024);
constexpr uint32_t CHECK_DIM_NUM = 5;
constexpr uint32_t INTERPOLATION_MODE_INDEX = 0;
constexpr uint32_t PADDING_MODE_INDEX = 1;
constexpr uint32_t ALIGN_CORNERS_INDEX = 2;
constexpr uint32_t INTERPOLATION_MODE_BILINEAR = 0;
constexpr uint32_t INTERPOLATION_MODE_NEAREST = 1;
constexpr uint32_t PADDING_MODE_ZEROS = 0;
constexpr uint32_t PADDING_MODE_BORDER = 1;
constexpr uint32_t PADDING_MODE_REFLECTION = 2;
constexpr uint32_t ALIGN_CORNERS_FALSE = 0;
constexpr uint32_t ALIGN_CORNERS_TRUE = 1;

template <typename TilingData>
class GridSampler3DGradTiling {
public:
    explicit GridSampler3DGradTiling(
        InputParamsInfo& param, const uint32_t inputCoreNum, const uint32_t inputUbSize, const uint32_t deterministic)
    {
        this->coreNum = inputCoreNum;
        this->batch = param.batch;
        this->channel = param.channel;
        this->xD = param.xD;
        this->xH = param.xH;
        this->xW = param.xW;
        this->gridD = param.gridD;
        this->gridH = param.gridH;
        this->gridW = param.gridW;
        this->interpolation = param.interpolation;
        this->padding = param.padding;
        this->alignCorners = param.alignCorners;
        this->tilingKey = param.tilingKey;
        this->ubSize = FloorAlign(inputUbSize, BYTE_BLOCK);
        this->isDeterministic = deterministic;
        return;
    }

    void GetTiling(TilingData* tilingData);

private:
    void GetUsedCore();
    void SplitUb();
    void FillTilingData(TilingData* tilingData);

    template <typename T1, typename T2>
    inline auto FloorDiv(T1 a, T2 b)
    {
        if (b == 0) {
            throw std::invalid_argument("FloorDiv Division by zero");
        }
        return (a) / (b);
    }
    template <typename T1, typename T2>
    inline auto CeilAlign(T1 a, T2 b)
    {
        if (b == 0) {
            throw std::invalid_argument("CeilAlign Division by zero");
        }
        return (a + b - 1) / b * b;
    }
    template <typename T1, typename T2>
    inline auto FloorAlign(T1 a, T2 b)
    {
        if (b == 0) {
            throw std::invalid_argument("FloorAlign Division by zero");
        }
        return (a) / b * b;
    }
    inline uint32_t Ceil(uint32_t a, uint32_t b)
    {
        if (b == 0) {
            throw std::invalid_argument("Ceil Division by zero");
        }
        uint32_t tmp = a % b;
        if (tmp > 0) {
            return a / b + 1;
        } else {
            return a / b;
        }
    }

private:
    uint32_t batch = 0;
    uint32_t channel = 0;
    uint32_t xD = 0;
    uint32_t xH = 0;
    uint32_t xW = 0;
    uint32_t gridD = 0;
    uint32_t gridH = 0;
    uint32_t gridW = 0;
    uint32_t interpolation = 0;
    uint32_t padding = 0;
    uint32_t alignCorners = 0;
    uint32_t usedCoreNum = 0;
    uint32_t tailPNum = 0;
    uint32_t coreNum = 0;
    uint32_t group = 0;
    uint32_t ubSize = 0;
    uint32_t pNumPerCore = 0;
    uint32_t tilingKey = 1;
    uint32_t divideUbNum = 1;
    uint32_t extraUbSize = 0;
    uint32_t ubFactorElement = 0;
    uint32_t isDeterministic = 0;
    uint32_t tailBNum = 0;
};

template <typename TilingData>
void GridSampler3DGradTiling<TilingData>::GetUsedCore()
{
    if (isDeterministic == 0) {
        uint64_t mulNDHW = static_cast<uint64_t>(batch * gridD * gridH * gridW);
        if (mulNDHW <= coreNum) {
            usedCoreNum = mulNDHW;
            pNumPerCore = static_cast<uint32_t>(1);
            tailPNum = static_cast<uint32_t>(0);
            return;
        }
        usedCoreNum = coreNum;
        pNumPerCore = FloorDiv(mulNDHW, coreNum);
        tailPNum = static_cast<uint32_t>(mulNDHW % usedCoreNum);
    } else {
        if (batch > coreNum) {
            usedCoreNum = coreNum;
            uint32_t bNumPerCore = FloorDiv(batch, usedCoreNum);
            tailBNum = static_cast<uint32_t>(batch % usedCoreNum);
            pNumPerCore = gridD * gridH * gridW * bNumPerCore;
            return;
        }
        usedCoreNum = batch;
        pNumPerCore = gridD * gridH * gridW;
    }
}

template <typename TilingData>
void GridSampler3DGradTiling<TilingData>::SplitUb()
{
    uint32_t alignChannel = 0;
    alignChannel = CeilAlign(channel, FP32_BLOCK_NUM);

    if (static_cast<int32_t>(interpolation) == 0) { // bilinear
        divideUbNum = BILINEAR_DIVIDE_UB_NUM;
        extraUbSize = CONST_SEVENTEEN * alignChannel * DTYPE_SIZE_FLOAT;
        group = static_cast<uint32_t>(1);
    } else if (static_cast<int32_t>(interpolation) == 1) { // nearest
        divideUbNum = NEAREST_DIVIDE_UB_NUM;
        if (channel <= CHANNEL_256) {
            extraUbSize = BUFFER_NUM * (FP32_GROUP_SIZE_LT_256 + 1) * alignChannel * DTYPE_SIZE_FLOAT;
            group = FP32_GROUP_SIZE_LT_256;
        } else if (channel <= CHANNEL_512) {
            extraUbSize = BUFFER_NUM * (FP32_GROUP_SIZE_GT_256_LT_512 + 1) * alignChannel * DTYPE_SIZE_FLOAT;
            group = FP32_GROUP_SIZE_GT_256_LT_512;
        } else if (channel <= CHANNEL_1024) {
            extraUbSize = BUFFER_NUM * (FP32_GROUP_SIZE_GT_512_LT_1024 + 1) * alignChannel * DTYPE_SIZE_FLOAT;
            group = FP32_GROUP_SIZE_GT_512_LT_1024;
        } else {
            extraUbSize = BUFFER_NUM * CONST_TWO * alignChannel * DTYPE_SIZE_FLOAT;
            group = static_cast<uint32_t>(1);
        }
    }
    uint32_t tilingDataSize = CeilAlign(sizeof(TilingData), BYTE_BLOCK);
    uint32_t canUseUbSize = FloorAlign(ubSize - tilingDataSize, BYTE_BLOCK);
    if (canUseUbSize <= extraUbSize) {
        ubFactorElement = static_cast<uint32_t>(0);
        return;
    }
    ubFactorElement = FloorAlign((canUseUbSize - extraUbSize) / divideUbNum, ALIGN_256_BYTES) / DTYPE_SIZE_FLOAT;
}

template <typename TilingData>
void GridSampler3DGradTiling<TilingData>::FillTilingData(TilingData* tilingData)
{
    tilingData->set_batch(batch);
    tilingData->set_channel(channel);
    tilingData->set_xD(xD);
    tilingData->set_xH(xH);
    tilingData->set_xW(xW);
    tilingData->set_gridD(gridD);
    tilingData->set_gridH(gridH);
    tilingData->set_gridW(gridW);
    tilingData->set_interpolation(interpolation);
    tilingData->set_padding(padding);
    tilingData->set_alignCorners(alignCorners);
    tilingData->set_blockNum(usedCoreNum);
    tilingData->set_pNumPerCore(pNumPerCore);
    tilingData->set_tailPNum(tailPNum);
    tilingData->set_ubFactorElement(ubFactorElement);
    tilingData->set_group(group);
    tilingData->set_isDeterministic(isDeterministic);
    tilingData->set_tailBNum(tailBNum);
}

template <typename TilingData>
void GridSampler3DGradTiling<TilingData>::GetTiling(TilingData* tilingData)
{
    GetUsedCore();
    SplitUb();
    FillTilingData(tilingData);
}

template <typename TilingData>
void GetGridSampler3DGradTiling(
    TilingData* tilingData, InputParamsInfo& params, uint32_t coreNum, uint32_t ubSize, uint32_t deterministic)
{
    class GridSampler3DGradTiling<TilingData> tilingObj(params, coreNum, ubSize, deterministic);
    tilingObj.GetTiling(tilingData);
}

static ge::graphStatus CheckShapes(
    gert::TilingContext* tilingContext, InputParamsInfo& params, const gert::StorageShape* gradShape,
    const gert::StorageShape* xShape, const gert::StorageShape* gridShape)
{
    OP_CHECK_IF((gradShape == nullptr), OP_LOGE(tilingContext->GetNodeName(), "Get gradShape Failed."), return false);
    OP_CHECK_IF((xShape == nullptr), OP_LOGE(tilingContext->GetNodeName(), "Get xShape Failed."), return false);
    OP_CHECK_IF((gridShape == nullptr), OP_LOGE(tilingContext->GetNodeName(), "Get gridShape Failed."), return false);
    if (xShape->GetStorageShape().GetDimNum() != CHECK_DIM_NUM) {
        OP_LOGE(tilingContext->GetNodeName(), "input dim is not 5, please check input");
        return ge::GRAPH_FAILED;
    }

    uint32_t outD = gradShape->GetStorageShape().GetDim(DIM_INDEX1);
    uint32_t outH = gradShape->GetStorageShape().GetDim(DIM_INDEX2);
    uint32_t outW = gradShape->GetStorageShape().GetDim(DIM_INDEX3);

    if (outD != params.gridD || outH != params.gridH || outW != params.gridW) {
        OP_LOGE(tilingContext->GetNodeName(), "Please check grad's dims and grid's dims");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus GetInputInfo(gert::TilingContext* tilingContext, InputParamsInfo& params)
{
    OP_LOGI(tilingContext->GetNodeName(), "GetInputInfo -> strat to get input dims.");
    const gert::StorageShape* gradShape = tilingContext->GetInputShape(GRAD_INPUT_INDEX);
    const gert::StorageShape* xShape = tilingContext->GetInputShape(X_INPUT_INDEX);
    const gert::StorageShape* gridShape = tilingContext->GetInputShape(GRID_INPUT_INDEX);

    params.batch = xShape->GetStorageShape().GetDim(DIM_INDEX0);
    params.xD = xShape->GetStorageShape().GetDim(DIM_INDEX1);
    params.xH = xShape->GetStorageShape().GetDim(DIM_INDEX2);
    params.xW = xShape->GetStorageShape().GetDim(DIM_INDEX3);
    params.channel = xShape->GetStorageShape().GetDim(DIM_INDEX4);
    params.gridD = gridShape->GetStorageShape().GetDim(DIM_INDEX1);
    params.gridH = gridShape->GetStorageShape().GetDim(DIM_INDEX2);
    params.gridW = gridShape->GetStorageShape().GetDim(DIM_INDEX3);

    CheckShapes(tilingContext, params, gradShape, xShape, gridShape);

    const gert::RuntimeAttrs* attrs = tilingContext->GetAttrs();
    const char* pInterpolationMode = attrs->GetAttrPointer<char>(static_cast<std::uint32_t>(INTERPOLATION_MODE_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, pInterpolationMode);
    if (strcmp(pInterpolationMode, "bilinear") == 0) {
        params.interpolation = INTERPOLATION_MODE_BILINEAR;
    } else if (strcmp(pInterpolationMode, "nearest") == 0) {
        params.interpolation = INTERPOLATION_MODE_NEAREST;
    } else {
        OP_LOGE(tilingContext->GetNodeName(), "interpolation_mode only support bilinear or nearest.");
        return ge::GRAPH_FAILED;
    }

    const char* pPaddingMode = attrs->GetAttrPointer<char>(static_cast<std::uint32_t>(PADDING_MODE_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, pPaddingMode);
    if (strcmp(pPaddingMode, "zeros") == 0) {
        params.padding = PADDING_MODE_ZEROS;
    } else if (strcmp(pPaddingMode, "border") == 0) {
        params.padding = PADDING_MODE_BORDER;
    } else if (strcmp(pPaddingMode, "reflection") == 0) {
        params.padding = PADDING_MODE_REFLECTION;
    } else {
        OP_LOGE(tilingContext->GetNodeName(), "padding_mode only support zeros or border or reflection.");
        return ge::GRAPH_FAILED;
    }

    const bool* pAlignCorners = attrs->GetAttrPointer<bool>(static_cast<std::uint32_t>(ALIGN_CORNERS_INDEX));
    OP_CHECK_NULL_WITH_CONTEXT(tilingContext, pAlignCorners);
    params.alignCorners = ALIGN_CORNERS_FALSE;
    if (*pAlignCorners) {
        params.alignCorners = ALIGN_CORNERS_TRUE;
    }

    size_t* currentWorkspace = tilingContext->GetWorkspaceSizes(1);
    size_t sysWorkspaceSize = WORKSPACE_SIZE;
    OP_CHECK_IF(
        (currentWorkspace == nullptr), OP_LOGE(tilingContext->GetNodeName(), "Get currentWorkspace Failed."),
        return false);
    currentWorkspace[0] = sysWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus tiling4GridSampler3DGradTiling(gert::TilingContext* tilingContext)
{
    OP_LOGI(tilingContext->GetNodeName(), "Tiling4GridSampler3DGradTiling start.");
    uint32_t deterministic = tilingContext->GetDeterministic();
    if (deterministic == 0) {
        OP_LOGI(tilingContext->GetNodeName(), "Deterministic status is closed.");
    } else {
        OP_LOGI(tilingContext->GetNodeName(), "Deterministic status is open.");
    }

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(tilingContext->GetPlatformInfo());
    auto coreNum = ascendcPlatform.GetCoreNum();
    uint64_t ubSizePlatForm = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);

    uint32_t ubSize = static_cast<uint32_t>(ubSizePlatForm);
    uint32_t availableUb = ubSize - RESERVED_UB;

    ge::DataType inputDatatype = tilingContext->GetInputDesc(GRAD_INPUT_INDEX)->GetDataType();
    OP_LOGI(tilingContext->GetNodeName(), "ubSizePlatForm: %lu, coreNum: %u", ubSizePlatForm, coreNum);

    InputParamsInfo params;
    if (GetInputInfo(tilingContext, params) != ge::GRAPH_SUCCESS) {
        OP_LOGE(tilingContext->GetNodeName(), "Failed to Parse input params, please check inputs.");
        return ge::GRAPH_FAILED;
    }

    GridSampler3DGradTilingData tilingData;
    GetGridSampler3DGradTiling<GridSampler3DGradTilingData>(&tilingData, params, coreNum, availableUb, deterministic);

    if (inputDatatype == ge::DT_FLOAT) {
        tilingContext->SetTilingKey(1);
    } else if (inputDatatype == ge::DT_FLOAT16) {
        tilingContext->SetTilingKey(2);
    } else if (inputDatatype == ge::DT_BF16) {
        tilingContext->SetTilingKey(3);
    }

    tilingContext->SetBlockDim(tilingData.get_blockNum());
    tilingContext->SetNeedAtomic(true);
    tilingData.SaveToBuffer(
        tilingContext->GetRawTilingData()->GetData(), tilingContext->GetRawTilingData()->GetCapacity());
    tilingContext->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
    OP_LOGI(tilingContext->GetNodeName(), "tiling4GridSampler3DGradTiling end.");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(GridSampler3DGrad)
    .Tiling(tiling4GridSampler3DGradTiling);

} // namespace optiling
