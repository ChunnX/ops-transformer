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
 * \file BatchToSpace_tiling.cpp
 * \brief BatchToSpace tiling implementation
 */

#include "log/log.h"
#include "util/math_util.h"
#include "tiling_base/tiling_util.h"
#include "tiling_base/tiling_templates_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/BatchToSpace_tiling_data.h"
#include "../op_kernel/BatchToSpace_tiling_key.h"

namespace optiling {

using namespace Ops::Transformer::OpTiling;
using namespace Ops::Base;

constexpr int32_t DIMS_LIMIT = 4;
constexpr uint32_t AXIS_COUNT = 4;
constexpr uint32_t INDEX_N = 0;
constexpr uint32_t INDEX_H = 1;
constexpr uint32_t INDEX_W = 2;
constexpr uint32_t INDEX_C = 3;
constexpr uint32_t CROPS_DIM_0 = 0;
constexpr uint32_t CROPS_DIM_1 = 1;
constexpr uint32_t CROPS_TOP = 0;
constexpr uint32_t CROPS_BOTTOM = 1;
constexpr uint32_t CROPS_LEFT = 0;
constexpr uint32_t CROPS_RIGHT = 1;
constexpr uint64_t MAX_BUFFER_SIZE = 64U * 1024U;
constexpr uint32_t WS_SYS_SIZE = 16U * 1024U * 1024U;

struct BatchtospaceCompileInfo {};

class BatchtospaceTilingBase {
public:
    BatchtospaceTilingBase(gert::TilingContext* context) : context_(context) {}

    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetInputShapeAndAttrs();
    ge::graphStatus CalculateTiling();
    ge::graphStatus SetTilingData();

private:
    gert::TilingContext* context_;
    uint64_t ubSize_ = 0;
    uint64_t ubBlockSize_ = 32;
    uint64_t cacheLineSize_ = 256;
    uint32_t coreNum_ = 0;
    int64_t inShape_[AXIS_COUNT] = {0};
    int64_t outShape_[AXIS_COUNT] = {0};
    int64_t blockSize_ = 0;
    int64_t crops_[2][2] = {{0}};
    uint64_t totalCount_ = 0;
    uint64_t perCoreCount_ = 0;
    uint8_t ubAxis_ = 0;
    uint32_t ubFactor_ = 0;
    uint32_t bufferSize_ = 0;
    bool isLargeC_ = false;
};

ge::graphStatus BatchtospaceTilingBase::GetPlatformInfo()
{
    fe::PlatFormInfos* platformInfoPtr = context_->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context_, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(coreNum_ == 0, OP_LOGE(context_, "coreNum is 0"), return ge::GRAPH_FAILED);
    
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    OP_CHECK_IF(ubSize_ == 0, OP_LOGE(context_, "ubSize is 0"), return ge::GRAPH_FAILED);
    
    ubBlockSize_ = Ops::Base::GetUbBlockSize(context_);
    cacheLineSize_ = Ops::Base::GetCacheLineSize(context_);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchtospaceTilingBase::GetInputShapeAndAttrs()
{
    auto inputX = context_->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputX);
    auto inputShapeX = EnsureNotScalar(inputX->GetStorageShape());
    
    OP_CHECK_IF(inputShapeX.GetDimNum() != DIMS_LIMIT,
                OP_LOGE(context_, "input x shape dim = %zu, should be 4", inputShapeX.GetDimNum()),
                return ge::GRAPH_FAILED);
    
    for (uint32_t i = 0; i < AXIS_COUNT; i++) {
        inShape_[i] = inputShapeX.GetDim(i);
    }
    
    auto cropsTensor = context_->GetInputTensor(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cropsTensor);
    
    auto cropsShape = cropsTensor->GetStorageShape();
    OP_CHECK_IF(cropsShape.GetDimNum() != 2 || cropsShape.GetDim(0) != 2 || cropsShape.GetDim(1) != 2,
                OP_LOGE(context_, "crops shape should be [2, 2]"),
                return ge::GRAPH_FAILED);
    
    auto cropsData = cropsTensor->GetData<int64_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, cropsData);
    
    crops_[CROPS_DIM_0][CROPS_TOP] = cropsData[0];
    crops_[CROPS_DIM_0][CROPS_BOTTOM] = cropsData[1];
    crops_[CROPS_DIM_1][CROPS_LEFT] = cropsData[2];
    crops_[CROPS_DIM_1][CROPS_RIGHT] = cropsData[3];
    
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
    
    const int64_t* blockSizePtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, blockSizePtr);
    blockSize_ = *blockSizePtr;
    
    OP_CHECK_IF(blockSize_ <= 0, OP_LOGE(context_, "block_size should be > 0"), return ge::GRAPH_FAILED);
    
    int64_t blockSizeSquare = blockSize_ * blockSize_;
    OP_CHECK_IF(inShape_[INDEX_N] % blockSizeSquare != 0,
                OP_LOGE(context_, "batch N=%ld must be divisible by block_size^2=%ld", 
                        inShape_[INDEX_N], blockSizeSquare),
                return ge::GRAPH_FAILED);
    
    int64_t maxCropH = blockSize_ * inShape_[INDEX_H];
    int64_t maxCropW = blockSize_ * inShape_[INDEX_W];
    OP_CHECK_IF(crops_[CROPS_DIM_0][CROPS_TOP] + crops_[CROPS_DIM_0][CROPS_BOTTOM] > maxCropH,
                OP_LOGE(context_, "crop height exceeds limit"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(crops_[CROPS_DIM_1][CROPS_LEFT] + crops_[CROPS_DIM_1][CROPS_RIGHT] > maxCropW,
                OP_LOGE(context_, "crop width exceeds limit"),
                return ge::GRAPH_FAILED);
    
    outShape_[INDEX_N] = inShape_[INDEX_N] / blockSizeSquare;
    outShape_[INDEX_H] = inShape_[INDEX_H] * blockSize_ - crops_[CROPS_DIM_0][CROPS_TOP] - crops_[CROPS_DIM_0][CROPS_BOTTOM];
    outShape_[INDEX_W] = inShape_[INDEX_W] * blockSize_ - crops_[CROPS_DIM_1][CROPS_LEFT] - crops_[CROPS_DIM_1][CROPS_RIGHT];
    outShape_[INDEX_C] = inShape_[INDEX_C];
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchtospaceTilingBase::CalculateTiling()
{
    isLargeC_ = (outShape_[INDEX_C] > cacheLineSize_ / 2);
    
    bufferSize_ = ubSize_ / 2;
    if (bufferSize_ > MAX_BUFFER_SIZE) {
        bufferSize_ = MAX_BUFFER_SIZE;
    }
    
    auto inputDesc = context_->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context_, inputDesc);
    ge::DataType dataType = inputDesc->GetDataType();
    uint32_t dtypeSize = ge::GetSizeByDataType(dataType);
    if (dtypeSize == 0) {
        OP_LOGE(context_, "unsupported dtype");
        return ge::GRAPH_FAILED;
    }
    
    uint64_t tmpBufferSize = bufferSize_ / dtypeSize;
    ubAxis_ = AXIS_COUNT;
    
    // 从最内轴（C轴）开始，逐轴计算bufferSize能容纳多少个该轴元素
    // tmpBufferSize每次更新为factor，表示能容纳内轴多少个元素组
    for (int32_t axis = AXIS_COUNT - 1; axis >= 0; axis--) {
        uint64_t axisSize = (axis == AXIS_COUNT - 1) ? 
                            CeilAlign(static_cast<uint64_t>(outShape_[axis]), ubBlockSize_) : 
                            static_cast<uint64_t>(outShape_[axis]);
        
        uint32_t factor = tmpBufferSize / axisSize;
        
        if (factor == 0) {
            // bufferSize不够容纳一个完整的(axis及其内轴)元素组
            // tmpBufferSize已经是内轴除法后的结果，直接作为ubFactor
            ubAxis_ = static_cast<uint8_t>(axis);
            ubFactor_ = static_cast<uint32_t>(tmpBufferSize);
            
            // 确保ubFactor不超过outShape[ubAxis]
            if (ubFactor_ > static_cast<uint32_t>(outShape_[ubAxis_])) {
                ubFactor_ = static_cast<uint32_t>(outShape_[ubAxis_]);
            }
            break;
        }
        
        tmpBufferSize = factor;
    }
    
    // 如果所有轴都能被bufferSize完整容纳，切分在最外轴(N轴)
    if (ubAxis_ == AXIS_COUNT) {
        ubAxis_ = 0;
        ubFactor_ = static_cast<uint32_t>(outShape_[INDEX_N]);
    }
    
    // 计算totalCount：ubAxis之前的轴累乘 × ubAxis的切分次数 × ubAxis之后的轴累乘
    totalCount_ = 1;
    for (uint32_t axis = 0; axis < ubAxis_; axis++) {
        totalCount_ *= static_cast<uint64_t>(outShape_[axis]);
    }
    
    uint64_t ubFactorCount = CeilDiv(static_cast<uint64_t>(outShape_[ubAxis_]), static_cast<uint64_t>(ubFactor_));
    totalCount_ *= ubFactorCount;
    
    for (uint32_t axis = ubAxis_ + 1; axis < AXIS_COUNT; axis++) {
        totalCount_ *= static_cast<uint64_t>(outShape_[axis]);
    }
    
    perCoreCount_ = CeilDiv(totalCount_, static_cast<uint64_t>(coreNum_));
    
    uint32_t usedCoreNum = CeilDiv(totalCount_, perCoreCount_);
    if (usedCoreNum > coreNum_) {
        usedCoreNum = coreNum_;
    }
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus BatchtospaceTilingBase::SetTilingData()
{
    BatchtospaceTilingData* tiling = context_->GetTilingData<BatchtospaceTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tiling);
    
    OP_CHECK_IF(memset_s(tiling, sizeof(BatchtospaceTilingData), 0, sizeof(BatchtospaceTilingData)) != EOK,
                OP_LOGE(context_, "memset tiling data failed"),
                return ge::GRAPH_FAILED);
    
    for (uint32_t i = 0; i < AXIS_COUNT; i++) {
        tiling->inShape[i] = inShape_[i];
        tiling->outShape[i] = outShape_[i];
    }
    tiling->blockSize = blockSize_;
    for (uint32_t i = 0; i < 2; i++) {
        for (uint32_t j = 0; j < 2; j++) {
            tiling->crops[i][j] = crops_[i][j];
        }
    }
    tiling->totalCount = totalCount_;
    tiling->perCoreCount = perCoreCount_;
    tiling->ubAxis = ubAxis_;
    tiling->ubFactor = ubFactor_;
    tiling->bufferSize = bufferSize_;
    
    uint64_t tilingKey = 0;
    if (isLargeC_) {
        tilingKey = GET_TPL_TILING_KEY(TPL_SCH_MODE_IS_LARGE_C);
    } else {
        tilingKey = GET_TPL_TILING_KEY(TPL_SCH_MODE_IS_SMALL_C);
    }
    context_->SetTilingKey(tilingKey);
    
    uint32_t usedCoreNum = CeilDiv(totalCount_, perCoreCount_);
    context_->SetBlockDim(usedCoreNum);
    
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus GetWorkspaceSize(gert::TilingContext* context)
{
    size_t* currentWorkspace = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = WS_SYS_SIZE;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus BatchtospaceTilingFunc(gert::TilingContext* context)
{
    BatchtospaceTilingBase tilingBase(context);
    
    OP_CHECK_IF(tilingBase.GetPlatformInfo() != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo failed"),
                return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(tilingBase.GetInputShapeAndAttrs() != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetInputShapeAndAttrs failed"),
                return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(tilingBase.CalculateTiling() != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "CalculateTiling failed"),
                return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(tilingBase.SetTilingData() != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "SetTilingData failed"),
                return ge::GRAPH_FAILED);
    
    OP_CHECK_IF(GetWorkspaceSize(context) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetWorkspaceSize failed"),
                return ge::GRAPH_FAILED);
    
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForBatchtospace([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(Batchtospace).Tiling(BatchtospaceTilingFunc).TilingParse<BatchtospaceCompileInfo>(TilingParseForBatchtospace);

} // namespace optiling