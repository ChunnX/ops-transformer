/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file multi_scale_deformable_attn_function_tiling.cpp
 * \brief
 */
#include "multi_scale_deformable_attn_function_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"

using namespace ge;
namespace {
const uint64_t INPUT_VALUE = 0;
const uint64_t INPUT_SPATIAL_SHAPE = 1;
const uint64_t INPUT_ATTN_WEIGHT = 4;

const uint64_t BATCH_SIZE_DIM = 0;
const uint64_t NUM_KEYS_DIM = 1;
const uint64_t NUM_HEADS_DIM = 2;
const uint64_t EMBED_DIMS_DIM = 3;
const uint64_t NUM_LEVEL_DIM = 0;
const uint64_t REAL_LEVEL_DIM = 3;
const uint64_t NUM_QUERIES_DIM = 1;
const uint64_t NUM_POINTS_DIM = 4;

const uint64_t NUM_KEYS_DIM_TRANSPOSE = 2;
const uint64_t NUM_HEADS_DIM_TRANSPOSE = 1;
const uint64_t REAL_LEVEL_DIM_TRANSPOSE = 2;
const uint64_t NUM_QUERIES_DIM_TRANSPOSE = 4;
const uint64_t NUM_POINTS_DIM_TRANSPOSE = 3;

const uint64_t NUM_POINTS_EIGHT = 8;
const uint64_t NUM_POINTS_FOUR = 4;
const uint64_t NUM_POINTS_TWO = 2;
const uint64_t EMBEDDIMS_SIXTEEN = 16;
const uint64_t sysWorkspaceSize = 16 * 1024 *1024;
const uint64_t TILING_KEY_WEIGHT = 1000;

const uint64_t NUM_QUERIES_MIN = 32;
const uint64_t NUM_HEADS_MAX = 16;
const uint64_t NUM_LEVELS_MAX = 16;
const uint64_t NUM_POINTS_MAX = 16;
const uint64_t EMBED_DIMS_DIV_EIGHT = 8;
const uint64_t EMBED_DIMS_MAX = 256;

std::tuple<uint64_t, uint64_t> GroupPoints(uint64_t numPoints)
{
    if (numPoints % NUM_POINTS_EIGHT == 0) {
        return std::make_tuple(NUM_POINTS_EIGHT, numPoints / NUM_POINTS_EIGHT);
    }
    if (numPoints % NUM_POINTS_FOUR == 0) {
        return std::make_tuple(NUM_POINTS_FOUR, numPoints / NUM_POINTS_FOUR);
    }
    if (numPoints % NUM_POINTS_TWO == 0) {
        return std::make_tuple(NUM_POINTS_TWO, numPoints / NUM_POINTS_TWO);
    }
    return std::make_tuple(1, numPoints);
}
} // namespace

namespace optiling {

static ge::graphStatus TilingFuncForMultiScaleDeformableAttn(gert::TilingContext *context)
{
    MultiScaleDeformableAttnFunctionTilingData tiling;
    auto valueTensorPtr = context->GetInputTensor(INPUT_VALUE);
    auto spatialTensorPtr = context->GetInputTensor(INPUT_SPATIAL_SHAPE);
    auto attnWeightTensorPtr = context->GetInputTensor(INPUT_ATTN_WEIGHT);
    if (valueTensorPtr == nullptr || spatialTensorPtr == nullptr || attnWeightTensorPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto valueShape = valueTensorPtr->GetStorageShape();
    auto spatialShape = spatialTensorPtr->GetStorageShape();
    auto attnWeightShape = attnWeightTensorPtr->GetStorageShape();

    auto platformInfo = context->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t coreNum = ascendcPlatform.GetCoreNumAiv();

    uint64_t deterministicFlag = context->GetDeterministic() == 1 ? 1 : 0;
    if (deterministicFlag == 1) {
        coreNum = 1;
    }

    context->SetBlockDim(coreNum);
    tiling.set_coreNum(coreNum);

    uint64_t isTranspose = attnWeightShape.GetDim(1) >= 32 ? 0 : 1;
    uint64_t numLevels = spatialShape.GetDim(NUM_LEVEL_DIM);
    uint64_t embedDims = valueShape.GetDim(EMBED_DIMS_DIM);
    uint64_t batchSize = valueShape.GetDim(BATCH_SIZE_DIM);
    uint64_t numPoints, numHeads, numKeys, numQueries, realLevels;
    if (isTranspose == 0) {
        numPoints = attnWeightShape.GetDim(NUM_POINTS_DIM);
        numHeads = attnWeightShape.GetDim(NUM_HEADS_DIM);
        numKeys = valueShape.GetDim(NUM_KEYS_DIM);
        numQueries = attnWeightShape.GetDim(NUM_QUERIES_DIM);
        realLevels = attnWeightShape.GetDim(REAL_LEVEL_DIM);
    } else {
        numPoints = attnWeightShape.GetDim(NUM_POINTS_DIM_TRANSPOSE);
        numHeads = attnWeightShape.GetDim(NUM_HEADS_DIM_TRANSPOSE);
        numKeys = valueShape.GetDim(NUM_KEYS_DIM_TRANSPOSE);
        numQueries = attnWeightShape.GetDim(NUM_QUERIES_DIM_TRANSPOSE);
        realLevels = attnWeightShape.GetDim(REAL_LEVEL_DIM_TRANSPOSE);
    }
    uint64_t optPoint = numLevels <= 8 && numHeads <= 8 && (embedDims == 16 || embedDims == 32) &&
                        (numPoints % 2 == 0);

    uint64_t pointLoops = 0;
    uint64_t point = 0;
    if (optPoint) {
        auto groups = GroupPoints(numPoints);
        pointLoops = std::get<1>(groups);
        point = std::get<0>(groups);
    }

    if (numQueries < NUM_QUERIES_MIN) {
        return ge::GRAPH_FAILED;
    }

    if (numHeads > NUM_HEADS_MAX) {
        return ge::GRAPH_FAILED;
    }

    if (numLevels > NUM_LEVELS_MAX) {
        return ge::GRAPH_FAILED;
    }

    if (numPoints > NUM_POINTS_MAX) {
        return ge::GRAPH_FAILED;
    }

    if (!((embedDims % EMBED_DIMS_DIV_EIGHT == 0) && (embedDims <= EMBED_DIMS_MAX))) {
        return ge::GRAPH_FAILED;
    }

    uint64_t TilingKey = optPoint == 1 ? (embedDims / EMBEDDIMS_SIXTEEN) * TILING_KEY_WEIGHT + point : 0;
    if (TilingKey == 0) {
        context->SetScheduleMode(1);
    }

    context->SetTilingKey(TilingKey);
    tiling.set_batchSize(batchSize);
    tiling.set_numKeys(numKeys);
    tiling.set_numHeads(numHeads);
    tiling.set_embedDims(embedDims);
    tiling.set_numLevels(numLevels);
    tiling.set_numQueries(numQueries);
    tiling.set_numPoints(numPoints);
    tiling.set_coreNum(coreNum);
    tiling.set_pointLoops(pointLoops);
    tiling.set_realLevels(realLevels);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = sysWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(MultiScaleDeformableAttnFunction)
    .Tiling(TilingFuncForMultiScaleDeformableAttn);
}  // namespace optiling