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
 * \file BatchToSpace_infershape.cpp
 * \brief BatchToSpace infer shape implementation
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr int64_t IDX_0 = 0;
static constexpr int64_t IDX_1 = 1;
static constexpr int64_t AXIS_N = 0;
static constexpr int64_t AXIS_H = 1;
static constexpr int64_t AXIS_W = 2;
static constexpr int64_t AXIS_C = 3;
static constexpr int64_t AXIS_COUNT = 4;

static ge::graphStatus InferShapeBatchtospace(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferShapeBatchtospace");

    const gert::Shape* xShape = context->GetInputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    gert::Shape* yShape = context->GetOutputShape(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    auto xShapeSize = xShape->GetDimNum();
    if (xShapeSize != AXIS_COUNT) {
        OP_LOGE(context->GetNodeName(), "input x shape dim = %zu, should be 4", xShapeSize);
        return GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    const int64_t* blockSizePtr = attrs->GetAttrPointer<int64_t>(IDX_0);
    OP_CHECK_NULL_WITH_CONTEXT(context, blockSizePtr);
    int64_t blockSize = *blockSizePtr;

    if (blockSize <= 0) {
        OP_LOGE(context->GetNodeName(), "block_size should be > 0");
        return GRAPH_FAILED;
    }

    int64_t batchSize = xShape->GetDim(AXIS_N);
    int64_t height = xShape->GetDim(AXIS_H);
    int64_t width = xShape->GetDim(AXIS_W);
    int64_t depth = xShape->GetDim(AXIS_C);

    int64_t blockSizeSquare = blockSize * blockSize;
    if (batchSize % blockSizeSquare != 0) {
        OP_LOGE(context->GetNodeName(), "batch N=%ld must be divisible by block_size^2=%ld", 
                batchSize, blockSizeSquare);
        return GRAPH_FAILED;
    }

    auto cropsTensor = context->GetInputTensor(IDX_1);
    if (cropsTensor == nullptr) {
        OP_LOGE(context->GetNodeName(), "crops tensor is nullptr");
        return GRAPH_FAILED;
    }

    auto cropsShape = cropsTensor->GetStorageShape();
    if (cropsShape.GetDimNum() != 2 || cropsShape.GetDim(0) != 2 || cropsShape.GetDim(1) != 2) {
        OP_LOGE(context->GetNodeName(), "crops shape should be [2, 2]");
        return GRAPH_FAILED;
    }

    auto cropsData = cropsTensor->GetData<int64_t>();
    if (cropsData == nullptr) {
        OP_LOGE(context->GetNodeName(), "crops data is nullptr");
        return GRAPH_FAILED;
    }

    int64_t cropTop = cropsData[0];
    int64_t cropBottom = cropsData[1];
    int64_t cropLeft = cropsData[2];
    int64_t cropRight = cropsData[3];

    int64_t maxCropH = blockSize * height;
    int64_t maxCropW = blockSize * width;
    if (cropTop + cropBottom > maxCropH) {
        OP_LOGE(context->GetNodeName(), "crop height exceeds limit: %ld > %ld", 
                cropTop + cropBottom, maxCropH);
        return GRAPH_FAILED;
    }
    if (cropLeft + cropRight > maxCropW) {
        OP_LOGE(context->GetNodeName(), "crop width exceeds limit: %ld > %ld", 
                cropLeft + cropRight, maxCropW);
        return GRAPH_FAILED;
    }

    int64_t outBatch = batchSize / blockSizeSquare;
    int64_t outHeight = height * blockSize - cropTop - cropBottom;
    int64_t outWidth = width * blockSize - cropLeft - cropRight;
    int64_t outDepth = depth;

    yShape->SetDimNum(AXIS_COUNT);
    yShape->SetDim(AXIS_N, outBatch);
    yShape->SetDim(AXIS_H, outHeight);
    yShape->SetDim(AXIS_W, outWidth);
    yShape->SetDim(AXIS_C, outDepth);

    OP_LOGD(context->GetNodeName(), "End to do InferShapeBatchtospace");
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Batchtospace).InferShape(InferShapeBatchtospace);
} // namespace ops