/* *
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file allto_allv_quant_grouped_mat_mul_infershape.cc
 * \brief
 */
#include "mc2_log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "graph/utils/type_utils.h"

using namespace ge;

namespace ops {
// output index
static const size_t INDEX_OUT_GMM_Y = 0;
static const size_t INDEX_OUT_MM_Y = 1;
static const size_t INDEX_PERMUTE_OUT = 2;
// input index
static const size_t INDEX_IN_GMM_X = 0;
static const size_t INDEX_IN_GMM_WEIGHT = 1;
static const size_t INDEX_IN_MM_X = 6;
static const size_t INDEX_IN_MM_WEIGHT = 7;
// dim num
static constexpr size_t DIM_NUM_0 = 0;
static constexpr size_t DIM_NUM_1 = 1;
static constexpr size_t DIM_NUM_2 = 2;
static constexpr size_t DIM_NUM_3 = 3;
// attr index
static const size_t INDEX_ATTR_EP_WORLD_SIZE = 1;
static const size_t INDEX_ATTR_SEND_COUNTS = 2;
static const size_t INDEX_ATTR_RECV_COUNTS = 3;
static const size_t INDEX_ATTR_TRANS_GMM_WEIGHT_INDEX = 6;
static const size_t INDEX_ATTR_TRANS_MM_WEIGHT_INDEX = 7;
static const size_t INDEX_ATTR_PERMUTE_OUT_FLAG_INDEX = 8;
static const size_t INDEX_ATTR_Y_DTYPE_INDEX = 12;
static const size_t INDEX_ATTR_MM_DTYPE_INDEX = 13;

static constexpr size_t DIM_0 = 0;
static constexpr size_t DIM_1 = 1;
static constexpr size_t DIM_2 = 2;

static constexpr int64_t FIRST_ELE_SIZE = -1;
static constexpr uint64_t MX_QUANT_MODE_NUM = 6;
static constexpr uint64_t TT_QUANT_MODE_NUM = 1;
static constexpr uint64_t SCALE_LAST_DIM = 2;

static graphStatus CheckDims(const gert::InferShapeContext *context, const gert::Shape *gmmXShape,
    const gert::Shape *gmmWeightShape, bool transGmmWeight)
{
    auto result = ge::GRAPH_SUCCESS;

    auto gmmXDimNum = gmmXShape->GetDimNum();
    auto gmmWeightDimNum = gmmWeightShape->GetDimNum();
    if (gmmXDimNum != DIM_NUM_2) {
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Only gmmX with dim 2 is supported.");
        result = ge::GRAPH_FAILED;
    }
    if (gmmWeightDimNum != DIM_NUM_3) {
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Only gmmWeight with dim 3 is supported.");
        result = ge::GRAPH_FAILED;
    }
    return result;
}

static graphStatus CheckDimsOptional(const gert::InferShapeContext *context, const gert::Shape *mmXShape,
    const gert::Shape *mmWeightShape, bool transMmWeight)
{
    auto result = ge::GRAPH_SUCCESS;
    auto xDimNum = mmXShape->GetDimNum();
    auto mmWeightDimNum = mmWeightShape->GetDimNum();
    if (xDimNum != DIM_NUM_2) {
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Only x with dim 2 is supported.");
        result = ge::GRAPH_FAILED;
    }
    if (mmWeightDimNum != DIM_NUM_2) {
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "Only mmWeight with dim 2 is supported.");
        result = ge::GRAPH_FAILED;
    }
    return result;
}

static ge::graphStatus InferGMMOutputShape(const gert::InferShapeContext *context, gert::Shape *gmmYShape,
    const int64_t *epWorldSizePtr, const gert::ContinuousVector *recvCountsPtr,
    const gert::ContinuousVector *sendCountsPtr, const int64_t e, int64_t &a, const int64_t n1)
{
    gmmYShape->SetDimNum(DIM_NUM_2);
    gmmYShape->SetDim(DIM_0, FIRST_ELE_SIZE);
    gmmYShape->SetDim(DIM_1, FIRST_ELE_SIZE);
    if (e != FIRST_ELE_SIZE) {
        int64_t arraySize = e * (*epWorldSizePtr);
        OPS_ERR_IF(recvCountsPtr->GetSize() < static_cast<size_t>(arraySize),
            VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
            "recvCounts size should not be smaller than e * epWorldSize."),
            return ge::GRAPH_FAILED);
        OPS_ERR_IF(sendCountsPtr->GetSize() < static_cast<size_t>(arraySize),
            VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
            "sendCounts size should not be smaller than e * epWorldSize."),
            return ge::GRAPH_FAILED);
        for (int64_t i = 0; i < arraySize; i++) {
            a += static_cast<const int64_t *>(recvCountsPtr->GetData())[i];
        }
        gmmYShape->SetDim(DIM_0, a);
        gmmYShape->SetDim(DIM_1, n1);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferMMOutputShape(const gert::InferShapeContext *context, const gert::Shape *mmXShape,
    const gert::Shape *mmWeightShape, const bool *transMmWeightPtr, gert::Shape *mmYShape)
{
    OPS_ERR_IF(mmYShape == nullptr, VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
        "the shape of output mm_y is nullptr."), return ge::GRAPH_FAILED);
    mmYShape->SetDimNum(DIM_NUM_0);
    if ((mmXShape != nullptr) && (mmWeightShape != nullptr) &&
        (mmYShape != nullptr) && (transMmWeightPtr != nullptr)) {
        int64_t bs = mmXShape->GetDim(DIM_0);
        mmYShape->SetDimNum(DIM_NUM_2);
        if (bs == FIRST_ELE_SIZE) {
            mmYShape->SetDim(DIM_0, FIRST_ELE_SIZE);
            mmYShape->SetDim(DIM_1, FIRST_ELE_SIZE);
        } else {
            int64_t n2 = *transMmWeightPtr ? mmWeightShape->GetDim(DIM_0) : mmWeightShape->GetDim(DIM_1);
            mmYShape->SetDim(DIM_0, bs);
            mmYShape->SetDim(DIM_1, n2);
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferPermuteOutputShape(const gert::InferShapeContext *context, const bool *permuteOutFlagPtr,
    const int64_t e, const int64_t a, const int64_t h, gert::Shape *permuteOutShape)
{
    OPS_ERR_IF(permuteOutShape == nullptr,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), 
        "the shape of output permute_out is nullptr."), return ge::GRAPH_FAILED);
    permuteOutShape->SetDimNum(DIM_NUM_0);
    if ((permuteOutShape != nullptr) && (permuteOutFlagPtr != nullptr) && (*permuteOutFlagPtr == true)) {
        permuteOutShape->SetDimNum(DIM_NUM_2);
        if (e == FIRST_ELE_SIZE) {
            permuteOutShape->SetDim(DIM_0, FIRST_ELE_SIZE);
            permuteOutShape->SetDim(DIM_1, FIRST_ELE_SIZE);
        } else {
            permuteOutShape->SetDim(DIM_0, a);
            permuteOutShape->SetDim(DIM_1, h);
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShapeAlltoAllvGroupedMatMul(gert::InferShapeContext *context)
{
    auto *gmmXShape = context->GetInputShape(INDEX_IN_GMM_X);
    auto *gmmWeightShape = context->GetInputShape(INDEX_IN_GMM_WEIGHT);
    auto *mmXShape = context->GetOptionalInputShape(INDEX_IN_MM_X);
    auto *mmWeightShape = context->GetOptionalInputShape(INDEX_IN_MM_WEIGHT);
    auto *gmmYShape = context->GetOutputShape(INDEX_OUT_GMM_Y);
    auto *mmYShape = context->GetOutputShape(INDEX_OUT_MM_Y);
    auto *permuteOutShape = context->GetOutputShape(INDEX_PERMUTE_OUT);

    auto *attrs = context->GetAttrs();
    auto *epWorldSizePtr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_EP_WORLD_SIZE);
    auto *recvCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_ATTR_RECV_COUNTS);
    auto *sendCountsPtr = attrs->GetAttrPointer<gert::ContinuousVector>(INDEX_ATTR_SEND_COUNTS);
    auto *transGmmWeightPtr = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANS_GMM_WEIGHT_INDEX);
    auto *transMmWeightPtr = attrs->GetAttrPointer<bool>(INDEX_ATTR_TRANS_MM_WEIGHT_INDEX);
    auto *permuteOutFlagPtr = attrs->GetAttrPointer<bool>(INDEX_ATTR_PERMUTE_OUT_FLAG_INDEX);
    auto *gmmXScaleShape = context->GetOptionalInputShape(INDEX_GMM_X_SCALE);
    auto *gmmWeightScaleShape = context->GetOptionalInputShape(INDEX_GMM_WEIGHT_SCALE);
    auto *mmXScaleShape = context->GetOptionalInputShape(INDEX_MM_X_SCALE);
    auto *mmWeightScaleShape = context->GetOptionalInputShape(INDEX_MM_WEIGHT_SCALE);
    const int64_t* gmmXQuantModePtr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_GMM_X_QUANT_MODE);
    const int64_t* gmmWeightQuantModePtr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_GMM_WEIGHT_QUANT_MODE);
    const int64_t* mmXQuantModePtr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_MM_X_QUANT_MODE);
    const int64_t* mmWeightQuantModePtr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_MM_WEIGHT_QUANT_MODE);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gmmXQuantModePtr);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gmmWeightQuantModePtr);

    OPS_CHECK_NULL_WITH_CONTEXT(context, recvCountsPtr);
    OPS_CHECK_NULL_WITH_CONTEXT(context, sendCountsPtr);
    OPS_CHECK_NULL_WITH_CONTEXT(context, transGmmWeightPtr);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gmmXShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gmmWeightShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gmmXScaleShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gmmWeightScaleShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, gmmYShape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, epWorldSizePtr);
    OPS_CHECK(CheckOptionalInputMatchValid(context, mmXShape, mmWeightShape, mmXScaleShape, mmWeightScaleShape) != ge::GRAPH_SUCCESS,
              CUBE_INNER_ERR_REPORT(context->GetNodeName(), "Failed to optional input valid."),
              return ge::GRAPH_FAILED);
    OPS_ERR_IF(CheckDims(context, gmmXShape, gmmWeightShape, *transGmmWeightPtr) != ge::GRAPH_SUCCESS,
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(), "CheckDims failed."), return ge::GRAPH_FAILED);
    if ((mmXShape != nullptr) && (mmWeightShape != nullptr) && (transMmWeightPtr != nullptr)) {
        OPS_ERR_IF(CheckDimsOptional(context, mmXShape, mmWeightShape,
            *transMmWeightPtr) != ge::GRAPH_SUCCESS, VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
            "CheckDimsOptional failed."), return ge::GRAPH_FAILED);
    }

    int64_t bsk = gmmXShape->GetDim(DIM_0);
    int64_t e = gmmWeightShape->GetDim(DIM_0);
    int64_t a = 0;
    int64_t h = gmmXShape->GetDim(DIM_1);
    int64_t n1 = *transGmmWeightPtr ? gmmWeightShape->GetDim(DIM_1) : gmmWeightShape->GetDim(DIM_2);
    int64_t ScaleN1 = *transGmmWeightPtr ? gmmWeightScaleShape->GetDim(DIM_1) : gmmWeightScaleShape->GetDim(DIM_2);
    OPS_CHECK(n1 != ScaleN1, CUBE_INNER_ERR_REPORT(context->GetNodeName(), "gmmWeight axis-N1 should equal to gmmWeightScale axis-N1,
                                                   but actual gmmWeight axis-N1 is:%lu, gmmWeightScale axis-N1 is:%lu.", n1, ScaleN1), return ge::GRAPH_FAILED);
    if (e != FIRST_ELE_SIZE) {
        OPS_CHECK(CheckAxisKShapeForAlltoAllvQuantGroupedMatmul(context, gmmXShape, gmmWeightShape, *transGmmWeightPtr, mmXShape,
                  mmWeightShape, *transMmWeightPtr) != ge::GRAPH_SUCCESS,
                  CUBE_INNER_ERR_REPORT(context->GetNodeName(), "Failed to check k-axis."),
                  return ge::GRAPH_FAILED);
        OPS_CHECK(CheckScaleShapeForAlltoAllvQuantGroupedMatmul(context, bsk, e, gmmXQuantModePtr, gmmWeightQuantModePtr, gmmXShape,
                  gmmWeightShape) != ge::GRAPH_SUCCESS,
                  CUBE_INNER_ERR_REPORT(context->GetNodeName(), "Failed to check gmmXScale and gmmWeightScale shape."),
                  return ge::GRAPH_FAILED);
        if ((mmXShape != nullptr) && (mmWeightShape != nullptr) && (mmXScaleShape != nullptr) && (mmWeightScaleShape != nullptr)
        && (mmXQuantModePtr != nullptr) && (mmWeightQuantModePtr != nullptr)) {
            int64_t bs = mmXShape->GetDim(DIM_0);
            OPS_CHECK(CheckOptionalScaleShapeForAlltoAllvQuantGroupedMatmul(context, bs, mmXQuantModePtr, mmWeightQuantModePtr,
                      mmXScaleShape, mmWeightScaleShape) != ge::GRAPH_SUCCESS,
                      CUBE_INNER_ERR_REPORT(context->GetNodeName(), "Failed to check mmXScale and mmWeightScale shape."),
                      return ge::GRAPH_FAILED);
        }
    }
    if (InferGMMOutputShape(context, gmmYShape, epWorldSizePtr, recvCountsPtr, sendCountsPtr, e, a, n1) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (InferMMOutputShape(context, mmXShape, mmWeightShape, transMmWeightPtr, mmYShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (InferPermuteOutputShape(context, permuteOutFlagPtr, e, a, h, permuteOutShape) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOptionalInputMatchValid(const gert::InferShapeContext *context, const gert::Shape *mmXShape, const gert::Shape *mmWeightShape,
                                               const gert::Shape *mmXScaleShape, const gert::Shape *mmWeightScaleShape)
{
    if (mmXShape != nullptr) {
        OPS_CHECK_NULL_WITH_CONTEXT(context, mmXScaleShape);
    }
    if (mmXScaleShape != nullptr) {
        OPS_CHECK_NULL_WITH_CONTEXT(context, mmXShape);
    }
    if (mmWeightShape != nullptr) {
        OPS_CHECK_NULL_WITH_CONTEXT(context, mmWeightScaleShape);
    }
    if (mmWeightScaleShape != nullptr) {
        OPS_CHECK_NULL_WITH_CONTEXT(context, mmWeightShape);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckAxisKShapeForAlltoAllvQuantGroupedMatmul(const gert::InferShapeContext *context, const gert::Shape *gmmXShape,
    const gert::Shape *gmmWeightShape, const bool transGmmWeight, const gert::Shape *mmXShape, const gert::Shape *mmWeightShape,
    const bool transMmWeightPtr)
{
    auto k1 = gmmXShape->GetDim(DIM_1);
    auto k2 = gmmWeightShape->GetDim(DIM_1);
    if (transGmmWeight) {
        k2 = gmmWeightShape->GetDim(DIM_2);
    }
    if (k1 != k2) {
        VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
            "Dim of gmmX and dim of gmmWeight do not match for MatMul");
        result = ge::GRAPH_FAILED;
    }
    if ((mmXShape != nullptr) && (mmWeightShape != nullptr) && (transMmWeightPtr != nullptr)) {
        auto k1 = mmXShape->GetDim(DIM_1);
        auto k2 = mmWeightShape->GetDim(DIM_0);
        if (transMmWeightPtr) {
            k2 = mmWeightShape->GetDim(DIM_1);
        }
        if (k1 != k2) {
            VECTOR_INFER_SHAPE_INNER_ERR_REPORT(context->GetNodeName(),
                "Dim of x and dim of mmWeight do not match for MatMul");
            result = ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckScaleShapeForAlltoAllvQuantGroupedMatmul(const gert::InferShapeContext* context, const int64_t bsk, const int64_t e,
                                                                     const bool gmmXQuantMode, const bool gmmWeightQuantMode,
                                                                     const gert::Shape *gmmXScaleShape, const gert::Shape *gmmWeightScaleShape)
{
    ////还有scale的H轴和N轴没有校验
    int64_t gmmXScaleDimNum = gmmXScaleShape->GetDimNum();
    int64_t gmmWeightScaleDimNum = gmmWeightScaleShape->GetDimNum();
    if (*gmmXQuantMode == MX_QUANT_MODE_NUM && *gmmWeightQuantMode == MX_QUANT_MODE_NUM) {
        OPS_CHECK(gmmXScaleDimNum != DIM_NUM_3, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                  "In MX quantization mode, gmmXScale dim num must be %ld, but actual value is: %ld",
                  DIM_NUM_3, gmmXScaleDimNum), return ge::GRAPH_FAILED);
        OPS_CHECK(gmmWeightScaleDimNum != DIM_NUM_4, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                  "In MX quantization mode, gmmWeightScale dim num must be %ld, but actual value is: %ld",
                  DIM_NUM_4, gmmWeightScaleDimNum), return ge::GRAPH_FAILED);
        OPS_CHECK(gmmXScaleDimNum->GetDim(gmmXScaleDimNum - 1) != SCALE_LAST_DIM,
                  CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                  "In MX quantization mode, gmmXScale last dim must be %ld, but actual value is: %ld",
                  SCALE_LAST_DIM, gmmXScaleDimNum->GetDim(gmmXScaleDimNum - 1)), return ge::GRAPH_FAILED);
        OPS_CHECK(gmmWeightScaleDimNum->GetDim(gmmWeightScaleDimNum - 1) != SCALE_LAST_DIM,
                  CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                  "In MX quantization mode, gmmWeightScale last dim must be %ld, but actual value is: %ld",
                  SCALE_LAST_DIM, gmmWeightScaleDimNum->GetDim(gmmWeightScaleDimNum - 1)), return ge::GRAPH_FAILED); 
        OPS_CHECK(gmmXScaleShape->GetDim(0) != bsk, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                  "In MX quantization mode, gmmXScale dim0 must be the same with gmmX first dim, "
                  "but actual gmmXScale dim0 is: %ld, gmmX first dim is: %ld",
                  gmmXScaleShape->GetDim(0), bsk), return ge::GRAPH_FAILED);
        OPS_CHECK(gmmWeightScaleShape->GetDim(0) != e, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                  "In MX quantization mode, gmmWeightScale dim0 must be the same with gmmWeight first dim, "
                  "but actual gmmWeightScale dim0 is: %ld, gmmWeight first dim is: %ld",
                  gmmWeightScaleShape->GetDim(0), e), return ge::GRAPH_FAILED);
    } else if (*gmmXQuantMode == TT_QUANT_MODE_NUM && *gmmWeightQuantMode == TT_QUANT_MODE_NUM) {
        OPS_CHECK(gmmXScaleDimNum != DIM_NUM_1, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                  "In TT quantization mode, gmmXScale dim num must be %ld, but actual value is: %ld",
                  DIM_NUM_1, gmmXScaleDimNum), return ge::GRAPH_FAILED);
        OPS_CHECK(gmmWeightScaleDimNum != DIM_NUM_1, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                    "In TT quantization mode, gmmWeightScale dim num must be %ld, but actual value is: %ld",
                    DIM_NUM_1, gmmWeightScaleDimNum), return ge::GRAPH_FAILED);
        int64_t gmmXScaleShapeDim0 = gmmXScaleShape->GetStorageShape().GetDim(DIM_ZERO);
        OPS_CHECK(gmmXScaleShapeDim0 != DIM_NUM_1, CUBE_INNER_ERR_REPORT(context_->GetNodeName(),
                  "In TT quantization mode, gmmXScale input shape should be [1], "
                  "but actual shape is [%lu].", gmmXScaleShapeDim0), return ge::GRAPH_FAILED);
        int64_t gmmWeightScaleShapeDim0 = gmmWeightScaleShape->GetStorageShape().GetDim(DIM_ZERO);
        OPS_CHECK(gmmWeightScaleShapeDim0 != DIM_NUM_1, CUBE_INNER_ERR_REPORT(context_->GetNodeName(),
                  "In TT quantization mode, gmmWeightScale input shape should be [1], "
                  "but actual shape is [%lu].", gmmWeightScaleShapeDim0), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckOptionalScaleShapeForAlltoAllvQuantGroupedMatmul(const gert::InferShapeContext* context, const int64_t bs,
                                                                             const bool mmXQuantMode, const bool mmWeightQuantMode,
                                                                             const gert::Shape *mmXScaleShape, const gert::Shape *mmWeightScaleShape)
{
    //还有scale的H轴和N轴没有校验
    int64_t mmXScaleDimNum = mmXScaleShape->GetDimNum();
    int64_t mmWeightScaleDimNum = mmWeightScaleShape->GetDimNum();
    if (*mmXQuantMode == MX_QUANT_MODE_NUM && *mmWeightQuantMode == MX_QUANT_MODE_NUM) {
        OPS_CHECK(mmXScaleDimNum != DIM_NUM_3, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                  "In MX quantization mode, mmXScale dim num must be %ld, but actual value is: %ld",
                  DIM_NUM_3, mmXScaleDimNum), return ge::GRAPH_FAILED);
        OPS_CHECK(mmWeightScaleDimNum != DIM_NUM_3, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                  "In MX quantization mode, mmWeightScale dim num must be %ld, but actual value is: %ld",
                  DIM_NUM_3, mmWeightScaleDimNum), return ge::GRAPH_FAILED);
        OPS_CHECK(mmXScaleDimNum->GetDim(mmXScaleDimNum - 1) != SCALE_LAST_DIM,
                  CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                  "In MX quantization mode, mmXScale last dim must be %ld, but actual value is: %ld",
                  SCALE_LAST_DIM, mmXScaleDimNum->GetDim(mmXScaleDimNum - 1)), return ge::GRAPH_FAILED);
        OPS_CHECK(mmWeightScaleDimNum->GetDim(mmWeightScaleDimNum - 1) != SCALE_LAST_DIM,
                  CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                  "In MX quantization mode, mmWeightScale last dim must be %ld, but actual value is: %ld",
                  SCALE_LAST_DIM, mmWeightScaleDimNum->GetDim(mmWeightScaleDimNum - 1)), return ge::GRAPH_FAILED);
        OPS_CHECK(mmXScaleShape->GetDim(0) != bs, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                  "In MX quantization mode, mmXScale dim0 must be the same with mmX first dim, "
                  "but actual gmmXScale dim0 is: %ld, gmmX first dim is: %ld",
                  mmXScaleShape->GetDim(0), bs), return ge::GRAPH_FAILED);
    } else if (*mmXQuantMode == TT_QUANT_MODE_NUM && *mmWeightQuantMode == TT_QUANT_MODE_NUM) {
        OPS_CHECK(mmXScaleDimNum != DIM_NUM_1, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                  "In TT quantization mode, gmmXScale dim num must be %ld, but actual value is: %ld",
                  DIM_NUM_1, mmXScaleDimNum), return ge::GRAPH_FAILED);
        OPS_CHECK(mmWeightScaleDimNum != DIM_NUM_1, CUBE_INNER_ERR_REPORT(context->GetNodeName(),
                    "In TT quantization mode, mmWeightScale dim num must be %ld, but actual value is: %ld",
                    DIM_NUM_1, mmWeightScaleDimNum), return ge::GRAPH_FAILED);
        int64_t mmXScaleShapeDim0 = mmXScaleShape->GetStorageShape().GetDim(DIM_ZERO);
        OPS_CHECK(mmXScaleShapeDim0 != DIM_NUM_1, CUBE_INNER_ERR_REPORT(context_->GetNodeName(),
                  "In TT quantization mode, mmXScale input shape should be [1], "
                  "but actual shape is [%lu].", mmXScaleShapeDim0), return ge::GRAPH_FAILED);
        int64_t mmWeightScaleShapeDim0 = mmWeightScaleShape->GetStorageShape().GetDim(DIM_ZERO);
        OPS_CHECK(mmWeightScaleShapeDim0 != DIM_NUM_1, CUBE_INNER_ERR_REPORT(context_->GetNodeName(),
                  "In TT quantization mode, mmWeightScale input shape should be [1], "
                  "but actual shape is [%lu].", mmWeightScaleShapeDim0), return ge::GRAPH_FAILED);
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeAlltoAllvGroupedMatMul(gert::InferDataTypeContext *context)
{
    auto dType = context->GetInputDataType(INDEX_IN_GMM_X);
    auto *attrs = context->GetAttrs();
    const int64_t *yDtypePtr = nullptr;
    if (attrs->GetAttrNum() > INDEX_ATTR_Y_DTYPE_INDEX) {
        yDtypePtr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_Y_DTYPE_INDEX);
    }
    const int64_t *mmDtypePtr = nullptr;
    if (attrs->GetAttrNum() > INDEX_ATTR_MM_DTYPE_INDEX) {
        mmDtypePtr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_MM_DTYPE_INDEX);
    }
    ge::DataType yDataType = (yDtypePtr == nullptr || *yDtypePtr == 28) ? dType : static_cast<ge::DataType>(*yDtypePtr);
    ge::DataType mmDataType =
        (mmDtypePtr == nullptr || *mmDtypePtr == 28) ? dType : static_cast<ge::DataType>(*mmDtypePtr);
    context->SetOutputDataType(INDEX_OUT_MM_Y, mmDataType);
    OP_LOGD(context->GetNodeName(), "infershape mmY data type: %s.",
        TypeUtils::DataTypeToAscendString(mmDataType).GetString());
    context->SetOutputDataType(INDEX_OUT_GMM_Y, yDataType);
    OP_LOGD(context->GetNodeName(), "infershape gmmY data type: %s.",
        TypeUtils::DataTypeToAscendString(yDataType).GetString());
    context->SetOutputDataType(INDEX_PERMUTE_OUT, dType);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AlltoAllvQuantGroupedMatMul)
    .InferShape(InferShapeAlltoAllvGroupedMatMul)
    .InferDataType(InferDataTypeAlltoAllvGroupedMatMul);
} // namespace ops
