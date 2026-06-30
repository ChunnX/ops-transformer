/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_grouped_matmul_activation_quant_weight_nz.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "grouped_matmul_activation_quant.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "opdev/shape_utils.h"
#include "util/math_util.h"

using namespace op;

namespace {
constexpr char OP_NAME[] = "aclnnGroupedMatmulActivationQuantWeightNz";
constexpr char GELU_TANH[] = "gelu_tanh";
constexpr char QUANT_MODE_MX[] = "MX";
constexpr char ROUND_MODE_RINT[] = "rint";
constexpr size_t SINGLE_TENSOR_SIZE = 1UL;
constexpr size_t FIRST_TENSOR_INDEX = 0UL;
constexpr size_t X_DIM_NUM = 2UL;
constexpr size_t SCALE_DIM_NUM = 3UL;
constexpr size_t WEIGHT_SCALE_DIM_NUM = 4UL;
constexpr size_t GROUP_LIST_DIM_NUM = 1UL;
constexpr size_t OUT_DIM_NUM = 2UL;
constexpr size_t OUT_SCALE_DIM_NUM = 3UL;
constexpr size_t WEIGHT_LOGICAL_DIM_NUM = 3UL;
constexpr size_t WEIGHT_NZ_DIM_NUM = 5UL;
constexpr int64_t DIM_0 = 0L;
constexpr int64_t DIM_1 = 1L;
constexpr int64_t DIM_2 = 2L;
constexpr int64_t DIM_3 = 3L;
constexpr int64_t MX_GROUP_SIZE = 64L;
constexpr int64_t MX_SCALE_PAIR = 2L;
constexpr int64_t NZ_C0_DIM = 16L;
constexpr int64_t NZ_LAST_DIM = 32L;
constexpr int64_t MAX_GROUP_NUM = 1024L;
constexpr int64_t FLOAT8_E5M2_VALUE = static_cast<int64_t>(DataType::DT_FLOAT8_E5M2);
constexpr int64_t FLOAT8_E4M3FN_VALUE = static_cast<int64_t>(DataType::DT_FLOAT8_E4M3FN);
constexpr int64_t SCALE_ALG_OCP = 0L;
constexpr int64_t SCALE_ALG_CUBLAS = 1L;
constexpr float DEFAULT_DST_TYPE_MAX = 0.0F;
constexpr float FLOAT_EPS = 1e-6F;

template <typename... Args>
void GmmaqPrint(const char *format, Args... args)
{
    std::printf("[GMMAQ aclnn] ");
    std::printf(format, args...);
    std::printf("\n");
    std::fflush(stdout);
}

bool IsFp8(DataType dtype)
{
    return dtype == DataType::DT_FLOAT8_E4M3FN || dtype == DataType::DT_FLOAT8_E5M2;
}

bool IsFp8DtypeValue(int64_t dtype)
{
    return dtype == FLOAT8_E4M3FN_VALUE || dtype == FLOAT8_E5M2_VALUE;
}

bool IsMxScaleDtype(DataType dtype)
{
    return dtype == DataType::DT_FLOAT8_E8M0;
}

bool IsEmptyQuantMode(const char *quantMode)
{
    return quantMode == nullptr || quantMode[0] == '\0';
}

aclnnStatus CheckMxQuantDtype(const aclTensor *x, const aclTensor *weight, const aclTensor *xScaleOptional,
                              const aclTensor *weightScale, const aclTensor *y, const aclTensor *yScale,
                              int64_t yDtype)
{
    CHECK_COND(IsFp8(x->GetDataType()), ACLNN_ERR_PARAM_INVALID,
               "In op [%s], when the quant mode is MX, x only supports FLOAT8_E4M3FN or FLOAT8_E5M2.", OP_NAME);
    CHECK_COND(weight->GetDataType() == DataType::DT_FLOAT8_E4M3FN, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], when the quant mode is MX, weight only supports FLOAT8_E4M3FN.", OP_NAME);
    CHECK_COND(IsMxScaleDtype(xScaleOptional->GetDataType()), ACLNN_ERR_PARAM_INVALID,
               "In op [%s], when the quant mode is MX, xScaleOptional only supports FLOAT8_E8M0.", OP_NAME);
    CHECK_COND(IsMxScaleDtype(weightScale->GetDataType()), ACLNN_ERR_PARAM_INVALID,
               "In op [%s], when the quant mode is MX, weightScale only supports FLOAT8_E8M0.", OP_NAME);
    CHECK_COND(IsFp8(y->GetDataType()), ACLNN_ERR_PARAM_INVALID,
               "In op [%s], when the quant mode is MX, y only supports FLOAT8_E4M3FN or FLOAT8_E5M2.", OP_NAME);
    CHECK_COND(y->GetDataType() == static_cast<DataType>(yDtype), ACLNN_ERR_PARAM_INVALID,
               "In op [%s], when the quant mode is MX, y dtype must be the same as yDtype.", OP_NAME);
    CHECK_COND(IsMxScaleDtype(yScale->GetDataType()), ACLNN_ERR_PARAM_INVALID,
               "In op [%s], when the quant mode is MX, yScale only supports FLOAT8_E8M0.", OP_NAME);
    return ACLNN_SUCCESS;
}

bool IsNzFormat(const aclTensor *tensor)
{
    return tensor != nullptr &&
           ge::GetPrimaryFormat(tensor->GetStorageFormat()) == op::Format::FORMAT_FRACTAL_NZ;
}

bool IsEmptyTensorList(const aclTensorList *tensorList)
{
    if (tensorList == nullptr || tensorList->Size() == 0) {
        return true;
    }
    if (tensorList->Size() != SINGLE_TENSOR_SIZE || (*tensorList)[FIRST_TENSOR_INDEX] == nullptr) {
        return false;
    }
    const auto &shape = (*tensorList)[FIRST_TENSOR_INDEX]->GetViewShape();
    return shape.GetDimNum() == 1 && shape.GetDim(DIM_0) == 0;
}

void PrintTensorDebugInfo(const char *stage, const char *name, const aclTensor *tensor)
{
    if (tensor == nullptr) {
        GmmaqPrint("%s tensor %s: nullptr.", stage, name);
        return;
    }
    GmmaqPrint("%s tensor %s: dtype=%s, storageFormat=%s(primary=%s), viewShape=%s, storageShape=%s, "
               "viewDim=%zu, storageDim=%zu, isEmpty=%d.",
               stage, name, op::ToString(tensor->GetDataType()).GetString(),
               op::ToString(tensor->GetStorageFormat()).GetString(),
               op::ToString(static_cast<op::Format>(ge::GetPrimaryFormat(tensor->GetStorageFormat()))).GetString(),
               op::ToString(tensor->GetViewShape()).GetString(),
               op::ToString(tensor->GetStorageShape()).GetString(),
               tensor->GetViewShape().GetDimNum(), tensor->GetStorageShape().GetDimNum(),
               static_cast<int>(tensor->IsEmpty()));
}

void PrintTensorListDebugInfo(const char *stage, const char *name, const aclTensorList *tensorList)
{
    if (tensorList == nullptr) {
        GmmaqPrint("%s tensorList %s: nullptr.", stage, name);
        return;
    }
    GmmaqPrint("%s tensorList %s: size=%zu.", stage, name, tensorList->Size());
    for (size_t i = 0; i < tensorList->Size(); ++i) {
        char tensorName[128] = {};
        std::snprintf(tensorName, sizeof(tensorName), "%s[%zu]", name, i);
        PrintTensorDebugInfo(stage, tensorName, (*tensorList)[i]);
    }
}

void PrintAclnnInputDebugInfo(const char *stage, const aclTensor *x, const aclTensor *groupList,
    const aclTensorList *weight, const aclTensorList *weightScale, const aclTensorList *bias,
    const aclTensor *xScaleOptional, const aclTensor *y, const aclTensor *yScale)
{
    GmmaqPrint("%s tensor snapshot.", stage);
    PrintTensorDebugInfo(stage, "x", x);
    PrintTensorDebugInfo(stage, "groupList", groupList);
    PrintTensorListDebugInfo(stage, "weight", weight);
    PrintTensorListDebugInfo(stage, "weightScale", weightScale);
    PrintTensorListDebugInfo(stage, "bias", bias);
    PrintTensorDebugInfo(stage, "xScaleOptional", xScaleOptional);
    PrintTensorDebugInfo(stage, "y", y);
    PrintTensorDebugInfo(stage, "yScale", yScale);
}

void NormalizeEmptyTensorList(const aclTensorList *&tensorList)
{
    if (IsEmptyTensorList(tensorList)) {
        tensorList = nullptr;
    }
}

aclnnStatus CheckTensorList(const aclTensorList *tensorList, const char *name)
{
    CHECK_COND(tensorList != nullptr, ACLNN_ERR_PARAM_NULLPTR, "In op [%s], [%s] must not be nullptr.",
               OP_NAME, name);
    CHECK_COND(tensorList->Size() == SINGLE_TENSOR_SIZE, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], [%s] only supports one tensor in current MX quant mode, but got %zu.",
               OP_NAME, name, tensorList->Size());
    CHECK_COND((*tensorList)[FIRST_TENSOR_INDEX] != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "In op [%s], [%s][0] must not be nullptr.", OP_NAME, name);
    return ACLNN_SUCCESS;
}

aclnnStatus CreateEmptyTensorList(const aclDataType dataType, const aclTensorList *&tensorList,
                                  aclOpExecutor *executor, const char *name)
{
    if (tensorList != nullptr) {
        return ACLNN_SUCCESS;
    }
    std::vector<aclTensor *> emptyTensors;
    aclTensor *emptyTensor = executor->AllocTensor({0}, static_cast<DataType>(dataType));
    CHECK_RET(emptyTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    emptyTensors.emplace_back(emptyTensor);
    tensorList = executor->AllocTensorList(emptyTensors.data(), emptyTensors.size());
    CHECK_RET(tensorList != nullptr, ACLNN_ERR_INNER_NULLPTR);
    GmmaqPrint("In op [%s], create empty tensorList for %s. size=%zu, dtype=%s, format=%s, shape=%s.",
               OP_NAME, name, tensorList->Size(),
               op::ToString((*tensorList)[FIRST_TENSOR_INDEX]->GetDataType()).GetString(),
               op::ToString((*tensorList)[FIRST_TENSOR_INDEX]->GetStorageFormat()).GetString(),
               op::ToString((*tensorList)[FIRST_TENSOR_INDEX]->GetViewShape()).GetString());
    return ACLNN_SUCCESS;
}

aclnnStatus WrapWeightNzTensorList(const aclTensorList *&weight, aclOpExecutor *executor)
{
    CHECK_RET(weight != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(executor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    std::vector<const aclTensor *> tensors;
    tensors.reserve(weight->Size());
    for (size_t i = 0; i < weight->Size(); ++i) {
        const aclTensor *tensor = (*weight)[i];
        CHECK_RET(tensor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        PrintTensorDebugInfo("WrapWeightNzTensorList input", "weight", tensor);
        op::Shape viewShape = tensor->GetViewShape();
        op::Shape storageShape = tensor->GetStorageShape();
        auto wrappedTensor = executor->CreateView(tensor, viewShape, tensor->GetViewOffset());
        CHECK_RET(wrappedTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        wrappedTensor->SetStorageFormat(op::Format::FORMAT_FRACTAL_NZ);
        wrappedTensor->SetOriginalFormat(tensor->GetViewFormat());
        wrappedTensor->SetViewShape(tensor->GetViewShape());
        wrappedTensor->SetStorageShape(storageShape);
        PrintTensorDebugInfo("WrapWeightNzTensorList output", "weight", wrappedTensor);
        tensors.emplace_back(wrappedTensor);
    }
    weight = executor->AllocTensorList(tensors.data(), tensors.size());
    CHECK_RET(weight != nullptr, ACLNN_ERR_INNER_NULLPTR);
    PrintTensorListDebugInfo("after WrapWeightNzTensorList", "weight", weight);
    return ACLNN_SUCCESS;
}

int64_t CeilDivInt64(int64_t dividend, int64_t divisor)
{
    return static_cast<int64_t>(Ops::Base::CeilDiv(dividend, divisor));
}

bool IsTransposeLastTwoDims(const aclTensor *tensor)
{
    auto shape = tensor->GetViewShape();
    if (shape.GetDimNum() < X_DIM_NUM) {
        return false;
    }
    int64_t lastDim = static_cast<int64_t>(shape.GetDimNum()) - 1;
    int64_t penultimateDim = static_cast<int64_t>(shape.GetDimNum()) - 2;
    auto strides = tensor->GetViewStrides();
    if (strides[penultimateDim] != 1 || strides[lastDim] != shape.GetDim(penultimateDim)) {
        return false;
    }
    int64_t expectedStride = shape.GetDim(lastDim) * shape.GetDim(penultimateDim);
    for (int64_t batchDim = static_cast<int64_t>(shape.GetDimNum()) - 3; batchDim >= 0; --batchDim) {
        if (strides[batchDim] != expectedStride) {
            return false;
        }
        expectedStride *= shape.GetDim(batchDim);
    }
    return true;
}

bool IsTransposeForMxShape(const aclTensor *tensor)
{
    auto shape = tensor->GetViewShape();
    if (shape.GetDimNum() < WEIGHT_SCALE_DIM_NUM) {
        return false;
    }
    auto strides = tensor->GetViewStrides();
    int64_t lastDim = static_cast<int64_t>(shape.GetDimNum()) - 1;
    int64_t secondLastDim = static_cast<int64_t>(shape.GetDimNum()) - 2;
    int64_t thirdLastDim = static_cast<int64_t>(shape.GetDimNum()) - 3;
    return strides[lastDim] == 1 && strides[thirdLastDim] == MX_SCALE_PAIR &&
           strides[secondLastDim] == shape.GetDim(thirdLastDim) * MX_SCALE_PAIR;
}

bool TryInferTransposeByShape(const aclTensor *x, const aclTensor *groupList, const aclTensor *weight,
                              const aclTensor *weightScale, bool inputTransposeWeight, bool &transposeWeight)
{
    const auto &weightShape = weight->GetViewShape();
    const auto &weightScaleShape = weightScale->GetViewShape();
    if (weightShape.GetDimNum() != WEIGHT_LOGICAL_DIM_NUM ||
        weightScaleShape.GetDimNum() != WEIGHT_SCALE_DIM_NUM) {
        return false;
    }
    int64_t e = groupList->GetViewShape().GetDim(DIM_0);
    int64_t k = x->GetViewShape().GetDim(DIM_1);

    int64_t nonTransN = weightScaleShape.GetDim(DIM_2);
    bool nonTransMatched = weightShape.GetDim(DIM_0) == e &&
                           weightShape.GetDim(DIM_1) == k &&
                           weightShape.GetDim(DIM_2) == nonTransN &&
                           weightScaleShape.GetDim(DIM_0) == e &&
                           weightScaleShape.GetDim(DIM_1) == CeilDivInt64(k, MX_GROUP_SIZE) &&
                           weightScaleShape.GetDim(DIM_3) == MX_SCALE_PAIR;

    int64_t transN = weightScaleShape.GetDim(DIM_1);
    bool transMatched = weightShape.GetDim(DIM_0) == e &&
                        weightShape.GetDim(DIM_1) == transN &&
                        weightShape.GetDim(DIM_2) == k &&
                        weightScaleShape.GetDim(DIM_0) == e &&
                        weightScaleShape.GetDim(DIM_2) == CeilDivInt64(k, MX_GROUP_SIZE) &&
                        weightScaleShape.GetDim(DIM_3) == MX_SCALE_PAIR;

    GmmaqPrint("In op [%s], infer transpose by shape. weight=%s, weightScale=%s, nonTransN=%ld matched=%d, "
               "transN=%ld matched=%d, inputAttr=%d.",
               OP_NAME, op::ToString(weightShape).GetString(), op::ToString(weightScaleShape).GetString(),
               nonTransN, static_cast<int>(nonTransMatched), transN, static_cast<int>(transMatched),
               static_cast<int>(inputTransposeWeight));
    if (!nonTransMatched && !transMatched) {
        return false;
    }
    if (nonTransMatched && transMatched) {
        transposeWeight = inputTransposeWeight;
        return true;
    }
    transposeWeight = transMatched;
    return true;
}

aclnnStatus CreateContiguousTensorListForMXTypeMScale(const aclTensorList *tensorList,
                                                      const aclTensorList *&newTensorList,
                                                      aclOpExecutor *executor)
{
    std::vector<aclTensor *> tensors;
    tensors.reserve(tensorList->Size());
    for (size_t idx = 0; idx < tensorList->Size(); ++idx) {
        const aclTensor *inputTensor = (*tensorList)[idx];
        CHECK_RET(inputTensor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        op::Shape viewShape = inputTensor->GetViewShape();
        op::Shape shape;
        shape.SetScalar();
        CHECK_COND(viewShape.GetDimNum() >= WEIGHT_SCALE_DIM_NUM, ACLNN_ERR_PARAM_INVALID,
                   "In op [%s], weightScale view shape dim num must be greater than or equal to 4.", OP_NAME);
        shape.AppendDim(viewShape.GetDim(DIM_0));
        shape.AppendDim(viewShape.GetDim(viewShape.GetDimNum() - DIM_2));
        shape.AppendDim(viewShape.GetDim(viewShape.GetDimNum() - DIM_3));
        shape.AppendDim(viewShape.GetDim(viewShape.GetDimNum() - DIM_1));
        GmmaqPrint("In op [%s], CreateView for transposed weightScale[%zu], old shape=%s, new shape=%s.",
                   OP_NAME, idx, op::ToString(viewShape).GetString(), op::ToString(shape).GetString());
        aclTensor *tensor = executor->CreateView(inputTensor, shape, inputTensor->GetViewOffset());
        CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        tensor->SetStorageFormat(inputTensor->GetStorageFormat());
        tensors.emplace_back(tensor);
    }
    newTensorList = executor->AllocTensorList(tensors.data(), tensors.size());
    CHECK_RET(newTensorList != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus CreateContiguousTensorList(const aclTensorList *tensorList, const aclTensorList *&newTensorList,
                                       aclOpExecutor *executor)
{
    std::vector<aclTensor *> tensors;
    tensors.reserve(tensorList->Size());
    for (size_t idx = 0; idx < tensorList->Size(); ++idx) {
        const aclTensor *inputTensor = (*tensorList)[idx];
        CHECK_RET(inputTensor != nullptr, ACLNN_ERR_PARAM_NULLPTR);
        op::Shape viewShape = inputTensor->GetViewShape();
        uint32_t viewShapeDimsNum = viewShape.GetDimNum();
        op::Shape shape;
        shape.SetScalar();
        CHECK_COND(viewShapeDimsNum >= X_DIM_NUM, ACLNN_ERR_PARAM_INVALID,
                   "In op [%s], weight view shape dim num must be greater than or equal to 2.", OP_NAME);
        for (uint32_t i = 0; i < viewShapeDimsNum - X_DIM_NUM; ++i) {
            shape.AppendDim(viewShape.GetDim(i));
        }
        shape.AppendDim(viewShape.GetDim(viewShapeDimsNum - 1));
        shape.AppendDim(viewShape.GetDim(viewShapeDimsNum - 2));
        GmmaqPrint("In op [%s], CreateView for transposed weight[%zu], old shape=%s, new shape=%s.",
                   OP_NAME, idx, op::ToString(viewShape).GetString(), op::ToString(shape).GetString());
        aclTensor *tensor = executor->CreateView(inputTensor, shape, inputTensor->GetViewOffset());
        CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        tensor->SetStorageFormat(inputTensor->GetStorageFormat());
        tensor->SetStorageShape(inputTensor->GetStorageShape());
        tensors.emplace_back(tensor);
    }
    newTensorList = executor->AllocTensorList(tensors.data(), tensors.size());
    CHECK_RET(newTensorList != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckMXTranspose(const aclTensor *x, const aclTensor *groupList, const aclTensor *xScaleOptional,
                             const aclTensorList *&weight, const aclTensorList *&weightScale, bool &transposeWeight,
                             aclOpExecutor *executor)
{
    const aclTensor *w = (*weight)[FIRST_TENSOR_INDEX];
    const aclTensor *wScale = (*weightScale)[FIRST_TENSOR_INDEX];
    bool transposeWeightScale = IsTransposeForMxShape(wScale);
    bool transposeWeightByStride = IsTransposeLastTwoDims(w);
    bool transposeX = IsTransposeLastTwoDims(x);
    bool transposeXScale = IsTransposeForMxShape(xScaleOptional);
    GmmaqPrint("In op [%s], CheckMXTranspose. weight=%d, weightScale=%d, x=%d, xScale=%d, inputAttr=%d.",
               OP_NAME, static_cast<int>(transposeWeightByStride), static_cast<int>(transposeWeightScale),
               static_cast<int>(transposeX), static_cast<int>(transposeXScale), static_cast<int>(transposeWeight));

    CHECK_COND(transposeWeightScale == transposeWeightByStride, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], the transposition of weightScale must be equal to the transposition of weight.",
               OP_NAME);
    CHECK_COND(!transposeX && !transposeXScale, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], x and xScaleOptional only support non-transposed input.", OP_NAME);

    if (transposeWeightScale && transposeWeightByStride) {
        transposeWeight = true;
        CHECK_RET(CreateContiguousTensorListForMXTypeMScale(weightScale, weightScale, executor) == ACLNN_SUCCESS,
                  ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(CreateContiguousTensorList(weight, weight, executor) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
        PrintTensorListDebugInfo("after CheckMXTranspose CreateView", "weight", weight);
        PrintTensorListDebugInfo("after CheckMXTranspose CreateView", "weightScale", weightScale);
        return ACLNN_SUCCESS;
    }

    bool shapeTransposeWeight = transposeWeight;
    if (TryInferTransposeByShape(x, groupList, w, wScale, transposeWeight, shapeTransposeWeight)) {
        transposeWeight = shapeTransposeWeight;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckMxQuantShape(const aclTensor *x, const aclTensor *groupList, const aclTensor *xScaleOptional,
                              const aclTensor *weight, const aclTensor *weightScale, bool transposeWeight,
                              const aclTensor *y, const aclTensor *yScale)
{
    int64_t m = x->GetViewShape().GetDim(DIM_0);
    int64_t k = x->GetViewShape().GetDim(DIM_1);
    int64_t e = groupList->GetViewShape().GetDim(DIM_0);
    int64_t n = transposeWeight ? weightScale->GetViewShape().GetDim(DIM_1) :
                                           weightScale->GetViewShape().GetDim(DIM_2);
    GmmaqPrint("In op [%s], CheckMxQuantShape start. M=%ld, K=%ld, E=%ld, N=%ld, transposeWeight=%d.",
               OP_NAME, m, k, e, n, static_cast<int>(transposeWeight));
    CHECK_COND(e >= 1 && e <= MAX_GROUP_NUM, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], the first dim of groupList must be in range [1, 1024], but got %ld.", OP_NAME, e);
    CHECK_COND(m == 0 || n == 0 || k > 0, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], K must be greater than 0 unless M or N is 0.", OP_NAME);
    CHECK_COND(n % MX_GROUP_SIZE == 0, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], N must be divisible by 64, but got %ld.", OP_NAME, n);

    const auto &weightShape = weight->GetStorageShape();
    const auto &weightViewShape = weight->GetViewShape();
    const auto &weightScaleShape = weightScale->GetViewShape();
    CHECK_COND(weightShape.GetDim(DIM_0) == e, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], weight dim 0 must equal E, but got %ld and %ld.", OP_NAME,
               weightShape.GetDim(DIM_0), e);
    CHECK_COND(weightScaleShape.GetDim(DIM_0) == e, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], weightScale dim 0 must equal E, but got %ld and %ld.", OP_NAME,
               weightScaleShape.GetDim(DIM_0), e);

    op::Shape xScaleExpectShape = {m, CeilDivInt64(k, MX_GROUP_SIZE), MX_SCALE_PAIR};
    op::Shape yExpectShape = {m, n};
    op::Shape yScaleExpectShape = {m, CeilDivInt64(n, MX_GROUP_SIZE), MX_SCALE_PAIR};
    GmmaqPrint("In op [%s], expected xScale shape %s, y shape %s, yScale shape %s.", OP_NAME,
               op::ToString(xScaleExpectShape).GetString(), op::ToString(yExpectShape).GetString(),
               op::ToString(yScaleExpectShape).GetString());
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(xScaleOptional, xScaleExpectShape, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(y, yExpectShape, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(yScale, yScaleExpectShape, return ACLNN_ERR_PARAM_INVALID);

    op::Shape weightExpectShape;
    op::Shape weightScaleExpectShape;
    if (transposeWeight) {
        weightExpectShape = {e, CeilDivInt64(k, NZ_LAST_DIM), CeilDivInt64(n, NZ_C0_DIM), NZ_C0_DIM, NZ_LAST_DIM};
        weightScaleExpectShape = {e, n, CeilDivInt64(k, MX_GROUP_SIZE), MX_SCALE_PAIR};
    } else {
        weightExpectShape = {e, CeilDivInt64(n, NZ_LAST_DIM), CeilDivInt64(k, NZ_C0_DIM), NZ_C0_DIM, NZ_LAST_DIM};
        weightScaleExpectShape = {e, CeilDivInt64(k, MX_GROUP_SIZE), n, MX_SCALE_PAIR};
    }
    GmmaqPrint("In op [%s], expected weight storage shape %s, weightScale shape %s.", OP_NAME,
               op::ToString(weightExpectShape).GetString(), op::ToString(weightScaleExpectShape).GetString());
    if (weightViewShape.GetDimNum() == WEIGHT_LOGICAL_DIM_NUM) {
        op::Shape weightViewExpectShape = transposeWeight ? op::Shape({e, n, k}) : op::Shape({e, k, n});
        GmmaqPrint("In op [%s], expected weight logical view shape %s.", OP_NAME,
                   op::ToString(weightViewExpectShape).GetString());
        CHECK_COND(weightViewShape == weightViewExpectShape, ACLNN_ERR_PARAM_INVALID,
                   "In op [%s], expected weight logical view shape to be %s, but got %s.", OP_NAME,
                   op::ToString(weightViewExpectShape).GetString(), op::ToString(weightViewShape).GetString());
    }
    CHECK_COND(weightShape == weightExpectShape, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], expected weight storage shape to be %s, but got %s.", OP_NAME,
               op::ToString(weightExpectShape).GetString(), op::ToString(weightShape).GetString());
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(weightScale, weightScaleExpectShape, return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckRequiredParams(const aclTensor *x, const aclTensor *groupList, const aclTensorList *weight,
                                const aclTensorList *weightScale, const char *activationType, const aclTensor *y,
                                const aclTensor *yScale)
{
    CHECK_COND(x != nullptr, ACLNN_ERR_PARAM_NULLPTR, "In op [%s], [x] must not be nullptr.", OP_NAME);
    CHECK_COND(groupList != nullptr, ACLNN_ERR_PARAM_NULLPTR, "In op [%s], [groupList] must not be nullptr.",
               OP_NAME);
    aclnnStatus checkTensorListRet = CheckTensorList(weight, "weight");
    CHECK_RET(checkTensorListRet == ACLNN_SUCCESS, checkTensorListRet);
    checkTensorListRet = CheckTensorList(weightScale, "weightScale");
    CHECK_RET(checkTensorListRet == ACLNN_SUCCESS, checkTensorListRet);
    CHECK_COND(y != nullptr, ACLNN_ERR_PARAM_NULLPTR, "In op [%s], [y] must not be nullptr.", OP_NAME);
    CHECK_COND(yScale != nullptr, ACLNN_ERR_PARAM_NULLPTR, "In op [%s], [yScale] must not be nullptr.", OP_NAME);
    CHECK_COND(activationType != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "In op [%s], [activationType] must not be nullptr.", OP_NAME);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckActivationType(const char *activationType)
{
    CHECK_COND(std::strcmp(activationType, GELU_TANH) == 0, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], activationType only supports gelu_tanh, but got %s.", OP_NAME, activationType);
    return ACLNN_SUCCESS;
}

aclnnStatus ResolveAndCheckQuantMode(const aclTensor *x, const aclTensor *xScaleOptional, const char *quantMode,
                                     const char *&effectiveQuantMode)
{
    if (!IsEmptyQuantMode(quantMode)) {
        CHECK_COND(std::strcmp(quantMode, QUANT_MODE_MX) == 0, ACLNN_ERR_PARAM_INVALID,
                   "In op [%s], quant_mode 参数当前仅支持为 MX", OP_NAME);
        effectiveQuantMode = quantMode;
        return ACLNN_SUCCESS;
    }

    if (xScaleOptional != nullptr && IsFp8(x->GetDataType()) && IsMxScaleDtype(xScaleOptional->GetDataType())) {
        effectiveQuantMode = QUANT_MODE_MX;
        return ACLNN_SUCCESS;
    }

    CHECK_COND(false, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], 当前数据类型组合无法匹配到该算子支持的任何一种量化模式，x数据类型：%s，"
               "x_scale数据类型：%s",
               OP_NAME, op::ToString(x->GetDataType()).GetString(),
               xScaleOptional == nullptr ? "nullptr" : op::ToString(xScaleOptional->GetDataType()).GetString());
    return ACLNN_ERR_PARAM_INVALID;
}

aclnnStatus ResolveAndCheckRoundMode(const char *roundMode, const char *&effectiveRoundMode)
{
    if (roundMode == nullptr || roundMode[0] == '\0') {
        effectiveRoundMode = ROUND_MODE_RINT;
        return ACLNN_SUCCESS;
    }
    CHECK_COND(std::strcmp(roundMode, ROUND_MODE_RINT) == 0, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], MX quant only supports roundMode rint, but got %s.",
               OP_NAME, roundMode);
    effectiveRoundMode = roundMode;
    return ACLNN_SUCCESS;
}

aclnnStatus CheckMxQuantAttrs(int64_t groupListType, int64_t yDtype, const char *roundMode,
                              const char *&effectiveRoundMode, int64_t scaleAlg, float dstTypeMax,
                              const aclTensorList *bias)
{
    CHECK_RET(ResolveAndCheckRoundMode(roundMode, effectiveRoundMode) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(IsFp8DtypeValue(yDtype), ACLNN_ERR_PARAM_INVALID,
               "In op [%s], yDtype only supports FLOAT8_E4M3FN or FLOAT8_E5M2, but got %ld.",
               OP_NAME, yDtype);
    CHECK_COND(groupListType == 0 || groupListType == 1, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], groupListType only supports 0 or 1, but got %ld.",
               OP_NAME, groupListType);
    CHECK_COND(scaleAlg == SCALE_ALG_OCP || scaleAlg == SCALE_ALG_CUBLAS, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], scaleAlg only supports 0 or 1 in MX quant mode, but got %ld.",
               OP_NAME, scaleAlg);
    CHECK_COND(std::fabs(dstTypeMax - DEFAULT_DST_TYPE_MAX) <= FLOAT_EPS, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], dstTypeMax only supports 0.0 in current MX quant mode, but got %f.",
               OP_NAME, dstTypeMax);
    CHECK_COND(IsEmptyTensorList(bias), ACLNN_ERR_PARAM_INVALID,
               "In op [%s], non-empty bias is not supported in current MX quant mode.", OP_NAME);
    GmmaqPrint("In op [%s], MX attribute check passed. bias size=%zu.", OP_NAME,
               bias == nullptr ? 0UL : bias->Size());
    return ACLNN_SUCCESS;
}

aclnnStatus CheckMxQuantDim(const aclTensor *x, const aclTensor *groupList, const aclTensor *weight,
                            const aclTensor *weightScale, const aclTensor *xScaleOptional,
                            const aclTensor *y, const aclTensor *yScale)
{
    GmmaqPrint("In op [%s], tensor dims: x=%zu, xScale=%zu, weight view=%zu, weight storage=%zu, "
               "weightScale=%zu, groupList=%zu, y=%zu, yScale=%zu.",
               OP_NAME, x->GetViewShape().GetDimNum(), xScaleOptional->GetViewShape().GetDimNum(),
               weight->GetViewShape().GetDimNum(), weight->GetStorageShape().GetDimNum(),
               weightScale->GetViewShape().GetDimNum(), groupList->GetViewShape().GetDimNum(),
               y->GetViewShape().GetDimNum(), yScale->GetViewShape().GetDimNum());
    OP_CHECK_WRONG_DIMENSION(x, X_DIM_NUM, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(xScaleOptional, SCALE_DIM_NUM, return ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(weight->GetViewShape().GetDimNum() == WEIGHT_LOGICAL_DIM_NUM,
               ACLNN_ERR_PARAM_INVALID,
               "In op [%s], weight view shape dim num must be 3, but got %zu.", OP_NAME,
               weight->GetViewShape().GetDimNum());
    OP_CHECK_WRONG_DIMENSION(weightScale, WEIGHT_SCALE_DIM_NUM, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(groupList, GROUP_LIST_DIM_NUM, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(y, OUT_DIM_NUM, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(yScale, OUT_SCALE_DIM_NUM, return ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(weight->GetStorageShape().GetDimNum() == WEIGHT_NZ_DIM_NUM, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], weight storage shape dim num must be 5.", OP_NAME);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckMxQuantFormat(const aclTensor *x, const aclTensor *groupList, const aclTensor *weight,
                               const aclTensor *weightScale, const aclTensor *xScaleOptional,
                               const aclTensor *y, const aclTensor *yScale)
{
    GmmaqPrint("In op [%s], storage formats: x=%s(primary=%s), xScaleOptional=%s(primary=%s), "
               "groupList=%s(primary=%s), weightScale=%s(primary=%s), y=%s(primary=%s), yScale=%s(primary=%s).",
               OP_NAME, op::ToString(x->GetStorageFormat()).GetString(),
               op::ToString(static_cast<op::Format>(ge::GetPrimaryFormat(x->GetStorageFormat()))).GetString(),
               op::ToString(xScaleOptional->GetStorageFormat()).GetString(),
               op::ToString(static_cast<op::Format>(ge::GetPrimaryFormat(xScaleOptional->GetStorageFormat()))).GetString(),
               op::ToString(groupList->GetStorageFormat()).GetString(),
               op::ToString(static_cast<op::Format>(ge::GetPrimaryFormat(groupList->GetStorageFormat()))).GetString(),
               op::ToString(weightScale->GetStorageFormat()).GetString(),
               op::ToString(static_cast<op::Format>(ge::GetPrimaryFormat(weightScale->GetStorageFormat()))).GetString(),
               op::ToString(y->GetStorageFormat()).GetString(),
               op::ToString(static_cast<op::Format>(ge::GetPrimaryFormat(y->GetStorageFormat()))).GetString(),
               op::ToString(yScale->GetStorageFormat()).GetString(),
               op::ToString(static_cast<op::Format>(ge::GetPrimaryFormat(yScale->GetStorageFormat()))).GetString());
    CHECK_COND(!op::IsPrivateFormat(x->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID,
               "In op [%s], x only supports ND, but got %s.", OP_NAME,
               op::ToString(x->GetStorageFormat()).GetString());
    CHECK_COND(!op::IsPrivateFormat(xScaleOptional->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID,
               "In op [%s], xScaleOptional only supports ND, but got %s.", OP_NAME,
               op::ToString(xScaleOptional->GetStorageFormat()).GetString());
    CHECK_COND(!op::IsPrivateFormat(groupList->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID,
               "In op [%s], groupList only supports ND, but got %s.", OP_NAME,
               op::ToString(groupList->GetStorageFormat()).GetString());
    CHECK_COND(!op::IsPrivateFormat(weightScale->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID,
               "In op [%s], weightScale only supports ND, but got %s.", OP_NAME,
               op::ToString(weightScale->GetStorageFormat()).GetString());
    CHECK_COND(!op::IsPrivateFormat(y->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID,
               "In op [%s], y only supports ND, but got %s.", OP_NAME,
               op::ToString(y->GetStorageFormat()).GetString());
    CHECK_COND(!op::IsPrivateFormat(yScale->GetStorageFormat()), ACLNN_ERR_PARAM_INVALID,
               "In op [%s], yScale only supports ND, but got %s.", OP_NAME,
               op::ToString(yScale->GetStorageFormat()).GetString());
    CHECK_COND(IsNzFormat(weight), ACLNN_ERR_PARAM_INVALID,
               "In op [%s], weight only supports FRACTAL_NZ format.", OP_NAME);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckMxQuantParams(const aclTensor *x, const aclTensor *groupList, const aclTensorList *&weight,
                               const aclTensorList *&weightScale, const aclTensorList *bias,
                               const aclTensor *xScaleOptional, bool &transposeWeight, int64_t groupListType,
                               int64_t yDtype, const char *roundMode, const char *&effectiveRoundMode,
                               int64_t scaleAlg, float dstTypeMax, const aclTensor *y, const aclTensor *yScale,
                               aclOpExecutor *executor)
{
    CHECK_COND(xScaleOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "In op [%s], [xScaleOptional] must not be nullptr in gelu_tanh MX quant mode.", OP_NAME);
    const aclTensor *w = (*weight)[FIRST_TENSOR_INDEX];
    const aclTensor *wScale = (*weightScale)[FIRST_TENSOR_INDEX];

    aclnnStatus checkRet = CheckMxQuantAttrs(groupListType, yDtype, roundMode, effectiveRoundMode, scaleAlg,
                                             dstTypeMax, bias);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);
    checkRet = CheckMxQuantDim(x, groupList, w, wScale, xScaleOptional, y, yScale);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);
    checkRet = CheckMxQuantDtype(x, w, xScaleOptional, wScale, y, yScale, yDtype);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);
    CHECK_COND(groupList->GetDataType() == DataType::DT_INT64, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], groupList only supports INT64.", OP_NAME);
    checkRet = CheckMxQuantFormat(x, groupList, w, wScale, xScaleOptional, y, yScale);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);
    checkRet = CheckMXTranspose(x, groupList, xScaleOptional, weight, weightScale, transposeWeight, executor);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);
    w = (*weight)[FIRST_TENSOR_INDEX];
    wScale = (*weightScale)[FIRST_TENSOR_INDEX];
    checkRet = CheckMxQuantShape(x, groupList, xScaleOptional, w, wScale, transposeWeight, y, yScale);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const aclTensor *x, const aclTensor *groupList, const aclTensorList *&weight,
                        const aclTensorList *&weightScale, const aclTensorList *bias,
                        const aclTensor *xScaleOptional, const char *activationType,
                        bool &transposeWeight, int64_t groupListType, const char *quantMode, int64_t yDtype,
                        const char *roundMode, int64_t scaleAlg, float dstTypeMax, const aclTensor *y,
                        const aclTensor *yScale, aclOpExecutor *executor, const char *&effectiveQuantMode,
                        const char *&effectiveRoundMode)
{
    GmmaqPrint("In op [%s], CheckParams start. activationType=%s, quantMode=%s, roundMode=%s, "
               "transposeWeight=%d, groupListType=%ld, yDtype=%ld, scaleAlg=%ld, dstTypeMax=%f.",
               OP_NAME, activationType == nullptr ? "nullptr" : activationType,
               quantMode == nullptr ? "nullptr" : quantMode, roundMode == nullptr ? "nullptr" : roundMode,
               static_cast<int>(transposeWeight), groupListType, yDtype, scaleAlg, dstTypeMax);
    aclnnStatus checkRet = CheckRequiredParams(x, groupList, weight, weightScale, activationType, y, yScale);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);
    checkRet = CheckActivationType(activationType);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);

    checkRet = ResolveAndCheckQuantMode(x, xScaleOptional, quantMode, effectiveQuantMode);
    CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);
    if (std::strcmp(effectiveQuantMode, QUANT_MODE_MX) == 0) {
        checkRet = CheckMxQuantParams(x, groupList, weight, weightScale, bias, xScaleOptional, transposeWeight,
                                      groupListType, yDtype, roundMode, effectiveRoundMode, scaleAlg, dstTypeMax,
                                      y, yScale, executor);
        CHECK_RET(checkRet == ACLNN_SUCCESS, checkRet);
    } else {
        CHECK_COND(false, ACLNN_ERR_PARAM_INVALID,
                   "In op [%s], quant_mode 参数当前仅支持为 MX", OP_NAME);
    }
    GmmaqPrint("In op [%s], CheckParams success.", OP_NAME);
    return ACLNN_SUCCESS;
}

aclnnStatus PrepareTensorListContiguous(const aclTensorList *&tensorList, aclOpExecutor *executor)
{
    GmmaqPrint("In op [%s], PrepareTensorListContiguous start. tensorList size=%zu.", OP_NAME, tensorList->Size());
    std::vector<const aclTensor *> tensorVec;
    tensorVec.reserve(tensorList->Size());
    for (size_t i = 0; i < tensorList->Size(); ++i) {
        GmmaqPrint("In op [%s], make tensorList[%zu] contiguous.", OP_NAME, i);
        const aclTensor *contiguousTensor = l0op::Contiguous((*tensorList)[i], executor);
        CHECK_RET(contiguousTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        tensorVec.push_back(contiguousTensor);
    }
    tensorList = executor->AllocTensorList(tensorVec.data(), tensorVec.size());
    CHECK_RET(tensorList != nullptr, ACLNN_ERR_INNER_NULLPTR);
    GmmaqPrint("In op [%s], PrepareTensorListContiguous success.", OP_NAME);
    return ACLNN_SUCCESS;
}
} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize(const aclTensor *x,
    const aclTensor *groupList, const aclTensorList *weight, const aclTensorList *weightScale,
    const aclTensorList *bias, const aclTensor *xScaleOptional, const char *activationType,
    int64_t groupListType, const aclIntArray *tuningConfig, const char *quantMode, int64_t yDtype,
    const char *roundMode, int64_t scaleAlg, float dstTypeMax, aclTensor *y, aclTensor *yScale,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnGroupedMatmulActivationQuantWeightNz,
                   DFX_IN(x, groupList, weight, weightScale, bias, xScaleOptional),
                   DFX_OUT(y, yScale));
    bool transposeWeight = false;
    GmmaqPrint("In op [%s], GetWorkspaceSize start. activationType=%s, quantMode=%s, roundMode=%s, "
               "groupListType=%ld, yDtype=%ld, scaleAlg=%ld, dstTypeMax=%f.",
               OP_NAME, activationType == nullptr ? "nullptr" : activationType,
               quantMode == nullptr ? "nullptr" : quantMode, roundMode == nullptr ? "nullptr" : roundMode,
               groupListType, yDtype, scaleAlg, dstTypeMax);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    const char *effectiveQuantMode = nullptr;
    const char *effectiveRoundMode = nullptr;
    CHECK_RET(CheckParams(x, groupList, weight, weightScale, bias, xScaleOptional, activationType,
                          transposeWeight, groupListType, quantMode, yDtype, roundMode, scaleAlg, dstTypeMax,
                          y, yScale, uniqueExecutor.get(), effectiveQuantMode, effectiveRoundMode) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    GmmaqPrint("In op [%s], parameter check passed. transposeWeight=%d.",
               OP_NAME, static_cast<int>(transposeWeight));

    if (x->IsEmpty() || y->IsEmpty() || yScale->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        GmmaqPrint("In op [%s], empty tensor path. xEmpty=%d, yEmpty=%d, yScaleEmpty=%d, workspaceSize=0.",
                   OP_NAME, static_cast<int>(x->IsEmpty()), static_cast<int>(y->IsEmpty()),
                   static_cast<int>(yScale->IsEmpty()));
        return ACLNN_SUCCESS;
    }

    GmmaqPrint("In op [%s], make x contiguous.", OP_NAME);
    x = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(x != nullptr, ACLNN_ERR_INNER_NULLPTR);
    GmmaqPrint("In op [%s], make xScaleOptional contiguous.", OP_NAME);
    xScaleOptional = l0op::Contiguous(xScaleOptional, uniqueExecutor.get());
    CHECK_RET(xScaleOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    GmmaqPrint("In op [%s], make groupList contiguous.", OP_NAME);
    groupList = l0op::Contiguous(groupList, uniqueExecutor.get());
    CHECK_RET(groupList != nullptr, ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(PrepareTensorListContiguous(weightScale, uniqueExecutor.get()) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    NormalizeEmptyTensorList(bias);
    CHECK_RET(CreateEmptyTensorList(aclDataType::ACL_FLOAT, bias, uniqueExecutor.get(), "bias") ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(WrapWeightNzTensorList(weight, uniqueExecutor.get()) == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    PrintAclnnInputDebugInfo("before l0op::GroupedMatmulActivationQuant", x, groupList, weight, weightScale, bias,
                             xScaleOptional, y, yScale);
    GmmaqPrint("In op [%s], call l0op::GroupedMatmulActivationQuant.", OP_NAME);
    auto result = l0op::GroupedMatmulActivationQuant(x, groupList, weight, weightScale, bias, xScaleOptional,
        activationType, transposeWeight, groupListType, tuningConfig, effectiveQuantMode, yDtype, effectiveRoundMode,
        scaleAlg, dstTypeMax, uniqueExecutor.get());
    CHECK_RET(result != std::tuple(nullptr, nullptr), ACLNN_ERR_INNER_NULLPTR);
    GmmaqPrint("In op [%s], l0op::GroupedMatmulActivationQuant success, do ViewCopy for y.", OP_NAME);
    auto yResult = l0op::ViewCopy(std::get<0>(result), y, uniqueExecutor.get());
    CHECK_RET(yResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    GmmaqPrint("In op [%s], ViewCopy for y success, do ViewCopy for yScale.", OP_NAME);
    auto yScaleResult = l0op::ViewCopy(std::get<1>(result), yScale, uniqueExecutor.get());
    CHECK_RET(yScaleResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    GmmaqPrint("In op [%s], GetWorkspaceSize success. workspaceSize=%lu.", OP_NAME, *workspaceSize);
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedMatmulActivationQuantWeightNz(void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGroupedMatmulActivationQuantWeightNz);
    GmmaqPrint("In op [%s], phase2 launch start. workspaceSize=%lu.", OP_NAME, workspaceSize);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in aclnnGroupedMatmulActivationQuantWeightNz launch aicore.");
    GmmaqPrint("In op [%s], phase2 launch success.", OP_NAME);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
