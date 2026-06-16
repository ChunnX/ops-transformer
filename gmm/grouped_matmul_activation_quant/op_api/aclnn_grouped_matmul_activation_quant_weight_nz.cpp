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
constexpr size_t WEIGHT_NZ_DIM_NUM = 5UL;
constexpr int64_t DIM_0 = 0L;
constexpr int64_t DIM_1 = 1L;
constexpr int64_t DIM_2 = 2L;
constexpr int64_t MX_GROUP_SIZE = 64L;
constexpr int64_t MX_SCALE_PAIR = 2L;
constexpr int64_t NZ_C0_DIM = 16L;
constexpr int64_t NZ_LAST_DIM = 32L;
constexpr int64_t MAX_GROUP_NUM = 1024L;
constexpr int64_t DEFAULT_DEQUANT_DTYPE = static_cast<int64_t>(DataType::DT_FLOAT);
constexpr int64_t FLOAT8_E5M2_VALUE = static_cast<int64_t>(DataType::DT_FLOAT8_E5M2);
constexpr int64_t FLOAT8_E4M3FN_VALUE = static_cast<int64_t>(DataType::DT_FLOAT8_E4M3FN);
constexpr int64_t SCALE_ALG_OCP = 0L;
constexpr int64_t SCALE_ALG_CUBLAS = 1L;
constexpr float DEFAULT_DST_TYPE_MAX = 0.0F;
constexpr float FLOAT_EPS = 1e-6F;

bool IsFp8(DataType dtype)
{
    return dtype == DataType::DT_FLOAT8_E4M3FN || dtype == DataType::DT_FLOAT8_E5M2;
}

bool IsFp8DtypeValue(int64_t dtype)
{
    return dtype == FLOAT8_E4M3FN_VALUE || dtype == FLOAT8_E5M2_VALUE;
}

bool IsWeightDtype(DataType dtype)
{
    return dtype == DataType::DT_FLOAT8_E4M3FN;
}

bool IsNzFormat(const aclTensor *tensor)
{
    return tensor != nullptr &&
           ge::GetPrimaryFormat(tensor->GetStorageFormat()) == op::Format::FORMAT_FRACTAL_NZ;
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

int64_t CeilDivInt64(int64_t dividend, int64_t divisor)
{
    return static_cast<int64_t>(Ops::Base::CeilDiv(dividend, divisor));
}

aclnnStatus CheckShapeValue(const aclTensor *x, const aclTensor *groupList, const aclTensor *xScaleOptional,
                            const aclTensor *weight, const aclTensor *weightScale, bool transposeWeightOptional,
                            const aclTensor *y, const aclTensor *yScale)
{
    int64_t m = x->GetViewShape().GetDim(DIM_0);
    int64_t k = x->GetViewShape().GetDim(DIM_1);
    int64_t e = groupList->GetViewShape().GetDim(DIM_0);
    int64_t n = transposeWeightOptional ? weightScale->GetViewShape().GetDim(DIM_1) :
                                           weightScale->GetViewShape().GetDim(DIM_2);
    CHECK_COND(e >= 1 && e <= MAX_GROUP_NUM, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], the first dim of groupList must be in range [1, 1024], but got %ld.", OP_NAME, e);
    CHECK_COND(m == 0 || n == 0 || k > 0, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], K must be greater than 0 unless M or N is 0.", OP_NAME);
    CHECK_COND(n % MX_GROUP_SIZE == 0, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], N must be divisible by 64, but got %ld.", OP_NAME, n);

    const auto &weightShape = weight->GetStorageShape();
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
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(xScaleOptional, xScaleExpectShape, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(y, yExpectShape, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(yScale, yScaleExpectShape, return ACLNN_ERR_PARAM_INVALID);

    op::Shape weightExpectShape;
    op::Shape weightScaleExpectShape;
    if (transposeWeightOptional) {
        weightExpectShape = {e, CeilDivInt64(k, NZ_LAST_DIM), CeilDivInt64(n, NZ_C0_DIM), NZ_C0_DIM, NZ_LAST_DIM};
        weightScaleExpectShape = {e, n, CeilDivInt64(k, MX_GROUP_SIZE), MX_SCALE_PAIR};
    } else {
        weightExpectShape = {e, CeilDivInt64(n, NZ_LAST_DIM), CeilDivInt64(k, NZ_C0_DIM), NZ_C0_DIM, NZ_LAST_DIM};
        weightScaleExpectShape = {e, CeilDivInt64(k, MX_GROUP_SIZE), n, MX_SCALE_PAIR};
    }
    CHECK_COND(weightShape == weightExpectShape, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], expected weight storage shape to be %s, but got %s.", OP_NAME,
               op::ToString(weightExpectShape).GetString(), op::ToString(weightShape).GetString());
    OP_CHECK_SHAPE_NOT_EQUAL_WITH_EXPECTED_SIZE(weightScale, weightScaleExpectShape, return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus CheckParams(const aclTensor *x, const aclTensor *groupList, const aclTensorList *weight,
                        const aclTensorList *weightScale, const aclTensorList *biasOptional,
                        const aclTensor *xScaleOptional, const char *activationType,
                        int64_t dequantDtypeOptional, bool transposeWeightOptional,
                        int64_t groupListTypeOptional, const char *quantModeOptional, int64_t yDtypeOptional,
                        const char *roundModeOptional, int64_t scaleAlgOptional,
                        float dstTypeMaxOptional, const aclTensor *y, const aclTensor *yScale)
{
    CHECK_COND(x != nullptr, ACLNN_ERR_PARAM_NULLPTR, "In op [%s], [x] must not be nullptr.", OP_NAME);
    CHECK_COND(groupList != nullptr, ACLNN_ERR_PARAM_NULLPTR, "In op [%s], [groupList] must not be nullptr.",
               OP_NAME);
    CHECK_COND(xScaleOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "In op [%s], [xScaleOptional] must not be nullptr in gelu_tanh MX quant mode.", OP_NAME);
    aclnnStatus checkTensorListRet = CheckTensorList(weight, "weight");
    CHECK_RET(checkTensorListRet == ACLNN_SUCCESS, checkTensorListRet);
    checkTensorListRet = CheckTensorList(weightScale, "weightScale");
    CHECK_RET(checkTensorListRet == ACLNN_SUCCESS, checkTensorListRet);
    CHECK_COND(y != nullptr, ACLNN_ERR_PARAM_NULLPTR, "In op [%s], [y] must not be nullptr.", OP_NAME);
    CHECK_COND(yScale != nullptr, ACLNN_ERR_PARAM_NULLPTR, "In op [%s], [yScale] must not be nullptr.", OP_NAME);
    CHECK_COND(activationType != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "In op [%s], [activationType] must not be nullptr.", OP_NAME);
    CHECK_COND(std::strcmp(activationType, GELU_TANH) == 0, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], activationType only supports gelu_tanh, but got %s.", OP_NAME, activationType);
    CHECK_COND(quantModeOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "In op [%s], [quantModeOptional] must not be nullptr.", OP_NAME);
    CHECK_COND(std::strcmp(quantModeOptional, QUANT_MODE_MX) == 0, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], quantModeOptional only supports MX, but got %s.", OP_NAME, quantModeOptional);
    CHECK_COND(roundModeOptional != nullptr, ACLNN_ERR_PARAM_NULLPTR,
               "In op [%s], [roundModeOptional] must not be nullptr.",
               OP_NAME);
    CHECK_COND(std::strcmp(roundModeOptional, ROUND_MODE_RINT) == 0, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], MX quant only supports roundModeOptional rint, but got %s.",
               OP_NAME, roundModeOptional);
    CHECK_COND(dequantDtypeOptional == DEFAULT_DEQUANT_DTYPE, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], dequantDtypeOptional must be FLOAT in MX quant mode, but got %ld.",
               OP_NAME, dequantDtypeOptional);
    CHECK_COND(IsFp8DtypeValue(yDtypeOptional), ACLNN_ERR_PARAM_INVALID,
               "In op [%s], yDtypeOptional only supports FLOAT8_E4M3FN or FLOAT8_E5M2, but got %ld.",
               OP_NAME, yDtypeOptional);
    CHECK_COND(groupListTypeOptional == 0 || groupListTypeOptional == 1, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], groupListTypeOptional only supports 0 or 1, but got %ld.",
               OP_NAME, groupListTypeOptional);
    CHECK_COND(scaleAlgOptional == SCALE_ALG_OCP || scaleAlgOptional == SCALE_ALG_CUBLAS, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], scaleAlgOptional only supports 0 or 1 in MX quant mode, but got %ld.",
               OP_NAME, scaleAlgOptional);
    CHECK_COND(std::fabs(dstTypeMaxOptional - DEFAULT_DST_TYPE_MAX) <= FLOAT_EPS, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], dstTypeMaxOptional only supports 0.0 in current MX quant mode, but got %f.",
               OP_NAME, dstTypeMaxOptional);
    CHECK_COND(biasOptional == nullptr || biasOptional->Size() == 0, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], non-empty bias is not supported in current MX quant mode.", OP_NAME);

    const aclTensor *w = (*weight)[FIRST_TENSOR_INDEX];
    const aclTensor *wScale = (*weightScale)[FIRST_TENSOR_INDEX];
    OP_CHECK_WRONG_DIMENSION(x, X_DIM_NUM, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(xScaleOptional, SCALE_DIM_NUM, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(w, WEIGHT_NZ_DIM_NUM, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(wScale, WEIGHT_SCALE_DIM_NUM, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(groupList, GROUP_LIST_DIM_NUM, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(y, OUT_DIM_NUM, return ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_WRONG_DIMENSION(yScale, OUT_SCALE_DIM_NUM, return ACLNN_ERR_PARAM_INVALID);
    CHECK_COND(w->GetStorageShape().GetDimNum() == WEIGHT_NZ_DIM_NUM, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], weight storage shape dim num must be 5.", OP_NAME);

    CHECK_COND(IsFp8(x->GetDataType()), ACLNN_ERR_PARAM_INVALID,
               "In op [%s], x only supports FLOAT8_E4M3FN or FLOAT8_E5M2.", OP_NAME);
    CHECK_COND(IsWeightDtype(w->GetDataType()), ACLNN_ERR_PARAM_INVALID,
               "In op [%s], weight only supports FLOAT8_E4M3FN.", OP_NAME);
    CHECK_COND(xScaleOptional->GetDataType() == DataType::DT_FLOAT8_E8M0, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], xScaleOptional only supports FLOAT8_E8M0.", OP_NAME);
    CHECK_COND(wScale->GetDataType() == DataType::DT_FLOAT8_E8M0, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], weightScale only supports FLOAT8_E8M0.", OP_NAME);
    CHECK_COND(groupList->GetDataType() == DataType::DT_INT64, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], groupList only supports INT64.", OP_NAME);
    CHECK_COND(y->GetDataType() == static_cast<DataType>(yDtypeOptional), ACLNN_ERR_PARAM_INVALID,
               "In op [%s], y dtype must be the same as yDtypeOptional.", OP_NAME);
    CHECK_COND(yScale->GetDataType() == DataType::DT_FLOAT8_E8M0, ACLNN_ERR_PARAM_INVALID,
               "In op [%s], yScale only supports FLOAT8_E8M0.", OP_NAME);
    CHECK_COND(x->GetStorageFormat() == op::Format::FORMAT_ND &&
                   xScaleOptional->GetStorageFormat() == op::Format::FORMAT_ND &&
                   groupList->GetStorageFormat() == op::Format::FORMAT_ND &&
                   wScale->GetStorageFormat() == op::Format::FORMAT_ND &&
                   y->GetStorageFormat() == op::Format::FORMAT_ND &&
                   yScale->GetStorageFormat() == op::Format::FORMAT_ND,
               ACLNN_ERR_PARAM_INVALID,
               "In op [%s], x, xScaleOptional, groupList, weightScale, y and yScale only support ND.", OP_NAME);
    CHECK_COND(IsNzFormat(w), ACLNN_ERR_PARAM_INVALID,
               "In op [%s], weight only supports FRACTAL_NZ format.", OP_NAME);
    CHECK_RET(CheckShapeValue(x, groupList, xScaleOptional, w, wScale, transposeWeightOptional, y, yScale) ==
                  ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus PrepareTensorListContiguous(const aclTensorList *&tensorList, aclOpExecutor *executor)
{
    std::vector<const aclTensor *> tensorVec;
    tensorVec.reserve(tensorList->Size());
    for (size_t i = 0; i < tensorList->Size(); ++i) {
        const aclTensor *contiguousTensor = l0op::Contiguous((*tensorList)[i], executor);
        CHECK_RET(contiguousTensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
        tensorVec.push_back(contiguousTensor);
    }
    tensorList = executor->AllocTensorList(tensorVec.data(), tensorVec.size());
    CHECK_RET(tensorList != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}
} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnGroupedMatmulActivationQuantWeightNzGetWorkspaceSize(const aclTensor *x,
    const aclTensor *groupList, const aclTensorList *weight, const aclTensorList *weightScale,
    const aclTensorList *biasOptional, const aclTensor *xScaleOptional, const char *activationType,
    int64_t dequantDtypeOptional, bool transposeWeightOptional, int64_t groupListTypeOptional,
    const aclIntArray *tuningConfigOptional, const char *quantModeOptional, int64_t yDtypeOptional,
    const char *roundModeOptional, int64_t scaleAlgOptional, float dstTypeMaxOptional, aclTensor *y, aclTensor *yScale,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnGroupedMatmulActivationQuantWeightNz,
                   DFX_IN(x, groupList, weight, weightScale, biasOptional, xScaleOptional),
                   DFX_OUT(y, yScale));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    CHECK_RET(CheckParams(x, groupList, weight, weightScale, biasOptional, xScaleOptional, activationType,
                          dequantDtypeOptional, transposeWeightOptional, groupListTypeOptional, quantModeOptional,
                          yDtypeOptional,
                          roundModeOptional, scaleAlgOptional, dstTypeMaxOptional, y, yScale) == ACLNN_SUCCESS,
              ACLNN_ERR_PARAM_INVALID);

    if (x->IsEmpty() || y->IsEmpty() || yScale->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    x = l0op::Contiguous(x, uniqueExecutor.get());
    CHECK_RET(x != nullptr, ACLNN_ERR_INNER_NULLPTR);
    xScaleOptional = l0op::Contiguous(xScaleOptional, uniqueExecutor.get());
    CHECK_RET(xScaleOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    groupList = l0op::Contiguous(groupList, uniqueExecutor.get());
    CHECK_RET(groupList != nullptr, ACLNN_ERR_INNER_NULLPTR);

    CHECK_RET(PrepareTensorListContiguous(weightScale, uniqueExecutor.get()) == ACLNN_SUCCESS,
              ACLNN_ERR_INNER_NULLPTR);
    if (biasOptional == nullptr) {
        biasOptional = uniqueExecutor->AllocTensorList(nullptr, 0);
        CHECK_RET(biasOptional != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    auto result = l0op::GroupedMatmulActivationQuant(x, groupList, weight, weightScale, biasOptional, xScaleOptional,
        activationType, dequantDtypeOptional, transposeWeightOptional, groupListTypeOptional, tuningConfigOptional,
        quantModeOptional, yDtypeOptional, roundModeOptional, scaleAlgOptional, dstTypeMaxOptional, uniqueExecutor.get());
    CHECK_RET(result != std::tuple(nullptr, nullptr), ACLNN_ERR_INNER_NULLPTR);
    auto yResult = l0op::ViewCopy(std::get<0>(result), y, uniqueExecutor.get());
    CHECK_RET(yResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto yScaleResult = l0op::ViewCopy(std::get<1>(result), yScale, uniqueExecutor.get());
    CHECK_RET(yScaleResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnGroupedMatmulActivationQuantWeightNz(void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnGroupedMatmulActivationQuantWeightNz);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "This is an error in aclnnGroupedMatmulActivationQuantWeightNz launch aicore.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
