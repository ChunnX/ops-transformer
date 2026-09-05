/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "aclnn_cla_gate_mx_quant_with_dual_axis.h"
#include "cla_gate_mx_quant_with_dual_axis.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "op_api/aclnn_util.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr int64_t EXPECTED_S = 256;
constexpr int64_t EXPECTED_B = 4;
constexpr int64_t EXPECTED_N = 96;
constexpr int64_t EXPECTED_D = 256;
constexpr int64_t NUM_TWO = 2;
constexpr int64_t SPLIT_BLOCK_SIZE = 64;
constexpr int64_t SCALE_ALG_OCP = 0;
constexpr int64_t SCALE_ALG_CUBLAS = 1;

static bool CheckNotNull(const aclTensor* globalOut, const aclTensor* localOut,
                         const aclTensor* globalGateLogits, const aclTensor* localGateLogits,
                         const aclTensor* rowData, const aclTensor* rowScale,
                         const aclTensor* colData, const aclTensor* colScale)
{
    OP_CHECK_NULL(globalOut, return false);
    OP_CHECK_NULL(localOut, return false);
    OP_CHECK_NULL(globalGateLogits, return false);
    OP_CHECK_NULL(localGateLogits, return false);
    OP_CHECK_NULL(rowData, return false);
    OP_CHECK_NULL(rowScale, return false);
    OP_CHECK_NULL(colData, return false);
    OP_CHECK_NULL(colScale, return false);
    return true;
}

static bool CheckDtypeValid(const aclTensor* globalOut, const aclTensor* localOut,
                            const aclTensor* globalGateLogits, const aclTensor* localGateLogits,
                            int64_t dstType)
{
    const std::initializer_list<op::DataType> inDtypeList = {op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
    OP_CHECK_DTYPE_NOT_SUPPORT(globalOut, inDtypeList, return false);
    OP_CHECK_DTYPE_NOT_SAME(localOut, globalOut, return false);
    OP_CHECK_DTYPE_NOT_SAME(globalGateLogits, globalOut, return false);
    OP_CHECK_DTYPE_NOT_SAME(localGateLogits, globalOut, return false);
    if (dstType != static_cast<int64_t>(op::DataType::DT_FLOAT8_E4M3FN) &&
        dstType != static_cast<int64_t>(op::DataType::DT_FLOAT8_E5M2)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dst_type must be float8_e4m3fn or float8_e5m2.");
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* globalOut, const aclTensor* localOut,
                       const aclTensor* globalGateLogits, const aclTensor* localGateLogits)
{
    auto goShape = globalOut->GetViewShape();
    auto loShape = localOut->GetViewShape();
    auto ggShape = globalGateLogits->GetViewShape();
    auto lgShape = localGateLogits->GetViewShape();
    if (goShape.GetDimNum() != 4 || loShape.GetDimNum() != 4 ||
        goShape.GetDim(0) != EXPECTED_S || goShape.GetDim(1) != EXPECTED_B ||
        goShape.GetDim(2) != EXPECTED_N || goShape.GetDim(3) != EXPECTED_D) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "global_out must be [256,4,96,256].");
        return false;
    }
    if (loShape.GetDimNum() != goShape.GetDimNum()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "local_out rank must match global_out.");
        return false;
    }
    for (size_t i = 0; i < goShape.GetDimNum(); ++i) {
        if (loShape.GetDim(i) != goShape.GetDim(i)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "local_out shape must match global_out.");
            return false;
        }
    }
    if (ggShape.GetDimNum() != 3 && ggShape.GetDimNum() != 4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "global_gate_logits must be [256,4,96] or [256,4,96,1].");
        return false;
    }
    if (lgShape.GetDimNum() != ggShape.GetDimNum()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "local_gate_logits rank must match global_gate_logits.");
        return false;
    }
    for (size_t i = 0; i < ggShape.GetDimNum(); ++i) {
        if (lgShape.GetDim(i) != ggShape.GetDim(i)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "local_gate_logits shape must match global_gate_logits.");
            return false;
        }
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* globalOut, const aclTensor* localOut,
                               const aclTensor* globalGateLogits, const aclTensor* localGateLogits,
                               int64_t scaleAlg, int64_t dstType,
                               const aclTensor* rowData, const aclTensor* rowScale,
                               const aclTensor* colData, const aclTensor* colScale)
{
    CHECK_RET(CheckNotNull(globalOut, localOut, globalGateLogits, localGateLogits,
                           rowData, rowScale, colData, colScale),
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(globalOut, localOut, globalGateLogits, localGateLogits, dstType),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(globalOut, localOut, globalGateLogits, localGateLogits), ACLNN_ERR_PARAM_INVALID);
    if (scaleAlg != SCALE_ALG_OCP && scaleAlg != SCALE_ALG_CUBLAS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "scale_alg must be 0 or 1.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnClaGateMxQuantWithDualAxisGetWorkspaceSize(
    const aclTensor* globalOut, const aclTensor* localOut,
    const aclTensor* globalGateLogits, const aclTensor* localGateLogits,
    int64_t scaleAlg, int64_t dstType,
    aclTensor* rowData, aclTensor* rowScale,
    aclTensor* colData, aclTensor* colScale,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnClaGateMxQuantWithDualAxis,
                   DFX_IN(globalOut, localOut, globalGateLogits, localGateLogits, scaleAlg, dstType),
                   DFX_OUT(rowData, rowScale, colData, colScale));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto ret = CheckParams(globalOut, localOut, globalGateLogits, localGateLogits,
                           scaleAlg, dstType, rowData, rowScale, colData, colScale);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto globalOutContiguous = l0op::Contiguous(globalOut, uniqueExecutor.get());
    CHECK_RET(globalOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto localOutContiguous = l0op::Contiguous(localOut, uniqueExecutor.get());
    CHECK_RET(localOutContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto globalGateContiguous = l0op::Contiguous(globalGateLogits, uniqueExecutor.get());
    CHECK_RET(globalGateContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto localGateContiguous = l0op::Contiguous(localGateLogits, uniqueExecutor.get());
    CHECK_RET(localGateContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::ClaGateMxQuantWithDualAxis(
        globalOutContiguous, localOutContiguous, globalGateContiguous, localGateContiguous,
        scaleAlg, dstType, uniqueExecutor.get());
    const aclTensor* rowDataOut = std::get<0>(result);
    const aclTensor* rowScaleOut = std::get<1>(result);
    const aclTensor* colDataOut = std::get<2>(result);
    const aclTensor* colScaleOut = std::get<3>(result);
    if (rowDataOut == nullptr || rowScaleOut == nullptr || colDataOut == nullptr || colScaleOut == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto viewRowData = l0op::ViewCopy(rowDataOut, rowData, uniqueExecutor.get());
    CHECK_RET(viewRowData != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewRowScale = l0op::ViewCopy(rowScaleOut, rowScale, uniqueExecutor.get());
    CHECK_RET(viewRowScale != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewColData = l0op::ViewCopy(colDataOut, colData, uniqueExecutor.get());
    CHECK_RET(viewColData != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewColScale = l0op::ViewCopy(colScaleOut, colScale, uniqueExecutor.get());
    CHECK_RET(viewColScale != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnClaGateMxQuantWithDualAxis(void* workspace, uint64_t workspaceSize,
                                            aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnClaGateMxQuantWithDualAxis);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
