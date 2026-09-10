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
#include <cstring>
#include "aclnn_cla_gate_quant.h"
#include "cla_gate_quant.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_util.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
constexpr int64_t MIN_N = 1;
constexpr int64_t MAX_N = 128;
constexpr int64_t ROW_BLOCK = 64;
constexpr int64_t SCALE_LAST_DIM = 2;
constexpr int64_t SCALE_ALG_OCP = 0;
constexpr int64_t SCALE_ALG_CUBLAS = 1;
constexpr int64_t FP8_E4M3FN = 36;
constexpr int64_t FP8_E5M2 = 35;
constexpr int64_t FP4_E2M1 = 40;
constexpr int64_t FP4_E1M2 = 41;
constexpr const char *INPUT_ATTN_LAYOUT_TND = "TND";

static bool CheckNotNull(const aclTensor *globalAttn, const aclTensor *localAttn, const aclTensor *globalGateLogits,
                         const aclTensor *localGateLogits, const aclTensor *rowData, const aclTensor *rowScale,
                         const aclTensor *colData, const aclTensor *colScale, bool dualAxisFlag)
{
    OP_CHECK_NULL(globalAttn, return false);
    OP_CHECK_NULL(localAttn, return false);
    OP_CHECK_NULL(globalGateLogits, return false);
    OP_CHECK_NULL(localGateLogits, return false);
    OP_CHECK_NULL(rowData, return false);
    OP_CHECK_NULL(rowScale, return false);
    // Single-axis mode must pass null col outputs; dual-axis mode must provide them.
    if (dualAxisFlag) {
        OP_CHECK_NULL(colData, return false);
        OP_CHECK_NULL(colScale, return false);
    } else if (colData != nullptr || colScale != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "colData/colScale must be nullptr when dualAxisFlag is false.");
        return false;
    }
    return true;
}

static bool IsRoundModeValid(const char *mode)
{
    return mode == nullptr || std::strcmp(mode, "rint") == 0 || std::strcmp(mode, "round") == 0 ||
           std::strcmp(mode, "floor") == 0;
}

static bool IsDstTypeValid(int64_t dstType)
{
    return dstType == FP8_E5M2 || dstType == FP8_E4M3FN || dstType == FP4_E2M1 || dstType == FP4_E1M2;
}

static bool IsInputAttnLayoutValid(const char *inputAttnLayout)
{
    return inputAttnLayout == nullptr || std::strcmp(inputAttnLayout, INPUT_ATTN_LAYOUT_TND) == 0;
}

static bool CheckDtypeValid(const aclTensor *globalAttn, const aclTensor *localAttn,
                            const aclTensor *globalGateLogits, const aclTensor *localGateLogits, int64_t dstType)
{
    const std::initializer_list<op::DataType> inDtypeList = {op::DataType::DT_FLOAT16, op::DataType::DT_BF16};
    OP_CHECK_DTYPE_NOT_SUPPORT(globalAttn, inDtypeList, return false);
    OP_CHECK_DTYPE_NOT_SAME(localAttn, globalAttn, return false);
    OP_CHECK_DTYPE_NOT_SAME(globalGateLogits, globalAttn, return false);
    OP_CHECK_DTYPE_NOT_SAME(localGateLogits, globalAttn, return false);
    if (!IsDstTypeValid(dstType)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "dst_type must be 35(float8_e5m2), 36(float8_e4m3fn), 40(float4_e2m1) or 41(float4_e1m2).");
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor *globalAttn, const aclTensor *localAttn,
                       const aclTensor *globalGateLogits, const aclTensor *localGateLogits)
{
    auto globalAttnShape = globalAttn->GetViewShape();
    auto localAttnShape = localAttn->GetViewShape();
    auto globalGateShape = globalGateLogits->GetViewShape();
    auto localGateShape = localGateLogits->GetViewShape();
    if (globalAttnShape.GetDimNum() != 3 || localAttnShape.GetDimNum() != 3 || globalAttnShape.GetDim(1) < MIN_N ||
        globalAttnShape.GetDim(1) > MAX_N || (globalAttnShape.GetDim(2) != 128 && globalAttnShape.GetDim(2) != 256)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "global_attn/local_attn must be [T,N,D], N in [1,128], D in {128,256}.");
        return false;
    }
    if (localAttnShape.GetDimNum() != globalAttnShape.GetDimNum()) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "local_attn rank must match global_attn.");
        return false;
    }
    for (size_t i = 0; i < globalAttnShape.GetDimNum(); ++i) {
        if (localAttnShape.GetDim(i) != globalAttnShape.GetDim(i)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "local_attn shape must match global_attn.");
            return false;
        }
    }
    auto gateValid = [&](const op::Shape &gate) {
        return gate.GetDimNum() == 2 && gate.GetDim(0) == globalAttnShape.GetDim(0) &&
               gate.GetDim(1) == globalAttnShape.GetDim(1);
    };
    if (!gateValid(globalGateShape) || !gateValid(localGateShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "gate logits must be [T,N].");
        return false;
    }
    for (size_t i = 0; i < globalGateShape.GetDimNum(); ++i) {
        if (localGateShape.GetDim(i) != globalGateShape.GetDim(i)) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "local_gate_logits shape must match global_gate_logits.");
            return false;
        }
    }
    return true;
}

static bool CheckOutputShape(const aclTensor *rowData, const aclTensor *rowScale, const aclTensor *colData,
                             const aclTensor *colScale, int64_t rowCount, int64_t columnCount, bool dualAxisFlag)
{
    auto rowDataShape = rowData->GetViewShape();
    auto rowScaleShape = rowScale->GetViewShape();
    if (rowDataShape.GetDimNum() != 2 || rowDataShape.GetDim(0) != rowCount || rowDataShape.GetDim(1) != columnCount) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "row_data must be [T,N*D].");
        return false;
    }
    if (rowScaleShape.GetDimNum() != 3 || rowScaleShape.GetDim(0) != rowCount ||
        rowScaleShape.GetDim(1) != (columnCount + ROW_BLOCK - 1) / ROW_BLOCK ||
        rowScaleShape.GetDim(2) != SCALE_LAST_DIM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "row_scale physical shape mismatch.");
        return false;
    }
    if (!dualAxisFlag) {
        return true;
    }
    auto colDataShape = colData->GetViewShape();
    auto colScaleShape = colScale->GetViewShape();
    if (colDataShape.GetDimNum() != 2 || colDataShape.GetDim(0) != rowCount || colDataShape.GetDim(1) != columnCount) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "col_data must be [T,N*D].");
        return false;
    }
    if (colScaleShape.GetDimNum() != 3 || colScaleShape.GetDim(0) != (rowCount + ROW_BLOCK - 1) / ROW_BLOCK ||
        colScaleShape.GetDim(1) != columnCount || colScaleShape.GetDim(2) != SCALE_LAST_DIM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "col_scale physical shape mismatch.");
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor *globalAttn, const aclTensor *localAttn,
                               const aclTensor *globalGateLogits, const aclTensor *localGateLogits, const char *roundMode,
                               int64_t scaleAlg, int64_t dstType, const char *inputAttnLayout, bool dualAxisFlag,
                               const aclTensor *rowData, const aclTensor *rowScale, const aclTensor *colData,
                               const aclTensor *colScale)
{
    CHECK_RET(CheckNotNull(globalAttn, localAttn, globalGateLogits, localGateLogits, rowData, rowScale, colData, colScale,
                           dualAxisFlag),
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValid(globalAttn, localAttn, globalGateLogits, localGateLogits, dstType),
              ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShape(globalAttn, localAttn, globalGateLogits, localGateLogits), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(IsRoundModeValid(roundMode), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(IsInputAttnLayoutValid(inputAttnLayout), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(scaleAlg == SCALE_ALG_OCP || scaleAlg == SCALE_ALG_CUBLAS, ACLNN_ERR_PARAM_INVALID);
    if ((dstType == FP4_E2M1 || dstType == FP4_E1M2) && scaleAlg != SCALE_ALG_OCP) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FP4 only supports scale_alg=0.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((dstType == FP8_E4M3FN || dstType == FP8_E5M2) && roundMode != nullptr && std::strcmp(roundMode, "rint") != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FP8 only supports round_mode=rint.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto globalAttnShape = globalAttn->GetViewShape();
    int64_t rowCount = globalAttnShape.GetDim(0);
    int64_t columnCount = globalAttnShape.GetDim(1) * globalAttnShape.GetDim(2);
    if ((dstType == FP4_E2M1 || dstType == FP4_E1M2) && (columnCount % 4) != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When dst_type is FP4, K=N*D must be divisible by 4.");
        return ACLNN_ERR_PARAM_INVALID;
    }
    CHECK_RET(CheckOutputShape(rowData, rowScale, colData, colScale, rowCount, columnCount, dualAxisFlag),
              ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnClaGateQuantGetWorkspaceSize(
    const aclTensor *globalAttn, const aclTensor *localAttn, const aclTensor *globalGateLogits,
    const aclTensor *localGateLogits, const char *roundModeOptional, int64_t scaleAlg, int64_t dstType,
    const char *inputAttnLayout, bool dualAxisFlag, aclTensor *rowData, aclTensor *rowScale, aclTensor *colData,
    aclTensor *colScale, uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(
        aclnnClaGateQuant,
        DFX_IN(globalAttn, localAttn, globalGateLogits, localGateLogits, roundModeOptional, scaleAlg, dstType,
               inputAttnLayout, dualAxisFlag),
        DFX_OUT(rowData, rowScale, colData, colScale));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto ret = CheckParams(globalAttn, localAttn, globalGateLogits, localGateLogits, roundModeOptional, scaleAlg, dstType,
                           inputAttnLayout, dualAxisFlag, rowData, rowScale, colData, colScale);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto globalAttnContiguous = l0op::Contiguous(globalAttn, uniqueExecutor.get());
    CHECK_RET(globalAttnContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto localAttnContiguous = l0op::Contiguous(localAttn, uniqueExecutor.get());
    CHECK_RET(localAttnContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto globalGateContiguous = l0op::Contiguous(globalGateLogits, uniqueExecutor.get());
    CHECK_RET(globalGateContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto localGateContiguous = l0op::Contiguous(localGateLogits, uniqueExecutor.get());
    CHECK_RET(localGateContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::ClaGateQuant(globalAttnContiguous, localAttnContiguous, globalGateContiguous,
                                     localGateContiguous, roundModeOptional, scaleAlg, dstType, inputAttnLayout,
                                     dualAxisFlag, uniqueExecutor.get());
    const aclTensor *rowDataOut = std::get<0>(result);
    const aclTensor *rowScaleOut = std::get<1>(result);
    const aclTensor *colDataOut = std::get<2>(result);
    const aclTensor *colScaleOut = std::get<3>(result);
    if (rowDataOut == nullptr || rowScaleOut == nullptr || colDataOut == nullptr || colScaleOut == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto viewRowData = l0op::ViewCopy(rowDataOut, rowData, uniqueExecutor.get());
    CHECK_RET(viewRowData != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewRowScale = l0op::ViewCopy(rowScaleOut, rowScale, uniqueExecutor.get());
    CHECK_RET(viewRowScale != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (dualAxisFlag) {
        auto viewColData = l0op::ViewCopy(colDataOut, colData, uniqueExecutor.get());
        CHECK_RET(viewColData != nullptr, ACLNN_ERR_INNER_NULLPTR);
        auto viewColScale = l0op::ViewCopy(colScaleOut, colScale, uniqueExecutor.get());
        CHECK_RET(viewColScale != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnClaGateQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnClaGateQuant);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
