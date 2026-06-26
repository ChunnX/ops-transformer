/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <initializer_list>
#include <tuple>

#include "aclnn_moe_token_permute_v2.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnnInner_moe_token_permute.h"  // 该文件为自动生成，在build/autogen/inner路径下
#include "external/aclnn_kernels/aclnn_platform.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_log.h"
#include "moe/moe_init_routing_v2/op_host/op_api/moe_init_routing_v2.h"
#include "moe/moe_init_routing_v3/op_host/op_api/moe_init_routing_v3.h"

using namespace op;

namespace MoeTokenPermuteV2Check {

static const std::initializer_list<op::DataType> MOE_DTYPE_SUPPORT_LIST_X = {DataType::DT_FLOAT16, DataType::DT_BF16,
                                                                             DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> MOE_DTYPE_SUPPORT_LIST_X_V3 = {
    DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_FLOAT, DataType::DT_INT8};
static const std::initializer_list<op::DataType> MOE_QUANT_DTYPE_SUPPORT_LIST_X = {DataType::DT_FLOAT16,
                                                                                   DataType::DT_BF16};
static const std::initializer_list<op::DataType> MOE_DTYPE_SUPPORT_LIST_EXPERT_IDX_REGBASE = {DataType::DT_INT32,
                                                                                              DataType::DT_INT64};
static const std::initializer_list<op::DataType> MOE_DTYPE_SUPPORT_LIST_ROW_IDX = {DataType::DT_INT32};
static constexpr int64_t QUANT_MODE_NONE = -1;
static constexpr int64_t QUANT_MODE_MXFP8_E5M2 = 2;
static constexpr int64_t QUANT_MODE_MXFP8_E4M3FN = 3;
static constexpr int64_t QUANT_MODE_MXFP4_E2M1 = 9;
static constexpr int64_t EXPERT_NUM = 10240;
static constexpr int64_t EXPERT_TOKENS_NUM_TYPE_COUNT = 1;
static constexpr int64_t ROW_IDX_TYPE_GATHER = 0;

static inline bool CheckNotNull(const aclTensor *tokens, const aclTensor *indices, const aclTensor *permuteTokensOut,
                                const aclTensor *sortedIndicesOut, const aclTensor *expandedScaleOut)
{
    OP_CHECK_NULL(tokens, return false);
    OP_CHECK_NULL(indices, return false);
    OP_CHECK_NULL(permuteTokensOut, return false);
    OP_CHECK_NULL(sortedIndicesOut, return false);
    OP_CHECK_NULL(expandedScaleOut, return false);
    return true;
}

static inline bool CheckDtypeValidRegbase(const aclTensor *tokens, const aclTensor *indices,
                                          const aclTensor *permuteTokensOut, const aclTensor *sortedIndicesOut)
{
    if (tokens != nullptr && tokens->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(tokens, MOE_DTYPE_SUPPORT_LIST_X, return false);
    }
    if (indices != nullptr && indices->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(indices, MOE_DTYPE_SUPPORT_LIST_EXPERT_IDX_REGBASE, return false);
    }
    if (permuteTokensOut != nullptr && permuteTokensOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(permuteTokensOut, MOE_DTYPE_SUPPORT_LIST_X, return false);
        OP_CHECK_DTYPE_NOT_SAME(tokens, permuteTokensOut, return false);
    }
    if (sortedIndicesOut != nullptr && sortedIndicesOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(sortedIndicesOut, MOE_DTYPE_SUPPORT_LIST_ROW_IDX, return false);
    }
    return true;
}

static inline bool CheckDtypeValidV2Regbase(const aclTensor *tokens, const aclTensor *indices,
                                            int64_t quantMode, const aclTensor *permuteTokensOut,
                                            const aclTensor *sortedIndicesOut, const aclTensor *expandedScaleOut)
{
    if (indices != nullptr && indices->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(indices, MOE_DTYPE_SUPPORT_LIST_EXPERT_IDX_REGBASE, return false);
    }
    if (sortedIndicesOut != nullptr && sortedIndicesOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(sortedIndicesOut, MOE_DTYPE_SUPPORT_LIST_ROW_IDX, return false);
    }

    if (quantMode == QUANT_MODE_NONE) {
        OP_CHECK_DTYPE_NOT_SUPPORT(tokens, MOE_DTYPE_SUPPORT_LIST_X_V3, return false);
        OP_CHECK_DTYPE_NOT_SUPPORT(permuteTokensOut, MOE_DTYPE_SUPPORT_LIST_X_V3, return false);
        OP_CHECK_DTYPE_NOT_SAME(tokens, permuteTokensOut, return false);
        OP_CHECK_DTYPE_NOT_MATCH(expandedScaleOut, DataType::DT_FLOAT, return false);
        return true;
    }

    OP_CHECK_DTYPE_NOT_SUPPORT(tokens, MOE_QUANT_DTYPE_SUPPORT_LIST_X, return false);
    if (quantMode == QUANT_MODE_MXFP8_E5M2) {
        OP_CHECK_DTYPE_NOT_MATCH(permuteTokensOut, DataType::DT_FLOAT8_E5M2, return false);
    } else if (quantMode == QUANT_MODE_MXFP8_E4M3FN) {
        OP_CHECK_DTYPE_NOT_MATCH(permuteTokensOut, DataType::DT_FLOAT8_E4M3FN, return false);
    } else if (quantMode == QUANT_MODE_MXFP4_E2M1) {
        OP_CHECK_DTYPE_NOT_MATCH(permuteTokensOut, DataType::DT_FLOAT4_E2M1, return false);
    } else {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "quantMode only supports -1, 2, 3 or 9 on Ascend 950.");
        return false;
    }
    OP_CHECK_DTYPE_NOT_MATCH(expandedScaleOut, DataType::DT_FLOAT8_E8M0, return false);
    return true;
}

static aclnnStatus CheckParamsV2Regbase(const aclTensor *tokens, const aclTensor *indices, int64_t quantMode,
                                        const aclTensor *permuteTokensOut, const aclTensor *sortedIndicesOut,
                                        const aclTensor *expandedScaleOut)
{
    CHECK_RET(CheckNotNull(tokens, indices, permuteTokensOut, sortedIndicesOut, expandedScaleOut),
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValidV2Regbase(tokens, indices, quantMode, permuteTokensOut, sortedIndicesOut,
                                       expandedScaleOut),
              ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParamsV2Fallback(const aclTensor *tokens, const aclTensor *indices,
                                         const aclTensor *permuteTokensOut, const aclTensor *sortedIndicesOut,
                                         const aclTensor *expandedScaleOut)
{
    CHECK_RET(CheckNotNull(tokens, indices, permuteTokensOut, sortedIndicesOut, expandedScaleOut),
              ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtypeValidRegbase(tokens, indices, permuteTokensOut, sortedIndicesOut), ACLNN_ERR_PARAM_INVALID);
    OP_CHECK_DTYPE_NOT_MATCH(expandedScaleOut, DataType::DT_FLOAT, return ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

} // namespace MoeTokenPermuteV2Check

static aclnnStatus BuildMoeInitRoutingV2Executor(
    const aclTensor* tokens, const aclTensor* indices, int64_t numOutTokens,
    const aclTensor* permuteTokensOut, const aclTensor* sortedIndicesOut, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto tokensContiguous = l0op::Contiguous(tokens, uniqueExecutor.get());
    CHECK_RET(tokensContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
    CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::MoeInitRoutingV2(tokensContiguous, indicesContiguous, numOutTokens, 0, 0, 0, 0, false,
                                         permuteTokensOut, sortedIndicesOut, nullptr, nullptr,
                                         uniqueExecutor.get());
    auto expandedXOut = std::get<0>(result);
    auto expandedRowIdxOut = std::get<1>(result);
    CHECK_RET(expandedXOut != nullptr && expandedRowIdxOut != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(expandedXOut, permuteTokensOut, uniqueExecutor.get()) != nullptr,
              ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(expandedRowIdxOut, sortedIndicesOut, uniqueExecutor.get()) != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMoeTokenPermuteV2GetWorkspaceSize(
    const aclTensor* tokens, const aclTensor* indices, int64_t numOutTokens, bool paddedMode, int64_t quantMode,
    const aclTensor* permuteTokensOut, const aclTensor* sortedIndicesOut, const aclTensor* expandedScaleOut,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnMoeTokenPermuteV2,
        DFX_IN(tokens, indices, numOutTokens, paddedMode, quantMode),
                   DFX_OUT(permuteTokensOut, sortedIndicesOut, expandedScaleOut));
    CHECK_RET(MoeTokenPermuteV2Check::CheckNotNull(tokens, indices, permuteTokensOut, sortedIndicesOut,
                                                   expandedScaleOut),
              ACLNN_ERR_PARAM_NULLPTR);

    bool isRegbase = Ops::Transformer::AclnnUtil::IsRegbase();
    if (!isRegbase) {
        return aclnnInnerMoeTokenPermuteGetWorkspaceSize(
            tokens, indices, numOutTokens, paddedMode, permuteTokensOut, sortedIndicesOut, workspaceSize, executor);
    }
    CHECK_RET(paddedMode == false, ACLNN_ERR_PARAM_INVALID);

    bool useMoeInitRoutingV3 = GetCurrentPlatformInfo().GetSocVersion() == SocVersion::ASCEND950;
    if (!useMoeInitRoutingV3) {
        aclnnStatus ret = MoeTokenPermuteV2Check::CheckParamsV2Fallback(
            tokens, indices, permuteTokensOut, sortedIndicesOut, expandedScaleOut);
        CHECK_RET(ret == ACLNN_SUCCESS, ret);
        return BuildMoeInitRoutingV2Executor(tokens, indices, numOutTokens, permuteTokensOut, sortedIndicesOut,
                                             workspaceSize, executor);
    }

    aclnnStatus ret = MoeTokenPermuteV2Check::CheckParamsV2Regbase(
        tokens, indices, quantMode, permuteTokensOut, sortedIndicesOut, expandedScaleOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto tokensContiguous = l0op::Contiguous(tokens, uniqueExecutor.get());
    CHECK_RET(tokensContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto indicesContiguous = l0op::Contiguous(indices, uniqueExecutor.get());
    CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    if (indicesContiguous->GetDataType() == DataType::DT_INT64) {
        indicesContiguous = l0op::Cast(indicesContiguous, DataType::DT_INT32, uniqueExecutor.get());
        CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }
    if (indicesContiguous->GetViewShape().GetDimNum() == 1) {
        op::Shape indicesShape;
        indicesShape.AppendDim(indicesContiguous->GetViewShape().GetDim(0));
        indicesShape.AppendDim(1);
        indicesContiguous = uniqueExecutor->CreateView(indicesContiguous, indicesShape, 0);
        CHECK_RET(indicesContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    }

    int64_t fullTokenNum = indicesContiguous->GetViewShape().GetShapeSize();
    op::Shape fullExpandedXShape = permuteTokensOut->GetViewShape();
    fullExpandedXShape.SetDim(0, fullTokenNum);
    auto fullExpandedX = uniqueExecutor->AllocTensor(
        fullExpandedXShape, permuteTokensOut->GetDataType(), Format::FORMAT_ND);
    op::Shape fullRowIdxShape;
    fullRowIdxShape.AppendDim(fullTokenNum);
    auto fullExpandedRowIdx = uniqueExecutor->AllocTensor(
        fullRowIdxShape, DataType::DT_INT32, Format::FORMAT_ND);
    op::Shape expertTokensShape;
    expertTokensShape.AppendDim(MoeTokenPermuteV2Check::EXPERT_NUM);
    auto expertTokensCount = uniqueExecutor->AllocTensor(
        expertTokensShape, DataType::DT_INT64, Format::FORMAT_ND);
    op::Shape fullExpandedScaleShape;
    if (quantMode == MoeTokenPermuteV2Check::QUANT_MODE_MXFP8_E5M2 ||
        quantMode == MoeTokenPermuteV2Check::QUANT_MODE_MXFP8_E4M3FN) {
        fullExpandedScaleShape = expandedScaleOut->GetViewShape();
        fullExpandedScaleShape.SetDim(0, fullTokenNum);
    } else if (quantMode == MoeTokenPermuteV2Check::QUANT_MODE_MXFP4_E2M1) {
        fullExpandedScaleShape = expandedScaleOut->GetViewShape();
        fullExpandedScaleShape.SetDim(0, fullTokenNum);
    } else {
        fullExpandedScaleShape.AppendDim(fullTokenNum);
    }
    auto fullExpandedScale = uniqueExecutor->AllocTensor(
        fullExpandedScaleShape,
        quantMode == MoeTokenPermuteV2Check::QUANT_MODE_NONE ? DataType::DT_FLOAT : DataType::DT_FLOAT8_E8M0,
        Format::FORMAT_ND);
    CHECK_RET(fullExpandedX != nullptr && fullExpandedRowIdx != nullptr && expertTokensCount != nullptr &&
                  fullExpandedScale != nullptr,
              ACLNN_ERR_INNER_NULLPTR);

    const int64_t activeExpertRangeData[] = {0, MoeTokenPermuteV2Check::EXPERT_NUM};
    auto activeExpertRange = uniqueExecutor->AllocIntArray(activeExpertRangeData, 2);
    CHECK_RET(activeExpertRange != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::MoeInitRoutingV3(
        tokensContiguous, indicesContiguous, nullptr, nullptr, fullTokenNum, 0, MoeTokenPermuteV2Check::EXPERT_NUM, 0,
        MoeTokenPermuteV2Check::EXPERT_TOKENS_NUM_TYPE_COUNT, true, quantMode, activeExpertRange,
        MoeTokenPermuteV2Check::ROW_IDX_TYPE_GATHER, fullExpandedX, fullExpandedRowIdx, expertTokensCount,
        fullExpandedScale, uniqueExecutor.get());
    auto expandedXOut = std::get<0>(result);
    auto expandedRowIdxOut = std::get<1>(result);
    auto expandedScale = std::get<3>(result);
    bool hasNullptr = (expandedXOut == nullptr) || (expandedRowIdxOut == nullptr) || (expandedScale == nullptr);
    CHECK_RET(hasNullptr != true, ACLNN_ERR_INNER_NULLPTR);

    auto expandedXView = uniqueExecutor->CreateView(expandedXOut, permuteTokensOut->GetViewShape(), 0);
    CHECK_RET(expandedXView != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(expandedXView, permuteTokensOut, uniqueExecutor.get()) != nullptr,
              ACLNN_ERR_INNER_NULLPTR);
    auto expandedRowIdxView = uniqueExecutor->CreateView(expandedRowIdxOut, sortedIndicesOut->GetViewShape(), 0);
    CHECK_RET(expandedRowIdxView != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(l0op::ViewCopy(expandedRowIdxView, sortedIndicesOut, uniqueExecutor.get()) != nullptr,
              ACLNN_ERR_INNER_NULLPTR);
    if (quantMode != MoeTokenPermuteV2Check::QUANT_MODE_NONE) {
        auto expandedScaleView = uniqueExecutor->CreateView(expandedScale, expandedScaleOut->GetViewShape(), 0);
        CHECK_RET(expandedScaleView != nullptr, ACLNN_ERR_INNER_NULLPTR);
        CHECK_RET(l0op::ViewCopy(expandedScaleView, expandedScaleOut, uniqueExecutor.get()) != nullptr,
                  ACLNN_ERR_INNER_NULLPTR);
    }

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMoeTokenPermuteV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    static bool isRegbase = Ops::Transformer::AclnnUtil::IsRegbase();
    if (!isRegbase) {
        return aclnnInnerMoeTokenPermute(workspace, workspaceSize, executor, stream);
    }

    L2_DFX_PHASE_2(aclnnMoeTokenPermuteV2);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
