/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <tuple>
#include <cstddef>
#include "opdev/make_op_executor.h"
#include "aclnn_kernels/contiguous.h"
#include "opdev/tensor_view_utils.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/common_types.h"
#include "moe_re_routing.h"
#include "aclnn_moe_re_routing.h"
#include "external/aclnn_kernels/aclnn_platform.h"
#include "aclnn_util.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_TOKENS = {DataType::DT_FLOAT16, DataType::DT_BF16,
    DataType::DT_INT8};
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_TOKENS_REGBASE = {
    DataType::DT_FLOAT16,  DataType::DT_BF16,        DataType::DT_INT8,
    DataType::DT_HIFLOAT8, DataType::DT_FLOAT8_E5M2, DataType::DT_FLOAT8_E4M3FN,
    DataType::DT_FLOAT4_E2M1, DataType::DT_FLOAT4_E1M2};
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_EXPERT_TOKEN_NUM = {
    DataType::DT_INT32, DataType::DT_INT64};
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_SCALE = {DataType::DT_FLOAT};
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_SCALE_REGBASE = {
    DataType::DT_FLOAT, DataType::DT_FLOAT8_E8M0};
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_PERMUTE_TOKENS = {
    DataType::DT_FLOAT16, DataType::DT_BF16, DataType::DT_INT8};
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_PERMUTE_TOKENS_REGBASE = {
    DataType::DT_FLOAT16,       DataType::DT_BF16,        DataType::DT_INT8,
    DataType::DT_HIFLOAT8,      DataType::DT_FLOAT8_E5M2, DataType::DT_FLOAT8_E4M3FN,
    DataType::DT_FLOAT4_E2M1,   DataType::DT_FLOAT4_E1M2};
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_PERMUTE_SCALE = {DataType::DT_FLOAT};
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_PERMUTE_SCALE_REGBASE = {
    DataType::DT_FLOAT, DataType::DT_FLOAT8_E8M0};
static const std::initializer_list<DataType> DTYPE_SUPPORT_LIST_PERMUTE_IDX = {DataType::DT_INT32};

static inline bool CheckNotNull(const aclTensor *tokens, const aclTensor *expertTokenNumPerRank,
                                const aclTensor *permuteTokensOut, const aclTensor *permutePerTokenScalesOut,
                                const aclTensor *permuteTokenIdxOut, const aclTensor *expertTokenNumOut)
{
    OP_CHECK_NULL(tokens, return false);
    OP_CHECK_NULL(expertTokenNumPerRank, return false);
    OP_CHECK_NULL(permuteTokensOut, return false);
    OP_CHECK_NULL(permutePerTokenScalesOut, return false);
    OP_CHECK_NULL(permuteTokenIdxOut, return false);
    OP_CHECK_NULL(expertTokenNumOut, return false);
    return true;
}

static inline bool CheckDtypeValidRegbase(const aclTensor *tokens, const aclTensor *expertTokenNumPerRank,
                                          const aclTensor *perTokenScalesOptional,
                                          const aclTensor *permuteTokensOut, const aclTensor *permutePerTokenScalesOut,
                                          const aclTensor *permuteTokenIdxOut, const aclTensor *expertTokenNumOut)
{
    if (tokens != nullptr && tokens->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(tokens, DTYPE_SUPPORT_LIST_TOKENS_REGBASE, return false);
    }
    if (expertTokenNumPerRank != nullptr && expertTokenNumPerRank->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expertTokenNumPerRank, DTYPE_SUPPORT_LIST_EXPERT_TOKEN_NUM, return false);
    }
    if (perTokenScalesOptional != nullptr && perTokenScalesOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(perTokenScalesOptional, DTYPE_SUPPORT_LIST_SCALE_REGBASE, return false);
    }
    if (permuteTokensOut != nullptr && permuteTokensOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(permuteTokensOut, DTYPE_SUPPORT_LIST_PERMUTE_TOKENS_REGBASE, return false);
    }
    if (permutePerTokenScalesOut != nullptr && permutePerTokenScalesOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(permutePerTokenScalesOut, DTYPE_SUPPORT_LIST_PERMUTE_SCALE_REGBASE, return false);
    }
    if (permuteTokenIdxOut != nullptr && permuteTokenIdxOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(permuteTokenIdxOut, DTYPE_SUPPORT_LIST_PERMUTE_IDX, return false);
    }
    if (expertTokenNumOut != nullptr && expertTokenNumOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expertTokenNumOut, DTYPE_SUPPORT_LIST_EXPERT_TOKEN_NUM, return false);
    }
    return true;
}

static inline bool CheckDtypeValid(const aclTensor *tokens, const aclTensor *expertTokenNumPerRank,
                                   const aclTensor *perTokenScalesOptional,
                                   const aclTensor *permuteTokensOut, const aclTensor *permutePerTokenScalesOut,
                                   const aclTensor *permuteTokenIdxOut, const aclTensor *expertTokenNumOut)
{
    if (tokens != nullptr && tokens->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(tokens, DTYPE_SUPPORT_LIST_TOKENS, return false);
    }
    if (expertTokenNumPerRank != nullptr && expertTokenNumPerRank->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expertTokenNumPerRank, DTYPE_SUPPORT_LIST_EXPERT_TOKEN_NUM, return false);
    }
    if (perTokenScalesOptional != nullptr && perTokenScalesOptional->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(perTokenScalesOptional, DTYPE_SUPPORT_LIST_SCALE, return false);
    }
    if (permuteTokensOut != nullptr && permuteTokensOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(permuteTokensOut, DTYPE_SUPPORT_LIST_PERMUTE_TOKENS, return false);
    }
    if (permutePerTokenScalesOut != nullptr && permutePerTokenScalesOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(permutePerTokenScalesOut, DTYPE_SUPPORT_LIST_PERMUTE_SCALE, return false);
    }
    if (permuteTokenIdxOut != nullptr && permuteTokenIdxOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(permuteTokenIdxOut, DTYPE_SUPPORT_LIST_PERMUTE_IDX, return false);
    }
    if (expertTokenNumOut != nullptr && expertTokenNumOut->GetViewShape().GetShapeSize() != 0) {
        OP_CHECK_DTYPE_NOT_SUPPORT(expertTokenNumOut, DTYPE_SUPPORT_LIST_EXPERT_TOKEN_NUM, return false);
    }
    return true;
}

ACLNN_API aclnnStatus aclnnMoeReRoutingGetWorkspaceSize(
    const aclTensor *tokens, const aclTensor *expertTokenNumPerRank, const aclTensor *perTokenScalesOptional,
    int64_t expertTokenNumType, int64_t idxType,
    const aclTensor *permuteTokensOut, const aclTensor *permutePerTokenScalesOut,
    const aclTensor *permuteTokenIdxOut, const aclTensor *expertTokenNumOut,
    uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnMoeReRouting,
                   DFX_IN(tokens, expertTokenNumPerRank, perTokenScalesOptional, expertTokenNumType, idxType),
                   DFX_OUT(permuteTokensOut, permutePerTokenScalesOut, permuteTokenIdxOut, expertTokenNumOut));

    auto ret = CheckNotNull(tokens, expertTokenNumPerRank, permuteTokensOut, permutePerTokenScalesOut,
                            permuteTokenIdxOut, expertTokenNumOut);
    CHECK_RET(ret, ACLNN_ERR_PARAM_NULLPTR);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (Ops::Transformer::AclnnUtil::IsRegbase()) {
        CHECK_RET(CheckDtypeValidRegbase(tokens, expertTokenNumPerRank, perTokenScalesOptional,
                                         permuteTokensOut, permutePerTokenScalesOut,
                                         permuteTokenIdxOut, expertTokenNumOut),
                  ACLNN_ERR_PARAM_INVALID);
    } else {
        CHECK_RET(CheckDtypeValid(tokens, expertTokenNumPerRank, perTokenScalesOptional,
                                  permuteTokensOut, permutePerTokenScalesOut,
                                  permuteTokenIdxOut, expertTokenNumOut),
                  ACLNN_ERR_PARAM_INVALID);
    }

    auto tokensContiguous = l0op::Contiguous(tokens, uniqueExecutor.get());
    CHECK_RET(tokensContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto expertTokenNumPerRankContiguous = l0op::Contiguous(expertTokenNumPerRank, uniqueExecutor.get());
    CHECK_RET(expertTokenNumPerRankContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    const aclTensor *perTokenScalesContiguous = nullptr;
    if (perTokenScalesOptional != nullptr) {
        perTokenScalesContiguous = l0op::Contiguous(perTokenScalesOptional, uniqueExecutor.get());
        CHECK_RET(perTokenScalesContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    }

    auto routingResult = l0op::MoeReRouting(
        tokensContiguous, expertTokenNumPerRankContiguous, perTokenScalesContiguous, nullptr,
        expertTokenNumType, idxType,
        permuteTokensOut, permutePerTokenScalesOut, permuteTokenIdxOut, expertTokenNumOut, nullptr,
        uniqueExecutor.get());
    auto [permuteTokensOut_, permutePerTokenScalesOut_, permuteTokenIdxOut_,
          expertTokenNumOut_, permuteTopkWeightOut_] = routingResult;
    bool hasNullptr = (permuteTokensOut_ == nullptr) || (permutePerTokenScalesOut_ == nullptr) ||
                      (permuteTokenIdxOut_ == nullptr) || (expertTokenNumOut_ == nullptr);
    CHECK_RET(hasNullptr != true, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(permuteTopkWeightOut_ == nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyPermuteTokensResult = l0op::ViewCopy(permuteTokensOut_, permuteTokensOut, uniqueExecutor.get());
    CHECK_RET(viewCopyPermuteTokensResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyPermutePerTokenScalesResult =
        l0op::ViewCopy(permutePerTokenScalesOut_, permutePerTokenScalesOut, uniqueExecutor.get());
    CHECK_RET(viewCopyPermutePerTokenScalesResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyPermuteTokenIdxResult = l0op::ViewCopy(permuteTokenIdxOut_, permuteTokenIdxOut, uniqueExecutor.get());
    CHECK_RET(viewCopyPermuteTokenIdxResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyExpertTokenNumResult = l0op::ViewCopy(expertTokenNumOut_, expertTokenNumOut, uniqueExecutor.get());
    CHECK_RET(viewCopyExpertTokenNumResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

ACLNN_API aclnnStatus aclnnMoeReRouting(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMoeReRouting);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
