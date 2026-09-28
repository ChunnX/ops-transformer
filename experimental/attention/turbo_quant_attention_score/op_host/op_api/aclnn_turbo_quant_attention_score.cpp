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
 * \file aclnn_turbo_quant_attention_score.cpp
 * \brief ACLNN C-API (GetWorkspaceSize + Execute).
 */

#include <dlfcn.h>
#include "aclnn_turbo_quant_attention_score.h"
#include "turbo_quant_attention_score.h"

#include "securec.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"

#include "aclnn_kernels/contiguous.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {

struct TurboQuantAttentionScoreParams {
    const aclTensor *queryRotated{nullptr};
    const aclTensor *queryQjl{nullptr};
    const aclTensor *keyCacheIdx{nullptr};
    const aclTensor *keyCacheQjl{nullptr};
    const aclTensor *keyCacheNorm{nullptr};
    const aclTensor *keyCacheGamma{nullptr};
    const aclTensor *blockTable{nullptr};
    const aclTensor *seqLens{nullptr};
    int64_t mseBits{3};
    aclTensor *out{nullptr};
};

static const std::initializer_list<op::DataType> BF16_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16};
static const std::initializer_list<op::DataType> UINT8_TYPE_SUPPORT_LIST = {op::DataType::DT_UINT8};
static const std::initializer_list<op::DataType> NORM_TYPE_SUPPORT_LIST = {op::DataType::DT_BF16,
                                                                           op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> INT32_TYPE_SUPPORT_LIST = {op::DataType::DT_INT32};
static const std::initializer_list<op::DataType> FP32_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};

static inline bool CheckNotNull(const TurboQuantAttentionScoreParams &params)
{
    OP_CHECK_NULL(params.queryRotated, return false);
    OP_CHECK_NULL(params.queryQjl, return false);
    OP_CHECK_NULL(params.keyCacheIdx, return false);
    OP_CHECK_NULL(params.keyCacheQjl, return false);
    OP_CHECK_NULL(params.keyCacheNorm, return false);
    OP_CHECK_NULL(params.keyCacheGamma, return false);
    OP_CHECK_NULL(params.blockTable, return false);
    OP_CHECK_NULL(params.seqLens, return false);
    OP_CHECK_NULL(params.out, return false);
    return true;
}

static inline bool CheckDtype(const TurboQuantAttentionScoreParams &params)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(params.queryRotated, BF16_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.queryQjl, BF16_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.keyCacheIdx, UINT8_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.keyCacheQjl, UINT8_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.keyCacheNorm, NORM_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.keyCacheGamma, NORM_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.blockTable, INT32_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.seqLens, INT32_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.out, FP32_TYPE_SUPPORT_LIST, return false);
    OP_CHECK(params.keyCacheNorm->GetDataType() == params.keyCacheGamma->GetDataType(),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "keyCacheNorm and keyCacheGamma must have the same dtype."),
             return false);
    return true;
}

static inline bool CheckAttr(const TurboQuantAttentionScoreParams &params)
{
    OP_CHECK(params.mseBits == 2 || params.mseBits == 3 || params.mseBits == 4,
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mseBits must be in {2, 3, 4}, got %ld.", params.mseBits), return false);
    return true;
}

static aclnnStatus CheckParams(const TurboQuantAttentionScoreParams &params)
{
    CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(CheckDtype(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckAttr(params), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

} // namespace

aclnnStatus aclnnTurboQuantAttentionScoreGetWorkspaceSize(const aclTensor *queryRotated, const aclTensor *queryQjl,
                                                          const aclTensor *keyCacheIdx, const aclTensor *keyCacheQjl,
                                                          const aclTensor *keyCacheNorm, const aclTensor *keyCacheGamma,
                                                          const aclTensor *blockTable, const aclTensor *seqLens,
                                                          int64_t mseBits, aclTensor *out, uint64_t *workspaceSize,
                                                          aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnTurboQuantAttentionScore,
                   DFX_IN(queryRotated, queryQjl, keyCacheIdx, keyCacheQjl, keyCacheNorm, keyCacheGamma, blockTable,
                          seqLens, mseBits),
                   DFX_OUT(out));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    TurboQuantAttentionScoreParams params{queryRotated,  queryQjl,   keyCacheIdx, keyCacheQjl, keyCacheNorm,
                                          keyCacheGamma, blockTable, seqLens,     mseBits,     out};
    CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    auto queryRotatedContig = l0op::Contiguous(queryRotated, uniqueExecutor.get());
    auto queryQjlContig = l0op::Contiguous(queryQjl, uniqueExecutor.get());
    auto keyCacheIdxContig = l0op::Contiguous(keyCacheIdx, uniqueExecutor.get());
    auto keyCacheQjlContig = l0op::Contiguous(keyCacheQjl, uniqueExecutor.get());
    auto keyCacheNormContig = l0op::Contiguous(keyCacheNorm, uniqueExecutor.get());
    auto keyCacheGammaContig = l0op::Contiguous(keyCacheGamma, uniqueExecutor.get());
    auto blockTableContig = l0op::Contiguous(blockTable, uniqueExecutor.get());
    auto seqLensContig = l0op::Contiguous(seqLens, uniqueExecutor.get());
    CHECK_RET(queryRotatedContig != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(queryQjlContig != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(keyCacheIdxContig != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(keyCacheQjlContig != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(keyCacheNormContig != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(keyCacheGammaContig != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(blockTableContig != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(seqLensContig != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto result = l0op::TurboQuantAttentionScore(queryRotatedContig, queryQjlContig, keyCacheIdxContig,
                                                 keyCacheQjlContig, keyCacheNormContig, keyCacheGammaContig,
                                                 blockTableContig, seqLensContig, mseBits, uniqueExecutor.get());
    CHECK_RET(result != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto vcOut = l0op::ViewCopy(result, out, uniqueExecutor.get());
    CHECK_RET(vcOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnTurboQuantAttentionScore(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                          aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnTurboQuantAttentionScore);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
