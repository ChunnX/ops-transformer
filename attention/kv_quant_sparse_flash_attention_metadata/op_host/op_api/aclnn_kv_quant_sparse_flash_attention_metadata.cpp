/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_kv_quant_sparse_flash_attention_metadata.cpp
 * \brief
 */

#include "aclnn_kv_quant_sparse_flash_attention_metadata.h"
#include "l0_kv_quant_sparse_flash_attention_metadata.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/make_op_executor.h"

#ifdef __cplusplus
extern "C" {
#endif

static aclnnStatus ParamsCheck(const aclTensor* actualSeqLengthsQueryOptional, 
                               const aclTensor* actualSeqLengthsKvOptional, 
                               int64_t numHeadsQ, int64_t numHeadsKv, int64_t batchSizeOptional, 
                               int64_t maxSeqlenQOptional, int64_t sparseModeOptional, int64_t preTokensOptional, 
                               int64_t  nextTokensOptional, char* layoutQueryOptional, char* layoutKvOptional) 
{
    if (CheckSingleParamQsfa(batchSizeOptional, maxSeqlenQOptional, numHeadsQ, numHeadsKv, sparseModeOptional, preTokensOptional, 
                             nextTokensOptional, layoutQueryOptional, 
                             layoutKvOptional) == ACLNN_SUCCESS && 
        CheckExistenceQsfa(actualSeqLengthsQueryOptional, actualSeqLengthsKvOptional, layoutQueryOptional, layoutKvOptional) == ACLNN_SUCCESS && 
        CheckConsistencyQsfa(actualSeqLengthsQueryOptional, actualSeqLengthsKvOptional, batchSizeOptional) == ACLNN_SUCCESS &&
        CheckFeatureQsfa() == ACLNN_SUCCESS) {
        return ACLNN_SUCCESS;
    } else {
        return ACLNN_ERR_PARAM_INVALID;
    }
}

int64_t GetQueryBatchSizeQsfa(const aclTensor* actualSeqLengthsQueryOptional, int64_t batchSizeOptional)
{
    // 如果 actualSeqLengthsQueryOptional 传了，使用sequsedQOptional获取BatchSize
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    auto ret = aclGetViewShape(actualSeqLengthsQueryOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        int64_t dimSize = viewDims[0];
        delete[] viewDims;
        return dimSize;
    }
    // 使用batchSizeOptional
    delete[] viewDims;
    return batchSizeOptional;
}

int64_t GetKvBatchSizeQsfa(const aclTensor* actualSeqLengthsKvOptional, int64_t batchSizeOptional)
{
    // 如果 sequsedKvOptional传了，使用sequsedKvOptional获取BatchSize
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    auto ret = aclGetViewShape(actualSeqLengthsKvOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        int64_t dimSize = viewDims[0];
        delete[] viewDims;
        return dimSize;
    }
    // 如果不是 TND，或者 cuSeqLensKvOptional 为空，使用 batchSizeOptional
    delete[] viewDims;
    return batchSizeOptional;
}

aclnnStatus CheckSingleParamQsfa(int64_t batchSizeOptional, int64_t maxSeqlenQOptional, int64_t numHeadsQ, 
                                 int64_t numHeadsKv, int64_t sparseModeOptional, int64_t preTokensOptional, 
                                 int64_t  nextTokensOptional, char* layoutQueryOptional, 
                                 char* layoutKvOptional) 
{
    // batch_size 非负校验
    if (batchSizeOptional < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "batch_size should not be negative, but got %d", batchSizeOptional);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // max_seqlen_q 非负校验
    if (maxSeqlenQOptional < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "max_seqlen_q should not be negative, but got %d", maxSeqlenQOptional);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // num_heads_kv 校验
    if (numHeadsKv != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "num_heads_kv should only be 1, but got %d", numHeadsKv);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // sparse_mode 校验
    if (sparseModeOptional != 0 && sparseModeOptional != 3 && sparseModeOptional != 4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sparse_mode should be 0, 3 or4, but got %d", sparseModeOptional);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // pre_tokens 校验
    if (preTokensOptional != INT64_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "pre_tokens should only be 2^63-1, but got %ld", preTokensOptional);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // next_tokens 校验
    if (nextTokensOptional != INT64_MAX) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "next_tokens should only be 2^63-1, but got %ld", nextTokensOptional);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_q 校验
    if ((strcmp(layoutQueryOptional, "TND") != 0) && (strcmp(layoutQueryOptional, "BSND") != 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layout_q must be TND or BSND, but got %s", layoutQueryOptional);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_kv 校验
    if ((strcmp(layoutKvOptional, "PA_BSND") != 0) && (strcmp(layoutKvOptional, "TND") != 0) && 
        (strcmp(layoutKvOptional, "BSND") != 0)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "layout_kv must be TND, BSND or PA_BSND, but got %s", layoutKvOptional);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckExistenceQsfa(const aclTensor* actualSeqLengthsQueryOptional, 
                               const aclTensor* actualSeqLengthsKvOptional, 
                               char* layoutQueryOptional, char* layoutKvOptional)
{
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    // Query 存在性逻辑
    if (strcmp(layoutQueryOptional, "TND") == 0) {
        auto ret = aclGetViewShape(actualSeqLengthsQueryOptional, &viewDims, &viewDimsNum);
        if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum == 0 || viewDims[0] == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_query TND, actual_seq_lengths_query must be provided!");
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    // KV 存在性逻辑
    if (strcmp(layoutKvOptional, "PA_BSND") == 0) {
        auto ret = aclGetViewShape(actualSeqLengthsKvOptional, &viewDims, &viewDimsNum);
        if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum == 0 || viewDims[0] == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_key PA_BSND, actual_seq_lengths_kv must be provided!");
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (strcmp(layoutKvOptional, "TND") == 0) {
        auto ret = aclGetViewShape(actualSeqLengthsKvOptional, &viewDims, &viewDimsNum);
        if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum == 0 || viewDims[0] == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_key TND, actual_seq_lengths_kv must be provided!");
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    delete[] viewDims;
    return ACLNN_SUCCESS;
}

aclnnStatus CheckConsistencyQsfa(const aclTensor* actualSeqLengthsQueryOptional, 
                                 char* actualSeqLengthsKvOptional, int64_t batchSizeOptional)
{
    int64_t queryBatchSize = GetQueryBatchSizeQsfa(actualSeqLengthsQueryOptional, 
                                                   batchSizeOptional);
    int64_t kvBatchSize = GetKvBatchSizeQsfa(actualSeqLengthsKvOptional, 
                                             batchSizeOptional);
    if (queryBatchSize != kvBatchSize) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The batch_size obtained from q Tensor should be the same as "
                "that obtained from kv tensor, but got %ld and %ld", queryBatchSize, kvBatchSize);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckFeatureQsfa()
{
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnKvQuantSparseFlashAttentionMetadataGetWorkspaceSize(
    const aclTensor* actualSeqLengthsQueryOptional, const aclTensor* actualSeqLengthsKvOptional, int64_t numHeadsQ, 
    int64_t numHeadsKv, int64_t headDim, int64_t keyQuantMode, int64_t valueQuantMode, int64_t batchSizeOptional, 
    int64_t maxSeqlenQOptional, int64_t maxSeqlenKvOptional, int64_t topKOptional, int64_t sparseBlockSizeOptional, 
    char *layoutQueryOptional, char *layoutKvOptional, int64_t sparseModeOptional, int64_t preTokensOptional, 
    int64_t  nextTokensOptional, int64_t  attentionModeOptional, int64_t  quantScaleRepoModeOptional, 
    int64_t tileSizeOptional, int64_t ropeHeadDimOptional, const aclTensor* metaData, uint64_t* workspaceSize, 
    aclOpExecutor** executor) 
{
    L2_DFX_PHASE_1(aclnnKvQuantSparseFlashAttentionMetadata,
                   DFX_IN(actualSeqLengthsQueryOptional, actualSeqLengthsKvOptional, numHeadsQ, numHeadsKv, headDim, 
                          keyQuantMode, valueQuantMode, batchSizeOptional, maxSeqlenQOptional, maxSeqlenKvOptional, 
                          topKOptional, sparseBlockSizeOptional, layoutQueryOptional, layoutKvOptional, 
                          sparseModeOptional, preTokensOptional, nextTokensOptional, attentionModeOptional, 
                          quantScaleRepoModeOptional, tileSizeOptional, ropeHeadDimOptional),
                   DFX_OUT(metaData));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = ParamsCheck(actualSeqLengthsQueryOptional, actualSeqLengthsKvOptional, numHeadsQ, 
                           numHeadsKv, batchSizeOptional, maxSeqlenQOptional, sparseModeOptional, 
                           preTokensOptional, nextTokensOptional, layoutQueryOptional, layoutKvOptional);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
    uint32_t aicCoreNum = npuInfo.GetCubeCoreNum();
    uint32_t aivCoreNum = npuInfo.GetVectorCoreNum();
    const char *socVersion = npuInfo.GetSocLongVersion().c_str();
    // 核心数校验
    if (aicCoreNum == 0 || aivCoreNum == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AIC num or AIV num should not be 0, but got %u and %u", aicCoreNum, aivCoreNum);
        return ACLNN_ERR_PARAM_INVALID;
    }
    auto actualSeqLengthsQueryOptionalContiguous = l0op::Contiguous(actualSeqLengthsQueryOptional, uniqueExecutor.get());
    CHECK_RET(actualSeqLengthsQueryOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto actualSeqLengthsKvOptionalContiguous = l0op::Contiguous(actualSeqLengthsKvOptional, uniqueExecutor.get());
    CHECK_RET(actualSeqLengthsKvOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto output = l0op::KvQuantSparseFlashAttentionMetadata(
        actualSeqLengthsQueryOptionalContiguous, actualSeqLengthsKvOptionalContiguous, numHeadsQ, numHeadsKv, headDim, 
        keyQuantMode, valueQuantMode, batchSizeOptional, maxSeqlenQOptional, maxSeqlenKvOptional, 
        topKOptional, sparseBlockSizeOptional, layoutQueryOptional, layoutKvOptional, 
        sparseModeOptional, preTokensOptional, nextTokensOptional, attentionModeOptional, 
        quantScaleRepoModeOptional, tileSizeOptional, ropeHeadDimOptional, socVersion, aicCoreNum, 
        aivCoreNum, metaData, uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus
aclnnKvQuantSparseFlashAttentionMetadata(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, 
                                         aclrtStream stream) 
{
    L2_DFX_PHASE_2(aclnnKvQuantSparseFlashAttentionMetadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
