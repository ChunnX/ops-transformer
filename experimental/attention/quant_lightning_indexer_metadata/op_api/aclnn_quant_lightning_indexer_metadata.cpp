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
 * \file aclnn_quant_lightning_indexer_metadata.cpp
 * \brief
 */

#include "aclnn_quant_lightning_indexer_metadata.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "l0_quant_lightning_indexer_metadata.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

#ifdef __cplusplus
extern "C" {
#endif

static aclnnStatus ParamsCheck(const aclTensor* actualSeqLengthsQueryOptional,
                               const aclTensor* actualSeqLengthsKeyOptional, int64_t numHeadsQ, int64_t numHeadsK,
                               int64_t batchSizeOptional, int64_t maxSeqlenQOptional, int64_t maxSeqlenKOptional, 
                               char* layoutQueryOptional, char* layoutKeyOptional, int64_t sparseModeOptional,
                               int64_t preTokensOptional, int64_t nextTokensOptional, int64_t cmpRatioOptional) 
{
    if (CheckSingleParamQli(batchSizeOptional, maxSeqlenQOptional, maxSeqlenKOptional, numHeadsQ, numHeadsK, 
                         layoutQueryOptional, layoutKeyOptional, sparseModeOptional, preTokensOptional, 
                         nextTokensOptional) == ACLNN_SUCCESS && 
        CheckExistenceQli(layoutQueryOptional, layoutKeyOptional, actualSeqLengthsQueryOptional, 
                       actualSeqLengthsKeyOptional) == ACLNN_SUCCESS && 
        CheckConsistencyQli(layoutQueryOptional, layoutKeyOptional, actualSeqLengthsQueryOptional, 
                         actualSeqLengthsKeyOptional, batchSizeOptional) == ACLNN_SUCCESS && 
        CheckFeatureQli(cmpRatioOptional) == ACLNN_SUCCESS) {
        return ACLNN_SUCCESS;
    } else {
        return ACLNN_ERR_PARAM_INVALID;
    }
}

aclnnStatus CheckSingleParamQli(int64_t batchSizeOptional, int64_t maxSeqlenQOptional, int64_t maxSeqlenKOptional, 
                             int64_t numHeadsQ, int64_t numHeadsK, char* layoutQueryOptional, char* layoutKeyOptional, 
                             int64_t sparseModeOptional, int64_t preTokensOptional, int64_t nextTokensOptional)
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
    // max_seqlen_k 非负校验
    if (maxSeqlenKOptional < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "max_seqlen_k should not be negative, but got %d", maxSeqlenKOptional);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // num_heads_q 校验
    if (numHeadsQ != 64) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "num_heads_q should only be 64, but got %d", numHeadsQ);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // num_heads_k 校验
    if (numHeadsK != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "num_heads_k should only be 1, but got %d", numHeadsK);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_query 校验
    if (strcmp(layoutQueryOptional, "TND") != 0 && strcmp(layoutQueryOptional, "BSND") != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_query, layout must be TND or BSND!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_key 校验
    if (strcmp(layoutKeyOptional, "PA_BSND") != 0 && strcmp(layoutKeyOptional, "TND") != 0 && 
        strcmp(layoutKeyOptional, "BSND") != 0) {
    	OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_key, layout must be PA_BSND/TND/BSND!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (strcmp(layoutQueryOptional, "TND") == 0 && strcmp(layoutKeyOptional, "BSND") == 0) {
    	OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_query TND, layout_key should be PA_BSND/TND!");
    	return ACLNN_ERR_PARAM_INVALID;
    }
    if (strcmp(layoutQueryOptional, "BSND") == 0 && strcmp(layoutKeyOptional, "TND") == 0) {
    	OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_query BSND, layout_key should be PA_BSND/BSND!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // sparse_mode 校验
    if (sparseModeOptional != 0 && sparseModeOptional != 3) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "sparse_mode should be 0/3, but got %d", sparseModeOptional);
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
    return ACLNN_SUCCESS;
}

aclnnStatus CheckExistenceQli(char* layoutQueryOptional, char* layoutKeyOptional, 
                           const aclTensor* actualSeqLengthsQueryOptional, const aclTensor* actualSeqLengthsKeyOptional)
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
    if (strcmp(layoutKeyOptional, "PA_BSND") == 0) {
        auto ret = aclGetViewShape(actualSeqLengthsKeyOptional, &viewDims, &viewDimsNum);
        if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum == 0 || viewDims[0] == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_key PA_BSND, actual_seq_lengths_key must be provided!");
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (strcmp(layoutKeyOptional, "TND") == 0) {
        auto ret = aclGetViewShape(actualSeqLengthsKeyOptional, &viewDims, &viewDimsNum);
        if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum == 0 || viewDims[0] == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_key TND, actual_seq_lengths_key must be provided!");
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    delete[] viewDims;
    return ACLNN_SUCCESS;
}

int64_t GetQueryBatchSizeQli(const aclTensor* actualSeqLengthsQueryOptional, int64_t batchSizeOptional)
{
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    auto ret = aclGetViewShape(actualSeqLengthsQueryOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        int64_t dimSize = viewDims[0];
        delete[] viewDims;
        return dimSize;
    }
    delete[] viewDims;
    return batchSizeOptional;
}

int64_t GetKvBatchSizeQli(const aclTensor* actualSeqLengthsKeyOptional, int64_t batchSizeOptional)
{
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    auto ret = aclGetViewShape(actualSeqLengthsKeyOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        int64_t dimSize = viewDims[0];
        delete[] viewDims;
        return dimSize;
    }
    delete[] viewDims;
    return batchSizeOptional;
}

aclnnStatus CheckConsistencyQli(char* layoutQueryOptional, char* layoutKeyOptional,
                             const aclTensor* actualSeqLengthsQueryOptional, 
                             const aclTensor* actualSeqLengthsKeyOptional, int64_t batchSizeOptional)
{
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    // 校验 cu_seqlens_q 维度
    auto ret = aclGetViewShape(actualSeqLengthsQueryOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of actual_seq_lengths_query must be 1, but got %ld", viewDimsNum);
        delete[] viewDims;
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 校验 cu_seqlens_ori_kv 维度
    ret = aclGetViewShape(actualSeqLengthsQueryOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of actual_seq_lengths_key must be 1, but got %ld", viewDimsNum);
        delete[] viewDims;
        return ACLNN_ERR_PARAM_INVALID;
    }
    int64_t queryBatchSize = GetQueryBatchSizeQli(actualSeqLengthsQueryOptional, batchSizeOptional);
    int64_t kvBatchSize = GetKvBatchSizeQli(actualSeqLengthsKeyOptional, batchSizeOptional);
    if ((strcmp(layoutQueryOptional, "BSND") == 0 || (strcmp(layoutQueryOptional, "TND") == 0 && 
        strcmp(layoutKeyOptional, "TND") == 0)) && queryBatchSize != kvBatchSize) {
    	OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For the layout_query is BSND or both layout_query and layout_key are TND, "
            "the batch_size obtained from q Tensor should be the same as that obtained from kv tensor, "
            "but got %ld and %ld", queryBatchSize, kvBatchSize);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if (std::abs(queryBatchSize - kvBatchSize) > 1) {
    	OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The difference between the dim of actual_seq_lengths_query and the dim of "
            "actual_seq_lengths_key should not be greater than 1, but got %ld and %ld", queryBatchSize, kvBatchSize);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckFeatureQli(int64_t cmpRatioOptional)
{
    // 压缩率校验
    if (cmpRatioOptional < 1 || cmpRatioOptional > 128) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cmp_ratio should be [1, 128], but got %d", cmpRatioOptional);
        return ACLNN_ERR_PARAM_INVALID;
    }
    const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
    const char* socVersion = npuInfo.GetSocLongVersion().c_str();
    if (strstr(socVersion, "Ascend950") == nullptr) {
        // 校验 2 的幂次方: 1, 2, 4, ..., 128
        if ((cmpRatioOptional & (cmpRatioOptional - 1)) != 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For Atlas A3, cmp_ratio should be 1/2/4/8/16/32/64/128, but got %d", 
                    cmpRatioOptional);
            return ACLNN_ERR_PARAM_INVALID;
        }
    } else {
        if (cmpRatioOptional != 1 && cmpRatioOptional != 4 && cmpRatioOptional != 128) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For Ascend950, cmp_ratio should be 1/4/128, but got %d", cmpRatioOptional);
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default")))
aclnnStatus aclnnQuantLightningIndexerMetadataGetWorkspaceSize(
    const aclTensor* actualSeqLengthsQueryOptional, const aclTensor* actualSeqLengthsKeyOptional, int64_t numHeadsQ,
    int64_t numHeadsK, int64_t headDim, int64_t queryQuantMode, int64_t keyQuantMode, int64_t batchSizeOptional,
    int64_t maxSeqlenQOptional, int64_t maxSeqlenKOptional, char* layoutQueryOptional, char* layoutKeyOptional,
    int64_t sparseCountOptional, int64_t sparseModeOptional, int64_t preTokensOptional, int64_t nextTokensOptional,
    int64_t cmpRatioOptional, const aclTensor* metaData, uint64_t* workspaceSize, aclOpExecutor** executor) 
{
    L2_DFX_PHASE_1(aclnnQuantLightningIndexerMetadata,
                   DFX_IN(actualSeqLengthsQueryOptional, actualSeqLengthsKeyOptional, numHeadsQ, numHeadsK, headDim, 
                          queryQuantMode, keyQuantMode, batchSizeOptional, maxSeqlenQOptional, maxSeqlenKOptional, 
                          layoutQueryOptional, layoutKeyOptional, sparseCountOptional, sparseModeOptional, 
                          preTokensOptional, nextTokensOptional, cmpRatioOptional),
                   DFX_OUT(metaData));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = ParamsCheck(actualSeqLengthsQueryOptional, actualSeqLengthsKeyOptional, numHeadsQ, numHeadsK,
                           batchSizeOptional, maxSeqlenQOptional, maxSeqlenKOptional, layoutQueryOptional, 
                           layoutKeyOptional, sparseModeOptional, preTokensOptional, nextTokensOptional, 
                           cmpRatioOptional);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    
    const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
    uint32_t aicCoreNum = npuInfo.GetCubeCoreNum();
    uint32_t aivCoreNum = npuInfo.GetVectorCoreNum();
    const char* socVersion = npuInfo.GetSocLongVersion().c_str();
    // 核心数校验
    if (aicCoreNum == 0 || aivCoreNum == 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "AIC num or AIV num should not be 0, but got %u and %u", aicCoreNum, aivCoreNum);
        return ACLNN_ERR_PARAM_INVALID;
    }

    auto actualSeqLengthsQueryOptionalContiguous = l0op::Contiguous(actualSeqLengthsQueryOptional, uniqueExecutor.get());
    CHECK_RET(actualSeqLengthsQueryOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto actualSeqLengthsKeyOptionalContiguous = l0op::Contiguous(actualSeqLengthsKeyOptional, uniqueExecutor.get());
    CHECK_RET(actualSeqLengthsKeyOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto output = l0op::QuantLightningIndexerMetadata(
        actualSeqLengthsQueryOptionalContiguous, actualSeqLengthsKeyOptionalContiguous, aicCoreNum, aivCoreNum, 
        socVersion, numHeadsQ, numHeadsK, headDim, queryQuantMode, keyQuantMode, batchSizeOptional, maxSeqlenQOptional, 
        maxSeqlenKOptional, layoutQueryOptional, layoutKeyOptional, sparseCountOptional, sparseModeOptional, 
        preTokensOptional, nextTokensOptional, cmpRatioOptional, metaData, uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus
aclnnQuantLightningIndexerMetadata(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) 
{
    L2_DFX_PHASE_2(aclnnQuantLightningIndexerMetadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
