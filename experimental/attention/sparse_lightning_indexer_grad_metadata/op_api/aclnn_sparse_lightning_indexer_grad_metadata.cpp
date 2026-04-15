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
 * \file aclnn_sparse_lightning_indexer_grad_metadata.cpp
 * \brief
 */

#include "aclnn_sparse_lightning_indexer_grad_metadata.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/reshape.h"
#include "l0_sparse_lightning_indexer_grad_metadata.h"
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

static aclnnStatus ParamsCheck(const aclTensor* cuSeqLensQOptional, const aclTensor* cuSeqLensKOptional, 
                               const aclTensor* sequsedQOptional, const aclTensor* sequsedKOptional, 
                               const aclTensor* cmpResidualKOptional, int64_t numHeadsQ, int64_t numHeadsK, 
                               int64_t batchSizeOptional, int64_t maxSeqlenQOptional, int64_t maxSeqlenKOptional, 
                               char* layoutQOptional, char* layoutKOptional, int64_t maskModeOptional, 
                               int64_t cmpRatioOptional) 
{
    if (CheckSingleParamSlig(batchSizeOptional, maxSeqlenQOptional, maxSeqlenKOptional, numHeadsQ, numHeadsK, 
                         layoutQOptional, layoutKOptional, maskModeOptional) == ACLNN_SUCCESS && 
        CheckExistenceSlig(layoutQOptional, layoutKOptional, cuSeqLensQOptional, cuSeqLensKOptional) == ACLNN_SUCCESS && 
        CheckConsistencySlig(layoutQOptional, layoutKOptional, cuSeqLensQOptional, cuSeqLensKOptional, sequsedQOptional, 
                             sequsedKOptional, cmpResidualKOptional, batchSizeOptional) == ACLNN_SUCCESS && 
        CheckFeatureSlig(cmpRatioOptional) == ACLNN_SUCCESS) {
        return ACLNN_SUCCESS;
    } else {
        return ACLNN_ERR_PARAM_INVALID;
    }
}

aclnnStatus CheckSingleParamSlig(int64_t batchSizeOptional, int64_t maxSeqlenQOptional, int64_t maxSeqlenKOptional, 
                                 int64_t numHeadsQ, int64_t numHeadsK, char* layoutQOptional, char* layoutKOptional, 
                                 int64_t maskModeOptional)
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
    if (numHeadsQ < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "num_heads_q should not be negative, but got %d", numHeadsQ);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // num_heads_k 校验
    if (numHeadsK < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "num_heads_k should not be negative1, but got %d", numHeadsK);
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_q 校验
    if (strcmp(layoutQOptional, "TND") != 0 && strcmp(layoutQOptional, "BSND") != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_q, layout must be TND or BSND!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // layout_k 校验
    if (strcmp(layoutKOptional, "TND") != 0 && strcmp(layoutKOptional, "BSND") != 0) {
    	OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_k, layout must be TND/BSND!");
        return ACLNN_ERR_PARAM_INVALID;
    }
    // mask_mode 校验
    if (maskModeOptional != 0 && maskModeOptional != 3 && maskModeOptional != 4) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "mask_mode should be 0/3/4, but got %d", maskModeOptional);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckExistenceSlig(char* layoutQOptional, char* layoutKOptional, const aclTensor* cuSeqLensQOptional, 
                               const aclTensor* cuSeqLensKOptional)
{
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    // Query 存在性逻辑
    if (strcmp(layoutQOptional, "TND") == 0) {
        auto ret = aclGetViewShape(cuSeqLensQOptional, &viewDims, &viewDimsNum);
        if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum == 0 || viewDims[0] == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_q TND, cu_seqlens_q must be provided!");
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    if (strcmp(layoutKOptional, "TND") == 0) {
        auto ret = aclGetViewShape(cuSeqLensKOptional, &viewDims, &viewDimsNum);
        if (ret == ACLNN_ERR_PARAM_INVALID || viewDimsNum == 0 || viewDims[0] == 0) {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "For layout_k TND, seqused_k must be provided!");
            delete[] viewDims;
            return ACLNN_ERR_PARAM_INVALID;
        }
    }
    return ACLNN_SUCCESS;
}

int64_t GetQueryBatchSizeSlig(const aclTensor* cuSeqLensQOptional, const aclTensor* sequsedQOptional, 
                              char* layoutQOptional, int64_t batchSizeOptional)
{
    // 1. 如果sequsedQOptional 传了，使用sequsedQOptional获取BatchSize
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    auto ret = aclGetViewShape(sequsedQOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        int64_t dimSize = viewDims[0];
        delete[] viewDims;
        return dimSize;
    }
    // 2. sequsedQOptional 没传，判断 Layout
    if (strcmp(layoutQOptional, "TND") == 0) {
        // 如果是 TND，尝试使用 cuSeqLensQOptional获取BatchSize
        ret = aclGetViewShape(cuSeqLensQOptional, &viewDims, &viewDimsNum);
        if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
            int64_t dimSize = viewDims[0];
            delete[] viewDims;
            return dimSize - 1;
        }
    }
    // 3. 如果不是 TND，或者 cuSeqLensQOptional 为空，使用batchSizeOptional
    delete[] viewDims;
    return batchSizeOptional;
}

int64_t GetKeyBatchSizeSlig(const aclTensor* cuSeqLensKOptional, const aclTensor* sequsedKOptional, 
                           char* layoutKOptional, int64_t batchSizeOptional)
{
    // 1. 如果sequsedKvOptional传了，使用sequsedKvOptional获取BatchSize
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    auto ret = aclGetViewShape(sequsedKOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
        int64_t dimSize = viewDims[0];
        delete[] viewDims;
        return dimSize;
    }
    // 2. sequsedKvOptional 没传，判断 Layout
    if (strcmp(layoutKOptional, "TND") == 0) {
        // 如果是 TND，尝试使用 cuSeqLensOriKvOptional获取BatchSize
        ret = aclGetViewShape(cuSeqLensKOptional, &viewDims, &viewDimsNum);
        if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum > 0 && viewDims[0] != 0) {
            int64_t dimSize = viewDims[0];
            delete[] viewDims;
            return dimSize - 1;
        }
    }
    // 3. 如果不是 TND，或者 cuSeqLensOriKvOptional 为空，使用 batchSizeOptional
    delete[] viewDims;
    return batchSizeOptional;
}

aclnnStatus CheckConsistencySlig(char* layoutQOptional, char* layoutKOptional, const aclTensor* cuSeqLensQOptional, 
                                 const aclTensor* cuSeqLensKOptional, const aclTensor* sequsedQOptional, 
                                 const aclTensor* sequsedKOptional, const aclTensor* cmpResidualKOptional, 
                                 int64_t batchSizeOptional)
{
    int64_t *viewDims = nullptr;
    uint64_t viewDimsNum = 0;
    // 校验 cu_seqlens_q 维度
    auto ret = aclGetViewShape(cuSeqLensQOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of cu_seqlens_q must be 1, but got %ld", viewDimsNum);
        delete[] viewDims;
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 校验 cu_seqlens_k 维度
    ret = aclGetViewShape(cuSeqLensKOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of cu_seqlens_k must be 1, but got %ld", viewDimsNum);
        delete[] viewDims;
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 校验 seqused_q 维度
    ret = aclGetViewShape(sequsedQOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of seqused_q must be 1, but got %ld", viewDimsNum);
        delete[] viewDims;
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 校验 seqused_k 维度
    ret = aclGetViewShape(sequsedKOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of seqused_k must be 1, but got %ld", viewDimsNum);
        delete[] viewDims;
        return ACLNN_ERR_PARAM_INVALID;
    }
    // 校验 cmp_residual_k 维度
    ret = aclGetViewShape(cmpResidualKOptional, &viewDims, &viewDimsNum);
    if (ret != ACLNN_ERR_PARAM_INVALID && viewDimsNum != 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The dim num of cmp_residual_k must be 1, but got %ld", viewDimsNum);
        delete[] viewDims;
        return ACLNN_ERR_PARAM_INVALID;
    }
    int64_t queryBatchSize = GetQueryBatchSizeSlig(cuSeqLensQOptional, sequsedQOptional, layoutQOptional, 
                                                   batchSizeOptional);
    int64_t keyBatchSize = GetKeyBatchSizeSlig(cuSeqLensKOptional, sequsedKOptional, layoutKOptional, batchSizeOptional);
    if (queryBatchSize != keyBatchSize) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The batch_size obtained from q Tensor should be the same as "
                "that obtained from k tensor, but got %ld and %ld", queryBatchSize, keyBatchSize);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

aclnnStatus CheckFeatureSlig(int64_t cmpRatioOptional)
{
    // 压缩率校验
    if (cmpRatioOptional < 1 || cmpRatioOptional > 128) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cmp_ratio should be [1, 128], but got %d", cmpRatioOptional);
        return ACLNN_ERR_PARAM_INVALID;
    }
    const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
    const char* socVersion = npuInfo.GetSocLongVersion().c_str();
    // 校验 2 的幂次方: 1, 2, 4, ..., 128
    if ((cmpRatioOptional & (cmpRatioOptional - 1)) != 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "cmp_ratio should be 1/2/4/8/16/32/64/128, but got %d", 
                cmpRatioOptional);
        return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default")))
aclnnStatus aclnnSparseLightningIndexerGradMetadataGetWorkspaceSize(
    const aclTensor* cuSeqLensQOptional, const aclTensor* cuSeqLensKOptional, const aclTensor* sequsedQOptional, 
    const aclTensor* sequsedKOptional, const aclTensor* cmpResidualKOptional, int64_t numHeadsQ, int64_t numHeadsK, 
    int64_t headDim, int64_t batchSizeOptional, int64_t maxSeqlenQOptional, int64_t maxSeqlenKOptional, 
    int64_t topkOptional, char* layoutQOptional, char* layoutKOptional, int64_t maskModeOptional, 
    int64_t cmpRatioOptional, int64_t deterministicOptional, const aclTensor* metaData, uint64_t* workspaceSize, 
    aclOpExecutor** executor) 
{
    L2_DFX_PHASE_1(aclnnSparseLightningIndexerGradMetadata,
                   DFX_IN(cuSeqLensQOptional, cuSeqLensKOptional, sequsedQOptional, sequsedKOptional, cmpResidualKOptional, 
                          numHeadsQ, numHeadsK, headDim, batchSizeOptional, maxSeqlenQOptional, maxSeqlenKOptional, 
                          topkOptional, layoutQOptional, layoutKOptional, maskModeOptional, cmpRatioOptional, 
                          deterministicOptional),
                   DFX_OUT(metaData));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = ParamsCheck(cuSeqLensQOptional, cuSeqLensKOptional, sequsedQOptional, sequsedKOptional, 
                           cmpResidualKOptional, numHeadsQ, numHeadsK, batchSizeOptional, maxSeqlenQOptional, 
                           maxSeqlenKOptional, layoutQOptional, layoutKOptional, maskModeOptional, cmpRatioOptional);
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

    auto cuSeqLensQOptionalContiguous = l0op::Contiguous(cuSeqLensQOptional, uniqueExecutor.get());
    CHECK_RET(cuSeqLensQOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto cuSeqLensKOptionalContiguous = l0op::Contiguous(cuSeqLensKOptional, uniqueExecutor.get());
    CHECK_RET(cuSeqLensKOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto sequsedQOptionalContiguous = l0op::Contiguous(sequsedQOptional, uniqueExecutor.get());
    CHECK_RET(sequsedQOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto sequsedKOptionalContiguous = l0op::Contiguous(sequsedKOptional, uniqueExecutor.get());
    CHECK_RET(sequsedKOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto cmpResidualKOptionalContiguous = l0op::Contiguous(cmpResidualKOptional, uniqueExecutor.get());
    CHECK_RET(cmpResidualKOptionalContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto output = l0op::SparseLightningIndexerGradMetadata(
        cuSeqLensQOptionalContiguous, cuSeqLensKOptionalContiguous, sequsedQOptionalContiguous, 
        sequsedKOptionalContiguous, cmpResidualKOptionalContiguous, aicCoreNum, aivCoreNum, socVersion, numHeadsQ, 
        numHeadsK, headDim, batchSizeOptional, maxSeqlenQOptional, maxSeqlenKOptional, topkOptional, layoutQOptional, 
        layoutKOptional, maskModeOptional, cmpRatioOptional, deterministicOptional, metaData, uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus
aclnnSparseLightningIndexerGradMetadata(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream) 
{
    L2_DFX_PHASE_2(aclnnSparseLightningIndexerGradMetadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
