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
 * \file aclnn_sparse_lightning_indexer_kl_loss_grad_metadata.cpp
 * \brief
 */

#include "aclnn_sparse_lightning_indexer_kl_loss_grad_metadata.h"
#include "sparse_lightning_indexer_kl_loss_grad_metadata.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/platform.h"
#include "../sparse_lightning_indexer_kl_loss_grad_metadata_check.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnSparseLightningIndexerKLLossGradMetadataGetWorkspaceSize(
    const aclTensor* cuSeqlensQOptional, const aclTensor* cuSeqlensKOptional, const aclTensor* sequsedQOptional,
    const aclTensor* sequsedKOptional, const aclTensor* cmpResidualKOptional, int64_t batchSize, int64_t maxSeqlenQ,
    int64_t maxSeqlenK, int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim, int64_t topk, char *layoutQOptional,
    char *layoutKOptional, int64_t maskMode, int64_t cmpRatio, const aclTensor* metadata, uint64_t* workspaceSize,
    aclOpExecutor** executor)
{
    if (workspaceSize == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "workspaceSize is nullptr");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (executor == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "executor is nullptr");
        return ACLNN_ERR_INNER_NULLPTR;
    }
    L2_DFX_PHASE_1(aclnnSparseLightningIndexerKLLossGradMetadata,
                   DFX_IN(cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional, sequsedKOptional,
                          cmpResidualKOptional, batchSize, maxSeqlenQ, maxSeqlenK, numHeadsQ, numHeadsK, headDim,
                          topk, layoutQOptional, layoutKOptional, maskMode, cmpRatio),
                   DFX_OUT(metadata));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
    uint32_t aicCoreNum = npuInfo.GetCubeCoreNum();
    uint32_t aivCoreNum = npuInfo.GetVectorCoreNum();
    const std::string socVersion = npuInfo.GetSocLongVersion();

    auto ret = ParamsCheckSli(cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional, sequsedKOptional,
                               cmpResidualKOptional, batchSize, maxSeqlenQ, maxSeqlenK, numHeadsQ, numHeadsK, headDim,
                               topk, layoutQOptional, layoutKOptional, maskMode, cmpRatio, metadata, aicCoreNum,
                               aivCoreNum, socVersion);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const aclTensor *cuSeqLensQContiguous = nullptr;
    if (cuSeqlensQOptional != nullptr) {
        cuSeqLensQContiguous = l0op::Contiguous(cuSeqlensQOptional, uniqueExecutor.get());
        if (cuSeqLensQContiguous == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "cu_seqlens_q contiguous is null");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    const aclTensor *cuSeqLensKContiguous = nullptr;
    if (cuSeqlensKOptional != nullptr) {
        cuSeqLensKContiguous = l0op::Contiguous(cuSeqlensKOptional, uniqueExecutor.get());
        if (cuSeqLensKContiguous == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "cu_seqlens_k contiguous is null");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    const aclTensor *seqUsedQContiguous = nullptr;
    if (sequsedQOptional != nullptr) {
        seqUsedQContiguous = l0op::Contiguous(sequsedQOptional, uniqueExecutor.get());
        if (seqUsedQContiguous == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "seqused_q contiguous is null");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    const aclTensor *seqUsedKContiguous = nullptr;
    if (sequsedKOptional != nullptr) {
        seqUsedKContiguous = l0op::Contiguous(sequsedKOptional, uniqueExecutor.get());
        if (seqUsedKContiguous == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "seqused_k contiguous is null");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }
    const aclTensor *cmpResidualKContiguous = nullptr;
    if (cmpResidualKOptional != nullptr) {
        cmpResidualKContiguous = l0op::Contiguous(cmpResidualKOptional, uniqueExecutor.get());
        if (cmpResidualKContiguous == nullptr) {
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "cmp_residual_k contiguous is null");
            return ACLNN_ERR_INNER_NULLPTR;
        }
    }

    auto output = l0op::SparseLightningIndexerKLLossGradMetadata(
        cuSeqLensQContiguous, cuSeqLensKContiguous, seqUsedQContiguous, seqUsedKContiguous, cmpResidualKContiguous,
        batchSize, maxSeqlenQ, maxSeqlenK, numHeadsQ, numHeadsK, headDim, topk, layoutQOptional, layoutKOptional, maskMode, cmpRatio,
        aicCoreNum, metadata, uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSparseLightningIndexerKLLossGradMetadata(void *workspace, uint64_t workspaceSize,
    aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSparseLightningIndexerKLLossGradMetadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
