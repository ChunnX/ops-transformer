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
 * \file aclnn_sparse_flash_mla_metadata.cpp
 * \brief
 */

#include "aclnn_sparse_flash_mla_metadata.h"
#include "l0_sparse_flash_mla_metadata.h"
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
#include "../op_host/sparse_flash_mla_metadata_check.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace {
aclnnStatus MakeContiguousOptional(const aclTensor* input, const aclTensor*& output, aclOpExecutor* executor,
                                   const char* name)
{
    if (input == nullptr) {
        output = nullptr;
        return ACLNN_SUCCESS;
    }
    output = l0op::Contiguous(input, executor);
    if (output == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "%s contiguous is none", name);
        return ACLNN_ERR_INNER_NULLPTR;
    }
    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnSparseFlashMlaMetadataGetWorkspaceSize(
    const aclTensor* cuSeqLensQOptional, const aclTensor* cuSeqLensOriKvOptional,
    const aclTensor* cuSeqLensCmpKvOptional, const aclTensor* sequsedQOptional, const aclTensor* sequsedKvOptional,
    int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv,
    int64_t oriTopK, int64_t cmpTopK, int64_t cmpRatio, int64_t oriMaskMode, int64_t cmpMaskMode, int64_t oriWinLeft,
    int64_t oriWinRight, char *layoutQOptional, char *layoutKvOptional, bool hasOriKv, bool hasCmpKv,
    const aclTensor* metadata, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnSparseFlashMlaMetadata,
                   DFX_IN(cuSeqLensQOptional, cuSeqLensOriKvOptional, cuSeqLensCmpKvOptional, sequsedQOptional,
                          sequsedKvOptional, numHeadsQ, numHeadsKv, headDim, batchSize, maxSeqlenQ, maxSeqlenKv,
                          oriTopK, cmpTopK, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight,
                          layoutQOptional, layoutKvOptional, hasOriKv, hasCmpKv),
                   DFX_OUT(metadata));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = ParamsCheck(cuSeqLensQOptional, cuSeqLensOriKvOptional, cuSeqLensCmpKvOptional, sequsedQOptional,
                           sequsedKvOptional, numHeadsQ, numHeadsKv, headDim, batchSize, maxSeqlenQ, maxSeqlenKv,
                           oriTopK, cmpTopK, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight,
                           layoutQOptional, layoutKvOptional, hasOriKv, hasCmpKv, metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
    uint32_t aicCoreNum = npuInfo.GetCubeCoreNum();
    uint32_t aivCoreNum = npuInfo.GetVectorCoreNum();
    const char *socVersion = npuInfo.GetSocLongVersion().c_str();

    const aclTensor* cuSeqLensQOptionalContiguous = nullptr;
    const aclTensor* cuSeqLensOriKvOptionalContiguous = nullptr;
    const aclTensor* cuSeqLensCmpKvOptionalContiguous = nullptr;
    const aclTensor* sequsedQOptionalContiguous = nullptr;
    const aclTensor* sequsedKvOptionalContiguous = nullptr;

    CHECK_RET(MakeContiguousOptional(cuSeqLensQOptional, cuSeqLensQOptionalContiguous, uniqueExecutor.get(),
                                     "cu_seqlens_q") == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguousOptional(cuSeqLensOriKvOptional, cuSeqLensOriKvOptionalContiguous, uniqueExecutor.get(),
                                     "cu_seqlens_ori_kv") == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguousOptional(cuSeqLensCmpKvOptional, cuSeqLensCmpKvOptionalContiguous, uniqueExecutor.get(),
                                     "cu_seqlens_cmp_kv") == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguousOptional(sequsedQOptional, sequsedQOptionalContiguous, uniqueExecutor.get(),
                                     "seqused_q") == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(MakeContiguousOptional(sequsedKvOptional, sequsedKvOptionalContiguous, uniqueExecutor.get(),
                                     "seqused_kv") == ACLNN_SUCCESS, ACLNN_ERR_INNER_NULLPTR);

    auto output = l0op::SparseFlashMlaMetadata(
        cuSeqLensQOptionalContiguous, cuSeqLensOriKvOptionalContiguous, cuSeqLensCmpKvOptionalContiguous,
        sequsedQOptionalContiguous, sequsedKvOptionalContiguous, numHeadsQ, numHeadsKv, headDim, batchSize,
        maxSeqlenQ, maxSeqlenKv, oriTopK, cmpTopK, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight,
        layoutQOptional, layoutKvOptional, hasOriKv, hasCmpKv, socVersion, aicCoreNum, aivCoreNum, metadata,
        uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus
aclnnSparseFlashMlaMetadata(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSparseFlashMlaMetadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
