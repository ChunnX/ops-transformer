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
#include "../sparse_flash_mla_metadata_check.h"

namespace {
const aclTensor* OptionalContiguous(const aclTensor* tensor, aclOpExecutor* executor, const char* tensorName,
                                    aclnnStatus &status)
{
    if (tensor == nullptr) {
        return nullptr;
    }
    auto contiguousTensor = l0op::Contiguous(tensor, executor);
    if (contiguousTensor == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "%s contiguous is none", tensorName);
        status = ACLNN_ERR_INNER_NULLPTR;
    }
    return contiguousTensor;
}
} // namespace

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnSparseFlashMlaMetadataGetWorkspaceSize(
    const aclTensor* cuSeqLensQOptional, const aclTensor* cuSeqLensOriKvOptional,
    const aclTensor* cuSeqLensCmpKvOptional, const aclTensor* sequsedQOptional, const aclTensor* sequsedOriKvOptional,
    int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv,
    int64_t oriTopK, int64_t cmpTopK, int64_t cmpRatio, int64_t oriMaskMode, int64_t cmpMaskMode, int64_t oriWinLeft,
    int64_t oriWinRight, char *layoutQOptional, char *layoutKvOptional, bool hasOriKv, bool hasCmpKv,
    const aclTensor* metadata, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnSparseFlashMlaMetadata,
                   DFX_IN(cuSeqLensQOptional, cuSeqLensOriKvOptional, cuSeqLensCmpKvOptional, sequsedQOptional,
                          sequsedOriKvOptional, numHeadsQ, numHeadsKv, headDim, batchSize, maxSeqlenQ, maxSeqlenKv,
                          oriTopK, cmpTopK, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight,
                          layoutQOptional, layoutKvOptional, hasOriKv, hasCmpKv),
                   DFX_OUT(metadata));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    int64_t effectiveCmpRatio = (cmpRatio == 0) ? 1 : cmpRatio;
    bool effectiveHasCmpKv = hasCmpKv && effectiveCmpRatio > 1;
    auto ret = ParamsCheck(cuSeqLensQOptional, cuSeqLensOriKvOptional, cuSeqLensCmpKvOptional, sequsedQOptional,
                           sequsedOriKvOptional, numHeadsQ, numHeadsKv, headDim, batchSize, maxSeqlenQ, maxSeqlenKv,
                           oriTopK, cmpTopK, effectiveCmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight,
                           layoutQOptional, layoutKvOptional, hasOriKv, effectiveHasCmpKv, metadata);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
    uint32_t aicCoreNum = npuInfo.GetCubeCoreNum();
    uint32_t aivCoreNum = npuInfo.GetVectorCoreNum();
    const char *socVersion = npuInfo.GetSocLongVersion().c_str();

    aclnnStatus contiguousStatus = ACLNN_SUCCESS;
    auto cuSeqLensQOptionalContiguous =
        OptionalContiguous(cuSeqLensQOptional, uniqueExecutor.get(), "cu_seqlens_q", contiguousStatus);
    CHECK_RET(contiguousStatus == ACLNN_SUCCESS, contiguousStatus);
    auto cuSeqLensOriKvOptionalContiguous =
        OptionalContiguous(cuSeqLensOriKvOptional, uniqueExecutor.get(), "cu_seqlens_ori_kv", contiguousStatus);
    CHECK_RET(contiguousStatus == ACLNN_SUCCESS, contiguousStatus);
    auto cuSeqLensCmpKvOptionalContiguous =
        OptionalContiguous(cuSeqLensCmpKvOptional, uniqueExecutor.get(), "cu_seqlens_cmp_kv", contiguousStatus);
    CHECK_RET(contiguousStatus == ACLNN_SUCCESS, contiguousStatus);
    auto sequsedQOptionalContiguous =
        OptionalContiguous(sequsedQOptional, uniqueExecutor.get(), "seqused_q", contiguousStatus);
    CHECK_RET(contiguousStatus == ACLNN_SUCCESS, contiguousStatus);
    auto sequsedOriKvOptionalContiguous =
        OptionalContiguous(sequsedOriKvOptional, uniqueExecutor.get(), "seqused_ori_kv", contiguousStatus);
    CHECK_RET(contiguousStatus == ACLNN_SUCCESS, contiguousStatus);

    auto output = l0op::SparseFlashMlaMetadata(
        cuSeqLensQOptionalContiguous, cuSeqLensOriKvOptionalContiguous, cuSeqLensCmpKvOptionalContiguous,
        sequsedQOptionalContiguous, sequsedOriKvOptionalContiguous, numHeadsQ, numHeadsKv, headDim, batchSize,
        maxSeqlenQ, maxSeqlenKv, oriTopK, cmpTopK, effectiveCmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft,
        oriWinRight, layoutQOptional, layoutKvOptional, hasOriKv, effectiveHasCmpKv, socVersion, aicCoreNum,
        aivCoreNum, metadata, uniqueExecutor.get());
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
