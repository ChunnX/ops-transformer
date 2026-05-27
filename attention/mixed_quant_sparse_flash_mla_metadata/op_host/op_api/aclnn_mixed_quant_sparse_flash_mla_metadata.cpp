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
 * \file aclnn_mixed_quant_sparse_flash_mla_metadata.cpp
 * \brief
 */

#include "aclnn_mixed_quant_sparse_flash_mla_metadata.h"
#include "l0_mixed_quant_sparse_flash_mla_metadata.h"
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
#include "../mixed_quant_sparse_flash_mla_metadata_check.h"

#ifdef __cplusplus
extern "C" {
#endif

aclnnStatus aclnnMixedQuantSparseFlashMlaMetadataGetWorkspaceSize(
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensOriKvOptional,
    const aclTensor *cuSeqlensCmpKvOptional, const aclTensor *sequsedQOptional, const aclTensor *sequsedOriKvOptional,
    const aclTensor *sequsedCmpKvOptional, const aclTensor *cmpResidualKvOptional,
    const aclTensor *oriTopkLengthOptional, const aclTensor *cmpTopkLengthOptional, int64_t numHeadsQ,
    int64_t numHeadsKv, int64_t headDim, int64_t quantMode, int64_t batchSize, int64_t maxSeqlenQ, int64_t maxSeqlenKv,
    int64_t oriTopk, int64_t cmpTopk, int64_t ropeHeadDim, int64_t cmpRatio, int64_t oriMaskMode, int64_t cmpMaskMode,
    int64_t oriWinLeft, int64_t oriWinRight, const char *layoutQOptional, const char *layoutKvOptional,
    bool hasOriKv, bool hasCmpKv, int64_t batchInvariant, const aclTensor *metaData, uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnMixedQuantSparseFlashMlaMetadata,
                   DFX_IN(cuSeqlensQOptional, cuSeqlensOriKvOptional, cuSeqlensCmpKvOptional, sequsedQOptional,
                          sequsedOriKvOptional, sequsedCmpKvOptional, cmpResidualKvOptional, oriTopkLengthOptional,
                          cmpTopkLengthOptional, numHeadsQ, numHeadsKv, headDim, quantMode, batchSize, maxSeqlenQ,
                          maxSeqlenKv, oriTopk, cmpTopk, ropeHeadDim, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft,
                          oriWinRight, layoutQOptional, layoutKvOptional, hasOriKv, hasCmpKv, batchInvariant),
                   DFX_OUT(metaData));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
    uint32_t aicCoreNum = npuInfo.GetCubeCoreNum();
    uint32_t aivCoreNum = npuInfo.GetVectorCoreNum();
    const char *socVersion = npuInfo.GetSocLongVersion().c_str();

    auto ret = ParamsCheck(cuSeqlensQOptional, cuSeqlensOriKvOptional, cuSeqlensCmpKvOptional, sequsedQOptional,
                           sequsedOriKvOptional, sequsedCmpKvOptional, cmpResidualKvOptional, oriTopkLengthOptional,
                           cmpTopkLengthOptional, numHeadsQ, numHeadsKv, headDim, quantMode, batchSize, maxSeqlenQ,
                           maxSeqlenKv, oriTopk, cmpTopk, ropeHeadDim, cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft,
                           oriWinRight, layoutQOptional, layoutKvOptional, hasOriKv, hasCmpKv, batchInvariant,
                           aicCoreNum, aivCoreNum, socVersion, metaData);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    auto cuSeqlensQOptionalContiguous = l0op::Contiguous(cuSeqlensQOptional, uniqueExecutor.get());
    if (cuSeqlensQOptionalContiguous == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "cu_seqlens_q contiguous is none");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto cuSeqlensOriKvOptionalContiguous = l0op::Contiguous(cuSeqlensOriKvOptional, uniqueExecutor.get());
    if (cuSeqlensOriKvOptionalContiguous == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "cu_seqlens_ori_kv contiguous is none");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto cuSeqlensCmpKvOptionalContiguous = l0op::Contiguous(cuSeqlensCmpKvOptional, uniqueExecutor.get());
    if (cuSeqlensCmpKvOptionalContiguous == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "cu_seqlens_cmp_kv contiguous is none");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto sequsedQOptionalContiguous = l0op::Contiguous(sequsedQOptional, uniqueExecutor.get());
    if (sequsedQOptionalContiguous == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "seqused_q contiguous is none");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto sequsedOriKvOptionalContiguous = l0op::Contiguous(sequsedOriKvOptional, uniqueExecutor.get());
    if (sequsedOriKvOptionalContiguous == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "seqused_ori_kv contiguous is none");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto sequsedCmpKvOptionalContiguous = l0op::Contiguous(sequsedCmpKvOptional, uniqueExecutor.get());
    if (sequsedCmpKvOptionalContiguous == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "seqused_cmp_kv contiguous is none");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto cmpResidualKvOptionalContiguous = l0op::Contiguous(cmpResidualKvOptional, uniqueExecutor.get());
    if (cmpResidualKvOptionalContiguous == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "cmp_residual_kv contiguous is none");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto oriTopkLengthOptionalContiguous = l0op::Contiguous(oriTopkLengthOptional, uniqueExecutor.get());
    if (oriTopkLengthOptionalContiguous == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ori_topk_length contiguous is none");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto cmpTopkLengthOptionalContiguous = l0op::Contiguous(cmpTopkLengthOptional, uniqueExecutor.get());
    if (cmpTopkLengthOptionalContiguous == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "cmp_topk_length contiguous is none");
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto output = l0op::MixedQuantSparseFlashMlaMetadata(
        cuSeqlensQOptionalContiguous, cuSeqlensOriKvOptionalContiguous, cuSeqlensCmpKvOptionalContiguous,
        sequsedQOptionalContiguous, sequsedOriKvOptionalContiguous, sequsedCmpKvOptionalContiguous,
        cmpResidualKvOptionalContiguous, oriTopkLengthOptionalContiguous, cmpTopkLengthOptionalContiguous,
        numHeadsQ, numHeadsKv, headDim, quantMode, batchSize, maxSeqlenQ, maxSeqlenKv, oriTopk, cmpTopk, ropeHeadDim,
        cmpRatio, oriMaskMode, cmpMaskMode, oriWinLeft, oriWinRight, layoutQOptional, layoutKvOptional, hasOriKv,
        hasCmpKv, batchInvariant, socVersion, aicCoreNum, aivCoreNum, metaData, uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus aclnnMixedQuantSparseFlashMlaMetadata(
    void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMixedQuantSparseFlashMlaMetadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif