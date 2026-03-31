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
 * \file aclnn_flash_attn_metadata.cpp
 * \brief
 */

#include "aclnn_flash_attn_metadata.h"
#include "l0_flash_attn_metadata.h"
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

static aclnnStatus ParamsCheck(
    const aclTensor* actualSeqLengthsQOptional,
    const aclTensor* actualSeqLengthsKvOptional,
    int64_t numHeadsQ,
    int64_t numHeadsKv,
    int64_t headDim,
    int64_t sparseMode,
    int64_t batchSizeOptional,
    int64_t maxSeqlenQOptional,
    int64_t maxSeqlenKvOptional,
    const aclTensor* metaData) {
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttnMetadataGetWorkspaceSize(
    const aclTensor* actualSeqLengthsQOptional,
    const aclTensor* actualSeqLengthsKvOptional,
    int64_t numHeadsQ,
    int64_t numHeadsKv,
    int64_t headDim,
    int64_t sparseMode,
    int64_t batchSizeOptional,
    int64_t maxSeqlenQOptional,
    int64_t maxSeqlenKvOptional,
    const aclTensor* metaData,
    uint64_t* workspaceSize,
    aclOpExecutor** executor) {
    L2_DFX_PHASE_1(aclnnFlashAttnMetadata,
                    DFX_IN(actualSeqLengthsQOptional, actualSeqLengthsKvOptional,
                        numHeadsQ, numHeadsKv, headDim, sparseMode, batchSizeOptional, maxSeqlenQOptional, maxSeqlenKvOptional),
                    DFX_OUT(metaData));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = ParamsCheck(actualSeqLengthsQOptional, actualSeqLengthsKvOptional,
                        numHeadsQ, numHeadsKv, headDim, sparseMode, batchSizeOptional, maxSeqlenQOptional, maxSeqlenKvOptional, metaData);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
    uint32_t aicCoreNum = npuInfo.GetCubeCoreNum();
    uint32_t aivCoreNum = npuInfo.GetVectorCoreNum();
    const char *socVersion = npuInfo.GetSocLongVersion().c_str();
    auto output = l0op::FlashAttnMetadata(
        actualSeqLengthsQOptional, actualSeqLengthsKvOptional, numHeadsQ, numHeadsKv, headDim, sparseMode,
        batchSizeOptional, maxSeqlenQOptional, maxSeqlenKvOptional,
        socVersion, aicCoreNum, aivCoreNum, metaData, 
        uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus
aclnnFlashAttnMetadata(void *workspace, uint64_t workspaceSize,
                                aclOpExecutor *executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnFlashAttnMetadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
