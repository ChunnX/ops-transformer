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
 * \file aclnn_attention_pioneer_metadata.cpp
 * \brief aclnn API implementation for AttentionPioneerMetadata AICPU operator.
 */

#include <string>
#include "aclnn_attention_pioneer_metadata.h"
#include "l0_attention_pioneer_metadata.h"
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

static aclnnStatus ParamsCheck(const aclTensor *actualSeqLengthsOptional,
                            const aclTensor *actualSeqLengthsKvOptional,
                            int64_t numHeadsQ,
                            int64_t numHeadsKv,
                            int64_t headDimQk,
                            int64_t headDimV,
                            int64_t batchSizeOptional,
                            int64_t sparseModeOptional,
                            int64_t preTokensOptional,
                            int64_t nextTokensOptional,
                            char *inputLayoutOptional,
                            int64_t kSinkNumOptional,
                            int64_t ropeHeadDimOptional,
                            int64_t blockSizeOptional,
                            int64_t aicCoreNumOptional,
                            int64_t aivCoreNumOptional,
                            const aclTensor* metaData) {
    // input / output tensor nullptr check
    CHECK_RET(actualSeqLengthsOptional != nullptr, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(actualSeqLengthsKvOptional != nullptr, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(metaData != nullptr, ACLNN_ERR_PARAM_INVALID);

    // head num validation
    CHECK_RET(numHeadsQ > 0, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(numHeadsKv > 0, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(numHeadsQ % numHeadsKv == 0, ACLNN_ERR_PARAM_INVALID);

    // head dim validation
    CHECK_RET(headDimQk > 0, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(headDimV > 0, ACLNN_ERR_PARAM_INVALID);

    // sparse_mode: only 2 (LEFT_UP_CAUSAL) and 4 (BAND) supported
    constexpr int64_t SPARSE_MODE_LEFT_UP = 2;
    constexpr int64_t SPARSE_MODE_BAND = 4;
    CHECK_RET(sparseModeOptional == SPARSE_MODE_LEFT_UP || sparseModeOptional == SPARSE_MODE_BAND,
              ACLNN_ERR_PARAM_INVALID);

    // pre_tokens / next_tokens: nexttoken line should not be higher than pretoken line
    CHECK_RET((-nextTokensOptional) <= preTokensOptional, ACLNN_ERR_PARAM_INVALID);

    // input_layout: only "TND" and "TND_NTD" supported
    if (inputLayoutOptional != nullptr) {
        std::string layout(inputLayoutOptional);
        CHECK_RET(layout == "TND" || layout == "TND_NTD", ACLNN_ERR_PARAM_INVALID);
    }

    // k_sink_num: only 0 (disabled) or 128 (enabled)
    CHECK_RET(kSinkNumOptional == 0 || kSinkNumOptional == 128, ACLNN_ERR_PARAM_INVALID);

    // block_size >= 0
    CHECK_RET(blockSizeOptional >= 0, ACLNN_ERR_PARAM_INVALID);

    // rope_head_dim >= 0
    CHECK_RET(ropeHeadDimOptional >= 0, ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnAttentionPioneerMetadataGetWorkspaceSize(
    const aclTensor *actualSeqLengthsOptional,
    const aclTensor *actualSeqLengthsKvOptional,
    int64_t numHeadsQ,
    int64_t numHeadsKv,
    int64_t headDimQk,
    int64_t headDimV,
    int64_t batchSizeOptional,
    int64_t sparseModeOptional,
    int64_t preTokensOptional,
    int64_t nextTokensOptional,
    char *inputLayoutOptional,
    int64_t kSinkNumOptional,
    int64_t ropeHeadDimOptional,
    int64_t blockSizeOptional,
    int64_t aicCoreNumOptional,
    int64_t aivCoreNumOptional,
    const aclTensor* metaData,
    uint64_t* workspaceSize,
    aclOpExecutor** executor) {
    L2_DFX_PHASE_1(aclnnAttentionPioneerMetadata,
                   DFX_IN(actualSeqLengthsOptional, actualSeqLengthsKvOptional, numHeadsQ, numHeadsKv,
                          headDimQk, headDimV, batchSizeOptional, sparseModeOptional, preTokensOptional,
                          nextTokensOptional, inputLayoutOptional,
                          kSinkNumOptional, ropeHeadDimOptional, blockSizeOptional,
                          aicCoreNumOptional, aivCoreNumOptional),
                   DFX_OUT(metaData));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    auto ret = ParamsCheck(
        actualSeqLengthsOptional, actualSeqLengthsKvOptional, numHeadsQ, numHeadsKv, headDimQk, headDimV,
        batchSizeOptional, sparseModeOptional, preTokensOptional, nextTokensOptional, inputLayoutOptional,
        kSinkNumOptional, ropeHeadDimOptional, blockSizeOptional,
        aicCoreNumOptional, aivCoreNumOptional, metaData);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    const op::PlatformInfo &npuInfo = op::GetCurrentPlatformInfo();
    const char *socVersion = npuInfo.GetSocLongVersion().c_str();
    aicCoreNumOptional = npuInfo.GetCubeCoreNum();
    aivCoreNumOptional = npuInfo.GetVectorCoreNum();
    auto output = l0op::AttentionPioneerMetadata(
        actualSeqLengthsOptional, actualSeqLengthsKvOptional, numHeadsQ, numHeadsKv, headDimQk, headDimV,
        batchSizeOptional, sparseModeOptional, preTokensOptional, nextTokensOptional, inputLayoutOptional,
        kSinkNumOptional, ropeHeadDimOptional, blockSizeOptional,
        socVersion, aicCoreNumOptional, aivCoreNumOptional, metaData, uniqueExecutor.get());
    CHECK_RET(output != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = 0;
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

__attribute__((visibility("default"))) aclnnStatus
aclnnAttentionPioneerMetadata(void *workspace, uint64_t workspaceSize,
                              aclOpExecutor *executor, aclrtStream stream) {
    L2_DFX_PHASE_2(aclnnAttentionPioneerMetadata);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
