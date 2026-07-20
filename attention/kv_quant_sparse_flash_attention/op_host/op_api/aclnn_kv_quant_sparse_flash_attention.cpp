/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstring>
#include "graph/types.h"
#include "aclnn_kv_quant_sparse_flash_attention.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
extern aclnnStatus aclnnInnerKvQuantSparseFlashAttentionGetWorkspaceSize(
    const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *sparseIndices,
    const aclTensor *keyDequantScaleOptional, const aclTensor *valueDequantScaleOptional,
    const aclTensor *blockTableOptional, const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKvOptional, const aclTensor *sinksOptional, double scaleValue,
    int64_t keyQuantMode, int64_t valueQuantMode, int64_t sparseBlockSizeOptional, char *layoutQueryOptional,
    char *layoutKvOptional, int64_t sparseModeOptional, int64_t preTokensOptional,
    int64_t nextTokensOptional, int64_t attentionModeOptional, int64_t quantScaleRepoModeOptional,
    int64_t tileSizeOptional, int64_t ropeHeadDimOptional, const aclTensor *attentionOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);

extern aclnnStatus aclnnInnerKvQuantSparseFlashAttention(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                         const aclrtStream stream);

aclnnStatus aclnnKvQuantSparseFlashAttentionGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *sparseIndices,
    const aclTensor *keyDequantScaleOptional,
    const aclTensor *valueDequantScaleOptional,
    const aclTensor *blockTableOptional,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKvOptional,
    double           scaleValue,
    int64_t          keyQuantMode,
    int64_t          valueQuantMode,
    int64_t          sparseBlockSizeOptional,
    char             *layoutQueryOptional,
    char             *layoutKvOptional,
    int64_t          sparseModeOptional,
    int64_t          preTokensOptional,
    int64_t          nextTokensOptional,
    int64_t          attentionModeOptional,
    int64_t          quantScaleRepoModeOptional,
    int64_t          tileSizeOptional,
    int64_t          ropeHeadDimOptional,
    const aclTensor *attentionOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    return aclnnInnerKvQuantSparseFlashAttentionGetWorkspaceSize(
        query, key, value, sparseIndices, keyDequantScaleOptional, valueDequantScaleOptional, blockTableOptional,
        actualSeqLengthsQueryOptional, actualSeqLengthsKvOptional, nullptr, scaleValue, keyQuantMode,
        valueQuantMode, sparseBlockSizeOptional, layoutQueryOptional, layoutKvOptional, sparseModeOptional,
        preTokensOptional, nextTokensOptional, attentionModeOptional, quantScaleRepoModeOptional, tileSizeOptional,
        ropeHeadDimOptional, attentionOut, workspaceSize, executor);
}

aclnnStatus aclnnKvQuantSparseFlashAttention(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                     const aclrtStream stream)
{
    return aclnnInnerKvQuantSparseFlashAttention(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif
