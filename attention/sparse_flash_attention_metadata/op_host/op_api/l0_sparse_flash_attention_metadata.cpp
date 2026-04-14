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
 * \file l0_sparse_flash_attention_metadata.cpp
 * \brief
 */

#include "l0_sparse_flash_attention_metadata.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(SparseFlashAttentionMetadata);

const aclTensor* SparseFlashAttentionMetadata(
    const aclTensor* actualSeqLengthsQueryOptional, const aclTensor* actualSeqLengthsKvOptional, int64_t numHeadsQ, 
    int64_t numHeadsKv, int64_t headDim, int64_t batchSizeOptional, int64_t maxSeqlenQOptional, 
    int64_t maxSeqlenKvOptional, int64_t ropeHeadDimOptional, int64_t topKOptional, int64_t sparseBlockSizeOptional, 
    char *layoutQueryOptional, char *layoutKvOptional, int64_t sparseModeOptional, int64_t preTokensOptional, 
    int64_t  nextTokensOptional, int64_t  attentionModeOptional, const char *socVersion, int64_t aicCoreNum, 
    int64_t aivCoreNum, const aclTensor* metaData, aclOpExecutor* executor) 
{
    L0_DFX(SparseFlashAttentionMetadata, actualSeqLengthsQueryOptional, actualSeqLengthsKvOptional, numHeadsQ, 
           numHeadsKv, headDim, batchSizeOptional, maxSeqlenQOptional, maxSeqlenKvOptional, ropeHeadDimOptional, 
           topKOptional, sparseBlockSizeOptional, layoutQueryOptional, layoutKvOptional, sparseModeOptional, 
           preTokensOptional, nextTokensOptional, attentionModeOptional, socVersion, aicCoreNum, aivCoreNum, metaData);

    static internal::AicpuTaskSpace space("SparseFlashAttentionMetadata");

    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
    SparseFlashAttentionMetadata,
    OP_ATTR_NAMES({"num_heads_q", "num_heads_kv", "head_dim", "batch_size", "max_seqlen_q", "max_seqlen_kv", 
                   "rope_head_dim", "topk", "sparse_block_size", "layout_query", "layout_kv", "sparse_mode", 
                   "pre_tokens", "next_tokens", "attention_mode", "soc_version", "aic_core_num", "aiv_core_num"}),
    OP_INPUT(actualSeqLengthsQueryOptional, actualSeqLengthsKvOptional), 
    OP_OUTPUT(metaData),
    OP_ATTR(numHeadsQ, numHeadsKv, headDim, batchSizeOptional, maxSeqlenQOptional, maxSeqlenKvOptional, 
            ropeHeadDimOptional, topKOptional, sparseBlockSizeOptional, layoutQueryOptional, layoutKvOptional, 
            sparseModeOptional, preTokensOptional, nextTokensOptional, attentionModeOptional, socVersion, aicCoreNum, 
            aivCoreNum));
    OP_CHECK(ret == ACL_SUCCESS,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                    "SparseFlashAttentionMetadata"
                    " ADD_TO_LAUNCHER_LIST_AICPU failed."),
            return nullptr);
    return metaData;
}

}  // namespace l0op
