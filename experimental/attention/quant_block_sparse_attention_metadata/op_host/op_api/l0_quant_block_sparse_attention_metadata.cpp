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
 * \file l0_quant_block_sparse_attention_metadata.cpp
 * \brief
 */

#include "l0_quant_block_sparse_attention_metadata.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(QuantBlockSparseAttentionMetadata);

const aclTensor *QuantBlockSparseAttentionMetadata(
    const aclTensor *sparseSeqLen, const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKvOptional,
    const aclTensor *sequsedQOptional, const aclTensor *sequsedKvOptional, int64_t batchSize, int64_t numHeadsQ,
    int64_t numHeadsKv, int64_t headDim, int64_t sparseBlockSizeQ, int64_t sparseBlockSizeKv, int64_t quantMode,
    int64_t maskMode, int64_t maxSeqlenQ, int64_t maxSeqlenKv, const char *layoutQOptional,
    const char *layoutKvOptional, const char *layoutSparseIndicesOptional, const char *socVersion,
    int64_t aicCoreNum, int64_t aivCoreNum, const aclTensor *metadata, aclOpExecutor *executor)
{
    L0_DFX(QuantBlockSparseAttentionMetadata, sparseSeqLen, cuSeqlensQOptional, cuSeqlensKvOptional,
            sequsedQOptional, sequsedKvOptional, batchSize, numHeadsQ, numHeadsKv, headDim, sparseBlockSizeQ,
            sparseBlockSizeKv, quantMode, maskMode, maxSeqlenQ, maxSeqlenKv, layoutQOptional, layoutKvOptional,
            layoutSparseIndicesOptional, socVersion, aicCoreNum, aivCoreNum, metadata);

    static internal::AicpuTaskSpace space("QuantBlockSparseAttentionMetadata");

    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        QuantBlockSparseAttentionMetadata,
        OP_ATTR_NAMES({"batch_size", "num_heads_q", "num_heads_kv", "head_dim", "sparse_block_size_q",
                       "sparse_block_size_kv", "quant_mode", "mask_mode", "max_seqlen_q", "max_seqlen_kv",
                       "layout_q", "layout_kv", "layout_sparse_indices", "soc_version", "aic_core_num",
                       "aiv_core_num"}),
        OP_INPUT(sparseSeqLen, cuSeqlensQOptional, cuSeqlensKvOptional, sequsedQOptional, sequsedKvOptional),
        OP_OUTPUT(metadata),
        OP_ATTR(batchSize, numHeadsQ, numHeadsKv, headDim, sparseBlockSizeQ, sparseBlockSizeKv, quantMode, maskMode,
                maxSeqlenQ, maxSeqlenKv, layoutQOptional, layoutKvOptional, layoutSparseIndicesOptional, socVersion,
                aicCoreNum, aivCoreNum));
    OP_CHECK(ret == ACL_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "QuantBlockSparseAttentionMetadata ADD_TO_LAUNCHER_LIST_AICPU failed."),
             return nullptr);
    return metadata;
}

} // namespace l0op
