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
 * \file l0_kv_quant_sparse_flash_attention_metadata.cpp
 * \brief
 */

#include "l0_kv_quant_sparse_flash_attention_metadata.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(KvQuantSparseFlashAttentionMetadata);

const aclTensor* KvQuantSparseFlashAttentionMetadata(
    const aclTensor* cuSeqLensQOptional, const aclTensor* cuSeqLensKvOptional, const aclTensor* sequsedQOptional,
    const aclTensor* sequsedKvOptional, const aclTensor* topkLengthOptional, int64_t numHeadsQ, int64_t numHeadsKv,
    int64_t headDim, int64_t batchSizeOptional, int64_t maxSeqlenQOptional, int64_t maxSeqlenKvOptional,
    int64_t topKOptional, int64_t kvQuantMode, int64_t tileSizeOptional, int64_t ropeHeadDimOptional, 
    int64_t maskModeOptional, int64_t winLeftOptional, int64_t winRightOptional, char *layoutQOptional, 
    char *layoutKvOptional, const char *socVersion, int64_t aicCoreNum, int64_t aivCoreNum, const aclTensor* metaData,
    aclOpExecutor* executor) 
{
    L0_DFX(KvQuantSparseFlashAttentionMetadata, cuSeqLensQOptional, cuSeqLensKvOptional, sequsedQOptional, 
           sequsedKvOptional, topkLengthOptional, numHeadsQ, numHeadsKv, headDim, batchSizeOptional, maxSeqlenQOptional, 
           maxSeqlenKvOptional, topKOptional, kvQuantMode, tileSizeOptional, ropeHeadDimOptional, maskModeOptional, 
           winLeftOptional, winRightOptional, layoutQOptional, layoutKvOptional, socVersion, aicCoreNum, aivCoreNum, 
           metaData);

    static internal::AicpuTaskSpace space("KvQuantSparseFlashAttentionMetadata");

    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
    KvQuantSparseFlashAttentionMetadata,
    OP_ATTR_NAMES({"num_heads_q", "num_heads_kv", "head_dim", "batch_size", "max_seqlen_q", "max_seqlen_kv", 
                   "topk", "kv_quant_mode", "tile_size", "rope_head_dim", "mask_mode", "win_left", "win_right", 
                   "layout_q", "layout_kv", "soc_version", "aic_core_num", "aiv_core_num"}),
    OP_INPUT(cuSeqLensQOptional, cuSeqLensKvOptional, sequsedQOptional, sequsedKvOptional, topkLengthOptional), 
    OP_OUTPUT(metaData),
    OP_ATTR(numHeadsQ, numHeadsKv, headDim, batchSizeOptional, maxSeqlenQOptional, maxSeqlenKvOptional, topKOptional, 
            kvQuantMode, tileSizeOptional, ropeHeadDimOptional, maskModeOptional, winLeftOptional, winRightOptional, 
            layoutQOptional, layoutKvOptional, socVersion, aicCoreNum, aivCoreNum));
    OP_CHECK(ret == ACL_SUCCESS,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                    "KvQuantSparseFlashAttentionMetadata"
                    " ADD_TO_LAUNCHER_LIST_AICPU failed."),
            return nullptr);
    return metaData;
}

}  // namespace l0op
