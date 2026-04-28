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
 * \file l0_sparse_flash_mla_metadata.cpp
 * \brief
 */

#include "l0_sparse_flash_mla_metadata.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(SparseFlashMlaMetadata);

const aclTensor* SparseFlashMlaMetadata(
    const aclTensor *cuSeqLensQOptional, const aclTensor *cuSeqLensOriKvOptional,
    const aclTensor *sequsedQOptional, const aclTensor *sequsedOriKvOptional,
    int64_t numHeadsQ, int64_t numHeadsKv, int64_t headDim, int64_t batchSizeOptional, int64_t maxSeqlenQOptional,
    int64_t maxSeqlenKvOptional, int64_t oriTopKOptional, int64_t oriMaskModeOptional,
    char *layoutQOptional, char *layoutKvOptional, bool hasOriKvOptional,
    const char *socVersion, int64_t aicCoreNum, int64_t aivCoreNum,
    const aclTensor* metaData, aclOpExecutor* executor)
{
    L0_DFX(SparseFlashMlaMetadata, 
        cuSeqLensQOptional, cuSeqLensOriKvOptional, sequsedQOptional, sequsedOriKvOptional,
        numHeadsQ, numHeadsKv, headDim,
        batchSizeOptional, maxSeqlenQOptional, maxSeqlenKvOptional, oriTopKOptional, oriMaskModeOptional,
        layoutQOptional, layoutKvOptional, hasOriKvOptional, socVersion, aicCoreNum, aivCoreNum, metaData);

    static internal::AicpuTaskSpace space("SparseFlashMlaMetadata");

    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
    SparseFlashMlaMetadata,
    OP_ATTR_NAMES({
        "num_heads_q", "num_heads_kv", "head_dim", "batch_size", "max_seqlen_q", "max_seqlen_kv", "ori_topk",
        "ori_mask_mode", "layout_q", "layout_kv", "has_ori_kv", "soc_version", "aic_core_num", "aiv_core_num"
    }),
    OP_INPUT(cuSeqLensQOptional, cuSeqLensOriKvOptional, sequsedQOptional, sequsedOriKvOptional),
    OP_ATTR(numHeadsQ, numHeadsKv, headDim, batchSizeOptional, maxSeqlenQOptional, maxSeqlenKvOptional, oriTopKOptional,
        oriMaskModeOptional, layoutQOptional, layoutKvOptional, hasOriKvOptional, socVersion, aicCoreNum, aivCoreNum),
    OP_OUTPUT(metaData));

    OP_CHECK(ret == ACL_SUCCESS,
            OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SparseFlashMlaMetadata ADD_TO_LAUNCHER_LIST_AICPU failed."),
            return nullptr);

    return metaData;
}

}  // namespace l0op
