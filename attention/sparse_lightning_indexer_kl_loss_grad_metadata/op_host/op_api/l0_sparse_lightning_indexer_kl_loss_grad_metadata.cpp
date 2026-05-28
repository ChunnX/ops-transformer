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
 * \file l0_sparse_lightning_indexer_kl_loss_grad_metadata.cpp
 * \brief
 */

#include "l0_sparse_lightning_indexer_kl_loss_grad_metadata.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(SparseLightningIndexerKLLossGradMetadata);

const aclTensor* SparseLightningIndexerKLLossGradMetadata(
    const aclTensor* cuSeqLensQOptional,
    const aclTensor* cuSeqLensKOptional,
    const aclTensor* seqUsedQOptional,
    const aclTensor* seqUsedKOptional,
    const aclTensor* cmpResidualKOptional,
    int64_t bSize,
    int64_t s1Size,
    int64_t s2Size,
    int64_t kSize,
    int64_t maskMode,
    int64_t cmpRatio,
    char *layoutQ,
    char *layoutK,
    int64_t aicCoreNum,
    const aclTensor* metadata,
    aclOpExecutor* executor)
{
    L0_DFX(SparseLightningIndexerKLLossGradMetadata, cuSeqLensQOptional, cuSeqLensKOptional,
           seqUsedQOptional, seqUsedKOptional, cmpResidualKOptional,
           bSize, s1Size, s2Size, kSize, maskMode, cmpRatio, layoutQ, layoutK, aicCoreNum, metadata);

    static internal::AicpuTaskSpace space("SparseLightningIndexerKLLossGradMetadata");

    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        SparseLightningIndexerKLLossGradMetadata,
        OP_ATTR_NAMES({"b_size", "s1_size", "s2_size", "k_size", "mask_mode", "cmp_ratio", "layout_q", "layout_k",
                       "aic_core_num"}),
        OP_INPUT(cuSeqLensQOptional, cuSeqLensKOptional, seqUsedQOptional, seqUsedKOptional, cmpResidualKOptional),
        OP_OUTPUT(metadata),
        OP_ATTR(bSize, s1Size, s2Size, kSize, maskMode, cmpRatio, layoutQ, layoutK, aicCoreNum));
    OP_CHECK(ret == ACL_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SparseLightningIndexerKLLossGradMetadata ADD_TO_LAUNCHER_LIST_AICPU failed."),
             return nullptr);
    return metadata;
}
} // namespace l0op
