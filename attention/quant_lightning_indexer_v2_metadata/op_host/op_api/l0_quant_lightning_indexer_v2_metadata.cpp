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
 * \file l0_quant_lightning_indexer_v2_metadata.cpp
 * \brief
 */

#include "l0_quant_lightning_indexer_v2_metadata.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(QuantLightningIndexerV2Metadata);

const aclTensor *QuantLightningIndexerV2Metadata(
    const aclTensor *cuSeqlensQOptional, const aclTensor *cuSeqlensKOptional, const aclTensor *sequsedQOptional,
    const aclTensor *sequsedKOptional, const aclTensor *cmpResidualKOptional,
    int64_t numHeadsQ, int64_t numHeadsK, int64_t headDim, int64_t topk, int64_t qQuantMode, int64_t kQuantMode,
    int64_t batchSizeOptional, int64_t maxSeqlenQOptional, int64_t maxSeqlenKOptional, char *layoutQOptional,
    char *layoutKOptional, int64_t maskModeOptional, int64_t cmpRatioOptional,
    int64_t aicCoreNum, int64_t aivCoreNum, const char *socVersion,
    const aclTensor *metaData, aclOpExecutor *executor)
{
    L0_DFX(QuantLightningIndexerV2Metadata, cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional, sequsedKOptional,
        cmpResidualKOptional,
        numHeadsQ, numHeadsK, headDim, topk, qQuantMode, kQuantMode,
        batchSizeOptional, maxSeqlenQOptional, maxSeqlenKOptional, layoutQOptional, layoutKOptional, maskModeOptional,
        cmpRatioOptional,
        aicCoreNum, aivCoreNum, socVersion, metaData);

    static internal::AicpuTaskSpace space("QuantLightningIndexerV2Metadata");

    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        QuantLightningIndexerV2Metadata,
        OP_ATTR_NAMES({ "num_heads_q", "num_heads_k", "head_dim", "topk", "q_quant_mode", "k_quant_mode",
                        "batch_size", "max_seqlen_q", "max_seqlen_k", "layout_q", "layout_k", "mask_mode", "cmp_ratio",
                        "aic_core_num", "aiv_core_num", "soc_version" }),
        OP_INPUT(cuSeqlensQOptional, cuSeqlensKOptional, sequsedQOptional, sequsedKOptional, cmpResidualKOptional),
        OP_OUTPUT(metaData),
        OP_ATTR(numHeadsQ, numHeadsK, headDim, topk, qQuantMode, kQuantMode, batchSizeOptional, maxSeqlenQOptional,
                maxSeqlenKOptional, layoutQOptional, layoutKOptional, maskModeOptional, cmpRatioOptional,
                aicCoreNum, aivCoreNum, socVersion));

    OP_CHECK(ret == ACL_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "QuantLightningIndexerV2Metadata ADD_TO_LAUNCHER_LIST_AICPU failed."),
             return nullptr);
    return metaData;
}
}  // namespace l0op
