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
 * \file l0_attention_pioneer_metadata.cpp
 * \brief L0 API implementation for AttentionPioneerMetadata AICPU operator.
 */

#include "l0_attention_pioneer_metadata.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(AttentionPioneerMetadata);

const aclTensor* AttentionPioneerMetadata(
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
    const char *socVersion,
    int64_t aicCoreNum,
    int64_t aivCoreNum,
    const aclTensor* metaData,
    aclOpExecutor* executor) {
    L0_DFX(AttentionPioneerMetadata, actualSeqLengthsOptional, actualSeqLengthsKvOptional,
            numHeadsQ, numHeadsKv, headDimQk, headDimV,
            batchSizeOptional, sparseModeOptional, preTokensOptional, nextTokensOptional,
            inputLayoutOptional, kSinkNumOptional,
            ropeHeadDimOptional, blockSizeOptional, socVersion, aicCoreNum, aivCoreNum, metaData);

    static internal::AicpuTaskSpace space("AttentionPioneerMetadata");

    auto ret = ADD_TO_LAUNCHER_LIST_AICPU(
        AttentionPioneerMetadata,
        OP_ATTR_NAMES({"num_heads_q", "num_heads_kv", "head_dim_qk", "head_dim_v", "batch_size", "sparse_mode",
                    "pre_tokens", "next_tokens", "input_layout", "k_sink_num",
                    "rope_head_dim", "block_size", "soc_version", "aic_core_num", "aiv_core_num"}),
        OP_INPUT(actualSeqLengthsOptional, actualSeqLengthsKvOptional),
        OP_OUTPUT(metaData),
        OP_ATTR(numHeadsQ, numHeadsKv, headDimQk, headDimV, batchSizeOptional, sparseModeOptional,
                preTokensOptional, nextTokensOptional, inputLayoutOptional,
                kSinkNumOptional, ropeHeadDimOptional, blockSizeOptional,
                socVersion, aicCoreNum, aivCoreNum));
    OP_CHECK(ret == ACL_SUCCESS,
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR,
                "AttentionPioneerMetadata ADD_TO_LAUNCHER_LIST_AICPU failed."),
        return nullptr);
    return metaData;
}

}  // namespace l0op
