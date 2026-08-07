/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS PROGRAM IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aclnn_pool_key_indexer.cpp
 * \brief PoolKeyIndexer two-phase aclnn API implementation
 */

#include <string.h>
#include "graph/types.h"
#include "aclnn_pool_key_indexer.h"

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

class TensorHolder {
public:
    TensorHolder(const aclTensor *&output, aclDataType dataType, std::string varName) {
        inner_ = nullptr;
        name_ = varName;
        if (output == nullptr) {
            std::vector<int64_t> shape = {0};
            int64_t addr = 0xff;
            inner_ = aclCreateTensor(shape.data(), shape.size(),
                dataType, shape.data(), 0, ACL_FORMAT_ND,
                shape.data(), shape.size(), static_cast<void *>(&addr));
            output = inner_;
        }
    }
    ~TensorHolder() {
        if (inner_) { aclDestroyTensor(inner_); inner_ = nullptr; }
    }
private:
    const aclTensor *inner_;
    std::string name_;
};

aclnnStatus aclnnPoolKeyIndexerGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *poolKey,
    const aclTensor *weights,
    const aclTensor *poolTailK,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKeyOptional,
    const aclTensor *blockTableOptional,
    const aclTensor *qDescaleOptional,
    const aclTensor *kDescaleOptional,
    char *layoutQueryOptional,
    char *layoutKeyOptional,
    int64_t topk,
    int64_t poolSize,
    int64_t maskMode,
    int64_t quantMode,
    bool returnValue,
    const aclTensor *sparseIndicesOut,
    const aclTensor *sparseValuesOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    OP_CHECK_NULL_WITH_MSG(query, ACLNN_ERR_PARAM_NULLPTR, "query is null");
    OP_CHECK_NULL_WITH_MSG(poolKey, ACLNN_ERR_PARAM_NULLPTR, "poolKey is null");
    OP_CHECK_NULL_WITH_MSG(weights, ACLNN_ERR_PARAM_NULLPTR, "weights is null");
    OP_CHECK_NULL_WITH_MSG(poolTailK, ACLNN_ERR_PARAM_NULLPTR, "poolTailK is null");
    OP_CHECK_NULL_WITH_MSG(sparseIndicesOut, ACLNN_ERR_PARAM_NULLPTR, "sparseIndicesOut is null");

    auto queryAclDataType = ToAclDataType(query->GetDataType());
    auto valuesHolder = TensorHolder(sparseValuesOut, ACL_FLOAT, std::string("sparseValuesOut"));

    auto executorPtr = op::MakeOpExecutor(*(op::OpExecutor::GetCurrentExecutor()));
    OP_CHECK_NULL_WITH_MSG(executorPtr, ACLNN_ERR_INNER, "make executor failed");

    auto opExecutor = op::OpExecutor::GetCurrentExecutor();
    auto opDesc = op::OpDesc("PoolKeyIndexer", "PoolKeyIndexer");

    auto queryTensor = opExecutor->ConvertToTensor(query, opDesc.GetInput(0U));
    auto poolKeyTensor = opExecutor->ConvertToTensor(poolKey, opDesc.GetInput(1U));
    auto weightsTensor = opExecutor->ConvertToTensor(weights, opDesc.GetInput(2U));
    auto poolTailKTensor = opExecutor->ConvertToTensor(poolTailK, opDesc.GetInput(3U));

    opDesc.AddInput(0U, "query", queryTensor);
    opDesc.AddInput(1U, "pool_key", poolKeyTensor);
    opDesc.AddInput(2U, "weights", weightsTensor);
    opDesc.AddInput(3U, "pool_tail_k", poolTailKTensor);

    if (actualSeqLengthsQueryOptional != nullptr) {
        auto t = opExecutor->ConvertToTensor(actualSeqLengthsQueryOptional, opDesc.GetInput(4U));
        opDesc.AddInput(4U, "actual_seq_q", t);
    }
    if (actualSeqLengthsKeyOptional != nullptr) {
        auto t = opExecutor->ConvertToTensor(actualSeqLengthsKeyOptional, opDesc.GetInput(5U));
        opDesc.AddInput(5U, "actual_seq_k", t);
    }
    if (blockTableOptional != nullptr) {
        auto t = opExecutor->ConvertToTensor(blockTableOptional, opDesc.GetInput(6U));
        opDesc.AddInput(6U, "block_table", t);
    }
    if (qDescaleOptional != nullptr) {
        auto t = opExecutor->ConvertToTensor(qDescaleOptional, opDesc.GetInput(7U));
        opDesc.AddInput(7U, "q_descale", t);
    }
    if (kDescaleOptional != nullptr) {
        auto t = opExecutor->ConvertToTensor(kDescaleOptional, opDesc.GetInput(8U));
        opDesc.AddInput(8U, "k_descale", t);
    }

    auto sparseIndicesTensor = opExecutor->ConvertToTensor(sparseIndicesOut, opDesc.GetOutput(0U));
    opDesc.AddOutput(0U, "sparse_indices", sparseIndicesTensor);

    if (returnValue && sparseValuesOut != nullptr) {
        auto sparseValuesTensor = opExecutor->ConvertToTensor(sparseValuesOut, opDesc.GetOutput(1U));
        opDesc.AddOutput(1U, "sparse_values", sparseValuesTensor);
    } else {
        auto sparseValuesTensor = opExecutor->ConvertToTensor(sparseValuesOut, opDesc.GetOutput(1U));
        opDesc.AddOutput(1U, "sparse_values", sparseValuesTensor);
    }

    opDesc.SetAttr("topk", topk);
    opDesc.SetAttr("pool_size", poolSize);
    opDesc.SetAttr("layout_q", layoutQueryOptional == nullptr ? std::string("BSND") : std::string(layoutQueryOptional));
    opDesc.SetAttr("layout_k", layoutKeyOptional == nullptr ? std::string("BSND") : std::string(layoutKeyOptional));
    opDesc.SetAttr("mask_mode", maskMode);
    opDesc.SetAttr("quant_mode", quantMode);
    opDesc.SetAttr("return_value", returnValue);

    *executor = executorPtr.release();

    if (workspaceSize != nullptr) {
        *workspaceSize = 0; // Phase 1: workspace computed in tiling
    }

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnPoolKeyIndexer(void *workspace, uint64_t workspaceSize,
                                 aclOpExecutor *executor, const aclrtStream stream)
{
    OP_CHECK_NULL_WITH_MSG(executor, ACLNN_ERR_PARAM_NULLPTR, "executor is null");
    auto opExecutor = op::OpExecutor::GetCurrentExecutor();
    opExecutor->ExecuteAsync(executor, stream, true);
    return ACLNN_SUCCESS;
}

} // namespace

#ifdef __cplusplus
}
#endif
