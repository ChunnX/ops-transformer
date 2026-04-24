/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_sparse_attention.cpp
 * \brief
 */


#include "block_sparse_attention_v2.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(BlockSparseAttentionV2);

static const aclTensor *ConvertIntArrayToTensor(const aclIntArray *intArray,
                                                aclOpExecutor *executor,
                                                DataType dtype)
{
    if (intArray != nullptr) {
        const aclTensor *tensor = executor->ConvertToTensor(intArray, dtype);
        auto mutableTensor = const_cast<aclTensor *>(tensor);
        mutableTensor->SetStorageFormat(Format::FORMAT_ND);
        mutableTensor->SetViewFormat(Format::FORMAT_ND);
        mutableTensor->SetOriginalFormat(Format::FORMAT_ND);
        return tensor;
    }
    return executor->AllocTensor(dtype, Format::FORMAT_ND, Format::FORMAT_ND);
}

const std::array<const aclTensor *, 2> BlockSparseAttentionV2(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *blockSparseMaskOptional,
    const aclTensor *attenMaskOptional,
    const aclIntArray *blockShapeOptional,
    const aclIntArray *actualSeqLengthsOptional,
    const aclIntArray *actualSeqLengthsKvOptional,
    const aclTensor *blockTableOptional,
    const aclTensor *queryDequantScaleOptional,
    const aclTensor *keyDequantScaleOptional,
    const aclTensor *valueDequantScaleOptional,
    const char *qInputLayout,
    const char *kvInputLayout,
    int64_t numKeyValueHeads,
    int64_t maskType,
    double scaleValue,
    int64_t innerPrecise,
    int64_t blockSize,
    int64_t preTokens,
    int64_t nextTokens,
    int64_t softmaxLseFlag,
    aclOpExecutor *executor)
{
    const char *safeKvInputLayout = (kvInputLayout != nullptr) ? kvInputLayout : qInputLayout;

    L0_DFX(BlockSparseAttentionV2, query, key, value, blockSparseMaskOptional, attenMaskOptional, blockShapeOptional,
           actualSeqLengthsOptional, actualSeqLengthsKvOptional, blockTableOptional, queryDequantScaleOptional,
           keyDequantScaleOptional, valueDequantScaleOptional, qInputLayout, safeKvInputLayout, numKeyValueHeads,
           maskType, scaleValue, innerPrecise, blockSize, preTokens, nextTokens, softmaxLseFlag);

    const aclTensor *blockShapeOptionalTensor = nullptr;
    if (blockShapeOptional) {
        blockShapeOptionalTensor = ConvertIntArrayToTensor(blockShapeOptional, executor, DataType::DT_INT64);
    }
    const aclTensor *actualSeqTensor = nullptr;
    if (actualSeqLengthsOptional) {
        actualSeqTensor = ConvertIntArrayToTensor(actualSeqLengthsOptional, executor, DataType::DT_INT64);
    }
    const aclTensor *actualSeqKvTensor = nullptr;
    if (actualSeqLengthsKvOptional) {
        actualSeqKvTensor = ConvertIntArrayToTensor(actualSeqLengthsKvOptional, executor, DataType::DT_INT64);
    }

    auto attentionOutTensor = executor->AllocTensor(query->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    auto softmaxLseTensor = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);

    // scaleValue is already float type, no need for cast
    auto ret = INFER_SHAPE(BlockSparseAttentionV2,
                           OP_INPUT(query, key, value, blockSparseMaskOptional, attenMaskOptional,
                                    blockShapeOptionalTensor, actualSeqTensor, actualSeqKvTensor, blockTableOptional,
                                    queryDequantScaleOptional, keyDequantScaleOptional, valueDequantScaleOptional),
                           OP_OUTPUT(attentionOutTensor, softmaxLseTensor),
                           OP_ATTR(qInputLayout, safeKvInputLayout, static_cast<int64_t>(numKeyValueHeads),
                                   static_cast<int64_t>(maskType), static_cast<float>(scaleValue),
                                   static_cast<int64_t>(innerPrecise), static_cast<int64_t>(blockSize),
                                   static_cast<uint32_t>(preTokens), static_cast<int64_t>(nextTokens),
                                   static_cast<int64_t>(softmaxLseFlag)));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "BlockSparseAttentionV2 infer shape failed, scaleValue: %f.", scaleValue);
        return {nullptr, nullptr};
    }
    
    ADD_TO_LAUNCHER_LIST_AICORE(BlockSparseAttentionV2,
                                OP_INPUT(query, key, value, blockSparseMaskOptional, attenMaskOptional,
                                         blockShapeOptionalTensor, actualSeqTensor, actualSeqKvTensor, blockTableOptional,
                                         queryDequantScaleOptional, keyDequantScaleOptional, valueDequantScaleOptional),
                                OP_OUTPUT(attentionOutTensor, softmaxLseTensor),
                                OP_ATTR(qInputLayout, safeKvInputLayout, static_cast<int64_t>(numKeyValueHeads),
                                        static_cast<int64_t>(maskType), static_cast<float>(scaleValue),
                                        static_cast<int64_t>(innerPrecise), static_cast<int64_t>(blockSize),
                                        static_cast<int64_t>(preTokens), static_cast<int64_t>(nextTokens),
                                        static_cast<int64_t>(softmaxLseFlag)));

    return {attentionOutTensor, softmaxLseTensor};
}
} // namespace l0op


