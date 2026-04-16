/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "attention_pioneer.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include <acl/acl.h> 
using namespace op;

namespace l0op {

OP_TYPE_REGISTER(AttentionPioneer);

const AttentionPioneerOutputs AttentionPioneer(
    const aclTensor *query,
    const aclTensorList *key,
    const aclTensorList *value,
    const aclTensor *pseShift,
    const aclTensor *attenMask,
    const aclIntArray *actualSeqLengths,
    const aclIntArray *actualSeqLengthsKv,
    const aclTensor *deqScale1,
    const aclTensor *quantScale1,
    const aclTensor *deqScale2,
    const aclTensor *quantScale2,
    const aclTensor *quantOffset2,
    const aclTensor *antiquantScale,
    const aclTensor *antiquantOffset,
    const aclTensor *blockTable,
    const aclTensor *queryPaddingSize,
    const aclTensor *kvPaddingSize,
    const aclTensor *keyAntiquantScale,
    const aclTensor *keyAntiquantOffset,
    const aclTensor *valueAntiquantScale,
    const aclTensor *valueAntiquantOffset,
    const aclTensor *keySharedPrefix,
    const aclTensor *valueSharedPrefix,
    const aclIntArray *actualSharedPrefixLen,
    const aclTensor *queryRope,
    const aclTensor *keyRope,
    const aclTensor *keyRopeAntiquantScale,
    const aclTensor *dequantScaleQuery,
    const aclTensor *keySink, 
    const aclTensor *keyRopeSink,
    const aclTensor *valueSink,
    int64_t numHeads,
    double scaleValue,
    int64_t preTokens,
    int64_t nextTokens,
    const char *inputLayout,
    int64_t numKeyValueHeads,
    int64_t sparseMode,
    int64_t innerPrecise,
    int64_t blockSize,
    int64_t antiquantMode,
    bool softmaxLseFlag,
    int64_t keyAntiquantMode,
    int64_t valueAntiquantMode,
    int64_t queryQuantMode,
    int64_t pseType, 
    int64_t outType,
    const aclTensor *attentionOut,
    const aclTensor *softmaxLse,
    aclOpExecutor *executor) {
    L0_DFX(AttentionPioneer, query, key, value, pseShift, attenMask, actualSeqLengths, actualSeqLengthsKv,
           deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2, antiquantScale, antiquantOffset,
           blockTable, queryPaddingSize, kvPaddingSize, keyAntiquantScale, keyAntiquantOffset,
           valueAntiquantScale, valueAntiquantOffset, keySharedPrefix, valueSharedPrefix, actualSharedPrefixLen,
           queryRope, keyRope, keyRopeAntiquantScale, dequantScaleQuery, 
           keySink, keyRopeSink, valueSink, numHeads, scaleValue, preTokens, nextTokens, inputLayout, numKeyValueHeads,
           sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag,
           keyAntiquantMode, valueAntiquantMode, queryQuantMode, pseType, outType);

    const aclTensor *actualSeqLengthsTensor = nullptr;
    const aclTensor *actualSeqLengthsKvTensor = nullptr;
    const aclTensor *actualSharedPrefixLenTensor = nullptr;
    if (executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FusedInferAttention: executor is nullptr.");
        return {nullptr, nullptr};
    }

    // 转换 aclIntArray 为 tensor
    if (actualSeqLengths != nullptr) {
        actualSeqLengthsTensor = executor->ConvertToTensor(actualSeqLengths, DataType::DT_INT64);
        const_cast<aclTensor *>(actualSeqLengthsTensor)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqLengthsTensor)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqLengthsTensor)->SetOriginalFormat(Format::FORMAT_ND);
    }

    if (actualSeqLengthsKv != nullptr) {
        actualSeqLengthsKvTensor = executor->ConvertToTensor(actualSeqLengthsKv, DataType::DT_INT64);
        const_cast<aclTensor *>(actualSeqLengthsKvTensor)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqLengthsKvTensor)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSeqLengthsKvTensor)->SetOriginalFormat(Format::FORMAT_ND);
    }

    if (actualSharedPrefixLen != nullptr) {
        actualSharedPrefixLenTensor = executor->ConvertToTensor(actualSharedPrefixLen, DataType::DT_INT64);
        const_cast<aclTensor *>(actualSharedPrefixLenTensor)->SetStorageFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSharedPrefixLenTensor)->SetViewFormat(Format::FORMAT_ND);
        const_cast<aclTensor *>(actualSharedPrefixLenTensor)->SetOriginalFormat(Format::FORMAT_ND);
    }

    // 分配输出 tensor
    auto attentionOutOut = executor->AllocTensor(query->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    auto softmaxLseOut = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    // 形状推导
    auto ret = INFER_SHAPE(AttentionPioneer,
        OP_INPUT(query, key, value, pseShift, attenMask, actualSeqLengthsTensor, actualSeqLengthsKvTensor,
                 deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2,
                 antiquantScale, antiquantOffset, blockTable, queryPaddingSize, kvPaddingSize,
                 keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset,
                 keySharedPrefix, valueSharedPrefix, actualSharedPrefixLenTensor,
                 queryRope, keyRope, keyRopeAntiquantScale, dequantScaleQuery,
                 keySink, keyRopeSink, valueSink),
        OP_OUTPUT(attentionOutOut, softmaxLseOut),
        OP_ATTR(numHeads, static_cast<float>(scaleValue), static_cast<int32_t>(preTokens),
                static_cast<int32_t>(nextTokens), inputLayout, numKeyValueHeads,
                static_cast<int32_t>(sparseMode), innerPrecise, blockSize,
                static_cast<int32_t>(antiquantMode), softmaxLseFlag,
                static_cast<int32_t>(keyAntiquantMode), static_cast<int32_t>(valueAntiquantMode),
                static_cast<int32_t>(queryQuantMode), 0, 0));
    const char* err_msg1 = aclGetRecentErrMsg();
    if (err_msg1 != nullptr) {
        std::cout << "INFER SHAPE 错误信息：" << err_msg1 << std::endl;
    }

    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FusedInferAttentionSink InferShape failed.");
        return {nullptr, nullptr};
    }
    // 添加到启动列表
    ret = ADD_TO_LAUNCHER_LIST_AICORE(AttentionPioneer,
        OP_INPUT(query, key, value, pseShift, attenMask, actualSeqLengthsTensor, actualSeqLengthsKvTensor,
                 deqScale1, quantScale1, deqScale2, quantScale2, quantOffset2,
                 antiquantScale, antiquantOffset, blockTable, queryPaddingSize, kvPaddingSize,
                 keyAntiquantScale, keyAntiquantOffset, valueAntiquantScale, valueAntiquantOffset,
                 keySharedPrefix, valueSharedPrefix, actualSharedPrefixLenTensor,
                 queryRope, keyRope, keyRopeAntiquantScale, dequantScaleQuery,
                 keySink, keyRopeSink, valueSink),
        OP_OUTPUT(attentionOutOut, softmaxLseOut),
        OP_ATTR(numHeads, static_cast<float>(scaleValue), static_cast<int32_t>(preTokens),
                static_cast<int32_t>(nextTokens), inputLayout, numKeyValueHeads,
                static_cast<int32_t>(sparseMode), innerPrecise, blockSize,
                static_cast<int32_t>(antiquantMode), softmaxLseFlag,
                static_cast<int32_t>(keyAntiquantMode), static_cast<int32_t>(valueAntiquantMode),
                static_cast<int32_t>(queryQuantMode), 0, 0));
    const char* err_msg = aclGetRecentErrMsg();
    if (err_msg != nullptr) {
        std::cout << "ACL 错误信息：" << err_msg << std::endl;
    }
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FusedInferAttention LaunchAicore failed.");
        return {nullptr, nullptr};
    }
    return {attentionOutOut, softmaxLseOut};
}

}