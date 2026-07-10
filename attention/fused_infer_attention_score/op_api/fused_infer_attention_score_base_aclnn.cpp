/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fused_infer_attention_score.h"

#include "opdev/make_op_executor.h"
#include "opdev/common_types.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"

#include <cstdio>
#include <string>

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(FusedInferAttentionScore);

namespace {
constexpr uint64_t DEBUG_TENSOR_LIST_LOG_LIMIT = 4;

template <typename... Args>
void FiaDebugPrintf(const char *format, Args... args)
{
    std::printf(format, args...);
    std::printf("\n");
}

void LogTensorDebugInfo(const char *name, const aclTensor *tensor)
{
    if (tensor == nullptr) {
        FiaDebugPrintf("[FIA_DEBUG][L0] %s is nullptr.", name);
        return;
    }
    FiaDebugPrintf("[FIA_DEBUG][L0] %s ptr[%p] dtype[%d] viewShape[%s] storageShape[%s] "
                   "viewOffset[%ld] storageFormat[%d].",
                   name, static_cast<const void *>(tensor), static_cast<int32_t>(tensor->GetDataType()),
                   op::ToString(tensor->GetViewShape()).GetString(),
                   op::ToString(tensor->GetStorageShape()).GetString(), tensor->GetViewOffset(),
                   static_cast<int32_t>(tensor->GetStorageFormat()));
}

void LogTensorListDebugInfo(const char *name, const aclTensorList *tensorList)
{
    if (tensorList == nullptr) {
        FiaDebugPrintf("[FIA_DEBUG][L0] %s tensorList is nullptr.", name);
        return;
    }
    FiaDebugPrintf("[FIA_DEBUG][L0] %s tensorList size[%lu].", name, tensorList->Size());
    uint64_t logSize = tensorList->Size() < DEBUG_TENSOR_LIST_LOG_LIMIT ?
        tensorList->Size() : DEBUG_TENSOR_LIST_LOG_LIMIT;
    for (uint64_t i = 0; i < logSize; ++i) {
        std::string itemName = std::string(name) + "[" + std::to_string(i) + "]";
        LogTensorDebugInfo(itemName.c_str(), (*tensorList)[i]);
    }
}

void LogIntArrayDebugInfo(const char *name, const aclIntArray *array)
{
    if (array == nullptr) {
        FiaDebugPrintf("[FIA_DEBUG][L0] %s aclIntArray is nullptr.", name);
        return;
    }
    FiaDebugPrintf("[FIA_DEBUG][L0] %s aclIntArray size[%lu].", name, array->Size());
}

const aclTensor *ConvertIntArrayToTensor(const aclIntArray *array, aclOpExecutor *executor)
{
    if (array == nullptr) {
        return nullptr;
    }

    auto tensor = executor->ConvertToTensor(array, DataType::DT_INT64);
    if (tensor == nullptr) {
        return nullptr;
    }
    const_cast<aclTensor *>(tensor)->SetStorageFormat(Format::FORMAT_ND);
    const_cast<aclTensor *>(tensor)->SetViewFormat(Format::FORMAT_ND);
    const_cast<aclTensor *>(tensor)->SetOriginalFormat(Format::FORMAT_ND);
    LogTensorDebugInfo("convertedIntArrayTensor", tensor);
    return tensor;
}
} // namespace

std::tuple<const aclTensor *, const aclTensor *> FusedInferAttentionScore(
    const aclTensor *query,
    const aclTensorList *key,
    const aclTensorList *value,
    const aclTensor *pseShiftOptional,
    const aclTensor *attenMaskOptional,
    const aclIntArray *actualSeqLengthsOptional,
    const aclIntArray *actualSeqLengthsKvOptional,
    const aclTensor *deqScale1Optional,
    const aclTensor *quantScale1Optional,
    const aclTensor *deqScale2Optional,
    const aclTensor *quantScale2Optional,
    const aclTensor *quantOffset2Optional,
    const aclTensor *antiquantScaleOptional,
    const aclTensor *antiquantOffsetOptional,
    const aclTensor *blockTableOptional,
    const aclTensor *queryPaddingSizeOptional,
    const aclTensor *kvPaddingSizeOptional,
    const aclTensor *keyAntiquantScaleOptional,
    const aclTensor *keyAntiquantOffsetOptional,
    const aclTensor *valueAntiquantScaleOptional,
    const aclTensor *valueAntiquantOffsetOptional,
    const aclTensor *keySharedPrefixOptional,
    const aclTensor *valueSharedPrefixOptional,
    const aclIntArray *actualSharedPrefixLenOptional,
    const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional,
    const aclTensor *keyRopeAntiquantScaleOptional,
    const aclTensor *dequantScaleQueryOptional,
    const aclTensor *learnableSinkOptional,
    const aclIntArray *qStartIdxOptional,
    const aclIntArray *kvStartIdxOptional,
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
    int64_t outDtype,
    const aclTensor *attentionOut,
    const aclTensor *softmaxLse,
    aclOpExecutor *executor)
{
    L0_DFX(FusedInferAttentionScore, query, key, value, pseShiftOptional, attenMaskOptional,
           actualSeqLengthsOptional, actualSeqLengthsKvOptional, deqScale1Optional, quantScale1Optional,
           deqScale2Optional, quantScale2Optional, quantOffset2Optional, antiquantScaleOptional,
           antiquantOffsetOptional, blockTableOptional, queryPaddingSizeOptional, kvPaddingSizeOptional,
           keyAntiquantScaleOptional, keyAntiquantOffsetOptional, valueAntiquantScaleOptional,
           valueAntiquantOffsetOptional, keySharedPrefixOptional, valueSharedPrefixOptional,
           actualSharedPrefixLenOptional, queryRopeOptional, keyRopeOptional, keyRopeAntiquantScaleOptional,
           dequantScaleQueryOptional, learnableSinkOptional, qStartIdxOptional, kvStartIdxOptional,
           numHeads, scaleValue, preTokens, nextTokens, inputLayout, numKeyValueHeads, sparseMode, innerPrecise,
           blockSize, antiquantMode, softmaxLseFlag, keyAntiquantMode, valueAntiquantMode, queryQuantMode, pseType,
           outDtype);

    if (executor == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FusedInferAttentionScore: executor is nullptr.");
        return {nullptr, nullptr};
    }
    if (attentionOut == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "FusedInferAttentionScore: attentionOut is nullptr.");
        return {nullptr, nullptr};
    }
    if (softmaxLse == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "FusedInferAttentionScore: softmaxLse is nullptr.");
        return {nullptr, nullptr};
    }

    FiaDebugPrintf("[FIA_DEBUG][L0] enter FusedInferAttentionScore attrs: numHeads[%ld] scale[%f] "
                   "preTokens[%ld] nextTokens[%ld] inputLayout[%s] numKeyValueHeads[%ld] sparseMode[%ld] "
                   "innerPrecise[%ld] blockSize[%ld] antiquantMode[%ld] softmaxLseFlag[%d] "
                   "keyAntiquantMode[%ld] valueAntiquantMode[%ld] queryQuantMode[%ld] pseType[%ld] outDtype[%ld].",
                   numHeads, scaleValue, preTokens, nextTokens, inputLayout == nullptr ? "nullptr" : inputLayout,
                   numKeyValueHeads, sparseMode, innerPrecise, blockSize, antiquantMode,
                   static_cast<int32_t>(softmaxLseFlag), keyAntiquantMode, valueAntiquantMode, queryQuantMode,
                   pseType, outDtype);
    LogTensorDebugInfo("query", query);
    LogTensorListDebugInfo("key", key);
    LogTensorListDebugInfo("value", value);
    LogTensorDebugInfo("pseShiftOptional", pseShiftOptional);
    LogTensorDebugInfo("attenMaskOptional", attenMaskOptional);
    LogTensorDebugInfo("blockTableOptional", blockTableOptional);
    LogTensorDebugInfo("attentionOut", attentionOut);
    LogTensorDebugInfo("softmaxLse", softmaxLse);
    LogIntArrayDebugInfo("actualSeqLengthsOptional", actualSeqLengthsOptional);
    LogIntArrayDebugInfo("actualSeqLengthsKvOptional", actualSeqLengthsKvOptional);
    LogIntArrayDebugInfo("actualSharedPrefixLenOptional", actualSharedPrefixLenOptional);
    LogIntArrayDebugInfo("qStartIdxOptional", qStartIdxOptional);
    LogIntArrayDebugInfo("kvStartIdxOptional", kvStartIdxOptional);

    const aclTensor *actualSeqLengthsTensor = ConvertIntArrayToTensor(actualSeqLengthsOptional, executor);
    const aclTensor *actualSeqLengthsKvTensor = ConvertIntArrayToTensor(actualSeqLengthsKvOptional, executor);
    const aclTensor *actualSharedPrefixLenTensor = ConvertIntArrayToTensor(actualSharedPrefixLenOptional, executor);
    const aclTensor *qStartIdxTensor = ConvertIntArrayToTensor(qStartIdxOptional, executor);
    const aclTensor *kvStartIdxTensor = ConvertIntArrayToTensor(kvStartIdxOptional, executor);
    if ((actualSeqLengthsOptional != nullptr && actualSeqLengthsTensor == nullptr) ||
        (actualSeqLengthsKvOptional != nullptr && actualSeqLengthsKvTensor == nullptr) ||
        (actualSharedPrefixLenOptional != nullptr && actualSharedPrefixLenTensor == nullptr) ||
        (qStartIdxOptional != nullptr && qStartIdxTensor == nullptr) ||
        (kvStartIdxOptional != nullptr && kvStartIdxTensor == nullptr)) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "FusedInferAttentionScore: convert IntArray to Tensor failed.");
        return {nullptr, nullptr};
    }

    auto attentionOutOut = executor->AllocTensor(attentionOut->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    if (attentionOutOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "FusedInferAttentionScore: alloc attentionOutOut failed.");
        return {nullptr, nullptr};
    }
    auto softmaxLseOut = executor->AllocTensor(softmaxLse->GetDataType(), Format::FORMAT_ND, Format::FORMAT_ND);
    if (softmaxLseOut == nullptr) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "FusedInferAttentionScore: alloc softmaxLseOut failed.");
        return {nullptr, nullptr};
    }
    LogTensorDebugInfo("attentionOutOut.beforeInfer", attentionOutOut);
    LogTensorDebugInfo("softmaxLseOut.beforeInfer", softmaxLseOut);

    FiaDebugPrintf("[FIA_DEBUG][L0] before INFER_SHAPE.");
    auto ret = INFER_SHAPE(FusedInferAttentionScore,
        OP_INPUT(query, key, value, pseShiftOptional, attenMaskOptional, actualSeqLengthsTensor,
                 actualSeqLengthsKvTensor, deqScale1Optional, quantScale1Optional, deqScale2Optional,
                 quantScale2Optional, quantOffset2Optional, antiquantScaleOptional, antiquantOffsetOptional,
                 blockTableOptional, queryPaddingSizeOptional, kvPaddingSizeOptional, keyAntiquantScaleOptional,
                 keyAntiquantOffsetOptional, valueAntiquantScaleOptional, valueAntiquantOffsetOptional,
                 keySharedPrefixOptional, valueSharedPrefixOptional, actualSharedPrefixLenTensor,
                 queryRopeOptional, keyRopeOptional, keyRopeAntiquantScaleOptional, dequantScaleQueryOptional,
                 learnableSinkOptional, qStartIdxTensor, kvStartIdxTensor),
        OP_OUTPUT(attentionOutOut, softmaxLseOut),
        OP_ATTR(numHeads, static_cast<float>(scaleValue), preTokens, nextTokens, inputLayout, numKeyValueHeads,
                sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag, keyAntiquantMode,
                valueAntiquantMode, queryQuantMode, pseType, outDtype));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FusedInferAttentionScore InferShape failed.");
        LogTensorDebugInfo("query.afterInferFailed", query);
        LogTensorListDebugInfo("key.afterInferFailed", key);
        LogTensorListDebugInfo("value.afterInferFailed", value);
        return {nullptr, nullptr};
    }
    FiaDebugPrintf("[FIA_DEBUG][L0] after INFER_SHAPE ret[%d].", static_cast<int32_t>(ret));
    LogTensorDebugInfo("attentionOutOut.afterInfer", attentionOutOut);
    LogTensorDebugInfo("softmaxLseOut.afterInfer", softmaxLseOut);

    FiaDebugPrintf("[FIA_DEBUG][L0] before ADD_TO_LAUNCHER_LIST_AICORE.");
    ret = ADD_TO_LAUNCHER_LIST_AICORE(FusedInferAttentionScore,
        OP_INPUT(query, key, value, pseShiftOptional, attenMaskOptional, actualSeqLengthsTensor,
                 actualSeqLengthsKvTensor, deqScale1Optional, quantScale1Optional, deqScale2Optional,
                 quantScale2Optional, quantOffset2Optional, antiquantScaleOptional, antiquantOffsetOptional,
                 blockTableOptional, queryPaddingSizeOptional, kvPaddingSizeOptional, keyAntiquantScaleOptional,
                 keyAntiquantOffsetOptional, valueAntiquantScaleOptional, valueAntiquantOffsetOptional,
                 keySharedPrefixOptional, valueSharedPrefixOptional, actualSharedPrefixLenTensor,
                 queryRopeOptional, keyRopeOptional, keyRopeAntiquantScaleOptional, dequantScaleQueryOptional,
                 learnableSinkOptional, qStartIdxTensor, kvStartIdxTensor),
        OP_OUTPUT(attentionOutOut, softmaxLseOut),
        OP_ATTR(numHeads, static_cast<float>(scaleValue), preTokens, nextTokens, inputLayout, numKeyValueHeads,
                sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag, keyAntiquantMode,
                valueAntiquantMode, queryQuantMode, pseType, outDtype));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "FusedInferAttentionScore LaunchAicore failed.");
        LogTensorDebugInfo("query.afterLaunchFailed", query);
        LogTensorListDebugInfo("key.afterLaunchFailed", key);
        LogTensorListDebugInfo("value.afterLaunchFailed", value);
        LogTensorDebugInfo("actualSeqLengthsTensor.afterLaunchFailed", actualSeqLengthsTensor);
        LogTensorDebugInfo("actualSeqLengthsKvTensor.afterLaunchFailed", actualSeqLengthsKvTensor);
        LogTensorDebugInfo("blockTableOptional.afterLaunchFailed", blockTableOptional);
        return {nullptr, nullptr};
    }
    FiaDebugPrintf("[FIA_DEBUG][L0] after ADD_TO_LAUNCHER_LIST_AICORE ret[%d].", static_cast<int32_t>(ret));

    return {attentionOutOut, softmaxLseOut};
}

} // namespace l0op
