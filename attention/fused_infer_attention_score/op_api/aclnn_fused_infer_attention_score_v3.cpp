/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_fused_infer_attention_score_v3.h"
#include <cstring>
#include "opdev/tensor_view_utils.h"
#include "opdev/format_utils.h"
#include "opdev/common_types.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "opdev/op_def.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "fused_infer_attention_score_inner.h"
#include "aclnnInner_fused_infer_attention_score.h" // 该文件为自动生成，在build/autogen/inner路径下

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
extern "C" void UnInitPTACacheThreadLocal() __attribute__((weak));
// Select opdev's complete argument key for the validated dense GQA decode path.
// Executor ownership, cached launch data, and address updates remain in opdev.
bool IsDenseCacheInput(const aclTensor* tensor)
{
    if (tensor == nullptr || tensor->IsEmpty() || tensor->GetPlacement() != gert::kOnDeviceHbm ||
        tensor->GetStorageAddr() == nullptr || tensor->GetViewOffset() != 0 || tensor->GetStorageOffset() != 0 ||
        !IsContiguous(tensor)) {
        return false;
    }
    const auto format = tensor->GetViewFormat();
    return !IsPrivateFormat(format) && tensor->GetStorageFormat() == format && tensor->GetOriginalFormat() == format &&
           tensor->GetOriginalShape() == tensor->GetStorageShape() &&
           tensor->GetStorageShape().GetShapeSize() == tensor->GetViewShape().GetShapeSize();
}

bool CanUseDenseFiaExecutorCache(const aclTensor* query, const aclTensorList* key, const aclTensorList* value,
                                 const aclTensor* mask, const aclTensor* blockTable, const aclTensor* output,
                                 const char* layout, int64_t heads, int64_t kvHeads, int64_t blockSize,
                                 int64_t sparseMode, const aclIntArray* queryLengths, const aclIntArray* kvLengths)
{
    if (UnInitPTACacheThreadLocal == nullptr || op::GetCurrentPlatformInfo().GetCurNpuArch() != NpuArch::DAV_2201 ||
        layout == nullptr || std::strcmp(layout, "TND") != 0 || heads <= 0 || kvHeads != 1 || queryLengths == nullptr ||
        kvLengths == nullptr || blockSize != 128 || (sparseMode != 0 && sparseMode != 3) || key == nullptr ||
        value == nullptr || key->Size() != 1 || value->Size() != 1) {
        return false;
    }
    const aclTensor* inputs[] = {query, (*key)[0], (*value)[0], blockTable, output, mask};
    for (size_t i = 0; i < sizeof(inputs) / sizeof(inputs[0]); ++i) {
        if (i == 5 && inputs[i] == nullptr && sparseMode == 0) {
            continue;
        }
        if (!IsDenseCacheInput(inputs[i])) {
            return false;
        }
        // opdev records storage-to-argument relations: keep this cache domain alias-free.
        for (size_t j = 0; j < i; ++j) {
            if (inputs[i]->GetStorageAddr() == inputs[j]->GetStorageAddr()) {
                return false;
            }
        }
    }
    const auto dtype = query->GetDataType();
    if ((dtype != DataType::DT_FLOAT16 && dtype != DataType::DT_BF16) || (*key)[0]->GetDataType() != dtype ||
        (*value)[0]->GetDataType() != dtype || output->GetDataType() != dtype) {
        return false;
    }
    const auto& q = query->GetViewShape();
    const auto& k = (*key)[0]->GetViewShape();
    const auto& v = (*value)[0]->GetViewShape();
    if (q.GetDimNum() != 3 || q.GetDim(1) != heads || q.GetDim(2) != 128 || k.GetDimNum() != 3 ||
        k.GetDim(1) != blockSize || k.GetDim(2) != 128 || k != v || q != output->GetViewShape() ||
        blockTable->GetDataType() != DataType::DT_INT32 || blockTable->GetViewShape().GetDimNum() != 2 ||
        blockTable->GetViewShape().GetDim(0) != q.GetDim(0) ||
        queryLengths->Size() != static_cast<uint64_t>(q.GetDim(0)) || kvLengths->Size() != queryLengths->Size()) {
        return false;
    }
    for (uint64_t i = 0; i < queryLengths->Size(); ++i) {
        if ((*queryLengths)[i] != static_cast<int64_t>(i + 1)) {
            return false;
        }
    }
    return true;
}

/**
 * @brief for acl graph calculates the max workspace size based on the specific calculation process.
 * declaration here for testcase to use by extern the interface
 * @domain aclnn_ops_infer
 */
__attribute__((visibility("default"))) aclnnStatus aclnnFusedInferAttentionScoreV3GetMaxWorkspaceSize(
    const aclTensor* query, const aclTensorList* tensorListKey, const aclTensorList* tensorListValue,
    const aclTensor* pseShiftOptional, const aclTensor* attenMaskOptional, const aclIntArray* actualSeqLengthsOptional,
    const aclIntArray* actualSeqLengthsKvOptional, const aclTensor* deqScale1Optional,
    const aclTensor* quantScale1Optional, const aclTensor* deqScale2Optional, const aclTensor* quantScale2Optional,
    const aclTensor* quantOffset2Optional, const aclTensor* antiquantScaleOptional,
    const aclTensor* antiquantOffsetOptional, const aclTensor* blockTableOptional,
    const aclTensor* queryPaddingSizeOptional, const aclTensor* kvPaddingSizeOptional,
    const aclTensor* keyAntiquantScaleOptional, const aclTensor* keyAntiquantOffsetOptional,
    const aclTensor* valueAntiquantScaleOptional, const aclTensor* valueAntiquantOffsetOptional,
    const aclTensor* tensorKeySharedPrefixOptional, const aclTensor* tensorValueSharedPrefixOptional,
    const aclIntArray* actualSharedPrefixLenOptional, const aclTensor* queryRopeOptional,
    const aclTensor* keyRopeOptional, const aclTensor* keyRopeAntiquantScaleOptional, int64_t numHeads,
    double scaleValue, int64_t preTokens, int64_t nextTokens, char* inputLayout, int64_t numKeyValueHeads,
    int64_t sparseMode, int64_t innerPrecise, int64_t blockSize, int64_t antiquantMode, bool softmaxLseFlag,
    int64_t keyAntiquantMode, int64_t valueAntiquantMode, const aclTensor* attentionOut, const aclTensor* softmaxLse,
    uint64_t* workspaceSize, aclOpExecutor** executor);

struct FiaActualSeqInArrays {
    const aclIntArray* actualSeqLengthsOptional = nullptr;
    const aclIntArray* actualSeqLengthsKvOptional = nullptr;
    const aclIntArray* actualSharedPrefixLenOptional = nullptr;
};

struct FiaActualSeqOutTensors {
    aclTensor* actualSeqLengthsOptional = nullptr;
    aclTensor* actualSeqLengthsKvOptional = nullptr;
    aclTensor* actualSharedPrefixLenOptional = nullptr;
};

aclnnStatus FakeFiaActualSeqArrays(const FiaActualSeqInArrays& inArrays, FiaActualSeqOutTensors& outTensors)
{
    // nullptr不处理， nullptr是空指针，这样不会影响原来就不传入actual seq length为空的逻辑
    aclnnStatus ret = FakeArray(inArrays.actualSeqLengthsOptional, outTensors.actualSeqLengthsOptional);
    CHECK_RET_CODE(ret, "Try alloc fake actualSeqLengthsOptional failed");

    ret = FakeArray(inArrays.actualSeqLengthsKvOptional, outTensors.actualSeqLengthsKvOptional);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try alloc fake actualSeqLengthsKvOptional failed");
        aclDestroyTensor(outTensors.actualSeqLengthsOptional); // 没有返回值无需校验
        return ret;
    }

    ret = FakeArray(inArrays.actualSharedPrefixLenOptional, outTensors.actualSharedPrefixLenOptional);
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "Try alloc fake actualSharedPrefixLenOptional failed");
        aclDestroyTensor(outTensors.actualSeqLengthsOptional); // 没有返回值无需校验
        aclDestroyTensor(outTensors.actualSeqLengthsKvOptional);
        return ret;
    }
    return ret;
}

aclnnStatus aclnnFusedInferAttentionScoreV3GetMaxWorkspaceSize(
    const aclTensor* query, const aclTensorList* tensorListKey, const aclTensorList* tensorListValue,
    const aclTensor* pseShiftOptional, const aclTensor* attenMaskOptional, const aclIntArray* actualSeqLengthsOptional,
    const aclIntArray* actualSeqLengthsKvOptional, const aclTensor* deqScale1Optional,
    const aclTensor* quantScale1Optional, const aclTensor* deqScale2Optional, const aclTensor* quantScale2Optional,
    const aclTensor* quantOffset2Optional, const aclTensor* antiquantScaleOptional,
    const aclTensor* antiquantOffsetOptional, const aclTensor* blockTableOptional,
    const aclTensor* queryPaddingSizeOptional, const aclTensor* kvPaddingSizeOptional,
    const aclTensor* keyAntiquantScaleOptional, const aclTensor* keyAntiquantOffsetOptional,
    const aclTensor* valueAntiquantScaleOptional, const aclTensor* valueAntiquantOffsetOptional,
    const aclTensor* tensorKeySharedPrefixOptional, const aclTensor* tensorValueSharedPrefixOptional,
    const aclIntArray* actualSharedPrefixLenOptional, const aclTensor* queryRopeOptional,
    const aclTensor* keyRopeOptional, const aclTensor* keyRopeAntiquantScaleOptional, int64_t numHeads,
    double scaleValue, int64_t preTokens, int64_t nextTokens, char* inputLayout, int64_t numKeyValueHeads,
    int64_t sparseMode, int64_t innerPrecise, int64_t blockSize, int64_t antiquantMode, bool softmaxLseFlag,
    int64_t keyAntiquantMode, int64_t valueAntiquantMode, const aclTensor* attentionOut, const aclTensor* softmaxLse,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR,
                "Interface aclnnFusedInferAttentionScore versions V1 to V4 are no longer supported on Ascend950.");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    OP_LOGD("start aclnnFusedInferAttentionScoreV3GetMaxWorkspaceSize");
    TensorPreProcess(tensorListKey, tensorListValue);
    PrefixTensorPreProcess(tensorKeySharedPrefixOptional, tensorValueSharedPrefixOptional);

    FiaActualSeqInArrays actualSeqArrays{actualSeqLengthsOptional, actualSeqLengthsKvOptional,
                                         actualSharedPrefixLenOptional};
    FiaActualSeqOutTensors fakeActualSeqTensors{};
    aclnnStatus ret = FakeFiaActualSeqArrays(actualSeqArrays, fakeActualSeqTensors);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    const aclTensor* placeHolder = nullptr;
    const aclTensor* tempTensor = nullptr;
    ret = FusedInferAttentionScoreProcessSoftmaxLse(softmaxLseFlag, softmaxLse, tempTensor, placeHolder);
    if (ret != ACLNN_SUCCESS) {
        aclDestroyTensor(fakeActualSeqTensors.actualSeqLengthsOptional);
        aclDestroyTensor(fakeActualSeqTensors.actualSeqLengthsKvOptional);
        aclDestroyTensor(fakeActualSeqTensors.actualSharedPrefixLenOptional);
        return ret;
    }

    ret = aclnnInnerFusedInferAttentionScoreTensorGetWorkspaceSize(
        query, tensorListKey, tensorListValue, pseShiftOptional, attenMaskOptional,
        fakeActualSeqTensors.actualSeqLengthsOptional, fakeActualSeqTensors.actualSeqLengthsKvOptional,
        deqScale1Optional, quantScale1Optional, deqScale2Optional, quantScale2Optional, quantOffset2Optional,
        antiquantScaleOptional, antiquantOffsetOptional, blockTableOptional, queryPaddingSizeOptional,
        kvPaddingSizeOptional, keyAntiquantScaleOptional, keyAntiquantOffsetOptional, valueAntiquantScaleOptional,
        valueAntiquantOffsetOptional, tensorKeySharedPrefixOptional, tensorValueSharedPrefixOptional,
        fakeActualSeqTensors.actualSharedPrefixLenOptional, queryRopeOptional, keyRopeOptional,
        keyRopeAntiquantScaleOptional, nullptr, nullptr, nullptr, nullptr, numHeads, scaleValue, preTokens, nextTokens,
        inputLayout, numKeyValueHeads, sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag,
        keyAntiquantMode, valueAntiquantMode, 0, 0, 0, attentionOut, placeHolder, workspaceSize, executor);
    if (softmaxLseFlag == false) {
        aclDestroyTensor(tempTensor);
    }
    aclDestroyTensor(fakeActualSeqTensors.actualSeqLengthsOptional); // 只会成功，无需校验
    aclDestroyTensor(fakeActualSeqTensors.actualSeqLengthsKvOptional);
    aclDestroyTensor(fakeActualSeqTensors.actualSharedPrefixLenOptional);
    return ret;
}

aclnnStatus aclnnFusedInferAttentionScoreV3GetWorkspaceSize(
    const aclTensor* query, const aclTensorList* key, const aclTensorList* value, const aclTensor* pseShiftOptional,
    const aclTensor* attenMaskOptional, const aclIntArray* actualSeqLengthsOptional,
    const aclIntArray* actualSeqLengthsKvOptional, const aclTensor* deqScale1Optional,
    const aclTensor* quantScale1Optional, const aclTensor* deqScale2Optional, const aclTensor* quantScale2Optional,
    const aclTensor* quantOffset2Optional, const aclTensor* antiquantScaleOptional,
    const aclTensor* antiquantOffsetOptional, const aclTensor* blockTableOptional,
    const aclTensor* queryPaddingSizeOptional, const aclTensor* kvPaddingSizeOptional,
    const aclTensor* keyAntiquantScaleOptional, const aclTensor* keyAntiquantOffsetOptional,
    const aclTensor* valueAntiquantScaleOptional, const aclTensor* valueAntiquantOffsetOptional,
    const aclTensor* keySharedPrefixOptional, const aclTensor* valueSharedPrefixOptional,
    const aclIntArray* actualSharedPrefixLenOptional, const aclTensor* queryRopeOptional,
    const aclTensor* keyRopeOptional, const aclTensor* keyRopeAntiquantScaleOptional, int64_t numHeads,
    double scaleValue, int64_t preTokens, int64_t nextTokens, char* inputLayout, int64_t numKeyValueHeads,
    int64_t sparseMode, int64_t innerPrecise, int64_t blockSize, int64_t antiquantMode, bool softmaxLseFlag,
    int64_t keyAntiquantMode, int64_t valueAntiquantMode, const aclTensor* attentionOut, const aclTensor* softmaxLse,
    uint64_t* workspaceSize, aclOpExecutor** executor)
{
    const bool useLocalExecutorCache =
        executor != nullptr && workspaceSize != nullptr && !softmaxLseFlag && antiquantMode == 0 &&
        keyAntiquantMode == 0 && valueAntiquantMode == 0 && pseShiftOptional == nullptr &&
        deqScale1Optional == nullptr && quantScale1Optional == nullptr && deqScale2Optional == nullptr &&
        quantScale2Optional == nullptr && quantOffset2Optional == nullptr && antiquantScaleOptional == nullptr &&
        antiquantOffsetOptional == nullptr && queryPaddingSizeOptional == nullptr && kvPaddingSizeOptional == nullptr &&
        keyAntiquantScaleOptional == nullptr && keyAntiquantOffsetOptional == nullptr &&
        valueAntiquantScaleOptional == nullptr && valueAntiquantOffsetOptional == nullptr &&
        keySharedPrefixOptional == nullptr && valueSharedPrefixOptional == nullptr &&
        actualSharedPrefixLenOptional == nullptr && queryRopeOptional == nullptr && keyRopeOptional == nullptr &&
        keyRopeAntiquantScaleOptional == nullptr &&
        CanUseDenseFiaExecutorCache(query, key, value, attenMaskOptional, blockTableOptional, attentionOut, inputLayout,
                                    numHeads, numKeyValueHeads, blockSize, sparseMode, actualSeqLengthsOptional,
                                    actualSeqLengthsKvOptional);
    if (useLocalExecutorCache) {
        // A caller-supplied empty PTA key disables plan reuse. Let the following
        // L2 macro build its full key, including sequence values, strides and offsets.
        UnInitPTACacheThreadLocal();
    }

    L2_DFX_PHASE_1(
        aclnnFusedInferAttentionScoreV3,
        DFX_IN(query, key, value, pseShiftOptional, attenMaskOptional, actualSeqLengthsOptional,
               actualSeqLengthsKvOptional, deqScale1Optional, quantScale1Optional, deqScale2Optional,
               quantScale2Optional, quantOffset2Optional, antiquantScaleOptional, antiquantOffsetOptional,
               blockTableOptional, queryPaddingSizeOptional, kvPaddingSizeOptional, keyAntiquantScaleOptional,
               keyAntiquantOffsetOptional, valueAntiquantScaleOptional, valueAntiquantOffsetOptional,
               keySharedPrefixOptional, valueSharedPrefixOptional, actualSharedPrefixLenOptional, queryRopeOptional,
               keyRopeOptional, keyRopeAntiquantScaleOptional, numHeads, scaleValue, preTokens, nextTokens, inputLayout,
               numKeyValueHeads, sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag, keyAntiquantMode,
               valueAntiquantMode, useLocalExecutorCache),
        DFX_OUT(attentionOut, softmaxLse));

    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR,
                "Interface aclnnFusedInferAttentionScore versions V1 to V4 are no longer supported on Ascend950.");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    static bool isFirstCall = true;
    if (isFirstCall) {
        OP_LOGW("aclnnFusedInferAttentionScoreV3GetWorkspaceSize is scheduled to be deprecated in December 2026, "
                "and will be replaced by the aclnnFusedInferAttentionScoreV5GetWorkspaceSize. "
                "We apologize for any inconvenience caused and appreciate your timely migration to the new interface.");
        isFirstCall = false;
    }
    const aclTensorList* tensorListKey = key;
    const aclTensorList* tensorListValue = value;
    TensorPreProcess(tensorListKey, tensorListValue);

    const aclTensor* tensorKeySharedPrefixOptional = keySharedPrefixOptional;
    const aclTensor* tensorValueSharedPrefixOptional = valueSharedPrefixOptional;
    PrefixTensorPreProcess(tensorKeySharedPrefixOptional, tensorValueSharedPrefixOptional);

    const aclTensor* placeHolder = nullptr;
    const aclTensor* tempTensor = nullptr;
    aclnnStatus ret = FusedInferAttentionScoreProcessSoftmaxLse(softmaxLseFlag, softmaxLse, tempTensor, placeHolder);
    if (ret != ACLNN_SUCCESS) {
        return ret;
    }

    ret = InnerFusedInferAttentionScoreGetWorkspaceSize(
        query, tensorListKey, tensorListValue, pseShiftOptional, attenMaskOptional, actualSeqLengthsOptional,
        actualSeqLengthsKvOptional, deqScale1Optional, quantScale1Optional, deqScale2Optional, quantScale2Optional,
        quantOffset2Optional, antiquantScaleOptional, antiquantOffsetOptional, blockTableOptional,
        queryPaddingSizeOptional, kvPaddingSizeOptional, keyAntiquantScaleOptional, keyAntiquantOffsetOptional,
        valueAntiquantScaleOptional, valueAntiquantOffsetOptional, tensorKeySharedPrefixOptional,
        tensorValueSharedPrefixOptional, actualSharedPrefixLenOptional, queryRopeOptional, keyRopeOptional,
        keyRopeAntiquantScaleOptional, nullptr, nullptr, nullptr, nullptr, numHeads, scaleValue, preTokens, nextTokens,
        inputLayout, numKeyValueHeads, sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag,
        keyAntiquantMode, valueAntiquantMode, 0, 0, 0, attentionOut, placeHolder, workspaceSize, executor);
    if (softmaxLseFlag == false) {
        aclDestroyTensor(tempTensor);
    }
    return ret;
}

aclnnStatus aclnnFusedInferAttentionScoreV3(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                            const aclrtStream stream)
{
    if (op::GetCurrentPlatformInfo().GetCurNpuArch() == NpuArch::DAV_3510) {
        OP_LOGE(ACLNN_ERR_RUNTIME_ERROR,
                "Interface aclnnFusedInferAttentionScore versions V1 to V4 are no longer supported on Ascend950.");
        return ACLNN_ERR_RUNTIME_ERROR;
    }
    static bool isFirstCall = true;
    if (isFirstCall) {
        OP_LOGW("aclnnFusedInferAttentionScoreV3 is scheduled to be deprecated in December 2026, "
                "and will be replaced by the aclnnFusedInferAttentionScoreV5. "
                "We apologize for any inconvenience caused and appreciate your timely migration to the new interface.");
        isFirstCall = false;
    }
    return InnerFusedInferAttentionScore(workspace, workspaceSize, executor, stream);
}

} // namespace

#ifdef __cplusplus
}
#endif
