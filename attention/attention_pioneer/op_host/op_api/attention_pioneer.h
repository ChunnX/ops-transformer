/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#ifndef OP_API_INC_LEVEL0_OP_FUSED_INFER_ATTENTION_OP_H_
#define OP_API_INC_LEVEL0_OP_FUSED_INFER_ATTENTION_OP_H_

#include "opdev/op_executor.h"

namespace l0op {
struct AttentionPioneerOutputs {
    const aclTensor *AttentionOutOut;
    const aclTensor *SoftmaxOutOut;
};

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
    aclOpExecutor *executor);
}
#endif