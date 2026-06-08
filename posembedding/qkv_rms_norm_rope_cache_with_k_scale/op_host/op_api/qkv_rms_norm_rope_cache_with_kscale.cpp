/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "qkv_rms_norm_rope_cache_with_kscale.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(QkvRmsNormRopeCacheWithKScale);

std::tuple<const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*>
QkvRmsNormRopeCacheWithKScale(const aclTensor *qkv, const aclTensor *qGamma, const aclTensor *kGamma,
                              const aclTensor *cosSin, const aclTensor *slotMapping, aclTensor *kCache,
                              aclTensor *vCache, aclTensor *kScaleCache, const aclTensor *cuSeqLen,
                              const aclTensor *optionalQRotation, const aclTensor *optionalKRotation,
                              const aclTensor *optionalVScale, const aclIntArray *headNums, float epsilon,
                              aclTensor *qOut, aclTensor *qScale, aclOpExecutor *executor)
{
    L0_DFX(QkvRmsNormRopeCacheWithKScale, qkv, qGamma, kGamma, cosSin, slotMapping, kCache, vCache, kScaleCache,
           cuSeqLen, optionalQRotation, optionalKRotation, optionalVScale, headNums, epsilon);

    if (optionalQRotation == nullptr) {
        optionalQRotation = executor->AllocTensor(DataType::DT_BF16, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (optionalKRotation == nullptr) {
        optionalKRotation = executor->AllocTensor(DataType::DT_BF16, Format::FORMAT_ND, Format::FORMAT_ND);
    }
    if (optionalVScale == nullptr) {
        optionalVScale = executor->AllocTensor(DataType::DT_FLOAT, Format::FORMAT_ND, Format::FORMAT_ND);
    }

    auto ret = INFER_SHAPE(QkvRmsNormRopeCacheWithKScale,
                           OP_INPUT(qkv, qGamma, kGamma, cosSin, slotMapping, kCache, vCache, kScaleCache,
                                    cuSeqLen, optionalQRotation, optionalKRotation, optionalVScale),
                           OP_OUTPUT(qOut, qScale, kCache, vCache, kScaleCache),
                           OP_ATTR(headNums, epsilon));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr};
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        QkvRmsNormRopeCacheWithKScale,
        OP_INPUT(qkv, qGamma, kGamma, cosSin, slotMapping, kCache, vCache, kScaleCache, cuSeqLen, optionalQRotation,
                 optionalKRotation, optionalVScale),
        OP_OUTPUT(qOut, qScale, kCache, vCache, kScaleCache),
        OP_ATTR(headNums, epsilon));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr};
    }

    return {qOut, qScale, kCache, vCache, kScaleCache};
}

} // namespace l0op
