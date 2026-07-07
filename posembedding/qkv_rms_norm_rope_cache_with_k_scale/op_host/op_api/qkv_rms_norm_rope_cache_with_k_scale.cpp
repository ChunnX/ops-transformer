/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "qkv_rms_norm_rope_cache_with_k_scale.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"

namespace l0op {
OP_TYPE_REGISTER(QkvRmsNormRopeCacheWithKScale);

namespace {
constexpr const char *DEFAULT_QKV_LAYOUT = "TND";
constexpr const char *DEFAULT_Q_OUT_LAYOUT = "NTD";

const char *GetLayoutOrDefault(const char *layout, const char *defaultLayout)
{
    return layout == nullptr || layout[0] == '\0' ? defaultLayout : layout;
}
} // namespace

std::tuple<const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *>
QkvRmsNormRopeCacheWithKScale(const aclTensor *qkv, const aclTensor *qGamma, const aclTensor *kGamma,
                              const aclTensor *cosSin, const aclTensor *slotMapping, aclTensor *kCache,
                              aclTensor *vCache, aclTensor *kScaleCache, const aclTensor *queryStartLoc,
                              const aclTensor *seqLens, const aclTensor *rotationOptional,
                              const aclTensor *vScaleOptional, const aclIntArray *headNums, const char *layoutQkv,
                              const char *layoutQOut, float epsilon, aclTensor *qOut, aclTensor *qScale,
                              aclOpExecutor *executor)
{
    const char *layoutQkvAttr = GetLayoutOrDefault(layoutQkv, DEFAULT_QKV_LAYOUT);
    const char *layoutQOutAttr = GetLayoutOrDefault(layoutQOut, DEFAULT_Q_OUT_LAYOUT);
    L0_DFX(QkvRmsNormRopeCacheWithKScale, qkv, qGamma, kGamma, cosSin, slotMapping, kCache, vCache, kScaleCache,
           queryStartLoc, seqLens, rotationOptional, vScaleOptional, headNums, layoutQkvAttr, layoutQOutAttr, epsilon);

    auto ret = INFER_SHAPE(QkvRmsNormRopeCacheWithKScale,
                           OP_INPUT(qkv, qGamma, kGamma, cosSin, slotMapping, kCache, vCache, kScaleCache,
                                    queryStartLoc, seqLens, rotationOptional, vScaleOptional),
                           OP_OUTPUT(qOut, qScale, kCache, vCache, kScaleCache),
                           OP_ATTR(headNums, layoutQkvAttr, layoutQOutAttr, epsilon));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "InferShape failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr};
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(QkvRmsNormRopeCacheWithKScale,
                                      OP_INPUT(qkv, qGamma, kGamma, cosSin, slotMapping, kCache, vCache, kScaleCache,
                                               queryStartLoc, seqLens, rotationOptional, vScaleOptional),
                                      OP_OUTPUT(qOut, qScale, kCache, vCache, kScaleCache),
                                      OP_ATTR(headNums, layoutQkvAttr, layoutQOutAttr, epsilon));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr};
    }

    return {qOut, qScale, kCache, vCache, kScaleCache};
}

} // namespace l0op
