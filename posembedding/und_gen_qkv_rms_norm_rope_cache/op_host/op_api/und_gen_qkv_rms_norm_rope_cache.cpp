/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "und_gen_qkv_rms_norm_rope_cache.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "aclnn_kernels/common/op_error_check.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(UndGenQkvRmsNormRopeCache);

aclnnStatus UndGenQkvRmsNormRopeCache(
    const aclTensor* undQkv, const aclTensor* undWeightsQ, const aclTensor* undWeightsK,
    const aclTensor* cosSinCache, aclTensor* kCache, aclTensor* vCache, const aclTensor* slotMapping,
    const aclTensor* positions, const aclTensor* genQkv, const aclTensor* genWeightsQ, const aclTensor* genWeightsK,
    const aclTensor* catIndices, int64_t numHeadsQ, int64_t numHeadsK, int64_t numHeadsV, float normEps,
    const aclIntArray* mropeSection, aclTensor* q, aclTensor* kCacheOut, aclTensor* vCacheOut,
    aclOpExecutor* executor)
{
    L0_DFX(UndGenQkvRmsNormRopeCache, undQkv, undWeightsQ, undWeightsK, cosSinCache, kCache, vCache, slotMapping,
           positions, genQkv, genWeightsQ, genWeightsK, catIndices, numHeadsQ, numHeadsK, numHeadsV, normEps,
           mropeSection);

    auto ret = INFER_SHAPE(
        UndGenQkvRmsNormRopeCache,
        OP_INPUT(undQkv, undWeightsQ, undWeightsK, cosSinCache, kCache, vCache, slotMapping, positions, genQkv,
                 genWeightsQ, genWeightsK, catIndices),
        OP_OUTPUT(q, kCacheOut, vCacheOut),
        OP_ATTR(numHeadsQ, numHeadsK, numHeadsV, normEps, mropeSection));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "UndGenQkvRmsNormRopeCache InferShape failed.");
        return ret;
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(
        UndGenQkvRmsNormRopeCache,
        OP_INPUT(undQkv, undWeightsQ, undWeightsK, cosSinCache, kCache, vCache, slotMapping, positions, genQkv,
                 genWeightsQ, genWeightsK, catIndices),
        OP_OUTPUT(q, kCacheOut, vCacheOut),
        OP_ATTR(numHeadsQ, numHeadsK, numHeadsV, normEps, mropeSection));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "UndGenQkvRmsNormRopeCache ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return ret;
    }

    return ACLNN_SUCCESS;
}
} // namespace l0op
