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
 * \file aclnn_und_gen_qkv_rms_norm_rope_cache.cpp
 * \brief
 */

#include "aclnn_und_gen_qkv_rms_norm_rope_cache.h"
#include "und_gen_qkv_rms_norm_rope_cache.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/shape_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_executor.h"
#include "opdev/make_op_executor.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace und_gen_qkv_rms_norm_rope_cache {

static inline bool CheckNotNull(
    const aclTensor* undQkv, const aclTensor* undWeightsQ, const aclTensor* undWeightsK, const aclTensor* cosSinCache,
    aclTensor* kCache, aclTensor* vCache, const aclTensor* slotMapping, const aclTensor* positions, aclTensor* q,
    aclTensor* kCacheOut, aclTensor* vCacheOut)
{
    OP_CHECK_NULL(undQkv, return false);
    OP_CHECK_NULL(undWeightsQ, return false);
    OP_CHECK_NULL(undWeightsK, return false);
    OP_CHECK_NULL(cosSinCache, return false);
    OP_CHECK_NULL(kCache, return false);
    OP_CHECK_NULL(vCache, return false);
    OP_CHECK_NULL(slotMapping, return false);
    OP_CHECK_NULL(positions, return false);
    OP_CHECK_NULL(q, return false);
    OP_CHECK_NULL(kCacheOut, return false);
    OP_CHECK_NULL(vCacheOut, return false);
    return true;
}

static inline bool CheckGenPaired(const aclTensor* genQkv, const aclTensor* genWeightsQ, const aclTensor* genWeightsK)
{
    if (genQkv == nullptr) {
        return true;
    }
    if (genWeightsQ == nullptr || genWeightsK == nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_NULLPTR, "genWeightsQ/genWeightsK are required when genQkv is provided.");
        return false;
    }
    return true;
}
} // namespace und_gen_qkv_rms_norm_rope_cache

aclnnStatus aclnnUndGenQkvRmsNormRopeCacheGetWorkspaceSize(
    const aclTensor* undQkv, const aclTensor* undWeightsQ, const aclTensor* undWeightsK, const aclTensor* cosSinCache,
    aclTensor* kCache, aclTensor* vCache, const aclTensor* slotMapping, const aclTensor* positions,
    const aclTensor* genQkv, const aclTensor* genWeightsQ, const aclTensor* genWeightsK, const aclTensor* catIndices,
    int64_t numHeadsQ, int64_t numHeadsK, int64_t numHeadsV, double normEps, const aclIntArray* mropeSection,
    aclTensor* q, aclTensor* kCacheOut, aclTensor* vCacheOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    OP_CHECK_COMM_INPUT(workspaceSize, executor);
    L2_DFX_PHASE_1(aclnnUndGenQkvRmsNormRopeCache,
                   DFX_IN(undQkv, undWeightsQ, undWeightsK, cosSinCache, kCache, vCache, slotMapping, positions, genQkv,
                          genWeightsQ, genWeightsK, catIndices, numHeadsQ, numHeadsK, numHeadsV, normEps, mropeSection),
                   DFX_OUT(q, kCacheOut, vCacheOut));

    // 参数检查：L2 只做空指针类校验，dtype/shape/属性取值的完整校验统一在 tiling 侧
    // （aclnn 单算子路径与图模式都会走 tiling，避免两处重复实现导致漂移）
    auto notNull = und_gen_qkv_rms_norm_rope_cache::CheckNotNull(undQkv, undWeightsQ, undWeightsK, cosSinCache, kCache,
                                                                 vCache, slotMapping, positions, q, kCacheOut,
                                                                 vCacheOut);
    CHECK_RET(notNull, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(und_gen_qkv_rms_norm_rope_cache::CheckGenPaired(genQkv, genWeightsQ, genWeightsK),
              ACLNN_ERR_PARAM_NULLPTR);

    // 创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 将输入转换成连续的tensor
    auto undQkvContiguous = l0op::Contiguous(undQkv, uniqueExecutor.get());
    CHECK_RET(undQkvContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto undWeightsQContiguous = l0op::Contiguous(undWeightsQ, uniqueExecutor.get());
    CHECK_RET(undWeightsQContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto undWeightsKContiguous = l0op::Contiguous(undWeightsK, uniqueExecutor.get());
    CHECK_RET(undWeightsKContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto cosSinCacheContiguous = l0op::Contiguous(cosSinCache, uniqueExecutor.get());
    CHECK_RET(cosSinCacheContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto slotMappingContiguous = l0op::Contiguous(slotMapping, uniqueExecutor.get());
    CHECK_RET(slotMappingContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto positionsContiguous = l0op::Contiguous(positions, uniqueExecutor.get());
    CHECK_RET(positionsContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 可选输入：为空时直接透传 nullptr
    const aclTensor* genQkvContiguous = nullptr;
    const aclTensor* genWeightsQContiguous = nullptr;
    const aclTensor* genWeightsKContiguous = nullptr;
    const aclTensor* catIndicesContiguous = nullptr;
    if (genQkv != nullptr) {
        genQkvContiguous = l0op::Contiguous(genQkv, uniqueExecutor.get());
        CHECK_RET(genQkvContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
        genWeightsQContiguous = l0op::Contiguous(genWeightsQ, uniqueExecutor.get());
        CHECK_RET(genWeightsQContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
        genWeightsKContiguous = l0op::Contiguous(genWeightsK, uniqueExecutor.get());
        CHECK_RET(genWeightsKContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    }
    if (catIndices != nullptr) {
        catIndicesContiguous = l0op::Contiguous(catIndices, uniqueExecutor.get());
        CHECK_RET(catIndicesContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    }

    // KV Cache 为原地更新的输入输出
    auto kCacheContiguous = l0op::Contiguous(kCache, uniqueExecutor.get());
    CHECK_RET(kCacheContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    kCache = const_cast<aclTensor*>(kCacheContiguous);
    auto vCacheContiguous = l0op::Contiguous(vCache, uniqueExecutor.get());
    CHECK_RET(vCacheContiguous != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    vCache = const_cast<aclTensor*>(vCacheContiguous);

    // 调用l0接口进行计算
    auto l0Ret = l0op::UndGenQkvRmsNormRopeCache(
        undQkvContiguous, undWeightsQContiguous, undWeightsKContiguous, cosSinCacheContiguous, kCache, vCache,
        slotMappingContiguous, positionsContiguous, genQkvContiguous, genWeightsQContiguous, genWeightsKContiguous,
        catIndicesContiguous, numHeadsQ, numHeadsK, numHeadsV, normEps, mropeSection, q, kCacheOut, vCacheOut,
        uniqueExecutor.get());
    CHECK_RET(l0Ret == ACLNN_SUCCESS, l0Ret);

    // 获取workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnUndGenQkvRmsNormRopeCache(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnUndGenQkvRmsNormRopeCache);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
