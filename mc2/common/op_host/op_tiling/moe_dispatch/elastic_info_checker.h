/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file elastic_info_checker.h
 * \brief MOE Distribute Dispatch ElasticInfo 可选特性校验
 *
 *   仅做互斥规则校验 + shape 精确值校验 + 产出 ElasticResult。
 *   dim/dtype/format 基础校验统一在阶段 E 通过 TensorCheckRule 处理。
 */

#ifndef MOE_DISPATCH_ELASTIC_INFO_CHECKER_H
#define MOE_DISPATCH_ELASTIC_INFO_CHECKER_H

#include "defs.h"
#include "utils.h"
#include "register/tilingdata_base.h"

namespace elastic_info_check {

// ========== ElasticResult 小结果结构 ==========
struct ElasticResult {
    bool hasElastic;
    uint32_t adjustedA;
    uint32_t adjustedExpertTokenDim0;
    uint32_t adjustedEpRecvDim0;
};

// ========== 互斥规则校验 ==========
// CheckCompatibility(nodeName, isLayered, hasTp)
inline ge::graphStatus CheckCompatibility(const char *nodeName, bool isLayered, bool hasTp)
{
    // elastic 与 layered 互斥
    if (isLayered) {
        OPS_LOG_E(nodeName, "Cannot support elasticInfo when comm_alg is hierarchy.");
        return ge::GRAPH_FAILED;
    }

    // elastic 与 tp>1 互斥
    if (hasTp) {
        OPS_LOG_E(nodeName, "Cannot support elasticInfo when tpWorldSize > 1.");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// ========== shape 校验 ==========
// dim0 = ELASTIC_INFO_OFFSET + RANK_LIST_NUM * epWorldSize = 4 + 2 * epWorldSize
inline ge::graphStatus CheckShape(gert::TilingContext *ctx, uint32_t inputIndex, uint32_t epWorldSize,
                                  const char *nodeName)
{
    const gert::StorageShape *shape = ctx->GetOptionalInputShape(inputIndex);
    if (shape == nullptr) {
        OPS_LOG_E(nodeName, "elasticInfo is null.");
        return ge::GRAPH_FAILED;
    }

    int64_t expectedDim0 =
        static_cast<int64_t>(moe_dispatch_defs::ELASTIC_INFO_OFFSET + moe_dispatch_defs::RANK_LIST_NUM * epWorldSize);

    if (shape->GetStorageShape().GetDimNum() != moe_dispatch_defs::ONE_DIM) {
        OPS_LOG_E(nodeName, "elasticInfo dim must be 1, but current dim num is %lu.",
                  shape->GetStorageShape().GetDimNum());
        return ge::GRAPH_FAILED;
    }

    if (shape->GetStorageShape().GetDim(0) != expectedDim0) {
        OPS_LOG_E(nodeName,
                  "elasticInfo's dim0 not equal to 4 + 2 * epWorldSize, "
                  "elasticInfo's dim0 is %ld, epWorldSize is %u.",
                  shape->GetStorageShape().GetDim(0), epWorldSize);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// ========== 产出 ElasticResult ==========
// CalcAdjustments(localExpertNum, sharedA, moeA, epWorldSize) → ElasticResult
//   sharedA: 共享专家路径的 A 值（maxBs * maxSharedGroupNum）
//   moeA   : moe 专家路径的 A 值（globalBs * min(localExpertNum, k)）
inline ElasticResult CalcAdjustments(uint32_t localExpertNum, uint32_t sharedA, uint32_t moeA, uint32_t epWorldSize)
{
    ElasticResult result{};
    result.hasElastic = true;
    // 弹性场景下 A 取两路最大值
    result.adjustedA = sharedA > moeA ? sharedA : moeA;
    // expertTokenNums dim0 = localExpertNum
    result.adjustedExpertTokenDim0 = localExpertNum;
    // epRecvCounts dim0 = epWorldSize * localExpertNum
    result.adjustedEpRecvDim0 = epWorldSize * localExpertNum;
    return result;
}

} // namespace elastic_info_check

#endif // MOE_DISPATCH_ELASTIC_INFO_CHECKER_H
