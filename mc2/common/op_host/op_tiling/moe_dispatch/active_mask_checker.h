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
 * \file active_mask_checker.h
 * \brief MOE Distribute Dispatch ActiveMask 可选特性校验
 *
 * 设计原则（architecture_v0.md §5.4）：
 *   仅做互斥规则校验 + shape 精确值校验。
 *   dim/dtype/format 基础校验统一在阶段 E 通过 TensorCheckRule 处理。
 */

#ifndef MOE_DISPATCH_ACTIVE_MASK_CHECKER_H
#define MOE_DISPATCH_ACTIVE_MASK_CHECKER_H

#include "defs.h"
#include "utils.h"
#include "register/tilingdata_base.h"

namespace active_mask_check {

// ========== 互斥/约束规则校验 ==========
// 接口签名遵循架构文档 §5.4：CheckCompatibility(nodeName, isLayered, isExpertMask, globalBs)
// globalBs 用于 globalBs > bs * epWorldSize 时不允许 ActiveMask 的约束校验
// （bs * epWorldSize 由调用方根据 globalBs 推断；当为 0 时跳过该项校验）
inline ge::graphStatus CheckCompatibility(const char *nodeName, bool isLayered, bool isExpertMask, uint32_t globalBs,
                                          uint32_t bsTimesEpWorldSize)
{
    // 2D mask (isExpertMask=true) 与 layered 互斥
    if (isLayered && isExpertMask) {
        OPS_LOG_E(nodeName, "Cannot support 2D xActiveMask when comm_alg is hierarchy.");
        return ge::GRAPH_FAILED;
    }

    // globalBs 超过 bs * epWorldSize 时不允许 ActiveMask
    if (bsTimesEpWorldSize != 0U && globalBs > bsTimesEpWorldSize) {
        OPS_LOG_E(nodeName, "Cannot support xActiveMask when globalBs(%u) > bs * epWorldSize(%u).", globalBs,
                  bsTimesEpWorldSize);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// ========== shape 值校验 ==========
// 1D 时 dim0=bs；2D 时 dim0=bs && dim1=k
inline ge::graphStatus CheckShape(gert::TilingContext *ctx, uint32_t inputIndex, uint32_t bs, uint32_t k,
                                  const char *nodeName)
{
    const gert::StorageShape *shape = ctx->GetOptionalInputShape(inputIndex);
    if (shape == nullptr) {
        OPS_LOG_E(nodeName, "xActiveMask shape is null.");
        return ge::GRAPH_FAILED;
    }

    uint32_t dimNum = static_cast<uint32_t>(shape->GetStorageShape().GetDimNum());
    int64_t dim0 = shape->GetStorageShape().GetDim(0);
    if (dim0 != static_cast<int64_t>(bs)) {
        OPS_LOG_E(nodeName, "The input of xActiveMask dim0 = %ld is not equal to x dim0 = %u.", dim0, bs);
        return ge::GRAPH_FAILED;
    }

    if (dimNum == moe_dispatch_defs::TWO_DIMS) {
        int64_t dim1 = shape->GetStorageShape().GetDim(1);
        if (dim1 != static_cast<int64_t>(k)) {
            OPS_LOG_E(nodeName, "The input of xActiveMask dim1 = %ld is not equal to expertId dim1 = %u.", dim1, k);
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

} // namespace active_mask_check

#endif // MOE_DISPATCH_ACTIVE_MASK_CHECKER_H
