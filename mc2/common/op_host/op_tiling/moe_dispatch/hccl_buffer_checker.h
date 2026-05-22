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
 * \file hccl_buffer_checker.h
 * \brief MOE Distribute Dispatch HCCL buffer 大小校验
 */

#ifndef MOE_DISPATCH_HCCL_BUFFER_CHECKER_H
#define MOE_DISPATCH_HCCL_BUFFER_CHECKER_H

#include "defs.h"
#include "utils.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace hccl_buffer_check {

// ========== 计算最小 buffer 需求 ==========

inline uint64_t CalcMinBuffSize(
    uint32_t localExpertNum, uint32_t maxBs, uint32_t epWorldSize,
    uint32_t k, uint32_t sharedExpertNum, uint32_t h, uint32_t quantMode) {
    // 基础 buffer 计算公式（由具体算子决定）
    // 这里提供通用计算框架，具体公式在算子主流程中实现

    uint64_t baseSize = static_cast<uint64_t>(maxBs) * static_cast<uint64_t>(k);
    baseSize *= static_cast<uint64_t>(localExpertNum);
    baseSize *= static_cast<uint64_t>(epWorldSize);
    baseSize *= static_cast<uint64_t>(h);

    // 根据 quantMode 调整
    // quantMode 0: h * dtype_size
    // quantMode 1-4: 需要考虑 scales 和 dynamicScales

    return baseSize;
}

// ========== 从 HCCL group 获取实际大小 ==========

inline ge::graphStatus CheckBuffSizeFromContext(
    gert::TilingContext* ctx, uint32_t groupIndex, uint64_t minRequired, const char* nodeName) {

    if (ctx == nullptr) {
        OPS_LOG_E(nodeName, "TilingContext is null when checking HCCL buffer size.");
        return ge::GRAPH_FAILED;
    }

    // 获取 HCCL group 中的 buffer 大小
    // 具体实现依赖框架 API
    uint64_t actualSize = 0UL;

    // 通过 HcclTopoInfo 获取 buffer 大小
    // actualSize = ctx->GetHcclBuffSize(groupIndex);

    if (actualSize < minRequired) {
        OPS_LOG_E(nodeName, "HCCL buffer size %lu is less than required minimum %lu.",
            actualSize, minRequired);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// ========== 从属性获取 buffer 大小 ==========

inline ge::graphStatus CheckBuffSizeFromAttr(
    uint64_t cclBufferSize, uint64_t minRequired, const char* nodeName) {

    if (cclBufferSize < minRequired) {
        OPS_LOG_E(nodeName, "CCL buffer size %lu is less than required minimum %lu.",
            cclBufferSize, minRequired);
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

// ========== Win Size 校验（dispatch_v2 特有）==========

inline uint64_t CalcWinSize(
    uint64_t hcclBufferSize, uint32_t aivNum, bool isCcu) {
    // MTE 模式: winSize = hcclBufferSize - 1MB (状态区)
    // CCU 模式: winSize = hcclBufferSize * 2

    if (isCcu) {
        return hcclBufferSize * 2UL;
    }

    uint64_t stateZoneSize = 1024UL * 1024UL;  // 1MB
    if (hcclBufferSize <= stateZoneSize) {
        return 0UL;
    }

    return hcclBufferSize - stateZoneSize;
}

// ========== 工作空间大小计算 ==========

inline uint64_t CalcWorkspaceSize(
    uint32_t aivNum, uint32_t quantMode, uint32_t sharedExpertNum) {
    // 基础工作空间: 16MB
    uint64_t baseWorkspace = moe_dispatch_defs::SYSTEM_NEED_WORKSPACE;

    // 额外工作空间: 512 * aivNum^2
    uint64_t extraWorkspace = static_cast<uint64_t>(aivNum) * static_cast<uint64_t>(aivNum);
    extraWorkspace *= static_cast<uint64_t>(moe_dispatch_defs::SYSTEM_NEED_WORKSPACE) / 32UL;

    return baseWorkspace + extraWorkspace;
}

} // namespace hccl_buffer_check

#endif // MOE_DISPATCH_HCCL_BUFFER_CHECKER_H