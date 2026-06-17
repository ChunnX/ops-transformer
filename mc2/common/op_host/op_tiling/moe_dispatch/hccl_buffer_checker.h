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
 * \file hccl_buffer_checker.h
 * \brief MOE Distribute Dispatch HCCL buffer 大小校验
 *
 *   - CalcMinBuffSize(params)              : 按 setup/teardown 通用公式计算最小 buffer 需求
 *   - CheckBuffSize(actualBuffSize, ...)   : 比较实际 buffer 大小与最小需求
 *   - CheckBuffSizeFromAttr(...)           : 从属性获取（V3/Mc2Context 场景）
 *
 * 注：v2 的 winSize 公式（layered/fullmesh/TP 多路）属算子层 dispatch_v2_win_size.h，
 *     不在公共层实现。
 */

#ifndef MOE_DISPATCH_HCCL_BUFFER_CHECKER_H
#define MOE_DISPATCH_HCCL_BUFFER_CHECKER_H

#include "defs.h"
#include "utils.h"

namespace hccl_buffer_check {

// ========== CalcMinBuffSize 入参 ==========
// 与 setup/teardown 现有公式一致：
//   align = ceilAlign(ceilAlign(2*h, 32) + HCCL_BUFFER_SIZE_HEAD, 512)
//   golden = MIN_AVAILABLE_BUFF_SIZE * (
//       localExpertNum * maxBs * epWorldSize * align +
//       (k + sharedExpertNum) * maxBs * ceilAlign(2*h, 512))
//
// 调用方填好以下字段即可（h、bs、k、epWorldSize、moe/shared 等）。
// HCCL_BUFFER_SIZE_HEAD 与 MIN_AVAILABLE_BUFF_SIZE 由调用方传入，
// 因为不同算子取值不同（teardown=2 / setup=2）。
struct MinBuffSizeParams {
    uint32_t h;
    uint32_t bs;
    uint32_t k;
    uint32_t epWorldSize;
    uint32_t globalBs;
    uint32_t localExpertNum;
    uint32_t sharedExpertNum;
    uint64_t hcclBufferSizeHead; // 头部对齐用，与算子约定一致
    uint64_t minAvailableFactor; // setup/teardown 一般为 2
};

// 计算最小 buffer 需求
inline uint64_t CalcMinBuffSize(const MinBuffSizeParams &p)
{
    if ((p.epWorldSize == 0U)) {
        return 0UL;
    }

    uint32_t maxBs = p.bs;
    if ((p.globalBs != 0U)) {
        maxBs = p.globalBs / p.epWorldSize;
    }

    uint64_t alignH32 = static_cast<uint64_t>(
        moe_dispatch_utils::AlignUp(2U * p.h, moe_dispatch_defs::ALIGN_32));
    uint64_t alignH512 = static_cast<uint64_t>(
        moe_dispatch_utils::AlignUp(2U * p.h, moe_dispatch_defs::ALIGN_512));

    uint64_t alignTotal = alignH32 + p.hcclBufferSizeHead;
    alignTotal = (alignTotal + moe_dispatch_defs::ALIGN_512 - 1UL) /
                 static_cast<uint64_t>(moe_dispatch_defs::ALIGN_512) *
                 static_cast<uint64_t>(moe_dispatch_defs::ALIGN_512);

    uint64_t expandPart = static_cast<uint64_t>(p.localExpertNum) * static_cast<uint64_t>(maxBs) *
                          static_cast<uint64_t>(p.epWorldSize) * alignTotal;
    uint64_t aggregatePart =
        static_cast<uint64_t>(p.k + p.sharedExpertNum) * static_cast<uint64_t>(maxBs) * alignH512;

    return p.minAvailableFactor * (expandPart + aggregatePart);
}

// ========== 从框架 HCCL group 获取实际 buffer 大小并比较 ==========
// 注：实际大小通过 mc2tiling::Mc2TilingUtils::GetMaxWindowSize() / 2 获取
//     （与 setup/teardown 现有实现一致），由调用方传入 actualBuffSize。
inline ge::graphStatus CheckBuffSize(uint64_t actualBuffSize, uint64_t minRequired, const char *nodeName)
{
    if ((actualBuffSize < minRequired)) {
        OPS_LOG_E(nodeName, "HCCL_BUFFSIZE [%lu] < required minimum [%lu].", actualBuffSize, minRequired);
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

// ========== 从属性获取（V3/Mc2Context 场景）==========
inline ge::graphStatus CheckBuffSizeFromAttr(uint64_t cclBufferSize, uint64_t minRequired, const char *nodeName)
{
    return CheckBuffSize(cclBufferSize, minRequired, nodeName);
}

} // namespace hccl_buffer_check

#endif // MOE_DISPATCH_HCCL_BUFFER_CHECKER_H
