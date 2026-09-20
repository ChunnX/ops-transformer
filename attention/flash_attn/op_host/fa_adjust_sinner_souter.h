/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTN_FA_ADJUST_SINNER_SOUTER_H
#define FLASH_ATTN_FA_ADJUST_SINNER_SOUTER_H

#include <cstdint>
#include <initializer_list>
#include "../op_kernel/utils/dn_unmerged_policy.h"

namespace optiling {
namespace flash_attn {
namespace fa_tiling_util {

constexpr int64_t MAX_SEQ_LEN_DEFAULT = 2147483647;

// layout 枚举值，与 FiaLayout 一致，供外部算子使用
constexpr uint32_t LAYOUT_BSH = 0;
constexpr uint32_t LAYOUT_BSND = 1;
constexpr uint32_t LAYOUT_BNSD = 2;
constexpr uint32_t LAYOUT_TND = 4;

// tiling 切块常量
constexpr uint32_t SOUTER_32 = 32;
constexpr uint32_t SOUTER_64 = 64;
constexpr uint32_t SINNER_128 = 128;
constexpr uint32_t SINNER_256 = 256;
constexpr uint32_t DSIZE_128 = 128;
constexpr uint32_t DSIZE_256 = 256;
constexpr uint32_t CUBE_M_128 = 128;

/**
 * @brief 根据算子参数决定 sOuter / sInner 切块大小，纯函数，不依赖任何类。
 *
 * @param vHeadDim   V 的 head dim
 * @param gSize      GQA 的 group 数（n1/n2）
 * @param maxSeqQ    Q 的 max sequence length，-1 表示未知（按极大值处理）
 * @param maxSeqKv   KV 的 max sequence length，-1 表示未知（按极大值处理）
 * @param maskMode   mask 模式（0/2/4 等）
 * @param winLeft    左侧窗口
 * @param winRight   右侧窗口
 * @param qLayout    Q 的 layout（使用 LAYOUT_BSH / LAYOUT_BSND / LAYOUT_TND 等）
 * @param sOuterFactor [out] sOuter 切块大小
 * @param sInnerFactor [out] sInner 切块大小
 */
inline void AdjustSinnerAndSouter(uint32_t vHeadDim, uint32_t gSize, int64_t maxSeqQ, int64_t maxSeqKv,
                                  int32_t maskMode, int64_t winLeft, int64_t winRight, uint32_t qLayout,
                                  uint32_t &sOuterFactor, uint32_t &sInnerFactor)
{
    // NO_MASK and CAUSAL ignore user windows. Normalize here so host and metadata
    // choose the same basic block, including short-Q fallback to Cube M=64.
    if (maskMode == 0 || maskMode == 3) {
        winLeft = MAX_SEQ_LEN_DEFAULT;
        winRight = MAX_SEQ_LEN_DEFAULT;
    }
    if (winLeft == -1) {
        winLeft = MAX_SEQ_LEN_DEFAULT;
    }
    if (winRight == -1) {
        winRight = MAX_SEQ_LEN_DEFAULT;
    }
    if (maxSeqQ == -1) {
        maxSeqQ = MAX_SEQ_LEN_DEFAULT;
    }
    if (maxSeqKv == -1) {
        maxSeqKv = MAX_SEQ_LEN_DEFAULT;
    }
    sOuterFactor = SOUTER_64;
    sInnerFactor = SINNER_128;

    bool checkQueryAndValueS = maxSeqQ <= SOUTER_64 && maxSeqKv > SINNER_128;

    if (vHeadDim <= DSIZE_128) {
        int64_t winLeftTmp = winLeft;
        int64_t winRightTmp = winRight;
        if (maskMode == 0) {
            winLeftTmp = (winLeftTmp > 0) ? 0 : winLeftTmp;
        } else if (maskMode == 4) {
            winRightTmp = (winRightTmp > 0) ? 0 : winRightTmp;
        }
        bool checkSparseMode = (maskMode != 2 && winLeftTmp + winRightTmp > 128);
        if (checkQueryAndValueS && checkSparseMode) {
            sOuterFactor = SOUTER_32;
            sInnerFactor = SINNER_256;
        }
    }
    if (vHeadDim == DSIZE_256) {
        if (gSize * maxSeqQ < SOUTER_64) {
            sOuterFactor = SOUTER_32;
            sInnerFactor = SINNER_256;
        } else {
            sOuterFactor = SOUTER_64;
            sInnerFactor = SINNER_128;
        }
    }
}

// Both tiling and metadata must use the same static bounds, without reading sequence tensors.
// Count dense (B, N2, ceil(Q * G / M), ceil(KV / N)) blocks to retain split-KV parallelism.
// Unknown/empty dimensions keep the platform limit, including the existing empty-output path.
inline uint32_t GetMaxUsedAicCores(uint32_t aicNum, uint32_t batchSize, uint32_t kvHeads, uint32_t groupSize,
                                   int64_t maxSeqQ, int64_t maxSeqKv, uint32_t mBaseSize, uint32_t s2BaseSize,
                                   bool dnUnmerged = false)
{
    if (aicNum == 0 || batchSize == 0 || kvHeads == 0 || groupSize == 0 || maxSeqQ <= 0 || maxSeqKv <= 0 ||
        mBaseSize == 0 || s2BaseSize == 0) {
        return aicNum;
    }
    // BN1_S1 keeps G outside the query ceil; legacy BN2_S1G merges Q and G.
    const uint32_t mergedGroup = dnUnmerged ? 1U : groupSize;
    // Cap before multiplying Q by G, so even very large static bounds cannot overflow.
    if (static_cast<uint64_t>(maxSeqQ) > static_cast<uint64_t>(aicNum) * mBaseSize / mergedGroup) {
        return aicNum;
    }
    const uint64_t mSize = static_cast<uint64_t>(maxSeqQ) * mergedGroup;
    const uint64_t mBlocks = (mSize - 1) / mBaseSize + 1;
    const uint64_t s2Blocks = (static_cast<uint64_t>(maxSeqKv) - 1) / s2BaseSize + 1;
    uint32_t cores = 1;
    for (uint64_t factor : {static_cast<uint64_t>(batchSize), static_cast<uint64_t>(kvHeads),
                            static_cast<uint64_t>(dnUnmerged ? groupSize : 1U), mBlocks, s2Blocks}) {
        if (factor >= (static_cast<uint64_t>(aicNum) - 1) / cores + 1) {
            return aicNum;
        }
        cores *= static_cast<uint32_t>(factor);
    }
    return cores;
}

constexpr uint32_t FA_TEMPLATE_ND = 0;
constexpr uint32_t FA_TEMPLATE_DN = 1;
constexpr uint32_t FA_TEMPLATE_DN_UNMERGED = 2; // 不合轴 DN(templateId=2), 预埋路由入口默认关闭

// Host and metadata share this gate. Migration is complete; keep disabled until
// masked prefill performance satisfies the per-case enablement gate.
constexpr bool DN_UNMERGED_ROUTE_ENABLED = false;

// Use the original Cube M, not per-AIV rows or actual tail rows. Eligible shapes
// serve multimodal attention and LLM prefill; M=64 and D=256 retain legacy routing.
inline uint32_t GetFlashAttnTemplateId(uint32_t qkHeadDim, uint32_t vHeadDim, uint32_t cubeM, bool hasAttenMask,
                                       bool enableDnUnmerged = DN_UNMERGED_ROUTE_ENABLED)
{
    const bool supportedEqualDim = (qkHeadDim == vHeadDim) && (qkHeadDim == 64 || qkHeadDim == 72 || qkHeadDim == 128);
    const bool supportedMlaDim = (qkHeadDim == 192) && (vHeadDim == 128);
    const bool supported = cubeM == CUBE_M_128 && (supportedEqualDim || supportedMlaDim);
    if (supported && enableDnUnmerged) {
        return FA_TEMPLATE_DN_UNMERGED;
    }
    return (supported && !hasAttenMask) ? FA_TEMPLATE_DN : FA_TEMPLATE_ND;
}

} // namespace fa_tiling_util
} // namespace flash_attn
} // namespace optiling

#endif // FLASH_ATTN_FA_ADJUST_SINNER_SOUTER_H
