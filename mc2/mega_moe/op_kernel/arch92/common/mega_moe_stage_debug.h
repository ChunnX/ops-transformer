/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_ARCH92_STAGE_DEBUG_H
#define MEGA_MOE_ARCH92_STAGE_DEBUG_H

#include "kernel_operator.h"

namespace MegaMoeImpl {

using namespace AscendC;

namespace StageDebug {

constexpr unsigned long GMM1_GATE_OUTPUT_MARKER = 910002UL;
constexpr unsigned long GMM1_UP_OUTPUT_MARKER = 910003UL;
constexpr unsigned long GMM2_OUTPUT_MARKER = 910004UL;
constexpr unsigned long GMM2_DEPENDENCY_HASH_MARKER = 910010UL;
constexpr unsigned long GMM1_DEPENDENCY_HASH_MARKER = 910011UL;
constexpr unsigned long UINT_BIAS = 1000000UL;
constexpr uint32_t DATA_SAMPLE_COUNT = 20U;
constexpr uint32_t BF16_FORMAT = 0U;
constexpr uint32_t HASH_FORMAT = 7U;
constexpr uint32_t GMM1_HASH_OUTPUT_COLUMNS = 4U;
constexpr uint32_t GMM2_HASH_OUTPUT_COLUMNS = DATA_SAMPLE_COUNT;
constexpr uint32_t HASH_OFFSET_BASIS = 2166136261U;
constexpr uint32_t HASH_PRIME = 16777619U;

__aicore__ inline void LogHeader(unsigned long marker, uint32_t format, uint32_t rankIdx,
                                 uint32_t expertIdx, uint32_t rowIdx)
{
    set_cond(marker);
    set_cond(UINT_BIAS + static_cast<unsigned long>(format));
    set_cond(UINT_BIAS + static_cast<unsigned long>(AscendC::GetBlockIdx()));
    set_cond(UINT_BIAS + static_cast<unsigned long>(rankIdx));
    set_cond(UINT_BIAS + static_cast<unsigned long>(expertIdx));
    set_cond(UINT_BIAS + static_cast<unsigned long>(rowIdx));
}

// Hash complete dependency rows with one 32-bit scalar GM read per word.
// Every row width is 4-byte aligned, so each loop body performs one scalar GM read.
__aicore__ inline uint32_t HashWordRows(GM_ADDR address, uint32_t rowCount,
                                        uint32_t rowBytes, uint32_t rowStrideBytes)
{
    uint32_t hash = HASH_OFFSET_BASIS;
    uint32_t wordsPerRow = rowBytes / sizeof(uint32_t);
    for (uint32_t row = 0U; row < rowCount; ++row) {
        GM_ADDR rowAddress = address + static_cast<uint64_t>(row) * rowStrideBytes;
        __gm__ int32_t *words = reinterpret_cast<__gm__ int32_t *>(rowAddress);
        for (uint32_t word = 0U; word < wordsPerRow; ++word) {
            uint32_t value = static_cast<uint32_t>(AscendC::ReadGmByPassDCache(words + word));
            hash = (hash ^ value) * HASH_PRIME;
        }
    }
    return hash;
}

__aicore__ inline void LogGmm1DependencyHashes(
    GM_ADDR aAddress, GM_ADDR scaleAAddress, GM_ADDR gateBAddress, GM_ADDR gateScaleBAddress,
    GM_ADDR upBAddress, GM_ADDR upScaleBAddress, uint32_t k, uint32_t scaleK,
    uint32_t rankIdx, uint32_t expertIdx)
{
    LogHeader(GMM1_DEPENDENCY_HASH_MARKER, HASH_FORMAT, rankIdx, expertIdx, 0U);
    set_cond(static_cast<unsigned long>(HashWordRows(aAddress, 1U, k, k)));
    set_cond(static_cast<unsigned long>(HashWordRows(scaleAAddress, 1U, scaleK, scaleK)));
    set_cond(static_cast<unsigned long>(
        HashWordRows(gateBAddress, GMM1_HASH_OUTPUT_COLUMNS, k, k)));
    set_cond(static_cast<unsigned long>(
        HashWordRows(gateScaleBAddress, GMM1_HASH_OUTPUT_COLUMNS, scaleK, scaleK)));
    set_cond(static_cast<unsigned long>(
        HashWordRows(upBAddress, GMM1_HASH_OUTPUT_COLUMNS, k, k)));
    set_cond(static_cast<unsigned long>(
        HashWordRows(upScaleBAddress, GMM1_HASH_OUTPUT_COLUMNS, scaleK, scaleK)));
}

__aicore__ inline void LogGmm2DependencyHashes(
    GM_ADDR aAddress, GM_ADDR scaleAAddress, GM_ADDR bAddress, GM_ADDR scaleBAddress,
    uint32_t k, uint32_t scaleK, uint32_t rankIdx, uint32_t expertIdx)
{
    LogHeader(GMM2_DEPENDENCY_HASH_MARKER, HASH_FORMAT, rankIdx, expertIdx, 0U);
    set_cond(static_cast<unsigned long>(HashWordRows(aAddress, 1U, k, k)));
    set_cond(static_cast<unsigned long>(HashWordRows(scaleAAddress, 1U, scaleK, scaleK)));
    // DNExt/ScaleBDN store B physically as [N, K] and ScaleB as [N, scaleK].
    set_cond(static_cast<unsigned long>(
        HashWordRows(bAddress, GMM2_HASH_OUTPUT_COLUMNS, k, k)));
    set_cond(static_cast<unsigned long>(
        HashWordRows(scaleBAddress, GMM2_HASH_OUTPUT_COLUMNS, scaleK, scaleK)));
}

__aicore__ inline void LogRawBf16(GM_ADDR address, uint32_t elementOffset, uint32_t count)
{
    GM_ADDR elementAddress = address + static_cast<uint64_t>(elementOffset) * sizeof(uint16_t);
    __gm__ int32_t *words = reinterpret_cast<__gm__ int32_t *>(elementAddress);
    for (uint32_t elementIdx = 0U; elementIdx < count; elementIdx += 2U) {
        uint32_t packed = static_cast<uint32_t>(AscendC::ReadGmByPassDCache(words + elementIdx / 2U));
        set_cond(static_cast<unsigned long>(packed & 0xFFFFU));
        if (elementIdx + 1U < count) {
            set_cond(static_cast<unsigned long>((packed >> 16U) & 0xFFFFU));
        }
    }
}

__aicore__ inline void LogBf16(unsigned long marker, GM_ADDR dataAddr, uint32_t rankIdx,
                               uint32_t expertIdx, uint32_t rowIdx)
{
    LogHeader(marker, BF16_FORMAT, rankIdx, expertIdx, rowIdx);
    LogRawBf16(dataAddr, 0U, DATA_SAMPLE_COUNT);
}

} // namespace StageDebug
} // namespace MegaMoeImpl

#endif // MEGA_MOE_ARCH92_STAGE_DEBUG_H
