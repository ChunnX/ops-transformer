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
 * \file attention_pioneer_metadata.h
 * \brief MetaData constants and structure definitions for AttentionPioneerMetadata AICPU operator.
 */

#ifndef ATTENTION_PIONEER_METADATA_H
#define ATTENTION_PIONEER_METADATA_H

#include <cstdint>

namespace optiling {

// A5 platform core numbers
constexpr uint32_t AP_AIC_CORE_NUM = 36;
constexpr uint32_t AP_AIV_CORE_NUM = 72;
constexpr uint32_t AP_CV_RATIO = AP_AIV_CORE_NUM / AP_AIC_CORE_NUM;
constexpr uint32_t AP_METADATA_SIZE = 1024;
using AP_METADATA_T = int32_t;

// Per-core layout: each core has SLOTS_PER_CORE reserved positions
constexpr uint32_t AP_SLOTS_PER_CORE = 10;

// Per-core field indices (within each core's slot)
constexpr uint32_t AIC_CORE_ENABLE_INDEX = 0;   // 0/1 enable flag
constexpr uint32_t BN2_START_PTR_INDEX = 1;      // batch*head start index
constexpr uint32_t GS1_START_PTR_INDEX = 2;      // sOuter block start index
// indices 3..9 reserved

// Per-core section total size
constexpr uint32_t AP_PER_CORE_SECTION_SIZE = AP_AIC_CORE_NUM * AP_SLOTS_PER_CORE;

// Base metadata field count
constexpr uint32_t AP_BASE_METADATA_SIZE = 13;

// Base metadata offset in flat array (right after per-core section)
constexpr uint32_t AP_BASE_OFFSET = AP_PER_CORE_SECTION_SIZE;

// Base metadata field indices (within baseMetadata array)
constexpr uint32_t USED_CORE_NUM_INDEX = 0;                    // cube cores actually used (before CV_RATIO)
constexpr uint32_t M_BASE_SIZE_INDEX = 1;                       // sOuterSize (M dimension tile size)
constexpr uint32_t S_INNER_SIZE_INDEX = 2;                      // sInnerSize (KV dimension tile size)
constexpr uint32_t S1_OUTER_SIZE_INDEX = 3;                     // total sOuter blocks
constexpr uint32_t S1_SIZE_INDEX = 4;                           // max Q sequence length
constexpr uint32_t S2_SIZE_INDEX = 5;                           // max KV sequence length
constexpr uint32_t ACTUAL_LEN_Q_DIM_INDEX = 6;                 // actualSeqLengths Q dim count
constexpr uint32_t ACTUAL_LEN_KV_DIM_INDEX = 7;                // actualSeqLengths KV dim count
constexpr uint32_t MULTI_SMAX_S_INNER_LOOP_TIMES_INDEX = 8;    // s2 direction loop count
constexpr uint32_t ACTUAL_CORE_NUMS_INDEX = 9;                  // actual vector core num (= cube cores * CV_RATIO)
constexpr uint32_t TOTAL_SIZE_INDEX = 10;                       // total work size for multi-core split
constexpr uint32_t SPLIT_FACTOR_SIZE_INDEX = 11;                // work size per core
constexpr uint32_t SPLIT_FACTOR_TAIL_SIZE_INDEX = 12;           // tail core work size

/**
 * Metadata layout (flat int32_t[1024]):
 *   [0..359]   Per-core data: 36 cores x 10 slots each
 *              Per core (core_id * 10 + offset):
 *                +0: AIC_CORE_ENABLE  (0/1)
 *                +1: BN2_START_PTR    (batch*head start index)
 *                +2: GS1_START_PTR    (sOuter block start index)
 *                +3..9: reserved
 *   [360..372]  baseMetadata[13] - scalar parameters
 */
namespace detail {
struct APCoreSlot {
    int32_t fields[AP_SLOTS_PER_CORE];
};
struct APMetaData {
    APCoreSlot coreSlots[AP_AIC_CORE_NUM];
    int32_t baseMetadata[AP_BASE_METADATA_SIZE];
};
} // namespace detail

static_assert(AP_METADATA_SIZE * sizeof(AP_METADATA_T) >= sizeof(detail::APMetaData),
    "APMetaData must fit within AP_METADATA_SIZE int32_t elements");

// Flat access helper: get offset for a given core's field
inline constexpr uint32_t GetCoreFieldOffset(uint32_t coreIdx, uint32_t fieldIdx)
{
    return coreIdx * AP_SLOTS_PER_CORE + fieldIdx;
}

#ifdef __CCE_AICORE__
__aicore__ inline uint32_t GetAPCoreEnable(uint32_t coreIdx)
{
    return GetCoreFieldOffset(coreIdx, AIC_CORE_ENABLE_INDEX);
}

__aicore__ inline uint32_t GetAPBnStartIdx(uint32_t coreIdx)
{
    return GetCoreFieldOffset(coreIdx, BN2_START_PTR_INDEX);
}

__aicore__ inline uint32_t GetAPGS1StartIdx(uint32_t coreIdx)
{
    return GetCoreFieldOffset(coreIdx, GS1_START_PTR_INDEX);
}

__aicore__ inline uint32_t GetAPBaseMetaIndex(uint32_t fieldIdx)
{
    return AP_BASE_OFFSET + fieldIdx;
}
#endif

} // namespace optiling

#endif // ATTENTION_PIONEER_METADATA_H
