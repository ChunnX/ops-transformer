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
constexpr uint32_t AP_AIC_CORE_NUM = 32;
constexpr uint32_t AP_AIV_CORE_NUM = 64;
constexpr uint32_t AP_CV_RATIO = AP_AIV_CORE_NUM / AP_AIC_CORE_NUM;
constexpr uint32_t AP_METADATA_SIZE = 1024;
using AP_METADATA_T = int32_t;

// Per-core split result: bnStartIdx and gS1StartIdx arrays
// Size = AP_AIC_CORE_NUM + 1 (extra sentinel entry for end boundary)
constexpr uint32_t AP_SPLIT_ARRAY_LEN = AP_AIC_CORE_NUM + 1;

// Base metadata field count
constexpr uint32_t AP_BASE_METADATA_SIZE = 8;

// Base metadata field indices (within baseMetadata array)
constexpr uint32_t AP_USED_CORE_NUM_INDEX = 0;    // cube cores actually used (before CV_RATIO)
constexpr uint32_t AP_M_BASE_SIZE_INDEX = 1;       // sOuterSize (M dimension tile size)
constexpr uint32_t AP_S_INNER_SIZE_INDEX = 2;      // sInnerSize (KV dimension tile size)
constexpr uint32_t AP_S1_OUTER_SIZE_INDEX = 3;     // total sOuter blocks
constexpr uint32_t AP_S1_SIZE_INDEX = 4;           // max Q sequence length
constexpr uint32_t AP_S2_SIZE_INDEX = 5;           // max KV sequence length
constexpr uint32_t AP_ACTUAL_LEN_Q_DIM_INDEX = 6;  // actualSeqLengths Q dim count
constexpr uint32_t AP_ACTUAL_LEN_KV_DIM_INDEX = 7; // actualSeqLengths KV dim count

/**
 * Metadata layout (flat int32_t[1024]):
 *   [0..32]   bnStartIdx[33]    - per-core batch*head start index
 *   [33..65]  gS1StartIdx[33]   - per-core sOuter block start index
 *   [66..73]  baseMetadata[8]   - scalar parameters
 */
namespace detail {
struct APMetaData {
    int32_t bnStartIdx[AP_SPLIT_ARRAY_LEN];
    int32_t gS1StartIdx[AP_SPLIT_ARRAY_LEN];
    int32_t baseMetadata[AP_BASE_METADATA_SIZE];
};
} // namespace detail

static_assert(AP_METADATA_SIZE * sizeof(AP_METADATA_T) >= sizeof(detail::APMetaData),
    "APMetaData must fit within AP_METADATA_SIZE int32_t elements");

// Offset constants for flat access
constexpr uint32_t AP_BN_START_IDX_OFFSET = 0;
constexpr uint32_t AP_GS1_START_IDX_OFFSET = AP_SPLIT_ARRAY_LEN;
constexpr uint32_t AP_BASE_OFFSET = AP_SPLIT_ARRAY_LEN * 2;

#ifdef __CCE_AICORE__
__aicore__ inline uint32_t GetAPBnStartIdx(uint32_t coreIdx)
{
    return AP_BN_START_IDX_OFFSET + coreIdx;
}

__aicore__ inline uint32_t GetAPGS1StartIdx(uint32_t coreIdx)
{
    return AP_GS1_START_IDX_OFFSET + coreIdx;
}

__aicore__ inline uint32_t GetAPBaseMetaIndex(uint32_t fieldIdx)
{
    return AP_BASE_OFFSET + fieldIdx;
}
#endif

} // namespace optiling

#endif // ATTENTION_PIONEER_METADATA_H
