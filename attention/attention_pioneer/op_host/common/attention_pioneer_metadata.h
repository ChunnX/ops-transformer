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

// A5 platform (V100) core numbers
constexpr uint32_t AP_AIC_CORE_NUM = 32;
constexpr uint32_t AP_AIV_CORE_NUM = 64;
constexpr uint32_t AP_METADATA_SIZE = 1024;
using AP_METADATA_T = int32_t;

constexpr uint32_t AP_AIC_METADATA_SIZE = 10;
constexpr uint32_t AP_AIV_METADATA_SIZE = 3;
constexpr uint32_t AP_BASE_METADATA_SIZE = 10;

// AIC Metadata Index Definitions
constexpr uint32_t AP_AIC_CORE_ENABLE_INDEX = 0;
constexpr uint32_t AP_BN2_END_PTR_INDEX = 1;
constexpr uint32_t AP_GS1_END_PTR_INDEX = 2;
constexpr uint32_t AP_S2_END_PTR_INDEX = 3;
constexpr uint32_t AP_BN2_IDX_OF_FD_HEAD_INDEX = 4;
constexpr uint32_t AP_GS1_IDX_OF_FD_HEAD_INDEX = 5;
constexpr uint32_t AP_S2_SPLIT_NUM_OF_FD_HEAD_INDEX = 6;
constexpr uint32_t AP_S2_SPLIT_START_IDX_OF_CORE_INDEX = 7;
constexpr uint32_t AP_GS1_SPLIT_NUM_OF_FD_HEAD_INDEX = 8;
constexpr uint32_t AP_GS1_LAST_PART_SIZE_OF_FD_HEAD_INDEX = 9;

// AIV Metadata Index Definitions
constexpr uint32_t AP_AIV_CORE_ENABLE_INDEX = 0;
constexpr uint32_t AP_GS1_IDX_END_OF_FD_HEAD_INDEX = 1;
constexpr uint32_t AP_GS1_IDX_END_OF_FD_HEAD_SPLIT_INDEX = 2;

// BASE Metadata Index Definitions
constexpr uint32_t AP_M_BASE_SIZE_INDEX = 0;
constexpr uint32_t AP_S_INNER_SIZE_INDEX = 1;
constexpr uint32_t AP_M_FD_BASE_SIZE_INDEX = 2;
constexpr uint32_t AP_NUM_OF_FD_INDEX = 3;
constexpr uint32_t AP_USED_CORE_NUM_INDEX = 4;
constexpr uint32_t AP_USED_VEC_NUM_OF_FD_INDEX = 5;
constexpr uint32_t AP_S1_SIZE_INDEX = 6;
constexpr uint32_t AP_S2_SIZE_INDEX = 7;
constexpr uint32_t AP_ACTUAL_LEN_Q_DIM_INDEX = 8;
constexpr uint32_t AP_ACTUAL_LEN_KV_DIM_INDEX = 9;

/**
 * @brief Get absolute index for AIC metadata
 */
#ifdef __CCE_AICORE__
__aicore__ inline uint32_t GetAPAICMetaAbsIndex(uint32_t coreIdx, uint32_t metaIdx)
{
    return coreIdx * AP_AIC_METADATA_SIZE + metaIdx;
}

__aicore__ inline uint32_t GetAPAIVMetaAbsIndex(uint32_t coreIdx, uint32_t metaIdx)
{
    const uint32_t aicTotalSize = AP_AIC_CORE_NUM * AP_AIC_METADATA_SIZE;
    return aicTotalSize + coreIdx * AP_AIV_METADATA_SIZE + metaIdx;
}

__aicore__ inline uint32_t GetAPBaseMetaAbsIndex(uint32_t metaIdx)
{
    const uint32_t aicTotalSize = AP_AIC_CORE_NUM * AP_AIC_METADATA_SIZE;
    const uint32_t aivTotalSize = AP_AIV_CORE_NUM * AP_AIV_METADATA_SIZE;
    return aicTotalSize + aivTotalSize + metaIdx;
}
#endif

namespace detail {
struct APMetaData {
    uint32_t aicMetadata[AP_AIC_CORE_NUM][AP_AIC_METADATA_SIZE];
    uint32_t aivMetadata[AP_AIV_CORE_NUM][AP_AIV_METADATA_SIZE];
    uint32_t baseMetadata[AP_BASE_METADATA_SIZE];
};
} // namespace detail

static_assert(AP_METADATA_SIZE * sizeof(AP_METADATA_T) >= sizeof(detail::APMetaData));
} // namespace optiling

#endif // ATTENTION_PIONEER_METADATA_H
