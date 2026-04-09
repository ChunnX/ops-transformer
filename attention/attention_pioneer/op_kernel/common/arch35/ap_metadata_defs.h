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
 * \file ap_metadata_defs.h
 * \brief Metadata layout constants for attention_pioneer split-core scheduling.
 *        Mirrors the layout defined in attention_pioneer_metadata, kept local
 *        to avoid cross-operator includes.
 */

#ifndef AP_METADATA_DEFS_H_
#define AP_METADATA_DEFS_H_

namespace ap_metadata {

constexpr uint32_t AIC_CORE_NUM = 36;
constexpr uint32_t SLOTS_PER_CORE = 10;
constexpr uint32_t BASE_OFFSET = AIC_CORE_NUM * SLOTS_PER_CORE;  // 360

// Per-core field indices
constexpr uint32_t BN2_START_PTR_INDEX = 1;
constexpr uint32_t GS1_START_PTR_INDEX = 2;

// Base metadata field indices
constexpr uint32_t USED_CORE_NUM_INDEX = 0;
constexpr uint32_t S1_SIZE_INDEX = 4;
constexpr uint32_t S2_SIZE_INDEX = 5;
constexpr uint32_t ACTUAL_LEN_Q_DIM_INDEX = 6;
constexpr uint32_t ACTUAL_LEN_KV_DIM_INDEX = 7;
constexpr uint32_t S1_OUTER_SIZE_INDEX = 3;
constexpr uint32_t TOTAL_SIZE_INDEX = 10;
constexpr uint32_t SPLIT_FACTOR_SIZE_INDEX = 11;
constexpr uint32_t SPLIT_FACTOR_TAIL_SIZE_INDEX = 12;
constexpr uint32_t T1_SIZE_INDEX = 13;
constexpr uint32_t T2_SIZE_INDEX = 14;

__aicore__ inline constexpr uint32_t GetCoreFieldOffset(uint32_t coreIdx, uint32_t fieldIdx)
{
    return coreIdx * SLOTS_PER_CORE + fieldIdx;
}

__aicore__ inline constexpr uint32_t GetBaseMetaIndex(uint32_t fieldIdx)
{
    return BASE_OFFSET + fieldIdx;
}

} // namespace ap_metadata

#endif // AP_METADATA_DEFS_H_
