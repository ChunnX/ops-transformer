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
 * \file lightning_indexer_v2_metadata_common.h
 * \brief Common metadata definitions shared by LightningIndexerV2 and QuantLightningIndexerV2.
 */

#ifndef LIGHTNING_INDEXER_V2_METADATA_COMMON_H
#define LIGHTNING_INDEXER_V2_METADATA_COMMON_H

#include <cstdint>

namespace optiling {

inline constexpr uint32_t AIC_CORE_MAX_NUM = 36;
inline constexpr uint32_t AIV_CORE_MAX_NUM = 72;
inline constexpr uint32_t INDEXER_V2_METADATA_SIZE = 8;
inline constexpr uint32_t LD_V2_METADATA_SIZE = 8;

inline constexpr uint32_t INDEXER_V2_CORE_ENABLE_INDEX = 0;
inline constexpr uint32_t INDEXER_V2_BN2_START_INDEX = 1;
inline constexpr uint32_t INDEXER_V2_M_START_INDEX = 2;
inline constexpr uint32_t INDEXER_V2_S2_START_INDEX = 3;
inline constexpr uint32_t INDEXER_V2_BN2_END_INDEX = 4;
inline constexpr uint32_t INDEXER_V2_M_END_INDEX = 5;
inline constexpr uint32_t INDEXER_V2_S2_END_INDEX = 6;
inline constexpr uint32_t INDEXER_V2_FIRST_LD_DATA_WORKSPACE_IDX_INDEX = 7;

inline constexpr uint32_t LD_V2_CORE_ENABLE_INDEX = 0;
inline constexpr uint32_t LD_V2_BN2_IDX_INDEX = 1;
inline constexpr uint32_t LD_V2_M_IDX_INDEX = 2;
inline constexpr uint32_t LD_V2_WORKSPACE_IDX_INDEX = 3;
inline constexpr uint32_t LD_V2_WORKSPACE_NUM_INDEX = 4;
inline constexpr uint32_t LD_V2_M_START_INDEX = 5;
inline constexpr uint32_t LD_V2_M_NUM_INDEX = 6;

#ifdef __CCE_AICORE__
__aicore__ inline uint32_t GetAttrAbsIndex(uint32_t coreIdx, uint32_t metaIdx, bool isAIV = false)
{
    if (isAIV) {
        return INDEXER_V2_METADATA_SIZE * AIC_CORE_MAX_NUM + LD_V2_METADATA_SIZE * coreIdx + metaIdx;
    }
    return INDEXER_V2_METADATA_SIZE * coreIdx + metaIdx;
}
#endif

} // namespace optiling

#endif // LIGHTNING_INDEXER_V2_METADATA_COMMON_H
