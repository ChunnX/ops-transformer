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
 * \file flash_attn_metadata.h
 * \brief
 */

#ifndef FLASH_ATTN_METADATA_H
#define FLASH_ATTN_METADATA_H

#include <cstdint>
#include <cassert>

namespace optiling {

// Constants
constexpr uint32_t AIC_CORE_NUM = 36;
constexpr uint32_t AIV_CORE_NUM = 72;
constexpr uint32_t FA_META_SIZE = 1024;
using FA_METADATA_T = uint32_t;

constexpr uint32_t FA_METADATA_SIZE = 16;
constexpr uint32_t FD_METADATA_SIZE = 16;

// FA Metadata Index Definitions
constexpr uint32_t FA_BN2_START_INDEX = 0;
constexpr uint32_t FA_M_START_INDEX = 1;
constexpr uint32_t FA_S2_START_INDEX = 2;
constexpr uint32_t FA_BN2_END_INDEX = 3;
constexpr uint32_t FA_M_END_INDEX = 4;
constexpr uint32_t FA_S2_END_INDEX = 5;
constexpr uint32_t FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 6;

// FD Metadata Index Definitions
constexpr uint32_t FD_BN2_IDX_INDEX = 0;
constexpr uint32_t FD_M_IDX_INDEX = 1;
constexpr uint32_t FD_WORKSPACE_IDX_INDEX = 2;
constexpr uint32_t FD_WORKSPACE_NUM_INDEX = 3;
constexpr uint32_t FD_M_START_INDEX = 4;
constexpr uint32_t FD_M_NUM_INDEX = 5;

#ifdef __CCE_AICORE__

/**
 * @brief 获取属性的绝对索引
 * @param coreIdx 核索引
 * @param metaIdx 元数据索引
 * @param isAIV 是否为AIV数据，默认为false
 * @return 返回属性的绝对索引
 */
__aicore__ inline uint32_t GetAttrAbsIndex(uint32_t idx, uint32_t coreIdx,
                                           uint32_t metaIdx, uint32_t num, bool isAIV = false)
{
    if (isAIV) {
        return num * AIC_CORE_NUM * FA_METADATA_SIZE + FD_METADATA_SIZE * coreIdx + metaIdx + 16U;
    } else {
        return FA_METADATA_SIZE * coreIdx + metaIdx + 16U;
    }
}
#endif

namespace detail {
struct FaMetaData {
    uint32_t *faMetadata; // [AIC_CORE_NUM][FA_METADATA_SIZE];
    uint32_t *fdMetadata; // [AIV_CORE_NUM][FD_METADATA_SIZE];
    FaMetaData(void *metadataPtr)
        : faMetadata(static_cast<uint32_t*>(metadataPtr) + 16U),
          fdMetadata(static_cast<uint32_t*>(metadataPtr) + 16U + AIC_CORE_NUM * FA_METADATA_SIZE)
    {
        static_cast<uint32_t*>(metadataPtr)[0] = 1;
    }
    void setFaMetadata(uint32_t aicIdx, uint32_t metaIdx, uint32_t val)
    {
        assert(aicIdx < AIC_CORE_NUM);
        assert(metaIdx < FA_METADATA_SIZE);
        faMetadata[FA_METADATA_SIZE * aicIdx + metaIdx] = val;
    }
    uint32_t getFaMetadata(uint32_t aicIdx, uint32_t metaIdx)
    {
        assert(aicIdx < AIC_CORE_NUM);
        assert(metaIdx < FA_METADATA_SIZE);
        return faMetadata[FA_METADATA_SIZE * aicIdx + metaIdx];
    }
    void setFdMetadata(uint32_t aivIdx, uint32_t metaIdx, uint32_t val)
    {
        assert(aivIdx < AIV_CORE_NUM);
        assert(metaIdx < FD_METADATA_SIZE);
        fdMetadata[FD_METADATA_SIZE * aivIdx + metaIdx] = val;
    }
    uint32_t getFdMetadata(uint32_t aivIdx, uint32_t metaIdx)
    {
        assert(aivIdx < AIV_CORE_NUM);
        assert(metaIdx < FD_METADATA_SIZE);
        return fdMetadata[FD_METADATA_SIZE * aivIdx + metaIdx];
    }
};
} // namespace detail

} // namespace optiling

#endif
