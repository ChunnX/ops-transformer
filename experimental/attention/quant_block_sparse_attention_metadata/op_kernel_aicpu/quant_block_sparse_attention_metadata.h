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
 * \file quant_block_sparse_attention_metadata.h
 * \brief
 */

#ifndef QUANT_BLOCK_SPARSE_ATTENTION_METADATA_H
#define QUANT_BLOCK_SPARSE_ATTENTION_METADATA_H

#include <cstdint>
#ifndef __CCE_AICORE__
#include <cassert>
#endif

namespace optiling {

constexpr uint32_t AIC_CORE_NUM = 36U;
constexpr uint32_t AIV_CORE_NUM = 72U;
constexpr uint32_t QBSA_METADATA_SIZE = 2048U;
using QBSA_METADATA_T = uint32_t;

// constexpr uint32_t METADATA_HEADER_SIZE = 8;
constexpr uint32_t FA_METADATA_SIZE = 8U;
constexpr uint32_t FD_METADATA_SIZE = 8U;

// FA Metadata Index Definitions
constexpr uint32_t FA_CORE_ENABLE_INDEX = 0U; // 当前AIC核是否有负载需要计算
constexpr uint32_t FA_BN1_START_INDEX = 1U; // 当前AIC核的起始bn1Idx
constexpr uint32_t FA_M_START_INDEX = 2U; // 当前AIC核的起始s1Idx
constexpr uint32_t FA_S2_START_INDEX = 3U; // 当前AIC核的起始s2Idx
constexpr uint32_t FA_BN1_END_INDEX = 4U; // 当前AIC核的终止bn1Idx
constexpr uint32_t FA_M_END_INDEX = 5U; // 当前AIC核的终止s1Idx
constexpr uint32_t FA_S2_END_INDEX = 6U; // 当前AIC核的终止s2Idx
constexpr uint32_t FA_FIRST_FD_DATA_WORKSPACE_IDX_INDEX = 7U; // 当前AIC核第一份规约任务在FD_DATA_WORKSPACE的idx

#ifndef __CCE_AICORE__
namespace detail {
struct FaMetaData {
    QBSA_METADATA_T *faMetadata; // [AIC_CORE_NUM][FA_METADATA_SIZE];
    QBSA_METADATA_T *fdMetadata; // [AIV_CORE_NUM][FD_METADATA_SIZE];

    explicit FaMetaData(void *metadataPtr)
        : faMetadata(static_cast<QBSA_METADATA_T *>(metadataPtr)),
          fdMetadata(static_cast<QBSA_METADATA_T *>(metadataPtr) + AIC_CORE_NUM * FA_METADATA_SIZE)
    {}

    void setFaMetadata(uint32_t aicIdx, uint32_t metaIdx, uint32_t val)
    {
        assert(aicIdx < AIC_CORE_NUM);
        assert(metaIdx < FA_METADATA_SIZE);
        faMetadata[FA_METADATA_SIZE * aicIdx + metaIdx] = static_cast<QBSA_METADATA_T>(val);
    }

    uint32_t getFaMetadata(uint32_t aicIdx, uint32_t metaIdx) const
    {
        assert(aicIdx < AIC_CORE_NUM);
        assert(metaIdx < FA_METADATA_SIZE);
        return static_cast<uint32_t>(faMetadata[FA_METADATA_SIZE * aicIdx + metaIdx]);
    }
};
} // namespace detail
#endif

static_assert(QBSA_METADATA_SIZE * sizeof(QBSA_METADATA_T) >=
              (AIC_CORE_NUM * FA_METADATA_SIZE + AIV_CORE_NUM * FD_METADATA_SIZE) * sizeof(QBSA_METADATA_T),
              "QBSA metadata output buffer is too small");

} // namespace optiling

#endif // QUANT_BLOCK_SPARSE_ATTENTION_METADATA_H
