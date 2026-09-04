/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QUANT_LIGHTNING_INDEXER_MTP_FAST_PATH_H
#define QUANT_LIGHTNING_INDEXER_MTP_FAST_PATH_H

#include <cstdint>

namespace QLIConfig {

inline constexpr uint32_t MTP_MIN_QUERY_SEQ_LEN = 2U;
inline constexpr uint32_t MTP_MAX_QUERY_SEQ_LEN = 4U;
inline constexpr uint32_t SPARSE_COUNT_ALIGNMENT = 256U;
inline constexpr uint32_t MTP_SPARSE_COUNT_512 = 512U;
inline constexpr uint32_t MTP_SPARSE_COUNT_1024 = 1024U;
inline constexpr uint32_t RIGHT_DOWN_CAUSAL_SPARSE_MODE = 3U;
inline constexpr uint32_t DEEPSEEK_V4_CMP_RATIO = 4U;

inline constexpr uint32_t DEFAULT_M_BASE_SIZE = 256U;
inline constexpr uint32_t DEFAULT_S2_BASE_SIZE = 128U;
inline constexpr uint32_t MTP_M_BASE_SIZE = 128U;
inline constexpr uint32_t MTP_S2_BASE_SIZE = 256U;

inline constexpr bool IsMtpShortQueryFastPath(bool isAscend950, bool isTndQuery, bool isPageAttention,
                                              uint32_t querySeqLen, uint32_t sparseCount, uint32_t sparseMode,
                                              uint32_t cmpRatio)
{
    const bool isMtpQuery = querySeqLen >= MTP_MIN_QUERY_SEQ_LEN && querySeqLen <= MTP_MAX_QUERY_SEQ_LEN;
    const bool isSupportedSparseCount = sparseCount == MTP_SPARSE_COUNT_512 || sparseCount == MTP_SPARSE_COUNT_1024;
    return isAscend950 && isTndQuery && isPageAttention && isMtpQuery && isSupportedSparseCount &&
           sparseMode == RIGHT_DOWN_CAUSAL_SPARSE_MODE && cmpRatio == DEEPSEEK_V4_CMP_RATIO;
}

static_assert(MTP_SPARSE_COUNT_512 % SPARSE_COUNT_ALIGNMENT == 0U);
static_assert(MTP_SPARSE_COUNT_1024 % SPARSE_COUNT_ALIGNMENT == 0U);

inline constexpr uint32_t GetMBaseSize(bool useMtpFastPath)
{
    return useMtpFastPath ? MTP_M_BASE_SIZE : DEFAULT_M_BASE_SIZE;
}

inline constexpr uint32_t GetS2BaseSize(bool useMtpFastPath)
{
    return useMtpFastPath ? MTP_S2_BASE_SIZE : DEFAULT_S2_BASE_SIZE;
}

} // namespace QLIConfig

#endif // QUANT_LIGHTNING_INDEXER_MTP_FAST_PATH_H
