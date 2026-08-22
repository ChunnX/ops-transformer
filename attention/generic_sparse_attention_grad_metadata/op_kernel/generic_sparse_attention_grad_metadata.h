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
 * \file generic_sparse_attention_grad_metadata.h
 * \brief Metadata layout shared by AICPU host and device kernel.
 */

#ifndef GENERIC_SPARSE_ATTENTION_GRAD_METADATA_H
#define GENERIC_SPARSE_ATTENTION_GRAD_METADATA_H

#include <cstdint>

namespace optiling {

constexpr uint32_t AIC_CORE_MAX_NUM = 36;
constexpr uint32_t AIV_CORE_MAX_NUM = 72;
constexpr uint32_t GSAG_METADATA_TOTAL_SIZE = 4096;
using GSAG_METADATA_T = int64_t;

constexpr uint32_t GRAD_METADATA_SIZE = 8;
constexpr uint32_t TASK_ENTRY_SIZE = 4;

constexpr uint32_t GSAG_DEFAULT_BASE_M = 128;
constexpr uint32_t GSAG_DEFAULT_BASE_N = 128;

// Grad metadata header indices
constexpr uint32_t TOTAL_NUM = 0;
constexpr uint32_t TOTAL_BLOCK_COST = 1;
constexpr uint32_t MAX_CORE_BLOCK_COST = 2;
constexpr uint32_t BASE_M = 3;
constexpr uint32_t BASE_N = 4;
constexpr uint32_t USED_CORE_NUM = 5;
constexpr uint32_t GROUP_SIZE = 6;
constexpr uint32_t MAX_TASK_COUNT = 7;

constexpr uint32_t CORE_TASK_START_OFFSET = GRAD_METADATA_SIZE;
constexpr uint32_t CORE_TASK_END_OFFSET = GRAD_METADATA_SIZE + AIC_CORE_MAX_NUM;
constexpr uint32_t TASK_LIST_OFFSET = GRAD_METADATA_SIZE + 2 * AIC_CORE_MAX_NUM;

constexpr uint32_t TASK_B = 0;
constexpr uint32_t TASK_N2 = 1;
constexpr uint32_t TASK_J = 2;
constexpr uint32_t TASK_G = 3;

#ifdef __CCE_AICORE__
__aicore__ inline uint32_t GetCoreTaskStartIndex(uint32_t coreIdx)
{
    return CORE_TASK_START_OFFSET + coreIdx;
}

__aicore__ inline uint32_t GetCoreTaskEndIndex(uint32_t coreIdx)
{
    return CORE_TASK_END_OFFSET + coreIdx;
}

__aicore__ inline uint32_t GetTaskListIndex(uint32_t taskIdx, uint32_t fieldIdx)
{
    return TASK_LIST_OFFSET + taskIdx * TASK_ENTRY_SIZE + fieldIdx;
}
#endif

namespace detail {
struct GsagMetadataHeader {
    int64_t gradMetadata[GRAD_METADATA_SIZE];
    int64_t coreTaskStart[AIC_CORE_MAX_NUM];
    int64_t coreTaskEnd[AIC_CORE_MAX_NUM];
};
} // namespace detail

static_assert(GSAG_METADATA_TOTAL_SIZE * sizeof(GSAG_METADATA_T) >= sizeof(detail::GsagMetadataHeader));
} // namespace optiling

#endif // GENERIC_SPARSE_ATTENTION_GRAD_METADATA_H
