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
 * \file sparse_attn_sharedkv_grad_metadata.h
 * \brief
 */

#ifndef SPARSE_ATTN_SHAREDKV_GRAD_METADATA_H
#define SPARSE_ATTN_SHAREDKV_GRAD_METADATA_H

#include <cstdint>

namespace optiling {

// Constants
constexpr uint32_t AIC_CORE_NUM = 36;
constexpr uint32_t AIV_CORE_NUM = 72;
constexpr uint32_t SASG_METADATA_SIZE = 1024;
using SASG_METADATA_T = int32_t;

constexpr uint32_t GRAD_METADATA_SIZE = 5;

// Grad Metadata Index Definitions
constexpr uint32_t TOTAL_NUM = 0;
constexpr uint32_t FORMER_CORE_PROCESS_NUM = 1;
constexpr uint32_t REMAIN_CORE_PROCESS_NUM = 2;
constexpr uint32_t REMAIN_CORE_NUM = 3;
constexpr uint32_t USED_CORE_NUM = 4;

/**
 * @brief 获取属性的绝对索引
 * @param coreIdx 核索引
 * @param metaIdx 元数据索引
 * @return 返回属性的绝对索引
 */
#ifdef __CCE_AICORE__
__aicore__ inline uint32_t GetGradAttrAbsIndex(uint32_t metaIdx)
{
    return metaIdx;
}
#endif

namespace detail {
struct SasgMetadata {
    uint32_t gradMetadata[GRAD_METADATA_SIZE];
};
} // namespace detail

static_assert(SASG_METADATA_SIZE * sizeof(SASG_METADATA_T) >= sizeof(detail::SasgMetadata));
} // namespace optiling

#endif