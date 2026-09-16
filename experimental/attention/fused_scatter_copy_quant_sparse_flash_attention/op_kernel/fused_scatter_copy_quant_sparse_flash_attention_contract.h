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
 * \file fused_scatter_copy_quant_sparse_flash_attention_contract.h
 * \brief
 */

#ifndef FUSED_SCATTER_COPY_QUANT_SPARSE_FLASH_ATTENTION_CONTRACT_H
#define FUSED_SCATTER_COPY_QUANT_SPARSE_FLASH_ATTENTION_CONTRACT_H

#include <cstdint>

#ifdef __CCE_AICORE__
#define FSCQSFA_CONTRACT_INLINE __aicore__ inline
#else
#define FSCQSFA_CONTRACT_INLINE constexpr
#endif

namespace fused_scatter_copy_qsfa {
constexpr uint32_t TOPK_CAPACITY = 2048;
constexpr uint32_t COPY_CAPACITY = 32768;
constexpr uint32_t MAX_QUERY_COUNT = 16;

FSCQSFA_CONTRACT_INLINE bool IsSupportedHeadCount(uint32_t heads)
{
    return heads >= 8 && heads <= 128 && (heads & (heads - 1)) == 0;
}

FSCQSFA_CONTRACT_INLINE bool IsValidQuerySegment(
    int32_t previous, int32_t end, int32_t total)
{
    return end > previous && end <= total &&
        end - previous <= static_cast<int32_t>(MAX_QUERY_COUNT);
}
} // namespace fused_scatter_copy_qsfa

#undef FSCQSFA_CONTRACT_INLINE

#endif
