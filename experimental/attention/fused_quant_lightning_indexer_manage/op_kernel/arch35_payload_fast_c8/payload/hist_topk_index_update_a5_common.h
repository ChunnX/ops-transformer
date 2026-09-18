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
 * \file hist_topk_index_update_a5_common.h
 * \brief Common definitions for hist_topk_index_update_a5 (vectorized classify)
 */
#ifndef VLLM_FAST_HIST_TOPK_INDEX_UPDATE_A5_COMMON_H
#define VLLM_FAST_HIST_TOPK_INDEX_UPDATE_A5_COMMON_H

using namespace AscendC;

namespace VllmFastHistTopkIndexUpdateA5Common {

constexpr uint32_t DEFAULT_TOPK_TARGET = 2048;
constexpr uint32_t DEFAULT_MAX_TOPK_OUT = 3072;

constexpr uint32_t TRUNK_LEN_16K = 16384;
constexpr uint32_t TRUNK_LEN_8K = 8192;
constexpr uint32_t TOPK_LEN_4K = 4096;

template <typename T>
__aicore__ inline T Align(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd) * (rnd)));
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

template <typename T1, typename T2>
__aicore__ inline T1 Max(T1 a, T2 b)
{
    return (a > b) ? (a) : (b);
}

template <typename T>
__aicore__ inline T CeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd)-1) / (rnd)));
}

} // namespace VllmFastHistTopkIndexUpdateA5Common

#endif // VLLM_FAST_HIST_TOPK_INDEX_UPDATE_A5_COMMON_H
