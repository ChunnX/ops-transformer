/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef FLASH_ATTN_DN_CONFIG_H
#define FLASH_ATTN_DN_CONFIG_H

namespace FlashAttnKernel {
// Storage dimensions select capacity; actual D/DV select arithmetic and tail handling.
// No Q prescale or UnitFlag is enabled in the shared DN implementation.
template <typename FA_T>
struct FaDnConfig {
    using INPUT_T = typename FA_T::inputType;
    using T = float;
    static constexpr uint32_t mBaseSize = static_cast<uint32_t>(FA_T::mBaseSize);
    static constexpr uint32_t s2BaseSize = static_cast<uint32_t>(FA_T::s2BaseSize);
    static constexpr uint32_t dBaseSize = static_cast<uint32_t>(FA_T::dBaseSize);
    static constexpr uint32_t dVBaseSize = static_cast<uint32_t>(FA_T::dVBaseSize);
    static_assert(mBaseSize == 128 && dVBaseSize <= 128, "PV DN requires 128 query rows and DV storage <=128");
    static constexpr uint32_t CV_RATIO = 2;
    static constexpr uint32_t dTemplateAlign64 = (dVBaseSize + 63U) / 64U * 64U;
    static_assert(s2BaseSize == 128, "DN uses the original KV128 configurations");
    static constexpr bool DEEP_PRELOAD = dBaseSize <= 64 && dVBaseSize <= 64;
    static constexpr uint32_t PRELOAD_N = DEEP_PRELOAD ? 3U : 2U;
    static constexpr uint32_t UB_MM_RES_BUFCNT = dBaseSize > 128 ? 2U : 4U;
    static constexpr uint32_t UB_MM_RES_BUF_BYTES =
        mBaseSize / CV_RATIO * s2BaseSize * sizeof(T) + (DEEP_PRELOAD ? 128U : 0U);
    static constexpr uint32_t UB_MM_RES_TOTAL_BYTES = UB_MM_RES_BUFCNT * UB_MM_RES_BUF_BYTES;
    static constexpr uint32_t UB_VEC2_RES_BUF_BYTES = mBaseSize / CV_RATIO * dTemplateAlign64 * sizeof(T);
    static constexpr uint32_t UB_VEC1_RES_BUFCNT = 2U;
    static constexpr uint32_t UB_VEC1_RES_BUF_BYTES = (mBaseSize / CV_RATIO + 1U) * s2BaseSize * sizeof(INPUT_T);
    static constexpr uint32_t UB_SOFTMAX_MAX_BUFCNT = DEEP_PRELOAD ? 4U : 3U;
    static constexpr uint32_t UB_SOFTMAX_MAX_BUF_BYTES =
        AttentionCommon::Align(mBaseSize / CV_RATIO * static_cast<uint32_t>(sizeof(T)), 256U);
    static constexpr uint32_t UB_SOFTMAX_SUM_BUFCNT = DEEP_PRELOAD ? 4U : 3U;
    static constexpr uint32_t UB_SOFTMAX_SUM_BUF_BYTES =
        AttentionCommon::Align(mBaseSize / CV_RATIO * static_cast<uint32_t>(sizeof(T)), 256U);
    static constexpr uint32_t UB_SOFTMAX_EXP_BUFCNT = DEEP_PRELOAD ? 4U : 3U;
    static constexpr uint32_t UB_SOFTMAX_EXP_BUF_BYTES =
        AttentionCommon::Align(mBaseSize / CV_RATIO * static_cast<uint32_t>(sizeof(T)), 256U);
    static constexpr uint32_t UB_LSE_OUT_BUFCNT = 2U;
    static constexpr uint32_t UB_LSE_OUT_BUF_BYTES =
        AttentionCommon::Align(mBaseSize / CV_RATIO * 8 * static_cast<uint32_t>(sizeof(T)), 256U);
    static constexpr uint32_t L1_P_BUFCNT = DEEP_PRELOAD ? 4U : 3U;
    static constexpr uint32_t L1_P_BUF_BYTES = mBaseSize * s2BaseSize * sizeof(INPUT_T);
    static constexpr uint32_t L1_Q_BUFCNT = 2U;
    static constexpr uint32_t L1_Q_BUF_BYTES = mBaseSize * dBaseSize * sizeof(INPUT_T);
    static constexpr uint32_t L1_KV_BUFCNT = 4U;
    static constexpr uint32_t L1_KV_BUF_BYTES = 64U * 1024U;
    static constexpr uint32_t L0C_BUFCNT = 4U;
    static constexpr uint32_t L0C_BUF_BYTES = 64U * 1024U;
    static constexpr uint32_t L1P_SYNC_BASE = DEEP_PRELOAD ? 4U : 5U;
    static constexpr uint32_t FA_UB_BYTES =
        UB_MM_RES_TOTAL_BYTES + UB_VEC2_RES_BUF_BYTES + UB_VEC1_RES_BUFCNT * UB_VEC1_RES_BUF_BYTES +
        UB_SOFTMAX_MAX_BUFCNT * UB_SOFTMAX_MAX_BUF_BYTES + UB_SOFTMAX_SUM_BUFCNT * UB_SOFTMAX_SUM_BUF_BYTES +
        UB_SOFTMAX_EXP_BUFCNT * UB_SOFTMAX_EXP_BUF_BYTES + UB_LSE_OUT_BUFCNT * UB_LSE_OUT_BUF_BYTES;
    static constexpr uint32_t L1_BYTES =
        L1_P_BUFCNT * L1_P_BUF_BYTES + L1_Q_BUFCNT * L1_Q_BUF_BYTES + L1_KV_BUFCNT * L1_KV_BUF_BYTES;
    static_assert(FA_UB_BYTES <= 248U * 1024U, "DN FA buffers exceed UB");
    static_assert(L1_BYTES <= 512U * 1024U, "DN buffers exceed L1");

    static __aicore__ inline uint32_t MmSlotOffset(uint32_t id)
    {
        return id * UB_MM_RES_BUF_BYTES;
    }
};
// Cube and Vector maintain independent instances with the same slot sequence.
template <typename Config>
class FaDnMmSlots {
public:
    __aicore__ inline uint32_t Acquire()
    {
        uint32_t id = shared_;
        shared_ = (shared_ + 1U) % Config::UB_MM_RES_BUFCNT;
        return id;
    }

private:
    uint32_t shared_ = 0U;
};
} // namespace FlashAttnKernel
#endif
