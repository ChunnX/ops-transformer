/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "../common/constants.h"
#include "../common/types.h"

#if defined(__CCE__)
#include "kernel_operator.h"
#include "catlass/arch/resource.hpp"

namespace sage::hardware {

using Arch = Catlass::Arch::Ascend950;

// Static buffer wrapper to avoid runtime TPipe bookkeeping
template <AscendC::TPosition POSITION, uint32_t BYTES>
struct FixedBuffer {
    template <class T>
    __aicore__ inline AscendC::LocalTensor<T> GetBufferByByte(uint32_t offset) const {
        return AscendC::LocalTensor<T>(POSITION, offset, (BYTES - offset) / sizeof(T));
    }
};

// Static physical hardware buffer plan across L1, L0A, L0B, L0C, and UB
struct FixedResource {
    FixedBuffer<AscendC::TPosition::A1, Arch::L1_SIZE> l1Buf;
    FixedBuffer<AscendC::TPosition::A2, Arch::L0A_SIZE> l0ABuf;
    FixedBuffer<AscendC::TPosition::B2, Arch::L0B_SIZE> l0BBuf;
    FixedBuffer<AscendC::TPosition::CO1, Arch::L0C_SIZE> l0CBuf;
    FixedBuffer<AscendC::TPosition::VECCALC, Arch::UB_SIZE> ubBuf;

    __aicore__ inline FixedResource() { AscendC::PipeBarrier<PIPE_ALL>(); }
    __aicore__ inline ~FixedResource() { AscendC::PipeBarrier<PIPE_ALL>(); }
};

// ============================================================================
// Standard Pipeline UB Layout Plan
// ============================================================================
constexpr uint32_t PV_SLOT_BYTES = HALF * 128 * 4;
constexpr uint32_t SCORE_UB = 0;
constexpr uint32_t PV_UB = PIPE_SLOTS * HALF * N * 2;
constexpr uint32_t ACCUM_UB = PV_UB + PV_SLOTS * PV_SLOT_BYTES;
constexpr uint32_t P_UB = ACCUM_UB + HALF * 128 * 4;
constexpr uint32_t KS_UB = P_UB + PIPE_SLOTS * P_SLOT_BYTES;
constexpr uint32_t QS_UB = KS_UB + 2 * N * 2;
constexpr uint32_t GROUP_STAT_BYTES = N * 2;
constexpr uint32_t MAX_UB = QS_UB + GROUP_STAT_BYTES;
constexpr uint32_t DEN_UB = MAX_UB + GROUP_STAT_BYTES;
constexpr uint32_t ALPHA_UB = DEN_UB + GROUP_STAT_BYTES;
constexpr uint32_t SUM_UB = ALPHA_UB + ALPHA_SLOTS * GROUP_STAT_BYTES;
constexpr uint32_t VM_UB = SUM_UB + HALF * 4;
constexpr uint32_t UB_END = VM_UB + 128 * sizeof(float);

static_assert(UB_END <= UB_HIGH_WATER, "Static UB plan must stay within 248KB budget");

// ============================================================================
// Wide Streaming Pipeline UB Layout Plan
// ============================================================================
struct WideVectorLayout {
    static constexpr uint32_t SCORE_UB = 0;
    static constexpr uint32_t PV_UB = 4 * HALF * N * 2;
    static constexpr uint32_t P_PART_BYTES = HALF * N;
    static constexpr uint32_t PV_SLOT_BYTES = 2 * P_PART_BYTES;
    static constexpr uint32_t ACCUM_UB = PV_UB + 2 * PV_SLOT_BYTES;
    static constexpr uint32_t P_UB = ACCUM_UB + HALF * 128 * 4;
    static constexpr uint32_t KS_UB = P_UB + P_SLOT_BYTES;
    static constexpr uint32_t QS_UB = KS_UB + 2 * 512 * 2;
    static constexpr uint32_t MAX_UB = QS_UB + 512;
    static constexpr uint32_t DEN_UB = MAX_UB + 512;
    static constexpr uint32_t ALPHA_UB = DEN_UB + 512;
    static constexpr uint32_t SUM_UB = ALPHA_UB + 3 * 512;
    static constexpr uint32_t VM_UB = SUM_UB + HALF * 4;
    static constexpr uint32_t PARTIAL_SUMS_UB = P_UB;
    static constexpr uint32_t END = VM_UB + 128 * sizeof(float);

    static_assert(PV_SLOT_BYTES >= HALF * 128 * 4, "PV slot cannot hold FP32 PV");
    static_assert(P_SLOT_BYTES >= HALF * 8 * 4, "Output scratch cannot hold softmax partial sums");
    static_assert(END <= UB_HIGH_WATER, "Wide softmax layout exceeds UB 248KB budget");
};

// Wide-pipeline L1 buffer allocations & constants.
// Note: MX_V_L1 (the fixed standard-path MX scale-cache slot, 320 KB) is owned
// by common/constants.h in namespace sage; the wide path streams scales through
// V_SCALE_L1_PING/PONG instead, so no alias is defined here — redefining the
// name in sage::hardware with a different value would silently fork the plan.
constexpr uint32_t MX_P_BLOCK_NUM = M * N / 32 / 32;
constexpr uint32_t MX_P_M_BLOCKS = M / 16;
constexpr uint32_t MX_P_N_BLOCKS = N / 64;
constexpr uint32_t L0A_MX_P_OFFSET = 32768;

} // namespace sage::hardware

namespace sage {
using hardware::Arch;
using hardware::FixedBuffer;
using hardware::FixedResource;
using hardware::WideVectorLayout;
} // namespace sage

#endif // __CCE__
