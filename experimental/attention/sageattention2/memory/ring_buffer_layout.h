#pragma once

#include "../core/platform_traits.h"
#include "../core/numerical_types.h"

#if defined(__CCE__)
#include "kernel_operator.h"
#include "catlass/arch/resource.hpp"

namespace sage::memory {

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

// Standard Pipeline UB Layout Plan
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

// Wide Streaming Vector UB Layout Plan
struct WideVectorLayout {
    static constexpr uint32_t PV_SLOT_BYTES = HALF * 128 * sizeof(float);
    static constexpr uint32_t SCORE_UB = 0;
    static constexpr uint32_t PV_UB = SCORE_UB + 2 * HALF * N * sizeof(half);
    static constexpr uint32_t ACCUM_UB = PV_UB + 2 * PV_SLOT_BYTES;
    static constexpr uint32_t P_UB = ACCUM_UB + HALF * 128 * sizeof(float);
    static constexpr uint32_t PARTIAL_SUMS_UB = P_UB + 2 * (HALF + 1) * N * sizeof(uint8_t);
    static constexpr uint32_t KS_UB = PARTIAL_SUMS_UB + HALF * 8 * sizeof(float);
    static constexpr uint32_t QS_UB = KS_UB + 2 * N * sizeof(half);
    static constexpr uint32_t MAX_UB = QS_UB + HALF * sizeof(half);
    static constexpr uint32_t DEN_UB = MAX_UB + 2 * HALF * sizeof(half);
    static constexpr uint32_t ALPHA_UB = DEN_UB + HALF * sizeof(float);
    static constexpr uint32_t SUM_UB = ALPHA_UB + 3 * 512;
    static constexpr uint32_t VM_UB = SUM_UB + HALF * sizeof(float);
    static constexpr uint32_t END = VM_UB + 128 * sizeof(float);

    static_assert(END <= UB_HIGH_WATER, "Wide streaming UB plan must stay within 248KB budget");
};

template <AscendC::HardEvent E>
__aicore__ inline void Sync()
{
    AscendC::SetFlag<E>(0);
    AscendC::WaitFlag<E>(0);
}

} // namespace sage::memory

namespace sage {
using memory::Arch;
using memory::FixedBuffer;
using memory::FixedResource;
using memory::WideVectorLayout;
using memory::Sync;
using AscendC::HardEvent;
}
#endif // __CCE__
