/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#if defined(__CCE__)
#include "kernel_operator.h"
#include "catlass/arch/resource.hpp"
#include "types.h"

namespace sage::device {

using Arch = Catlass::Arch::Ascend950;

// Static buffer wrapper to avoid runtime TPipe bookkeeping
template <AscendC::TPosition POSITION, uint32_t BYTES>
struct FixedBuffer {
    template <class T>
    __aicore__ inline AscendC::LocalTensor<T> GetBufferByByte(uint32_t offset) const
    {
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

template <AscendC::HardEvent E>
__aicore__ inline void Sync()
{
    AscendC::SetFlag<E>(0);
    AscendC::WaitFlag<E>(0);
}

template <class T>
__aicore__ inline void ReadRow(AscendC::LocalTensor<T> dst, AscendC::GlobalTensor<T> src, uint64_t off, uint32_t d, uint32_t dp)
{
    AscendC::Duplicate(dst, T(0), dp);
    Sync<AscendC::HardEvent::V_MTE2>();
    AscendC::DataCopyExtParams copy{1, d * uint32_t(sizeof(T)), 0, 0, 0};
    constexpr uint32_t blockElements = 32 / sizeof(T);
    uint8_t rightPadding = static_cast<uint8_t>((blockElements - d % blockElements) % blockElements);
    AscendC::DataCopyPadExtParams<T> pad{true, 0, rightPadding, T(0)};
    AscendC::DataCopyPad(dst, src[off], copy, pad);
    Sync<AscendC::HardEvent::MTE2_V>();
}

template <class T>
__aicore__ inline void WriteRow(AscendC::GlobalTensor<T> dst, uint64_t off, AscendC::LocalTensor<T> src, uint32_t n)
{
    Sync<AscendC::HardEvent::V_MTE3>();
    AscendC::DataCopyExtParams copy{1, n * uint32_t(sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(dst[off], src, copy);
}

} // namespace sage::device

namespace sage {
using device::Arch;
using device::FixedBuffer;
using device::FixedResource;
using device::Sync;
using device::ReadRow;
using device::WriteRow;
using AscendC::RoundMode;
using AscendC::CMPMODE;
} // namespace sage
#endif
