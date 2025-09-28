/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file common_header.h
 * \brief
 */

#ifndef _COMMON_HEADER_H_
#define _COMMON_HEADER_H_


#include <limits>
#include <type_traits>
#include "kernel_operator.h"
#include "kernel_event.h"
#include "kernel_tensor.h"
#include "kernel_macros.h"

#define SET_FLAG(trigger, waiter, e) AscendC::SetFlag<AscendC::HardEvent::trigger##_##waiter>((e))
#define WAIT_FLAG(trigger, waiter, e) AscendC::WaitFlag<AscendC::HardEvent::trigger##_##waiter>((e))
#define PIPE_BARRIER(pipe) AscendC::PipeBarrier<PIPE_##pipe>()

/////////////////////////////////////////////////////
// common struct
/////////////////////////////////////////////////////
struct DynamicParams {
    int32_t l1_m_size;
    int32_t l1_n_size;
    int32_t l1_k_size;
    int32_t baseM;
    int32_t baseN;
};

struct AddrInfo {
    uint64_t left;
    uint64_t right;
    uint64_t out;
    int32_t kx = 0;
    int32_t ky = 0;
    int32_t lineStride = 0;
    bool lowerLeft;
    bool upperRight;
    int32_t S1Idx;
    int32_t S2Idx;
    int32_t blockStart;
};

struct CubeAddrInfo {
    int32_t taskId;
    int32_t blockLength;
    AddrInfo addrInfo[16];
};

struct VecBlockInfo {
    int32_t S1Idx;
    int32_t S2Idx;
    int32_t batchIdx;
    int32_t headNumIdx;
    int32_t n2Idx;
    int32_t gIdx = 0;
    int32_t offset;
    int32_t lengthx = 0;
    int32_t lengthy = 0;
    int8_t mask = 0;
};

struct VecAddrInfo {
    int32_t taskId;
    int32_t blockLength = 0;
    VecBlockInfo VecBlkInfo[16];
};

constexpr uint32_t CUBE2VEC = 7;
constexpr uint32_t VEC2CUBE = 8;
constexpr uint32_t CUBE2POST = 9;


/////////////////////////////////////////////////////
// hardware
/////////////////////////////////////////////////////
enum class ArchType {
    ASCEND_V220,
    ASCEND_V200,
    ASCEND_M200
};

template <ArchType ArchTag> struct HardwareInfo {
    static uint32_t const l2BW = 5;
    static uint32_t const hbmBW = 1;
    static uint32_t const supportMix = 0;
    static uint32_t const l1Size = 512 * 1024;
    static uint32_t const l0ASize = 64 * 1024;
    static uint32_t const l0BSize = 64 * 1024;
    static uint32_t const l0CSize = 128 * 1024;
    static uint32_t const l2Size = 192 * 1024 * 1024;
    static uint32_t const biasSize = 1024;
    static uint32_t const fixBufSize = 7 * 1024;
    static uint32_t const ubSize = 192 * 1024;
    static uint32_t const fractalSize = 512;
    static uint32_t const l1l0BlockSize = 32;
    static uint32_t const btBlockSize = 64;
    static uint32_t const fbBlockSize = 128;
};


/////////////////////////////////////////////////////
// mem
/////////////////////////////////////////////////////
enum class BufferType {
    ASCEND_UB,
    ASCEND_CB,
    ASCEND_L0A,
    ASCEND_L0B,
    ASCEND_L0C,
    ASCEND_MAX
};

template <ArchType ArchTag> struct AsdopsBuffer {
public:
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    __aicore__ AsdopsBuffer()
    {
        constexpr uint32_t bufferSize[(uint32_t)BufferType::ASCEND_MAX] = {
            HardwareInfo<ArchTag>::ubSize, HardwareInfo<ArchTag>::l1Size, HardwareInfo<ArchTag>::l0ASize,
            HardwareInfo<ArchTag>::l0BSize, HardwareInfo<ArchTag>::l0CSize};
#ifdef __DAV_C220_VEC__
        tensor[(uint32_t)BufferType::ASCEND_UB].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_UB]);
        tensor[(uint32_t)BufferType::ASCEND_UB].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
#elif __DAV_C220_CUBE__
        tensor[(uint32_t)BufferType::ASCEND_CB].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_CB]);
        tensor[(uint32_t)BufferType::ASCEND_CB].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::A1);
        tensor[(uint32_t)BufferType::ASCEND_L0A].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_L0A]);
        tensor[(uint32_t)BufferType::ASCEND_L0A].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::A2);
        tensor[(uint32_t)BufferType::ASCEND_L0B].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_L0B]);
        tensor[(uint32_t)BufferType::ASCEND_L0B].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::B2);
        tensor[(uint32_t)BufferType::ASCEND_L0C].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_L0C]);
        tensor[(uint32_t)BufferType::ASCEND_L0C].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::CO1);
#else
        tensor[(uint32_t)BufferType::ASCEND_UB].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_UB]);
        tensor[(uint32_t)BufferType::ASCEND_UB].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECIN);
        tensor[(uint32_t)BufferType::ASCEND_CB].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_CB]);
        tensor[(uint32_t)BufferType::ASCEND_CB].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::A1);
        tensor[(uint32_t)BufferType::ASCEND_L0A].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_L0A]);
        tensor[(uint32_t)BufferType::ASCEND_L0A].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::A2);
        tensor[(uint32_t)BufferType::ASCEND_L0B].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_L0B]);
        tensor[(uint32_t)BufferType::ASCEND_L0B].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::B2);
        tensor[(uint32_t)BufferType::ASCEND_L0C].InitBuffer(0, bufferSize[(uint32_t)BufferType::ASCEND_L0C]);
        tensor[(uint32_t)BufferType::ASCEND_L0C].address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::CO1);
#endif
    };
#pragma GCC diagnostic pop

    template <BufferType BufferType_, typename DstDataType>
    __aicore__ AscendC::LocalTensor<DstDataType> GetBuffer(const uint32_t offset) const
    {
        return tensor[(uint32_t)BufferType_][offset].template ReinterpretCast<DstDataType>();
    }

public:
    AscendC::LocalTensor<uint8_t> tensor[(uint32_t)BufferType::ASCEND_MAX];
};


/////////////////////////////////////////////////////
// common function
/////////////////////////////////////////////////////

template <typename T> inline __aicore__ T RoundUp(const T val, const T align)
{
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    if (align == 0 || val + align - 1 < val) {
        return val;
    }
    return (val + align - 1) / align * align;
}

template <typename T> constexpr T T_MAX = std::numeric_limits<T>::max();

template <typename T> inline __aicore__ T CeilDiv(const T dividend, const T divisor)
{
    static_assert(std::is_arithmetic<T>::value, "T must be an arithmetic type");
    if (divisor == 0 || dividend + divisor - 1 < dividend) {
        return T_MAX<T>;
    }
    return (dividend + divisor - 1) / divisor;
}

template <typename T> __aicore__ inline T Min(const T lhs, const T rhs)
{
    return lhs < rhs ? lhs : rhs;
}

#endif // COMMON_HEADER_H