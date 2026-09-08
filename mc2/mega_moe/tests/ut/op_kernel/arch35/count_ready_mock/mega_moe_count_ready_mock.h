/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

// CPU register semantics and deferred MTE2/V execution for the production count reader.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>

#define __aicore__
#define __simd_vf__
#define __ubuf__
#define __gm__

namespace AscendC {
enum CoreType {
    AIC,
    AIV
};
enum class CMPMODE {
    NE
};
enum class HardEvent {
    V_MTE2,
    MTE2_V,
    V_S
};
constexpr uint32_t GetVecLen()
{
    return 256U;
}
namespace Reg {
constexpr uint32_t LANES = GetVecLen() / sizeof(uint32_t);
enum class MaskPattern {
    ALL,
    VL1
};
using MaskReg = std::array<bool, LANES>;
template <typename T>
using RegTensor = std::array<T, LANES>;
template <typename T, MaskPattern pattern>
MaskReg CreateMask()
{
    MaskReg mask{};
    std::fill_n(mask.begin(), pattern == MaskPattern::ALL ? LANES : 1U, true);
    return mask;
}
template <typename T>
MaskReg UpdateMask(uint32_t &count)
{
    MaskReg mask{};
    auto valid = std::min(count, LANES);
    std::fill_n(mask.begin(), valid, true);
    count -= valid;
    return mask;
}
template <typename T>
void Duplicate(RegTensor<T> &dst, T value, MaskReg &mask)
{
    for (uint32_t i = 0; i < LANES; ++i)
        if (mask[i])
            dst[i] = value;
}
template <typename T>
void LoadAlign(RegTensor<T> &dst, T *src)
{
    std::copy_n(src, LANES, dst.begin());
}
template <typename T>
void ShiftRights(RegTensor<T> &dst, RegTensor<T> &src, uint32_t shift, MaskReg &mask)
{
    for (uint32_t i = 0; i < LANES; ++i)
        if (mask[i])
            dst[i] = src[i] >> shift;
}
template <typename T, CMPMODE mode>
void Compares(MaskReg &dst, RegTensor<T> &src, T value, MaskReg &mask)
{
    static_assert(mode == CMPMODE::NE);
    for (uint32_t i = 0; i < LANES; ++i)
        dst[i] = mask[i] && src[i] != value;
}
template <typename T>
void Select(RegTensor<T> &dst, RegTensor<T> &yes, RegTensor<T> &no, MaskReg &mask)
{
    for (uint32_t i = 0; i < LANES; ++i)
        dst[i] = mask[i] ? yes[i] : no[i];
}
template <typename T>
void And(RegTensor<T> &dst, RegTensor<T> &a, RegTensor<T> &b, MaskReg &mask)
{
    for (uint32_t i = 0; i < LANES; ++i)
        if (mask[i])
            dst[i] = a[i] & b[i];
}
template <typename T>
void StoreAlign(T *dst, RegTensor<T> &src, MaskReg &mask)
{
    for (uint32_t i = 0; i < LANES; ++i)
        if (mask[i])
            dst[i] = src[i];
}
template <typename T>
void ReduceMax(RegTensor<T> &dst, RegTensor<T> &src, MaskReg &mask)
{
    T result = 0;
    for (uint32_t i = 0; i < LANES; ++i)
        if (mask[i])
            result = std::max(result, src[i]);
    dst.fill(0xDEADBEEF);
    dst[0] = result;
}
} // namespace Reg

struct Pipeline {
    uint32_t copies = 0;
    uint32_t scalarReads = 0;
    uint32_t clockReads = 0;
    int64_t cycle = 0;
    int64_t lastResultCycle = 0;
    uint64_t copiedBytes = 0;
    bool canCopy = false;
    bool copyComplete = false;
    bool vectorComplete = false;
    bool resultRead = false;
    int32_t *resultAddress = nullptr;
    std::array<bool, 3> eventSet{};
    std::function<void()> mte2;
    std::function<void()> vector;
    std::function<void(uint32_t)> beforeCopy;
};
inline Pipeline pipeline;

template <typename T>
struct LocalTensor {
    T *data;
    T *GetPhyAddr() const
    {
        return data;
    }
    T GetValue(uint32_t index) const
    {
        assert(index == 0 && data == pipeline.resultAddress);
        assert(pipeline.vectorComplete && !pipeline.resultRead);
        pipeline.resultRead = true;
        pipeline.lastResultCycle = pipeline.cycle;
        ++pipeline.scalarReads;
        return data[index];
    }
};
template <typename T>
struct GlobalTensor {
    T *data;
};
struct DataCopyExtParams {
    uint16_t blockCount;
    uint32_t blockLen;
    int64_t srcStride;
    int64_t dstStride;
    uint32_t reserved;
};
template <typename T>
struct DataCopyPadExtParams {
    bool isPad;
    uint8_t left;
    uint8_t right;
    T value;
};
template <typename T>
void DataCopyPad(LocalTensor<T> dst, GlobalTensor<T> src, DataCopyExtParams copy, DataCopyPadExtParams<T>)
{
    assert(pipeline.canCopy && !pipeline.mte2 && !pipeline.vector);
    assert(copy.blockCount == 1 && copy.srcStride == 0 && copy.dstStride == 0);
    if (pipeline.copies != 0) {
        assert(pipeline.resultRead && pipeline.cycle - pipeline.lastResultCycle >= 500);
    } else {
        assert(pipeline.clockReads == 0); // first snapshot has no backoff
    }
    assert(++pipeline.copies <= 8); // fail instead of hanging on a broken retry
    pipeline.copiedBytes += copy.blockLen;
    pipeline.canCopy = false;
    pipeline.copyComplete = pipeline.vectorComplete = pipeline.resultRead = false;
    pipeline.mte2 = [=] {
        if (pipeline.beforeCopy)
            pipeline.beforeCopy(pipeline.copies);
        std::copy_n(src.data, copy.blockLen / sizeof(T), dst.data);
    };
}
template <HardEvent event>
void SetFlag(uint32_t id)
{
    assert(id == 2 && !pipeline.eventSet[static_cast<int>(event)]);
    pipeline.eventSet[static_cast<int>(event)] = true;
}
template <HardEvent event>
void WaitFlag(uint32_t id)
{
    assert(id == 2 && pipeline.eventSet[static_cast<int>(event)]);
    pipeline.eventSet[static_cast<int>(event)] = false;
    if constexpr (event == HardEvent::V_MTE2) {
        assert(!pipeline.vector);
        assert(pipeline.copies == 0 || pipeline.vectorComplete);
        pipeline.canCopy = true;
    } else if constexpr (event == HardEvent::MTE2_V) {
        assert(pipeline.mte2);
        pipeline.mte2();
        pipeline.mte2 = {};
        pipeline.copyComplete = true;
    } else {
        assert(pipeline.vector);
        pipeline.vector();
        pipeline.vector = {};
        pipeline.vectorComplete = true;
    }
}
inline int64_t GetSystemCycle()
{
    assert(pipeline.resultRead);
    ++pipeline.clockReads;
    pipeline.cycle += 100;
    return pipeline.cycle;
}
} // namespace AscendC

template <auto function, typename... Args>
void asc_vf_call(Args... args)
{
    auto &pipeline = AscendC::pipeline;
    assert(pipeline.copyComplete && !pipeline.vector);
    pipeline.vector = [=] { function(args...); };
}
#ifdef TEST_AIC
constexpr auto g_coreType = AscendC::AIC;
#else
constexpr auto g_coreType = AscendC::AIV;
#endif
