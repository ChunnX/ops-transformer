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

// Deterministic CPU model: scalar calls enqueue MTE3 work; a blocked MTE3 wait
// does not block scalar execution. This models protocol ordering, not NPU timing.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <vector>

#define __aicore__
#define __gm__
using GM_ADDR = uint8_t *;
enum pipe_t {
    PIPE_MTE3
};

namespace AscendC {
enum CoreType {
    AIC,
    AIV
};
enum class HardEvent {
    S_MTE3
};
enum class PaddingMode {
    Normal
};
struct DataCopyExtParams {
    uint16_t blockCount;
    uint32_t blockLen;
    int64_t srcStride;
    int64_t dstStride;
    uint32_t reserved;
};
struct Pipeline {
    std::deque<std::function<bool()>> work;
    uint32_t sets = 0;
    uint32_t waits = 0;
    uint32_t consumed = 0;
    bool scalarEvent = false;
    int32_t *ownedSequence = nullptr;
};
struct Rank {
    std::vector<Pipeline> pipes;
    std::vector<uint32_t> arrived;
    explicit Rank(uint32_t cores)
        : pipes(cores),
          arrived(cores)
    {}
    bool Step(uint32_t core)
    {
        auto &queue = pipes[core].work;
        if (queue.empty() || !queue.front()()) {
            return false;
        }
        queue.pop_front();
        return true;
    }
    void Drain()
    {
        for (;;) {
            bool pending = false, progress = false;
            for (uint32_t core = 0; core < pipes.size(); ++core) {
                pending |= !pipes[core].work.empty();
                progress |= Step(core);
            }
            if (!pending) {
                return;
            }
            assert(progress); // missing Set / circular pipeline dependency
        }
    }
};
inline Rank *rank = nullptr;
inline uint32_t core = 0;
inline std::map<const int32_t *, uint32_t> pendingReads;
inline uint32_t scalarReads = 0;
inline Pipeline &Current()
{
    assert(rank != nullptr);
    return rank->pipes[core];
}
inline int32_t ReadGmByPassDCache(int32_t *address)
{
    assert(address == Current().ownedSequence); // no peer-ready polling
    ++scalarReads;
    return *address;
}
inline void WriteGmByPassDCache(int32_t *address, int32_t value)
{
    assert(address == Current().ownedSequence);
    *address = value; // epoch bookkeeping may precede quant completion
}
template <uint8_t mode, pipe_t pipe>
inline void CrossCoreSetFlag(uint16_t flag)
{
    static_assert(mode == 0 && pipe == PIPE_MTE3);
    assert(flag == 8);
    auto *r = rank;
    auto c = core;
    auto generation = ++Current().sets;
    Current().work.emplace_back([r, c, generation] {
        r->arrived[c] = generation;
        return true;
    });
}
template <uint8_t mode, pipe_t pipe>
inline void CrossCoreWaitFlag(uint16_t flag)
{
    static_assert(mode == 0 && pipe == PIPE_MTE3);
    assert(flag == 8);
    auto *r = rank;
    auto c = core;
    auto generation = ++Current().waits;
    assert(generation == Current().sets);
    Current().work.emplace_back([r, c, generation] {
        for (auto arrived : r->arrived) {
            if (arrived < generation) {
                return false;
            }
        }
        assert(++r->pipes[c].consumed == generation);
        return true;
    });
}
template <HardEvent event>
inline void SetFlag(uint32_t)
{
    static_assert(event == HardEvent::S_MTE3);
    assert(!Current().scalarEvent);
    Current().scalarEvent = true;
}
template <HardEvent event>
inline void WaitFlag(uint32_t)
{
    static_assert(event == HardEvent::S_MTE3);
    assert(Current().scalarEvent);
    Current().scalarEvent = false;
}
template <typename T>
struct LocalTensor {
    T *data = nullptr;
    uint32_t size = 0;
    T GetValue(uint32_t idx) const
    {
        assert(idx < size);
        return data[idx];
    }
    void SetValue(uint32_t idx, T value) const
    {
        assert(idx < size && pendingReads[data + idx] == 0);
        data[idx] = value;
    }
    LocalTensor operator[](uint32_t idx) const
    {
        assert(idx < size);
        return {data + idx, size - idx};
    }
};
template <typename T>
struct GlobalTensor {
    T *data = nullptr;
    void SetGlobalBuffer(T *p)
    {
        data = p;
    }
};
template <typename T, PaddingMode mode>
inline void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src, DataCopyExtParams copy)
{
    static_assert(mode == PaddingMode::Normal);
    assert(copy.blockLen == sizeof(T) && copy.srcStride == 0);
    assert(reinterpret_cast<uintptr_t>(src.data) % 32 == 0);
    auto sourceStride = 32U / sizeof(T);
    auto destStride = (copy.blockLen + copy.dstStride) / sizeof(T);
    std::vector<T> expected;
    for (uint32_t i = 0; i < copy.blockCount; ++i) {
        expected.push_back(src.GetValue(i * sourceStride));
        ++pendingReads[src.data + i * sourceStride];
    }
    Current().work.emplace_back([=] {
        for (uint32_t i = 0; i < copy.blockCount; ++i) {
            assert(src.GetValue(i * sourceStride) == expected[i]);
            dst.data[i * destStride] = src.GetValue(i * sourceStride);
            --pendingReads[src.data + i * sourceStride];
        }
        return true;
    });
}
} // namespace AscendC
#ifdef TEST_AIC
constexpr auto g_coreType = AscendC::AIC;
#else
constexpr auto g_coreType = AscendC::AIV;
#endif
