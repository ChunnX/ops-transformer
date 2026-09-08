/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_INPUT_PREPARE_MOCK_H
#define MEGA_MOE_INPUT_PREPARE_MOCK_H

#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <vector>

#define __aicore__
#define __gm__
using GM_ADDR = uint8_t *;
enum pipe_t {
    PIPE_S,
    PIPE_MTE3,
    PIPE_FIX,
    PIPE_M,
    PIPE_COUNT
};
namespace AscendC {
enum Core {
    AIC,
    AIV
};
} // namespace AscendC
#ifdef TEST_AIC
#define g_coreType AscendC::AIC
#else
#define g_coreType AscendC::AIV
#endif

namespace InputPrepareMock {
struct Operation {
    uint32_t scalarDependency;
    std::function<bool()> execute;
};
struct Actor {
    std::array<std::deque<Operation>, PIPE_COUNT> pipes;
    uint32_t scalarIssued = 0;
    uint32_t scalarDone = 0;
    std::array<uint32_t, 32> localFlags{};
    std::array<uint32_t, 16> globalFlags{};
};
struct Machine {
    uint32_t blocks;
    uint32_t current = 0;
    std::vector<Actor> actors;
    std::array<uint32_t, 16> globalArrivals{};
    explicit Machine(uint32_t count)
        : blocks(count),
          actors(count * 3)
    {}
    void Enqueue(pipe_t pipe, std::function<bool()> action)
    {
        auto &actor = actors[current];
        actor.pipes[pipe].push_back({actor.scalarIssued, std::move(action)});
        if (pipe == PIPE_S) {
            ++actor.scalarIssued;
        }
    }
    void Run()
    {
        bool progress;
        do {
            progress = false;
            for (auto &actor : actors) {
                for (uint32_t pipe = 0; pipe < PIPE_COUNT; ++pipe) {
                    auto &queue = actor.pipes[pipe];
                    if (queue.empty() || actor.scalarDone < queue.front().scalarDependency) {
                        continue;
                    }
                    if (queue.front().execute()) {
                        queue.pop_front();
                        actor.scalarDone += pipe == PIPE_S;
                        progress = true;
                    }
                }
            }
        } while (progress);
    }
    void AssertDrained() const
    {
        for (const auto &actor : actors) {
            assert(actor.scalarDone == actor.scalarIssued);
            for (const auto &queue : actor.pipes) {
                assert(queue.empty());
            }
            for (auto count : actor.localFlags) {
                assert(count == 0);
            }
            for (auto count : actor.globalFlags) {
                assert(count == 0);
            }
        }
        for (auto count : globalArrivals) {
            assert(count == 0);
        }
    }
};
inline Machine *machine;
inline void Event(uint8_t mode, pipe_t pipe, uint16_t flag, bool set)
{
    assert(mode == 0 || mode == 4);
    auto &sim = *machine;
    uint32_t actor = sim.current;
    bool cube = actor < sim.blocks;
    assert(flag < (cube && mode == 4 ? 32 : 16));
    sim.Enqueue(pipe, [&sim, actor, cube, mode, flag, set]() {
        if (!set) {
            auto &counter = mode == 4 ? sim.actors[actor].localFlags[flag] : sim.actors[actor].globalFlags[flag];
            if (counter == 0) {
                return false;
            }
            --counter;
        } else if (mode == 4) {
            uint32_t target = cube ? sim.blocks + actor * 2 + flag / 16 : (actor - sim.blocks) / 2;
            uint32_t targetFlag = cube ? flag % 16 : flag + ((actor - sim.blocks) % 2) * 16;
            ++sim.actors[target].localFlags[targetFlag];
        } else {
            assert(cube);
            if (++sim.globalArrivals[flag] == sim.blocks) {
                sim.globalArrivals[flag] = 0;
                for (uint32_t index = 0; index < sim.blocks; ++index) {
                    ++sim.actors[index].globalFlags[flag];
                }
            }
        }
        return true;
    });
}
} // namespace InputPrepareMock

namespace AscendC {
template <uint8_t Mode, pipe_t Pipe>
void CrossCoreSetFlag(uint16_t flag)
{
    static_assert(Mode != 0 || Pipe != PIPE_S); // 950 ffts_cross_core_sync cannot publish on S.
    InputPrepareMock::Event(Mode, Pipe, flag, true);
}
template <uint8_t Mode, pipe_t Pipe>
void CrossCoreWaitFlag(uint16_t flag)
{
    InputPrepareMock::Event(Mode, Pipe, flag, false);
}
template <typename T>
struct LocalTensor {
    T *data;
    LocalTensor operator[](uint64_t offset) const
    {
        return {data + offset};
    }
    template <typename U>
    LocalTensor<U> ReinterpretCast() const
    {
        return {reinterpret_cast<U *>(data)};
    }
};
template <typename T>
struct GlobalTensor {
    T *data;
    void SetGlobalBuffer(T *address)
    {
        data = address;
    }
    GlobalTensor operator[](uint64_t offset) const
    {
        return {data + offset};
    }
};
struct DataCopyExtParams {
    uint16_t blocks;
    uint32_t bytes;
    uint32_t srcStride;
    uint32_t dstStride;
    uint32_t reserved;
};
template <typename T>
void DataCopyPad(GlobalTensor<T> dst, LocalTensor<T> src, DataCopyExtParams params)
{
    assert(params.blocks == 1 && params.srcStride == 0 && params.dstStride == 0);
    InputPrepareMock::machine->Enqueue(PIPE_MTE3, [=]() {
        // Read UB at execution, not submission, to expose buffer lifetime bugs.
        std::memcpy(dst.data, src.data, params.bytes);
        return true;
    });
}
} // namespace AscendC
#endif
