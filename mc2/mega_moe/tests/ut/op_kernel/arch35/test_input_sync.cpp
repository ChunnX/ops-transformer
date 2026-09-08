/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <iostream>
#include "../../../../op_kernel/arch35/common/mega_moe_input_sync.h"

using namespace MegaMoeImpl;
namespace {
constexpr uint32_t CORES = 4;
constexpr uint32_t RANKS = 3;
constexpr uint32_t MAX_EXPERTS = 257;
constexpr uint32_t COUNT_OFFSET = 256;
constexpr int32_t CANARY = 0x34567812;
struct alignas(64) Window {
    std::array<int32_t, 60 * 1024 / sizeof(int32_t)> words;
    Window()
    {
        words.fill(CANARY);
    }
    GM_ADDR Base()
    {
        return reinterpret_cast<GM_ADDR>(words.data());
    }
};
struct alignas(32) Counts {
    // Canary block before and after every source buffer.
    std::array<int32_t, (RANKS * MAX_EXPERTS + 2) * MTE_COUNT_UB_STRIDE> words;
    Counts()
    {
        words.fill(CANARY);
    }
    AscendC::LocalTensor<int32_t> Tensor()
    {
        return {words.data() + MTE_COUNT_UB_STRIDE, static_cast<uint32_t>(words.size()) - 2 * MTE_COUNT_UB_STRIDE};
    }
};
uint32_t RawCount(uint32_t launch, uint32_t source, uint32_t expert)
{
    // Include zero, maximum 24-bit count, and values sensitive to source/segment mixups.
    return expert % 7 == 0 ? 0 : (expert % 11 == 0 ? 0xFFFFFFU : launch * 100U + source * 10U + expert);
}
void TestLaunches(uint32_t seed, uint32_t expertsPerRank)
{
    std::array<Window, RANKS> windows;
    std::array<AscendC::Rank, RANKS> ranks{AscendC::Rank(CORES), AscendC::Rank(CORES), AscendC::Rank(CORES)};
    std::array<std::array<Counts, CORES>, RANKS> ub;
    std::array<GM_ADDR, RANKS> destinations;
    std::array<std::array<uint32_t, CORES>, RANKS> quant{}, index{}, independentScalar{};
    for (uint32_t r = 0; r < RANKS; ++r) {
        destinations[r] = windows[r].Base();
        for (uint32_t c = 0; c < CORES; ++c) {
            auto *slot = GetMteSyncCountAddress(windows[r].Base(), c);
            *slot = static_cast<int32_t>(seed);
            ranks[r].pipes[c].ownedSequence = slot;
        }
    }
    for (uint32_t launch = 1; launch <= 260; ++launch) {
        uint32_t total = RANKS * expertsPerRank;
        uint32_t sequence = seed + 2U * launch - 1U;
        uint32_t epoch = GetMteCountEpoch(static_cast<int32_t>(sequence));
        for (uint32_t r = 0; r < RANKS; ++r) {
            AscendC::rank = &ranks[r];
            for (uint32_t c = 0; c < CORES; ++c) {
                AscendC::core = c;
                auto &pipe = ranks[r].pipes[c];
                bool hasTokens = (launch + c) % 3 != 0;
                if (hasTokens) {
                    pipe.work.emplace_back([&, r, c, launch] {
                        quant[r][c] = launch;
                        return true;
                    });
                }
                auto seq = PublishMteQuantReady(windows[r].Base(), c);
                assert(static_cast<uint32_t>(seq) == sequence);
                assert(ranks[r].arrived[c] == launch - 1); // Set has not run; scalar did not drain MTE3
                uint32_t count = total / CORES + (c < total % CORES ? 1U : 0U);
                uint32_t begin = c * (total / CORES) + std::min(c, total % CORES);
                // Route writes precede the quant wait, even when another quant producer is delayed.
                pipe.work.emplace_back([&, r, c, launch] {
                    index[r][c] = launch;
                    return true;
                });
                auto tensor = ub[r][c].Tensor();
                for (uint32_t e = 0; e < count; ++e) {
                    tensor.SetValue(e * MTE_COUNT_UB_STRIDE, static_cast<int32_t>(RawCount(launch, r, begin + e)));
                }
                WaitForMteQuantReady(); // empty-token and no-expert cores must consume the event too
                pipe.work.emplace_back([&, r, c, launch] {
                    assert(index[r][c] == launch);
                    for (uint32_t peer = 0; peer < CORES; ++peer) {
                        if ((launch + peer) % 3 != 0) {
                            assert(quant[r][peer] == launch);
                        }
                    }
                    return true;
                });
                PublishMteExpertCounts(destinations.data(), COUNT_OFFSET, r, RANKS, expertsPerRank, begin, count, seq,
                                       tensor);
                independentScalar[r][c] = launch;
                assert(!pipe.work.empty()); // count publish returns with DMA still queued
            }
            // Hold the last quant producer. Other cores issue index and then stall only at the MTE3 wait.
            for (uint32_t c = 0; c < CORES - 1; ++c) {
                while (ranks[r].Step(c)) {
                }
                assert(index[r][c] == launch && ranks[r].pipes[c].consumed == launch - 1);
                assert(independentScalar[r][c] == launch);
            }
            ranks[r].Drain(); // retained input PIPE_ALL / SyncAll before source UB can be reused
        }
        for (uint32_t dst = 0; dst < RANKS; ++dst) {
            for (uint32_t e = 0; e < expertsPerRank; ++e) {
                for (uint32_t src = 0; src < RANKS; ++src) {
                    auto word = static_cast<uint32_t>(windows[dst].words[COUNT_OFFSET / 4 + e * RANKS + src]);
                    assert(word == ((epoch << 24U) | RawCount(launch, src, dst * expertsPerRank + e)));
                }
            }
            AscendC::rank = &ranks[dst];
            for (uint32_t c = 0; c < CORES; ++c) {
                AscendC::core = c;
                assert(ranks[dst].pipes[c].sets == launch && ranks[dst].pipes[c].consumed == launch);
                AscendC::WriteGmByPassDCache(ranks[dst].pipes[c].ownedSequence, static_cast<int32_t>(sequence + 1U));
            }
        }
    }
    for (uint32_t r = 0; r < RANKS; ++r) {
        for (uint32_t i = 0; i < windows[r].words.size(); ++i) {
            bool count = i >= COUNT_OFFSET / 4 && i < COUNT_OFFSET / 4 + RANKS * expertsPerRank;
            uint32_t offset = i * 4;
            bool sequence = offset >= MTE_SYNC_COUNT_OFFSET &&
                            offset < MTE_SYNC_COUNT_OFFSET + CORES * MTE_SYNC_COUNT_STRIDE &&
                            (offset - MTE_SYNC_COUNT_OFFSET) % MTE_SYNC_COUNT_STRIDE == 0;
            assert(count || sequence || windows[r].words[i] == CANARY);
        }
        for (uint32_t c = 0; c < CORES; ++c) {
            uint32_t total = RANKS * expertsPerRank;
            uint32_t count = total / CORES + (c < total % CORES ? 1U : 0U);
            for (uint32_t i = 0; i < ub[r][c].words.size(); ++i) {
                bool value =
                    i >= MTE_COUNT_UB_STRIDE && i < (count + 1) * MTE_COUNT_UB_STRIDE && i % MTE_COUNT_UB_STRIDE == 0;
                assert(value || ub[r][c].words[i] == CANARY);
            }
        }
    }
    for (const auto &[address, readers] : AscendC::pendingReads) {
        assert(readers == 0);
    }
}
} // namespace
int main()
{
    if constexpr (g_coreType == AscendC::AIC) {
        // No rank/pipeline/GM configured: any accidental AIC access fails immediately.
        assert(PublishMteQuantReady(nullptr, 0) == 0);
        WaitForMteQuantReady();
        PublishMteExpertCounts(nullptr, 0, 0, 1, 1, 0, 1, 1, {});
        std::cout << "PASS: AIC does not access epoch, UB, or AIV events\n";
    } else {
        for (uint32_t seed : {0U, 0x7FFFFFFEU, 0xFFFFFFFEU}) {
            for (uint32_t experts : {1U, 3U, 65U, MAX_EXPERTS}) {
                TestLaunches(seed, experts);
            }
        }
        assert(AscendC::scalarReads == 3120U * RANKS * CORES);
        std::cout << "PASS: 3120 launches, delayed/empty producers, index overlap, asynchronous count DMA, "
                     "cross-rank segments, zero/max counts, epoch/full wrap, UB/GM canaries and event pairing\n";
    }
}
