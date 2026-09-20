/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Standalone CPU test of the exact interval helpers used by the device kernel.
#define __aicore__
#include "../../../op_kernel/moe_ep_dispatch_schedule.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <random>
#include <vector>

using MoeEpDispatchSchedule::Overlap;
using MoeEpDispatchSchedule::Owner;
using MoeEpDispatchSchedule::Partition;

void CheckReadyDependencies(uint32_t bs, uint32_t prepareCount)
{
    std::array<uint32_t, 32> published{};
    for (uint32_t prepare = 0; prepare < prepareCount; ++prepare) {
        auto range = Partition(bs, prepareCount, prepare);
        auto receivers = Overlap(bs, 32, range.begin, range.end);
        for (uint32_t sender = receivers.begin; sender < receivers.end; ++sender) {
            ++published[sender];
        }
    }
    uint32_t covered = 0;
    for (uint32_t sender = 0; sender < 32; ++sender) {
        auto range = Partition(bs, 32, sender);
        auto producers = Overlap(bs, prepareCount, range.begin, range.end);
        uint32_t expected = 0;
        for (uint32_t p = 0; p < prepareCount; ++p) {
            auto other = Partition(bs, prepareCount, p);
            bool intersects = std::max(range.begin, other.begin) < std::min(range.end, other.end);
            assert(intersects == (p >= producers.begin && p < producers.end));
            expected += intersects ? 1U : 0U;
        }
        assert(published[sender] == expected);
        assert(range.begin == covered);
        assert(range.end - range.begin == bs / 32 + (sender < bs % 32 ? 1U : 0U));
        covered = range.end;
    }
    assert(covered == bs);
}

// Model the kernel's prefix plus boundary-producer search; the oracle scans the full sorted peer list.
uint32_t Boundary(uint32_t bs, uint32_t prepareCount, uint32_t token,
                  const std::vector<uint32_t> &indices, const std::vector<uint32_t> &counts,
                  MoeEpDispatchSchedule::Range readyProducers)
{
    uint32_t prepare = token == bs ? prepareCount : Owner(bs, prepareCount, token);
    uint32_t base = 0;
    for (uint32_t i = 0; i < prepare; ++i) {
        base += counts[i];
    }
    if (token == bs || token == Partition(bs, prepareCount, prepare).begin) {
        return base;
    }
    assert(prepare >= readyProducers.begin && prepare < readyProducers.end);
    auto first = indices.begin() + base;
    return static_cast<uint32_t>(std::lower_bound(first, first + counts[prepare], token) - indices.begin());
}

MoeEpDispatchSchedule::Range SharedPrefixRange(uint32_t bs, uint32_t prepareCount,
    MoeEpDispatchSchedule::Range range, const std::vector<uint32_t> &indices,
    const std::vector<uint32_t> &counts, MoeEpDispatchSchedule::Range ready)
{
    uint32_t first = range.begin == bs ? prepareCount : Owner(bs, prepareCount, range.begin);
    uint32_t last = range.end == bs ? prepareCount : Owner(bs, prepareCount, range.end);
    uint32_t base = 0;
    for (uint32_t row = 0; row < first; ++row) { base += counts[row]; }
    auto boundary = [&](uint32_t token, uint32_t prepare, uint32_t prefix) {
        if (token == bs || token == Partition(bs, prepareCount, prepare).begin) { return prefix; }
        assert(prepare >= ready.begin && prepare < ready.end);
        auto begin = indices.begin() + prefix;
        return static_cast<uint32_t>(std::lower_bound(begin, begin + counts[prepare], token) - indices.begin());
    };
    uint32_t begin = boundary(range.begin, first, base);
    if (range.begin == range.end) { return {begin, begin}; }
    for (uint32_t row = first; row < last; ++row) { base += counts[row]; }
    return {begin, boundary(range.end, last, base)};
}

void CheckRouting(uint32_t bs, uint32_t prepareCount, uint32_t peers, std::mt19937 &rng)
{
    std::vector<std::vector<uint32_t>> indices(peers);
    for (uint32_t token = 0; token < bs; ++token) {
        std::vector<bool> hits(peers);
        for (uint32_t k = 0; k < 6; ++k) {
            hits[rng() % peers] = true;
        }
        for (uint32_t peer = 0; peer < peers; ++peer) {
            if (hits[peer]) { indices[peer].push_back(token); }
        }
    }
    for (const auto &peer : indices) {
        std::vector<uint32_t> counts(prepareCount);
        for (uint32_t token : peer) { ++counts[Owner(bs, prepareCount, token)]; }
        uint32_t previous = 0;
        for (uint32_t sender = 0; sender < 32; ++sender) {
            auto range = Partition(bs, 32, sender);
            auto ready = Overlap(bs, prepareCount, range.begin, range.end);
            uint32_t begin = Boundary(bs, prepareCount, range.begin, peer, counts, ready);
            uint32_t end = Boundary(bs, prepareCount, range.end, peer, counts, ready);
            auto merged = SharedPrefixRange(bs, prepareCount, range, peer, counts, ready);
            assert(merged.begin == begin && merged.end == end);
            assert(begin == static_cast<uint32_t>(std::lower_bound(peer.begin(), peer.end(), range.begin) - peer.begin()));
            assert(end == static_cast<uint32_t>(std::lower_bound(peer.begin(), peer.end(), range.end) - peer.begin()));
            assert(begin == previous);
            previous = end;
        }
        assert(previous == peer.size());
    }
}

int main()
{
    std::mt19937 rng(20260919);
    for (uint32_t prepare = 1; prepare <= 68; ++prepare) {
        for (uint32_t bs = 0; bs <= 140; ++bs) { CheckReadyDependencies(bs, prepare); }
        for (uint32_t bs : {0U, 1U, 7U, 31U, 32U, 33U, 4095U, 4096U, 4097U}) {
            CheckReadyDependencies(bs, prepare);
            for (uint32_t peers : {2U, 4U, 8U, 16U, 33U}) { CheckRouting(bs, prepare, peers, rng); }
        }
    }
    std::cout << "PASS: independent 32-way send ranges, exact ready intersections, and peer slot coverage\n";
}
