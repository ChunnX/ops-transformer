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
#include <cassert>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <vector>

namespace {
constexpr uint32_t CHUNK_TOKENS = (160U * 1024U / 2U / sizeof(int32_t)) / 6U * 6U;
struct Range { uint32_t start; uint32_t count; };
struct Chunk { uint32_t rank; uint32_t start; uint32_t count; bool first; bool last; };
using Event = std::tuple<uint32_t, uint32_t, bool>;

// Scheduling model; range correctness is tested separately in test_dispatch_schedule.cpp.
struct Schedule {
    const std::vector<Range> &ranges;
    uint32_t local;
    uint32_t sender;
    uint32_t peerOffset = 0U;
    uint32_t start = 0U;
    uint32_t remaining = 0U;
    bool Next(Chunk &chunk)
    {
        uint32_t peers = static_cast<uint32_t>(ranges.size());
        if (peerOffset == peers) { return false; }
        uint32_t rank = (sender + peerOffset) % peers;
        bool first = remaining == 0U;
        if (first && rank != local) { start = ranges[rank].start; remaining = ranges[rank].count; }
        uint32_t count = remaining < CHUNK_TOKENS ? remaining : CHUNK_TOKENS;
        chunk = {rank, start, count, first, count == remaining};
        start += count;
        remaining -= count;
        if (chunk.last) { ++peerOffset; start = 0U; }
        return true;
    }
};

std::vector<Event> Expected(const Schedule &schedule)
{
    std::vector<Event> result;
    uint32_t peers = static_cast<uint32_t>(schedule.ranges.size());
    for (uint32_t p = 0U; p < peers; ++p) {
        uint32_t rank = (schedule.sender + p) % peers;
        if (rank != schedule.local) {
            auto range = schedule.ranges[rank];
            for (uint32_t row = 0U; row < range.count; ++row) {
                result.emplace_back(rank, range.start + row, false);
            }
        }
        result.emplace_back(rank, 0U, true);
    }
    return result;
}

void Consume(const Chunk &chunk, uint32_t local, bool &pending, uint32_t &boundPeer, std::vector<Event> &output)
{
    if (chunk.rank != local) {
        if (chunk.first) { boundPeer = chunk.rank; }
        assert(boundPeer == chunk.rank);
    } else {
        assert(chunk.count == 0U && chunk.first && chunk.last);
    }
    if (chunk.count != 0U) { assert(pending); pending = false; }
    assert(chunk.last || chunk.count % 6U == 0U);
    for (uint32_t offset = 0U; offset < chunk.count; offset += 6U) {
        uint32_t slots = chunk.count - offset < 6U ? chunk.count - offset : 6U;
        assert(slots * 2U <= 12U);
        for (uint32_t slot = 0U; slot < slots; ++slot) {
            output.emplace_back(chunk.rank, chunk.start + offset + slot, false);
        }
    }
    if (chunk.last) { output.emplace_back(chunk.rank, 0U, true); }
}

void CheckPipeline(Schedule schedule)
{
    auto expected = Expected(schedule);
    std::vector<Event> actual;
    std::array<bool, 2> pending{false, false};
    Chunk chunks[2]{};
    auto prefetch = [&](uint32_t buffer) {
        assert(!pending[buffer]);
        assert((chunks[buffer].count * sizeof(int32_t) + 31U) / 32U * 32U <= 80U * 1024U);
        pending[buffer] = chunks[buffer].count != 0U;
    };
    uint32_t current = 0U;
    uint32_t boundPeer = 0U;
    bool hasCurrent = schedule.Next(chunks[current]);
    if (hasCurrent) { prefetch(current); }
    while (hasCurrent) {
        uint32_t next = current ^ 1U;
        bool hasNext = schedule.Next(chunks[next]);
        if (hasNext) { prefetch(next); }
        Consume(chunks[current], schedule.local, pending[current], boundPeer, actual);
        current = next;
        hasCurrent = hasNext;
    }
    assert(!pending[0] && !pending[1]);
    assert(actual == expected);
}
} // namespace

int main()
{
    uint32_t cases = 0U;
    for (uint32_t peers : {1U, 2U, 3U, 8U, 33U, 1024U}) {
        for (uint32_t count : {0U, 1U, 6U, 7U, CHUNK_TOKENS - 1U, CHUNK_TOKENS, CHUNK_TOKENS + 1U, 40961U}) {
            std::vector<Range> ranges(peers);
            for (uint32_t p = 0U; p < peers; ++p) {
                ranges[p] = {p * 17U, p == peers - 1U ? count : p % 5U};
            }
            for (uint32_t local : {0U, peers - 1U}) {
                for (uint32_t sender = 0U; sender < 32U; ++sender) {
                    CheckPipeline({ranges, local, sender});
                    ++cases;
                }
            }
        }
    }
    std::cout << "PASS: " << cases << " Dispatch prefetch schedules, event balance, SGE slot coverage\n";
}
