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
#include <limits>
#include <tuple>
#include <vector>

namespace {
constexpr uint32_t SENDERS = 32U;
struct Chunk {
    uint32_t rank;
    uint64_t begin;
    uint32_t count;
    bool last;
};
using Event = std::tuple<uint32_t, uint64_t, bool>; // peer, metadata row, completion flag

// CPU model of NextMetadataChunk; the oracle below independently enumerates the old per-peer schedule.
struct Schedule {
    const std::vector<uint64_t> &offsets;
    uint32_t local;
    uint32_t sender;
    uint32_t capacity;
    uint32_t peerOffset = 0U;
    uint64_t begin = 0U;
    uint64_t end = 0U;

    bool Next(Chunk &chunk)
    {
        uint32_t peers = static_cast<uint32_t>(offsets.size() - 1U);
        if (peerOffset == peers) { return false; }
        uint32_t rank = (sender + peerOffset) % peers;
        if (begin == end && rank != local && offsets.back() != 0U) {
            uint64_t count = offsets[rank + 1U] - offsets[rank];
            uint64_t base = count / SENDERS;
            uint64_t remainder = count % SENDERS;
            begin = offsets[rank] + base * sender + (sender < remainder ? sender : remainder);
            end = begin + base + (sender < remainder ? 1U : 0U);
        }
        uint64_t remaining = end - begin;
        uint32_t count = remaining < capacity ? static_cast<uint32_t>(remaining) : capacity;
        chunk = {rank, begin, count, count == remaining};
        begin += count;
        if (chunk.last) { ++peerOffset; begin = 0U; end = 0U; }
        return true;
    }
};

std::vector<Event> Expected(const Schedule &schedule)
{
    std::vector<Event> result;
    uint32_t peers = static_cast<uint32_t>(schedule.offsets.size() - 1U);
    for (uint32_t p = 0; p < peers; ++p) {
        uint32_t rank = (schedule.sender + p) % peers;
        if (rank != schedule.local) {
            uint64_t count = schedule.offsets[rank + 1U] - schedule.offsets[rank];
            uint64_t base = count / SENDERS;
            uint64_t remainder = count % SENDERS;
            uint64_t cursor = schedule.offsets[rank];
            for (uint32_t owner = 0U; owner < SENDERS; ++owner) {
                uint64_t length = base + (owner < remainder ? 1U : 0U);
                if (owner == schedule.sender) {
                    for (uint64_t i = 0U; i < length; ++i) { result.emplace_back(rank, cursor + i, false); }
                }
                cursor += length;
            }
        }
        result.emplace_back(rank, 0U, true);
    }
    return result;
}

void CheckPipeline(Schedule schedule)
{
    auto expected = Expected(schedule);
    std::vector<Event> actual;
    std::array<bool, 2> pending{false, false};
    Chunk chunks[2]{};
    auto prefetch = [&](uint32_t buffer) {
        assert(!pending[buffer]); // no overwrite or second set before the previous wait
        pending[buffer] = chunks[buffer].count != 0U;
    };
    uint32_t current = 0U;
    bool hasCurrent = schedule.Next(chunks[current]);
    if (hasCurrent) { prefetch(current); }
    while (hasCurrent) {
        uint32_t next = current ^ 1U;
        bool hasNext = schedule.Next(chunks[next]);
        if (hasNext) { prefetch(next); }
        const auto &chunk = chunks[current];
        if (chunk.count != 0U) {
            assert(pending[current]);
            pending[current] = false;
            for (uint32_t i = 0U; i < chunk.count; ++i) {
                actual.emplace_back(chunk.rank, chunk.begin + i, false);
            }
        }
        if (chunk.last) { actual.emplace_back(chunk.rank, 0U, true); }
        current = next;
        hasCurrent = hasNext;
    }
    assert(!pending[0] && !pending[1]);
    assert(actual == expected);
}

void CheckMaskPredicate()
{
    const std::array<int32_t, 9> ids = {std::numeric_limits<int32_t>::min(), -16777217, -1, 0, 1, 255,
                                      16777215, 16777217, std::numeric_limits<int32_t>::max()};
    for (int32_t id : ids) {
        bool oldMask = static_cast<float>(id) >= 0.0F;
        bool directMask = id >= 0;
        assert(oldMask == directMask);
    }
}

void CheckFullBatches()
{
    for (uint32_t payloadWqebb : {1U, 2U}) {
        for (uint32_t requests : {0U, 1U, 15U, 16U, 17U, 32U, 128U, 129U, 4096U}) {
            for (uint32_t flagPeriod : {1U, 7U, 33U}) {
                std::vector<uint32_t> batches;
                uint32_t pending = 0U;
                uint32_t total = 0U;
                auto append = [&](uint32_t count) {
                    if (pending + count > 256U) { batches.push_back(pending); pending = 0U; }
                    pending += count;
                    total += count;
                };
                for (uint32_t i = 0U; i < requests; ++i) {
                    append(payloadWqebb);
                    if ((i + 1U) % flagPeriod == 0U) { append(1U); }
                }
                append(1U); // final completion, including the empty-peer case
                if (pending != 0U) { batches.push_back(pending); }
                assert(batches.front() > 0U && batches.front() <= 256U);
                for (std::size_t i = 0U; i + 1U < batches.size(); ++i) {
                    assert(batches[i] >= 255U); // Only a two-WQEBB request can leave one unused block.
                }
                uint32_t committed = 0U;
                for (uint32_t batch : batches) {
                    assert(batch > 0U && batch <= 256U && batch < 4096U);
                    committed += batch;
                }
                assert(committed == total);
            }
        }
    }
}
} // namespace

int main()
{
    uint32_t cases = 0U;
    for (uint32_t peers : {1U, 2U, 3U, 8U, 16U, 33U, 64U, 1024U}) {
        for (uint32_t pattern = 0U; pattern < 4U; ++pattern) {
            std::vector<uint64_t> offsets(peers + 1U, 0U);
            for (uint32_t p = 0U; p < peers; ++p) {
                uint64_t count = pattern == 0U ? 0U : (pattern == 1U ? p % 3U : (p * 37U + 31U) % 257U);
                if (pattern == 3U && p == peers - 1U) { count = 131105U; } // multiple chunks plus a tail
                offsets[p + 1U] = offsets[p] + count;
            }
            for (uint32_t local : {0U, peers - 1U}) {
                for (uint32_t sender = 0U; sender < SENDERS; ++sender) {
                    for (uint32_t capacity : {1U, 7U, 1024U, 2048U}) {
                        CheckPipeline({offsets, local, sender, capacity});
                        ++cases;
                    }
                }
            }
        }
    }
    CheckMaskPredicate();
    CheckFullBatches();
    std::cout << "PASS: " << cases << " Combine schedules, ping-pong events, completion order, mask, full batches\n";
}
