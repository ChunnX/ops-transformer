/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Standalone CPU schedule/layout model, not an AscendC DMA or synchronization simulator.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
using Row = std::array<int32_t, 5>;
constexpr int32_t GUARD = 0x12345678;
constexpr uint32_t UB_ROW_ELEMENTS = 8U;
constexpr uint32_t PACKED_CHUNK_ROWS = 128U;
constexpr uint32_t MAX_CHUNK_ROWS = PACKED_CHUNK_ROWS * sizeof(Row) / 32U;
struct Range { uint32_t begin; uint32_t end; };
struct Case {
    uint32_t bs;
    uint32_t k;
    uint32_t cores;
    std::vector<Row> metadata;
    int32_t localBegin;
    int32_t localEnd;
    int32_t actual;
    uint32_t capacity;
};

void Require(bool condition, const char *message)
{
    if (!condition) { throw std::runtime_error(message); }
}

Range Split(uint32_t total, uint32_t cores, uint32_t core)
{
    uint32_t base = total / cores;
    uint32_t remainder = total % cores;
    uint32_t begin = base * core + std::min(core, remainder);
    return {begin, begin + base + (core < remainder ? 1U : 0U)};
}

uint32_t Actual(const Case &test)
{
    return std::min(static_cast<uint32_t>(std::max(test.actual, 0)), test.capacity);
}

Range LocalRange(const Case &test)
{
    uint32_t begin = std::min(static_cast<uint32_t>(std::max(test.localBegin, 0)), Actual(test));
    uint32_t end = std::min(static_cast<uint32_t>(std::max(test.localEnd, 0)), Actual(test));
    return {begin, std::max(begin, end)};
}

uint32_t OldMetadataBytes(uint32_t actual)
{
    uint32_t packedRows = std::max(1U, std::min(actual, PACKED_CHUNK_ROWS));
    return (packedRows * static_cast<uint32_t>(sizeof(Row)) + 31U) / 32U * 32U;
}

uint32_t ChunkRows(uint32_t actual, uint32_t ownRows)
{
    return std::max(1U, std::min(ownRows, OldMetadataBytes(actual) / 32U));
}

// Independent oracle: old output-owner scan, not the shared row partition/scatter algorithm.
std::vector<int32_t> Reference(const Case &test)
{
    std::vector<int32_t> expected(static_cast<size_t>(test.bs) * test.k, -1);
    Range local = LocalRange(test);
    uint32_t cursor = 0U;
    for (uint32_t core = 0U; core < test.cores; ++core) {
        uint32_t count = test.bs / test.cores + (core < test.bs % test.cores ? 1U : 0U);
        for (uint32_t row = local.begin; row < local.end; ++row) {
            const Row &r = test.metadata[row];
            if (r[1] < static_cast<int32_t>(cursor) || r[1] >= static_cast<int32_t>(cursor + count) ||
                r[2] < 0 || r[2] >= static_cast<int32_t>(test.k) || r[4] < 0 ||
                static_cast<uint32_t>(r[4]) >= Actual(test)) { continue; }
            expected[static_cast<size_t>(r[1]) * test.k + r[2]] = r[4];
        }
        cursor += count;
    }
    Require(cursor == test.bs, "reference output coverage");
    return expected;
}

Case MakeCase(uint32_t bs, uint32_t k, uint32_t cores, uint32_t density, std::mt19937 &rng)
{
    Case test{bs, k, cores, {}, 2, 2, 0, 0U};
    std::vector<Row> local;
    for (uint32_t token = 0U; token < bs; ++token) {
        for (uint32_t topk = 0U; topk < k; ++topk) {
            if (density != 0U && (token * k + topk) % density == 0U) {
                local.push_back({3, static_cast<int32_t>(token), static_cast<int32_t>(topk), 0, 0});
            }
        }
    }
    std::shuffle(local.begin(), local.end(), rng); // no token ordering assumption
    test.metadata.assign(2U, Row{0, 0, 0, 0, 0}); // valid-looking rows outside local-rank interval
    test.metadata.insert(test.metadata.end(), local.begin(), local.end());
    for (Row invalid : {Row{3, -1, 0, 0, 0}, Row{3, static_cast<int32_t>(bs), 0, 0, 0},
                        Row{3, 0, -1, 0, 0}, Row{3, 0, static_cast<int32_t>(k), 0, 0}}) {
        test.metadata.push_back(invalid);
    }
    test.localEnd = static_cast<int32_t>(test.metadata.size());
    test.metadata.insert(test.metadata.end(), 3U, Row{4, 0, 0, 0, 0});
    test.actual = static_cast<int32_t>(test.metadata.size());
    test.capacity = static_cast<uint32_t>(test.metadata.size());
    for (uint32_t row = 0U; row < test.metadata.size(); ++row) {
        test.metadata[row][4] = static_cast<int32_t>(row);
    }
    return test;
}

void Initialize(const Case &test, std::vector<int32_t> &workspace, const std::vector<uint32_t> &order)
{
    std::vector<uint32_t> writers(static_cast<size_t>(test.bs) * test.k, 0U);
    for (uint32_t core : order) {
        Range tokens = Split(test.bs, test.cores, core);
        for (size_t key = static_cast<size_t>(tokens.begin) * test.k;
             key < static_cast<size_t>(tokens.end) * test.k; ++key) {
            workspace[key + 1U] = -1;
            ++writers[key];
        }
    }
    Require(std::all_of(writers.begin(), writers.end(), [](uint32_t n) { return n == 1U; }),
            "initialization ownership must cover each key once");
}

void ScatterChunk(const Case &test, Range chunk, std::vector<int32_t> &workspace,
                  std::vector<uint32_t> &visits, std::vector<uint32_t> &writers)
{
    // DataCopyPad: packed five-int GM rows become independent 32-byte-aligned UB rows.
    alignas(32) std::array<int32_t, MAX_CHUNK_ROWS * UB_ROW_ELEMENTS> ub{};
    uint32_t count = chunk.end - chunk.begin;
    Require(count <= MAX_CHUNK_ROWS, "chunk exceeds padded metadata UB capacity");
    for (uint32_t row = 0U; row < count; ++row) {
        std::copy(test.metadata[chunk.begin + row].begin(), test.metadata[chunk.begin + row].end(),
                  ub.begin() + row * UB_ROW_ELEMENTS);
        ++visits[chunk.begin + row];
        ub[row * UB_ROW_ELEMENTS] = ub[row * UB_ROW_ELEMENTS + 4U];
    }
    for (uint32_t row = 0U; row < count; ++row) {
        size_t offset = static_cast<size_t>(row) * UB_ROW_ELEMENTS;
        int32_t token = ub[offset + 1U];
        int32_t topk = ub[offset + 2U];
        int32_t recv = ub[offset];
        if (token < 0 || static_cast<uint32_t>(token) >= test.bs || topk < 0 ||
            static_cast<uint32_t>(topk) >= test.k || recv < 0 ||
            static_cast<uint32_t>(recv) >= Actual(test)) { continue; }
        size_t key = static_cast<size_t>(token) * test.k + topk;
        Require(++writers[key] == 1U, "valid metadata must have unique (token, topk) keys");
        Require(reinterpret_cast<uintptr_t>(&ub[offset]) % 32U == 0U, "unaligned UB scatter source");
        std::memcpy(&workspace[key + 1U], &ub[offset], sizeof(int32_t)); // exact four-byte GM write
    }
}

void BuildShared(const Case &test, std::vector<int32_t> &workspace, std::mt19937 &rng)
{
    std::vector<uint32_t> order(test.cores);
    std::iota(order.begin(), order.end(), 0U);
    std::shuffle(order.begin(), order.end(), rng);
    Initialize(test, workspace, order); // all initialization completes before any scatter
    Range local = LocalRange(test);
    std::vector<uint32_t> visits(test.metadata.size(), 0U);
    std::vector<uint32_t> writers(static_cast<size_t>(test.bs) * test.k, 0U);
    std::vector<Range> chunks;
    for (uint32_t core : order) {
        Range rows = Split(local.end - local.begin, test.cores, core);
        uint32_t chunkRows = ChunkRows(Actual(test), rows.end - rows.begin);
        Require(chunkRows * 32U <= OldMetadataBytes(Actual(test)), "metadata UB grew beyond old budget");
        for (uint32_t begin = local.begin + rows.begin; begin < local.begin + rows.end; begin += chunkRows) {
            chunks.push_back({begin, std::min(begin + chunkRows, local.begin + rows.end)});
        }
    }
    std::shuffle(chunks.begin(), chunks.end(), rng); // vary inter-core completion order
    for (Range chunk : chunks) { ScatterChunk(test, chunk, workspace, visits, writers); }
    for (uint32_t row = 0U; row < visits.size(); ++row) {
        Require(visits[row] == (row >= local.begin && row < local.end ? 1U : 0U), "metadata scan coverage");
    }
    Require(workspace.front() == GUARD && workspace.back() == GUARD, "workspace overrun");
    auto expected = Reference(test); // existing completion barrier precedes all slice reads
    for (uint32_t core : order) {
        Range tokens = Split(test.bs, test.cores, core);
        for (size_t key = static_cast<size_t>(tokens.begin) * test.k;
             key < static_cast<size_t>(tokens.end) * test.k; ++key) {
            Require(workspace[key + 1U] == expected[key], "shared index differs from old per-core lookup");
        }
    }
}

void CheckBoundaryVariants(Case test, std::vector<int32_t> &workspace, std::mt19937 &rng)
{
    Case modified = test;
    for (size_t row = 2U; row < static_cast<size_t>(test.localEnd); row += 2U) {
        modified.metadata[row][4] = row % 4U == 0U ? -1 : test.actual;
    }
    BuildShared(modified, workspace, rng); // invalid recvXIdx must remain -1
    modified = test;
    modified.capacity /= 2U;
    BuildShared(modified, workspace, rng); // capacity clamps actual and local end
    modified = test;
    modified.localEnd = modified.localBegin;
    BuildShared(modified, workspace, rng); // valid empty local rank, reuse dirty workspace
    modified.localEnd = 0;
    BuildShared(modified, workspace, rng); // preserve old empty/reversed-range behavior
    modified = test;
    modified.actual = -1;
    BuildShared(modified, workspace, rng);
}

uint64_t WorkspaceBytes(uint32_t bs, uint32_t k)
{
    return (static_cast<uint64_t>(bs) * k * sizeof(int32_t) + 511U) / 512U * 512U;
}

void CheckUbBudget()
{
    for (uint32_t actual = 0U; actual <= 512U; ++actual) {
        for (uint32_t ownRows = 0U; ownRows <= actual; ++ownRows) {
            Require(ChunkRows(actual, ownRows) * 32U <= OldMetadataBytes(actual), "metadata UB regression");
        }
    }
    // P=8, 56 AIV, K32, H8192, weights: old code fits one input/output/weight group.
    for (uint32_t totalUb : {248U * 1024U, 256U * 1024U}) {
        uint32_t tokensPerCore = totalUb == 248U * 1024U ? 560U : 592U;
        uint32_t fixedBytes = 32U + 64U + 8192U + 32U + 32U;
        uint32_t indexBytes = 2U * tokensPerCore * 32U * sizeof(int32_t);
        uint32_t fp32Bytes = 2U * 8192U * sizeof(float);
        uint32_t ioBytes = 2U * 8192U * sizeof(uint16_t) + 32U;
        uint32_t otherBytes = fixedBytes + indexBytes + fp32Bytes + ioBytes;
        Require(otherBytes + 128U * 32U > totalUb, "test must catch uncapped padded UB allocation");
        Require(otherBytes + ChunkRows(10000U, 1000U) * 32U <= totalUb, "one-buffer shape no longer fits");
    }
}
} // namespace

int main()
{
    std::mt19937 rng(20260920U);
    uint32_t cases = 0U;
    for (uint32_t bs : {1U, 7U, 33U, 129U, 4096U}) {
        for (uint32_t k : {1U, 6U, 32U}) {
            for (uint32_t cores : {1U, 2U, 7U, 32U, 33U, 56U, 64U, 100U}) {
                std::vector<int32_t> workspace(static_cast<size_t>(bs) * k + 2U, GUARD);
                for (uint32_t density : {1U, 8U, 0U, 3U}) {
                    Case test = MakeCase(bs, k, cores, density, rng);
                    BuildShared(test, workspace, rng);
                    CheckBoundaryVariants(test, workspace, rng);
                    cases += 6U;
                }
            }
        }
    }
    Require(WorkspaceBytes(4096U, 6U) == 98304U, "typical workspace size");
    Require(WorkspaceBytes(1U, 1U) == 512U, "workspace alignment");
    Require(WorkspaceBytes(0xffffffffU, 32U) == 549755813888ULL, "64-bit workspace arithmetic");
    CheckUbBudget();
    std::cout << "PASS: " << cases << " shared-index model cases; ownership, layout, bounds, reuse and UB budget\n";
    return 0;
}
