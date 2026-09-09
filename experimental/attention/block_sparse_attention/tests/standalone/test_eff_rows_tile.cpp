/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Host-only scheduling regression; no Ascend SDK required.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#define __aicore__
#include "../../op_kernel/attn_infra/gemm/block/bsa_eff_rows_tile.hpp"
#undef __aicore__

using NpuArch::Gemm::Block::EffRowsTile;
using NpuArch::Gemm::Block::NextEffRowsTile;

struct Tensor {
    const std::vector<int32_t> &data;
    int32_t GetValue(uint64_t offset) const { return data.at(offset); }
};

void Check(const std::vector<int32_t> &indices, const std::vector<int32_t> &effectiveRows,
           uint32_t blockRows, int64_t seqLen, uint32_t tileRows)
{
    // Nonzero metadata base catches missing batch/head offsets.
    constexpr uint64_t base = 14;
    std::vector<int32_t> metadata(base, -1);
    for (int32_t rows : effectiveRows) {
        metadata.push_back(7); // Q rows must not be used as KV rows.
        metadata.push_back(rows);
    }
    std::vector<int64_t> expected;
    uint32_t tileCount = 0;
    for (int32_t block : indices) {
        uint32_t count = 0;
        for (uint32_t row = 0; row < blockRows; ++row) {
            int64_t offset = static_cast<int64_t>(block) * blockRows + row;
            if (row < static_cast<uint32_t>(effectiveRows.at(block)) && offset < seqLen) {
                expected.push_back(offset);
                ++count;
            }
        }
        tileCount += (count + tileRows - 1) / tileRows;
    }

    uint32_t kBlock = 0, kRow = 0, vBlock = 0, vRow = 0;
    std::vector<EffRowsTile> launched;
    std::vector<int64_t> copied;
    double numerator = 0, denominator = 0;
    // Follow the kernel's two-tile prelaunch, including drain for 0/1/2 tiles.
    for (uint32_t step = 0; step < tileCount + 2; ++step) {
        if (step < tileCount) {
            auto tile = NextEffRowsTile(Tensor{indices}, Tensor{metadata}, base,
                indices.size(), blockRows, seqLen, tileRows, kBlock, kRow);
            assert(tile.rows > 0 && tile.rows <= tileRows);
            assert(tile.srcOffset / blockRows == tile.blockIndex);
            assert((tile.srcOffset + tile.rows - 1) / blockRows == tile.blockIndex);
            launched.push_back(tile);
            for (uint32_t row = 0; row < tile.rows; ++row) {
                copied.push_back(tile.srcOffset + row);
            }
        }
        if (step >= 2) {
            auto tile = NextEffRowsTile(Tensor{indices}, Tensor{metadata}, base,
                indices.size(), blockRows, seqLen, tileRows, vBlock, vRow);
            auto qk = launched.at(step - 2);
            assert(tile.blockIndex == qk.blockIndex && tile.srcOffset == qk.srcOffset && tile.rows == qk.rows);
            // Different K/V scales for each ORIGINAL block, with skipped sparse indices.
            double kScale = 0.125 * (tile.blockIndex + 1);
            double vScale = 0.25 * (tile.blockIndex + 2);
            for (uint32_t row = 0; row < tile.rows; ++row) {
                int64_t source = tile.srcOffset + row;
                double weight = std::exp(0.01 * (source % 13) * kScale);
                denominator += weight;
                numerator += weight * (source % 17) * vScale;
            }
        }
    }
    assert(copied == expected);
    assert(NextEffRowsTile(Tensor{indices}, Tensor{metadata}, base, indices.size(), blockRows,
                          seqLen, tileRows, kBlock, kRow).rows == 0);
    double refNumerator = 0, refDenominator = 0;
    for (int64_t source : expected) {
        auto block = source / blockRows;
        double weight = std::exp(0.01 * (source % 13) * 0.125 * (block + 1));
        refDenominator += weight;
        refNumerator += weight * (source % 17) * 0.25 * (block + 2);
    }
    if (!expected.empty()) {
        assert(std::abs(numerator / denominator - refNumerator / refDenominator) < 1e-12);
    }
}

int main()
{
    Check({}, {0}, 512, 512, 512);
    Check({0, 1, 2}, {0, 0, 0}, 512, 1536, 512);
    Check({0, 2, 3}, {1, 512, 0, 0}, 512, 2048, 512);
    Check({0, 2, 3}, {1, 512, 17, 0}, 512, 2048, 512);
    Check({0, 2, 3}, {33, 512, 65, 511}, 512, 1601, 512);
    Check({0, 1, 3}, {511, 257, 0, 1024}, 512, 1700, 128);
    Check({3, 0, 2}, {257, 0, 17, 513}, 768, 3000, 256);
    std::mt19937 random(9334);
    for (uint32_t trial = 0; trial < 2000; ++trial) {
        uint32_t blockRows = 16 * (1 + random() % 64);
        uint32_t count = 1 + random() % 16;
        uint32_t tileRows = std::min(blockRows, 128U << (random() % 3));
        std::vector<int32_t> rows(count), indices;
        for (uint32_t block = 0; block < count; ++block) {
            rows[block] = random() % (blockRows + 33);
            if (random() % 3 != 0) {
                indices.push_back(block);
            }
        }
        Check(indices, rows, blockRows, random() % (count * blockRows + 1), tileRows);
    }
    std::cout << "Passed 2007 effRows scheduling and scale-index regression cases\n";
}
