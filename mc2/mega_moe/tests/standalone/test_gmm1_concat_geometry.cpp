/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../../op_kernel/arch35/common/mega_moe_concat_geometry.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
// Independent logical [N,K] input; gate/up, K blocks and experts have different values.
uint8_t Weight(size_t n, size_t k, size_t expert, size_t elementsPerByte)
{
    return (n * 7 + k * 3 + n / 128 + k / 32 + expert * 5) % (elementsPerByte == 2 ? 16 : 127);
}

size_t NzOffset(size_t n, size_t k, size_t nSize, size_t c0)
{
    return (k / c0 * nSize + n) * c0 + k % c0;
}

void Store(std::vector<uint8_t> &bytes, size_t offset, uint8_t value, size_t elementsPerByte)
{
    if (elementsPerByte == 1) {
        bytes.at(offset) = value;
    } else {
        bytes.at(offset / 2) |= value << (4 * (offset % 2));
    }
}

uint8_t Load(const std::vector<uint8_t> &bytes, size_t offset, size_t elementsPerByte)
{
    return elementsPerByte == 1 ? bytes.at(offset) : (bytes.at(offset / 2) >> (4 * (offset % 2))) & 15;
}

double Decode(uint8_t value, size_t elementsPerByte)
{
    const double fp4[] = {0, .5, 1, 1.5, 2, 3, 4, 6};
    return elementsPerByte == 1 ? (static_cast<int>(value) - 63) / 32.0 : fp4[value & 7] * ((value & 8) ? -1 : 1);
}

void Check(size_t halfN, size_t fullK, size_t c0, size_t elementsPerByte, size_t expert)
{
    constexpr size_t singleN = 128;
    std::vector<uint8_t> source(2 * halfN * fullK / elementsPerByte);
    for (size_t n = 0; n < 2 * halfN; ++n) {
        for (size_t k = 0; k < fullK; ++k) {
            Store(source, NzOffset(n, k, 2 * halfN, c0), Weight(n, k, expert, elementsPerByte), elementsPerByte);
        }
    }
    for (size_t j = 0; j < halfN; j += singleN) {
        std::vector<double> actual(2 * singleN), reference(2 * singleN);
        // Includes nonzero K origins and a short final K tile.
        for (size_t kStart = 0; kStart < fullK; kStart += 192) {
            const size_t kSize = std::min(size_t(192), fullK - kStart);
            const auto copy = MegaMoeImpl::MakeGmm1ConcatZnCopy(kSize, singleN, halfN, c0, elementsPerByte);
            const size_t bytes = 2 * singleN * kSize / elementsPerByte;
            std::vector<uint8_t> tile(bytes + 32, 0xA5);
            const size_t sourceStart = NzOffset(j, kStart, 2 * halfN, c0) / elementsPerByte;
            // Execute the burst descriptor, including bounds and a destination guard.
            for (size_t burst = 0; burst < copy.blockCount; ++burst) {
                for (size_t b = 0; b < copy.blockBytes; ++b) {
                    tile.at(burst * copy.destinationStrideBytes + b) =
                        source.at(sourceStart + burst * copy.sourceStrideBytes + b);
                }
            }
            if (!std::all_of(tile.begin() + bytes, tile.end(), [](uint8_t v) { return v == 0xA5; })) {
                throw std::runtime_error("concat overwrote the destination guard");
            }
            for (size_t n = 0; n < 2 * singleN; ++n) {
                const size_t sourceN = j + n % singleN + (n / singleN) * halfN;
                for (size_t k = 0; k < kSize; ++k) {
                    const auto expected = Weight(sourceN, kStart + k, expert, elementsPerByte);
                    const auto got = Load(tile, NzOffset(n, k, 2 * singleN, c0), elementsPerByte);
                    if (got != expected) {
                        throw std::runtime_error("gate/up or K-block address mismatch");
                    }
                    const double a = (static_cast<int>((kStart + k) % 7) - 3) / 8.0;
                    actual[n] += a * Decode(got, elementsPerByte);
                    reference[n] += a * Decode(expected, elementsPerByte);
                }
            }
        }
        for (size_t n = 0; n < singleN; ++n) {
            const double got = actual[n] / (1 + std::exp(-actual[n])) * actual[n + singleN];
            const double expected = reference[n] / (1 + std::exp(-reference[n])) * reference[n + singleN];
            if (std::abs(got - expected) > 1e-12) {
                throw std::runtime_error("SwiGLU pairing mismatch");
            }
        }
    }
}
} // namespace

int main()
{
    size_t cases = 0;
    for (size_t halfN : {128, 256, 384, 1024}) {
        for (size_t k : {256, 320, 768}) {
            for (size_t expert : {0, 1, 7}) {
                Check(halfN, k, 32, 1, expert); // A8W8 NZ
                cases += 1;
            }
        }
    }
    std::cout << cases << " NZ concat geometry cases passed\n";
}
