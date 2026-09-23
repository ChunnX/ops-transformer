/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Host-only test of the production tiling selector plus a model of device tile ownership/ranges.
#include "../../../op_kernel/moe_ep_dispatch_schedule.h"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

using MoeEpDispatchSchedule::METADATA_TILE_BYTES;
using MoeEpDispatchSchedule::ROUTE_TILE_ELEMENTS;
using MoeEpDispatchSchedule::TileCount;
using MoeEpDispatchSchedule::TileTokens;

void CheckTiles(uint32_t bs, uint32_t cores, uint32_t topK, uint32_t metadataBytes)
{
    uint32_t tokens = TileTokens(bs, cores, topK, metadataBytes);
    uint32_t tiles = TileCount(bs, tokens);
    assert(tokens > 0 && tokens * topK <= ROUTE_TILE_ELEMENTS);
    assert(tokens * metadataBytes <= METADATA_TILE_BYTES);
    std::vector<uint32_t> visits(tiles, 0);
    for (uint32_t core = 0; core < cores; ++core) {
        for (uint32_t tile = core; tile < tiles; tile += cores) { ++visits[tile]; }
    }
    uint32_t end = 0;
    for (uint32_t tile = 0; tile < tiles; ++tile) {
        uint32_t begin = tile * tokens;
        uint32_t count = bs - begin < tokens ? bs - begin : tokens;
        assert(visits[tile] == 1 && begin == end && count > 0);
        end = begin + count;
    }
    assert(end == bs);
    assert(tiles >= (bs < cores ? bs : cores));
}

int main()
{
    for (uint32_t bs : {1U, 31U, 33U, 56U, 129U, 4096U, 18000U, 65537U}) {
        for (uint32_t cores : {33U, 48U, 56U, 64U, 100U}) {
            for (uint32_t topK : {1U, 6U, 7U, 32U}) {
                for (uint32_t bytes : {512U, 1024U}) { CheckTiles(bs, cores, topK, bytes); }
            }
        }
    }
    assert(TileTokens(4096, 56, 6, 512) == 32);
    assert(TileCount(4096, 32) == 128);
    std::cout << "Dispatch tile schedule checks passed\n";
}
