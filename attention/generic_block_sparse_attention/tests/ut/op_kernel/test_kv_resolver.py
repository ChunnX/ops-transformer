# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compile and exercise the production KV resolver on CPU (no CANN runtime required)."""
import pathlib
import re
import subprocess
import tempfile
import unittest


class KvResolverTest(unittest.TestCase):
    def test_registered_keys_are_unique_and_preserve_pa_values(self):
        kernel = pathlib.Path(__file__).resolve().parents[3] / "op_kernel"
        header = (kernel / "generic_block_sparse_attention_tilingkey.h").read_text()
        entry = (kernel / "generic_block_sparse_attention.cpp").read_text()
        sections = [header.split("#if (__CCE_AICORE__ == 220)")[1].split("#elif")[0],
                    header.split("#elif (__CCE_AICORE__ == 310)")[1]]
        for arch, section in zip((220, 310), sections):
            keys = dict((name, int(value)) for name, value in re.findall(
                r"#define (GBSA_(?:FP16|BF16|FP8)_\w+) (\d+)ULL", section))
            self.assertEqual(len(keys), 12)
            self.assertEqual(len(set(keys.values())), len(keys))
            for name, value in keys.items():
                base = 9200000000000000 if arch == 220 else 9250000000000000
                paged = "PA_BBND" in name
                expected = base + (71000000 if paged else 30000000) + 1002
                if name.startswith("GBSA_BF16"):
                    expected += 22220
                elif name.startswith("GBSA_FP8"):
                    expected += 20 if "_BF16_" in name else 10
                if "HALFSM" in name:
                    expected += 100000
                if name.endswith("LSE_OUT"):
                    expected += 100000000
                self.assertEqual(value, expected, name)
                self.assertIn(f"TILING_KEY_IS({name});", entry)

    def test_addressing_lengths_and_sparse_filtering(self):
        kernel = pathlib.Path(__file__).resolve().parents[3] / "op_kernel"
        with tempfile.TemporaryDirectory() as tmp:
            tmp = pathlib.Path(tmp)
            # Only emulate GM tensor reads; all address and length logic is production C++.
            (tmp / "kernel_operator.h").write_text(r'''
#pragma once
#include <cstdint>
#include <cassert>
#define __aicore__
#define __gm__
using GM_ADDR = uint8_t *;
namespace AscendC {
template <class T> struct GlobalTensor {
    T *data = nullptr;
    void SetGlobalBuffer(T *p) { data = p; }
    T GetValue(int64_t i) { assert(data != nullptr && i >= 0); return data[i]; }
};
}
''')
            (tmp / "test.cpp").write_text(r'''
#include "generic_block_sparse_attention_kv_resolver.h"
#include <climits>
#include <vector>
using GbsaKernel::KvBlockResolver;
using GbsaKernel::KvTokenRange;
using GbsaKernel::KvTile;
template <class T> GM_ADDR gm(T *p) { return reinterpret_cast<GM_ADDR>(p); }
int main() {
    int64_t cu[] = {0, 131, 390};
    int32_t used[] = {129, 258};
    int32_t table[] = {5, 2, 0, 7, 1, 3};
    KvBlockResolver<false> tnd, padded;
    tnd.Init(gm(cu), nullptr, nullptr);
    padded.Init(gm(cu), gm(used), nullptr);
    KvBlockResolver<true> pa;
    pa.Init(nullptr, gm(used), gm(table));
    int64_t start = -1;
    uint32_t len = 0;
    KvTile tile{};
    KvTokenRange range{};
    assert(tnd.ReadSequence(1, start, len) && start == 131 && len == 259);
    assert(padded.ReadSequence(1, start, len) && start == 131 && len == 258);
    assert(padded.ResolveSingleTileBlock(2, 128, len, 1, 0, 0, 128, tile));
    assert(tile.tokenOffset == 256 && tile.tokenCount == 2);
    auto base = padded.BatchHeadOffset(start, 1024, 2, 128);
    // Non-PA offsets are independent of page stride, including a zero stride.
    assert(padded.Offset(tile, base, 0, 1024) == (131 + 256) * 1024 + 256);
    assert(padded.Offset(tile, base, 999999, 1024) == padded.Offset(tile, base, 0, 1024));
    assert(pa.ReadSequence(1, start, len) && start == 0 && len == 258);
    assert(pa.ResolveSingleTileBlock(2, 128, len, 1, 3, 128, 128, tile));
    assert(tile.pageId == 3 && tile.tokenOffset == 0 && tile.tokenCount == 2);
    base = pa.BatchHeadOffset(start, 1024, 2, 128);
    // K and V may have different physical page strides.
    assert(pa.Offset(tile, base, 256 * 1024, 1024) == 3 * 256 * 1024 + 256);
    assert(pa.Offset(tile, base, 384 * 1024, 1024) == 3 * 384 * 1024 + 256);
    assert(!pa.ResolveSingleTileBlock(3, 128, 1000, 1, 3, 128, 128, tile));
    assert(!tnd.ResolveSparseRange(-1, 128, 258, range));
    assert(!tnd.ResolveSparseRange(INT_MAX, 128, 258, range));
    assert(!tnd.ResolveSparseRange(2, 128, 256, range));
    assert(!tnd.ResolveSparseRange(0, 0, 258, range));
    assert(tnd.ResolveSingleTileBlock(1, 128, 129, 1, 0, 0, 128, tile) && tile.tokenCount == 1);
    // Raw FD slices preserve ordering and duplicates, with the same causal filtering.
    std::vector<int32_t> sparse = {2, 0, -1, 1, 2, 100};
    std::vector<uint32_t> whole, split;
    for (int i = 0; i < 6; ++i)
        if (tnd.ResolveSingleTileBlock(sparse[i], 128, 258, 1, 0, 0, 128, tile))
            whole.push_back(tile.tokenOffset);
    for (int part = 0; part < 3; ++part)
        for (int i = part * 2; i < part * 2 + 2; ++i)
            if (tnd.ResolveSingleTileBlock(sparse[i], 128, 258, 1, 0, 0, 128, tile))
                split.push_back(tile.tokenOffset);
    assert(whole == split && whole == std::vector<uint32_t>({256, 0, 128, 256}));

    // Exercise independent sparse/page/compute sizes in the addressing layer.
    // sparse=192, PA page=256, tile=128: sparse block 1 starts inside page 0,
    // spans page 1, and ends at the causal boundary 350.
    assert(pa.ResolveSparseRange(1, 192, 350, range) && range.start == 192 && range.count == 158);
    assert(pa.ResolveTile(range, 1, 3, 256, 128, tile));
    assert(tile.pageId == 7 && tile.tokenOffset == 192 && tile.tokenCount == 64);
    assert(pa.Offset(tile, 256, 512 * 1024, 1024) == 7 * 512 * 1024 + 192 * 1024 + 256);
    range.start += tile.tokenCount;
    range.count -= tile.tokenCount;
    assert(pa.ResolveTile(range, 1, 3, 256, 128, tile));
    assert(tile.pageId == 1 && tile.tokenOffset == 0 && tile.tokenCount == 94);
    // PA bounds use logical page indices, not sparse block indices.
    assert(pa.ResolveSingleTileBlock(5, 32, 192, 1, 1, 256, 128, tile));
    assert(tile.pageId == 2 && tile.tokenOffset == 160 && tile.tokenCount == 32);
    assert(!pa.ResolveSingleTileBlock(1, 512, 1024, 1, 3, 128, 128, tile));
    // The current kernel's one-tile adapter must never truncate a sparse range.
    assert(!pa.ResolveSingleTileBlock(1, 192, 350, 1, 3, 256, 128, tile));
    assert(!tnd.ResolveSingleTileBlock(1, 192, 350, 1, 0, 0, 128, tile));
    // TND has no page boundaries; compute tiles alone divide the selected range.
    assert(tnd.ResolveSparseRange(1, 192, 350, range));
    assert(tnd.ResolveTile(range, 1, 0, 0, 128, tile));
    assert(tile.tokenOffset == 192 && tile.tokenCount == 128);
    range.start += tile.tokenCount;
    range.count -= tile.tokenCount;
    assert(tnd.ResolveTile(range, 1, 0, 777, 128, tile));
    assert(tile.tokenOffset == 320 && tile.tokenCount == 30);
    // A page smaller than the compute tile also clips the contiguous span.
    range = {48, 96};
    assert(pa.ResolveTile(range, 1, 3, 64, 128, tile));
    assert(tile.pageId == 7 && tile.tokenOffset == 48 && tile.tokenCount == 16);
    // A sparse block spanning >256 compute tiles does not need an expanded array
    // in the address resolver (kernel scheduling still requires separate work).
    assert(tnd.ResolveSparseRange(1, 65536, 131071, range));
    uint32_t total = 0, tiles = 0;
    while (range.count != 0) {
        assert(tnd.ResolveTile(range, 0, 0, 0, 128, tile));
        total += tile.tokenCount;
        ++tiles;
        range.start += tile.tokenCount;
        range.count -= tile.tokenCount;
    }
    assert(total == 65535 && tiles == 512);
    assert(!tnd.ResolveTile(range, 0, 0, 0, 128, tile) && tile.tokenCount == 0);
    range = {0, 128};
    assert(!pa.ResolveTile(range, 0, 3, 0, 128, tile) && tile.tokenCount == 0);
    assert(!tnd.ResolveTile(range, 0, 0, 0, 0, tile) && tile.tokenCount == 0);

    used[1] = 260;
    assert(!padded.ReadSequence(1, start, len));
    used[1] = -1;
    assert(!padded.ReadSequence(1, start, len));
    assert(!pa.ReadSequence(1, start, len));
    used[1] = 0;
    assert(padded.ReadSequence(1, start, len) && len == 0);
    assert(!padded.ResolveSparseRange(0, 128, len, range));
    cu[2] = 130;
    assert(!tnd.ReadSequence(1, start, len));
    cu[1] = -1;
    assert(!tnd.ReadSequence(1, start, len));
    cu[1] = 0; cu[2] = INT64_MAX;
    assert(!tnd.ReadSequence(1, start, len));
    cu[0] = 1; cu[1] = 2;
    assert(!tnd.ReadSequence(0, start, len));
    table[3] = -1;
    range = {0, 128};
    assert(!pa.ResolveTile(range, 1, 3, 128, 128, tile) && tile.tokenCount == 128);
}

''')
            subprocess.run(
                ["g++", "-std=c++17", "-Wall", "-Wextra", "-Werror", "-fsanitize=undefined",
                 f"-I{tmp}", f"-I{kernel}", str(tmp / "test.cpp"), "-o", str(tmp / "test")],
                check=True,
            )
            subprocess.run([str(tmp / "test")], check=True)


if __name__ == "__main__":
    unittest.main()
