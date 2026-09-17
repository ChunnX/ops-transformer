# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""CPU validation of diagnostic guards, not an AscendC compilation/device test."""

import pathlib
import subprocess
import tempfile
import unittest


HEADER = (
    pathlib.Path(__file__).resolve().parents[1]
    / "op_kernel/arch35/common/mega_moe_mte_diagnostics.h"
)

STUBS = r"""
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <limits>
#define MEGA_MOE_UTILS_H
#define __aicore__
// CPU stubs do not model the device GM address space.
#define __gm__
using GM_ADDR = unsigned char *;
constexpr int AIV = 1;
constexpr int g_coreType = AIV;
uint32_t core = 5;
uint32_t GetBlockIdx() { return core; }
uint32_t GetSubBlockIdx() { return core % 2; }
uint64_t GetSystemCycle() { return 1; }
namespace AscendC { void Trap() { throw 1; } }
namespace MegaMoeImpl {
constexpr int TOPO_TYPE_MTE = 0;
struct Tiling {
    int64_t topoType=0;
    uint32_t maxOutputSize=2310, bs=256, numMaxTokensPerRank=256;
    uint32_t h=7168, hiddenDim=4096, topK=9, moeExpertPerRank=10;
    int32_t topkWeightsPrefetch=0;
    int64_t combineQuantMode=0;
    uint32_t epWorldSize=2;
};
struct Peermem {
    GM_ADDR maskRecvPtr{}, expertCountRecvPtr{}, quantTokenScalePtr{}, combineSendPtr{};
};
struct Params {
    uint32_t diagRank=0;
    mutable uint32_t diagRecords[2] = {0, 0};
    Tiling *tilingData;
    Peermem peermemInfo;
};
struct WorkspaceLayout {
    uint64_t workspaceSize=1024;
    explicit WorkspaceLayout(Tiling *) {}
};
GM_ADDR g_winRankAddr_[2]{};
}
"""

CHECKS = r"""
int main() {
    using namespace MegaMoeImpl;
    Tiling t;
    Params p{0, {0, 0}, &t, {}};
    unsigned char memory[32];
    assert(MteDiagEnabled(p));
    assert(MteDiagSelected(p));
    MteDiagRange(p, "exact_end", memory, 16, 16, 32, 0);
    MteDiagRange(p, "zero_at_end", memory, 32, 0, 32, 0);
    int traps = 0;
    auto bad = [&](uint64_t offset, uint64_t bytes) {
        try { MteDiagRange(p, "bad", memory, offset, bytes, 32, 0); }
        catch (int) { ++traps; }
    };
    bad(16, 17);
    bad(33, 0);
    bad(std::numeric_limits<uint64_t>::max(), 2);
    try { MteDiagRange(p, "null", nullptr, 0, 1, 32, 0); }
    catch (int) { ++traps; }
    assert(traps == 4);
    core = 7;
    assert(!MteDiagSelected(p));
    bad(32, 1); // Unselected cores still check invalid accesses.
    assert(traps == 5);
    t.maxOutputSize = 1050;
    assert(!MteDiagEnabled(p));
    bad(32, 1); // Other cases preserve normal execution.
    assert(traps == 5);
    t.maxOutputSize = 2310;
    t.topoType = 1;
    bad(32, 1); // URMA is outside this diagnostic.
    assert(traps == 5);
    t.topoType = 0;
    core = 5;
    p.diagRecords[0] = MEGA_MOE_MTE_DIAG_RECORDS;
    bad(32, 1); // Log budget does not disable bounds checks.
    assert(traps == 6);
    MteDiagRange(p, "combine_budget", memory, 0, 1, 32, 0, true);
    assert(p.diagRecords[1] == 1);
    try { MteDiagRequire(p, false, "negative_index", -1, 2); }
    catch (int) { ++traps; }
    assert(traps == 7);
    MteDiagInit(p, memory, memory);
    MteDiagStage(p, "test");
}
"""


class MteDiagnosticsTest(unittest.TestCase):
    def test_guards_and_disabled_build(self):
        with tempfile.TemporaryDirectory() as temp:
            root = pathlib.Path(temp)
            for enabled in (0, 1):
                source = root / f"diag_{enabled}.cpp"
                source.write_text(
                    STUBS
                    + f'\n#include "{HEADER}"\n'
                    + (CHECKS if enabled else "int main() {}\n")
                )
                binary = root / f"diag_{enabled}"
                subprocess.run(
                    [
                        "g++",
                        "-std=c++17",
                        "-Wall",
                        "-Wextra",
                        "-Werror",
                        "-Wno-sign-compare",
                        f"-DMEGA_MOE_MTE_DIAG={enabled}",
                        str(source),
                        "-o",
                        str(binary),
                    ],
                    check=True,
                )
                result = subprocess.run(
                    [str(binary)], check=True, capture_output=True, text=True
                )
                if enabled:
                    self.assertIn("MM_DIAG_INVALID", result.stdout)
                    self.assertIn("valid=0", result.stdout)
                else:
                    self.assertNotIn(b"MM_DIAG", binary.read_bytes())


if __name__ == "__main__":
    unittest.main()
