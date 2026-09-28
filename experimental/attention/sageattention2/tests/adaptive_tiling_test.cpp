/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "../op_host/adaptive_tiling.h"

namespace {

int g_failures = 0;

void Report(const char *name, double val, double bound, bool ok)
{
    if (ok) {
        std::printf("  %-42s val = %-12g bound = %-12g PASS\n", name, val, bound);
    } else {
        std::printf("  %-42s val = %-12g bound = %-12g FAIL\n", name, val, bound);
        g_failures++;
    }
}

void ReportBool(const char *name, bool ok)
{
    if (ok) {
        std::printf("  %-42s PASS\n", name);
    } else {
        std::printf("  %-42s FAIL\n", name);
        g_failures++;
    }
}

struct BenchmarkCase {
    const char *name;
    uint32_t b, hq, hk, sq, sk, d;
    bool causal;
};

// [1] P-01 ~ P-07 Full Benchmark Suite Verification
void TestBenchmarkCases()
{
    std::printf("[1] P-01 ~ P-07 Benchmark Suite Wave Utilization & Hardware Fit\n");
    const std::vector<BenchmarkCase> cases = {
        {"P-01 (1,32,4096,128)", 1, 32, 32, 4096, 4096, 128, false},
        {"P-02 (1,32,8192,128)", 1, 32, 32, 8192, 8192, 128, false},
        {"P-03 (1,32,8192,128) causal", 1, 32, 32, 8192, 8192, 128, true},
        {"P-04 (2,32,8192,128)", 2, 32, 32, 8192, 8192, 128, false},
        {"P-05 (1,32:8,4096,128) GQA", 1, 32, 8, 4096, 4096, 128, false},
        {"P-06 (1,32,4096,128) BF16", 1, 32, 32, 4096, 4096, 128, false},
        {"P-07 (1,32,16384,128)", 1, 32, 32, 16384, 16384, 128, false},
    };

    double minUtil = 1.0;
    bool allFit = true;
    for (const auto &tc : cases) {
        auto sel = optiling::AdaptiveTileSelect(tc.b, tc.hq, tc.hk, tc.sq, tc.sk, tc.d, tc.causal, 24);
        minUtil = std::fmin(minUtil, static_cast<double>(sel.waveUtilization));
        allFit = allFit && (sel.ubBytes <= 384ULL * 1024ULL) && (sel.l1Bytes <= 512ULL * 1024ULL);
    }
    Report("benchmark minimum wave utilization (>=85%)", minUtil, 0.85, minUtil >= 0.85);
    Report("benchmark minimum wave utilization (>=99%)", minUtil, 0.99, minUtil >= 0.99);
    ReportBool("benchmark UB & L1 constraints satisfied", allFit);
}

// [2] Few-Head Variants (H_Q in {8, 12, 16, 20}) & Wave Balancer Fallback Verification
void TestFewHeadVariants()
{
    std::printf("[2] Few-Head Variants & Tail Bubble Reduction (Wave Balancer)\n");
    const std::vector<uint32_t> headConfigs = {8, 12, 16, 20};
    const std::vector<uint32_t> seqConfigs = {512, 1024, 2048};

    double minUtil = 1.0;
    for (uint32_t hq : headConfigs) {
        for (uint32_t sq : seqConfigs) {
            auto sel = optiling::AdaptiveTileSelect(1, hq, hq, sq, sq, 128, false, 24);
            minUtil = std::fmin(minUtil, static_cast<double>(sel.waveUtilization));
        }
    }
    Report("few-head minimum wave utilization (>=85%)", minUtil, 0.85, minUtil >= 0.85);

    // Deep dive into severe tail-bubble scenario: B=1, H_Q=8, S_Q=512
    // Baseline M=128: 8 * 4 = 32 tasks, waves = 2 (slots = 48), util = 32/48 = 66.67% (< 75%)
    // Adaptive M=64:  8 * 8 = 64 tasks, waves = 3 (slots = 72), util = 64/72 = 88.89% (>= 85%)
    auto selBaseline = optiling::EvaluateCandidate(1, 8, 512, 512, 128, 128, 256, 24);
    auto selAdaptive = optiling::AdaptiveTileSelect(1, 8, 8, 512, 512, 128, false, 24);

    Report("baseline M=128 utilization (<75% bubble)", selBaseline.waveUtilization, 0.75, selBaseline.waveUtilization < 0.75);
    Report("adaptive M=64 utilization (>=85%)", selAdaptive.waveUtilization, 0.85, selAdaptive.waveUtilization >= 0.85);
    ReportBool("fallback to M=64 triggered for H_Q=8, S_Q=512", selAdaptive.isFallbackM64);

    // Verify tail bubble reduction: idle cores drop from 16 to 8 (50% reduction)
    uint32_t baselineIdleCores = 24 - selBaseline.tailCores; // 24 - 8 = 16
    uint32_t adaptiveIdleCores = 24 - selAdaptive.tailCores; // 24 - 16 = 8
    double bubbleReduction = static_cast<double>(baselineIdleCores - adaptiveIdleCores) / baselineIdleCores;
    Report("tail idle core reduction ratio (>=50%)", bubbleReduction, 0.50, bubbleReduction >= 0.50);
}

// [3] Hardware Capacity Bounds & Strict Vector Alignment
void TestHardwareConstraintsAndAlignment()
{
    std::printf("[3] Hardware Capacity Bounds & Alignment Checks\n");
    const uint32_t testSq[] = {384, 512, 768, 1024, 2048, 4096, 8192};
    bool alignPass = true;
    bool capacityPass = true;

    for (uint32_t sq : testSq) {
        auto sel = optiling::AdaptiveTileSelect(1, 16, 16, sq, sq, 128, false, 24);
        uint32_t qp = (sq + sel.tileM - 1U) & ~(sel.tileM - 1U);
        uint32_t kp = (sq + sel.tileN - 1U) & ~(sel.tileN - 1U);

        // Alignment check
        if (qp % sel.tileM != 0 || kp % sel.tileN != 0 || qp % 16 != 0 || kp % 16 != 0) {
            alignPass = false;
        }
        // Capacity check: UB <= 384 KiB, L1 <= 512 KiB
        if (sel.ubBytes > 384ULL * 1024ULL || sel.l1Bytes > 512ULL * 1024ULL) {
            capacityPass = false;
        }
    }
    ReportBool("tile-boundary and 16-element vector alignment", alignPass);
    ReportBool("UB (<=384 KiB) and L1 (<=512 KiB) capacity bounds", capacityPass);
}

// [4] Asymmetric & Non-Power-of-2 Head Counts Edge Cases
void TestEdgeCases()
{
    std::printf("[4] Non-Power-of-2 Head Counts & Edge Cases\n");
    const std::vector<uint32_t> oddHeads = {6, 10, 14, 18, 22};
    const std::vector<uint32_t> seqLens = {768, 1536, 3072};

    double minUtil = 1.0;
    for (uint32_t hq : oddHeads) {
        for (uint32_t sq : seqLens) {
            auto sel = optiling::AdaptiveTileSelect(1, hq, hq, sq, sq, 128, false, 24);
            minUtil = std::fmin(minUtil, static_cast<double>(sel.waveUtilization));
        }
    }
    Report("non-power-of-2 heads wave utilization (>=85%)", minUtil, 0.85, minUtil >= 0.85);
}

} // namespace

int main()
{
    std::printf("================================================================\n");
    std::printf("SageAttention2 Adaptive Tiling & Wave Balancer Conformance Test\n");
    std::printf("================================================================\n");

    TestBenchmarkCases();
    TestFewHeadVariants();
    TestHardwareConstraintsAndAlignment();
    TestEdgeCases();

    std::printf("================================================================\n");
    if (g_failures == 0) {
        std::printf("ALL ADAPTIVE TILING & WAVE BALANCER TESTS PASSED\n");
        return 0;
    } else {
        std::printf("%d TESTS FAILED\n", g_failures);
        return 1;
    }
}
