/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_apply_rotary_pos_emb_grad.cpp
 * \brief kernel UT for apply_rotary_pos_emb_grad (mla_prolog_v3 template mode).
 *
 * Covers five cases:
 *   - BAB(203) / dcos=0
 *   - AB(204)  / dcos=0
 *   - A(205)   / dcos=0
 *   - A(205)   / dcos=1
 *   - AB(204)  / dcos=1 / directPartial=1 (部分积核内归约直写路径)
 *
 * The test data is self-contained: C++ generated inputs plus CPU golden checks.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <random>
#include <functional>
#include <cstring>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "apply_rotary_pos_emb_grad_tiling.h"

// Include the top-level kernel template and its transitive dependencies.
#include "../../../op_kernel/apply_rotary_pos_emb_grad_apt.cpp"

using namespace std;

// ============================================================================
// PARAM_LIST helpers for the 12 GM arguments in the kernel signature.
// ============================================================================
#define PARAM_LIST_DEF  __gm__ uint8_t* gradQueryEmbed,\
    __gm__ uint8_t* gradKeyEmbed,\
    __gm__ uint8_t* cos,\
    __gm__ uint8_t* sin,\
    __gm__ uint8_t* query,\
    __gm__ uint8_t* key,\
    __gm__ uint8_t* gradQueryOut,\
    __gm__ uint8_t* gradKeyOut,\
    __gm__ uint8_t* gradCosOut,\
    __gm__ uint8_t* gradSinOut,\
    __gm__ uint8_t* workspace,\
    __gm__ uint8_t* tiling

#define PARAM_LIST gradQueryEmbed, gradKeyEmbed, cos, sin, query, key,\
    gradQueryOut, gradKeyOut, gradCosOut, gradSinOut, workspace, tiling

// ============================================================================
// CPU reference helpers.
// ============================================================================

// HalfGradAlignVF: out[0:half] = cos[0:half]*in[0:half] + sin[half:]*in[half:]
//                  out[half:]  = cos[half:]*in[half:] - sin[0:half]*in[0:half]
static void RefHalfGrad(float* __restrict__ out, const float* __restrict__ in,
                         const float* __restrict__ c, const float* __restrict__ s, int D)
{
    int half = D / 2;
    for (int i = 0; i < half; i++) {
        out[i]        = c[i] * in[i] + s[half + i] * in[half + i];
        out[half + i] = c[half + i] * in[half + i] - s[i] * in[i];
    }
}

// HalfRotaryVF: rot(x) = cat(-x[half:], x[:half])
static void RefHalfRotary(float* __restrict__ out, const float* __restrict__ in, int D)
{
    int half = D / 2;
    for (int i = 0; i < half; i++) {
        out[i]        = -in[half + i];
        out[half + i] =  in[i];
    }
}

// ============================================================================
// Test fixture.
// ============================================================================
class apply_rotary_pos_emb_grad_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
    }
    static void TearDownTestCase() {}
};

// ============================================================================
// Case 1: BAB(203) / dcos=0
//   B=2, S=4, nQ=4, nK=2, D=32
//   blockNumB=2, blockFactorB=1, blockNumS=2, blockFactorS=2
//   ubFactorS=2, ubLoopNumN=1, ubFactorN=4, ubTailFactorN=4
//   cos 1S1D: flattened (S, D), one row per s.
//   blockDim = blockNumB * blockNumS = 4
// ============================================================================
TEST_F(apply_rotary_pos_emb_grad_test, bab_dcos0)
{
    const int B = 2, S = 4, nQ = 4, nK = 2, D = 32;
    const int gradQNum = B * S * nQ * D;   // 1024
    const int gradKNum = B * S * nK * D;   // 512
    const int cosNum   = S * D;             // 128  (1S1D: (S,D))
    const int qkNumQ   = B * S * nQ * D;    // 1024 query
    const int qkNumK   = B * S * nK * D;    // 512  key

    // Allocate GM buffers.
    uint8_t* gradQ   = (uint8_t*)AscendC::GmAlloc(gradQNum * sizeof(float));
    uint8_t* gradK   = (uint8_t*)AscendC::GmAlloc(gradKNum * sizeof(float));
    uint8_t* cosBuf  = (uint8_t*)AscendC::GmAlloc(cosNum   * sizeof(float));
    uint8_t* sinBuf  = (uint8_t*)AscendC::GmAlloc(cosNum   * sizeof(float));
    uint8_t* queryBuf = (uint8_t*)AscendC::GmAlloc(qkNumQ  * sizeof(float));
    uint8_t* keyBuf   = (uint8_t*)AscendC::GmAlloc(qkNumK  * sizeof(float));
    uint8_t* gradQOut = (uint8_t*)AscendC::GmAlloc(gradQNum * sizeof(float));
    uint8_t* gradKOut = (uint8_t*)AscendC::GmAlloc(gradKNum * sizeof(float));
    uint8_t* gradCosOut = (uint8_t*)AscendC::GmAlloc(gradQNum * sizeof(float)); // Unused when dcos=0.
    uint8_t* gradSinOut = (uint8_t*)AscendC::GmAlloc(gradQNum * sizeof(float));
    uint8_t* workspace  = (uint8_t*)AscendC::GmAlloc(4096);  // Kernel requires a non-null workspace.
    uint8_t* tiling     = (uint8_t*)AscendC::GmAlloc(sizeof(ApplyRopeGradTilingData));

    // Fill random inputs.
    mt19937 rng(1234);
    uniform_real_distribution<float> dist(-1.0f, 1.0f);
    auto fill = [&](float* buf, int n) { for (int i = 0; i < n; i++) buf[i] = dist(rng); };
    fill(reinterpret_cast<float*>(gradQ),   gradQNum);
    fill(reinterpret_cast<float*>(gradK),   gradKNum);
    fill(reinterpret_cast<float*>(cosBuf),  cosNum);
    fill(reinterpret_cast<float*>(sinBuf),  cosNum);
    fill(reinterpret_cast<float*>(queryBuf), qkNumQ);
    fill(reinterpret_cast<float*>(keyBuf),   qkNumK);
    memset(gradQOut, 0, gradQNum * sizeof(float));
    memset(gradKOut, 0, gradKNum * sizeof(float));

    // Fill tiling.
    memset(tiling, 0, sizeof(ApplyRopeGradTilingData));
    ApplyRopeGradTilingData* td = reinterpret_cast<ApplyRopeGradTilingData*>(tiling);
    td->ropeGradParams.b              = B;
    td->ropeGradParams.s              = S;
    td->ropeGradParams.d              = D;
    td->ropeGradParams.nQ             = nQ;
    td->ropeGradParams.nK             = nK;
    td->ropeGradParams.blockNumB      = 2;
    td->ropeGradParams.blockFactorB   = 1;
    td->ropeGradParams.blockNumS      = 2;
    td->ropeGradParams.blockFactorS   = 2;
    td->ropeGradParams.ubFactorS      = 2;
    td->ropeGradParams.ubLoopNumN     = 1;
    td->ropeGradParams.ubFactorN      = 4;
    td->ropeGradParams.ubTailFactorN  = 4;
    td->ropeGradParams.usedCoreNum    = 4;
    td->ropeGradParams.rotaryMode     = 0;
    td->ropeGradParams.layout         = 0;
    td->ropeGradParams.dCosFlag       = 0;
    td->dCosFlag = 0;
    td->layout    = 0;

    // golden
    const float* fGradQ   = reinterpret_cast<float*>(gradQ);
    const float* fGradK   = reinterpret_cast<float*>(gradK);
    const float* fCos     = reinterpret_cast<float*>(cosBuf);
    const float* fSin     = reinterpret_cast<float*>(sinBuf);
    vector<float> goldenQ(gradQNum, 0.0f), goldenK(gradKNum, 0.0f);
    for (int b = 0; b < B; b++) {
        for (int s = 0; s < S; s++) {
            const float* cosRow = fCos + s * D;
            const float* sinRow = fSin + s * D;
            int bsBase = (b * S + s);
            for (int n = 0; n < nQ; n++) {
                int base = (bsBase * nQ + n) * D;
                RefHalfGrad(&goldenQ[base], fGradQ + base, cosRow, sinRow, D);
            }
            for (int n = 0; n < nK; n++) {
                int base = (bsBase * nK + n) * D;
                RefHalfGrad(&goldenK[base], fGradK + base, cosRow, sinRow, D);
            }
        }
    }

    // Run kernel.
    std::function<void(PARAM_LIST_DEF)> func = [](PARAM_LIST_DEF) {
        apply_rotary_pos_emb_grad<true, false, 0, 0, 0, 203, 0>(PARAM_LIST);
    };
    ICPU_RUN_KF(func, 4, gradQ, gradK, cosBuf, sinBuf, queryBuf, keyBuf,
                    gradQOut, gradKOut, gradCosOut, gradSinOut, workspace, tiling);

    // Verify.
    const float* outQ = reinterpret_cast<float*>(gradQOut);
    const float* outK = reinterpret_cast<float*>(gradKOut);
    for (int i = 0; i < gradQNum; i++) {
        EXPECT_NEAR(outQ[i], goldenQ[i], 1e-5f) << "gradQ idx=" << i;
    }
    for (int i = 0; i < gradKNum; i++) {
        EXPECT_NEAR(outK[i], goldenK[i], 1e-5f) << "gradK idx=" << i;
    }

    // Release GM buffers.
    AscendC::GmFree(gradQ);   AscendC::GmFree(gradK);
    AscendC::GmFree(cosBuf);  AscendC::GmFree(sinBuf);
    AscendC::GmFree(queryBuf); AscendC::GmFree(keyBuf);
    AscendC::GmFree(gradQOut); AscendC::GmFree(gradKOut);
    AscendC::GmFree(gradCosOut); AscendC::GmFree(gradSinOut);
    AscendC::GmFree(workspace);  AscendC::GmFree(tiling);
}

// ============================================================================
// Case 2: AB(204) / dcos=0
//   B=2, S=4, nQ=4, nK=2, D=32
//   blockNumBS=2, blockFactorBS=4, blockTailBS=4
//   blockNumN=2, blockFactorN=2, blockTailN=2
//   ubFactorBS=4, ubFactorN=2
//   cos flattened as (B*S, D).
//   blockDim = blockNumBS * blockNumN = 4
// ============================================================================
TEST_F(apply_rotary_pos_emb_grad_test, ab_dcos0)
{
    const int B = 2, S = 4, nQ = 4, nK = 2, D = 32;
    const int bsTotal = B * S;  // 8
    const int gradQNum = bsTotal * nQ * D;  // 1024
    const int gradKNum = bsTotal * nK * D;  // 512
    const int cosNum   = bsTotal * D;        // 256

    // Allocate GM buffers.
    uint8_t* gradQ   = (uint8_t*)AscendC::GmAlloc(gradQNum * sizeof(float));
    uint8_t* gradK   = (uint8_t*)AscendC::GmAlloc(gradKNum * sizeof(float));
    uint8_t* cosBuf  = (uint8_t*)AscendC::GmAlloc(cosNum   * sizeof(float));
    uint8_t* sinBuf  = (uint8_t*)AscendC::GmAlloc(cosNum   * sizeof(float));
    uint8_t* queryBuf = (uint8_t*)AscendC::GmAlloc(gradQNum * sizeof(float));
    uint8_t* keyBuf   = (uint8_t*)AscendC::GmAlloc(gradKNum * sizeof(float));
    uint8_t* gradQOut = (uint8_t*)AscendC::GmAlloc(gradQNum * sizeof(float));
    uint8_t* gradKOut = (uint8_t*)AscendC::GmAlloc(gradKNum * sizeof(float));
    uint8_t* gradCosOut = (uint8_t*)AscendC::GmAlloc(gradQNum * sizeof(float));
    uint8_t* gradSinOut = (uint8_t*)AscendC::GmAlloc(gradQNum * sizeof(float));
    uint8_t* workspace  = (uint8_t*)AscendC::GmAlloc(4096);
    uint8_t* tiling     = (uint8_t*)AscendC::GmAlloc(sizeof(ApplyRopeGradTilingData));

    // Fill random inputs.
    mt19937 rng(2345);
    uniform_real_distribution<float> dist(-1.0f, 1.0f);
    auto fill = [&](float* buf, int n) { for (int i = 0; i < n; i++) buf[i] = dist(rng); };
    fill(reinterpret_cast<float*>(gradQ),   gradQNum);
    fill(reinterpret_cast<float*>(gradK),   gradKNum);
    fill(reinterpret_cast<float*>(cosBuf),  cosNum);
    fill(reinterpret_cast<float*>(sinBuf),  cosNum);
    fill(reinterpret_cast<float*>(queryBuf), gradQNum);
    fill(reinterpret_cast<float*>(keyBuf),   gradKNum);
    memset(gradQOut, 0, gradQNum * sizeof(float));
    memset(gradKOut, 0, gradKNum * sizeof(float));

    // Fill tiling.
    memset(tiling, 0, sizeof(ApplyRopeGradTilingData));
    ApplyRopeGradTilingData* td = reinterpret_cast<ApplyRopeGradTilingData*>(tiling);
    td->ropeGradABParams.b              = B;
    td->ropeGradABParams.s              = S;
    td->ropeGradABParams.d              = D;
    td->ropeGradABParams.nQ             = nQ;
    td->ropeGradABParams.nK             = nK;
    td->ropeGradABParams.dAlign         = 32;
    td->ropeGradABParams.dSplitCoef     = 2;
    td->ropeGradABParams.blockNumBS     = 2;
    td->ropeGradABParams.blockFactorBS  = 4;
    td->ropeGradABParams.blockTailBS    = 4;
    td->ropeGradABParams.blockNumN      = 2;
    td->ropeGradABParams.blockFactorN   = 2;
    td->ropeGradABParams.blockTailN     = 2;
    td->ropeGradABParams.ubFactorBS     = 4;
    td->ropeGradABParams.ubFactorN      = 2;
    td->ropeGradABParams.usedCoreNum    = 4;
    td->ropeGradABParams.rotaryMode     = 0;
    td->dCosFlag = 0;
    td->layout    = 0;

    // golden
    const float* fGradQ = reinterpret_cast<float*>(gradQ);
    const float* fGradK = reinterpret_cast<float*>(gradK);
    const float* fCos   = reinterpret_cast<float*>(cosBuf);
    const float* fSin   = reinterpret_cast<float*>(sinBuf);
    vector<float> goldenQ(gradQNum, 0.0f), goldenK(gradKNum, 0.0f);
    for (int bs = 0; bs < bsTotal; bs++) {
        const float* cosRow = fCos + bs * D;
        const float* sinRow = fSin + bs * D;
        for (int n = 0; n < nQ; n++) {
            int base = (bs * nQ + n) * D;
            RefHalfGrad(&goldenQ[base], fGradQ + base, cosRow, sinRow, D);
        }
        for (int n = 0; n < nK; n++) {
            int base = (bs * nK + n) * D;
            RefHalfGrad(&goldenK[base], fGradK + base, cosRow, sinRow, D);
        }
    }

    // Run kernel.
    std::function<void(PARAM_LIST_DEF)> func = [](PARAM_LIST_DEF) {
        apply_rotary_pos_emb_grad<true, false, 0, 0, 0, 204, 0>(PARAM_LIST);
    };
    ICPU_RUN_KF(func, 4, gradQ, gradK, cosBuf, sinBuf, queryBuf, keyBuf,
                    gradQOut, gradKOut, gradCosOut, gradSinOut, workspace, tiling);

    // Verify.
    const float* outQ = reinterpret_cast<float*>(gradQOut);
    const float* outK = reinterpret_cast<float*>(gradKOut);
    for (int i = 0; i < gradQNum; i++) {
        EXPECT_NEAR(outQ[i], goldenQ[i], 1e-5f) << "gradQ idx=" << i;
    }
    for (int i = 0; i < gradKNum; i++) {
        EXPECT_NEAR(outK[i], goldenK[i], 1e-5f) << "gradK idx=" << i;
    }

    AscendC::GmFree(gradQ);   AscendC::GmFree(gradK);
    AscendC::GmFree(cosBuf);  AscendC::GmFree(sinBuf);
    AscendC::GmFree(queryBuf); AscendC::GmFree(keyBuf);
    AscendC::GmFree(gradQOut); AscendC::GmFree(gradKOut);
    AscendC::GmFree(gradCosOut); AscendC::GmFree(gradSinOut);
    AscendC::GmFree(workspace);  AscendC::GmFree(tiling);
}

// ============================================================================
// Case 3: A(205) / dcos=0
//   B=2, S=2, N=2 (nQ=nK=2), D=32
//   totalRow = B*S*N = 8
//   blockNumB=2, blockFactorB=4, ubFactorS=4, ubFactorN=4, usedCoreNum=2
//   cos flattened as (totalRow, D), one row per grad row.
//   blockDim = 2
// ============================================================================
TEST_F(apply_rotary_pos_emb_grad_test, a_dcos0)
{
    const int B = 2, S = 2, N = 2, D = 32;
    const int totalRow = B * S * N;  // 8
    const int totalNum = totalRow * D;  // 256

    uint8_t* gradQ   = (uint8_t*)AscendC::GmAlloc(totalNum * sizeof(float));
    uint8_t* gradK   = (uint8_t*)AscendC::GmAlloc(totalNum * sizeof(float));
    uint8_t* cosBuf  = (uint8_t*)AscendC::GmAlloc(totalNum * sizeof(float));
    uint8_t* sinBuf  = (uint8_t*)AscendC::GmAlloc(totalNum * sizeof(float));
    uint8_t* queryBuf = (uint8_t*)AscendC::GmAlloc(totalNum * sizeof(float));
    uint8_t* keyBuf   = (uint8_t*)AscendC::GmAlloc(totalNum * sizeof(float));
    uint8_t* gradQOut = (uint8_t*)AscendC::GmAlloc(totalNum * sizeof(float));
    uint8_t* gradKOut = (uint8_t*)AscendC::GmAlloc(totalNum * sizeof(float));
    uint8_t* gradCosOut = (uint8_t*)AscendC::GmAlloc(totalNum * sizeof(float));
    uint8_t* gradSinOut = (uint8_t*)AscendC::GmAlloc(totalNum * sizeof(float));
    uint8_t* workspace  = (uint8_t*)AscendC::GmAlloc(4096);
    uint8_t* tiling     = (uint8_t*)AscendC::GmAlloc(sizeof(ApplyRopeGradTilingData));

    // Fill random inputs.
    mt19937 rng(3456);
    uniform_real_distribution<float> dist(-1.0f, 1.0f);
    auto fill = [&](float* buf, int n) { for (int i = 0; i < n; i++) buf[i] = dist(rng); };
    fill(reinterpret_cast<float*>(gradQ),   totalNum);
    fill(reinterpret_cast<float*>(gradK),   totalNum);
    fill(reinterpret_cast<float*>(cosBuf),  totalNum);
    fill(reinterpret_cast<float*>(sinBuf),  totalNum);
    fill(reinterpret_cast<float*>(queryBuf), totalNum);
    fill(reinterpret_cast<float*>(keyBuf),   totalNum);
    memset(gradQOut, 0, totalNum * sizeof(float));
    memset(gradKOut, 0, totalNum * sizeof(float));

    // Fill tiling.
    memset(tiling, 0, sizeof(ApplyRopeGradTilingData));
    ApplyRopeGradTilingData* td = reinterpret_cast<ApplyRopeGradTilingData*>(tiling);
    td->ropeGradParams.b              = totalRow;
    td->ropeGradParams.d              = D;
    td->ropeGradParams.blockNumB      = 2;
    td->ropeGradParams.blockFactorB   = 4;
    td->ropeGradParams.ubFactorS      = 4;
    td->ropeGradParams.ubFactorN      = 4;
    td->ropeGradParams.usedCoreNum    = 2;
    td->ropeGradParams.rotaryMode     = 0;
    td->ropeGradParams.layout         = 3;   // NO_BROADCAST
    td->ropeGradParams.dCosFlag       = 0;
    td->dCosFlag = 0;
    td->layout    = 3;

    // golden
    const float* fGradQ = reinterpret_cast<float*>(gradQ);
    const float* fGradK = reinterpret_cast<float*>(gradK);
    const float* fCos   = reinterpret_cast<float*>(cosBuf);
    const float* fSin   = reinterpret_cast<float*>(sinBuf);
    vector<float> goldenQ(totalNum, 0.0f), goldenK(totalNum, 0.0f);
    for (int r = 0; r < totalRow; r++) {
        RefHalfGrad(&goldenQ[r * D], fGradQ + r * D, fCos + r * D, fSin + r * D, D);
        RefHalfGrad(&goldenK[r * D], fGradK + r * D, fCos + r * D, fSin + r * D, D);
    }

    // Run kernel.
    std::function<void(PARAM_LIST_DEF)> func = [](PARAM_LIST_DEF) {
        apply_rotary_pos_emb_grad<true, false, 0, 0, 0, 205, 0>(PARAM_LIST);
    };
    ICPU_RUN_KF(func, 2, gradQ, gradK, cosBuf, sinBuf, queryBuf, keyBuf,
                    gradQOut, gradKOut, gradCosOut, gradSinOut, workspace, tiling);

    // Verify.
    const float* outQ = reinterpret_cast<float*>(gradQOut);
    const float* outK = reinterpret_cast<float*>(gradKOut);
    for (int i = 0; i < totalNum; i++) {
        EXPECT_NEAR(outQ[i], goldenQ[i], 1e-5f) << "gradQ idx=" << i;
    }
    for (int i = 0; i < totalNum; i++) {
        EXPECT_NEAR(outK[i], goldenK[i], 1e-5f) << "gradK idx=" << i;
    }

    AscendC::GmFree(gradQ);   AscendC::GmFree(gradK);
    AscendC::GmFree(cosBuf);  AscendC::GmFree(sinBuf);
    AscendC::GmFree(queryBuf); AscendC::GmFree(keyBuf);
    AscendC::GmFree(gradQOut); AscendC::GmFree(gradKOut);
    AscendC::GmFree(gradCosOut); AscendC::GmFree(gradSinOut);
    AscendC::GmFree(workspace);  AscendC::GmFree(tiling);
}

// ============================================================================
// Case 4: A(205) / dcos=1
//   Same shape as case 3, with dCosFlag=1.
//   workspace size = 2 * totalRow * D * sizeof(float) = 2048B.
//   golden: dx matches case 3; dcos/dsin = q*gradQ + k*gradK / rot(q)*gradQ + rot(k)*gradK.
// ============================================================================
TEST_F(apply_rotary_pos_emb_grad_test, a_dcos1)
{
    const int B = 2, S = 2, N = 2, D = 32;
    const int totalRow = B * S * N;  // 8
    const int totalNum = totalRow * D;  // 256
    const int wsSize = 2 * totalRow * D * sizeof(float);  // 2048

    uint8_t* gradQ   = (uint8_t*)AscendC::GmAlloc(totalNum * sizeof(float));
    uint8_t* gradK   = (uint8_t*)AscendC::GmAlloc(totalNum * sizeof(float));
    uint8_t* cosBuf  = (uint8_t*)AscendC::GmAlloc(totalNum * sizeof(float));
    uint8_t* sinBuf  = (uint8_t*)AscendC::GmAlloc(totalNum * sizeof(float));
    uint8_t* queryBuf = (uint8_t*)AscendC::GmAlloc(totalNum * sizeof(float));
    uint8_t* keyBuf   = (uint8_t*)AscendC::GmAlloc(totalNum * sizeof(float));
    uint8_t* gradQOut = (uint8_t*)AscendC::GmAlloc(totalNum * sizeof(float));
    uint8_t* gradKOut = (uint8_t*)AscendC::GmAlloc(totalNum * sizeof(float));
    uint8_t* gradCosOut = (uint8_t*)AscendC::GmAlloc(totalNum * sizeof(float));
    uint8_t* gradSinOut = (uint8_t*)AscendC::GmAlloc(totalNum * sizeof(float));
    uint8_t* workspace  = (uint8_t*)AscendC::GmAlloc(wsSize);
    uint8_t* tiling     = (uint8_t*)AscendC::GmAlloc(sizeof(ApplyRopeGradTilingData));

    // Fill random inputs.
    mt19937 rng(4567);
    uniform_real_distribution<float> dist(-1.0f, 1.0f);
    auto fill = [&](float* buf, int n) { for (int i = 0; i < n; i++) buf[i] = dist(rng); };
    fill(reinterpret_cast<float*>(gradQ),   totalNum);
    fill(reinterpret_cast<float*>(gradK),   totalNum);
    fill(reinterpret_cast<float*>(cosBuf),  totalNum);
    fill(reinterpret_cast<float*>(sinBuf),  totalNum);
    fill(reinterpret_cast<float*>(queryBuf), totalNum);
    fill(reinterpret_cast<float*>(keyBuf),   totalNum);
    memset(gradQOut, 0, totalNum * sizeof(float));
    memset(gradKOut, 0, totalNum * sizeof(float));
    memset(gradCosOut, 0, totalNum * sizeof(float));
    memset(gradSinOut, 0, totalNum * sizeof(float));

    // Fill tiling.
    memset(tiling, 0, sizeof(ApplyRopeGradTilingData));
    ApplyRopeGradTilingData* td = reinterpret_cast<ApplyRopeGradTilingData*>(tiling);
    td->ropeGradParams.b              = totalRow;
    td->ropeGradParams.d              = D;
    td->ropeGradParams.blockNumB      = 2;
    td->ropeGradParams.blockFactorB   = 4;
    td->ropeGradParams.ubFactorS      = 4;
    td->ropeGradParams.ubFactorN      = 4;
    td->ropeGradParams.usedCoreNum    = 2;
    td->ropeGradParams.rotaryMode     = 0;
    td->ropeGradParams.layout         = 3;
    td->ropeGradParams.dCosFlag       = 1;
    td->dCosFlag = 1;
    td->layout    = 3;

    // golden
    const float* fGradQ = reinterpret_cast<float*>(gradQ);
    const float* fGradK = reinterpret_cast<float*>(gradK);
    const float* fCos   = reinterpret_cast<float*>(cosBuf);
    const float* fSin   = reinterpret_cast<float*>(sinBuf);
    const float* fQ     = reinterpret_cast<float*>(queryBuf);
    const float* fK     = reinterpret_cast<float*>(keyBuf);
    vector<float> goldenQ(totalNum, 0.0f), goldenK(totalNum, 0.0f);
    vector<float> goldenCos(totalNum, 0.0f), goldenSin(totalNum, 0.0f);
    vector<float> rotQ(D), rotK(D);
    for (int r = 0; r < totalRow; r++) {
        // dx
        RefHalfGrad(&goldenQ[r * D], fGradQ + r * D, fCos + r * D, fSin + r * D, D);
        RefHalfGrad(&goldenK[r * D], fGradK + r * D, fCos + r * D, fSin + r * D, D);
        // dcos/dsin
        // gradCos = query * gradQ + key * gradK
        // gradSin = rot(query) * gradQ + rot(key) * gradK
        for (int i = 0; i < D; i++) {
            goldenCos[r * D + i] = fQ[r * D + i] * fGradQ[r * D + i] +
                                   fK[r * D + i] * fGradK[r * D + i];
        }
        RefHalfRotary(rotQ.data(), fQ + r * D, D);
        RefHalfRotary(rotK.data(), fK + r * D, D);
        for (int i = 0; i < D; i++) {
            goldenSin[r * D + i] = rotQ[i] * fGradQ[r * D + i] +
                                   rotK[i] * fGradK[r * D + i];
        }
    }

    // Run kernel.
    std::function<void(PARAM_LIST_DEF)> func = [](PARAM_LIST_DEF) {
        apply_rotary_pos_emb_grad<true, false, 0, 0, 0, 205, 1>(PARAM_LIST);
    };
    ICPU_RUN_KF(func, 2, gradQ, gradK, cosBuf, sinBuf, queryBuf, keyBuf,
                    gradQOut, gradKOut, gradCosOut, gradSinOut, workspace, tiling);

    // Verify.
    const float* outQ = reinterpret_cast<float*>(gradQOut);
    const float* outK = reinterpret_cast<float*>(gradKOut);
    const float* outCos = reinterpret_cast<float*>(gradCosOut);
    const float* outSin = reinterpret_cast<float*>(gradSinOut);
    for (int i = 0; i < totalNum; i++) {
        EXPECT_NEAR(outQ[i], goldenQ[i], 1e-5f) << "gradQ idx=" << i;
    }
    for (int i = 0; i < totalNum; i++) {
        EXPECT_NEAR(outK[i], goldenK[i], 1e-5f) << "gradK idx=" << i;
    }
    for (int i = 0; i < totalNum; i++) {
        EXPECT_NEAR(outCos[i], goldenCos[i], 1e-5f) << "gradCos idx=" << i;
    }
    for (int i = 0; i < totalNum; i++) {
        EXPECT_NEAR(outSin[i], goldenSin[i], 1e-5f) << "gradSin idx=" << i;
    }

    AscendC::GmFree(gradQ);   AscendC::GmFree(gradK);
    AscendC::GmFree(cosBuf);  AscendC::GmFree(sinBuf);
    AscendC::GmFree(queryBuf); AscendC::GmFree(keyBuf);
    AscendC::GmFree(gradQOut); AscendC::GmFree(gradKOut);
    AscendC::GmFree(gradCosOut); AscendC::GmFree(gradSinOut);
    AscendC::GmFree(workspace);  AscendC::GmFree(tiling);
}

// ============================================================================
// Case 5: AB(204) / dcos=1 / directPartial=1 (P2 部分积核内归约直写路径)
//   SBND, cosb=1 (B=1): bs=S=8, totalNQ=4, totalNK=2 (kernel nQ/nK 填 totalN)
//   B=1, S=8, nQ=4, nK=2, D=32 (f32, D/2 对齐)
//   blockNumBS=4, blockFactorBS=2, blockTailBS=2
//   blockNumN=1 (每核独占完整 N 范围), blockFactorN=4, blockTailN=4
//   ubFactorBS=2, ubFactorN=4 (单 N tile → directPartial=1)
//   cos 1S1D: (S, D), one row per s.
//   blockDim = blockNumBS * blockNumN = 4
//   golden: dx per row; gradCos/gradSin 在 s 行内对 nQ/nK 全量求和后直写 (S,D)。
// ============================================================================
TEST_F(apply_rotary_pos_emb_grad_test, ab_dcos1_direct)
{
    const int B = 1, S = 8, nQ = 4, nK = 2, D = 32;
    const int bsTotal = S;                  // cosb=1: bs = s
    const int gradQNum = bsTotal * nQ * D;  // 1024
    const int gradKNum = bsTotal * nK * D;  // 512
    const int cosNum   = S * D;             // 256

    uint8_t* gradQ   = (uint8_t*)AscendC::GmAlloc(gradQNum * sizeof(float));
    uint8_t* gradK   = (uint8_t*)AscendC::GmAlloc(gradKNum * sizeof(float));
    uint8_t* cosBuf  = (uint8_t*)AscendC::GmAlloc(cosNum   * sizeof(float));
    uint8_t* sinBuf  = (uint8_t*)AscendC::GmAlloc(cosNum   * sizeof(float));
    uint8_t* queryBuf = (uint8_t*)AscendC::GmAlloc(gradQNum * sizeof(float));
    uint8_t* keyBuf   = (uint8_t*)AscendC::GmAlloc(gradKNum * sizeof(float));
    uint8_t* gradQOut = (uint8_t*)AscendC::GmAlloc(gradQNum * sizeof(float));
    uint8_t* gradKOut = (uint8_t*)AscendC::GmAlloc(gradKNum * sizeof(float));
    uint8_t* gradCosOut = (uint8_t*)AscendC::GmAlloc(cosNum * sizeof(float));
    uint8_t* gradSinOut = (uint8_t*)AscendC::GmAlloc(cosNum * sizeof(float));
    uint8_t* workspace  = (uint8_t*)AscendC::GmAlloc(4096); // direct 模式 kernel 不写 ws, 仅需非空
    uint8_t* tiling     = (uint8_t*)AscendC::GmAlloc(sizeof(ApplyRopeGradTilingData));

    mt19937 rng(5678);
    uniform_real_distribution<float> dist(-1.0f, 1.0f);
    auto fill = [&](float* buf, int n) { for (int i = 0; i < n; i++) buf[i] = dist(rng); };
    fill(reinterpret_cast<float*>(gradQ),   gradQNum);
    fill(reinterpret_cast<float*>(gradK),   gradKNum);
    fill(reinterpret_cast<float*>(cosBuf),  cosNum);
    fill(reinterpret_cast<float*>(sinBuf),  cosNum);
    fill(reinterpret_cast<float*>(queryBuf), gradQNum);
    fill(reinterpret_cast<float*>(keyBuf),   gradKNum);
    memset(gradQOut, 0, gradQNum * sizeof(float));
    memset(gradKOut, 0, gradKNum * sizeof(float));
    memset(gradCosOut, 0, cosNum * sizeof(float));
    memset(gradSinOut, 0, cosNum * sizeof(float));

    // Fill tiling: nQ/nK 为 cosb==1 合并 B 后的 totalN (与 tiling_ab.cpp PostTiling 一致)
    memset(tiling, 0, sizeof(ApplyRopeGradTilingData));
    ApplyRopeGradTilingData* td = reinterpret_cast<ApplyRopeGradTilingData*>(tiling);
    td->ropeGradABParams.b              = B;
    td->ropeGradABParams.s              = S;
    td->ropeGradABParams.d              = D;
    td->ropeGradABParams.nQ             = nQ;
    td->ropeGradABParams.nK             = nK;
    td->ropeGradABParams.dAlign         = 32;
    td->ropeGradABParams.dSplitCoef     = 2;
    td->ropeGradABParams.blockNumBS     = 4;
    td->ropeGradABParams.blockFactorBS  = 2;
    td->ropeGradABParams.blockTailBS    = 2;
    td->ropeGradABParams.blockNumN      = 1;
    td->ropeGradABParams.blockFactorN   = 4;
    td->ropeGradABParams.blockTailN     = 4;
    td->ropeGradABParams.ubFactorBS     = 2;
    td->ropeGradABParams.ubFactorN      = 4;
    td->ropeGradABParams.usedCoreNum    = 4;
    td->ropeGradABParams.rotaryMode     = 0;
    td->ropeGradABParams.directPartial  = 1;
    td->dCosFlag = 1;
    td->layout    = 1;

    // golden
    const float* fGradQ = reinterpret_cast<float*>(gradQ);
    const float* fGradK = reinterpret_cast<float*>(gradK);
    const float* fCos   = reinterpret_cast<float*>(cosBuf);
    const float* fSin   = reinterpret_cast<float*>(sinBuf);
    const float* fQ     = reinterpret_cast<float*>(queryBuf);
    const float* fK     = reinterpret_cast<float*>(keyBuf);
    vector<float> goldenQ(gradQNum, 0.0f), goldenK(gradKNum, 0.0f);
    vector<float> goldenCos(cosNum, 0.0f), goldenSin(cosNum, 0.0f);
    vector<float> rot(D);
    for (int s = 0; s < S; s++) {
        const float* cosRow = fCos + s * D;
        const float* sinRow = fSin + s * D;
        for (int n = 0; n < nQ; n++) {
            int base = (s * nQ + n) * D;
            RefHalfGrad(&goldenQ[base], fGradQ + base, cosRow, sinRow, D);
        }
        for (int n = 0; n < nK; n++) {
            int base = (s * nK + n) * D;
            RefHalfGrad(&goldenK[base], fGradK + base, cosRow, sinRow, D);
        }
        // gradCos[s] = sum_n q*gq + sum_n k*gk; gradSin[s] = sum_n rot(q)*gq + sum_n rot(k)*gk
        for (int n = 0; n < nQ; n++) {
            int base = (s * nQ + n) * D;
            RefHalfRotary(rot.data(), fQ + base, D);
            for (int i = 0; i < D; i++) {
                goldenCos[s * D + i] += fQ[base + i] * fGradQ[base + i];
                goldenSin[s * D + i] += rot[i] * fGradQ[base + i];
            }
        }
        for (int n = 0; n < nK; n++) {
            int base = (s * nK + n) * D;
            RefHalfRotary(rot.data(), fK + base, D);
            for (int i = 0; i < D; i++) {
                goldenCos[s * D + i] += fK[base + i] * fGradK[base + i];
                goldenSin[s * D + i] += rot[i] * fGradK[base + i];
            }
        }
    }

    // Run kernel: AB(204) / dcos=1
    std::function<void(PARAM_LIST_DEF)> func = [](PARAM_LIST_DEF) {
        apply_rotary_pos_emb_grad<true, false, 0, 0, 0, 204, 1>(PARAM_LIST);
    };
    ICPU_RUN_KF(func, 4, gradQ, gradK, cosBuf, sinBuf, queryBuf, keyBuf,
                    gradQOut, gradKOut, gradCosOut, gradSinOut, workspace, tiling);

    // Verify.
    const float* outQ   = reinterpret_cast<float*>(gradQOut);
    const float* outK   = reinterpret_cast<float*>(gradKOut);
    const float* outCos = reinterpret_cast<float*>(gradCosOut);
    const float* outSin = reinterpret_cast<float*>(gradSinOut);
    for (int i = 0; i < gradQNum; i++) {
        EXPECT_NEAR(outQ[i], goldenQ[i], 1e-5f) << "gradQ idx=" << i;
    }
    for (int i = 0; i < gradKNum; i++) {
        EXPECT_NEAR(outK[i], goldenK[i], 1e-5f) << "gradK idx=" << i;
    }
    for (int i = 0; i < cosNum; i++) {
        EXPECT_NEAR(outCos[i], goldenCos[i], 1e-4f) << "gradCos idx=" << i;
    }
    for (int i = 0; i < cosNum; i++) {
        EXPECT_NEAR(outSin[i], goldenSin[i], 1e-4f) << "gradSin idx=" << i;
    }

    AscendC::GmFree(gradQ);   AscendC::GmFree(gradK);
    AscendC::GmFree(cosBuf);  AscendC::GmFree(sinBuf);
    AscendC::GmFree(queryBuf); AscendC::GmFree(keyBuf);
    AscendC::GmFree(gradQOut); AscendC::GmFree(gradKOut);
    AscendC::GmFree(gradCosOut); AscendC::GmFree(gradSinOut);
    AscendC::GmFree(workspace);  AscendC::GmFree(tiling);
}
