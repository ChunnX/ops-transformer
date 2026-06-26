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
 * \file test_inplace_partial_rotary_mul_grad_kernel.cpp
 * \brief OpKernel UT for InplacePartialRotaryMulGrad using TTK (ESL/CA simulation)
 *
 * Coverage per template:
 *   BAB (203): BSND, cosb_==1
 *   AB  (204): SBND
 *   ABA (201): BNSD, cosb_!=1
 *   BA  (202): BNSD, cosb_==1
 *   A   (205): NO_BROADCAST
 *   B   (206): BROADCAST_BSN
 *   EMPTY (403): sliceLength==0
 *
 * Requires:
 *   - Data scripts: tests/ut/op_kernel/inplace_partial_rotary_mul_grad_data/gen_data.py
 *   - Tiling script: tests/ut/op_kernel/inplace_partial_rotary_mul_grad_data/gen_tiling.py
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <cmath>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_inplace_partial_rotary_mul_grad_kernel.h"
#include "data_utils.h"

using namespace std;

extern "C" __global__ __aicore__ void inplace_partial_rotary_mul_grad(
    GM_ADDR dy, GM_ADDR cos, GM_ADDR sin, GM_ADDR x,
    GM_ADDR dx, GM_ADDR dcos, GM_ADDR dsin,
    GM_ADDR workspace, GM_ADDR tiling);

class InplacePartialRotaryMulGradKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "InplacePartialRotaryMulGradKernelTest SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "InplacePartialRotaryMulGradKernelTest TearDown\n" << endl;
    }

    void RunTest(const vector<int64_t>& dyShape, const vector<int64_t>& cosShape,
                 const string& dyDtype, const string& cosDtype,
                 uint64_t tilingKey, uint32_t blockDim,
                 int64_t sliceStart, int64_t sliceEnd,
                 const string& layout,
                 const vector<float>& dyData, const vector<float>& cosData,
                 const vector<float>& sinData, const vector<float>& refDx)
    {
        int64_t dyDtypeSize = (dyDtype == "float32") ? 4 : 2;
        int64_t cosDtypeSize = (cosDtype == "float32") ? 4 : 2;

        size_t dyByteSize = 1;
        for (auto d : dyShape) dyByteSize *= d;
        dyByteSize *= dyDtypeSize;

        size_t cosByteSize = 1;
        for (auto d : cosShape) cosByteSize *= d;
        cosByteSize *= cosDtypeSize;

        size_t tilingByteSize = sizeof(InplacePartialRotaryMulGradUTTilingData);

        // Allocate GM memory
        uint8_t* dyGm = (uint8_t*)AscendC::GmAlloc(dyByteSize);
        uint8_t* cosGm = (uint8_t*)AscendC::GmAlloc(cosByteSize);
        uint8_t* sinGm = (uint8_t*)AscendC::GmAlloc(cosByteSize);
        uint8_t* dxGm = (uint8_t*)AscendC::GmAlloc(dyByteSize);  // inplace: dx overwrites dy
        uint8_t* dcosGm = (uint8_t*)AscendC::GmAlloc(cosByteSize);
        uint8_t* dsinGm = (uint8_t*)AscendC::GmAlloc(cosByteSize);
        uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
        uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingByteSize);

        // Copy host data to GM
        if (dyDtype == "float32") {
            AscendC::GmMemcpy(dyGm, (uint8_t*)dyData.data(), dyByteSize);
        }
        AscendC::GmMemcpy(cosGm, (uint8_t*)cosData.data(), cosByteSize);
        AscendC::GmMemcpy(sinGm, (uint8_t*)sinData.data(), cosByteSize);

        // Set up tiling data with minimal valid values
        auto* tilingData = reinterpret_cast<InplacePartialRotaryMulGradUTTilingData*>(tiling);
        tilingData->b = dyShape[0];
        tilingData->s = dyShape[1];
        tilingData->n = dyShape[2];
        tilingData->d = dyShape[3];
        tilingData->blockNumB = blockDim;
        tilingData->blockFactorB = 1;
        tilingData->blockNumS = 1;
        tilingData->blockFactorS = 1;
        tilingData->ubFactorS = 1;
        tilingData->ubFactorB = 1;
        tilingData->ubLoopNumN = 1;
        tilingData->ubFactorN = 1;
        tilingData->ubTailFactorN = 1;
        tilingData->usedCoreNum = blockDim;
        tilingData->rotaryMode = 1; // interleave
        tilingData->sliceStart = sliceStart;
        tilingData->sliceEnd = sliceEnd;
        tilingData->sliceLength = sliceEnd - sliceStart;
        tilingData->dSplitCoef = 1;

        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        ICPU_SET_TILING_KEY(tilingKey);
        ICPU_RUN_KF(inplace_partial_rotary_mul_grad, blockDim,
                    dyGm, cosGm, sinGm, nullptr,  // x=nullptr (optional)
                    dxGm, dcosGm, dsinGm,
                    workspace, tiling);

        // Read back dx (inplace on dy)
        vector<float> resultDy(dyByteSize / sizeof(float), 0);
        AscendC::GmMemcpy((uint8_t*)resultDy.data(), dxGm, dyByteSize);

        // Verify
        const float epsilon = 1e-5f;
        int64_t totalElems = dyByteSize / sizeof(float);
        bool allPassed = true;
        for (int64_t i = 0; i < totalElems; i++) {
            if (std::fabs(resultDy[i] - refDx[i]) >= epsilon) {
                cout << "MISMATCH at [" << i << "]: got " << resultDy[i]
                     << ", expected " << refDx[i] << " (ref " << refDx[i] << ")" << endl;
                allPassed = false;
                break;  // stop on first mismatch
            }
        }
        EXPECT_TRUE(allPassed) << "Result mismatch for tilingKey=" << tilingKey
                               << " layout=" << layout;

        AscendC::GmFree(dyGm);
        AscendC::GmFree(cosGm);
        AscendC::GmFree(sinGm);
        AscendC::GmFree(dxGm);
        AscendC::GmFree(dcosGm);
        AscendC::GmFree(dsinGm);
        AscendC::GmFree(workspace);
        AscendC::GmFree(tiling);
    }
};

// ===================== BAB (203): BSND, cosb_==1 =====================
TEST_F(InplacePartialRotaryMulGradKernelTest, BAB_BSND_interleave)
{
    int64_t B = 2, S = 2, N = 2, D = 8;
    int64_t sliceStart = 2, sliceEnd = 6;
    int64_t cosD = sliceEnd - sliceStart;

    vector<int64_t> dyShape = {B, S, N, D};
    vector<int64_t> cosShape = {1, S, 1, cosD};

    // dy: each (B,S,N) group = [1,2,3,4,5,6,7,8]
    // cos: S=2 groups of [0.5,0.6,0.7,0.8]
    // sin: S=2 groups of [0.1,0.2,0.3,0.4]
    // Expected per-group after interleave on [2,6): [1,2,2.3,2.1,5.9,3.3,7,8]
    vector<float> dyData, expectedDx;
    for (int i = 0; i < B * S * N; i++) {
        dyData.insert(dyData.end(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
        expectedDx.insert(expectedDx.end(), {1.0f, 2.0f, 2.3f, 2.1f, 5.9f, 3.3f, 7.0f, 8.0f});
    }
    vector<float> cosData = {0.5f, 0.6f, 0.7f, 0.8f, 0.5f, 0.6f, 0.7f, 0.8f};
    vector<float> sinData = {0.1f, 0.2f, 0.3f, 0.4f, 0.1f, 0.2f, 0.3f, 0.4f};

    vector<float> refDx(expectedDx.size(), 0.0f);
    InterleaveGradRefBSND(dyData, cosData, sinData, refDx, B, S, N, D, sliceStart, sliceEnd);

    RunTest(dyShape, cosShape, "float32", "float32", TILING_KEY_BAB, B * S,
            sliceStart, sliceEnd, "BSND", dyData, cosData, sinData, refDx);
}

// ===================== ABA (201): BNSD, cosb_!=1 =====================
TEST_F(InplacePartialRotaryMulGradKernelTest, ABA_BNSD_interleave)
{
    int64_t B = 2, N = 2, S = 2, D = 8;
    int64_t sliceStart = 2, sliceEnd = 6;
    int64_t cosD = sliceEnd - sliceStart;

    vector<int64_t> dyShape = {B, N, S, D};
    vector<int64_t> cosShape = {B, 1, S, cosD};

    // dy: BNSD layout, each (B,N,S) = [1..8]
    vector<float> dyData;
    for (int i = 0; i < B * N * S; i++) {
        dyData.insert(dyData.end(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    }
    // cos: B=2, S=2 → shape [2,1,2,4], each (B,S) group = [0.5,0.6,0.7,0.8]
    vector<float> cosData = {
        0.5f, 0.6f, 0.7f, 0.8f,  // B=0,S=0
        0.9f, 1.0f, 1.1f, 1.2f,  // B=0,S=1
        1.3f, 1.4f, 1.5f, 1.6f,  // B=1,S=0
        1.7f, 1.8f, 1.9f, 2.0f   // B=1,S=1
    };
    vector<float> sinData = {
        0.1f, 0.2f, 0.3f, 0.4f,
        0.5f, 0.6f, 0.7f, 0.8f,
        0.9f, 1.0f, 1.1f, 1.2f,
        1.3f, 1.4f, 1.5f, 1.6f
    };

    vector<float> refDx(dyData.size(), 0.0f);
    InterleaveGradRefBNSD(dyData, cosData, sinData, refDx, B, N, S, D, sliceStart, sliceEnd, false);

    RunTest(dyShape, cosShape, "float32", "float32", TILING_KEY_ABA, B * S,
            sliceStart, sliceEnd, "BNSD", dyData, cosData, sinData, refDx);
}

// ===================== BA (202): BNSD, cosb_==1 =====================
TEST_F(InplacePartialRotaryMulGradKernelTest, BA_BNSD_interleave)
{
    int64_t B = 2, N = 2, S = 2, D = 8;
    int64_t sliceStart = 2, sliceEnd = 6;
    int64_t cosD = sliceEnd - sliceStart;

    vector<int64_t> dyShape = {B, N, S, D};
    vector<int64_t> cosShape = {1, 1, S, cosD};

    vector<float> dyData;
    for (int i = 0; i < B * N * S; i++) {
        dyData.insert(dyData.end(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    }
    // cos: S=2 broad-cast on B,N
    vector<float> cosData = {0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};
    vector<float> sinData = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};

    vector<float> refDx(dyData.size(), 0.0f);
    InterleaveGradRefBNSD(dyData, cosData, sinData, refDx, B, N, S, D, sliceStart, sliceEnd, true);

    RunTest(dyShape, cosShape, "float32", "float32", TILING_KEY_BA, B * S,
            sliceStart, sliceEnd, "BNSD_BA", dyData, cosData, sinData, refDx);
}

// ===================== AB (204): SBND =====================
TEST_F(InplacePartialRotaryMulGradKernelTest, AB_SBND_interleave)
{
    int64_t S = 2, B = 2, N = 2, D = 8;
    int64_t sliceStart = 2, sliceEnd = 6;
    int64_t cosD = sliceEnd - sliceStart;

    vector<int64_t> dyShape = {S, B, N, D};
    vector<int64_t> cosShape = {S, 1, 1, cosD};

    // dy: SBND layout
    vector<float> dyData;
    for (int i = 0; i < S * B * N; i++) {
        dyData.insert(dyData.end(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    }
    vector<float> cosData = {0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};
    vector<float> sinData = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};

    vector<float> refDx(dyData.size(), 0.0f);
    InterleaveGradRefSBND(dyData, cosData, sinData, refDx, S, B, N, D, sliceStart, sliceEnd);

    RunTest(dyShape, cosShape, "float32", "float32", TILING_KEY_AB, S * B,
            sliceStart, sliceEnd, "SBND", dyData, cosData, sinData, refDx);
}

// ===================== A (205): NO_BROADCAST =====================
TEST_F(InplacePartialRotaryMulGradKernelTest, A_NO_BROADCAST_interleave)
{
    int64_t B = 2, N = 2, S = 2, D = 4;
    int64_t sliceStart = 0, sliceEnd = 4;

    // After MergeDim: dy = [B*N*S, D] = [8, 4]
    vector<int64_t> dyShape = {B, N, S, D};  // BNSD form
    vector<int64_t> cosShape = {B, N, S, D};  // full broadcast

    vector<float> dyData(B * N * S * D);
    vector<float> cosData(B * N * S * D);
    vector<float> sinData(B * N * S * D);
    for (int i = 0; i < B * N * S; i++) {
        dyData[i * D + 0] = 1.0f;
        dyData[i * D + 1] = 2.0f;
        dyData[i * D + 2] = 3.0f;
        dyData[i * D + 3] = 4.0f;
        cosData[i * D + 0] = 0.5f;
        cosData[i * D + 1] = 0.6f;
        cosData[i * D + 2] = 0.7f;
        cosData[i * D + 3] = 0.8f;
        sinData[i * D + 0] = 0.1f;
        sinData[i * D + 1] = 0.2f;
        sinData[i * D + 2] = 0.3f;
        sinData[i * D + 3] = 0.4f;
    }

    vector<float> refDx(dyData.size(), 0.0f);
    InterleaveGradRefAB(dyData, cosData, sinData, refDx, B, N, S, D, sliceStart, sliceEnd, false);

    RunTest(dyShape, cosShape, "float32", "float32", TILING_KEY_A, B,
            sliceStart, sliceEnd, "NO_BROADCAST", dyData, cosData, sinData, refDx);
}

// ===================== B (206): BROADCAST_BSN =====================
TEST_F(InplacePartialRotaryMulGradKernelTest, B_BROADCAST_BSN_interleave)
{
    int64_t B = 2, N = 2, S = 2, D = 4;
    int64_t sliceStart = 0, sliceEnd = 4;

    vector<int64_t> dyShape = {B, N, S, D};  // BNSD form
    vector<int64_t> cosShape = {1, 1, 1, D};  // broadcast all dims

    vector<float> dyData(B * N * S * D);
    for (int i = 0; i < B * N * S; i++) {
        dyData[i * D + 0] = 1.0f;
        dyData[i * D + 1] = 2.0f;
        dyData[i * D + 2] = 3.0f;
        dyData[i * D + 3] = 4.0f;
    }
    vector<float> cosData = {0.5f, 0.6f, 0.7f, 0.8f};
    vector<float> sinData = {0.1f, 0.2f, 0.3f, 0.4f};

    vector<float> refDx(dyData.size(), 0.0f);
    InterleaveGradRefAB(dyData, cosData, sinData, refDx, B, N, S, D, sliceStart, sliceEnd, true);

    RunTest(dyShape, cosShape, "float32", "float32", TILING_KEY_B, B,
            sliceStart, sliceEnd, "BROADCAST_BSN", dyData, cosData, sinData, refDx);
}

// ===================== EMPTY (403): sliceLength==0 =====================
TEST_F(InplacePartialRotaryMulGradKernelTest, EMPTY_slice)
{
    int64_t B = 2, S = 2, N = 2, D = 8;
    int64_t sliceStart = 4, sliceEnd = 4;  // empty slice

    vector<int64_t> dyShape = {B, S, N, D};
    vector<int64_t> cosShape = {1, S, 1, 0};

    vector<float> dyData(B * S * N * D);
    for (int i = 0; i < B * S * N; i++) {
        dyData[i * D + 0] = 1.0f;
        dyData[i * D + 1] = 2.0f;
        dyData[i * D + 2] = 3.0f;
        dyData[i * D + 3] = 4.0f;
        dyData[i * D + 4] = 5.0f;
        dyData[i * D + 5] = 6.0f;
        dyData[i * D + 6] = 7.0f;
        dyData[i * D + 7] = 8.0f;
    }
    vector<float> cosData = {};
    vector<float> sinData = {};

    // Empty slice: dx = dy (no change)
    vector<float> refDx = dyData;

    RunTest(dyShape, cosShape, "float32", "float32", TILING_KEY_EMPTY, 1,
            sliceStart, sliceEnd, "EMPTY", dyData, cosData, sinData, refDx);
}
