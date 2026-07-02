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
 * \brief OpKernel UT for InplacePartialRotaryMulGrad using TTK simulation
 *
 * Each test sets up template-specific tiling data matching the kernel's expectations.
 */

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

extern "C" __global__ __aicore__ void inplace_partial_rotary_mul_grad(GM_ADDR dy, GM_ADDR cos, GM_ADDR sin, GM_ADDR dx,
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

    // Generic GM alloc + copy
    void AllocAndCopy(uint8_t *&dyGm, uint8_t *&cosGm, uint8_t *&sinGm, uint8_t *&dxGm, uint8_t *&workspace,
                      uint8_t *&tilingGm, const vector<float> &dyData, const vector<float> &cosData,
                      const vector<float> &sinData, size_t dyBytes, size_t cosBytes, size_t tilingBytes)
    {
        dyGm = (uint8_t *)AscendC::GmAlloc(dyBytes);
        cosGm = (uint8_t *)AscendC::GmAlloc(cosBytes);
        sinGm = (uint8_t *)AscendC::GmAlloc(cosBytes);
        dxGm = (uint8_t *)AscendC::GmAlloc(dyBytes);
        workspace = (uint8_t *)AscendC::GmAlloc(16 * 1024 * 1024);
        tilingGm = (uint8_t *)AscendC::GmAlloc(tilingBytes);

        AscendC::GmMemcpy(dyGm, (uint8_t *)dyData.data(), dyBytes);
        AscendC::GmMemcpy(cosGm, (uint8_t *)cosData.data(), cosBytes);
        AscendC::GmMemcpy(sinGm, (uint8_t *)sinData.data(), cosBytes);
    }

    void FreeAll(uint8_t *dyGm, uint8_t *cosGm, uint8_t *sinGm, uint8_t *dxGm, uint8_t *workspace, uint8_t *tilingGm)
    {
        AscendC::GmFree(dyGm);
        AscendC::GmFree(cosGm);
        AscendC::GmFree(sinGm);
        AscendC::GmFree(dxGm);
        AscendC::GmFree(workspace);
        AscendC::GmFree(tilingGm);
    }

    // ---- BAB template: BSND layout, cosb_==1 ----
    void RunBAB(const vector<int64_t> &dyShape, const vector<int64_t> &cosShape, const vector<float> &dyData,
                const vector<float> &cosData, const vector<float> &sinData, const vector<float> &refDx,
                int64_t sliceStart, int64_t sliceEnd)
    {
        int64_t B = dyShape[0], S = dyShape[1], N = dyShape[2], D = dyShape[3];
        int64_t cosD = sliceEnd - sliceStart;
        size_t dyBytes = B * S * N * D * sizeof(float);
        size_t cosBytes = cosShape[0] * cosShape[1] * cosShape[2] * cosShape[3] * sizeof(float);
        size_t tilingBytes = sizeof(InplacePartialRotaryMulGradUTTilingData);

        uint8_t *dyGm, *cosGm, *sinGm, *dxGm, *workspace, *tilingGm;
        AllocAndCopy(dyGm, cosGm, sinGm, dxGm, workspace, tilingGm, dyData, cosData, sinData, dyBytes, cosBytes,
                     tilingBytes);

        auto *td = reinterpret_cast<InplacePartialRotaryMulGradUTTilingData *>(tilingGm);
        td->b = B;
        td->s = S;
        td->d = D;
        td->n = N;
        td->blockNumB = B;
        td->blockFactorB = 1; // one block per B
        td->blockNumS = 1;
        td->blockFactorS = S; // one block for all S
        td->ubFactorS = 1;
        td->ubFactorB = 1;
        td->ubLoopNumN = N;
        td->ubFactorN = 1;
        td->ubTailFactorN = 1;
        td->usedCoreNum = B * 1; // blockNumB * blockNumS
        td->rotaryMode = 1;
        td->dSplitCoef = 1;
        td->sliceStart = sliceStart;
        td->sliceEnd = sliceEnd;
        td->sliceLength = cosD;

        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        ICPU_SET_TILING_KEY(TILING_KEY_BAB);
        ICPU_RUN_KF(inplace_partial_rotary_mul_grad, static_cast<uint32_t>(td->usedCoreNum), dyGm, cosGm, sinGm, dxGm,
                    workspace, tilingGm);

        vector<float> result(dyBytes / sizeof(float));
        AscendC::GmMemcpy((uint8_t *)result.data(), dxGm, dyBytes);
        Verify(result, refDx, "BAB");
        FreeAll(dyGm, cosGm, sinGm, dxGm, workspace, tilingGm);
    }

    // ---- ABA&BA template: BNSD layout ----
    void RunABAorBA(const vector<int64_t> &dyShape, const vector<int64_t> &cosShape, const vector<float> &dyData,
                    const vector<float> &cosData, const vector<float> &sinData, const vector<float> &refDx,
                    int64_t sliceStart, int64_t sliceEnd, uint64_t tilingKey, bool isBroadCast)
    {
        int64_t B = dyShape[0], N = dyShape[1], S = dyShape[2], D = dyShape[3];
        int64_t cosD = sliceEnd - sliceStart;
        size_t dyBytes = B * N * S * D * sizeof(float);
        size_t cosBytes = cosShape[0] * cosShape[1] * cosShape[2] * cosShape[3] * sizeof(float);
        size_t tilingBytes = sizeof(InplacePartialRotaryMulGradUTTilingData);

        uint8_t *dyGm, *cosGm, *sinGm, *dxGm, *workspace, *tilingGm;
        AllocAndCopy(dyGm, cosGm, sinGm, dxGm, workspace, tilingGm, dyData, cosData, sinData, dyBytes, cosBytes,
                     tilingBytes);

        auto *td = reinterpret_cast<InplacePartialRotaryMulGradUTTilingData *>(tilingGm);
        td->b = B;
        td->s = S;
        td->d = D;
        td->n = N;
        td->blockNumB = B;
        td->blockFactorB = 1;
        td->blockNumS = S;
        td->blockFactorS = 1;
        td->ubFactorB = 1;
        td->ubFactorS = 1;
        td->ubLoopNumN = 0;
        td->ubFactorN = 1;
        td->ubTailFactorN = 0;
        td->usedCoreNum = B * S;
        td->rotaryMode = 1;
        td->dSplitCoef = 1;
        td->sliceStart = sliceStart;
        td->sliceEnd = sliceEnd;
        td->sliceLength = cosD;

        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        ICPU_SET_TILING_KEY(tilingKey);
        ICPU_RUN_KF(inplace_partial_rotary_mul_grad, static_cast<uint32_t>(td->usedCoreNum), dyGm, cosGm, sinGm, dxGm,
                    workspace, tilingGm);

        vector<float> result(dyBytes / sizeof(float));
        AscendC::GmMemcpy((uint8_t *)result.data(), dxGm, dyBytes);
        Verify(result, refDx, isBroadCast ? "BA" : "ABA");
        FreeAll(dyGm, cosGm, sinGm, dxGm, workspace, tilingGm);
    }

    // ---- AB template: SBND layout ----
    void RunAB(const vector<int64_t> &dyShape, const vector<int64_t> &cosShape, const vector<float> &dyData,
               const vector<float> &cosData, const vector<float> &sinData, const vector<float> &refDx,
               int64_t sliceStart, int64_t sliceEnd)
    {
        int64_t S = dyShape[0], B = dyShape[1], N = dyShape[2], D = dyShape[3];
        // AB for SBND+cosb==1: bs = s (B folded into N), n_eff = b * n
        int64_t bs = S;
        int64_t nEff = B * N;
        int64_t cosD = sliceEnd - sliceStart;
        size_t dyBytes = S * B * N * D * sizeof(float);
        size_t cosBytes = cosShape[0] * cosShape[1] * cosShape[2] * cosShape[3] * sizeof(float);
        size_t tilingBytes = sizeof(InplacePartialRotaryMulGradABUTTilingData);

        uint8_t *dyGm, *cosGm, *sinGm, *dxGm, *workspace, *tilingGm;
        AllocAndCopy(dyGm, cosGm, sinGm, dxGm, workspace, tilingGm, dyData, cosData, sinData, dyBytes, cosBytes,
                     tilingBytes);

        auto *td = reinterpret_cast<InplacePartialRotaryMulGradABUTTilingData *>(tilingGm);
        td->b = B;
        td->s = S;
        td->d = D;
        td->n = nEff;
        td->dAlign = 0;
        td->dSplitCoef = 1;
        td->blockNumBS = bs;
        td->blockFactorBS = 1;
        td->blockTailBS = 1;
        td->blockNumN = 1;
        td->blockFactorN = nEff;
        td->blockTailN = nEff;
        td->ubFactorBS = 1;
        td->ubFactorN = 1;
        td->usedCoreNum = bs;
        td->rotaryMode = 1;
        td->sliceStart = sliceStart;
        td->sliceEnd = sliceEnd;
        td->sliceLength = cosD;

        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        ICPU_SET_TILING_KEY(TILING_KEY_AB);
        ICPU_RUN_KF(inplace_partial_rotary_mul_grad, static_cast<uint32_t>(td->usedCoreNum), dyGm, cosGm, sinGm, dxGm,
                    workspace, tilingGm);

        vector<float> result(dyBytes / sizeof(float));
        AscendC::GmMemcpy((uint8_t *)result.data(), dxGm, dyBytes);
        Verify(result, refDx, "AB");
        FreeAll(dyGm, cosGm, sinGm, dxGm, workspace, tilingGm);
    }

    // ---- A&B template: NO_BROADCAST / BROADCAST_BSN (merged B) ----
    void RunAorB(const vector<int64_t> &dyShape, const vector<int64_t> &cosShape, const vector<float> &dyData,
                 const vector<float> &cosData, const vector<float> &sinData, const vector<float> &refDx,
                 int64_t sliceStart, int64_t sliceEnd, uint64_t tilingKey)
    {
        int64_t B = dyShape[0] * dyShape[1] * dyShape[2]; // merged
        int64_t D = dyShape[3];
        int64_t cosD = sliceEnd - sliceStart;
        size_t dyBytes = B * D * sizeof(float);
        size_t cosBytes = cosShape[0] * cosShape[1] * cosShape[2] * cosShape[3] * sizeof(float);
        size_t tilingBytes = sizeof(InplacePartialRotaryMulGradUTTilingData);

        uint8_t *dyGm, *cosGm, *sinGm, *dxGm, *workspace, *tilingGm;
        AllocAndCopy(dyGm, cosGm, sinGm, dxGm, workspace, tilingGm, dyData, cosData, sinData, dyBytes, cosBytes,
                     tilingBytes);

        auto *td = reinterpret_cast<InplacePartialRotaryMulGradUTTilingData *>(tilingGm);
        td->b = B;
        td->s = 1;
        td->d = D;
        td->n = 1; // after MergeDim
        td->blockNumB = (B > 1) ? 2 : 1;
        td->blockFactorB = (B > 1) ? B / 2 : B;
        td->blockNumS = 1;
        td->blockFactorS = 1;
        td->ubFactorB = 1;
        td->ubFactorS = 1;
        td->ubLoopNumN = 0;
        td->ubFactorN = 1;
        td->usedCoreNum = td->blockNumB;
        td->rotaryMode = 1;
        td->dSplitCoef = 1;
        td->sliceStart = sliceStart;
        td->sliceEnd = sliceEnd;
        td->sliceLength = cosD;

        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        ICPU_SET_TILING_KEY(tilingKey);
        ICPU_RUN_KF(inplace_partial_rotary_mul_grad, static_cast<uint32_t>(td->usedCoreNum), dyGm, cosGm, sinGm, dxGm,
                    workspace, tilingGm);

        vector<float> result(dyBytes / sizeof(float));
        AscendC::GmMemcpy((uint8_t *)result.data(), dxGm, dyBytes);
        Verify(result, refDx, (tilingKey == TILING_KEY_A) ? "A" : "B");
        FreeAll(dyGm, cosGm, sinGm, dxGm, workspace, tilingGm);
    }

    // ---- EMPTY slice: just verify no-op ----
    void RunEmpty(const vector<float> &dyData, const vector<float> &refDx, const vector<int64_t> &dyShape,
                  int64_t sliceStart, int64_t sliceEnd)
    {
        size_t dyBytes = dyData.size() * sizeof(float);
        size_t cosBytes = 0;
        size_t tilingBytes = sizeof(InplacePartialRotaryMulGradUTTilingData);

        uint8_t *dyGm, *cosGm, *sinGm, *dxGm, *workspace, *tilingGm;
        vector<float> emptyCos;
        AllocAndCopy(dyGm, cosGm, sinGm, dxGm, workspace, tilingGm, dyData, emptyCos, emptyCos, dyBytes, cosBytes,
                     tilingBytes);

        auto *td = reinterpret_cast<InplacePartialRotaryMulGradUTTilingData *>(tilingGm);
        td->usedCoreNum = 1;
        td->sliceStart = sliceStart;
        td->sliceEnd = sliceEnd;
        td->sliceLength = 0;

        AscendC::SetKernelMode(KernelMode::AIV_MODE);
        ICPU_SET_TILING_KEY(TILING_KEY_EMPTY);
        ICPU_RUN_KF(inplace_partial_rotary_mul_grad, 1, dyGm, cosGm, sinGm, dxGm, workspace, tilingGm);

        vector<float> result(dyBytes / sizeof(float));
        AscendC::GmMemcpy((uint8_t *)result.data(), dxGm, dyBytes);
        Verify(result, refDx, "EMPTY");
        FreeAll(dyGm, cosGm, sinGm, dxGm, workspace, tilingGm);
    }

    void Verify(const vector<float> &result, const vector<float> &ref, const string &tag)
    {
        const float epsilon = 1e-5f;
        bool ok = true;
        for (size_t i = 0; i < ref.size(); i++) {
            if (std::fabs(result[i] - ref[i]) >= epsilon) {
                cout << "[" << tag << "] MISMATCH at [" << i << "]: got " << result[i] << ", expected " << ref[i]
                     << endl;
                ok = false;
                break;
            }
        }
        EXPECT_TRUE(ok) << tag << " result mismatch";
    }
};

// ===================== BAB (203): BSND, cosb_==1 =====================
TEST_F(InplacePartialRotaryMulGradKernelTest, BAB_BSND_interleave)
{
    int64_t B = 2, S = 2, N = 2, D = 8;
    int64_t sliceStart = 2, sliceEnd = 6, cosD = sliceEnd - sliceStart;

    vector<int64_t> dyShape = {B, S, N, D};
    vector<int64_t> cosShape = {1, S, 1, cosD};

    vector<float> dyData, expectedDx;
    for (int i = 0; i < B * S * N; i++) {
        dyData.insert(dyData.end(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
        expectedDx.insert(expectedDx.end(), {1.0f, 2.0f, 2.3f, 2.1f, 5.9f, 3.3f, 7.0f, 8.0f});
    }
    vector<float> cosData = {0.5f, 0.6f, 0.7f, 0.8f, 0.5f, 0.6f, 0.7f, 0.8f};
    vector<float> sinData = {0.1f, 0.2f, 0.3f, 0.4f, 0.1f, 0.2f, 0.3f, 0.4f};

    vector<float> refDx(expectedDx.size(), 0.0f);
    InterleaveGradRefBSND(dyData, cosData, sinData, refDx, B, S, N, D, sliceStart, sliceEnd);

    RunBAB(dyShape, cosShape, dyData, cosData, sinData, refDx, sliceStart, sliceEnd);
}

// ===================== ABA (201): BNSD, cosb_!=1 =====================
TEST_F(InplacePartialRotaryMulGradKernelTest, ABA_BNSD_interleave)
{
    int64_t B = 2, N = 2, S = 2, D = 8;
    int64_t sliceStart = 2, sliceEnd = 6, cosD = sliceEnd - sliceStart;

    vector<int64_t> dyShape = {B, N, S, D};
    vector<int64_t> cosShape = {B, 1, S, cosD};

    vector<float> dyData;
    for (int i = 0; i < B * N * S; i++) {
        dyData.insert(dyData.end(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    }
    // B=2,S=2: cos[B,1,S,cosD] — each (B,S) unique
    vector<float> cosData = {0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f,
                             1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f, 1.9f, 2.0f};
    vector<float> sinData = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
                             0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f};

    vector<float> refDx(dyData.size(), 0.0f);
    InterleaveGradRefBNSD(dyData, cosData, sinData, refDx, B, N, S, D, sliceStart, sliceEnd, false);

    RunABAorBA(dyShape, cosShape, dyData, cosData, sinData, refDx, sliceStart, sliceEnd, TILING_KEY_ABA, false);
}

// ===================== BA (202): BNSD, cosb_==1 =====================
TEST_F(InplacePartialRotaryMulGradKernelTest, BA_BNSD_interleave)
{
    int64_t B = 2, N = 2, S = 2, D = 8;
    int64_t sliceStart = 2, sliceEnd = 6, cosD = sliceEnd - sliceStart;

    vector<int64_t> dyShape = {B, N, S, D};
    vector<int64_t> cosShape = {1, 1, S, cosD};

    vector<float> dyData;
    for (int i = 0; i < B * N * S; i++) {
        dyData.insert(dyData.end(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    }
    vector<float> cosData = {0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};
    vector<float> sinData = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};

    vector<float> refDx(dyData.size(), 0.0f);
    InterleaveGradRefBNSD(dyData, cosData, sinData, refDx, B, N, S, D, sliceStart, sliceEnd, true);

    RunABAorBA(dyShape, cosShape, dyData, cosData, sinData, refDx, sliceStart, sliceEnd, TILING_KEY_BA, true);
}

// ===================== AB (204): SBND =====================
TEST_F(InplacePartialRotaryMulGradKernelTest, AB_SBND_interleave)
{
    int64_t S = 2, B = 2, N = 2, D = 8;
    int64_t sliceStart = 2, sliceEnd = 6, cosD = sliceEnd - sliceStart;

    vector<int64_t> dyShape = {S, B, N, D};
    vector<int64_t> cosShape = {S, 1, 1, cosD};

    vector<float> dyData;
    for (int i = 0; i < S * B * N; i++) {
        dyData.insert(dyData.end(), {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f});
    }
    vector<float> cosData = {0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f};
    vector<float> sinData = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};

    vector<float> refDx(dyData.size(), 0.0f);
    InterleaveGradRefSBND(dyData, cosData, sinData, refDx, S, B, N, D, sliceStart, sliceEnd);

    RunAB(dyShape, cosShape, dyData, cosData, sinData, refDx, sliceStart, sliceEnd);
}

// ===================== A (205): NO_BROADCAST =====================
TEST_F(InplacePartialRotaryMulGradKernelTest, A_NO_BROADCAST_interleave)
{
    int64_t B = 2, N = 2, S = 2, D = 4;
    int64_t sliceStart = 0, sliceEnd = 4;
    int64_t mergedB = B * N * S; // 8

    vector<int64_t> dyShape = {B, N, S, D};
    vector<int64_t> cosShape = {B, N, S, D};

    vector<float> dyData(mergedB * D), cosData(mergedB * D), sinData(mergedB * D);
    for (int i = 0; i < mergedB; i++) {
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

    RunAorB(dyShape, cosShape, dyData, cosData, sinData, refDx, sliceStart, sliceEnd, TILING_KEY_A);
}

// ===================== B (206): BROADCAST_BSN =====================
TEST_F(InplacePartialRotaryMulGradKernelTest, B_BROADCAST_BSN_interleave)
{
    int64_t B = 2, N = 2, S = 2, D = 4;
    int64_t sliceStart = 0, sliceEnd = 4;
    int64_t mergedB = B * N * S;

    vector<int64_t> dyShape = {B, N, S, D};
    vector<int64_t> cosShape = {1, 1, 1, D};

    vector<float> dyData(mergedB * D);
    for (int i = 0; i < mergedB; i++) {
        dyData[i * D + 0] = 1.0f;
        dyData[i * D + 1] = 2.0f;
        dyData[i * D + 2] = 3.0f;
        dyData[i * D + 3] = 4.0f;
    }
    vector<float> cosData = {0.5f, 0.6f, 0.7f, 0.8f};
    vector<float> sinData = {0.1f, 0.2f, 0.3f, 0.4f};

    vector<float> refDx(dyData.size(), 0.0f);
    InterleaveGradRefAB(dyData, cosData, sinData, refDx, B, N, S, D, sliceStart, sliceEnd, true);

    RunAorB(dyShape, cosShape, dyData, cosData, sinData, refDx, sliceStart, sliceEnd, TILING_KEY_B);
}

// ===================== EMPTY (403): sliceLength==0 =====================
TEST_F(InplacePartialRotaryMulGradKernelTest, EMPTY_slice)
{
    int64_t B = 2, S = 2, N = 2, D = 8;
    int64_t sliceStart = 4, sliceEnd = 4;

    vector<int64_t> dyShape = {B, S, N, D};
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
    vector<float> refDx = dyData; // no change

    RunEmpty(dyData, refDx, dyShape, sliceStart, sliceEnd);
}
