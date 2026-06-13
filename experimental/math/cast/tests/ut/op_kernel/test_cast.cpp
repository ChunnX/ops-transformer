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
 * \file test_cast.cpp
 * \brief Cast ascend310p kernel UT (tikicpulib CPU model + numeric assertions).
 *        Covers the software paths the device/aclnn simulation could not validate end-to-end:
 *        bf16->f32 bit ops, i64<->f32 split/sign-extend, plus f16->f32 /
 *        f32->i32(NaN->0) regressions. Directly instantiates CastKernel<X,Y> per pair (entry symbol
 *        is fixed to one dtype pair at compile time), self-allocates + memset0 GM, and EXPECT_EQ/NEAR
 *        every element against an independent host oracle (numpy-equivalent bit math).
 */
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <iostream>
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#endif

// Pull in the kernel TU for direct CastKernel<X,Y> instantiation (no duplicate cast entry symbol).
#define CAST_UT_NO_ENTRY
#include "../../../op_kernel/cast.cpp"

using namespace std;
using cast_310p::CastKernel;

class cast_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "cast_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "cast_test TearDown\n" << endl; }
};

// ---- tiling helper: fill a self-consistent CastTilingData for total N elements over blockNum blocks.
// N must be divisible by blockNum; each block does exactly one UB loop of (N/blockNum) elements.
static void FillTiling(CastTilingData* t, int64_t N, int64_t blockNum)
{
    int64_t blockFormer = N / blockNum;  // caller guarantees divisibility
    t->blockNum = blockNum;
    t->ubFormer = blockFormer;
    t->blockFormer = blockFormer;
    t->ubLoopOfFormerBlock = 1;
    t->ubLoopOfTailBlock = 1;
    t->ubTailOfFormerBlock = blockFormer;
    t->ubTailOfTailBlock = blockFormer;
}

// ---- bit helpers (host oracle, independent of kernel) ----
// reference bf16 -> f32 (bit splice into high16).
static uint32_t ref_bf16_to_f32(uint16_t b) { return static_cast<uint32_t>(b) << 16; }

// generic GM-driven kernel launcher templated on (X,Y). Returns raw output bytes.
template <int X, int Y>
static void RunCast(CastTilingData* tiling, uint8_t* xGm, uint8_t* yGm)
{
    uint8_t* ws = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    memset(ws, 0, 16 * 1024 * 1024);
#if (__CCE_AICORE__ == 200)
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
#else
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
#endif
    ICPU_SET_TILING_KEY(0);
    auto launch = [](GM_ADDR x, GM_ADDR y, GM_ADDR w, GM_ADDR tl) {
        CastKernel<X, Y> op;
        CastTilingData td;
        memcpy(&td, tl, sizeof(CastTilingData));
        op.Init(x, y, td);
        op.Process();
    };
    ICPU_RUN_KF(static_cast<void (*)(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR)>(launch),
                static_cast<uint32_t>(tiling->blockNum), xGm, yGm, ws, reinterpret_cast<uint8_t*>(tiling));
    AscendC::GmFree(ws);
}

// allocate a 32B-padded, memset0 GM buffer
static uint8_t* AllocGm(size_t bytes)
{
    uint8_t* p = (uint8_t*)AscendC::GmAlloc(bytes + 32);
    memset(p, 0, bytes + 32);
    return p;
}


// Tile a vector of meaningful values to length n (so per-block UB tensor >=128 bytes for the
// hardware Cast intrinsic, which the CPU model enforces). Oracle still checks every element.
template <typename T>
static std::vector<T> Tile(const std::vector<T>& v, int n)
{
    std::vector<T> out(n);
    for (int i = 0; i < n; ++i) out[i] = v[i % v.size()];
    return out;
}
static constexpr int64_t NTOT = 64;     // total elements
static constexpr int64_t NBLK = 2;      // 32 elems/block -> f32/i32 buffer = 128 bytes

// ============================ bf16 -> f32 ============================
// boundary bf16 bit patterns: +0,-0,1.0,-2.0,+Inf,-Inf,NaN,0x7FFF(max qNaN payload),min-normal,large.
// Output f32 must equal the exact bit-spliced value (lossless), incl NaN/Inf bit patterns.
TEST_F(cast_test, bf16_to_f32_boundaries)
{
    std::vector<uint16_t> vals = {
        0x0000, 0x8000, 0x3F80, 0xC000, 0x7F80, 0xFF80, 0x7FC0, 0x7FFF, 0x0080, 0x7F7F,
        0x4049, 0xBF80, 0x0001, 0x4120, 0xC120, 0x3F00};
    auto in = Tile(vals, NTOT);
    CastTilingData t; FillTiling(&t, NTOT, NBLK);
    uint8_t* x = AllocGm(NTOT * sizeof(uint16_t));
    uint8_t* y = AllocGm(NTOT * sizeof(float));
    memcpy(x, in.data(), NTOT * sizeof(uint16_t));
    RunCast<DT_BF16, DT_FLOAT>(&t, x, y);
    const uint32_t* yo = reinterpret_cast<const uint32_t*>(y);
    for (int i = 0; i < NTOT; ++i)
        EXPECT_EQ(yo[i], ref_bf16_to_f32(in[i])) << "i=" << i << " bf16=0x" << std::hex << in[i];
    AscendC::GmFree(x); AscendC::GmFree(y);
}


// ============================ i64 -> f32 ============================
TEST_F(cast_test, i64_to_f32_within_2p31)
{
    std::vector<int64_t> vals = {0, 1, -1, 2, -2, 1000000, -1000000,
                                 2147483647LL, -2147483648LL, 123456789LL, -123456789LL,
                                 16777217LL, -16777217LL, 100LL, -100LL, 65536LL};
    auto in = Tile(vals, NTOT);
    CastTilingData t; FillTiling(&t, NTOT, NBLK);
    uint8_t* x = AllocGm(NTOT * sizeof(int64_t));
    uint8_t* y = AllocGm(NTOT * sizeof(float));
    memcpy(x, in.data(), NTOT * sizeof(int64_t));
    RunCast<DT_INT64, DT_FLOAT>(&t, x, y);
    const float* yo = reinterpret_cast<const float*>(y);
    for (int i = 0; i < NTOT; ++i)
        EXPECT_FLOAT_EQ(yo[i], static_cast<float>(in[i])) << "i=" << i << " v=" << in[i];
    AscendC::GmFree(x); AscendC::GmFree(y);
}

// ============================ f32 -> i64 ============================
TEST_F(cast_test, f32_to_i64_within_2p31)
{
    std::vector<float> vals = {0.0f, 1.4f, 1.6f, -1.4f, -1.6f, 2.5f, 3.5f, -2.5f,
                               100.0f, -100.0f, 65536.0f, -65536.0f,
                               2147483520.0f, -2147483648.0f, 12345.0f, -0.5f};
    auto in = Tile(vals, NTOT);
    CastTilingData t; FillTiling(&t, NTOT, NBLK);
    uint8_t* x = AllocGm(NTOT * sizeof(float));
    uint8_t* y = AllocGm(NTOT * sizeof(int64_t));
    memcpy(x, in.data(), NTOT * sizeof(float));
    RunCast<DT_FLOAT, DT_INT64>(&t, x, y);
    const int64_t* yo = reinterpret_cast<const int64_t*>(y);
    for (int i = 0; i < NTOT; ++i) {
        long ref = std::lrint(in[i]);
        EXPECT_EQ(yo[i], static_cast<int64_t>(ref)) << "i=" << i << " v=" << in[i];
    }
    AscendC::GmFree(x); AscendC::GmFree(y);
}

// ============================ f16 -> f32 (regression) ============================
TEST_F(cast_test, f16_to_f32_regression)
{
    std::vector<float> src = {0.0f, 1.0f, -1.0f, 2.5f, -2.5f, 0.5f, -0.5f, 100.0f,
                              -100.0f, 0.25f, 1024.0f, -1024.0f, 3.140625f, -7.0f, 0.125f, 255.0f};
    auto sv = Tile(src, NTOT);
    std::vector<uint16_t> in(NTOT);
    for (int i = 0; i < NTOT; ++i) { half h = static_cast<half>(sv[i]); memcpy(&in[i], &h, 2); }
    CastTilingData t; FillTiling(&t, NTOT, NBLK);
    uint8_t* x = AllocGm(NTOT * sizeof(uint16_t));
    uint8_t* y = AllocGm(NTOT * sizeof(float));
    memcpy(x, in.data(), NTOT * sizeof(uint16_t));
    RunCast<DT_FLOAT16, DT_FLOAT>(&t, x, y);
    const float* yo = reinterpret_cast<const float*>(y);
    for (int i = 0; i < NTOT; ++i) {
        half h; memcpy(&h, &in[i], 2);
        EXPECT_FLOAT_EQ(yo[i], static_cast<float>(h)) << "i=" << i;
    }
    AscendC::GmFree(x); AscendC::GmFree(y);
}

// ============================ f32 -> i32, NaN -> 0 (regression) ============================
TEST_F(cast_test, f32_to_i32_nan_to_zero_regression)
{
    std::vector<float> vals = {0.0f, 1.4f, 1.6f, -1.4f, -1.6f, 2.5f, 3.5f, -2.5f,
                               bits_f32(0x7FC00000u), bits_f32(0xFFC00000u),
                               100.0f, -100.0f, 2147483520.0f, -2147483648.0f, 65536.0f, -65536.0f};
    auto in = Tile(vals, NTOT);
    CastTilingData t; FillTiling(&t, NTOT, NBLK);
    uint8_t* x = AllocGm(NTOT * sizeof(float));
    uint8_t* y = AllocGm(NTOT * sizeof(int32_t));
    memcpy(x, in.data(), NTOT * sizeof(float));
    RunCast<DT_FLOAT, DT_INT32>(&t, x, y);
    const int32_t* yo = reinterpret_cast<const int32_t*>(y);
    for (int i = 0; i < NTOT; ++i) {
        int32_t ref = std::isnan(in[i]) ? 0 : static_cast<int32_t>(std::lrint(in[i]));
        EXPECT_EQ(yo[i], ref) << "i=" << i << " v=" << in[i];
    }
    AscendC::GmFree(x); AscendC::GmFree(y);
}
