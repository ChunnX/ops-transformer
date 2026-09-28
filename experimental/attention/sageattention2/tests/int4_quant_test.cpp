/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* Standalone unit test suite for Native INT4 QK Quantization and Bit-Packing Engine.
 *
 * Portable C++17 test verifying numerical accuracy, saturation clamping bounds,
 * bit-exact lossless pack/unpack reversibility, SQNR conformance (> 24 dB),
 * and Catlass Atlas 950PR Cube MMAD fractal layout alignment.
 *
 * Compilation & Execution:
 *   g++ -O2 -std=c++17 tests/int4_quant_test.cpp -o build/int4_quant_test
 *   ./build/int4_quant_test
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "../op_kernel/quant_int4.h"

namespace {

int g_failures = 0;

void Report(const char *name, double measured, double bound, bool ok)
{
    std::printf("  %-48s val = %-12.6g bound = %-10g  %s\n",
                name, measured, bound, ok ? "PASS" : "FAIL");
    if (!ok) {
        ++g_failures;
    }
}

void ReportExact(const char *name, int passed, int total)
{
    bool ok = (passed == total);
    std::printf("  %-48s passed = %d/%-9d status = %s\n",
                name, passed, total, ok ? "PASS" : "FAIL");
    if (!ok) {
        ++g_failures;
    }
}

// ============================================================================
// Test 1: Boundary Saturation & Clamping Tests
// ============================================================================
void TestBoundarySaturation()
{
    std::printf("[1] INT4 Boundary Saturation & Clamp Tests\n");

    const float scale = 1.0f;
    int clampFailures = 0;

    // Check extreme positive saturation
    for (float v : {7.0f, 7.1f, 7.5f, 8.0f, 15.0f, 127.0f, 1e5f, std::numeric_limits<float>::infinity()}) {
        int8_t q = Sage::Int4::QuantizeInt4(v, scale);
        if (q != Sage::Int4::INT4_MAX) {
            ++clampFailures;
        }
    }
    ReportExact("positive saturation clamp (+7)", 8 - clampFailures, 8);

    // Check extreme negative saturation
    int negClampFailures = 0;
    for (float v : {-8.0f, -8.1f, -8.5f, -9.0f, -16.0f, -128.0f, -1e5f, -std::numeric_limits<float>::infinity()}) {
        int8_t q = Sage::Int4::QuantizeInt4(v, scale);
        if (q != Sage::Int4::INT4_MIN) {
            ++negClampFailures;
        }
    }
    ReportExact("negative saturation clamp (-8)", 8 - negClampFailures, 8);

    // Check NaN safety: maps cleanly to 0
    int8_t qNan = Sage::Int4::QuantizeInt4(std::numeric_limits<float>::quiet_NaN(), scale);
    Report("NaN safe handling (maps to 0)", static_cast<double>(qNan), 0.0, qNan == 0);

    // Verify exact integer quantization within linear range [-8, 7]
    int linearFailures = 0;
    for (int expected = -8; expected <= 7; ++expected) {
        float inVal = static_cast<float>(expected);
        int8_t q = Sage::Int4::QuantizeInt4(inVal, scale);
        if (q != expected) {
            ++linearFailures;
        }
    }
    ReportExact("exact integer quantization [-8, 7]", 16 - linearFailures, 16);

    // Check round-to-nearest transition points with scale = 2.0
    const float scale2 = 2.0f;
    // 3.25 * 2.0 = 6.5 -> rounds to 6 or 7, within bounds
    int8_t qUpper = Sage::Int4::QuantizeInt4(3.49f, scale2); // 6.98 -> 7
    int8_t qSat = Sage::Int4::QuantizeInt4(3.75f, scale2);   // 7.50 -> 7 (clamped)
    int8_t qLower = Sage::Int4::QuantizeInt4(-3.99f, scale2); // -7.98 -> -8
    int8_t qNegSat = Sage::Int4::QuantizeInt4(-4.25f, scale2); // -8.50 -> -8 (clamped)

    bool transitionsOk = (qUpper == 7) && (qSat == 7) && (qLower == -8) && (qNegSat == -8);
    Report("scaled round and clamp transitions", transitionsOk ? 1.0 : 0.0, 1.0, transitionsOk);
}

// ============================================================================
// Test 2: Bit-Exact Lossless Pack / Unpack Reversibility (All 256 Combinations)
// ============================================================================
void TestPackUnpackReversibility()
{
    std::printf("[2] Bit-Exact Lossless Pack / Unpack Reversibility\n");

    // Test 2a: Exhaustive test over all 256 (low, high) integer pairs in [-8, 7] x [-8, 7]
    int pairPassCount = 0;
    for (int high = Sage::Int4::INT4_MIN; high <= Sage::Int4::INT4_MAX; ++high) {
        for (int low = Sage::Int4::INT4_MIN; low <= Sage::Int4::INT4_MAX; ++low) {
            int8_t b0 = static_cast<int8_t>(low);
            int8_t b1 = static_cast<int8_t>(high);

            uint8_t packed = Sage::Int4::PackInt4Pair(b0, b1);

            int8_t out0 = 0, out1 = 0;
            Sage::Int4::UnpackInt4Pair(packed, out0, out1);

            int8_t lowDirect = Sage::Int4::UnpackInt4Low(packed);
            int8_t highDirect = Sage::Int4::UnpackInt4High(packed);

            if (out0 == b0 && out1 == b1 && lowDirect == b0 && highDirect == b1) {
                ++pairPassCount;
            }
        }
    }
    ReportExact("all 256 [-8, 7]x[-8, 7] pairs lossless roundtrip", pairPassCount, 256);

    // Test 2b: Exhaustive test over all 256 possible packed byte values [0, 255]
    int bytePassCount = 0;
    for (int b = 0; b < 256; ++b) {
        uint8_t originalByte = static_cast<uint8_t>(b);
        int8_t low = 0, high = 0;
        Sage::Int4::UnpackInt4Pair(originalByte, low, high);

        // Verify decoded values fall strictly within INT4 limits
        if (low < Sage::Int4::INT4_MIN || low > Sage::Int4::INT4_MAX) continue;
        if (high < Sage::Int4::INT4_MIN || high > Sage::Int4::INT4_MAX) continue;

        uint8_t repacked = Sage::Int4::PackInt4Pair(low, high);
        if (repacked == originalByte) {
            ++bytePassCount;
        }
    }
    ReportExact("all 256 byte codes pack/unpack bijection", bytePassCount, 256);
}

// ============================================================================
// Test 3: SQNR Validation (> 24 dB on Smooth Distributions)
// ============================================================================
void TestSqnrConformance()
{
    std::printf("[3] INT4 Quantization Noise & SQNR Validation (> 24 dB)\n");

    // 3a: Smooth harmonic distribution (representing normalized RoPE / rotary embeddings)
    constexpr size_t N_SAMPLES = 8192;
    std::vector<float> xSin(N_SAMPLES);
    std::vector<float> xSinDequant(N_SAMPLES);
    std::vector<uint8_t> packedSin(N_SAMPLES / 2);

    float amaxSin = 0.0f;
    for (size_t i = 0; i < N_SAMPLES; ++i) {
        float t = static_cast<float>(i) * (8.0f * 3.141592653589793f / static_cast<float>(N_SAMPLES));
        xSin[i] = std::sin(t);
        amaxSin = std::fmax(amaxSin, std::fabs(xSin[i]));
    }

    float scaleSin = 0.0f, invScaleSin = 0.0f;
    Sage::Int4::CalcInt4Scale(amaxSin, scaleSin, invScaleSin);
    Sage::Int4::QuantizeAndPackRow(xSin.data(), packedSin.data(), static_cast<uint32_t>(N_SAMPLES), scaleSin);
    Sage::Int4::UnpackAndDequantizeRow(packedSin.data(), xSinDequant.data(), static_cast<uint32_t>(N_SAMPLES), invScaleSin);

    double signalPowerSin = 0.0;
    double noisePowerSin = 0.0;
    double maxErrSin = 0.0;
    for (size_t i = 0; i < N_SAMPLES; ++i) {
        double s = xSin[i];
        double err = s - xSinDequant[i];
        signalPowerSin += s * s;
        noisePowerSin += err * err;
        maxErrSin = std::fmax(maxErrSin, std::fabs(err));
    }
    double sqnrSin = 10.0 * std::log10(signalPowerSin / noisePowerSin);

    Report("sinusoidal harmonic SQNR (> 24.0 dB)", sqnrSin, 24.0, sqnrSin > 24.0);
    // Theoretical max quantization step error = 0.5 * invScale
    Report("sinusoidal max quant error bound (<= 0.5*invScale)", maxErrSin, 0.5 * invScaleSin + 1e-6,
           maxErrSin <= 0.5 * invScaleSin + 1e-6);

    // 3b: Smooth Hann-windowed centered activation distribution
    std::vector<float> xHann(N_SAMPLES);
    std::vector<float> xHannDequant(N_SAMPLES);
    std::vector<uint8_t> packedHann(N_SAMPLES / 2);

    float amaxHann = 0.0f;
    for (size_t i = 0; i < N_SAMPLES; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(N_SAMPLES);
        xHann[i] = 0.5f * (1.0f - std::cos(2.0f * 3.141592653589793f * t)) - 0.5f;
        amaxHann = std::fmax(amaxHann, std::fabs(xHann[i]));
    }

    float scaleHann = 0.0f, invScaleHann = 0.0f;
    Sage::Int4::CalcInt4Scale(amaxHann, scaleHann, invScaleHann);
    Sage::Int4::QuantizeAndPackRow(xHann.data(), packedHann.data(), static_cast<uint32_t>(N_SAMPLES), scaleHann);
    Sage::Int4::UnpackAndDequantizeRow(packedHann.data(), xHannDequant.data(), static_cast<uint32_t>(N_SAMPLES), invScaleHann);

    double signalPowerHann = 0.0;
    double noisePowerHann = 0.0;
    for (size_t i = 0; i < N_SAMPLES; ++i) {
        double s = xHann[i];
        double err = s - xHannDequant[i];
        signalPowerHann += s * s;
        noisePowerHann += err * err;
    }
    double sqnrHann = 10.0 * std::log10(signalPowerHann / noisePowerHann);
    Report("Hann-windowed activation SQNR (> 24.0 dB)", sqnrHann, 24.0, sqnrHann > 24.0);

    // 3c: Centered sigmoid smooth activation curve
    std::vector<float> xSigmoid(N_SAMPLES);
    std::vector<float> xSigmoidDequant(N_SAMPLES);
    std::vector<uint8_t> packedSigmoid(N_SAMPLES / 2);

    float amaxSig = 0.0f;
    for (size_t i = 0; i < N_SAMPLES; ++i) {
        float z = -4.0f + 8.0f * static_cast<float>(i) / static_cast<float>(N_SAMPLES);
        xSigmoid[i] = (1.0f / (1.0f + std::exp(-z))) - 0.5f;
        amaxSig = std::fmax(amaxSig, std::fabs(xSigmoid[i]));
    }

    float scaleSig = 0.0f, invScaleSig = 0.0f;
    Sage::Int4::CalcInt4Scale(amaxSig, scaleSig, invScaleSig);
    Sage::Int4::QuantizeAndPackRow(xSigmoid.data(), packedSigmoid.data(), static_cast<uint32_t>(N_SAMPLES), scaleSig);
    Sage::Int4::UnpackAndDequantizeRow(packedSigmoid.data(), xSigmoidDequant.data(), static_cast<uint32_t>(N_SAMPLES), invScaleSig);

    double signalPowerSig = 0.0;
    double noisePowerSig = 0.0;
    for (size_t i = 0; i < N_SAMPLES; ++i) {
        double s = xSigmoid[i];
        double err = s - xSigmoidDequant[i];
        signalPowerSig += s * s;
        noisePowerSig += err * err;
    }
    double sqnrSig = 10.0 * std::log10(signalPowerSig / noisePowerSig);
    Report("centered sigmoid activation SQNR (> 24.0 dB)", sqnrSig, 24.0, sqnrSig > 24.0);
}

// ============================================================================
// Test 4: Atlas 950PR Catlass Cube MMAD Layout Engine & Fractal Alignment
// ============================================================================
void TestCatlassLayoutEngine()
{
    std::printf("[4] Atlas 950PR Catlass Cube MMAD Layout & Fractal Alignment\n");

    // 4a: Geometry round-up tests
    uint32_t roundM = Sage::Int4::LayoutEngine::RoundUpM(25);
    uint32_t roundK = Sage::Int4::LayoutEngine::RoundUpK(100);
    bool geomOk = (roundM == 32) && (roundK == 128);
    Report("fractal dimension alignment (M->32, K->128)", geomOk ? 1.0 : 0.0, 1.0, geomOk);

    // 4b: End-to-end ND to Fractal zZ and reverse roundtrip test
    // Shape: M = 32, K = 128 -> exactly 2 x 2 = 4 fractals of 512 bytes each = 2048 bytes
    const uint32_t M = 32;
    const uint32_t K = 128;
    const uint32_t packedStride = K / 2; // 64 bytes per row
    std::vector<uint8_t> origNd(M * packedStride);

    for (size_t i = 0; i < origNd.size(); ++i) {
        origNd[i] = static_cast<uint8_t>(i & 0xFF);
    }

    uint32_t totalFractalBytes = Sage::Int4::LayoutEngine::TotalFractalBytes(M, K);
    std::vector<uint8_t> fractalZz(totalFractalBytes, 0);
    std::vector<uint8_t> reconstructedNd(M * packedStride, 0);

    Sage::Int4::LayoutEngine::NdToFractalZz(origNd.data(), fractalZz.data(), M, K, packedStride);
    Sage::Int4::LayoutEngine::FractalZzToNd(fractalZz.data(), reconstructedNd.data(), M, K, packedStride);

    int diffBytes = 0;
    for (size_t i = 0; i < origNd.size(); ++i) {
        if (origNd[i] != reconstructedNd[i]) {
            ++diffBytes;
        }
    }
    ReportExact("ND -> Fractal zZ -> ND bit-exact roundtrip (32x128)",
                static_cast<int>(origNd.size()) - diffBytes, static_cast<int>(origNd.size()));

    // 4c: Non-aligned shape roundtrip (M = 20, K = 70)
    const uint32_t M_odd = 20;
    const uint32_t K_odd = 70;
    const uint32_t strideOdd = K_odd / 2; // 35 bytes
    std::vector<uint8_t> origOdd(M_odd * strideOdd);
    for (size_t i = 0; i < origOdd.size(); ++i) {
        origOdd[i] = static_cast<uint8_t>((i * 7 + 13) & 0xFF);
    }

    uint32_t oddFractalBytes = Sage::Int4::LayoutEngine::TotalFractalBytes(M_odd, K_odd);
    std::vector<uint8_t> fractalOdd(oddFractalBytes, 0);
    std::vector<uint8_t> reconstructedOdd(M_odd * strideOdd, 0);

    Sage::Int4::LayoutEngine::NdToFractalZz(origOdd.data(), fractalOdd.data(), M_odd, K_odd, strideOdd);
    Sage::Int4::LayoutEngine::FractalZzToNd(fractalOdd.data(), reconstructedOdd.data(), M_odd, K_odd, strideOdd);

    int diffOdd = 0;
    for (size_t i = 0; i < origOdd.size(); ++i) {
        if (origOdd[i] != reconstructedOdd[i]) {
            ++diffOdd;
        }
    }
    ReportExact("odd dimensions padded fractal roundtrip (20x70)",
                static_cast<int>(origOdd.size()) - diffOdd, static_cast<int>(origOdd.size()));
}

// ============================================================================
// Test 5: Full Row Quantization & Auto-Scale Pipeline
// ============================================================================
void TestRowQuantizationPipeline(std::mt19937 &rng)
{
    std::printf("[5] Full Row Quantization & Auto-Scale Pipeline\n");

    const uint32_t len = 128;
    std::vector<float> row(len);
    std::uniform_real_distribution<float> dist(-12.5f, 12.5f);
    for (uint32_t i = 0; i < len; ++i) {
        row[i] = dist(rng);
    }

    std::vector<uint8_t> packed(len / 2);
    float scale = 0.0f, invScale = 0.0f;
    Sage::Int4::QuantizePackRowAuto(row.data(), packed.data(), len, scale, invScale);

    std::vector<float> reconstructed(len);
    Sage::Int4::UnpackAndDequantizeRow(packed.data(), reconstructed.data(), len, invScale);

    double maxErr = 0.0;
    for (uint32_t i = 0; i < len; ++i) {
        maxErr = std::fmax(maxErr, std::fabs(row[i] - reconstructed[i]));
    }
    double allowedErr = 0.5 * static_cast<double>(invScale) + 1e-5;
    Report("row auto-scale max quantization error (<= 0.5*invScale)", maxErr, allowedErr, maxErr <= allowedErr);
}

} // namespace

int main()
{
    std::mt19937 rng(20260929u);
    std::printf("================================================================\n");
    std::printf("Atlas 950PR Native INT4 Quantization & Bit-Packing Engine Unit Test\n");
    std::printf("================================================================\n");

    TestBoundarySaturation();
    TestPackUnpackReversibility();
    TestSqnrConformance();
    TestCatlassLayoutEngine();
    TestRowQuantizationPipeline(rng);

    std::printf("================================================================\n");
    if (g_failures == 0) {
        std::printf("ALL INT4 QUANTIZATION TESTS PASSED (100%% OK)\n");
        return 0;
    }
    std::printf("%d INT4 QUANTIZATION TEST(S) FAILED\n", g_failures);
    return 1;
}
