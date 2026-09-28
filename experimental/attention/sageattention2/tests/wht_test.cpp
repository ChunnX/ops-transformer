/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define SAGE_STANDALONE_TEST 1
#include "../op_kernel/wht.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>

namespace {

int g_failures = 0;

void Report(const char *name, double measured, double bound, bool ok)
{
    std::printf("  %-48s measured = %-12.6g bound = %-10g  %s\n", name, measured, bound, ok ? "PASS" : "FAIL");
    if (!ok) {
        ++g_failures;
    }
}

// Compute sample kurtosis: mu4 / (variance^2)
template <typename T>
double ComputeKurtosis(const std::vector<T> &x)
{
    const size_t n = x.size();
    if (n == 0) return 0.0;
    double mean = 0.0;
    for (auto v : x) mean += static_cast<double>(v);
    mean /= n;

    double m2 = 0.0, m4 = 0.0;
    for (auto v : x) {
        double d = static_cast<double>(v) - mean;
        m2 += d * d;
        m4 += d * d * d * d;
    }
    m2 /= n;
    m4 /= n;
    if (m2 < 1e-15) return 0.0;
    return m4 / (m2 * m2);
}

// -------------------------------------------------------------------------------------------------
// Test 1: Orthogonality & Reconstruction: |H_D * H_D^T - I| < 1e-12
// -------------------------------------------------------------------------------------------------
template <uint32_t D>
void TestOrthogonality()
{
    std::vector<double> H(D * D, 0.0);
    Sage::FastWalshHadamardTransform<D>::GenerateMatrix(H.data(), true);

    // Verify symmetry: H == H^T
    double maxSymmErr = 0.0;
    for (uint32_t r = 0; r < D; ++r) {
        for (uint32_t c = 0; c < D; ++c) {
            double diff = std::fabs(H[r * D + c] - H[c * D + r]);
            maxSymmErr = std::fmax(maxSymmErr, diff);
        }
    }

    // Verify H * H^T == I
    double maxOrthoErr = 0.0;
    for (uint32_t r = 0; r < D; ++r) {
        for (uint32_t c = 0; c < D; ++c) {
            double dot = 0.0;
            for (uint32_t k = 0; k < D; ++k) {
                dot += H[r * D + k] * H[c * D + k];
            }
            double expected = (r == c) ? 1.0 : 0.0;
            maxOrthoErr = std::fmax(maxOrthoErr, std::fabs(dot - expected));
        }
    }

    // Verify FWHT on basis vectors matches explicit matrix columns
    double maxBasisErr = 0.0;
    for (uint32_t i = 0; i < D; ++i) {
        std::vector<double> basis(D, 0.0);
        basis[i] = 1.0;
        Sage::FastWalshHadamardTransform<D>(basis.data(), true);
        for (uint32_t j = 0; j < D; ++j) {
            double diff = std::fabs(basis[j] - H[i * D + j]);
            maxBasisErr = std::fmax(maxBasisErr, diff);
        }
    }

    char nameSymm[64], nameOrtho[64], nameBasis[64];
    std::snprintf(nameSymm, sizeof(nameSymm), "D=%u Hadamard symmetry |H - H^T|", D);
    std::snprintf(nameOrtho, sizeof(nameOrtho), "D=%u Orthogonality |H * H^T - I|", D);
    std::snprintf(nameBasis, sizeof(nameBasis), "D=%u FWHT butterfly vs Matrix basis", D);

    Report(nameSymm, maxSymmErr, 1e-12, maxSymmErr < 1e-12);
    Report(nameOrtho, maxOrthoErr, 1e-12, maxOrthoErr < 1e-12);
    Report(nameBasis, maxBasisErr, 1e-12, maxBasisErr < 1e-12);
}

// -------------------------------------------------------------------------------------------------
// Test 2: Attention Score Strict Invariance: |(Q*H)*(K*H)^T - Q*K^T| < 1e-12
// -------------------------------------------------------------------------------------------------
template <uint32_t D>
void TestAttentionScoreInvariance(std::mt19937 &rng, uint32_t N = 32)
{
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<double> Q(N * D), K(N * D);
    for (auto &v : Q) v = dist(rng);
    for (auto &v : K) v = dist(rng);

    // Compute true original attention scores: S = Q * K^T
    std::vector<double> S_orig(N * N, 0.0);
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            double dot = 0.0;
            for (uint32_t d = 0; d < D; ++d) {
                dot += Q[i * D + d] * K[j * D + d];
            }
            S_orig[i * N + j] = dot;
        }
    }

    // Apply WHT to Q and K: Q_tilde = Q * H, K_tilde = K * H
    std::vector<double> Q_tilde = Q;
    std::vector<double> K_tilde = K;
    Sage::FastWalshHadamardTransform<D>::TransformRows(Q_tilde.data(), N, D, true);
    Sage::FastWalshHadamardTransform<D>::TransformRows(K_tilde.data(), N, D, true);

    // Compute transformed attention scores: S_wht = Q_tilde * K_tilde^T
    std::vector<double> S_wht(N * N, 0.0);
    double maxScoreErr = 0.0;
    for (uint32_t i = 0; i < N; ++i) {
        for (uint32_t j = 0; j < N; ++j) {
            double dot = 0.0;
            for (uint32_t d = 0; d < D; ++d) {
                dot += Q_tilde[i * D + d] * K_tilde[j * D + d];
            }
            S_wht[i * N + j] = dot;
            maxScoreErr = std::fmax(maxScoreErr, std::fabs(dot - S_orig[i * N + j]));
        }
    }

    char name[64];
    std::snprintf(name, sizeof(name), "D=%u Attention score invariance |Q_wht*K_wht^T - Q*K^T|", D);
    Report(name, maxScoreErr, 1e-12, maxScoreErr < 1e-12);
}

// -------------------------------------------------------------------------------------------------
// Test 3: Extreme Outlier Channel Suppression (> 3.0x peak reduction) & Kurtosis Reduction
// -------------------------------------------------------------------------------------------------
template <uint32_t D>
void TestOutlierSuppression(std::mt19937 &rng)
{
    std::normal_distribution<float> noiseDist(0.0f, 0.5f);
    const uint32_t numTokens = 64;
    std::vector<float> tokens(numTokens * D);

    // Create realistic activation matrix with extreme outlier channels (e.g. channel 7, 23, 51)
    for (uint32_t r = 0; r < numTokens; ++r) {
        for (uint32_t c = 0; c < D; ++c) {
            tokens[r * D + c] = noiseDist(rng);
        }
        tokens[r * D + 7] += 120.0f;  // Dominant LLM outlier channel
        if (D == 128) {
            tokens[r * D + 73] -= 95.0f; // Additional outlier for D=128
        }
    }

    // Measure peak and kurtosis before WHT
    float peakBefore = 0.0f;
    for (auto v : tokens) peakBefore = std::fmax(peakBefore, std::fabs(v));
    double kurtosisBefore = ComputeKurtosis(tokens);

    // Apply WHT
    std::vector<float> tokensTrans = tokens;
    Sage::FastWalshHadamardTransform<D>::TransformRows(tokensTrans.data(), numTokens, D, true);

    // Measure peak and kurtosis after WHT
    float peakAfter = 0.0f;
    for (auto v : tokensTrans) peakAfter = std::fmax(peakAfter, std::fabs(v));
    double kurtosisAfter = ComputeKurtosis(tokensTrans);

    double suppressionRatio = static_cast<double>(peakBefore) / static_cast<double>(peakAfter);

    char namePeak[64], nameKurt[64];
    std::snprintf(namePeak, sizeof(namePeak), "D=%u Outlier peak suppression ratio (>3.0x)", D);
    std::snprintf(nameKurt, sizeof(nameKurt), "D=%u Kurtosis reduction (Gaussian target ~3.0)", D);

    Report(namePeak, suppressionRatio, 3.0, suppressionRatio > 3.0);
    // Gaussian kurtosis is 3.0; outlier kurtosis before is typically > 25.0
    Report(nameKurt, kurtosisAfter, 6.0, kurtosisAfter < 6.0 && kurtosisAfter < kurtosisBefore * 0.3);
}

// -------------------------------------------------------------------------------------------------
// Test 4: Invertibility (Round-Trip Identity): FWHT(FWHT(x)) == x
// -------------------------------------------------------------------------------------------------
template <uint32_t D>
void TestRoundTrip(std::mt19937 &rng)
{
    std::normal_distribution<double> dist(0.0, 5.0);
    std::vector<double> original(D), transformed(D);
    for (auto &v : original) v = dist(rng);
    transformed = original;

    // Apply FWHT twice (normalized)
    Sage::FastWalshHadamardTransform<D>(transformed.data(), true);
    Sage::FastWalshHadamardTransform<D>(transformed.data(), true);

    double maxRoundTripErr = 0.0;
    for (uint32_t i = 0; i < D; ++i) {
        maxRoundTripErr = std::fmax(maxRoundTripErr, std::fabs(transformed[i] - original[i]));
    }

    char name[64];
    std::snprintf(name, sizeof(name), "D=%u Round-trip self-inversion |FWHT(FWHT(x)) - x|", D);
    Report(name, maxRoundTripErr, 1e-12, maxRoundTripErr < 1e-12);
}

} // namespace

int main()
{
    std::mt19937 rng(20260929u);
    std::printf("====================================================================================\n");
    std::printf(" SageAttention2 Fast Walsh-Hadamard Transform (FWHT) Conformance Test\n");
    std::printf("====================================================================================\n");

    std::printf("\n[1] Orthogonality & Matrix Reconstruction (|H_D * H_D^T - I| < 1e-12)\n");
    TestOrthogonality<64>();
    TestOrthogonality<128>();

    std::printf("\n[2] Attention Score Strict Invariance (|(Q*H)*(K*H)^T - Q*K^T| < 1e-12)\n");
    TestAttentionScoreInvariance<64>(rng);
    TestAttentionScoreInvariance<128>(rng);

    std::printf("\n[3] Extreme Outlier Peak Suppression (> 3.0x) & Kurtosis Flattening\n");
    TestOutlierSuppression<64>(rng);
    TestOutlierSuppression<128>(rng);

    std::printf("\n[4] Invertibility / Round-Trip Identity (|FWHT(FWHT(x)) - x| < 1e-12)\n");
    TestRoundTrip<64>(rng);
    TestRoundTrip<128>(rng);

    std::printf("====================================================================================\n");
    if (g_failures == 0) {
        std::printf("ALL FAST WALSH-HADAMARD TRANSFORM CONFORMANCE TESTS PASSED!\n");
        return 0;
    }
    std::printf("%d TEST(S) FAILED!\n", g_failures);
    return 1;
}
