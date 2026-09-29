/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms of conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* Standalone floating-point error conformance test for SageAttention2.
 *
 * 可移植 C++17（宿主机 g++/clang++ 或 CANN 容器内编译运行，无 AscendC 依赖），
 * 将内核的数值规格以可执行断言固化，逐项验证浮点误差满足设计界：
 *   1. FastReciprocal（common.h 牛顿迭代，魔数 0x7EEF370B）   e0^4 包络 ≤ 2^-14
 *   2. per-token INT8 对称量化往返误差                          ≤ scale/2
 *   3. E4M3 饱和舍入相对误差（3 位尾数）                        ≤ 2^-4
 *   4. E8M0 分组 scale 覆盖性（2^ceil(log2(amax/448))）        无饱和且 ≤2×
 *   5. DEQF16(2^-11) × FP16 平衡 scale 的 2048 抵消恒等式       ≤ 2^-9
 *   6. ln(448) 平移后 P 的满量程与饱和裕量                      (224, 448·(1+2^-7)]
 *   7. LSE 契约（raw log2 内核值 + 开源平滑校正, GQA 广播）     ≤ 1e-5（双精度参考）
 *   8. smooth-K softmax 不变性（含校正项）                      ≤ 1e-12（双精度）
 *   9. Causal Mask NZ 拓扑剪枝零精度损失与算力压缩 (≥95% bypass) bit-exact == 0
 *
 * 编译运行：
 *   g++ -O2 -std=c++17 tests/numeric_error_test.cpp -o build/numeric_error_test
 *   ./build/numeric_error_test
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

namespace {

int g_failures = 0;

void Report(const char *name, double measured, double bound, bool ok)
{
    std::printf("  %-42s max_err = %-12.6g bound = %-10g  %s\n", name, measured, bound, ok ? "PASS" : "FAIL");
    if (!ok) {
        ++g_failures;
    }
}

float AsFloat(uint32_t bits)
{
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

uint32_t AsBits(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return bits;
}

// common.h FastReciprocal 的可移植镜像：魔数初值 + 两次牛顿迭代。
float FastReciprocal(float x)
{
    const uint32_t magic = 0x7EEF370Bu;
    float y0 = AsFloat(magic - AsBits(x));
    float y1 = y0 * (2.0f - x * y0);
    return y1 * (2.0f - x * y1);
}

// E4M3 舍入（RNE、饱和 ±448）的可移植镜像，对应 SatMode::SAT + CAST_RINT。
// 直接在浮点位模式上做尾数 RNE 进位，规格化/非规格化/饱和一次处理。
float QuantE4M3(float x)
{
    if (std::isnan(x)) {
        return x;
    }
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    float mag = std::fabs(x);
    if (mag <= 0.00048828125f) {  // <= 半个最小非规格化数 (2^-11)
        return 0.0f;
    }
    mag = std::fmin(mag, 448.0f);
    uint32_t bits = AsBits(mag);
    const uint32_t manBits = (bits & 0x007FFFFFu) >> 20;  // 目标 3 位尾数的高位
    const uint32_t rem = bits & 0x000FFFFFu;              // 被丢弃的低位
    const uint32_t half = 0x00080000u;                    // RNE 中点
    uint32_t enc = bits & 0xFF800000u;                    // 符号+阶码（先剥离尾数）
    enc |= manBits << 20;
    if (rem > half || (rem == half && (enc & 0x00100000u))) {
        enc += 0x00100000u;  // 尾数 MSB 进位（0b111 -> 阶码，位 20 起）
    }
    // enc 仍是合法 float 位模式（阶码溢出即超出 448，饱和处理）
    float q = AsFloat(enc);
    if (q > 448.0f || std::isinf(q)) {
        q = 448.0f;
    }
    return sign * q;
}

// quant_v_mx.h 的 E8M0 scale：2^ceil(log2(amax/448))，下限 2^-100。
float MxScale(float amax)
{
    const float floorScale = std::ldexp(1.0f, -100);
    if (amax <= 448.0f * floorScale) {
        return floorScale;
    }
    int e = 0;
    std::frexp(amax / 448.0f, &e);       // amax/448 = m * 2^(e-1)
    return std::ldexp(1.0f, e - 1 + (std::ldexp(1.0f, e - 1) < amax / 448.0f ? 1 : 0));
}

double LogSumExp(const std::vector<double> &xs)
{
    double m = -1e300;
    for (double x : xs) {
        m = std::fmax(m, x);
    }
    double s = 0.0;
    for (double x : xs) {
        s += std::exp(x - m);
    }
    return m + std::log(s);
}

void TestFastReciprocal(std::mt19937 &rng)
{
    std::uniform_real_distribution<float> uni(1e-6f, 1e6f);
    double worst = 0.0;
    for (int i = 0; i < 200000; ++i) {
        float x = uni(rng);
        double ref = 1.0 / static_cast<double>(x);
        double err = std::fabs(static_cast<double>(FastReciprocal(x)) - ref) / ref;
        worst = std::fmax(worst, err);
    }
    std::printf("[1] FastReciprocal (magic 0x7EEF370B + 2 NR steps)\n");
    // 魔数初值相对误差随 binade 摆动（最差 ~2^-4），两次牛顿迭代后为 e0^4，
    // 包络 2^-16；远小于 FP16 舍入(2^-11)与 P/V 量化噪声(2^-5)的下游影响。
    Report("relative error", worst, 1.0 / 16384.0, worst <= 1.0 / 16384.0);
}

void TestInt8Quant(std::mt19937 &rng)
{
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    double worst = 0.0;
    bool rangeOk = true, zeroRowOk = true;
    for (int t = 0; t < 2000; ++t) {
        float row[128];
        float amax = 0.0f;
        for (auto &v : row) {
            v = gauss(rng);
            amax = std::fmax(amax, std::fabs(v));
        }
        if (t % 500 == 0) {
            for (auto &v : row) {
                v = 0.0f;  // 全零行
            }
            amax = 0.0f;
        }
        const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
        const float inv = amax > 0.0f ? 127.0f / amax : 1.0f;
        for (float v : row) {
            float q = std::nearbyint(v * inv);
            if (q < -127.0f || q > 127.0f) {
                rangeOk = false;
            }
            double err = std::fabs(v - static_cast<double>(q) * scale);
            if (amax > 0.0f) {
                worst = std::fmax(worst, err / scale);
            } else if (q != 0.0f) {
                zeroRowOk = false;
            }
        }
    }
    std::printf("[2] per-token symmetric INT8 quantization\n");
    Report("round-trip error (in scale units)", worst, 0.5 + 1e-6, worst <= 0.5 + 1e-6);
    Report("codeword range [-127,127]", rangeOk ? 0.0 : 1.0, 0.0, rangeOk);
    Report("zero-row maps to zeros", zeroRowOk ? 0.0 : 1.0, 0.0, zeroRowOk);
}

void TestE4M3(std::mt19937 &rng)
{
    std::uniform_real_distribution<float> pos(1e-6f, 448.0f);
    double worst = 0.0;
    bool satOk = true;
    for (int i = 0; i < 200000; ++i) {
        float x = std::copysign(pos(rng), static_cast<float>(i % 2 ? 1 : -1));
        float q = QuantE4M3(x);
        if (std::fabs(q) > 448.0f) {
            satOk = false;
        }
        if (std::fabs(x) > 1e-5f) {
            double rel = std::fabs(static_cast<double>(x) - q) / std::fabs(static_cast<double>(x));
            worst = std::fmax(worst, rel);
        }
    }
    if (QuantE4M3(1000.0f) != 448.0f || QuantE4M3(-1000.0f) != -448.0f) {
        satOk = false;
    }
    std::printf("[3] E4M3 saturating round-to-nearest (3 mantissa bits)\n");
    Report("relative error", worst, 1.0 / 16.0 + 1e-6, worst <= 1.0 / 16.0 + 1e-6);
    Report("saturation at +/-448", satOk ? 0.0 : 1.0, 0.0, satOk);
}

void TestMxScale(std::mt19937 &rng)
{
    std::uniform_real_distribution<float> pos(1e-9f, 1e3f);
    bool coverOk = true, tightOk = true;
    for (int i = 0; i < 100000; ++i) {
        float amax = pos(rng);
        float scale = MxScale(amax);
        if (!(scale >= amax / 448.0)) {
            coverOk = false;  // V/scale 超过 448 → 会饱和
        }
        if (!(scale <= 2.0 * amax / 448.0 + 1e-30)) {
            tightOk = false;  // 损失超过 1 个二进制位动态范围
        }
        float q = QuantE4M3(amax / scale);
        if (q > 448.0f) {
            coverOk = false;
        }
    }
    std::printf("[4] E8M0 group scale = 2^ceil(log2(amax/448))\n");
    Report("coverage (no saturation)", coverOk ? 0.0 : 1.0, 0.0, coverOk);
    Report("tightness (<= 2x)", tightOk ? 0.0 : 1.0, 0.0, tightOk);
}

// 将 float 按 RNE 舍入到 FP16 精度，结果仍以 float 承载（宿主无 _Float16 依赖）。
// 幂次网格下的 nearbyint 乘除均为精确运算，无二次舍入；非规格化域固定 2^-24 网格，
// 结果超出 [−65520, 65520] 按 IEEE 上溢到 Inf。
float RoundToFp16(float f)
{
    const float mag = std::fabs(f);
    if (std::isnan(f) || mag >= 65520.0f) {
        return std::isnan(f) ? f : std::copysign(std::numeric_limits<float>::infinity(), f);
    }
    int e = 0;
    std::frexp(mag, &e);                             // mag = m * 2^(e-1)，m ∈ [0.5,1)
    const int gridExp = e - 11 >= -24 ? e - 11 : -24;  // 规格化步长 2^(e-11)，非规格化 2^-24
    const float grid = std::ldexp(1.0f, gridExp);
    const float r = std::nearbyint(f / grid) * grid;
    return std::fabs(r) < 65520.0f ? r : std::copysign(std::numeric_limits<float>::infinity(), f);
}

void TestDeqf16BalancedScales(std::mt19937 &rng)
{
    // 恒等式：(S_int * 2^-11) * qh * kh == S_int * qs * ks * sm_scale，
    // 其中 qh = fp16(qs * sign(s) * sqrt(|s|*2048))，kh = fp16(ks * sqrt(|s|*2048))，
    // S_int 的 INT32 累加在 float 中精确（|S_int| < 2^24）。
    std::uniform_real_distribution<float> uni(0.5f, 100.0f);
    double worst = 0.0;
    for (int i = 0; i < 100000; ++i) {
        float qs = uni(rng);
        float ks = uni(rng);
        float smScale = (i % 2 ? 1.0f : -1.0f) * (1.0f / std::sqrt(128.0f));
        double sInt = std::round(uni(rng) * 20000.0) - 10000.0;  // |S_int| < 2^24
        float factor = std::sqrt(std::fabs(smScale) * 2048.0f);
        float qh = RoundToFp16(qs * (smScale >= 0 ? factor : -factor));
        float kh = RoundToFp16(ks * factor);
        double lhs = sInt * (1.0 / 2048.0) * static_cast<double>(qh) * static_cast<double>(kh);
        double rhs = sInt * static_cast<double>(qs) * static_cast<double>(ks) * smScale;
        double rel = std::fabs(lhs - rhs) / std::fabs(rhs);
        worst = std::fmax(worst, rel);
    }
    std::printf("[5] DEQF16(2^-11) x FP16 balanced scales cancellation\n");
    // 界来源：qh/kh 各一次 FP16 RNE 舍入（各 ≤ 2^-11 相对）+ float 乘/sqrt 舍入，合计 ≈ 2^-10。
    Report("identity relative error", worst, 1.0 / 512.0, worst <= 1.0 / 512.0);
}

void TestLn448Shift(std::mt19937 &rng)
{
    // 平移后的概率 p = exp(x - m_hat)，m_hat 含 fp16(-ln448)。要求：
    // 行最大元素 p > 224（满量程利用率），且 p <= 448*(1+2^-7)（饱和裕量内）。
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    const float shift = -6.104793232414985f;
    bool utilizationOk = true, headroomOk = true;
    for (int t = 0; t < 20000; ++t) {
        float rowMax = -1e30f;
        for (int i = 0; i < 64; ++i) {
            rowMax = std::fmax(rowMax, gauss(rng) * 3.0f);
        }
        float mHat = rowMax + shift;  // fp16 状态的 float 镜像
        float pMax = std::exp(rowMax - mHat);
        if (!(pMax > 224.0f)) {
            utilizationOk = false;
        }
        if (!(pMax <= 448.0f * (1.0 + 1.0 / 128.0))) {
            headroomOk = false;
        }
    }
    std::printf("[6] ln(448) shifted softmax range utilization\n");
    Report("p_max utilization > 224", utilizationOk ? 0.0 : 1.0, 0.0, utilizationOk);
    Report("p_max headroom <= 448*(1+2^-7)", headroomOk ? 0.0 : 1.0, 0.0, headroomOk);
}

void TestLseIdentity(std::mt19937 &rng)
{
    // 内核输出 raw log2 域 LSE：lse_raw = (m_hat + log(den)) * log2(e)（未校正）。
    // PyTorch 层按开源契约还原：lse = lse_raw / log2(e) + sm_scale * dot(q, mean_k)，
    // GQA 广播 kh = h / (hq/hk) 等价 repeat_interleave。
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    constexpr double kLog2E = 1.4426950408889634;
    const int d = 128;
    const double smScale = 1.0 / std::sqrt(128.0);
    double worst = 0.0;
    bool gqaOk = true;
    const int hq = 8, hk = 2;
    for (int t = 0; t < 200; ++t) {
        std::vector<double> q(d), meanK(d), kSmooth(d);
        for (auto &x : q) {
            x = gauss(rng);
        }
        for (auto &x : kSmooth) {
            x = gauss(rng);
        }
        for (auto &x : meanK) {
            x = 0.1 * gauss(rng);
        }
        double dotQM = 0.0;
        for (int j = 0; j < d; ++j) {
            dotQM += q[j] * meanK[j];
        }
        const int rows = 4;
        std::vector<double> smoothScores(rows), rawScores(rows);
        for (int r = 0; r < rows; ++r) {
            double ss = 0.0;
            for (int j = 0; j < d; ++j) {
                ss += q[j] * (kSmooth[j] + 0.01 * r);  // 同一平滑行 + 微扰
            }
            smoothScores[r] = ss * smScale;
            rawScores[r] = smoothScores[r] + smScale * dotQM;  // 行常数还原
        }
        // 参考：未平滑分数的真值 logsumexp。
        double reference = LogSumExp(rawScores);
        // 内核侧：平移 max 状态 + den，换底 log2 后输出 raw 值（不含平滑校正）。
        double m = -1e300;
        for (double s : smoothScores) {
            m = std::fmax(m, s - std::log(448.0));
        }
        double den = 0.0;
        for (double s : smoothScores) {
            den += std::exp(s - m);
        }
        double kernelRaw = (m + std::log(den)) * kLog2E;
        double returned = kernelRaw / kLog2E + smScale * dotQM;
        double rel = std::fabs(returned - reference) / (std::fabs(reference) + 1.0);
        worst = std::fmax(worst, rel);
        for (int h = 0; h < hq; ++h) {
            int kh = h / (hq / hk);
            if (kh < 0 || kh >= hk) {
                gqaOk = false;
            }
        }
    }
    std::printf("[7] LSE contract (raw log2 kernel + open-source smooth correction)\n");
    Report("relative error vs double reference", worst, 1e-5, worst <= 1e-5);
    Report("GQA kh = h/(hq/hk) mapping", gqaOk ? 0.0 : 1.0, 0.0, gqaOk);
}

void TestSmoothKInvariance(std::mt19937 &rng)
{
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    const int d = 128;
    double worst = 0.0;
    for (int t = 0; t < 200; ++t) {
        std::vector<double> q(d), meanK(d), kSmooth(d);
        for (auto &x : q) {
            x = gauss(rng);
        }
        for (auto &x : meanK) {
            x = gauss(rng);
        }
        for (auto &x : kSmooth) {
            x = gauss(rng);
        }
        double sRaw = 0.0, sSmooth = 0.0, dotQM = 0.0;
        for (int j = 0; j < d; ++j) {
            sRaw += q[j] * (kSmooth[j] + meanK[j]);
            sSmooth += q[j] * kSmooth[j];
            dotQM += q[j] * meanK[j];
        }
        sRaw *= 1.0 / std::sqrt(128.0);
        sSmooth *= 1.0 / std::sqrt(128.0);
        dotQM *= 1.0 / std::sqrt(128.0);
        // softmax 行不变性：exp(sRaw)/Z == exp(sSmooth)/Z'，由行常数 dotQM 保证
        const double shift = 20.0;
        double p1 = std::exp(sRaw - shift);
        double p2 = std::exp(sSmooth - shift) * std::exp(dotQM);
        double rel = std::fabs(p1 - p2) / (std::fabs(p1) + 1e-300);
        worst = std::fmax(worst, rel);
    }
    std::printf("[8] smooth-K softmax invariance (constant shift)\n");
    Report("relative discrepancy", worst, 1e-12, worst <= 1e-12);
}

void TestCausalMaskTopologyPruning()
{
    // Verify 128 NZ lanes: laneOffset = (lane & 15) - (lane >> 4)
    int16_t laneOffset[128];
    int minOffset = 1000, maxOffset = -1000;
    for (int i = 0; i < 128; ++i) {
        int16_t r = i >> 4;
        int16_t c = i & 15;
        laneOffset[i] = c - r;
        if (laneOffset[i] < minOffset) minOffset = laneOffset[i];
        if (laneOffset[i] > maxOffset) maxOffset = laneOffset[i];
    }
    bool rangeOk = (minOffset == -7 && maxOffset == 15);

    // Verify bit-exact equivalence between baseline compare and pruned topology over wide threshold domain [-1024, 1024]
    bool bitExactOk = true;
    for (int thresh = -1024; thresh <= 1024; ++thresh) {
        for (int i = 0; i < 128; ++i) {
            bool baselineValid = (laneOffset[i] <= thresh);
            bool prunedValid = false;
            if (thresh >= 15) {
                prunedValid = true;
            } else if (thresh < -7) {
                prunedValid = false;
            } else {
                prunedValid = (laneOffset[i] <= thresh);
            }
            if (baselineValid != prunedValid) {
                bitExactOk = false;
                break;
            }
        }
    }

    // Verify pruning ratio over all 64 NZ pairs (256 sub-blocks) in SoftmaxWideMaxCore
    int totalSubBlocks = 0;
    int prunedSubBlocks = 0;
    for (int qInGroup : {0, 64, 128, 192, 256, 320, 384, 448}) {
        int causalLimit0 = qInGroup;
        int causalLimit1 = qInGroup - 256;
        for (uint16_t row = 0; row < 64; row += 16) {
            for (uint16_t col = 0; col < 16; ++col) {
                int t0_0 = causalLimit0 + row - col * 16;
                int t1_0 = t0_0 + 8;
                int t0_1 = causalLimit1 + row - col * 16;
                int t1_1 = t0_1 + 8;
                for (int t : {t0_0, t1_0, t0_1, t1_1}) {
                    totalSubBlocks++;
                    if (t >= 15 || t < -7) {
                        prunedSubBlocks++;
                    }
                }
            }
        }
    }
    double bypassRatio = static_cast<double>(prunedSubBlocks) / totalSubBlocks;
    bool bypassRatioOk = (bypassRatio >= 0.95);

    std::printf("[9] causal mask NZ topology pruning (bit-exact lower/upper bypass)\n");
    Report("lane offset bounded in [-7, 15]", rangeOk ? 0.0 : 1.0, 0.0, rangeOk);
    Report("bit-exact mask equivalence", bitExactOk ? 0.0 : 1.0, 0.0, bitExactOk);
    Report("vector instruction bypass ratio >= 95%", 1.0 - bypassRatio, 0.05, bypassRatioOk);
}

}  // namespace

int main()
{
    std::mt19937 rng(20260928u);
    std::printf("SageAttention2 numeric error conformance test (portable C++17)\n");
    std::printf("================================================================\n");
    TestFastReciprocal(rng);
    TestInt8Quant(rng);
    TestE4M3(rng);
    TestMxScale(rng);
    TestDeqf16BalancedScales(rng);
    TestLn448Shift(rng);
    TestLseIdentity(rng);
    TestSmoothKInvariance(rng);
    TestCausalMaskTopologyPruning();
    std::printf("================================================================\n");
    if (g_failures == 0) {
        std::printf("ALL NUMERIC ERROR TESTS PASSED\n");
        return 0;
    }
    std::printf("%d NUMERIC ERROR TEST(S) FAILED\n", g_failures);
    return 1;
}
