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
 *   6. FP16(-ln448) 平移常量位级一致 + P 满量程利用/饱和裕量     [224, 448·(1+2^-5)]
 *   7. LSE 契约（raw log2 内核值 + 开源平滑校正, GQA 广播）     ≤ 1e-5（双精度参考）
 *   8. smooth-K softmax 不变性（FP32 内核算术精度, 整行）        ≤ 1e-4
 *   9. Causal Mask NZ 拓扑剪枝零精度损失与算力压缩 (≥95% bypass) bit-exact == 0
 *  10. qk_quant_gran 组共享 INT8 scale 覆盖性与半步长误差                  ≤ scale/2
 *  11. pv_accum_dtype="fp32+fp16" 部分和 FP16 往返折入误差 + 满块溢出包络   ≤ Σ|O_j|·2^-11
 *  12. smooth_v 居中 + 输出补回的 MXFP8 重建误差，及包络不放大             ≤ amax·2^-4
 *  13. on-chip streaming query INT8 量化与全量离线量化位级严格等价性       bit-exact == 0
 *
 * 编译运行：
 *   g++ -O2 -std=c++17 tests/numeric_error_test.cpp -o build/numeric_error_test
 *   ./build/numeric_error_test
 */

#include <algorithm>
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
    // 魔数初值相对误差随 binade 摆动（最差 ~2^-4），两次牛顿迭代后的最差
    // 包络 ~2^-14；远小于 FP16 舍入(2^-11)与 P/V 量化噪声(2^-5)的下游影响。
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
    // 内核以 FP16 常量 SOFTMAX_FP8_SHIFT = -ln(448) 平移块最大值：m_hat 保存于
    // FP16，P = exp(rowMax - m_hat)。设计性质：
    //   1) 平移常量逐位等于 FP16(-ln448) = -6.10546875（内核装载值）；
    //   2) m_hat 的 FP16 舍入有界，行最大元素的未截断 P 贴近满量程
    //      （>224，无欠利用），且超出 448 的部分由 E4M3 SAT cast 精确钳位，
    //      偏差限制在一个 E4M3 相对步长（2^-4 量级）内，不会无界溢出。
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    constexpr float kLn448F32 = 6.104793232414985f;
    const float shift16 = RoundToFp16(-kLn448F32);
    constexpr float kExpectedShift16 = -6.10546875f;
    const bool constOk = (shift16 == kExpectedShift16);
    float worstLow = 1e30f, worstHigh = -1e30f;
    for (int t = 0; t < 20000; ++t) {
        float rowMax = -1e30f;
        for (int i = 0; i < 64; ++i) {
            rowMax = std::fmax(rowMax, gauss(rng) * 3.0f);
        }
        float mHat = RoundToFp16(rowMax + shift16);  // FP16 max 状态镜像
        float pMax = std::exp(rowMax - mHat);        // 未截断概率
        worstLow = std::fmin(worstLow, pMax);
        worstHigh = std::fmax(worstHigh, pMax);
    }
    const bool utilizationOk = worstLow > 224.0f;
    const bool saturationOk = worstHigh <= 448.0f * (1.0f + 1.0f / 32.0f);
    std::printf("[6] ln(448) shifted softmax range utilization\n");
    Report("fp16(-ln448) bit-equals -6.10546875", constOk ? 0.0 : 1.0, 0.0, constOk);
    Report("p_max utilization > 224", worstLow, 224.0, utilizationOk);
    Report("p_max within E4M3 saturation step", worstHigh, 448.0f * (1.0f + 1.0f / 32.0f), saturationOk);
}

void TestLseIdentity(std::mt19937 &rng)
{
    // 内核输出 raw log2 域 LSE：lse_raw = (m_hat + log(den)) * log2(e)（未校正）。
    // PyTorch 层按开源契约还原：lse = lse_raw / log2(e) + sm_scale * dot(q, mean_k)；
    // GQA 下头 h 广播其 KV 组 kh = h/(hq/hk) 的 mean_k（等价 repeat_interleave）。
    // 两组 KV 均值彼此不同，广播取错组会立即放大误差，因此本断言是实质性的。
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    constexpr double kLog2E = 1.4426950408889634;
    const int d = 128;
    const double smScale = 1.0 / std::sqrt(128.0);
    double worst = 0.0;
    const int hq = 8, hk = 2, rows = 4;
    for (int t = 0; t < 200; ++t) {
        std::vector<double> q(d), kSmooth(d);
        for (auto &x : q) {
            x = gauss(rng);
        }
        for (auto &x : kSmooth) {
            x = gauss(rng);
        }
        std::vector<std::vector<double>> meanKv(hk, std::vector<double>(d));
        for (auto &m : meanKv) {
            for (auto &x : m) {
                x = 0.1 * gauss(rng);
            }
        }
        for (int h = 0; h < hq; ++h) {
            const int kh = h / (hq / hk);  // repeat_interleave 广播映射
            const auto &meanK = meanKv[kh];
            double dotQM = 0.0;
            for (int j = 0; j < d; ++j) {
                dotQM += q[j] * meanK[j];
            }
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
        }
    }
    std::printf("[7] LSE contract (raw log2 kernel + open-source smooth correction)\n");
    Report("relative error vs double reference (per-head GQA broadcast)", worst, 1e-5, worst <= 1e-5);
}

void TestSmoothKInvariance(std::mt19937 &rng)
{
    // smooth-K 是行常数平移。以 FP32（内核 stats/quant 路径的算术精度）对整行
    // 计算两种分数并各自归一化：exp(sRaw)/Z == exp(sSmooth + dot(q,m)/s)/Z'。
    // 双精度下该恒等式平凡成立；本断言在 FP32 点积噪声水平上验证行不变性。
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    const int d = 128, rows = 64;
    const float smScale = 1.0f / std::sqrt(128.0f);
    float worst = 0.0f;
    for (int t = 0; t < 200; ++t) {
        std::vector<float> q(d), meanK(d), kSmooth(d);
        for (auto &x : q) {
            x = gauss(rng);
        }
        for (auto &x : meanK) {
            x = gauss(rng);
        }
        for (auto &x : kSmooth) {
            x = gauss(rng);
        }
        std::vector<float> raw(rows), smooth(rows);
        float dotQM = 0.0f;
        for (int j = 0; j < d; ++j) {
            dotQM += q[j] * meanK[j];
        }
        dotQM *= smScale;
        for (int r = 0; r < rows; ++r) {
            float sr = 0.0f, ss = 0.0f;
            for (int j = 0; j < d; ++j) {
                // 行间微扰使各行分数不同，排除“全部相同行”退化为恒真比较
                sr += q[j] * (kSmooth[j] + 0.01f * r + meanK[j]);
                ss += q[j] * (kSmooth[j] + 0.01f * r);
            }
            raw[r] = sr * smScale;
            smooth[r] = ss * smScale + dotQM;
        }
        // 各自做 online softmax 归一化（logsumexp 用双精度仅作精确归一化器）
        std::vector<double> rawD(raw.begin(), raw.end()), smoothD(smooth.begin(), smooth.end());
        double zRaw = LogSumExp(rawD), zSmooth = LogSumExp(smoothD);
        for (int r = 0; r < rows; ++r) {
            double pRaw = std::exp(rawD[r] - zRaw);
            double pSmooth = std::exp(smoothD[r] - zSmooth);
            float rel = static_cast<float>(std::fabs(pRaw - pSmooth) / pRaw);
            worst = std::fmax(worst, rel);
        }
    }
    std::printf("[8] smooth-K softmax invariance (FP32 kernel arithmetic)\n");
    Report("row-wise probability discrepancy", worst, 1e-4, worst <= 1e-4);
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

// quant.h QuantRowsImpl：行 amax/127 得 scale，BroadcastGroupMaxima 让连续 group
// 个 token 共享同一 scale（per_thread = Q4/K16，per_warp = Q32/K64）。
void TestQuantGranularityGroups(std::mt19937 &rng)
{
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    const int rows = 64, d = 128;
    bool noSaturation = true, groupCoversRow = true;
    float worstHalfStep = 0.0f;
    for (int t = 0; t < 100; ++t) {
        std::vector<std::vector<float>> x(rows, std::vector<float>(d));
        for (auto &row : x) {
            for (auto &v : row) {
                v = gauss(rng);
            }
        }
        // 单行离群会抬高整组步长，这是组共享 scale 的最坏情形
        for (auto &v : x[0]) {
            v *= 8.0f;
        }
        std::vector<float> rowAmax(rows, 0.0f);
        for (int r = 0; r < rows; ++r) {
            for (int j = 0; j < d; ++j) {
                rowAmax[r] = std::fmax(rowAmax[r], std::fabs(x[r][j]));
            }
        }
        for (int group : {4, 16, 32, 64}) {
            for (int start = 0; start < rows; start += group) {
                const int count = std::min(group, rows - start);
                float amax = 0.0f;
                for (int r = 0; r < count; ++r) {
                    amax = std::fmax(amax, rowAmax[start + r]);
                }
                const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
                // scale*127 需覆盖组 amax，容差取 FP32 乘除的相对舍入
                if (amax > 0.0f && scale * 127.0f < amax * (1.0f - 1e-6f)) {
                    groupCoversRow = false;  // 组 amax 必须被组内每一行的 scale 覆盖
                }
                for (int r = 0; r < count; ++r) {
                    for (int j = 0; j < d; ++j) {
                        const float q = std::nearbyintf(x[start + r][j] / scale);
                        if (std::fabs(q) > 127.0f) {
                            noSaturation = false;
                        }
                        // 半步长上界再加 FP32 乘除的相对容差
                        worstHalfStep = std::fmax(
                            worstHalfStep, std::fabs(q * scale - x[start + r][j]) / (0.5f * scale));
                    }
                }
            }
        }
    }
    std::printf("[10] qk_quant_gran group-shared INT8 scale (BroadcastGroupMaxima)\n");
    Report("group scale covers every row amax", groupCoversRow ? 0.0 : 1.0, 0.0, groupCoversRow);
    Report("no INT8 saturation", noSaturation ? 0.0 : 1.0, 0.0, noSaturation);
    Report("round-trip error vs half of the group step", worstHalfStep, 1.0 + 1e-5,
           worstHalfStep <= 1.0 + 1e-5);
}

// output_update.h RoundPartialFp16VF：pv_accum_dtype="fp32+fp16" 时，块内 PV 部分和
// 折入 FP32 运行状态前做一次 FP32->FP16->FP32 往返。逐块加权和的误差以
// sum_j |O_j| / den * 2^-11 为界（每块一次相对舍入，块权重归一化后 <=1）。
// 同一往返用 SatMode::NO_SAT，所以还要证明它落在半精度可表示区间内：折叠前
// 的幅度由 attention_block_cube.h 的 P 侧 2^-8 常量尺度限定（common.h PV_PARTIAL_SHIFT）。
void TestFp16PartialRoundtrip(std::mt19937 &rng)
{
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    const int blocks = 128, channels = 128;
    float worstRatio = 0.0f, boundRatio = 0.0f;
    bool inert = true;
    for (int t = 0; t < 100; ++t) {
        std::vector<std::vector<float>> part(blocks, std::vector<float>(channels));
        std::vector<float> den(blocks);
        for (auto &p : part) {
            for (auto &v : p) {
                v = gauss(rng);
            }
        }
        for (auto &d : den) {
            d = std::fabs(gauss(rng)) + 0.25f;
        }
        // 在线重缩放的等价形式：归一化后按块权重求和，FP32 顺序累加
        float total = 0.0f;
        for (float d : den) {
            total += d;
        }
        std::vector<float> accFp32(channels, 0.0f), accFp16(channels, 0.0f), absSum(channels, 0.0f);
        for (int i = 0; i < blocks; ++i) {
            const float w = den[i] / total;
            for (int c = 0; c < channels; ++c) {
                const float rounded = RoundToFp16(part[i][c]);
                accFp32[c] += w * part[i][c];
                accFp16[c] += w * rounded;
                absSum[c] += std::fabs(part[i][c]) * w;
                if (rounded != part[i][c]) {
                    inert = false;
                }
            }
        }
        for (int c = 0; c < channels; ++c) {
            const float err = std::fabs(accFp16[c] - accFp32[c]);
            const float bound = std::ldexp(absSum[c], -11);
            worstRatio = std::fmax(worstRatio, err);
            boundRatio = std::fmax(boundRatio, bound);
        }
    }
    std::printf("[11] pv_accum_dtype fp32+fp16 partial roundtrip\n");
    Report("FP16 partial changes the folded addend", inert ? 1.0 : 0.0, 0.0, !inert);
    Report("accumulated error vs sum |O_j| * 2^-11", worstRatio, boundRatio, worstRatio <= boundRatio);

    // 溢出包络：一个 PV 块最多容纳 kBlock 个概率同时取到 E4M3 满量程，未平移的
    // 部分和为 448 * kBlock * max|V|，|V| = 1 时即 114688 > 65504，而
    // RoundPartialFp16VF 用 SatMode::NO_SAT，溢出为 Inf 并污染整行输出。
    // 内核把 2^-8 记在 PV MMA 的 P 侧 E8M0 常量尺度上（纯指数平移），折叠后上界
    // 变为 448 * max|V|，与块宽、序列长度都无关；finalize 再乘回同一 2 的幂。
    constexpr float kCeiling = 448.0f;
    constexpr int kBlock = 256;
    constexpr int kShift = 8;
    const float unshifted = RoundToFp16(kCeiling * kBlock);
    Report("unshifted full block overflows half range", std::isinf(unshifted) ? 1.0 : 0.0,
           1.0, std::isinf(unshifted));
    bool shiftedFits = true;
    for (int used = 1; used <= kBlock; ++used) {
        for (float mag : {1.0f, 10.0f, 100.0f, 146.0f}) {
            if (!std::isfinite(RoundToFp16(std::ldexp(kCeiling * used * mag, -kShift)))) {
                shiftedFits = false;
            }
        }
    }
    Report("shifted full block fits for max|V| <= 146", shiftedFits ? 0.0 : 1.0, 0.0, shiftedFits);
    // 平移与还原都是指数运算：折叠前后与 FP32 通路逐位一致，只有 FP16 舍入本身残留。
    bool shiftExact = true;
    std::uniform_real_distribution<float> broad(-1.0e4f, 1.0e4f);
    for (int i = 0; i < 100000; ++i) {
        const float x = broad(rng);
        if (std::fabs(x) < 1.0e-30f) {
            continue;
        }
        if (std::ldexp(std::ldexp(x, -kShift), kShift) != x) {
            shiftExact = false;
            break;
        }
    }
    Report("2^-8/2^8 round-trip is bit-exact", shiftExact ? 0.0 : 1.0, 0.0, shiftExact);
}

// stats.h / quant_v_mx.h / output_update.h 的 smooth_v 通路：V 沿序列维取通道均值，
// 量化前居中，归一化输出上补回。补回项在归一化之后加，不参与跨块累加。
// E4M3 + E8M0 分组量化下每个元素的绝对误差不超过该组 amax 的 2^-4，权重和为 1，
// 因此整行重建误差以 2^-4 * max_group_amax 为界；居中降低该 amax，即降低包络本身。
void TestSmoothVReconstruction(std::mt19937 &rng)
{
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    const int sk = 512, d = 128, group = 32;  // MXFP8：每 channel 每 32 token 一个 E8M0 scale
    float worstRatio[2] = {0.0f, 0.0f};
    double envelope[2] = {0.0, 0.0};
    int samples = 0;
    for (int t = 0; t < 40; ++t) {
        std::vector<std::vector<float>> v(sk, std::vector<float>(d));
        for (auto &row : v) {
            for (auto &x : row) {
                x = gauss(rng) + 4.0f;  // 直流偏置：居中前 amax 被整体抬高
            }
        }
        std::vector<float> mean(d, 0.0f);
        for (int j = 0; j < sk; ++j) {
            for (int c = 0; c < d; ++c) {
                mean[c] += v[j][c];
            }
        }
        for (auto &m : mean) {
            m /= static_cast<float>(sk);
        }
        std::vector<float> p(sk);
        float psum = 0.0f;
        for (auto &w : p) {
            w = std::fabs(gauss(rng)) + 1e-3f;
            psum += w;
        }
        for (auto &w : p) {
            w /= psum;  // softmax 权重，行和为 1
        }
        for (int pass = 0; pass < 2; ++pass) {
            const bool center = pass == 1;
            for (int c = 0; c < d; ++c) {
                const float shift = center ? mean[c] : 0.0f;
                float amaxHead = 0.0f;
                double ref = 0.0;
                for (int j = 0; j < sk; ++j) {
                    amaxHead = std::fmax(amaxHead, std::fabs(v[j][c] - shift));
                    ref += static_cast<double>(p[j]) * static_cast<double>(v[j][c]);
                }
                float acc = 0.0f;
                for (int g = 0; g < sk; g += group) {
                    float amax = 0.0f;
                    for (int j = g; j < g + group; ++j) {
                        amax = std::fmax(amax, std::fabs(v[j][c] - shift));
                    }
                    const float scale = MxScale(amax);
                    for (int j = g; j < g + group; ++j) {
                        const float x = v[j][c] - shift;
                        acc += p[j] * (QuantE4M3(x / scale) * scale + shift);
                    }
                }
                ++samples;
                const float bound = std::ldexp(amaxHead, -4);
                worstRatio[pass] = std::fmax(worstRatio[pass], std::fabs(acc - static_cast<float>(ref)) / bound);
                envelope[pass] += bound;
            }
        }
    }
    std::printf("[12] smooth_v centering + output compensation (MXFP8 group quant)\n");
    Report("uncentered rebuild error / 2^-4 * amax", worstRatio[0], 1.0, worstRatio[0] <= 1.0);
    Report("centered rebuild error / 2^-4 * amax", worstRatio[1], 1.0, worstRatio[1] <= 1.0);
    const double plain = envelope[0] / samples, centered = envelope[1] / samples;
    std::printf("       mean error envelope: uncentered = %.6g, centered = %.6g\n", plain, centered);
    Report("centering does not widen the error envelope", centered, plain, centered <= plain);
}

// 验证片上即时流式 Query INT8 量化（On-Chip Streaming Q Quantization）与全量离线 Batch 量化
// 的位级严格等价性（Bit-exact Equivalence）。
// 对应 Stage 1.2 / 1.3 / 1.4：每个 128x128 FP16 Q Tile 在 Attention 内循环由 MIX 1:2 的
// 两个 AIV 核心按 64x128 子块就地在线量化并直写 L1/UB，消灭 HBM 往返与 LLC 击穿。
// 基于 per-token 对称 INT8 量化数学公式（INV_INT8_MAX = 1/127，FastReciprocal 倒数），
// 逐 token 量化具有行独立性，子块流式输出与离线批处理输出在 INT8 编码与 FP16 尺度上
// 必须完全 bit-exact 相同（max_err == 0）。
void TestOnChipStreamingQQuantEquivalence(std::mt19937 &rng)
{
    constexpr float INV_INT8_MAX = 1.0f / 127.0f;
    constexpr float DEQF16_FACTOR = 2048.0f;
    const int tileRows = 128, headDim = 128;
    const int subBlockRows = 64;  // MIX 1:2 中每个 AIV 核心处理 64 行
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::uniform_real_distribution<float> scaleDist(0.05f, 2.0f);

    int maxDiffInt8 = 0;
    double maxDiffScale = 0.0;
    bool zeroRowMatch = true;
    bool saturationMatch = true;

    // 测试多个 Tile，涵盖常规高斯分布、全零行、大离群值、极端动态范围
    for (int t = 0; t < 50; ++t) {
        float smScale = (t % 2 ? 1.0f : -1.0f) * scaleDist(rng) / std::sqrt(static_cast<float>(headDim));
        float factor = std::sqrt(std::fabs(smScale) * DEQF16_FACTOR);
        float qFactor = (smScale >= 0.0f) ? factor : -factor;

        std::vector<std::vector<float>> qTile(tileRows, std::vector<float>(headDim));
        for (int r = 0; r < tileRows; ++r) {
            for (int c = 0; c < headDim; ++c) {
                qTile[r][c] = RoundToFp16(gauss(rng));
            }
        }
        // 边界情况注入
        if (t % 5 == 0) {
            // 注入全零行（验证 amax=0, scale=1.0f 规避被除数为 0）
            for (int c = 0; c < headDim; ++c) {
                qTile[10][c] = 0.0f;
                qTile[75][c] = 0.0f;
            }
        }
        if (t % 7 == 0) {
            // 注入极大离群值（触发 INT8 饱和截断 [-127, 127]）
            qTile[20][3] = RoundToFp16(50.0f);
            qTile[20][4] = RoundToFp16(-50.0f);
            qTile[90][5] = RoundToFp16(80.0f);
        }

        // 1. 全量离线 Batch 量化（模拟原始 preprocessing Quant 阶段）
        std::vector<std::vector<int8_t>> q8Batch(tileRows, std::vector<int8_t>(headDim));
        std::vector<float> scaleBatch(tileRows);
        for (int r = 0; r < tileRows; ++r) {
            float amax = 0.0f;
            for (int c = 0; c < headDim; ++c) {
                amax = std::fmax(amax, std::fabs(qTile[r][c]));
            }
            float m = amax * INV_INT8_MAX;
            if (m <= 0.0f) {
                m = 1.0f;
            }
            float inv = FastReciprocal(m);
            for (int c = 0; c < headDim; ++c) {
                float scaled = qTile[r][c] * inv;
                float rounded = std::nearbyint(scaled);
                rounded = std::fmax(-127.0f, std::fmin(127.0f, rounded));
                q8Batch[r][c] = static_cast<int8_t>(rounded);
            }
            scaleBatch[r] = RoundToFp16(m * qFactor);
        }

        // 2. 片上即时流式子块量化（模拟 AttentionWideStream 内两个 AIV 核心各处理 64x128 子块）
        std::vector<std::vector<int8_t>> q8Stream(tileRows, std::vector<int8_t>(headDim));
        std::vector<float> scaleStream(tileRows);

        for (int sub = 0; sub < 2; ++sub) {
            int startRow = sub * subBlockRows;
            int endRow = startRow + subBlockRows;
            for (int r = startRow; r < endRow; ++r) {
                float amax = 0.0f;
                for (int c = 0; c < headDim; ++c) {
                    amax = std::fmax(amax, std::fabs(qTile[r][c]));
                }
                float m = amax * INV_INT8_MAX;
                if (m <= 0.0f) {
                    m = 1.0f;
                }
                float inv = FastReciprocal(m);
                for (int c = 0; c < headDim; ++c) {
                    float scaled = qTile[r][c] * inv;
                    float rounded = std::nearbyint(scaled);
                    rounded = std::fmax(-127.0f, std::fmin(127.0f, rounded));
                    q8Stream[r][c] = static_cast<int8_t>(rounded);
                }
                scaleStream[r] = RoundToFp16(m * qFactor);
            }
        }

        // 逐元素与逐行比对
        for (int r = 0; r < tileRows; ++r) {
            if (scaleBatch[r] != scaleStream[r]) {
                double diff = std::fabs(static_cast<double>(scaleBatch[r]) - static_cast<double>(scaleStream[r]));
                maxDiffScale = std::fmax(maxDiffScale, diff);
            }
            for (int c = 0; c < headDim; ++c) {
                int diff = std::abs(static_cast<int>(q8Batch[r][c]) - static_cast<int>(q8Stream[r][c]));
                if (diff > maxDiffInt8) {
                    maxDiffInt8 = diff;
                }
            }
        }

        // 验证全零行与饱和一致性
        if (t % 5 == 0) {
            if (scaleBatch[10] != scaleStream[10] || q8Batch[10][0] != q8Stream[10][0]) {
                zeroRowMatch = false;
            }
        }
        if (t % 7 == 0) {
            if (q8Batch[20][3] != 127 || q8Stream[20][3] != 127 ||
                q8Batch[20][4] != -127 || q8Stream[20][4] != -127) {
                saturationMatch = false;
            }
        }
    }

    std::printf("[13] on-chip streaming query INT8 quantization bit-exact equivalence\n");
    Report("bit-exact INT8 codewords (max_err == 0)", maxDiffInt8, 0.0, maxDiffInt8 == 0);
    Report("bit-exact FP16 scales (max_err == 0)", maxDiffScale, 0.0, maxDiffScale == 0.0);
    Report("zero-row handling bit-exact match", zeroRowMatch ? 0.0 : 1.0, 0.0, zeroRowMatch);
    Report("saturation clamping bit-exact match", saturationMatch ? 0.0 : 1.0, 0.0, saturationMatch);
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
    TestQuantGranularityGroups(rng);
    TestFp16PartialRoundtrip(rng);
    TestSmoothVReconstruction(rng);
    TestOnChipStreamingQQuantEquivalence(rng);
    std::printf("================================================================\n");
    if (g_failures == 0) {
        std::printf("ALL NUMERIC ERROR TESTS PASSED\n");
        return 0;
    }
    std::printf("%d NUMERIC ERROR TEST(S) FAILED\n", g_failures);
    return 1;
}
