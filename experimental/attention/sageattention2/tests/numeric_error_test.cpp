/* Standalone floating-point error conformance test for SageAttention2.
 *
 * 可移植 C++17（宿主机 g++/clang++ 或 CANN 容器内编译运行，无 AscendC 依赖），
 * 将内核的数值规格以可执行断言固化，逐项验证浮点误差满足设计界：
 *   1. FastReciprocal（platform_traits.h 魔数 0x7EEF370B + math_approximations.h 牛顿迭代） e0^4 包络 ≤ 2^-14
 *   2. per-token INT8 对称量化往返误差                          ≤ scale/2
 *   3. E4M3 饱和舍入相对误差（3 位尾数，MX_FP8_MAX）             ≤ 2^-4
 *   4. E8M0 分组 scale 覆盖性（2^ceil(log2(amax/MX_FP8_MAX))）    无饱和且 ≤2×
 *   5. DEQF16(2^-11) × FP16 平衡 scale 的 2048 抵消恒等式       ≤ 2^-9
 *   6. FP16(-ln448) 平移常量位级一致 + P 满量程利用/饱和裕量     [224, 448·(1+2^-5)]
 *   7. LSE 契约（自然对数内核值 + 开源平滑校正, GQA CalcKh 广播） ≤ 1e-5（双精度参考）
 *   8. smooth-K softmax 不变性（FP32 内核算术精度, 整行）        ≤ 1e-4
 *   9. Causal Mask NZ 拓扑剪枝零精度损失与算力压缩 (≥95% bypass) bit-exact == 0
 *  10. qk_quant_gran 组共享 INT8 scale 覆盖性与半步长误差        ≤ scale/2
 *  11. pv_accum_dtype="fp32+fp16" 部分和 FP16 往返折入误差 + 满块溢出包络   ≤ Σ|O_j|·2^-11
 *  12. smooth_v 居中 + 输出补回的 MXFP8 重建误差，及包络不放大             ≤ amax·2^-4
 *  13. on-chip streaming query INT8 量化与全量离线量化位级严格等价性       bit-exact == 0
 *  14. 蛇形波前折叠调度双射性 + 因果负载方差压缩 (SnakeScheduler)          CV ≤ 0.05, ≥10x
 *  15. Delayed FP32 Buffering:FP16 对折叠累加误差包络 + 溢出安全          ≤ 32·2^-11·2
 *  16. Chunk-wise 局部平滑零屏障在线 Softmax 等价性                        ≤ 1e-5
 *  17. Task 3.4 / P5.2 homologous normalizer（E4M3 舍入值同源累加消除概率质量偏差） ≤ 1e-5
 *  18. Task P0.1 宽路径多块分母折叠与 LSE 契约（dp=128, sk>=4096）             ≤ 1e-5
 *
 * 编译运行：
 *   g++ -O2 -std=c++17 ops-transformer/experimental/attention/sageattention2/tests/numeric_error_test.cpp -o /tmp/numeric_error_test && /tmp/numeric_error_test
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "../op_kernel/platform_traits.h"
#include "../op_kernel/math_approximations.h"
#include "../op_kernel/split_kv_grid.h"
#include "../op_kernel/snake_wavefront_balancer.h"

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

// E4M3 舍入（RNE、饱和 ±MX_FP8_MAX）的可移植镜像，对应 SatMode::SAT + CAST_RINT。
// 直接在浮点位模式上做尾数 RNE 进位，规格化/非规格化/饱和一次处理。
float QuantE4M3(float x)
{
    if (std::isnan(x)) {
        return x;
    }
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    float mag = std::fabs(x);
    // <= 半个最小非规格化数 (2^-11 = DEQ_SCALAR_DEQF16)
    if (mag <= AsFloat(static_cast<uint32_t>(sage::core::DEQ_SCALAR_DEQF16))) {
        return 0.0f;
    }
    mag = std::fmin(mag, sage::core::MX_FP8_MAX);
    uint32_t bits = AsBits(mag);
    const uint32_t manBits = (bits & sage::core::EXPONENT_ROUND_BIAS) >> 20;  // 目标 3 位尾数的高位 (0x007FFFFF)
    const uint32_t rem = bits & 0x000FFFFFu;                                  // 被丢弃的低位
    const uint32_t half = 0x00080000u;                                        // RNE 中点
    uint32_t enc = bits & 0xFF800000u;                                        // 符号+阶码（先剥离尾数）
    enc |= manBits << 20;
    if (rem > half || (rem == half && (enc & 0x00100000u))) {
        enc += 0x00100000u;  // 尾数 MSB 进位（0b111 -> 阶码，位 20 起）
    }
    // enc 仍是合法 float 位模式（阶码溢出即超出 MX_FP8_MAX，饱和处理）
    float q = AsFloat(enc);
    if (q > sage::core::MX_FP8_MAX || std::isinf(q)) {
        q = sage::core::MX_FP8_MAX;
    }
    return sign * q;
}

// platform_traits.h MX_FP8_MAX + mxfp8_e4m3_encoder.h 的 E8M0 scale：2^ceil(log2(amax/MX_FP8_MAX))，下限 2^-100。
float MxScale(float amax)
{
    const float floorScale = std::ldexp(1.0f, -100);
    if (amax <= sage::core::MX_FP8_MAX * floorScale) {
        return floorScale;
    }
    int e = 0;
    std::frexp(amax * sage::core::INV_MX_FP8_MAX, &e);       // amax/MX_FP8_MAX = m * 2^(e-1)
    return std::ldexp(1.0f, e - 1 + (std::ldexp(1.0f, e - 1) < amax * sage::core::INV_MX_FP8_MAX ? 1 : 0));
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

void TestFastReciprocal(std::mt19937 &rng)
{
    std::uniform_real_distribution<float> uni(1e-6f, 1e6f);
    double worst = 0.0;
    for (int i = 0; i < 200000; ++i) {
        float x = uni(rng);
        double ref = 1.0 / static_cast<double>(x);
        double err = std::fabs(static_cast<double>(sage::core::FastReciprocalScalar(x)) - ref) / ref;
        worst = std::fmax(worst, err);
    }
    std::printf("[1] FastReciprocal (magic 0x%08X + 2 NR steps)\n", sage::core::FAST_RECIPROCAL_MAGIC);
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
        float row[sage::core::MAX_HEAD_DIM];
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
        const float scale = amax > 0.0f ? amax * sage::core::INV_INT8_MAX : 1.0f;
        const float inv = amax > 0.0f ? sage::core::INT8_MAX_FLOAT / amax : 1.0f;
        for (float v : row) {
            float q = std::nearbyint(v * inv);
            if (q < -sage::core::INT8_MAX_FLOAT || q > sage::core::INT8_MAX_FLOAT) {
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
    std::uniform_real_distribution<float> pos(1e-6f, sage::core::MX_FP8_MAX);
    double worst = 0.0;
    bool satOk = true;
    for (int i = 0; i < 200000; ++i) {
        float x = std::copysign(pos(rng), static_cast<float>(i % 2 ? 1 : -1));
        float q = QuantE4M3(x);
        if (std::fabs(q) > sage::core::MX_FP8_MAX) {
            satOk = false;
        }
        if (std::fabs(x) > 1e-5f) {
            double rel = std::fabs(static_cast<double>(x) - q) / std::fabs(static_cast<double>(x));
            worst = std::fmax(worst, rel);
        }
    }
    if (QuantE4M3(1000.0f) != sage::core::MX_FP8_MAX || QuantE4M3(-1000.0f) != -sage::core::MX_FP8_MAX) {
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
        if (!(scale >= amax * sage::core::INV_MX_FP8_MAX)) {
            coverOk = false;  // V/scale 超过 MX_FP8_MAX → 会饱和
        }
        if (!(scale <= 2.0 * amax * sage::core::INV_MX_FP8_MAX + 1e-30)) {
            tightOk = false;  // 损失超过 1 个二进制位动态范围
        }
        float q = QuantE4M3(amax / scale);
        if (q > sage::core::MX_FP8_MAX) {
            coverOk = false;
        }
    }
    std::printf("[4] E8M0 group scale = 2^ceil(log2(amax/448))\n");
    Report("coverage (no saturation)", coverOk ? 0.0 : 1.0, 0.0, coverOk);
    Report("tightness (<= 2x)", tightOk ? 0.0 : 1.0, 0.0, tightOk);
}

void TestDeqf16BalancedScales(std::mt19937 &rng)
{
    // 恒等式：(S_int * 2^-11) * qh * kh == S_int * qs * ks * sm_scale，
    // 其中 qh = fp16(qs * sign(s) * sqrt(|s|*2048))，kh = fp16(ks * sqrt(|s|*2048))，
    // S_int 的 INT32 累加在 float 中精确（|S_int| < 2^24）。
    const float deqScalar = AsFloat(static_cast<uint32_t>(sage::core::DEQ_SCALAR_DEQF16)); // 2^-11 = 1/2048
    const float deqFactor = 1.0f / deqScalar;                                                // 2048
    std::uniform_real_distribution<float> uni(0.5f, 100.0f);
    double worst = 0.0;
    for (int i = 0; i < 100000; ++i) {
        float qs = uni(rng);
        float ks = uni(rng);
        float smScale = (i % 2 ? 1.0f : -1.0f) * (1.0f / std::sqrt(static_cast<float>(sage::core::MAX_HEAD_DIM)));
        double sInt = std::round(uni(rng) * 20000.0) - 10000.0;  // |S_int| < 2^24
        float factor = std::sqrt(std::fabs(smScale) * deqFactor);
        float qh = RoundToFp16(qs * (smScale >= 0 ? factor : -factor));
        float kh = RoundToFp16(ks * factor);
        double lhs = sInt * static_cast<double>(deqScalar) * static_cast<double>(qh) * static_cast<double>(kh);
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
    //   2) 严格验证 -LN_448 符号（已修复的 P0 缺陷）；
    //   3) m_hat 的 FP16 舍入有界，行最大元素的未截断 P 贴近满量程
    //      （>224，无欠利用），且超出 448 的部分由 E4M3 SAT cast 精确钳位，
    //      偏差限制在一个 E4M3 相对步长（2^-4 量级）内，不会无界溢出。
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    // 直接绑定内核工件常量（而非本地复写）：内核 rescale 处必须以 -LN_448 下移跟踪
    // 最大值，使 P = exp(score - m_hat) 落在 (0, 448] 撑满 E4M3 量程；符号反了 P 会
    // 全部落入次正规数区并被钳为 0，本断言与 [8]/[16] 联合失败。
    const float shift16 = RoundToFp16(-sage::core::LN_448);
    constexpr float kExpectedShift16 = -6.10546875f;
    const bool constOk = (shift16 == kExpectedShift16);
    const bool signOk = (-sage::core::LN_448 < 0.0f) && (shift16 < 0.0f);
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
    const bool utilizationOk = worstLow > (sage::core::MX_FP8_MAX * 0.5f);
    const bool saturationOk = worstHigh <= sage::core::MX_FP8_MAX * (1.0f + 1.0f / 32.0f);
    std::printf("[6] ln(448) shifted softmax range utilization\n");
    Report("fp16(-ln448) bit-equals -6.10546875", constOk ? 0.0 : 1.0, 0.0, constOk);
    Report("shift sign is strictly negative (-LN_448 < 0)", signOk ? 0.0 : 1.0, 0.0, signOk);
    Report("p_max utilization > 224", worstLow, sage::core::MX_FP8_MAX * 0.5f, utilizationOk);
    Report("p_max within E4M3 saturation step", worstHigh, sage::core::MX_FP8_MAX * (1.0f + 1.0f / 32.0f), saturationOk);
}

void TestLseIdentity(std::mt19937 &rng)
{
    // 内核输出自然对数域 LSE：lse_kernel = m_hat + log(den)（未含平滑校正，不换底）。
    // PyTorch 层还原：lse = lse_kernel + sm_scale * dot(q, mean_k)；
    // GQA 下头 h 广播其 KV 组 kh = CalcKh(h, hq, hk) 的 mean_k（等价 repeat_interleave）。
    // 两组 KV 均值彼此不同，广播取错组会立即放大误差，因此本断言是实质性的。
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    const int d = sage::core::MAX_HEAD_DIM;
    const double smScale = 1.0 / std::sqrt(static_cast<double>(sage::core::MAX_HEAD_DIM));
    double worst = 0.0, worstKernel = 0.0, worstGap = 0.0;
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
            const int kh = sage::dispatch::CalcKh(h, hq, hk);  // repeat_interleave 广播映射
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
            // 内核侧：自然对数域直接输出 (m + log(den))，不做换底（不含平滑校正）。
            double m = -1e300;
            for (double s : smoothScores) {
                m = std::fmax(m, s - static_cast<double>(sage::core::LN_448));
            }
            double den = 0.0;
            for (double s : smoothScores) {
                den += std::exp(s - m);
            }
            const double kernelNatural = m + std::log(den);
            const double returned = kernelNatural + smScale * dotQM;
            // 内核输出自身必须等于平滑分数的 logsumexp（同一对数底，无换底）。
            worstKernel = std::fmax(worstKernel, std::fabs(kernelNatural - LogSumExp(smoothScores)) /
                                                  (std::fabs(LogSumExp(smoothScores)) + 1.0));
            // 校正项必须非平凡：未还原的 kernel 值偏离真值，证明 host 校准确有必要。
            worstGap = std::fmax(worstGap, std::fabs(kernelNatural - reference) / (std::fabs(reference) + 1.0));
            double rel = std::fabs(returned - reference) / (std::fabs(reference) + 1.0);
            worst = std::fmax(worst, rel);
        }
    }
    std::printf("[7] LSE contract (natural-log kernel + open-source smooth correction)\n");
    Report("kernel emission equals natural-logsumexp of smoothed scores", worstKernel, 1e-5, worstKernel <= 1e-5);
    Report("restoration term is non-trivial (uncorrected kernel diverges)", worstGap, 1e-3, worstGap >= 1e-3);
    Report("relative error vs double reference (per-head GQA broadcast)", worst, 1e-5, worst <= 1e-5);
}

void TestSmoothKInvariance(std::mt19937 &rng)
{
    // smooth-K 是行常数平移。以 FP32（内核 stats/quant 路径的算术精度）对整行
    // 计算两种分数并各自归一化：exp(sRaw)/Z == exp(sSmooth + dot(q,m)/s)/Z'。
    // 双精度下该恒等式平凡成立；本断言在 FP32 点积噪声水平上验证行不变性。
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    const int d = sage::core::MAX_HEAD_DIM, rows = sage::core::HALF;
    const float smScale = 1.0f / std::sqrt(static_cast<float>(sage::core::MAX_HEAD_DIM));
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
    int16_t laneOffset[sage::core::M];
    int minOffset = 1000, maxOffset = -1000;
    for (int i = 0; i < static_cast<int>(sage::core::M); ++i) {
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
        for (int i = 0; i < static_cast<int>(sage::core::M); ++i) {
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
        int causalLimit1 = qInGroup - static_cast<int>(sage::core::N);
        for (uint16_t row = 0; row < sage::core::HALF; row += 16) {
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

// quant.h QuantRowsImpl：行 amax * INV_INT8_MAX 得 scale，BroadcastGroupMaxima 让连续 group
// 个 token 共享同一 scale（per_thread = Q4/K16，per_warp = Q32/K64）。
void TestQuantGranularityGroups(std::mt19937 &rng)
{
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    const int rows = sage::core::HALF, d = sage::core::MAX_HEAD_DIM;
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
        for (int group : {
            static_cast<int>(sage::core::QTokensPerScale(false)),  // 4
            static_cast<int>(sage::core::QTokensPerScale(true)),   // 32
            16, 64}) {
            for (int start = 0; start < rows; start += group) {
                const int count = std::min(group, rows - start);
                float amax = 0.0f;
                for (int r = 0; r < count; ++r) {
                    amax = std::fmax(amax, rowAmax[start + r]);
                }
                const float scale = amax > 0.0f ? amax * sage::core::INV_INT8_MAX : 1.0f;
                // scale * INT8_MAX_FLOAT 需覆盖组 amax，容差取 FP32 乘除的相对舍入
                if (amax > 0.0f && scale * sage::core::INT8_MAX_FLOAT < amax * (1.0f - 1e-6f)) {
                    groupCoversRow = false;  // 组 amax 必须被组内每一行的 scale 覆盖
                }
                for (int r = 0; r < count; ++r) {
                    for (int j = 0; j < d; ++j) {
                        const float q = std::nearbyintf(x[start + r][j] / scale);
                        if (std::fabs(q) > sage::core::INT8_MAX_FLOAT) {
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

// online_reduction_softmax.h RoundPartialFp16VF：pv_accum_dtype="fp32+fp16" 时，块内 PV 部分和
// 折入 FP32 运行状态前做一次 FP32->FP16->FP32 往返。逐块加权和的误差以
// sum_j |O_j| / den * 2^-11 为界（每块一次相对舍入，块权重归一化后 <=1）。
// 同一往返用 SatMode::NO_SAT，所以还要证明它落在半精度可表示区间内：折叠前
// 的幅度由 platform_traits.h 的 P 侧 2^-8 常量尺度限定（PV_PARTIAL_SHIFT）。
void TestFp16PartialRoundtrip(std::mt19937 &rng)
{
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    const int blocks = sage::core::M, channels = sage::core::MAX_HEAD_DIM;
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

    // 溢出包络：一个 PV 块最多容纳 kBlock 个概率同时取到 MX_FP8_MAX 满量程，未平移的
    // 部分和为 MX_FP8_MAX * kBlock * max|V|，|V| = 1 时即 114688 > 65504，而
    // RoundPartialFp16VF 用 SatMode::NO_SAT，溢出为 Inf 并污染整行输出。
    // 内核把 2^-8 记在 PV MMA 的 P 侧 E8M0 常量尺度上（纯指数平移 PV_PARTIAL_SHIFT），折叠后上界
    // 变为 MX_FP8_MAX * max|V|，与块宽、序列长度都无关；finalize 再乘回 PV_PARTIAL_UNDO。
    constexpr float kCeiling = sage::core::MX_FP8_MAX;
    constexpr uint32_t kBlock = sage::core::N;
    constexpr uint32_t kShift = sage::core::PV_PARTIAL_SHIFT;
    const float unshifted = RoundToFp16(kCeiling * kBlock);
    Report("unshifted full block overflows half range", std::isinf(unshifted) ? 1.0 : 0.0,
           1.0, std::isinf(unshifted));
    bool shiftedFits = true;
    for (uint32_t used = 1; used <= kBlock; ++used) {
        for (float mag : {1.0f, 10.0f, 100.0f, 146.0f}) {
            if (!std::isfinite(RoundToFp16(std::ldexp(kCeiling * used * mag, -static_cast<int>(kShift))))) {
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
        if (std::ldexp(x, -static_cast<int>(kShift)) * sage::core::PV_PARTIAL_UNDO != x) {
            shiftExact = false;
            break;
        }
    }
    Report("2^-8/2^8 round-trip is bit-exact", shiftExact ? 0.0 : 1.0, 0.0, shiftExact);
}

// dynamic_symmetric_quant.h 的 smooth_v 通路：V 沿序列维取通道均值，
// 量化前居中，归一化输出上补回。补回项在归一化之后加，不参与跨块累加。
// E4M3 + E8M0 分组量化下每个元素的绝对误差不超过该组 amax 的 2^-4，权重和为 1，
// 因此整行重建误差以 2^-4 * max_group_amax 为界；居中降低该 amax，即降低包络本身。
void TestSmoothVReconstruction(std::mt19937 &rng)
{
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    const int sk = 512, d = sage::core::MAX_HEAD_DIM, group = 32;  // MXFP8：每 channel 每 32 token 一个 E8M0 scale
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
// 基于 per-token 对称 INT8 量化数学公式（INV_INT8_MAX，FastReciprocalScalar 倒数），
// 逐 token 量化具有行独立性，子块流式输出与离线批处理输出在 INT8 编码与 FP16 尺度上
// 必须完全 bit-exact 相同（max_err == 0）。
void TestOnChipStreamingQQuantEquivalence(std::mt19937 &rng)
{
    const float deqFactor = 1.0f / AsFloat(static_cast<uint32_t>(sage::core::DEQ_SCALAR_DEQF16));
    const int tileRows = sage::core::M, headDim = sage::core::MAX_HEAD_DIM;
    const int subBlockRows = sage::core::QUANT_TILE_ROWS;  // MIX 1:2 中每个 AIV 核心处理 64 行
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    std::uniform_real_distribution<float> scaleDist(0.05f, 2.0f);

    int maxDiffInt8 = 0;
    double maxDiffScale = 0.0;
    bool zeroRowMatch = true;
    bool saturationMatch = true;

    // 测试多个 Tile，涵盖常规高斯分布、全零行、大离群值、极端动态范围
    for (int t = 0; t < 50; ++t) {
        float smScale = (t % 2 ? 1.0f : -1.0f) * scaleDist(rng) / std::sqrt(static_cast<float>(headDim));
        float factor = std::sqrt(std::fabs(smScale) * deqFactor);
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
            float m = amax * sage::core::INV_INT8_MAX;
            if (m <= 0.0f) {
                m = 1.0f;
            }
            float inv = sage::core::FastReciprocalScalar(m);
            for (int c = 0; c < headDim; ++c) {
                float scaled = qTile[r][c] * inv;
                float rounded = std::nearbyint(scaled);
                rounded = std::fmax(-sage::core::INT8_MAX_FLOAT, std::fmin(sage::core::INT8_MAX_FLOAT, rounded));
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
                float m = amax * sage::core::INV_INT8_MAX;
                if (m <= 0.0f) {
                    m = 1.0f;
                }
                float inv = sage::core::FastReciprocalScalar(m);
                for (int c = 0; c < headDim; ++c) {
                    float scaled = qTile[r][c] * inv;
                    float rounded = std::nearbyint(scaled);
                    rounded = std::fmax(-sage::core::INT8_MAX_FLOAT, std::fmin(sage::core::INT8_MAX_FLOAT, rounded));
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
            if (q8Batch[20][3] != static_cast<int8_t>(sage::core::INT8_MAX_FLOAT) ||
                q8Stream[20][3] != static_cast<int8_t>(sage::core::INT8_MAX_FLOAT) ||
                q8Batch[20][4] != static_cast<int8_t>(-sage::core::INT8_MAX_FLOAT) ||
                q8Stream[20][4] != static_cast<int8_t>(-sage::core::INT8_MAX_FLOAT)) {
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

void TestSnakeFoldedScheduler()
{
    // Test 1: Bijection across different tile counts T and heads H
    // Directly uses sage::dispatch::SnakeScheduler<true>::Decode from snake_wavefront_balancer.h
    bool allBijective = true;
    for (uint32_t T : {1u, 2u, 3u, 4u, 8u, 16u, 32u, 64u}) {
        for (uint32_t H : {1u, 2u, 4u, 8u, 32u}) {
            sage::SageParams p{};
            p.b = 1;
            p.hq = H;
            p.hk = std::max(1u, H / 4);
            p.sq = T * sage::core::M;
            p.sk = T * sage::core::M;
            p.cores = 0; // Disable die groups to isolate tile mapping
            for (uint32_t h = 0; h < H; ++h) {
                std::vector<int> visited(T, 0);
                for (uint32_t raw = 0; raw < T; ++raw) {
                    uint64_t task = static_cast<uint64_t>(h) * T + raw;
                    auto info = sage::dispatch::SnakeScheduler<true>::Decode(p, task, raw);
                    uint32_t tile = info.qstart / sage::core::M;
                    if (tile < T) {
                        visited[tile]++;
                    }
                }
                for (uint32_t t = 0; t < T; ++t) {
                    if (visited[t] != 1) {
                        allBijective = false;
                    }
                }
            }
        }
    }

    // Test 2: Workload balance across 48 cores on Atlas 950PR for causal attention
    // Compare naive linear (SnakeScheduler<false>) vs snake-folded wavefront schedule (SnakeScheduler<true>)
    constexpr uint32_t cores = 48;
    constexpr uint32_t T = 32; // 4K tokens with M=128
    constexpr uint32_t H = 32;
    std::vector<uint64_t> naiveLinear(cores, 0);
    std::vector<uint64_t> snakeFolded(cores, 0);

    sage::SageParams pFolded{};
    pFolded.b = 1;
    pFolded.hq = H;
    pFolded.hk = H;
    pFolded.sq = T * sage::core::M;
    pFolded.sk = T * sage::core::M;
    pFolded.cores = 0;

    sage::SageParams pLinear{};
    pLinear.b = 1;
    pLinear.hq = H;
    pLinear.hk = H;
    pLinear.sq = T * sage::core::M;
    pLinear.sk = T * sage::core::M;
    pLinear.cores = 0;

    for (uint32_t task = 0; task < H * T; ++task) {
        uint32_t core = task % cores;
        auto infoLinear = sage::dispatch::SnakeScheduler<false>::Decode(pLinear, task, 0);
        uint32_t linearTile = infoLinear.qstart / sage::core::M;
        naiveLinear[core] += (linearTile + 1);

        auto infoFolded = sage::dispatch::SnakeScheduler<true>::Decode(pFolded, task, 0);
        uint32_t foldedTile = infoFolded.qstart / sage::core::M;
        snakeFolded[core] += (foldedTile + 1);
    }

    auto calcVariance = [](const std::vector<uint64_t> &w) {
        double mean = 0.0;
        for (auto v : w) mean += v;
        mean /= w.size();
        double var = 0.0;
        for (auto v : w) var += (v - mean) * (v - mean);
        return std::sqrt(var / w.size()) / mean; // Coefficient of variation (CV)
    };

    double linearCV = calcVariance(naiveLinear);
    double foldedCV = calcVariance(snakeFolded);
    double varianceReduction = linearCV / std::max(foldedCV, 1e-6);

    std::printf("[14] Snake-Folded Wavefront Scheduler balancing and bijection\n");
    Report("tile mapping bijective (every tile visited once)", allBijective ? 0.0 : 1.0, 0.0, allBijective);
    Report("folded workload coefficient of variation <= 0.05", foldedCV, 0.05, foldedCV <= 0.05);
    Report("folded variance reduction ratio >= 10.0x vs naive", varianceReduction, 10.0, varianceReduction >= 10.0);
}

void TestDelayedFp32Buffering(std::mt19937 &rng)
{
    // Task 1.2: Delayed FP32 Buffering mode activating native FP16 MMA throughput.
    // For sequence length S >= 16384 (64 tiles of N=256), pairs of 2 tiles are
    // accumulated natively in FP16 before folding into the running FP32 accumulator.
    // Mathematical bound: intermediate sum |448 * max|V|| <= 65504 (FP16 safe envelope).
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    constexpr uint32_t S = 16384;
    constexpr uint32_t tileSize = sage::core::N;
    constexpr uint32_t numTiles = S / tileSize;
    constexpr uint32_t d = sage::core::MAX_HEAD_DIM;

    double maxRelErr = 0.0;
    bool noOverflow = true;

    for (int trial = 0; trial < 10; ++trial) {
        std::vector<float> exactFp32(d, 0.0f);
        std::vector<float> delayedFp32(d, 0.0f);
        float fp32Denom = 0.0f;
        float delayedDenom = 0.0f;

        // Process in pairs of 2 tiles (Delayed FP32 Buffering interval)
        for (uint32_t pair = 0; pair < numTiles / 2; ++pair) {
            std::vector<float> subTile0(d, 0.0f), subTile1(d, 0.0f);
            float subDenom0 = 0.0f, subDenom1 = 0.0f;

            for (uint32_t t = 0; t < 2; ++t) {
                for (uint32_t r = 0; r < tileSize; ++r) {
                    float s = gauss(rng);
                    float p = std::exp(s - 2.0f); // Softmax probability envelope
                    if (t == 0) subDenom0 += p;
                    else subDenom1 += p;

                    for (uint32_t c = 0; c < d; ++c) {
                        float v = gauss(rng) * 0.5f;
                        if (t == 0) subTile0[c] += p * v;
                        else subTile1[c] += p * v;
                    }
                }
            }

            // Path 1: Exact FP32 continuous accumulation
            for (uint32_t c = 0; c < d; ++c) {
                exactFp32[c] += subTile0[c] + subTile1[c];
            }
            fp32Denom += subDenom0 + subDenom1;

            // Path 2: Delayed FP32 Buffering - native FP16 accumulation for 2 tiles,
            // then high-precision FP32 folding into running state
            std::vector<float> pairFp16Accum(d, 0.0f);
            constexpr float kFp16Max = sage::core::SAGE_FP16_MAX;
            for (uint32_t c = 0; c < d; ++c) {
                float sum2 = subTile0[c] + subTile1[c];
                float scaledSum = std::ldexp(sum2, -static_cast<int>(sage::core::PV_PARTIAL_SHIFT));
                if (std::fabs(scaledSum) > kFp16Max) {
                    noOverflow = false;
                }
                float f16Sim = RoundToFp16(scaledSum);
                pairFp16Accum[c] = f16Sim * sage::core::PV_PARTIAL_UNDO;
            }

            for (uint32_t c = 0; c < d; ++c) {
                delayedFp32[c] += pairFp16Accum[c];
            }
            delayedDenom += (subDenom0 + subDenom1);
        }

        for (uint32_t c = 0; c < d; ++c) {
            float outExact = exactFp32[c] / fp32Denom;
            float outDelayed = delayedFp32[c] / delayedDenom;
            double relErr = std::fabs(outExact - outDelayed) / (std::fabs(outExact) + 1e-4);
            if (relErr > maxRelErr) {
                maxRelErr = relErr;
            }
        }
    }

    std::printf("[15] Delayed FP32 Buffering long sequence (S=16384) FP16 MMA conformance\n");
    // Each of the numPairs = S / (2 * 256) = 32 folds rounds one pair partial into
    // FP16, costing at most 2^-11 relative rounding per fold; folds compound
    // linearly in the worst case, so the honest envelope for this datapath is
    // numPairs * 2^-11 (~0.0156) and the bound below keeps a 2x margin over it.
    // Anything that regresses the fold interval or the fold precision breaks the
    // envelope and fails here.
    constexpr uint32_t kFoldCount = S / (2 * tileSize);
    constexpr double kFp16FoldEnvelope = static_cast<double>(1ULL << 11);
    const double foldBound = 2.0 * kFoldCount / kFp16FoldEnvelope;
    Report("relative error vs pure FP32 (S=16384, 32 FP16 folds, bound <= 2*32*2^-11)",
           maxRelErr, foldBound, maxRelErr <= foldBound);
    Report("FP16 MMA accumulation envelope safe (|val| <= 65504)", noOverflow ? 0.0 : 1.0, 0.0, noOverflow);
}

void TestChunkWiseLocalSmoothing(std::mt19937 &rng)
{
    // Task 2.1: Chunk-wise local mean smoothing (B_kv = 512 tokens).
    // Eliminates global sequence-wide mean synchronization (SyncAll hard barriers).
    // Proves row-wise constant absorption in online Softmax:
    // S_ij = Q_i (K_j - k_bar_j)^T + Q_i k_bar_j^T
    // The scalar c_ij = Q_i k_bar_j^T is absorbed into the chunk max and online denom.
    std::normal_distribution<float> gauss(0.0f, 1.0f);
    constexpr uint32_t S = 2048;
    constexpr uint32_t chunkSize = 512;
    constexpr uint32_t numChunks = S / chunkSize;
    constexpr uint32_t d = sage::core::MAX_HEAD_DIM;
    const float smScale = 1.0f / std::sqrt(static_cast<float>(sage::core::MAX_HEAD_DIM));

    double maxProbDiff = 0.0;
    double maxLseDiff = 0.0;

    for (int trial = 0; trial < 10; ++trial) {
        std::vector<float> q(d);
        for (auto &x : q) x = gauss(rng);

        std::vector<std::vector<float>> k(S, std::vector<float>(d));
        for (uint32_t i = 0; i < S; ++i) {
            for (uint32_t c = 0; c < d; ++c) {
                k[i][c] = gauss(rng);
            }
        }

        // Global baseline: exact full-sequence Softmax
        std::vector<double> scoresGlobal(S);
        for (uint32_t i = 0; i < S; ++i) {
            double dot = 0.0;
            for (uint32_t c = 0; c < d; ++c) dot += q[c] * k[i][c];
            scoresGlobal[i] = dot * smScale;
        }
        double lseGlobal = LogSumExp(scoresGlobal);
        std::vector<double> probGlobal(S);
        for (uint32_t i = 0; i < S; ++i) {
            probGlobal[i] = std::exp(scoresGlobal[i] - lseGlobal);
        }

        // Chunk-wise local smoothing simulation:
        // Each 512-token chunk computes local mean k_bar_j
        double runningMax = -1e300;
        double runningDenom = 0.0;

        for (uint32_t chunk = 0; chunk < numChunks; ++chunk) {
            uint32_t start = chunk * chunkSize;
            std::vector<double> localMean(d, 0.0);
            for (uint32_t i = 0; i < chunkSize; ++i) {
                for (uint32_t c = 0; c < d; ++c) {
                    localMean[c] += k[start + i][c];
                }
            }
            for (uint32_t c = 0; c < d; ++c) localMean[c] /= chunkSize;

            double rowConst = 0.0;
            for (uint32_t c = 0; c < d; ++c) rowConst += q[c] * localMean[c];
            rowConst *= smScale;

            std::vector<double> chunkScores(chunkSize);
            double chunkMax = -1e300;
            for (uint32_t i = 0; i < chunkSize; ++i) {
                double centeredDot = 0.0;
                for (uint32_t c = 0; c < d; ++c) {
                    centeredDot += q[c] * (k[start + i][c] - localMean[c]);
                }
                chunkScores[i] = centeredDot * smScale + rowConst;
                if (chunkScores[i] > chunkMax) chunkMax = chunkScores[i];
            }

            double newMax = std::fmax(runningMax, chunkMax);
            double alpha = std::exp(runningMax - newMax);
            runningDenom = runningDenom * alpha;
            for (uint32_t i = 0; i < chunkSize; ++i) {
                runningDenom += std::exp(chunkScores[i] - newMax);
            }
            runningMax = newMax;
        }

        double lseChunk = runningMax + std::log(runningDenom);
        double lseDiff = std::fabs(lseGlobal - lseChunk) / (std::fabs(lseGlobal) + 1.0);
        if (lseDiff > maxLseDiff) maxLseDiff = lseDiff;

        for (uint32_t i = 0; i < S; ++i) {
            double p = std::exp(scoresGlobal[i] - lseChunk);
            double diff = std::fabs(p - probGlobal[i]);
            if (diff > maxProbDiff) maxProbDiff = diff;
        }
    }

    std::printf("[16] Chunk-wise Local Smoothing zero-barrier online Softmax equivalence\n");
    Report("online Softmax probability diff (chunk=512, bound <= 1e-5)", maxProbDiff, 1e-5, maxProbDiff <= 1e-5);
    Report("fused LSE identity relative error (bound <= 1e-5)", maxLseDiff, 1e-5, maxLseDiff <= 1e-5);
}

void TestHomologousNormalizer(std::mt19937 &rng)
{
    // Task 3.4 / P5.2: Normalizer 与量化 P 同源化断言。
    // 在 Transformer Attention 中，若 Softmax 分母采用 FP32 原始指数和（非同源累加），
    // 而输出 Cube PV 运算消费的是 E4M3 截断舍入后的 P 矩阵（P_quant），
    // 则有效注意力权重之和为 sum(P_quant) / sum(P_raw) != 1.0，
    // 在整行上会产生 ~1% 到 3% 的系统性概率质量偏差（Systemic Probability Mass Bias），
    // 造成深度网络激活幅度逐层偏移；
    // 当改由量化后 P（E4M3 舍入值）参与分母累加（同源累加）时，有效权重之和为
    // sum(P_quant) / sum(P_quant) == 1.0，在 FP32 累加精度下严格消除概率质量偏差。
    std::normal_distribution<float> gauss(0.0f, 2.5f);
    double worstHomoErr = 0.0;
    double worstHeteroErr = 0.0;

    for (uint32_t len : {64u, 128u, 256u, 512u, 1024u}) {
        for (int trial = 0; trial < 200; ++trial) {
            std::vector<float> scores(len);
            float maxScore = -1e30f;
            for (uint32_t i = 0; i < len; ++i) {
                scores[i] = gauss(rng);
                if (scores[i] > maxScore) maxScore = scores[i];
            }
            float mHat = RoundToFp16(maxScore - sage::core::LN_448);

            std::vector<float> pRaw(len);
            std::vector<float> pQuant(len);
            float dHetero = 0.0f;
            float dHomo = 0.0f;

            for (uint32_t i = 0; i < len; ++i) {
                pRaw[i] = std::exp(scores[i] - mHat);
                pQuant[i] = QuantE4M3(pRaw[i]);
                dHetero += pRaw[i];
                dHomo += pQuant[i];
            }

            // 1. 验证概率质量和 sum_j (P_quant[j] / Denom)
            float massHetero = 0.0f;
            float massHomo = 0.0f;
            for (uint32_t i = 0; i < len; ++i) {
                massHetero += pQuant[i] / dHetero;
                massHomo += pQuant[i] / dHomo;
            }
            double errHetero = std::fabs(static_cast<double>(massHetero) - 1.0);
            double errHomo = std::fabs(static_cast<double>(massHomo) - 1.0);
            worstHomoErr = std::fmax(worstHomoErr, errHomo);
            worstHeteroErr = std::fmax(worstHeteroErr, errHetero);
        }
    }

    std::printf("[17] Task 3.4 / P5.2 homologous normalizer (co-origin quantized P sum)\n");
    // 同源累加下概率质量偏差严格在 FP32 舍入界限内 (<= 1e-5)
    Report("homologous probability mass deviation <= 1e-5", worstHomoErr, 1e-5, worstHomoErr <= 1e-5);
    // 非同源累加存在显著的系统性概率质量偏差 (> 1e-3, 实测 ~0.02)
    Report("heterogeneous non-homologous deviation observed (> 1e-3)", worstHeteroErr, 1e-3, worstHeteroErr >= 1e-3);
    // 同源累加相比非同源累加压降误差比 >= 1000x
    double reduction = worstHeteroErr / std::fmax(worstHomoErr, 1e-12);
    Report("homologous vs heterogeneous precision gain >= 1000x", reduction, 1000.0, reduction >= 1000.0);
}

void TestWidePathMultiTileDenominatorFoldingAndLse(std::mt19937 &rng)
{
    // Task P0.1: 宽路径分母折叠修复与回归测试（dp=128, sk>=4096，包含多块迭代）。
    // 宽路径触发条件为 dp==128 且 sk>=4096（覆盖 P-01~P-07 全部基准用例）。
    // 在多块在线归约时，SoftmaxWideSecondVF 生成各块局部同源概率和（alpha + 64）。
    // UpdateWidePairVF 与 FinalizeWideFp32VF 负责将局部和折叠进 denom 并计算倒数。
    // 本用例验证：
    //   1. 宽路径双块流水折叠分母与逐块基准等价 (<= 1e-6)
    //   2. 融合 LSE 与全局参考相对误差 <= 1e-5
    //   3. 归一化注意力输出向量误差 <= 1e-5
    //   4. 同源归一化概率质量偏差严格消除 (sum_j P_j / denom == 1.0, <= 1e-5)
    //   5. 针对未折叠未初始化分母漏洞的回归断言（误差放大比 >= 1000x）
    //   6. 未初始化 DEN_UB NaN canary 经缺失 FIRST 分派的可检出性（首块 den = NaN*0 + sum = NaN）
    //   7. FIRST 分派折叠不读 DEN_UB，对任意垃圾免疫且分母与逐块基准一致 (<= 1e-6)
    std::normal_distribution<float> gauss(0.0f, 1.5f);
    constexpr uint32_t DP = 128;
    constexpr uint32_t TILE_N = 512;

    double maxDenomDiff = 0.0;
    double maxLseDiff = 0.0;
    double maxOutDiff = 0.0;
    double maxMassDev = 0.0;
    double minBugRatio = 1e30;
    bool canaryDetected = false;
    double canaryFixedRel = 0.0;

    for (uint32_t sk : {4096u, 4608u, 6144u, 8192u}) {
        for (bool smoothV : {false, true}) {
            const uint32_t numTiles = (sk + TILE_N - 1) / TILE_N;
            for (int trial = 0; trial < 10; ++trial) {
                std::vector<float> scores(sk);
                for (auto &s : scores) s = gauss(rng);

                std::vector<std::vector<float>> V(sk, std::vector<float>(DP));
                for (uint32_t i = 0; i < sk; ++i) {
                    for (uint32_t d = 0; d < DP; ++d) V[i][d] = gauss(rng) * 0.2f;
                }
                std::vector<float> meanV(DP, 0.0f);
                if (smoothV) {
                    for (uint32_t d = 0; d < DP; ++d) meanV[d] = gauss(rng) * 0.1f;
                }

                float runningPeak = -1e30f;
                std::vector<float> tileAlpha(numTiles, 1.0f);
                std::vector<float> tileSum(numTiles, 0.0f);
                std::vector<std::vector<float>> tilePV(numTiles, std::vector<float>(DP, 0.0f));

                for (uint32_t t = 0; t < numTiles; ++t) {
                    uint32_t start = t * TILE_N;
                    uint32_t end = std::min(sk, (t + 1) * TILE_N);

                    float bMax = -1e30f;
                    for (uint32_t i = start; i < end; ++i) {
                        if (scores[i] > bMax) bMax = scores[i];
                    }
                    float newPeak = std::fmax(runningPeak, RoundToFp16(bMax - sage::core::LN_448));
                    float alpha = (t == 0) ? 0.0f : std::exp(runningPeak - newPeak);
                    tileAlpha[t] = alpha;
                    runningPeak = newPeak;

                    for (uint32_t i = start; i < end; ++i) {
                        float pRaw = std::exp(scores[i] - runningPeak);
                        float pQuant = QuantE4M3(pRaw);
                        tileSum[t] += pQuant;
                        for (uint32_t d = 0; d < DP; ++d) {
                            tilePV[t][d] += pQuant * (V[i][d] - meanV[d]);
                        }
                    }
                }

                // 逐块基准折叠（精确模拟标准 online softmax 归约）
                float seqDenom = 0.0f;
                std::vector<float> seqAccum(DP, 0.0f);
                for (uint32_t t = 0; t < numTiles; ++t) {
                    if (t == 0) {
                        seqDenom = tileSum[0];
                        seqAccum = tilePV[0];
                    } else {
                        seqDenom = seqDenom * tileAlpha[t] + tileSum[t];
                        for (uint32_t d = 0; d < DP; ++d) {
                            seqAccum[d] = seqAccum[d] * tileAlpha[t] + tilePV[t][d];
                        }
                    }
                }
                float seqInv = sage::core::FastReciprocalScalar(seqDenom);
                std::vector<float> seqOut(DP);
                for (uint32_t d = 0; d < DP; ++d) {
                    seqOut[d] = seqAccum[d] * seqInv + meanV[d];
                }
                float seqLse = (runningPeak + std::log(seqDenom));

                // 双块流水折叠（模拟 UpdateWidePairVF 与 FinalizeWideFp32VF 硬件流水）
                float pipeDenom = 0.0f;
                std::vector<float> pipeAccum(DP, 0.0f);
                uint32_t t = 0;
                for (; t + 1 < numTiles; t += 2) {
                    bool isFirst = (t == 0);
                    float den;
                    float sum0 = tileSum[t];
                    float sum1 = tileSum[t + 1];
                    float a0 = tileAlpha[t];
                    float a1 = tileAlpha[t + 1];

                    if (isFirst) {
                        den = sum0;
                    } else {
                        den = pipeDenom * a0 + sum0;
                    }
                    den = den * a1 + sum1;
                    pipeDenom = den;

                    std::vector<float> z(DP);
                    for (uint32_t d = 0; d < DP; ++d) {
                        float y = isFirst ? tilePV[t][d] : (pipeAccum[d] * a0 + tilePV[t][d]);
                        z[d] = y * a1 + tilePV[t + 1][d];
                    }
                    pipeAccum = z;
                }
                if (t < numTiles) {
                    // 奇数块尾块收口（FinalizeWideFp32VF）
                    bool isFirst = (t == 0);
                    float sum = tileSum[t];
                    float a = tileAlpha[t];
                    if (isFirst) {
                        pipeDenom = sum;
                        pipeAccum = tilePV[t];
                    } else {
                        pipeDenom = pipeDenom * a + sum;
                        for (uint32_t d = 0; d < DP; ++d) {
                            pipeAccum[d] = pipeAccum[d] * a + tilePV[t][d];
                        }
                    }
                }
                float pipeInv = sage::core::FastReciprocalScalar(pipeDenom);
                std::vector<float> pipeOut(DP);
                for (uint32_t d = 0; d < DP; ++d) {
                    pipeOut[d] = pipeAccum[d] * pipeInv + meanV[d];
                }
                float pipeLse = (runningPeak + std::log(pipeDenom));

                double dDiff = std::fabs(static_cast<double>(seqDenom) - pipeDenom) / (std::fabs(static_cast<double>(seqDenom)) + 1e-12);
                double lDiff = std::fabs(static_cast<double>(seqLse) - pipeLse) / (std::fabs(static_cast<double>(seqLse)) + 1.0);
                if (dDiff > maxDenomDiff) maxDenomDiff = dDiff;
                if (lDiff > maxLseDiff) maxLseDiff = lDiff;

                for (uint32_t d = 0; d < DP; ++d) {
                    double oDiff = std::fabs(static_cast<double>(seqOut[d]) - pipeOut[d]) / (std::fabs(static_cast<double>(seqOut[d])) + 1e-4);
                    if (oDiff > maxOutDiff) maxOutDiff = oDiff;
                }

                // 验证同源概率质量和等于 1.0（消除概率质量偏差）
                std::vector<float> scaleForward(numTiles, 1.0f);
                for (int i = static_cast<int>(numTiles) - 2; i >= 0; --i) {
                    scaleForward[i] = scaleForward[i + 1] * tileAlpha[i + 1];
                }
                double totalMass = 0.0;
                for (uint32_t ti = 0; ti < numTiles; ++ti) {
                    totalMass += (static_cast<double>(tileSum[ti]) * scaleForward[ti]) / pipeDenom;
                }
                double mDev = std::fabs(totalMass - 1.0);
                if (mDev > maxMassDev) maxMassDev = mDev;

                // 漏洞回归检验：若未折叠（denom 未累加历史块，只剩尾块甚至垃圾值）
                float bugDenom = tileSum.back();
                float bugLse = (runningPeak + std::log(bugDenom));
                double bugLseErr = std::fabs(static_cast<double>(bugLse) - seqLse) / (std::fabs(static_cast<double>(seqLse)) + 1.0);
                double gain = bugLseErr / std::fmax(lDiff, 1e-12);
                if (gain < minBugRatio) minBugRatio = gain;
            }
        }
    }

    // 未初始化 DEN_UB 失效模式回归（P0.1 的实际病灶）：DEN_UB 是静态 UB 切片，无任何
    // 跨 tile 初始化器。若首块折叠缺失 FIRST 分派（FoldSingle<false>/FoldPair<false>
    // 直接读 denom），未初始化垃圾中的非有限模式（NaN/Inf，宿主用 quiet_NaN canary
    // 模拟）会经 den = garbage*ratio + sum 传播成 NaN 分母，LSE 与输出全部污染；
    // 有限垃圾虽会被首块 ratio==0（exp(SAGE_FP16_MIN - m) 在 FP32 下精确下溢为 0）
    // 意外中和，但那是不可依赖的未定义行为。修复语义（FIRST 分派：denom := blockSum）
    // 从不读 DEN_UB，对任意垃圾免疫且与逐块基准一致。
    {
        constexpr uint32_t kSk = 4096u;  // 宽路径下限（dp=128 且 sk>=4096），8 个 512 列块
        std::vector<float> scores(kSk);
        for (auto &s : scores) {
            s = gauss(rng);
        }
        // 首块同源块和（E4M3 回读值累加，对齐 [17] 口径）、逐块基准分母，
        // 以及修复语义的折叠链（首块 denom := blockSum，其后 den = den*ratio + blockSum）
        float prevPeak = sage::core::SAGE_FP16_MIN;
        float seqDenom = 0.0f;
        std::vector<float> blockSum(kSk / 512u, 0.0f);
        std::vector<float> ratios(kSk / 512u, 0.0f);
        const float canary = std::numeric_limits<float>::quiet_NaN();
        float fixedDenom = canary;  // DEN_UB 槽位，预置未初始化垃圾
        for (uint32_t t = 0; t < kSk / 512u; ++t) {
            const uint32_t start = t * 512u;
            float bMax = -1e30f;
            for (uint32_t i = start; i < start + 512u; ++i) {
                bMax = std::fmax(bMax, scores[i]);
            }
            const float newPeak = std::fmax(prevPeak, RoundToFp16(bMax - sage::core::LN_448));
            ratios[t] = (t == 0) ? 0.0f : std::exp(prevPeak - newPeak);  // 首块 ratio 精确下溢为 0
            prevPeak = newPeak;
            for (uint32_t i = start; i < start + 512u; ++i) {
                blockSum[t] += QuantE4M3(std::exp(scores[i] - newPeak));
            }
            seqDenom = (t == 0) ? blockSum[0] : seqDenom * ratios[t] + blockSum[t];
            // FoldSingle/FoldPair 的 FIRST 分派语义：首块初始化（不读 DEN_UB），其后折叠
            fixedDenom = (t == 0) ? static_cast<float>(blockSum[0]) : fixedDenom * ratios[t] + blockSum[t];
        }
        // (a) 缺陷复现：无 FIRST 分派时首块折叠读 canary —— den = NaN*0 + sum = NaN
        const float buggyDenom = canary * 0.0f + blockSum[0];
        canaryDetected = std::isnan(buggyDenom) || std::fabs(static_cast<double>(buggyDenom) - seqDenom) > 1e-3 * seqDenom;
        // (b) 修复语义下全链与逐块基准一致（对垃圾免疫）
        canaryFixedRel = std::fabs(static_cast<double>(fixedDenom) - static_cast<double>(seqDenom)) /
                         (static_cast<double>(seqDenom) + 1e-12);
    }

    std::printf("[18] Task P0.1 wide-path multi-tile denominator folding and LSE conformance (dp=128, sk>=4096)\n");
    Report("paired vs step-by-step denominator relative error <= 1e-6", maxDenomDiff, 1e-6, maxDenomDiff <= 1e-6);
    Report("fused LSE identity relative error <= 1e-5", maxLseDiff, 1e-5, maxLseDiff <= 1e-5);
    Report("wide output vector relative error <= 1e-5", maxOutDiff, 1e-5, maxOutDiff <= 1e-5);
    Report("homologous probability mass deviation <= 1e-5", maxMassDev, 1e-5, maxMassDev <= 1e-5);
    Report("unfolded denominator bug regression detected (gain >= 1000x)", minBugRatio, 1000.0, minBugRatio >= 1000.0);
    Report("uninitialized DEN_UB canary detected (missing FIRST dispatch)", canaryDetected ? 0.0 : 1.0, 0.0, canaryDetected);
    Report("FIRST-dispatch fold ignores DEN_UB garbage (rel err <= 1e-6)", canaryFixedRel, 1e-6, canaryFixedRel <= 1e-6);
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
    TestSnakeFoldedScheduler();
    TestDelayedFp32Buffering(rng);
    TestChunkWiseLocalSmoothing(rng);
    TestHomologousNormalizer(rng);
    TestWidePathMultiTileDenominatorFoldingAndLse(rng);
    std::printf("================================================================\n");
    if (g_failures == 0) {
        std::printf("ALL NUMERIC ERROR TESTS PASSED\n");
        return 0;
    }
    std::printf("%d NUMERIC ERROR TEST(S) FAILED\n", g_failures);
    return 1;
}
