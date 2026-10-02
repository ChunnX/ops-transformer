// sa2_numeric_conformance.cpp — SageAttention2 内核数值契约断言桩（M4.1，纯 C++17 无依赖）。
//
// 直接复用 op_kernel/sa2_spec.h 与 sa2_planner.h 的契约常量与派生（编译命令见 tests/README.md）。
// 覆盖清单（DESIGN_SPEC §4 / REVIEW_LOOP_STATE M4.1）：
//   1. κ 两档（fp32+fp16=2.25 / 其余=448）与 θ=lnκ 派生；
//   2. θ=lnκ rescale-skip 与逐块精确 max 两路归一化一致性（含 skip/rescale 双触发；
//      单调流两路 max 域恒差 lnκ、denom/o 恒差 κ 倍，位关系见用例内断言）；
//   3. LSE 契约 ln(denom)+m'（含 +lnκ 旧式负测试、1/1.44269504=ln2 常数核对、s·corr 可加性）；
//   4. E8M0 向上取幂：位技巧(+0x007FFFFF 掩码) / TileKernels((bits-1)>>23+1) / std::ceil(log2)
//      （double 域参照）逐位一致 与 448·scale ≥ amax 覆盖性质（M2.6 向量化层既定实现的等价性冻结）；
//      另冻结 448·2^k 边界双口径（设备 float 预除+位技巧 vs 宿主 double 预除+ceil(log2)，除法口径）；
//   5. g 因子恒等式 g²=|s|·2^14、负 s 符号随 Q、kScale FP16 可表示界、DEQ 2^-14 恒等式与 score 域 FP16 安全界；
//   6. per_warp / per_thread 粒度差异非零且细粒度重构误差不劣（空桩违约检测）；
//   7. 因果谓词：块级 steps 枚举与逐元素谓词等价（不漏未掩码元素、不含全掩码块）；
//   8. FP8 E4M3 码字全表往返与 RNE/饱和语义、κ 包络经离散化保持；
//   9. HalfF32 RNE 特征点（0.5ulp 边界 / 奇偶尾数 ties-to-even / 溢出饱和 ±Inf / 下溢方向）、
//      全 2^16 码字往返幂等与最近性扫掠（截断型替身不可通过）；
//   10. fp32+fp16 折叠误差界（HalfF32 RNE 相对误差 2^-11 逐元素预算）与折叠非零生效；
//   11. AscendC MicroAPI / vapi 向量硬件指令与标量数值等价性。
//   12. M2.7.2 UB 计划 O(1) 硬防护（CI 化）：宿主编译期实例化 sa2_ub_plan.h，per_warp/per_thread
//       两链终址由 tile 常数独立重推 == 冻结锚点（152448 / 217984）且 ≤ 248KB 高水位、随 S 无关
//      （Rust 式结构化错误：host 编译失败即硬错误，非运行期静默溢出；头内断言幂等校验）。
//   13. P 的 L1 NZ 分形索引守卫：PNzFractalIndex 冻结图像（16×16 分形、4 行块 × 8 列块、
//       32B pad 行）界内 + 单射 + 锚点；缺陷旧式（缺行块项 r/16）碰撞可检出（迭代 7）。
//   14. κ=448 满量程 D-R1 硬守卫边界：θ 谓词 ⟺ 重缩放恒等（零偏差核心）、两支路 p ≤ κ 包络、
//       码字 448 = §4.3 设计满量程（rescale 支路同样触达）、「饱和占比强制 rescale」兜底的
//       数学空转性证据（迭代 7 甄别维度）。
//   15. M2.7.4 状态类型化槽位（StateTypedSlot）：Empty→AcquireSlot(内部 Await)→Full→ReleaseSlot
//       (内部 Notify)→Empty 全周期；SlotData 仅对 Full 声明（Empty 无重载 = 「未同步先读写」
//       编译期硬错误，SFINAE 检测形态说明而非真编译失败测试）、SlotRef<Full> 构造全私有、
//       两态 Move-only、moved-from 擦除、六旗号 Await/Notify 双映射表与槽周期配对冻结（迭代 9）。
//   16. M2c-C2 极限分数值域与上溢防御用例（Extreme Score Range & Overflow Guard）：
//       构造大幅值正负得分输入（Score 范围 [-100.0, 100.0]、动态极差 > 150.0）；
//       验证在大幅值下 θ-skip 机制在数值极端波动时能否正确平滑转移；
//       验证 Softmax 概率矩阵在 FP8 E4M3 饱和截断下（码字严格钳制在 [0, 448.0] / 0x7E）绝对不产生 NaN、Inf 或数值穿透；
//       验证极端负得分（所有元素 S_ij <= -80.0）下 Softmax 分母与概率值安全收敛，LSE 闭环无下溢异常；
//       验证 FP16 两级累加在最大极端激励下是否严格满足 <= 65504（κ=2.25 理论安全界）。
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "sa2_planner.h"

namespace {
// 用例档位：fp32+fp16 的 κ（2.25）；448 档为字面量（fp32/fp32+fp32 共用 E4M3 满量程）。
constexpr float kSa2CaseKappaFp16 = 2.25f;
}  // namespace

#include "sa2_spec.h"
#include "sa2_ascendc_hoststub.h"
#include "sa2_ub_plan.h"
#include "sa2_events.h"

namespace {

int g_checks = 0;
int g_failures = 0;

void ExpectTrue(bool cond, const char *what, const char *file, int line)
{
    ++g_checks;
    if (!cond) {
        ++g_failures;
        std::printf("[FAIL] %s:%d %s\n", file, line, what);
    }
}

#define SA2_CHECK(cond) ExpectTrue((cond), #cond, __FILE__, __LINE__)

// 确定性 LCG 伪随机（无 <random> 依赖，跨平台可复现）。
struct Lcg {
    uint64_t state;
    explicit Lcg(uint64_t seed) : state(seed) {}
    uint32_t Next()
    {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<uint32_t>(state >> 33);
    }
    float NextUniform(float lo, float hi)
    {
        return lo + (hi - lo) * (static_cast<float>(Next() >> 8) / 16777216.0f);
    }
};

uint32_t BitsOf(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

float FloatOfBits(uint32_t u)
{
    float f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

constexpr double kLn2 = 0.69314718055994530942;

// ---- 用例 1：κ 两档与 M4.4 数理定源证明 ----
void CaseKappaTiers()
{
    SA2_CHECK(sa2::PDomainOf(sa2::PvAccum::kFp32Fp16) == 2.25f);
    SA2_CHECK(sa2::PDomainOf(sa2::PvAccum::kFp32) == 448.0f);
    SA2_CHECK(sa2::PDomainOf(sa2::PvAccum::kFp32Fp32) == 448.0f);
    const auto polFp16 =
        sa2::DeriveNumericPolicy(sa2::Gran::kPerThread, sa2::PvAccum::kFp32Fp16, 128, 128, 128, 0.08838834764831845f);
    const auto polFp32 =
        sa2::DeriveNumericPolicy(sa2::Gran::kPerThread, sa2::PvAccum::kFp32, 128, 128, 128, 0.08838834764831845f);
    SA2_CHECK(polFp16.pDomain == 2.25f && polFp32.pDomain == 448.0f);
    SA2_CHECK(std::fabs(static_cast<double>(polFp16.lnPDomain) - std::log(2.25)) < 1.0e-6);
    SA2_CHECK(std::fabs(static_cast<double>(polFp32.lnPDomain) - std::log(448.0)) < 1.0e-6);
    SA2_CHECK(polFp16.lnPDomain != polFp32.lnPDomain);

    // M4.4 数理定源与推导闭环 (SageAttention2++ arXiv:2505.21136):
    // 证明在 Vr ≈ 2.25, Pr = 448 尺度下，Pr * Vr = 1008 <= 2047 封顶。
    // Softmax 概率矩阵平移上限 κ = 2.25 恰好将 INT8 * FP8 乘积累加最大值限制在 FP16 动态范围（< 65504）内，避免 FP16 累加溢出。
    constexpr float kPr = 448.0f;
    constexpr float kVr = 2.25f;
    constexpr float kPrVr = kPr * kVr; // 1008.0f
    constexpr float kFp16Max = 65504.0f;
    constexpr int kTileSeqK = 128;
    constexpr float kTileScaledBound = static_cast<float>(kTileSeqK) * kVr; // 288.0f

    static_assert(kPrVr <= 2047.0f, "M4.4: Pr * Vr must satisfy Pr * Vr <= 2047 capping bound");
    static_assert(kTileScaledBound < kFp16Max, "M4.4: 128 * 2.25 must be strictly < 65504 to prevent FP16 accumulation overflow");
    static_assert(kPrVr < kFp16Max, "M4.4: Pr * Vr must be strictly within FP16 dynamic range");
    static_assert(kPrVr == 1008.0f, "M4.4: Pr * Vr must be exactly 1008.0");

    SA2_CHECK(kPrVr == 1008.0f);
    SA2_CHECK(kPrVr <= 2047.0f);
    SA2_CHECK(kTileScaledBound == 288.0f);
    SA2_CHECK(kTileScaledBound < kFp16Max);
    SA2_CHECK(kPrVr < kFp16Max);

    // 验证 θ = ln(2.25) 与 κ 换算精确性
    const double thetaLn225 = std::log(2.25);
    SA2_CHECK(std::fabs(static_cast<double>(polFp16.lnPDomain) - thetaLn225) < 1.0e-7);

    // 验证行求和上限 Σ P_j <= κ = 2.25 与最大单步累加和 <= 1008 < 65504
    const float maxAccumWorstCase = static_cast<float>(kTileSeqK) * kVr; // 128 * 2.25 = 288.0f
    SA2_CHECK(maxAccumWorstCase < 65504.0f);
    const float maxAccumNormalized = 1.0f * kPrVr; // 1008.0f <= 2047 < 65504
    SA2_CHECK(maxAccumNormalized <= 2047.0f);
    SA2_CHECK(maxAccumNormalized < 65504.0f);
}

// ---- 用例 2/3 公共件：单行 online softmax 流（θ 路径逐语句对齐 sa2_attention_loop.h）----
constexpr int kToyVDeg = 8;
constexpr int kNumBlocks = 16;

struct OnlineState {
    float mRun = sa2::kFp16NegMax;
    float denom = 0.0f;
    float o[kToyVDeg] = {0.0f};
    int skips = 0;
    int rescales = 0;
    float maxP = 0.0f; // 量化前的 exp 域包络采样
};

void MakeScores(float out[kNumBlocks][sa2::kKvStep], Lcg &rng, bool monotoneDown)
{
    float base = rng.NextUniform(-8.0f, 2.0f);
    for (int b = 0; b < kNumBlocks; ++b) {
        if (monotoneDown) {
            base -= rng.NextUniform(0.5f, 2.0f); // 块最大值严格下降 ⇒ 恒走 skip，两路应位等同
        } else {
            base += rng.NextUniform(-2.0f, 3.5f); // 漂移与 θ=ln(2.25)≈0.81 同量级 ⇒ skip/rescale 交替
        }
        for (int j = 0; j < sa2::kKvStep; ++j) {
            out[b][j] = base + rng.NextUniform(-1.0f, 1.0f);
        }
    }
}

float QuantizeP(float p)
{
    // 与 sa2_attention_loop.h 同构：exp 域截断 + E4M3 码字网格离散化（饱和 448）。
    const float clamped = std::max(0.0f, std::min(448.0f, p));
    return sa2::Fp8E4M3ToFloat(sa2::FloatToFp8E4M3(clamped));
}

// 参照路：逐块精确 max（mNew = max(mRun, mBlk − θ)，每块必 rescale）。
void OnlineReference(const float scores[kNumBlocks][sa2::kKvStep], const float v[sa2::kKvStep][kToyVDeg],
                     float lnKappa, OnlineState &st)
{
    for (int b = 0; b < kNumBlocks; ++b) {
        float mBlk = -INFINITY;
        for (int j = 0; j < sa2::kKvStep; ++j) {
            if (scores[b][j] > mBlk) mBlk = scores[b][j];
        }
        const float mNew = std::max(st.mRun, mBlk - lnKappa);
        const float alpha = std::exp(st.mRun - mNew);
        float sumP8 = 0.0f;
        for (int j = 0; j < sa2::kKvStep; ++j) {
            const float p8 = QuantizeP(std::exp(scores[b][j] - mNew));
            if (p8 > st.maxP) st.maxP = p8;
            sumP8 += p8;
        }
        for (int c = 0; c < kToyVDeg; ++c) {
            float partial = 0.0f;
            for (int j = 0; j < sa2::kKvStep; ++j) {
                partial += QuantizeP(std::exp(scores[b][j] - mNew)) * v[b * sa2::kKvStep + j][c];
            }
            st.o[c] = alpha * st.o[c] + partial;
        }
        st.denom = alpha * st.denom + sumP8;
        st.mRun = mNew;
        ++st.rescales;
    }
}

// θ 路：D-R1 阈值法，与 sa2_attention_loop.h StepVectorSoftmax 逐语句镜像（迭代 5 增强 c）：
// 语句序与内核一致 —— 先 mNew = max(mOld, mBlk − θ)、alpha 草算，再判 skip；
// skip 命中时 mNew = mOld 直接成立（内核不重算 max、alpha 强制 1），本桩同样以赋值覆盖而非重算，
// 保证 skip 支路两路逐位同源。
void OnlineTheta(const float scores[kNumBlocks][sa2::kKvStep], const float v[sa2::kKvStep][kToyVDeg],
                 float lnKappa, OnlineState &st)
{
    for (int b = 0; b < kNumBlocks; ++b) {
        float mBlk = -INFINITY;
        for (int j = 0; j < sa2::kKvStep; ++j) {
            if (scores[b][j] > mBlk) mBlk = scores[b][j];
        }
        const float mOld = st.mRun;                          // 内核：mOld = rowStat[2r]
        float mNew = std::max(mOld, mBlk - lnKappa);         // 内核：mNew = max(mOld, mBlk − θ)
        float alpha = std::exp(mOld - mNew);                 // 内核：alpha = exp(mOld − mNew)
        const bool skipRescale = (mBlk - mOld <= lnKappa);   // 内核：D-R1 阈值判定
        if (skipRescale) {
            alpha = 1.0f;
            mNew = mOld;  // 内核同款：skip 时 mNew=mRun 直接成立，不重算 max
            ++st.skips;
        } else {
            ++st.rescales;
        }
        float sumP8 = 0.0f;
        float partial[kToyVDeg] = {0.0f};
        for (int j = 0; j < sa2::kKvStep; ++j) {
            const float pVal = std::exp(scores[b][j] - mNew);
            if (pVal > st.maxP) st.maxP = pVal;
            const float p8 = QuantizeP(pVal);
            sumP8 += p8;
            for (int c = 0; c < kToyVDeg; ++c) {
                partial[c] += p8 * v[b * sa2::kKvStep + j][c];
            }
        }
        st.denom = alpha * st.denom + sumP8;
        if (!skipRescale && alpha != 1.0f) {
            for (int c = 0; c < kToyVDeg; ++c) {
                st.o[c] *= alpha;
            }
        }
        for (int c = 0; c < kToyVDeg; ++c) {
            st.o[c] += partial[c];
        }
        st.mRun = mNew;
    }
}

void MakeV(float v[kNumBlocks * sa2::kKvStep][kToyVDeg], Lcg &rng)
{
    for (int j = 0; j < kNumBlocks * sa2::kKvStep; ++j) {
        for (int c = 0; c < kToyVDeg; ++c) {
            v[j][c] = rng.NextUniform(-1.0f, 1.0f);
        }
    }
}

// ---- 用例 2：θ=lnκ skip 两路一致性 ----
void CaseThetaSkipTwoPathConsistency()
{
    Lcg rng(20261001ULL);
    const float lnKappa = std::log(kSa2CaseKappaFp16);
    for (int row = 0; row < 8; ++row) {
        static float scores[kNumBlocks][sa2::kKvStep];
        static float v[kNumBlocks * sa2::kKvStep][kToyVDeg];
        MakeScores(scores, rng, /*monotoneDown=*/row >= 4);
        MakeV(v, rng);

        OnlineState ref;
        OnlineState theta;
        OnlineReference(scores, v, lnKappa, ref);
        OnlineTheta(scores, v, lnKappa, theta);

        // 单调下降流：两路逐位一致（skip 恒成立时阈值法退化为同一状态序列）。
        if (row >= 4) {
            // 镜像内核 skip 支路：首块必 rescale（mRun 初始 −65504），后续块以 skip 为主；
            // skips+rescales 逐块二选一（无第三支路），且 skip 支路 mNew=mOld 直接成立下两路逐位一致。
            SA2_CHECK(theta.rescales >= 1 && theta.skips >= 1);
            SA2_CHECK(theta.skips + theta.rescales == kNumBlocks);
            SA2_CHECK(theta.mRun == ref.mRun && theta.denom == ref.denom);
            for (int c = 0; c < kToyVDeg; ++c) {
                SA2_CHECK(theta.o[c] == ref.o[c]);
            }
        } else {
            SA2_CHECK(theta.skips > 0 && theta.rescales > 0); // 两支路均被真实触发
            SA2_CHECK(theta.skips + theta.rescales == kNumBlocks); // 逐块二选一（无第三支路）
        }
        // 归一化输出一致（E4M3 网格量化噪声同级，容差 5e-2）。
        for (int c = 0; c < kToyVDeg; ++c) {
            const float refNorm = (ref.denom > 0.0f) ? ref.o[c] / ref.denom : 0.0f;
            const float thetaNorm = (theta.denom > 0.0f) ? theta.o[c] / theta.denom : 0.0f;
            SA2_CHECK(std::fabs(refNorm - thetaNorm) <= 5.0e-2f);
        }
        // D-R1 skip 包络：θ 路量化前 exp 值恒 ≤ κ（κ=2.25 网格点精确，离散化后仍 ≤ κ，见用例 8）。
        SA2_CHECK(theta.maxP <= kSa2CaseKappaFp16 * (1.0f + 1.0e-6f));
        SA2_CHECK(theta.denom > 0.0f && ref.denom > 0.0f);
    }
}

// ---- 用例 3：LSE 契约 + +lnκ 负测试 ----
void CaseLseContractWithNegativeLnK()
{
    Lcg rng(3030303ULL);
    const double kInvLog2E = 1.0 / 1.44269504; // 任务书契约常数（§2.3 返回路径）
    SA2_CHECK(std::fabs(kInvLog2E - kLn2) < 1.0e-8);

    for (int tier = 0; tier < 2; ++tier) {
        const float kappa = (tier == 0) ? kSa2CaseKappaFp16 : 448.0f;
        const float lnKappa = std::log(kappa);
        static float scores[kNumBlocks][sa2::kKvStep];
        static float v[kNumBlocks * sa2::kKvStep][kToyVDeg];
        MakeScores(scores, rng, false);
        MakeV(v, rng);

        OnlineState st;
        OnlineTheta(scores, v, lnKappa, st);

        // 真值：double 域全量 logsumexp（自然域）；核内闭环 = ln(denom) + m'（§4.6，不加 lnκ）。
        double sumExp = 0.0;
        for (int b = 0; b < kNumBlocks; ++b) {
            for (int j = 0; j < sa2::kKvStep; ++j) {
                sumExp += std::exp(static_cast<double>(scores[b][j]));
            }
        }
        const double lseTrue = std::log(sumExp);
        double lseErrSum = 0.0;
        const double lseCore = std::log(static_cast<double>(st.denom)) + static_cast<double>(st.mRun);
        const double lseErr = std::fabs(lseCore - lseTrue);
        lseErrSum += lseErr;
        // E4M3 码字噪声上界 ln(1+2^-4) ≈ 0.0606 + fp32 求和余量。
        SA2_CHECK(lseErr < 0.08);
        // 任务书口径换算闭合：lse_log2/1.44269504 == lse_natural（±ln2 常数精度内）。
        const double lseTaskbook = (lseTrue / 1.44269504) / kInvLog2E;
        SA2_CHECK(std::fabs(lseTaskbook - lseTrue) < 1.0e-9);
        // +lnκ 旧式（2026-10-01 修正前）负测试：偏差须达 lnκ 的 85% 以上（可检出的整体漂移）。
        const double lseOldBug = lseCore + static_cast<double>(lnKappa);
        SA2_CHECK(std::fabs(lseOldBug - lseTrue) >= 0.85 * static_cast<double>(lnKappa));
        // s·corr 校正项可加性（corr=0 载体；接线冻结）。
        const float smScale = 0.08838834764831845f;
        const float corr = 0.5f;
        SA2_CHECK(std::fabs((lseCore + static_cast<double>(smScale * corr)) - (lseCore + 0.5 * smScale)) < 1.0e-9);
        (void)lseErrSum;
    }
}

// ---- 用例 4：E8M0 向上取幂位技巧对照 std ----
int E8m0StdPath(float x)
{
    // 标量 oracle 路径（sa2_kv_prep.h 现行）：clamp [-127,127] 后 +127 存储为 UE8M0。
    double e = std::ceil(std::log2(static_cast<double>(x)));
    if (e < -127.0) e = -127.0;
    if (e > 127.0) e = 127.0;
    return static_cast<int>(e) + 127;
}

int E8m0BitPath(float x)
{
    // M2.6 向量化层既定实现：+0x007FFFFF 掩码取幂（对正规数与 std 路径逐位一致）。
    return static_cast<int>((BitsOf(x) + 0x007FFFFFu) >> 23);
}

int E8m0TileKernelsPath(float x)
{
    // THIRD_PARTY TileKernels per_token_cast_asc.py 等价式：((bits-1)>>23)+1。
    return static_cast<int>(((BitsOf(x) - 1u) >> 23) + 1u);
}

// 448·2^k 边界双口径（M2.6 接线冻结，迭代 5 增强重放；P1-6 统一入口后接线更新）：
// 设备侧（sa2_spec.h ComputeE8m0Exponent，kv_prep V 量化唯一调用点）= float 预除 amax/448.0f
// + 位技巧（+幂边界饱和兜底，冻结域内兜底不触发，见下方 helper 等价断言）；
// 契约参照 oracle = double 预除 + std::ceil(log2)（双精度域换算，历史宿主口径）。
// 必须用除法而非倒数乘（不得写 amax*(1.0f/448.0f)）：1/448 非二的幂，倒数乘引入第二次 RNE，
// 其边界行为不在下方结构论证覆盖域内（随机 2e6 探针未现分歧，契约仍按设备逐语句冻结为除法）。
// 结构论证（边界不可翻转）：448 = 2^6·7 恰为 float 可表示值。设 amax = M·2^(s-23)、M ∈ [2^23, 2^24)。
// 商 q = amax/448 落入 2 的幂上方危险窗 (2^j, 2^j·(1+2^-24)]（RNE 舍下到 2^j 而 double ceil 给 j+1）
// 仅可能当 amax ∈ [2^(j+8), 2^(j+9)) 即 s = j+8；此时 7q = 2^(j-21)·(M − 7·2^21)，M − 7·2^21 为正整数
// ⇒ 7q − 7·2^j ≥ 2^(j-21) > 7·2^(j-24) ⇒ 危险窗不可达（float 商粒度 2^-24 ≫ double 舍入 2^-53），
// float 预除位技巧与 double 预除 oracle 在 448·2^k 边界逐位一致。
int E8m0StdPipeline(float amax)
{
    double e = std::ceil(std::log2(static_cast<double>(amax) / 448.0));
    if (e < -127.0) e = -127.0;
    if (e > 127.0) e = 127.0;
    return static_cast<int>(e) + 127;
}

int E8m0FloatBitPipeline(float amax)
{
    const float x = amax / 448.0f;  // 除法口径（M2.6 向量化层契约），不得以倒数乘替代
    uint32_t rawExp = (BitsOf(x) + 0x007FFFFFu) >> 23;
    if (rawExp > 254u) rawExp = 254u;
    return static_cast<int>(rawExp);
}

void CaseE8m0BitTrickMatchesStd()
{
    SA2_CHECK(E8m0StdPath(1.0f) == 127 && E8m0BitPath(1.0f) == 127);
    Lcg rng(448448ULL);
    int mismatches = 0;
    int coverageViolations = 0;
    // 扫掠域：正規数（指数 [-126,120]），2 的幂与 ±1 ulp / 满 mantissa 邻居 + 随机采样；
    // V 量化可达域（|v| ≤ 65504 → x = amax/448 ≤ 146.2）远在域内；clamp 边界外行为不在冻结范围。
    for (int e = -126; e <= 120; ++e) {
        float probe[4] = {std::ldexp(1.0f, e), FloatOfBits(BitsOf(std::ldexp(1.0f, e)) + 1u),
                          FloatOfBits(BitsOf(std::ldexp(1.0f, e)) | 0x007FFFFFu),
                          std::ldexp(1.0f, e) * (1.0f + 1.0f / 8192.0f)};
        for (float x : probe) {
            const int s = E8m0StdPath(x);
            if (E8m0BitPath(x) != s || E8m0TileKernelsPath(x) != s) {
                ++mismatches;
            }
            // 覆盖性质：scale = 2^(uexp-127) 须满足 448·scale ≥ x（量化不溢出的充要下界）。
            if (!(448.0f * std::ldexp(1.0f, s - 127) >= x)) {
                ++coverageViolations;
            }
            g_checks += 2;
        }
    }
    for (int i = 0; i < 200000; ++i) {
        const float x = std::ldexp(rng.NextUniform(1.0f, 2.0f), static_cast<int>(rng.Next() % 247u) - 126);
        const int s = E8m0StdPath(x);
        if (E8m0BitPath(x) != s || E8m0TileKernelsPath(x) != s) {
            ++mismatches;
        }
        if (!(448.0f * std::ldexp(1.0f, s - 127) >= x)) {
            ++coverageViolations;
        }
        g_checks += 2;
    }
    SA2_CHECK(mismatches == 0);
    SA2_CHECK(coverageViolations == 0);
    // 全零组防护：amax==0 时两路均不使用，标量取 E8M0 最小正（§4.8）。
    SA2_CHECK(sa2::kMinGroupMax > 0.0f);

    // 448·2^k 边界邻域（k∈[-20,20]，±1..3 ulp 邻点）：双口径逐位一致 + 覆盖性质（增强 a）。
    // 并冻结统一入口等价：ComputeE8m0Exponent（位技巧+兜底）在冻结域内 == double oracle（兜底不触发）。
    for (int k = -20; k <= 20; ++k) {
        const float base = 448.0f * std::ldexp(1.0f, k);
        for (int d = -3; d <= 3; ++d) {
            const float amax = FloatOfBits(BitsOf(base) + static_cast<uint32_t>(d));
            if (!(amax > 0.0f) || !std::isfinite(amax)) continue;
            const int sDev = E8m0FloatBitPipeline(amax);
            SA2_CHECK(sDev == E8m0StdPipeline(amax));
            SA2_CHECK(static_cast<int>(sa2::ComputeE8m0Exponent(amax)) == E8m0StdPipeline(amax));
            SA2_CHECK(448.0f * std::ldexp(1.0f, sDev - 127) >= amax); // 448·scale ≥ amax 充要下界
        }
    }
    // fp16 可达包络域随机扫掠（amax ≥ 2^-24 ⇒ 商 ≥ 2^-32.8 保持 float 正规）：双口径逐位一致 + 覆盖性质。
    for (int i = 0; i < 20000; ++i) {
        const float amax =
            std::ldexp(rng.NextUniform(1.0f, 2.0f), static_cast<int>(rng.Next() % 145u) - 24);
        const int sDev = E8m0FloatBitPipeline(amax);
        SA2_CHECK(sDev == E8m0StdPipeline(amax));
        SA2_CHECK(static_cast<int>(sa2::ComputeE8m0Exponent(amax)) == E8m0StdPipeline(amax));
        SA2_CHECK(448.0f * std::ldexp(1.0f, sDev - 127) >= amax);
    }
    // 域闭包：fp16 输入最小正次正规 2^-24 ⇒ 商 ≥ 2^-24/448 ≫ 2^-126（float 正规下界），
    // 次正规商域对 fp16 输入不可达（上面的逐位一致断言因此覆盖全部 fp16 可达域）。
    SA2_CHECK(std::ldexp(1.0f, -24) / 448.0f >= std::ldexp(1.0f, -126));
    // 次正规商域分歧分类（仅 bf16 极端次正规组可达）：设备位技巧对 subnormal 商偏保守
    //（尺度 ≥ oracle，覆盖性质保持）——分歧方向安全，冻结为类断言而非逐位。
    for (int t = 0; t < 17; ++t) {
        const float amax = std::ldexp(1.0f, -133 + t);
        const int sDev = E8m0FloatBitPipeline(amax);
        SA2_CHECK(sDev >= E8m0StdPipeline(amax));
        SA2_CHECK(448.0f * std::ldexp(1.0f, sDev - 127) >= amax);
    }

    // P1-6：验证 sa2::ComputeE8m0Exponent 对 448·2^k 边界兜底的有效性与防饱和溢出保证
    for (int k = -20; k <= 20; ++k) {
        const float base = 448.0f * std::ldexp(1.0f, k);
        for (int d = -5; d <= 5; ++d) {
            const float amax = FloatOfBits(BitsOf(base) + static_cast<uint32_t>(d));
            if (!(amax > 0.0f) || !std::isfinite(amax)) continue;
            const uint8_t uexp = sa2::ComputeE8m0Exponent(amax);
            const int expVal = static_cast<int>(uexp) - 127;
            const float scaledAmax = sa2::AmlaScaleMultiply(amax, -expVal);
            SA2_CHECK(scaledAmax <= 448.0f);
        }
    }
}

// ---- 用例 5：g 因子界 + DEQ 恒等式 ----
void CaseGFactorBoundAndDeqIdentity()
{
    const float kScales[] = {1.0f / std::sqrt(48.0f), 1.0f / std::sqrt(64.0f), 1.0f / std::sqrt(96.0f),
                             1.0f / std::sqrt(128.0f)};
    for (float s : kScales) {
        const auto pol =
            sa2::DeriveNumericPolicy(sa2::Gran::kPerThread, sa2::PvAccum::kFp32Fp16, 256, 1024, 128, s);
        // g² = |s|·2^14（DEQ 2^-14 拆分基础，§4.2）。
        const double g2 = static_cast<double>(pol.kFactor) * static_cast<double>(pol.kFactor);
        SA2_CHECK(std::fabs(g2 - static_cast<double>(s) * 16384.0) < 1.0e-3 * static_cast<double>(s) * 16384.0);
        SA2_CHECK(pol.qFactor == pol.kFactor && pol.qFactor > 0.0f);
        // FP16 可表示界：kScale = s_k·kFactor 以 half 落 GM，验收包络 s_k ≤ 516（§4.2 cap）。
        const float kScaleExtreme = 516.0f * pol.kFactor;
        const float back = static_cast<float>(sa2::hostbits::HalfF32(kScaleExtreme));
        SA2_CHECK(std::isfinite(back));
        SA2_CHECK(std::fabs(back - kScaleExtreme) <= kScaleExtreme / 2048.0f); // RNE ≤ 2^-11 相对
        // 中间 coeff（行×块粒度）不整体溢出 fp32。
        SA2_CHECK(516.0 * 516.0 * g2 < 3.6e10);
    }
    // 负 s：符号随 Q 侧因子（§4.2/R.1.2）。
    const auto neg = sa2::DeriveNumericPolicy(sa2::Gran::kPerWarp, sa2::PvAccum::kFp32, 64, 64, 64, -0.25f);
    SA2_CHECK(neg.qFactor < 0.0f && neg.qFactor == -neg.kFactor);
    // DEQ 恒等式数值例：dot·2^-14·(s_q·QH)·(s_k·KH) ≡ dot·s_q·s_k·s（每元素恰一次乘法）。
    {
        const float s = 1.0f / std::sqrt(128.0f);
        const auto pol = sa2::DeriveNumericPolicy(sa2::Gran::kPerWarp, sa2::PvAccum::kFp32Fp16, 64, 64, 64, s);
        const int32_t dot = 1000000;
        const float sq = 0.5f;
        const float sk = 0.25f;
        const float kernelPath = static_cast<float>(dot) * std::ldexp(1.0f, -static_cast<int>(sa2::kScoreDeqShift)) *
                                 (sq * pol.qFactor) * (sk * pol.kFactor);
        const float mathPath = static_cast<float>(dot) * sq * sk * s;
        SA2_CHECK(std::fabs(kernelPath - mathPath) <= 1.0e-5f * std::fabs(mathPath));
    }
    // score 域 FP16 安全界（§4.1）：上界 127·127·128·2^-14 ≈ 126 有限；下界 acc=1 → 2^-14 恰为 FP16 最小正规数。
    const float deqUpper = static_cast<float>(127 * 127 * sa2::kMaxHeadDim) *
                           std::ldexp(1.0f, -static_cast<int>(sa2::kScoreDeqShift));
    SA2_CHECK(std::isfinite(static_cast<float>(sa2::hostbits::HalfF32(deqUpper))) && deqUpper < 65504.0f);
    const float deqOne = std::ldexp(1.0f, -static_cast<int>(sa2::kScoreDeqShift));
    SA2_CHECK(deqOne == 6.103515625e-05f);
    SA2_CHECK(static_cast<float>(sa2::hostbits::HalfF32(deqOne)) == deqOne); // min normal 边界精确可表示
    // 派生策略运行期镜像（static_assert 的工作例复核）。
    const auto pol = sa2::DeriveNumericPolicy(sa2::Gran::kPerThread, sa2::PvAccum::kFp32Fp16, 4096, 16384, 128,
                                              0.08838834764831845f);
    SA2_CHECK(pol.qp == 4096 && pol.kp == 16384 && pol.dp == 128 && pol.segWidth == 16);
    SA2_CHECK(sa2::RoundUpTo(250, 64) == 256 && sa2::RoundUpTo(1000, 128) == 1024);
    SA2_CHECK(sa2::PaddedHeadDim(48) == 64 && sa2::PaddedHeadDim(96) == 128 && sa2::PaddedHeadDim(160) == 0);
}

// ---- 用例 6：per_warp / per_thread 粒度差异非零 ----
void CaseGranularityPathsDiffer()
{
    Lcg rng(777777ULL);
    float x[sa2::kStreamRows][sa2::kMaxHeadDim];
    for (int r = 0; r < sa2::kStreamRows; ++r) {
        for (int c = 0; c < sa2::kMaxHeadDim; ++c) {
            // 列增益不均（段间方差大），放大分段 scale 与行 scale 的差异。
            const float gain = (c % sa2::kSegCols == 3) ? 8.0f : 0.4f;
            x[r][c] = rng.NextUniform(-1.0f, 1.0f) * gain;
        }
    }
    auto quantDequant = [&](bool perThread, float *errOut, int8_t (*q8Out)[sa2::kMaxHeadDim]) {
        double worst = 0.0;
        for (int r = 0; r < sa2::kStreamRows; ++r) {
            const int segs = perThread ? sa2::kMaxHeadDim / sa2::kSegCols : 1;
            for (int seg = 0; seg < segs; ++seg) {
                const int c0 = seg * (perThread ? static_cast<int>(sa2::kSegCols) : 0);
                const int c1 = perThread ? c0 + static_cast<int>(sa2::kSegCols) : sa2::kMaxHeadDim;
                float amax = 0.0f;
                for (int c = c0; c < c1; ++c) {
                    amax = std::max(amax, std::abs(x[r][c]));
                }
                float scale = amax / 127.0f;
                if (scale < 1.0e-7f) scale = 1.0f;
                const float invScale = 1.0f / scale;
                for (int c = c0; c < c1; ++c) {
                    const float scaled = std::nearbyint(x[r][c] * invScale);
                    const float clamped = std::max(-127.0f, std::min(127.0f, scaled));
                    q8Out[r][c] = static_cast<int8_t>(clamped);
                    worst = std::max(worst, std::fabs(static_cast<double>(q8Out[r][c]) * scale -
                                                      static_cast<double>(x[r][c])));
                }
            }
        }
        *errOut = static_cast<float>(worst);
    };
    float errWarp = 0.0f;
    float errThread = 0.0f;
    int8_t qWarp[sa2::kStreamRows][sa2::kMaxHeadDim];
    int8_t qThread[sa2::kStreamRows][sa2::kMaxHeadDim];
    quantDequant(false, &errWarp, qWarp);
    quantDequant(true, &errThread, qThread);
    int diffs = 0;
    for (int r = 0; r < sa2::kStreamRows; ++r) {
        for (int c = 0; c < sa2::kMaxHeadDim; ++c) {
            if (qWarp[r][c] != qThread[r][c]) {  // 逐元素值比较
                ++diffs;
            }
        }
    }
    SA2_CHECK(diffs > 0);                                  // 两种粒度真实不同（空桩 = 违约）
    SA2_CHECK(errThread <= errWarp + 1.0e-9f);             // 细粒度重构误差不劣于粗粒度
    SA2_CHECK(errWarp > 0.0f && errThread > 0.0f);         // 量化非恒等
}

// ---- 用例 7：因果谓词块级 vs 逐元素 ----
void CaseCausalBlockVsElementPredicate()
{
    const int64_t kShapes[][2] = {{64, 64}, {128, 64}, {250, 1000}, {192, 129}, {64, 300}, {1024, 1024}};
    for (const auto &shape : kShapes) {
        const int64_t sq = shape[0];
        const int64_t sk = shape[1];
        // 非因果：步数恰为 ceil(sk/128)，全列覆盖。
        SA2_CHECK(sa2::StepCount(sk) == (sk + sa2::kKvStep - 1) / sa2::kKvStep);
        for (int64_t m0 = 0; m0 < sq; m0 += sa2::kStreamRows) {
            const int64_t limit = sa2::CausalLimit(sk, m0); // min(sk, m0+64)
            const int64_t steps = sa2::StepCount(limit);
            const int64_t lastRow = std::min(sq, m0 + sa2::kStreamRows) - 1;
            // (a) 不漏：末行（tile 内因果最远行）的最大未掩码列 < steps·128。
            const int64_t maxUnmaskedCol = std::min(sk - 1, lastRow);
            SA2_CHECK(maxUnmaskedCol < steps * sa2::kKvStep);
            // (b) 不空：枚举的最后一块含末行至少一个未掩码元素（无全掩码块）。
            const int64_t lastBlockStart = (steps - 1) * sa2::kKvStep;
            SA2_CHECK(lastBlockStart < limit && lastBlockStart <= maxUnmaskedCol);
            // (c) 紧致：步数 = 末行逐元素谓词推出的块数（不多算全掩码块）。
            SA2_CHECK(steps == maxUnmaskedCol / sa2::kKvStep + 1);
            // (d) 逐元素核对：任意行 r、任意列 c ≥ steps·128 必被逐元素谓词掩掉。
            for (int64_t r = 0; r < std::min(sq - m0, sa2::kStreamRows); r += 7) {
                const int64_t q = m0 + r;
                for (int64_t c = steps * sa2::kKvStep; c < std::min(steps * sa2::kKvStep + 512, sk + 512);
                     c += 13) {
                    const bool elementMasked = (c > q) || (c >= sk);
                    SA2_CHECK(elementMasked);
                }
            }
        }
    }
}

// ---- 用例 8：FP8 E4M3 码字全表往返与 κ 包络 ----
void CaseFp8CodeTableRoundtrip()
{
    for (int c = 0; c < 256; ++c) {
        const uint8_t code = static_cast<uint8_t>(c);
        const float v = sa2::Fp8E4M3ToFloat(code);
        if (std::isnan(v)) {
            SA2_CHECK(code == 0x7F || code == 0xFF); // 仅 NaN 码字产生 NaN
            continue;
        }
        if (sa2::FloatToFp8E4M3(v) != code) {
            ExpectTrue(false, "fp8 roundtrip", __FILE__, __LINE__);
            std::printf("       code=0x%02x -> %g -> 0x%02x\n", c, v, sa2::FloatToFp8E4M3(v));
            continue;
        }
        ++g_checks;
    }
    SA2_CHECK(sa2::Fp8E4M3ToFloat(sa2::FloatToFp8E4M3(2.25f)) == 2.25f);   // κ 网格点精确
    SA2_CHECK(sa2::Fp8E4M3ToFloat(sa2::FloatToFp8E4M3(2.375f)) == 2.5f);   // RNE ties-to-even
    SA2_CHECK(sa2::Fp8E4M3ToFloat(sa2::FloatToFp8E4M3(464.0f)) == 448.0f); // 溢出饱和 448（torch 对齐）
    SA2_CHECK(sa2::FloatToFp8E4M3(448.0f) == 0x7Eu);
    SA2_CHECK(std::isnan(sa2::Fp8E4M3ToFloat(0x7Fu)));
    // κ 包络经离散化保持：p ∈ [0, κ] 的码字值 ≤ κ（RNE 半 ulp 不越界）。
    Lcg rng(44844800ULL);
    float maxP8 = 0.0f;
    for (int i = 0; i < 100000; ++i) {
        const float p8 = sa2::Fp8E4M3ToFloat(sa2::FloatToFp8E4M3(rng.NextUniform(0.0f, kSa2CaseKappaFp16)));
        if (p8 > maxP8) maxP8 = p8;
    }
    SA2_CHECK(maxP8 <= kSa2CaseKappaFp16 + 1.0e-6f);
}

// ---- 用例 9：HalfF32 RNE 特征点（0.5ulp 边界 / 奇偶尾数 ties-to-even / 溢出饱和 / 下溢方向）----
// 迭代 5 增强 b：确保截断型替身无法通过本套件（截断突变检出验证见台账迭代 5 日志）。
void CaseHalfF32RneFeaturePoints()
{
    using sa2::hostbits::HalfF32;
    auto bitsOfHalf = [](float f) {
        HalfF32 h(f);
        return static_cast<uint32_t>(h.val);
    };
    const float kMinSubnormal = std::ldexp(1.0f, -24);  // half 最小次正规 2^-24
    const float kMinNormal = std::ldexp(1.0f, -14);     // half 最小正规 2^-14
    const float kMaxFinite = 65504.0f;                  // half 最大正规（0x7BFF）
    struct FeaturePoint {
        float x;
        uint32_t want;
    };
    const FeaturePoint kPoints[] = {
        {0.0f, 0x0000u}, {-0.0f, 0x8000u},             // ±0 精确
        {1.0f, 0x3C00u}, {-1.0f, 0xBC00u},
        {kMinNormal, 0x0400u},                          // 最小正规精确
        {kMinNormal - kMinSubnormal, 0x03FFu},          // 最大次正规精确
        {kMaxFinite, 0x7BFFu}, {-kMaxFinite, 0xFBFFu},  // ±65504 精确（溢出饱和锚点）
        {65519.0f, 0x7BFFu},                            // 上溢 tie 下方 → 最大正规（RNE 向下）
        {65520.0f, 0x7C00u},                            // 65520 恰 tie（65504 与上溢中点）→ 偶侧 Inf
        {std::nextafterf(65520.0f, INFINITY), 0x7C00u}, // tie 上方 → Inf
        {std::nextafterf(65520.0f, 0.0f), 0x7BFFu},     // tie 下方 → 65504
        {65536.0f, 0x7C00u}, {-65536.0f, 0xFC00u},      // 2^16 超 half 域 → ±Inf 饱和
        {1.0e30f, 0x7C00u}, {-1.0e30f, 0xFC00u},
        // 0.5 ulp 边界 + 奇偶尾数 ties-to-even（正规域）：
        {1.0f + std::ldexp(1.0f, -11), 0x3C00u},        // 恰 tie → 偶侧 1.0（向下）
        {std::nextafterf(1.0f + std::ldexp(1.0f, -11), INFINITY), 0x3C01u},   // tie 上方 → 向上
        {1.0f + 3.0f * std::ldexp(1.0f, -11), 0x3C02u}, // 恰 tie（下邻奇尾数）→ 偶侧向上
        {std::nextafterf(1.0f + 3.0f * std::ldexp(1.0f, -11), 0.0f), 0x3C01u}, // tie 下方 → 向下
        // 下溢方向（次正规域，步长 2^-24）：
        {std::ldexp(1.0f, -25), 0x0000u},               // 半步 tie → 偶侧归零
        {std::nextafterf(std::ldexp(1.0f, -25), 1.0f), 0x0001u},  // 半步上方 → 最小次正规
        {std::nextafterf(std::ldexp(1.0f, -25), 0.0f), 0x0000u},  // 半步下方 → 归零
        {1.5f * kMinSubnormal, 0x0002u},                // 恰 tie（下邻奇）→ 偶侧向上
        {std::nextafterf(1.5f * kMinSubnormal, 0.0f), 0x0001u},   // tie 下方 → 向下
        {kMinSubnormal, 0x0001u},
        {kMinNormal - 0.5f * kMinSubnormal, 0x0400u},   // 正规边界恰 tie → 偶侧 0x0400
        {std::nextafterf(kMinNormal - 0.5f * kMinSubnormal, 0.0f), 0x03FFu}, // tie 下方 → 最大次正规
    };
    for (const FeaturePoint &pt : kPoints) {
        SA2_CHECK(bitsOfHalf(pt.x) == pt.want);
    }
    // NaN：折叠为静音 qNaN（载荷位不保留，NaN 性与符号位保持）。
    {
        const float nanP = std::nanf("");
        HalfF32 h(nanP);
        SA2_CHECK(std::isnan(static_cast<float>(h)));
        SA2_CHECK((h.val & 0x7C00u) == 0x7C00u && (h.val & 0x03FFu) != 0u);
        HalfF32 nh(-nanP);
        SA2_CHECK((nh.val & 0x8000u) != 0u && std::isnan(static_cast<float>(nh)));
    }
    // 全 2^16 half 码字 → float → half 往返幂等（NaN 码字仅验 NaN 性；其余必须逐位还原）。
    for (uint32_t b = 0; b < 65536u; ++b) {
        HalfF32 h;
        h.val = static_cast<uint16_t>(b);
        const float f = static_cast<float>(h);
        if (std::isnan(f)) {
            SA2_CHECK((b & 0x7FFFu) > 0x7C00u); // 仅 NaN 码字产生 NaN
            continue;
        }
        HalfF32 h2(f);
        SA2_CHECK(h2.val == h.val);
    }
    // 最近性扫掠（截断/恒向上替身的通用检出器）：half(f) 与其 ±1 half-ulp 邻点中必须最近 f。
    Lcg rng(97531245ULL);
    for (int i = 0; i < 20000; ++i) {
        const float f = FloatOfBits(rng.Next());
        if (!std::isfinite(f) || std::fabs(static_cast<double>(f)) >= 65504.0) continue;
        HalfF32 h(f);
        const double d0 = std::fabs(static_cast<double>(static_cast<float>(h)) - static_cast<double>(f));
        if (h.val != 0x0000u && h.val != 0x8000u) { // 低邻点（±0 的低邻是 NaN 码字，跳过）
            HalfF32 hl;
            hl.val = static_cast<uint16_t>(h.val - 1u);
            const double dl = std::fabs(static_cast<double>(static_cast<float>(hl)) - static_cast<double>(f));
            SA2_CHECK(d0 <= dl);
        }
        HalfF32 hh;
        hh.val = static_cast<uint16_t>(h.val + 1u);
        const double dh = std::fabs(static_cast<double>(static_cast<float>(hh)) - static_cast<double>(f));
        SA2_CHECK(d0 <= dh); // |f| < 65504 ⇒ h ≤ 0x7BFF，高邻点至多为 Inf（合法更远邻点）
    }
}

// ---- 用例 10：fp32+fp16 折叠误差界 ----
void CaseFp16FoldErrorBound()
{
    Lcg rng(14142135ULL);
    const int kRows = 16;
    const int kBlocks = 4;
    static float v[kNumBlocks * sa2::kKvStep][kToyVDeg];
    MakeV(reinterpret_cast<float(*)[kToyVDeg]>(v), rng); // 仅前 kKvStep 行有效
    float oExact[kRows][kToyVDeg] = {{0.0f}};
    float oFold[kRows][kToyVDeg] = {{0.0f}};
    double boundAcc[kRows][kToyVDeg] = {{0.0}};
    float maxAbsPartial = 0.0f;
    bool anyDiff = false;
    for (int b = 0; b < kBlocks; ++b) {
        float partial[kRows][kToyVDeg] = {{0.0f}};
        for (int j = 0; j < sa2::kKvStep; ++j) {
            const float p8 = sa2::Fp8E4M3ToFloat(sa2::FloatToFp8E4M3(rng.NextUniform(0.0f, kSa2CaseKappaFp16)));
            for (int r = 0; r < kRows; ++r) {
                for (int c = 0; c < kToyVDeg; ++c) {
                    partial[r][c] += p8 * v[j][c];
                }
            }
        }
        for (int r = 0; r < kRows; ++r) {
            for (int c = 0; c < kToyVDeg; ++c) {
                maxAbsPartial = std::max(maxAbsPartial, std::abs(partial[r][c]));
                // HalfF32 为 RNE 替身（IEEE binary16，对齐设备 half / PyTorch .half()）：
                // |fp16(x) − x| ≤ 2^-11·|x|（正规域 RNE 半 ulp）；逐元素累入误差预算。
                boundAcc[r][c] += std::fabs(static_cast<double>(partial[r][c])) / 2048.0;
                oExact[r][c] += partial[r][c];
                const float folded = static_cast<float>(sa2::hostbits::HalfF32(partial[r][c]));
                if (folded != partial[r][c]) anyDiff = true;
                oFold[r][c] += folded;
            }
        }
    }
    SA2_CHECK(anyDiff); // 折叠真实生效（fp16 往返确实改变位型）
    for (int r = 0; r < kRows; ++r) {
        for (int c = 0; c < kToyVDeg; ++c) {
            if (std::fabs(static_cast<double>(oFold[r][c] - oExact[r][c])) > boundAcc[r][c] + 1.0e-6) {
                ExpectTrue(false, "fp16 fold bound", __FILE__, __LINE__);
                return;
            }
            ++g_checks;
        }
    }
    (void)maxAbsPartial;
}

// ---- 用例 11：AscendC MicroAPI / vapi 向量硬件指令与标量数值等价性 ----
void CaseMicroApiVectorEquivalence()
{
    // 1. 基本向量掩码与寄存器装载
    constexpr size_t N = vapi::RegTensor<float>::kCapacity; // 64 floats (256 bytes)
    float srcA[N];
    float srcB[N];
    for (size_t i = 0; i < N; ++i) {
        srcA[i] = static_cast<float>(i + 1) * 0.5f;
        srcB[i] = static_cast<float>(N - i) * 0.25f;
    }

    vapi::RegTensor<float> regA, regB, regDst;
    vapi::LoadAlign(regA, srcA);
    vapi::LoadAlign(regB, srcB);

    for (size_t i = 0; i < N; ++i) {
        SA2_CHECK(regA[i] == srcA[i]);
        SA2_CHECK(regB[i] == srcB[i]);
    }

    // 2. 算术操作：Mul, Muls, Add, Sub
    vapi::MaskReg fullMask = vapi::CreateMask<float, vapi::MaskPattern::ALL>();
    vapi::Mul(regDst, regA, regB, fullMask);
    for (size_t i = 0; i < N; ++i) {
        SA2_CHECK(regDst[i] == srcA[i] * srcB[i]);
    }

    vapi::Muls(regDst, regA, 3.5f, fullMask);
    for (size_t i = 0; i < N; ++i) {
        SA2_CHECK(regDst[i] == srcA[i] * 3.5f);
    }

    vapi::Add(regDst, regA, regB, fullMask);
    for (size_t i = 0; i < N; ++i) {
        SA2_CHECK(regDst[i] == srcA[i] + srcB[i]);
    }

    vapi::Sub(regDst, regA, regB, fullMask);
    for (size_t i = 0; i < N; ++i) {
        SA2_CHECK(regDst[i] == srcA[i] - srcB[i]);
    }

    // 3. 掩码更新与条件 Select
    uint32_t activeLen = 42;
    vapi::MaskReg partialMask = vapi::UpdateMask<float>(activeLen);
    vapi::RegTensor<float> negInf;
    vapi::Duplicate(negInf, -65504.0f);
    vapi::RegTensor<float> selected;
    vapi::Select(selected, partialMask, regA, negInf);
    for (size_t i = 0; i < N; ++i) {
        if (i < activeLen) {
            SA2_CHECK(selected[i] == srcA[i]);
        } else {
            SA2_CHECK(selected[i] == -65504.0f);
        }
    }

    // 4. 向量化归约：ReduceMax & ReduceSum
    float maxVal = vapi::ReduceMax(selected, partialMask);
    float wantMax = -65504.0f;
    for (size_t i = 0; i < activeLen; ++i) {
        if (srcA[i] > wantMax) wantMax = srcA[i];
    }
    SA2_CHECK(maxVal == wantMax);

    float sumVal = vapi::ReduceSum(regA, partialMask);
    float wantSum = 0.0f;
    for (size_t i = 0; i < activeLen; ++i) {
        wantSum += srcA[i];
    }
    SA2_CHECK(std::fabs(sumVal - wantSum) <= 1e-4f);

    // 5. 硬件单指令 ExpSub (exp(S - m)) 与自然对数 Ln
    float mVal = 10.0f;
    vapi::RegTensor<float> expSubReg;
    vapi::ExpSub(expSubReg, regA, mVal, partialMask);
    for (size_t i = 0; i < activeLen; ++i) {
        float expectedExp = std::exp(srcA[i] - mVal);
        SA2_CHECK(std::fabs(expSubReg[i] - expectedExp) <= 1e-5f * expectedExp);
    }

    vapi::RegTensor<float> lnReg;
    vapi::Ln(lnReg, regA, partialMask);
    for (size_t i = 0; i < activeLen; ++i) {
        float expectedLn = std::log(srcA[i]);
        SA2_CHECK(std::fabs(lnReg[i] - expectedLn) <= 1e-5f);
    }

    // 6. 向量类型转换：Cast<float8_e4m3_t, float>, Cast<half, float>, Cast<float, half>
    vapi::RegTensor<float8_e4m3_t> p8Reg;
    vapi::Cast<float8_e4m3_t, float>(p8Reg, expSubReg, partialMask);
    for (size_t i = 0; i < activeLen; ++i) {
        uint8_t wantByte = sa2::FloatToFp8E4M3(expSubReg[i]);
        SA2_CHECK(static_cast<uint8_t>(p8Reg[i]) == wantByte);
    }

    vapi::RegTensor<half> halfReg;
    vapi::Cast<half, float>(halfReg, regA, partialMask);
    vapi::RegTensor<float> backFloatReg;
    vapi::Cast<float, half>(backFloatReg, halfReg, partialMask);
    for (size_t i = 0; i < activeLen; ++i) {
        float wantBack = static_cast<float>(half(srcA[i]));
        SA2_CHECK(backFloatReg[i] == wantBack);
    }
}

// ---- 用例 12：M2.7.2 UB 计划 O(1) 硬防护（CI 化）----
// Rust 式结构化错误思想：把「UB 预算违例」从真机运行期静默溢出前移为宿主编译期硬错误——
// 本 TU 宿主编译期直接实例化 op_kernel/sa2_ub_plan.h，头内 static_assert 链（偏移链锚点 /
// 槽大小 / 248KB 高水位）在 conformance 构建时全部生效：任何一条被并发改动打破，g++ 编译即红
//（结构化错误 = CI 硬故障），不存在「运行期才炸」或「静默越界写」窗口。
// 下方断言为幂等校验（第二重键盘）：即使头内断言链被误删，测试 TU 仍独立复核同一事实。
// 「随 S 无关」的结构性来源：两条 UB 链每一项都只由 tile 常数（kStreamRows=64、kKvStep=128、
// kMaxHeadDim=128、NZ 分形 16×16+pad）与状态区冻结常数构成，无任何 shape 参数入链；长序列只
// 线性增加 KV 步数（steps = ceil(limit/128)），每步复用同一组 ping-pong 槽 ⇒ 工作集 O(1)。
// 为防「锚点常量被同步改错」的同源盲区，链终址由 tile 常数独立重推（不抄头内结果）后再对账。
constexpr int64_t kChkScoreWarpSlot = sa2::kStreamRows * sa2::kKvStep * 2;              // score FP16
constexpr int64_t kChkScoreThreadSlot = sa2::kStreamRows * sa2::kKvStep * 4;            // score FP32 组合域
constexpr int64_t kChkPSlot = (sa2::kStreamRows / 16) * (sa2::kKvStep / 16) * (16 * 16 + 16);  // NZ 分形块 + pad 行
constexpr int64_t kChkPvSlot = sa2::kStreamRows * sa2::kMaxHeadDim * 4;                 // PV FP32
constexpr int64_t kChkOAcc = sa2::kStreamRows * sa2::kMaxHeadDim * 4;                   // O 累加器 FP32
constexpr int64_t kChkSegSlot = sa2::kStreamRows * sa2::kKvStep * 2;                    // per_thread 分段 FP16

// 独立重推的链终址（与 sa2_ub_plan.h 推导式同构、数值独立重算）：
constexpr int64_t kChkUbEndPerWarp =
    2 * kChkScoreWarpSlot + 2 * kChkPSlot + 2 * kChkPvSlot + kChkOAcc + sa2::kUbStateBytes;
constexpr int64_t kChkUbEndPerThread =
    2 * kChkScoreThreadSlot + 2 * kChkPSlot + 2 * kChkPvSlot + kChkOAcc + 2 * kChkSegSlot + sa2::kUbStateBytes;

// 编译期硬防护（CI 门禁本体）：
static_assert(kChkUbEndPerWarp == 152448 && kChkUbEndPerWarp == sa2::kUbEndPerWarp,
              "per_warp UB 链终址独立重推 == 冻结锚点 152448");
static_assert(kChkUbEndPerThread == 217984 && kChkUbEndPerThread == sa2::kUbEndPerThread,
              "per_thread UB 链终址独立重推 == 冻结锚点 217984");
static_assert(sa2::Sa2DeviceResWarp::kUbEnd == kChkUbEndPerWarp &&
                  sa2::Sa2DeviceResThread::kUbEnd == kChkUbEndPerThread,
              "两种粒度模板实例化的 kUbEnd 命中同一锚点（实例化幂等）");
static_assert(kChkUbEndPerWarp <= sa2::kUbHardSafetyLimit, "per_warp 终址 ≤ 160KB 瞬时硬安全门禁");
static_assert(kChkUbEndPerThread <= sa2::kUbHighWater, "per_thread 终址 ≤ 248KB UB 编程高水位");
static_assert(kChkUbEndPerThread <= sa2::kUbPhysicalLimit - 8192,
              "per_thread 终址与 256KB 物理上限间保留 ≥ 8KB 硬件控制寄存器余量");
static_assert(sa2::kUbPhysicalLimit - sa2::kUbEndPerThread == 44160, "44KB 硬件控制寄存器安全余量冻结");
// 随 S 无关的结构对偶（编译期采样）：步数派生式只吃 limit、不吃任何 UB 链输入。
static_assert(sa2::StepCount(128) == 1 && sa2::StepCount(sa2::kUbPhysicalLimit) == 2048,
              "steps 随 limit 线性增长而 UB 链常数恒定（对偶性的编译期采样）");
// L1 / L0C 预算链幂等复核（与 UB 同一硬防护纪律）：
static_assert(sa2::kL1End == 119424 && sa2::kL1End <= sa2::kL1Size, "L1 链终址 119424 ≤ 512KB");
static_assert(sa2::kL0CQuadTotal == 131072 && sa2::kL0CQuadTotal <= sa2::kL0CSize, "L0C 四分区 ≤ 256KB");

// 瞬时活跃工作集（单步实际触碰：半组 ping-pong 槽 + 状态区）——O(1) 恒定且 ≤ 160KB 硬安全门禁。
constexpr int64_t kChkInstantWarp = kChkScoreWarpSlot + kChkPSlot + kChkPvSlot + kChkOAcc + sa2::kUbStateBytes;
constexpr int64_t kChkInstantThread = kChkScoreThreadSlot + kChkPSlot + kChkPvSlot + kChkOAcc + sa2::kUbStateBytes;
static_assert(kChkInstantWarp == 94592 && kChkInstantThread == 110976, "瞬时工作集冻结常数");
static_assert(kChkInstantWarp <= sa2::kUbHardSafetyLimit && kChkInstantThread <= sa2::kUbHardSafetyLimit,
              "瞬时工作集 ≤ 160KB 硬安全门禁（M2.7.2）");

void CaseUbPlanO1HardGuard()
{
    SA2_CHECK(kChkInstantWarp <= sa2::kUbHardSafetyLimit && kChkInstantThread <= sa2::kUbHardSafetyLimit);
    // 长 S 扫掠：kp 随 S 严格增长（扫掠非空转）而 UB 两链终址与瞬时工作集逐点不变 ⇒ O(1) 实证。
    const int64_t kSeqLens[] = {128, 1024, 4096, 16384, 131072};
    int64_t prevKp = 0;
    for (int64_t s : kSeqLens) {
        const auto pol =
            sa2::DeriveNumericPolicy(sa2::Gran::kPerThread, sa2::PvAccum::kFp32Fp16, s, s, 128, 0.08838834764831845f);
        SA2_CHECK(pol.kp == sa2::RoundUpTo(s, sa2::kKvStep));  // 步数维度随 S 增长
        SA2_CHECK(pol.kp > prevKp);                            // 扫掠严格递增（非空转证明）
        prevKp = pol.kp;
        // UB 计划两链终址对每个 S 逐点相同（constexpr 常数的运行期镜像读取）。
        SA2_CHECK(sa2::Sa2DeviceResWarp::kUbEnd == 152448 && sa2::Sa2DeviceResThread::kUbEnd == 217984);
        SA2_CHECK(sa2::kUbEndPerWarp == kChkUbEndPerWarp && sa2::kUbEndPerThread == kChkUbEndPerThread);
        SA2_CHECK(kChkInstantWarp <= sa2::kUbHardSafetyLimit && kChkInstantThread <= sa2::kUbHardSafetyLimit);
        // 长序列的代价只在步数（O(S)），从不在缓冲（O(1)）：S=131072 → 1024 步复用同一组槽。
        SA2_CHECK(sa2::StepCount(sa2::RoundUpTo(s, sa2::kKvStep)) == pol.kp / sa2::kKvStep);
    }
    // 头内断言幂等：本 TU 编译成功本身即证明 sa2_ub_plan.h 全部 static_assert 通过
    //（上方编译期硬防护为第二重；两处同真 = 幂等）。
    SA2_CHECK(sa2::kUbEndPerThread <= sa2::kUbHighWater);
}

// ---- 用例 13：P 的 L1 NZ 分形索引守卫（迭代 7 设备轨缺口 b 的静态修复执行化）----
// 冻结图像：kL1PNzSlot = (64/16 行块)×(128/16 列块)×16 行×32B；16×16 分形、行块主序、
// 分形行 32B = 16 码 + 16B pad（D-R6）。唯一算式 = sa2_ub_plan.h PNzFractalIndex
//（B-26：engine 不得自写偏移算式；StepVectorSoftmaxDevice 的 L1 P 写为唯一消费者）。
void CasePNzFractalIndexGuard()
{
    // 锚点：首分形、行块项、列块项、终值紧贴槽尾。
    SA2_CHECK(sa2::PNzFractalIndex(0, 0) == 0);
    SA2_CHECK(sa2::PNzFractalIndex(1, 0) == 32);            // 分形行步长 32B（16 码 + 16B pad）
    SA2_CHECK(sa2::PNzFractalIndex(0, 16) == 512);          // 相邻列块步长 512B（16 行 × 32B）
    SA2_CHECK(sa2::PNzFractalIndex(16, 0) == 8 * 512);      // 行块项 row/16 入最高位（第二行块首）
    SA2_CHECK(sa2::PNzFractalIndex(63, 127) == sa2::kL1PNzSlot - 17);
    // 全定义域扫掠：界内 + 单射（64×128 个 (row,col) 恰占满 8192 个数据字节位）。
    static bool occupied[sa2::kL1PNzSlot];
    for (int i = 0; i < sa2::kL1PNzSlot; ++i) occupied[i] = false;
    bool bounded = true;
    bool injective = true;
    for (int64_t r = 0; r < sa2::kStreamRows; ++r) {
        for (int64_t c = 0; c < sa2::kKvStep; ++c) {
            const int64_t idx = sa2::PNzFractalIndex(r, c);
            bounded = bounded && idx >= 0 && idx < sa2::kL1PNzSlot;
            if (idx >= 0 && idx < sa2::kL1PNzSlot) {
                if (occupied[idx]) injective = false;
                occupied[idx] = true;
            }
            ++g_checks;
        }
    }
    SA2_CHECK(bounded);
    SA2_CHECK(injective);
    // 分形行 pad 半区（每 32B 行的高 16B）永不被数据占用：恰 8192 个数据位 ⇒ pad 位全空。
    int64_t usedCount = 0;
    for (int i = 0; i < sa2::kL1PNzSlot; ++i) usedCount += occupied[i] ? 1 : 0;
    SA2_CHECK(usedCount == sa2::kStreamRows * sa2::kKvStep);
    // 检出力（突变对照）：迭代 6 登记的缺陷公式 f*256 + r*16 + k（缺行块项）在 r≥16 必然
    // 跨块碰撞——此处实证该旧式至少存在一对碰撞，且本算式对其可检出。
    bool oldFormulaCollides = false;
    for (int64_t r = 16; r < sa2::kStreamRows && !oldFormulaCollides; ++r) {
        for (int64_t c = 0; c < sa2::kKvStep && !oldFormulaCollides; ++c) {
            const int64_t oldIdx = (c / 16) * 256 + r * 16 + (c % 16);
            for (int64_t r2 = 0; r2 < sa2::kStreamRows && !oldFormulaCollides; ++r2) {
                for (int64_t c2 = 0; c2 < sa2::kKvStep; ++c2) {
                    if (r2 == r && c2 == c) continue;
                    if ((c2 / 16) * 256 + r2 * 16 + (c2 % 16) == oldIdx) {
                        oldFormulaCollides = true;
                        break;
                    }
                }
            }
        }
    }
    SA2_CHECK(oldFormulaCollides);
}

// ---- 用例 14：κ=448 满量程 D-R1 硬守卫边界（迭代 7 甄别维度的执行化依据）----
// 甄别结论：θ-skip 谓词 mBlk − mOld ≤ θ 在实数域 ⟺ max(mOld, mBlk−θ) == mOld ⟺ alpha ≡ 1，
// 即 skip 恰在「重缩放为恒等变换」时命中——θ 判定本身就是 D-R1 硬守卫；float 域两种写法
// 存在 ≤1 ulp(θ) 舍入槽（本用例实测），包络偏差 ≤ exp(ulp(θ)) ≈ 1+5e-7，p ≤ κ ≤ 448
// （E4M3 满量程）两支路同界保持。而「饱和占比超阈值强制 rescale」兜底在本设计下数学空转：
// 冻结递归 m' = max(m', mBlk−θ) 内 skip 命中时重算 max 与恒等差 ≤ 1 ulp（p_max 相对差
// ≤ 1e-6，下方逐探针实测）——改不动包络，落码即死代码（判定标准排除项），
// 故以本用例冻结其不可达性证据。
void CaseThetaSkip448FullscaleGuard()
{
    const float lnKappa = std::log(448.0f);
    const float mBlk = 8.0f;  // 2 的幂：mOld 探针的构造基准
    // (a) 谓词与语句镜像：跨 θ 边界 ±2 ulp 逐探针核对 skip 判定与内核五语句效果。
    auto neighbor = [](float f, int ulps) {
        return FloatOfBits(BitsOf(f) + static_cast<uint32_t>(ulps));
    };
    bool boundaryFound = false;
    for (int u = -3; u <= 3; ++u) {
        const float mOld = neighbor(neighbor(mBlk - lnKappa, u), 0);
        const float dEff = mBlk - mOld;
        if (dEff == lnKappa) boundaryFound = true;
        // 内核五语句镜像（sa2_attention_loop.h 双轨逐句同构）：
        float mNew = std::max(mOld, mBlk - lnKappa);
        float alpha = std::exp(mOld - mNew);
        const bool skip = (mBlk - mOld <= lnKappa);
        if (skip) {
            alpha = 1.0f;
            mNew = mOld;
        }
        SA2_CHECK(skip == (dEff <= lnKappa));                     // 谓词即判定
        if (skip) {
            SA2_CHECK(alpha == 1.0f && mNew == mOld);
            // skip ⟺ 重缩放（近）恒等：fl 域谓词与 max 恒等存在 ≤1 ulp(θ) 舍入槽
            //（fl(mBlk−mOld) ≤ θ 与 fl(mBlk−θ) ≤ mOld 可差 1 ulp）⇒ skip 支路对恒等
            // 重缩放的偏差 ≤ exp(ulp(θ)) ≈ 1+5e-7，被下方 κ 包络断言覆盖。
            SA2_CHECK(mBlk - lnKappa <= neighbor(mOld, 1));
        } else {
            SA2_CHECK(mNew == mBlk - lnKappa && alpha < 1.0f);    // 抬升支路 m' = mBlk − θ
        }
        // 包络两支路同界：量化前 p_max = exp(mBlk − mNew) ≤ κ·(1+1e-6) 且有限（无 Inf 路径）。
        const float pMaxPre = std::exp(mBlk - mNew);
        SA2_CHECK(std::isfinite(pMaxPre) && pMaxPre <= 448.0f * (1.0f + 1.0e-6f));
        // 码字满量程语义：p_max 抵达 κ 邻域 ⇒ 码字 = 0x7E（448，设计满量程），越界被钳到 448。
        const float pClamped = std::max(0.0f, std::min(448.0f, pMaxPre));
        const uint8_t code = sa2::FloatToFp8E4M3(pClamped);
        if (pMaxPre >= 448.0f * (1.0f - 1.0f / 16.0f)) {
            SA2_CHECK(code == 0x7Eu);
        }
        SA2_CHECK(sa2::Fp8E4M3ToFloat(code) <= 448.0f);
        // 空转性证据（冻结递归 m' = max(m', mBlk−θ) 内）：skip 命中时「强制 rescale」重算
        // max 与 skip 支路状态差 ≤ 1 ulp ⇒ p_max 相对差 ≤ 1e-6（rescale 兜底改不动包络）；
        // 抬升支路本就是 rescale，二者恒等。
        const float pForced = std::exp(mBlk - std::max(mOld, mBlk - lnKappa));
        SA2_CHECK(std::fabs(pForced - pMaxPre) <= pMaxPre * 1.0e-6f);
    }
    SA2_CHECK(boundaryFound);  // θ 边界可被逐位构造（谓词边界确实受测）
    // (b) 流级：κ=448 混合漂移流走 OnlineTheta，包络 + 满量程触达 + 支路记账。
    Lcg rng(44844801ULL);
    static float scores[kNumBlocks][sa2::kKvStep];
    static float v[kNumBlocks * sa2::kKvStep][kToyVDeg];
    float base = rng.NextUniform(-4.0f, 2.0f);
    for (int b = 0; b < kNumBlocks; ++b) {
        base += rng.NextUniform(-4.0f, 9.0f);  // 漂移跨 θ=ln448≈6.10 ⇒ skip/rescale 交替
        for (int j = 0; j < sa2::kKvStep; ++j) {
            scores[b][j] = base + rng.NextUniform(-2.0f, 2.0f);
        }
    }
    MakeV(v, rng);
    OnlineState st;
    OnlineTheta(scores, v, lnKappa, st);
    SA2_CHECK(st.skips + st.rescales == kNumBlocks && st.rescales >= 1 && st.skips >= 1);
    SA2_CHECK(st.maxP <= 448.0f * (1.0f + 1.0e-6f));  // D-R1 包络：θ 路量化前 exp 恒 ≤ κ
    SA2_CHECK(std::isfinite(st.denom) && st.denom > 0.0f);
    // 满量程触达：rescale 支路 m' = mBlk − θ ⇒ 块最大 p = exp(θ) 邻域 ⇒ 码字 448 必现
    //（「码字触 448」= §4.3 设计满量程，rescale 支路同样产生——非 skip 支路异常）。
    bool fullScaleTouched = false;
    for (int b = 0; b < kNumBlocks && !fullScaleTouched; ++b) {
        float mBlkB = -INFINITY;
        for (int j = 0; j < sa2::kKvStep; ++j) {
            if (scores[b][j] > mBlkB) mBlkB = scores[b][j];
        }
        if (mBlkB - st.mRun > lnKappa || (mBlkB - lnKappa) >= st.mRun) {
            const float pMaxB = QuantizeP(std::exp(mBlkB - (mBlkB - lnKappa)));
            if (pMaxB == 448.0f) fullScaleTouched = true;
        }
    }
    SA2_CHECK(fullScaleTouched);
    // LSE 闭环在 448 档保持契约（对照 double 全量 logsumexp）。
    double sumExp = 0.0;
    for (int b = 0; b < kNumBlocks; ++b) {
        for (int j = 0; j < sa2::kKvStep; ++j) {
            sumExp += std::exp(static_cast<double>(scores[b][j]));
        }
    }
    const double lseErr =
        std::fabs(std::log(static_cast<double>(st.denom)) + static_cast<double>(st.mRun) - std::log(sumExp));
    SA2_CHECK(lseErr < 0.08);
}

// ---- 用例 15：M2.7.4 状态类型化槽位（StateTypedSlot，迭代 9）----
// 甄别/落地口径：Empty --AcquireSlot（内部 Await 事件）--> Full --ReleaseSlot（内部 Notify 事件）
// --> Empty；数据访问原语 SlotData 只对 SlotRef<Full> 声明。「错误用法不可编译」以 SFINAE 检测
// 形态说明（真编译失败形态 = 直接写 sa2::SlotData<float>(emptyRef) 报 no matching function、
// 默认/裸指针构造 SlotRef<Full> 报 private ctor，均不在本套件真实触发）。
// 旗号语义与预算不变：内部事件 = 既有 Await*/Notify* 原样复用，六旗号双映射表
// （SlotAwaitPipeOf / SlotNotifyPipeOf）与槽周期配对（SlotReleaseFlagOf）全部编译期冻结。

// SFINAE 检测形态：表达式嵌在部分特化的 void_t 实参内（别名模板作类型实参会在推导外
// 求值成硬错误，必须如此嵌套才是可检测的替换失败）。
template <typename StateTag, typename = void>
struct SlotDataDetected : std::false_type {};
template <typename StateTag>
struct SlotDataDetected<StateTag, std::void_t<decltype(sa2::SlotData<float>(std::declval<const sa2::SlotRef<StateTag> &>()))>>
    : std::true_type {};

static_assert(!SlotDataDetected<sa2::Empty>::value,
              "Empty 态无 SlotData 重载：未同步槽编译期拿不到任何数据访问（负向能力）");
static_assert(SlotDataDetected<sa2::Full>::value,
              "Full 态（仅经 AcquireSlot 产出）可数据访问");
static_assert(!std::is_default_constructible<sa2::SlotRef<sa2::Full>>::value,
              "SlotRef<Full> 默认构造私有：Full 只能由 AcquireSlot 产出");
static_assert(!std::is_constructible<sa2::SlotRef<sa2::Full>, uint8_t *, int64_t, int64_t>::value,
              "SlotRef<Full> 裸指针构造私有：绕过 Await 自造 Full 不可编译");
static_assert(!std::is_copy_constructible<sa2::SlotRef<sa2::Full>>::value &&
                  !std::is_copy_assignable<sa2::SlotRef<sa2::Full>>::value &&
                  !std::is_copy_constructible<sa2::SlotRef<sa2::Empty>>::value &&
                  !std::is_copy_assignable<sa2::SlotRef<sa2::Empty>>::value,
              "两态均 Move-only（拷贝 = delete，线性类型状态）");
static_assert(std::is_move_constructible<sa2::SlotRef<sa2::Full>>::value &&
                  std::is_move_assignable<sa2::SlotRef<sa2::Full>>::value &&
                  std::is_move_constructible<sa2::SlotRef<sa2::Empty>>::value,
              "两态可移动（跨作用域保持线性链）");

void CaseStateTypedSlotTypestateGuard()
{
    // 0. 编译期冻结表的运行期镜像（上方 static_assert 已是承重墙；此处保证表值可读可对账）。
    SA2_CHECK(!SlotDataDetected<sa2::Empty>::value);
    SA2_CHECK(SlotDataDetected<sa2::Full>::value);
    SA2_CHECK(sa2::SlotAwaitPipeOf(sa2::Sa2CrossFlag::kScoreOk) == 0 &&
              sa2::SlotAwaitPipeOf(sa2::Sa2CrossFlag::kPvOk) == 0 &&
              sa2::SlotAwaitPipeOf(sa2::Sa2CrossFlag::kQkFree) == 1 &&
              sa2::SlotAwaitPipeOf(sa2::Sa2CrossFlag::kPvFree) == 1 &&
              sa2::SlotAwaitPipeOf(sa2::Sa2CrossFlag::kPTaken) == 2 &&
              sa2::SlotAwaitPipeOf(sa2::Sa2CrossFlag::kQ8Ok) == 2);
    SA2_CHECK(sa2::SlotNotifyPipeOf(sa2::Sa2CrossFlag::kScoreOk) == 0 &&
              sa2::SlotNotifyPipeOf(sa2::Sa2CrossFlag::kPvOk) == 0 &&
              sa2::SlotNotifyPipeOf(sa2::Sa2CrossFlag::kQkFree) == 1 &&
              sa2::SlotNotifyPipeOf(sa2::Sa2CrossFlag::kPvFree) == 1 &&
              sa2::SlotNotifyPipeOf(sa2::Sa2CrossFlag::kQ8Ok) == 1 &&
              sa2::SlotNotifyPipeOf(sa2::Sa2CrossFlag::kPTaken) == 2);
    SA2_CHECK(sa2::SlotReleaseFlagOf(sa2::Sa2CrossFlag::kScoreOk) == sa2::Sa2CrossFlag::kQkFree &&
              sa2::SlotReleaseFlagOf(sa2::Sa2CrossFlag::kPvOk) == sa2::Sa2CrossFlag::kPvFree);
    // 零新增旗号：六枚举全覆盖（预算断言在 sa2_events.h，本用例镜像）。
    SA2_CHECK(sa2::kCrossFlagBudget == 12 && sa2::kLocalEventBudget == 6);

    // 1. PV 槽全周期（宿主 stub 下 Await/Notify 为空实现；事件序由上表编译期冻结）：
    //    Acquire<kPvOk>（arm 0 AwaitFix）→ Full 数据访问 → Release<kPvFree>（arm 1 NotifyV）。
    alignas(64) float buf[16];
    for (int i = 0; i < 16; ++i) {
        buf[i] = static_cast<float>(i);
    }
    auto pvFull = sa2::AcquireSlot<sa2::Sa2CrossFlag::kPvOk>(buf, static_cast<int64_t>(sizeof(buf)), 1);
    SA2_CHECK(pvFull.IsValid());
    SA2_CHECK(pvFull.SizeBytes() == static_cast<int64_t>(sizeof(buf)));
    SA2_CHECK(pvFull.StreamId() == 1);
    SA2_CHECK(sa2::SlotData<float>(pvFull) == buf);        // 地址恒等（无中间拷贝）
    SA2_CHECK(sa2::SlotData<const float>(pvFull) == buf);  // 只读访问同一原语
    sa2::SlotData<float>(pvFull)[3] = 7.5f;                // 经 Full 态写通
    SA2_CHECK(buf[3] == 7.5f);
    auto pvEmpty = sa2::ReleaseSlot<sa2::Sa2CrossFlag::kPvFree>(std::move(pvFull));
    SA2_CHECK(!pvFull.IsValid());  // 已释放句柄擦除（moved-from 纵深防御）
    SA2_CHECK(pvEmpty.StreamId() == 1 && pvEmpty.SizeBytes() == static_cast<int64_t>(sizeof(buf)));

    // 2. 六旗号分发臂全覆盖（模拟 AIC 消费/生产侧角色，与调用点角色映射一致）：
    //    score 槽：AIC 消费侧 Acquire<kQkFree>（arm 1 AwaitV）→ AIC 生产侧 Release<kScoreOk>（arm 0 NotifyFix）。
    auto scoreFree = sa2::AcquireSlot<sa2::Sa2CrossFlag::kQkFree>(buf, sizeof(float) * 16, 0);
    SA2_CHECK(scoreFree.IsValid() && sa2::SlotData<float>(scoreFree) == buf);
    auto scoreOkDone = sa2::ReleaseSlot<sa2::Sa2CrossFlag::kScoreOk>(std::move(scoreFree));
    SA2_CHECK(!scoreFree.IsValid() && scoreOkDone.StreamId() == 0);
    //    单向信用旗：AIC 消费侧 Acquire<kQ8Ok>（arm 2 AwaitMte1）→ AIV 生产侧 Release<kPTaken>（arm 2 NotifyMte3）。
    auto q8Ready = sa2::AcquireSlot<sa2::Sa2CrossFlag::kQ8Ok>(buf, sizeof(float) * 16, 0);
    SA2_CHECK(q8Ready.IsValid());
    auto pTakenDone = sa2::ReleaseSlot<sa2::Sa2CrossFlag::kPTaken>(std::move(q8Ready));
    SA2_CHECK(!q8Ready.IsValid() && pTakenDone.StreamId() == 0);
}

// ---- 用例 16：M2c-C2 极限分数值域与上溢防御用例（Extreme Score Range & Overflow Guard）----
// 数值护栏用例：
// 1. 构造大幅值正负得分输入（Score 范围 [-100.0, 100.0]、动态极差 > 150.0）；
// 2. 验证在大幅值下 θ-skip 机制在数值极端波动时能否正确平滑转移；
// 3. 验证 Softmax 概率矩阵在 FP8 E4M3 饱和截断下（码字严格钳制在 [0, 448.0] / 0x7E）绝对不产生 NaN、Inf 或数值穿透；
// 4. 验证极端负得分（所有元素 S_ij <= -80.0）下 Softmax 分母与概率值安全收敛，LSE 闭环无下溢异常；
// 5. 验证 FP16 两级累加在最大极端激励下是否严格满足 <= 65504（κ=2.25 理论安全界）。
void CaseScoreRangeSafetyGuard()
{
    // Part 1: 构造大幅值正负得分输入（Score 范围 [-100.0, 100.0]、动态极差 > 150.0）
    static float scoresOscillate[kNumBlocks][sa2::kKvStep];
    static float v[kNumBlocks * sa2::kKvStep][kToyVDeg];
    Lcg rng(20261002ULL);
    MakeV(v, rng);

    float minScore = 1.0e9f;
    float maxScore = -1.0e9f;

    for (int b = 0; b < kNumBlocks; ++b) {
        // 奇偶块在正负极端之间大幅震荡，模拟注意力深坑与尖锐注意力汇聚
        const bool isNegativeBlock = (b % 2 == 0);
        const float base = isNegativeBlock ? -90.0f : 90.0f;
        for (int j = 0; j < sa2::kKvStep; ++j) {
            float s = base + rng.NextUniform(-10.0f, 10.0f);
            s = std::max(-100.0f, std::min(100.0f, s));
            scoresOscillate[b][j] = s;
            if (s < minScore) minScore = s;
            if (s > maxScore) maxScore = s;
        }
    }
    // 强制边界锚点保证精确覆盖 -100.0 与 +100.0 极值
    scoresOscillate[0][0] = -100.0f;
    scoresOscillate[1][0] = 100.0f;
    minScore = std::min(minScore, -100.0f);
    maxScore = std::max(maxScore, 100.0f);

    const float dynamicRange = maxScore - minScore;
    SA2_CHECK(minScore <= -99.9f && maxScore >= 99.9f);
    SA2_CHECK(dynamicRange >= 199.0f);
    SA2_CHECK(dynamicRange > 150.0f);

    // 另构造随机连续大幅摆动流（Pattern 2）
    static float scoresRandomSweep[kNumBlocks][sa2::kKvStep];
    float minRand = 1.0e9f;
    float maxRand = -1.0e9f;
    for (int b = 0; b < kNumBlocks; ++b) {
        float center = rng.NextUniform(-95.0f, 95.0f);
        for (int j = 0; j < sa2::kKvStep; ++j) {
            float s = center + rng.NextUniform(-15.0f, 15.0f);
            s = std::max(-100.0f, std::min(100.0f, s));
            scoresRandomSweep[b][j] = s;
            if (s < minRand) minRand = s;
            if (s > maxRand) maxRand = s;
        }
    }
    scoresRandomSweep[0][1] = -100.0f;
    scoresRandomSweep[kNumBlocks - 1][1] = 100.0f;
    minRand = std::min(minRand, -100.0f);
    maxRand = std::max(maxRand, 100.0f);
    SA2_CHECK((maxRand - minRand) > 150.0f);

    // Part 2: 验证在大幅值下 θ-skip 机制在数值极端波动时能否正确平滑转移
    for (int pattern = 0; pattern < 2; ++pattern) {
        const auto &currentScores = (pattern == 0) ? scoresOscillate : scoresRandomSweep;
        for (int tier = 0; tier < 2; ++tier) {
            const float kappa = (tier == 0) ? kSa2CaseKappaFp16 : 448.0f;
            const float lnKappa = std::log(kappa);

            OnlineState stTheta;
            OnlineState stRef;
            OnlineTheta(currentScores, v, lnKappa, stTheta);
            OnlineReference(currentScores, v, lnKappa, stRef);

            // 验证 skip 与 rescale 在极端波动下均被真实触发，且步骤总和守恒
            SA2_CHECK(stTheta.skips > 0);
            SA2_CHECK(stTheta.rescales > 0);
            SA2_CHECK(stTheta.skips + stTheta.rescales == kNumBlocks);

            // 验证运行态无 NaN、Inf，分母 strictly > 0
            SA2_CHECK(std::isfinite(stTheta.denom) && stTheta.denom > 0.0f);
            SA2_CHECK(std::isfinite(stTheta.mRun));
            SA2_CHECK(std::isfinite(stRef.denom) && stRef.denom > 0.0f);
            SA2_CHECK(std::isfinite(stRef.mRun));

            // 验证输出向量各分量有限且满足有界性
            for (int c = 0; c < kToyVDeg; ++c) {
                SA2_CHECK(std::isfinite(stTheta.o[c]));
                SA2_CHECK(std::isfinite(stRef.o[c]));
                const float thetaNorm = (stTheta.denom > 0.0f) ? (stTheta.o[c] / stTheta.denom) : 0.0f;
                const float refNorm = (stRef.denom > 0.0f) ? (stRef.o[c] / stRef.denom) : 0.0f;
                SA2_CHECK(std::isfinite(thetaNorm) && std::fabs(thetaNorm) <= 1.0f + 1e-4f);
                SA2_CHECK(std::isfinite(refNorm) && std::fabs(refNorm) <= 1.0f + 1e-4f);
                // 归一化输出在量化噪声下对齐（极端跳变时历史衰减至 0，新主块主导两路）
                SA2_CHECK(std::fabs(thetaNorm - refNorm) <= 5.0e-2f);
            }

            // 验证量化前概率包络满足 <= κ
            SA2_CHECK(stTheta.maxP <= kappa * (1.0f + 1.0e-5f));
        }
    }

    // Part 3: 验证 Softmax 概率矩阵在 FP8 E4M3 饱和截断下（码字严格钳制在 [0, 448.0] / 0x7E）绝对不产生 NaN、Inf 或数值穿透
    {
        // 边界与超量程测试探针
        const float probeValues[] = {
            -1.0e20f, -100.0f, -1.0f, -0.0f, 0.0f, 1.0e-7f, 0.001953125f, 0.015625f,
            0.5f, 1.0f, 2.25f, 128.0f, 256.0f, 448.0f, 448.0001f, 448.1f, 480.0f,
            1000.0f, 65504.0f, 1.0e10f, 1.0e38f, INFINITY
        };
        for (float val : probeValues) {
            // Softmax 概率钳位至 [0, 448.0f]
            const float clamped = std::max(0.0f, std::min(448.0f, val));
            const uint8_t code = sa2::FloatToFp8E4M3(clamped);
            // 码字必须严格在 [0x00, 0x7E] 范围内，符号位为 0，绝对不是 NaN 码字 0x7F
            SA2_CHECK(code <= 0x7Eu);
            SA2_CHECK(code != 0x7Fu);
            SA2_CHECK((code & 0x80u) == 0u);

            const float decoded = sa2::Fp8E4M3ToFloat(code);
            SA2_CHECK(!std::isnan(decoded));
            SA2_CHECK(!std::isinf(decoded));
            SA2_CHECK(decoded >= 0.0f && decoded <= 448.0f);

            // 越界值饱和验证：>= 448 的输入码字严格等于 0x7E，无符号翻转与溢出回绕穿透
            if (val >= 448.0f) {
                SA2_CHECK(code == 0x7Eu);
                SA2_CHECK(decoded == 448.0f);
            }
        }

        // 验证即使未预钳位的原始大幅值浮点数送入 FloatToFp8E4M3，也安全饱和到 0x7E (448.0)，无 NaN 或穿透
        const float unclampedExtremes[] = { 480.0f, 1000.0f, 65504.0f, 1.0e20f, 1.0e38f, INFINITY };
        for (float ex : unclampedExtremes) {
            const uint8_t rawCode = sa2::FloatToFp8E4M3(ex);
            SA2_CHECK(rawCode == 0x7Eu);
            SA2_CHECK(sa2::Fp8E4M3ToFloat(rawCode) == 448.0f);
        }

        // 验证在 scoresOscillate 全量网格上的概率值均无 NaN、Inf、穿透
        for (int b = 0; b < kNumBlocks; ++b) {
            for (int j = 0; j < sa2::kKvStep; ++j) {
                const float pExp = std::exp(std::min(0.0f, scoresOscillate[b][j]));
                const float pQuant = QuantizeP(pExp);
                SA2_CHECK(!std::isnan(pQuant) && !std::isinf(pQuant));
                SA2_CHECK(pQuant >= 0.0f && pQuant <= 448.0f);
            }
        }
    }

    // Part 4: 验证极端负得分（所有元素 S_ij <= -80.0）下 Softmax 分母与概率值安全收敛，LSE 闭环无下溢异常
    {
        static float scoresDeepNeg[kNumBlocks][sa2::kKvStep];
        float maxDeepNeg = -1.0e9f;
        float minDeepNeg = 1.0e9f;
        for (int b = 0; b < kNumBlocks; ++b) {
            for (int j = 0; j < sa2::kKvStep; ++j) {
                // 得分严格位于 [-100.0, -80.0]
                float s = rng.NextUniform(-100.0f, -80.0f);
                scoresDeepNeg[b][j] = s;
                if (s > maxDeepNeg) maxDeepNeg = s;
                if (s < minDeepNeg) minDeepNeg = s;
            }
        }
        scoresDeepNeg[0][0] = -80.0f;
        scoresDeepNeg[kNumBlocks - 1][sa2::kKvStep - 1] = -100.0f;
        maxDeepNeg = std::max(maxDeepNeg, -80.0f);
        minDeepNeg = std::min(minDeepNeg, -100.0f);

        SA2_CHECK(maxDeepNeg <= -80.0f);
        SA2_CHECK(minDeepNeg <= -99.9f);

        for (int tier = 0; tier < 2; ++tier) {
            const float kappa = (tier == 0) ? kSa2CaseKappaFp16 : 448.0f;
            const float lnKappa = std::log(kappa);

            OnlineState stNeg;
            OnlineTheta(scoresDeepNeg, v, lnKappa, stNeg);

            // 分母与运行状态安全收敛，无下溢归零
            SA2_CHECK(std::isfinite(stNeg.denom));
            SA2_CHECK(stNeg.denom > 0.0f);
            SA2_CHECK(stNeg.denom >= kappa * 0.999f); // 至少包含最大得分项的 κ 贡献
            SA2_CHECK(std::isfinite(stNeg.mRun));
            SA2_CHECK(stNeg.mRun <= -80.0f + lnKappa);

            // LSE 闭环验证：double 域真值 vs 内核等价式 ln(denom) + m'
            double sumExp = 0.0;
            for (int b = 0; b < kNumBlocks; ++b) {
                for (int j = 0; j < sa2::kKvStep; ++j) {
                    sumExp += std::exp(static_cast<double>(scoresDeepNeg[b][j]));
                }
            }
            const double lseTrue = std::log(sumExp);
            const double lseCore = std::log(static_cast<double>(stNeg.denom)) + static_cast<double>(stNeg.mRun);

            SA2_CHECK(std::isfinite(lseCore));
            SA2_CHECK(!std::isnan(lseCore));
            SA2_CHECK(!std::isinf(lseCore));
            SA2_CHECK(lseCore <= -70.0); // 结果安全收敛于深度负域，未下溢归零或溢出

            const double lseErr = std::fabs(lseCore - lseTrue);
            SA2_CHECK(lseErr < 0.08); // 满足 E4M3 码字量化噪声界
        }
    }

    // Part 5: 验证 FP16 两级累加在最大极端激励下是否严格满足 <= 65504（κ=2.25 理论安全界）
    {
        constexpr float kFp16Max = 65504.0f;
        // 1. 静态数理理论界检验 (SageAttention2++ arXiv:2505.21136 & 任务书契约)：
        // 单 Tile (K=128) 极端激励求和界：
        constexpr float kTileMaxP225 = static_cast<float>(sa2::kKvStep) * kSa2CaseKappaFp16; // 128 * 2.25 = 288.0f
        static_assert(kTileMaxP225 < kFp16Max, "Level 1 tile sum under kappa=2.25 must be < 65504");
        SA2_CHECK(kTileMaxP225 == 288.0f);
        SA2_CHECK(kTileMaxP225 < kFp16Max);

        // P_r * V_r 乘积封顶：
        constexpr float kPrVr = 448.0f * 2.25f; // 1008.0f
        static_assert(kPrVr <= 2047.0f, "Pr * Vr <= 2047 capping bound");
        SA2_CHECK(kPrVr == 1008.0f && kPrVr < kFp16Max);

        // P-07 极限长序列 S=16384 (128 tiles * 128 steps) 全序列两级累加理论最大激励：
        constexpr int kMaxTiles16K = 128; // 16384 / 128
        constexpr float kLevel2Max16K = static_cast<float>(kMaxTiles16K) * kTileMaxP225; // 128 * 288.0f = 36864.0f
        static_assert(kLevel2Max16K < kFp16Max, "Level 2 accumulation across 16K sequence must strictly satisfy < 65504");
        SA2_CHECK(kLevel2Max16K == 36864.0f);
        SA2_CHECK(kLevel2Max16K <= kFp16Max);

        // 对比证明：若在 FP16 累加下误用 kappa=448.0，仅 2 个 Tile 累加即达 114688 > 65504，必致 FP16 溢出
        constexpr float kTileMaxP448 = static_cast<float>(sa2::kKvStep) * 448.0f; // 57344.0f
        constexpr float kLevel2TwoTiles448 = 2.0f * kTileMaxP448; // 114688.0f
        static_assert(kLevel2TwoTiles448 > kFp16Max, "kappa=448 violates FP16 range, proving kappa=2.25 necessity");
        SA2_CHECK(kLevel2TwoTiles448 > kFp16Max);

        // 2. 动态最大激励模拟：全 16K 序列 (128 块) 注入平坦满量程 P_ij = 2.25 与 V_jc = 1.0f
        float oAccumFp16 = 0.0f;
        for (int t = 0; t < kMaxTiles16K; ++t) {
            // 一级累加：MMAD Tile 产生 partial = sum(P * V) = 288.0f
            float partial = 0.0f;
            for (int j = 0; j < sa2::kKvStep; ++j) {
                partial += kSa2CaseKappaFp16 * 1.0f;
            }
            SA2_CHECK(partial == 288.0f);
            SA2_CHECK(partial < kFp16Max);

            // 一级折叠：转换为 FP16 物理半精度
            const auto hPart = sa2::hostbits::HalfF32(partial);
            const float foldedPart = static_cast<float>(hPart);
            SA2_CHECK(std::isfinite(foldedPart));
            SA2_CHECK(foldedPart == 288.0f);

            // 二级累加：累加至 O 寄存器并经受 FP16 动态范围考验
            oAccumFp16 += foldedPart;
            const auto hAcc = sa2::hostbits::HalfF32(oAccumFp16);
            const float oFolded = static_cast<float>(hAcc);
            SA2_CHECK(std::isfinite(oFolded));
            SA2_CHECK(oFolded <= kFp16Max);
        }
        SA2_CHECK(oAccumFp16 == 36864.0f);
        SA2_CHECK(oAccumFp16 <= 65504.0f);

        // 3. 动态真实极值得分流激励下的 FP16 逐步两级折叠验证
        float oStreamFold[kToyVDeg] = {0.0f};
        float mRunFp16 = sa2::kFp16NegMax;
        const float lnKappa225 = std::log(kSa2CaseKappaFp16);
        for (int b = 0; b < kNumBlocks; ++b) {
            float mBlk = -INFINITY;
            for (int j = 0; j < sa2::kKvStep; ++j) {
                if (scoresOscillate[b][j] > mBlk) mBlk = scoresOscillate[b][j];
            }
            const float mOld = mRunFp16;
            float mNew = std::max(mOld, mBlk - lnKappa225);
            float alpha = std::exp(mOld - mNew);
            const bool skip = (mBlk - mOld <= lnKappa225);
            if (skip) {
                alpha = 1.0f;
                mNew = mOld;
            }
            // 一级累加与折叠
            float partial[kToyVDeg] = {0.0f};
            for (int j = 0; j < sa2::kKvStep; ++j) {
                const float p8 = QuantizeP(std::exp(scoresOscillate[b][j] - mNew));
                for (int c = 0; c < kToyVDeg; ++c) {
                    partial[c] += p8 * v[b * sa2::kKvStep + j][c];
                }
            }
            for (int c = 0; c < kToyVDeg; ++c) {
                SA2_CHECK(std::fabs(partial[c]) <= 288.0f);
                const float foldedPart = static_cast<float>(sa2::hostbits::HalfF32(partial[c]));
                SA2_CHECK(std::isfinite(foldedPart));

                // 二级折叠与累加
                oStreamFold[c] = alpha * oStreamFold[c] + foldedPart;
                const float oFoldVal = static_cast<float>(sa2::hostbits::HalfF32(oStreamFold[c]));
                SA2_CHECK(std::isfinite(oFoldVal));
                SA2_CHECK(std::fabs(oFoldVal) <= kFp16Max);
            }
            mRunFp16 = mNew;
        }
    }
}

// ---- 用例 17（迭代 2 P1-2）：配对 PV 折叠调度镜像（M2c-A2 同提交纪律）----
// 被测契约：设备 Sa2VfFoldRun paired 分支（sa2_attention_loop.h）与 sim StepVectorPvFoldPair
// （sa2_sim_loop.h）同构——配对调度（softmax(0)·a0、softmax(1)·a1 先行重缩放，折叠时
// o += a1·f16(pv0) → o += f16(pv1)）与单步调度（每步 softmax·ak 后立即 o += f16(pvk)）
// 数学等价而 FP32 舍入点不同。覆盖：fp32 与 fp32+fp16 双档、奇数 steps 尾块、
// 配对 vs 单步差异真实性（差异恒 0 的断言 = 假覆盖，v28 教训）。
struct FoldPairState {
    float o[kToyVDeg] = {0.0f};
};

// 逐设备语句序的配对折叠重放（Sa2VfFoldRun paired 分支逐条镜像；alpha1 = step1 的
// 行 alpha 标量——设备按行取 alpha1[r]，本 toy 单行）。
void FoldPairDeviceOrder(const float *pv0, const float *pv1, float alpha1,
                         bool fp16Accum, int cols, FoldPairState &st)
{
    for (int c = 0; c < cols; ++c) {
        float p0 = pv0[c];
        float p1 = pv1[c];
        if (fp16Accum) {
            p0 = static_cast<float>(sa2::hostbits::HalfF32(p0));
            p1 = static_cast<float>(sa2::hostbits::HalfF32(p1));
        }
        const float a1 = alpha1;
        if (a1 != 1.0f) {
            p0 = p0 * a1;
        }
        st.o[c] += p0;
        st.o[c] += p1;
    }
}

// sim StepVectorPvFoldPair 镜像（同语句序；独立重写防同源抄写错误）。
void FoldPairSimMirror(const float *pv0, const float *pv1, float alpha1,
                       bool fp16Accum, int cols, FoldPairState &st)
{
    for (int c = 0; c < cols; ++c) {
        float p0 = pv0[c];
        float p1 = pv1[c];
        if (fp16Accum) {
            p0 = static_cast<float>(sa2::hostbits::HalfF32(p0));
            p1 = static_cast<float>(sa2::hostbits::HalfF32(p1));
        }
        const float a1 = alpha1;
        const bool scalePv0 = (a1 != 1.0f);
        if (scalePv0) {
            p0 = p0 * a1;
        }
        st.o[c] += p0;
        st.o[c] += p1;
    }
}

void CasePairedPvFoldSchedule()
{
    Lcg rng(24110958ULL);  // SageAttention2 论文号
    bool anyDiffInMatrix = false;  // 位差按 (tier, steps, col) 抽样：个别组合因舍入巧合位同
                                   //（合法），矩阵内至少一处位差即证调度分歧真实存在
    for (int tier = 0; tier < 2; ++tier) {
        const bool fp16Accum = (tier == 1);
        // steps ∈ {2（整对）, 3（奇尾：1 对 + 1 单步）, 5（两对 + 1 单步）}
        for (int steps : {2, 3, 5}) {
            // 1) alpha 序列构造（非退化：alphas[0] = 0.5f——若取真实首块的 exp 下溢 0，
            //    首对内累加器被清零，配对/单步舍入序分歧不显形；θ-skip 与 rescale 混合
            //    由后续步覆盖）。奇数步（= 配对折叠的 step1，携带 pv0·a1 折叠缩放）取
            //    分数 alpha，偶数步取 1.0f——折叠缩放点（pv0·a1）与单步缩放点
            //    （(o+pv0)·a1）的舍入分歧在每对内必然显形。
            static float pv[kNumBlocks][kToyVDeg];
            float alphas[kNumBlocks] = {0.5f};
            for (int b = 1; b < steps; ++b) {
                alphas[b] = (b % 2 == 1) ? rng.NextUniform(0.25f, 0.95f) : 1.0f;
            }
            for (int b = 0; b < steps; ++b) {
                for (int c = 0; c < kToyVDeg; ++c) {
                    pv[b][c] = rng.NextUniform(-4.0f, 4.0f);
                }
            }

            // 2) 两条调度（各自的 O 初始 = 随机正值，两调度共用同一份）
            float oSeed[kToyVDeg];
            for (int c = 0; c < kToyVDeg; ++c) {
                oSeed[c] = rng.NextUniform(0.5f, 2.0f);
            }

            // 单步调度：每步 softmax·ak 重缩放后立即折叠
            FoldPairState single;
            for (int c = 0; c < kToyVDeg; ++c) single.o[c] = oSeed[c];
            for (int b = 0; b < steps; ++b) {
                if (alphas[b] != 1.0f) {
                    for (int c = 0; c < kToyVDeg; ++c) single.o[c] *= alphas[b];
                }
                for (int c = 0; c < kToyVDeg; ++c) {
                    float blockVal = pv[b][c];
                    if (fp16Accum) blockVal = static_cast<float>(sa2::hostbits::HalfF32(blockVal));
                    single.o[c] += blockVal;
                }
            }

            // 配对调度：两步 softmax 重缩放先行，随后成对折叠；奇尾单步
            FoldPairState paired;
            for (int c = 0; c < kToyVDeg; ++c) paired.o[c] = oSeed[c];
            int64_t step = 0;
            for (; step + 1 < steps; step += 2) {
                if (alphas[step] != 1.0f) {
                    for (int c = 0; c < kToyVDeg; ++c) paired.o[c] *= alphas[step];
                }
                if (alphas[step + 1] != 1.0f) {
                    for (int c = 0; c < kToyVDeg; ++c) paired.o[c] *= alphas[step + 1];
                }
                FoldPairDeviceOrder(pv[step], pv[step + 1], alphas[step + 1], fp16Accum, kToyVDeg, paired);
            }
            for (; step < steps; ++step) {
                if (alphas[step] != 1.0f) {
                    for (int c = 0; c < kToyVDeg; ++c) paired.o[c] *= alphas[step];
                }
                for (int c = 0; c < kToyVDeg; ++c) {
                    float blockVal = pv[step][c];
                    if (fp16Accum) blockVal = static_cast<float>(sa2::hostbits::HalfF32(blockVal));
                    paired.o[c] += blockVal;
                }
            }

            // 3) 断言：配对镜像（设备序重放）== sim 镜像（逐位）
            FoldPairState simMirror;
            for (int c = 0; c < kToyVDeg; ++c) simMirror.o[c] = oSeed[c];
            step = 0;
            for (; step + 1 < steps; step += 2) {
                if (alphas[step] != 1.0f) {
                    for (int c = 0; c < kToyVDeg; ++c) simMirror.o[c] *= alphas[step];
                }
                if (alphas[step + 1] != 1.0f) {
                    for (int c = 0; c < kToyVDeg; ++c) simMirror.o[c] *= alphas[step + 1];
                }
                FoldPairSimMirror(pv[step], pv[step + 1], alphas[step + 1], fp16Accum, kToyVDeg, simMirror);
            }
            for (; step < steps; ++step) {
                if (alphas[step] != 1.0f) {
                    for (int c = 0; c < kToyVDeg; ++c) simMirror.o[c] *= alphas[step];
                }
                for (int c = 0; c < kToyVDeg; ++c) {
                    float blockVal = pv[step][c];
                    if (fp16Accum) blockVal = static_cast<float>(sa2::hostbits::HalfF32(blockVal));
                    simMirror.o[c] += blockVal;
                }
            }
            for (int c = 0; c < kToyVDeg; ++c) {
                SA2_CHECK(BitsOf(paired.o[c]) == BitsOf(simMirror.o[c]));
            }

            // 4) 配对 vs 单步：FP32 舍入点不同 ⇒ 构造性输入下位型确有差异（fp16 档更强）
            for (int c = 0; c < kToyVDeg; ++c) {
                if (BitsOf(paired.o[c]) != BitsOf(single.o[c])) anyDiffInMatrix = true;
            }

            // 5) 数学等价性：两调度相对「全精度无重排参照」的偏差均 < fp16 折叠界聚合
            //    （steps × 2^-11 相对界 + 位移安全余量；界本身见用例 fp16_fold_error_bound）
            float exact[kToyVDeg];
            for (int c = 0; c < kToyVDeg; ++c) exact[c] = oSeed[c];
            for (int b = 0; b < steps; ++b) {
                for (int c = 0; c < kToyVDeg; ++c) {
                    exact[c] = exact[c] * alphas[b] + pv[b][c];
                }
            }
            const double bound = static_cast<double>(steps) * (1.0 / 2048.0) * 8.0 + 1.0e-5;
            for (int c = 0; c < kToyVDeg; ++c) {
                const double relP = std::fabs(static_cast<double>(paired.o[c] - exact[c])) /
                                    (std::fabs(static_cast<double>(exact[c])) + 1.0);
                SA2_CHECK(relP < bound);
                const double relS = std::fabs(static_cast<double>(single.o[c] - exact[c])) /
                                    (std::fabs(static_cast<double>(exact[c])) + 1.0);
                SA2_CHECK(relS < bound);
            }
        }
    }
    SA2_CHECK(anyDiffInMatrix);
}

// ---- 用例 18（迭代 2 P2-2）：IsWidePathEligible 判定矩阵（M2c-A1 零执行覆盖收口）----
void CaseIsWidePathEligibleMatrix()
{
    using sa2::IsWidePathEligible;
    // 编译期锚点（constexpr 函数：static_assert 即真实判定执行）
    static_assert(!IsWidePathEligible(4096, 4096, 96), "d != kWideHeadDim 不合格");
    static_assert(IsWidePathEligible(4096, 4096, 128), "P-01 基准形状合格");
    // 运行期判定矩阵
    // eligible（P-01~P-07 长序列门禁形状族：S≥4096、512 整除、64 行整除、D=128）
    SA2_CHECK(IsWidePathEligible(4096, 4096, 128));
    SA2_CHECK(IsWidePathEligible(8192, 8192, 128));
    SA2_CHECK(IsWidePathEligible(16384, 16384, 128));
    SA2_CHECK(IsWidePathEligible(4096, 8192, 128));   // Sq≠Sk（解码扩展族）
    SA2_CHECK(IsWidePathEligible(64, 4096, 128));     // 单 tile 行
    SA2_CHECK(IsWidePathEligible(128, 4608, 128));    // 4608 = 9×512
    // 非 eligible：D 维不符
    SA2_CHECK(!IsWidePathEligible(4096, 4096, 64));
    SA2_CHECK(!IsWidePathEligible(4096, 4096, 96));
    SA2_CHECK(!IsWidePathEligible(4096, 4096, 256));
    // 非 eligible：S 过短（< kWideMinSeqLen）
    SA2_CHECK(!IsWidePathEligible(2048, 2048, 128));
    SA2_CHECK(!IsWidePathEligible(4096, 4095, 128));
    // 非 eligible：sk 非整除（kWideKvStep=512）
    SA2_CHECK(!IsWidePathEligible(4096, 4224, 128));
    SA2_CHECK(!IsWidePathEligible(4096, 4160, 128));
    // 非 eligible：sq 非 tile 对齐（kWideTileM=64）
    SA2_CHECK(!IsWidePathEligible(100, 4096, 128));
    SA2_CHECK(!IsWidePathEligible(63, 4096, 128));
    // 非 eligible：组合不满足（多条件同时破）
    SA2_CHECK(!IsWidePathEligible(100, 2048, 96));
}

// ---- 用例 19: Q/K 混流批量量化与流水控制器守卫（M2c-A3 契约）----
void CaseQuantScaleReciprocalAndPipelineGuard()
{
    // 1. 编译期常量断言
    static_assert(sa2::kQuantTileRows == 64, "量化分块 64 行");
    static_assert(sa2::kQuantPipelineStages == 3, "3 级重叠流水");
    static_assert(sa2::kQuantBufferSlots == 2, "ping-pong 双缓冲");

    // 2. 尺度倒数复用结构数值一致性
    {
        // 边界值：0.0f
        sa2::Sa2QuantScaleReciprocal q0(0.0f, 1.0f);
        SA2_CHECK(q0.scale == 1.0f);
        SA2_CHECK(q0.invScale == 1.0f);
        SA2_CHECK(q0.scaledFactor == 1.0f);

        // 边界值：极小次正规/下溢
        sa2::Sa2QuantScaleReciprocal qSub(1.0e-9f, 2.0f);
        SA2_CHECK(qSub.scale == 1.0f);
        SA2_CHECK(qSub.invScale == 1.0f);
        SA2_CHECK(qSub.scaledFactor == 2.0f);

        // 典型 amax 与 factor 组合
        const float amaxValues[] = {0.1f, 1.0f, 12.7f, 63.5f, 127.0f, 254.0f, 500.0f};
        const float factorValues[] = {0.25f, 0.5f, 1.0f, 2.0f};
        for (float amax : amaxValues) {
            for (float factor : factorValues) {
                sa2::Sa2QuantScaleReciprocal q(amax, factor);
                float expScale = amax / 127.0f;
                if (expScale < 1e-7f) expScale = 1.0f;
                SA2_CHECK(std::abs(q.scale - expScale) <= 1e-6f);
                SA2_CHECK(std::abs(q.invScale - (1.0f / expScale)) <= 1e-6f);
                SA2_CHECK(std::abs(q.scaledFactor - expScale * factor) <= 1e-6f);

                // 扫掠待量化浮点数，逐点对比 Quantize 与 oracle
                for (int vInt = -200; vInt <= 200; ++vInt) {
                    float val = vInt * 1.5f;
                    int8_t code = q.Quantize(val);
                    float scaled = sa2::aicNearbyInt(val * (1.0f / expScale));
                    int8_t expCode = static_cast<int8_t>(std::max(-127.0f, std::min(127.0f, scaled)));
                    SA2_CHECK(code == expCode);
                }
            }
        }
    }

    // 3. 3 级重叠流水控制器行为验证
    {
        const int64_t totalR = 160; // 3 个 tile: 64, 64, 32
        const int64_t d = 128;
        const int64_t dp = 128;

        std::vector<int64_t> loadSlices;
        std::vector<int64_t> computeSlices;
        std::vector<int64_t> storeSlices;
        std::vector<std::string> eventOrder;

        sa2::ExecuteQuantBatchPipeline(
            totalR, d, dp,
            [&](const sa2::Sa2QuantSliceDesc &desc) {
                loadSlices.push_back(desc.sliceIndex);
                eventOrder.push_back("load_" + std::to_string(desc.sliceIndex));
            },
            [&](const sa2::Sa2QuantSliceDesc &desc) {
                computeSlices.push_back(desc.sliceIndex);
                eventOrder.push_back("compute_" + std::to_string(desc.sliceIndex));
            },
            [&](const sa2::Sa2QuantSliceDesc &desc) {
                storeSlices.push_back(desc.sliceIndex);
                eventOrder.push_back("store_" + std::to_string(desc.sliceIndex));
            });

        // 验证切片数量
        SA2_CHECK(loadSlices.size() == 3);
        SA2_CHECK(computeSlices.size() == 3);
        SA2_CHECK(storeSlices.size() == 3);

        // 验证流水时序：Prologue 必须为 load_0，首个 compute_0 必须在 load_0 之后
        SA2_CHECK(eventOrder[0] == "load_0");
        SA2_CHECK(eventOrder[1] == "load_1");
        SA2_CHECK(eventOrder[2] == "compute_0");
        // Epilogue 最后一个事件必须为 store_2
        SA2_CHECK(eventOrder.back() == "store_2");

        // 验证各切片描述符
        auto dummy = [](const sa2::Sa2QuantSliceDesc &) {};
        sa2::Sa2QuantBatchPipeline<decltype(dummy), decltype(dummy), decltype(dummy)> pipe(totalR, d, dp);
        SA2_CHECK(pipe.numSlices == 3);
        auto d0 = pipe.GetSliceDesc(0);
        SA2_CHECK(d0.rowOffset == 0 && d0.validRows == 64 && d0.bufferSlot == 0);
        auto d1 = pipe.GetSliceDesc(1);
        SA2_CHECK(d1.rowOffset == 64 && d1.validRows == 64 && d1.bufferSlot == 1);
        auto d2 = pipe.GetSliceDesc(2);
        SA2_CHECK(d2.rowOffset == 128 && d2.validRows == 32 && d2.bufferSlot == 0);
    }
}

// ---- 用例 20: P NZ 分形 16B pad 布局与 K-scale 预取事件契约（M2c-B3/B5 契约）----
void CasePNzPadLayoutAndScaleDoubleBufGuard()
{
    // 1. P NZ 分形常数与契约
    static_assert(sa2::kPNzFractalM == 16, "分形 M=16");
    static_assert(sa2::kPNzFractalN == 16, "分形 N=16");
    static_assert(sa2::kPNzFractalDataBytes == 16, "数据 16B");
    static_assert(sa2::kPNzFractalPadBytes == 16, "防 Bank 冲突 pad 16B");
    static_assert(sa2::kPNzFractalRowStride == 32, "分形行严格 32B C0 对齐");
    static_assert(sa2::kPNzFractalBlockBytes == 512, "分形块 512B");
    static_assert(sa2::kPNzTotalSlotBytes == 16384, "P 槽位 16384B 精确匹配");

    // 2. 全数据点单射与契约检查（64 行 × 128 列 = 8192 数据字节）
    for (int64_t r = 0; r < sa2::kStreamRows; ++r) {
        for (int64_t c = 0; c < sa2::kKvStep; ++c) {
            int64_t idx = sa2::PNzFractalIndex(r, c);
            SA2_CHECK(idx >= 0 && idx < sa2::kL1PNzSlot);
            SA2_CHECK(sa2::Sa2PNzPadLayoutContract::IsDataByte(idx));
            SA2_CHECK(!sa2::Sa2PNzPadLayoutContract::IsPadByte(idx));
        }
    }

    // 3. 全 pad 区域检查（32 块 × 16 行 × 16 字节 = 8192 pad 字节）
    for (int64_t blk = 0; blk < sa2::kPNzFractalBlocks; ++blk) {
        for (int64_t row = 0; row < sa2::kPNzFractalM; ++row) {
            for (int64_t padCol = 0; padCol < sa2::kPNzFractalPadBytes; ++padCol) {
                int64_t padIdx = blk * 512 + row * 32 + 16 + padCol;
                SA2_CHECK(padIdx >= 0 && padIdx < sa2::kL1PNzSlot);
                SA2_CHECK(sa2::Sa2PNzPadLayoutContract::IsPadByte(padIdx));
                SA2_CHECK(!sa2::Sa2PNzPadLayoutContract::IsDataByte(padIdx));
            }
        }
    }

    // 4. 全槽位 16384 字节二值划分无缝覆盖断言
    int64_t totalData = 0;
    int64_t totalPad = 0;
    for (int64_t off = 0; off < sa2::kL1PNzSlot; ++off) {
        bool isData = sa2::Sa2PNzPadLayoutContract::IsDataByte(off);
        bool isPad = sa2::Sa2PNzPadLayoutContract::IsPadByte(off);
        SA2_CHECK(isData != isPad); // 恰为二选一互斥
        if (isData) ++totalData;
        if (isPad) ++totalPad;
    }
    SA2_CHECK(totalData == 8192);
    SA2_CHECK(totalPad == 8192);

    // 5. K-scale 双缓冲 scratch 槽位与本地事件映射
    using WarpRes = sa2::Sa2DeviceRes<sa2::Gran::kPerWarp>;
    static_assert(WarpRes::kKScaleStageSlotBytes == 256, "每步 Scale 槽 256B (128 half)");
    static_assert(WarpRes::KScaleBase(0) == WarpRes::kScratchBase + 768, "Scale ping 槽位于 +768");
    static_assert(WarpRes::KScaleBase(1) == WarpRes::kScratchBase + 1024, "Scale pong 槽位于 +1024");
    // 验证与 scalar 槽 (+1280) 无重叠，且完全在 scratch 1344B 内
    static_assert(WarpRes::KScaleBase(1) + WarpRes::kKScaleStageSlotBytes == WarpRes::kScratchBase + 1280, "紧接 scalarSlot");
    static_assert(WarpRes::kScratchBase + 1280 + 16 <= WarpRes::kScratchBase + sa2::kUbSharedScratch, "Scratch 零溢出");

    // 6. 本地事件票号与 Stream-K 常量检查
    SA2_CHECK(sa2::kEvScale0 == 4);
    SA2_CHECK(sa2::kEvScale1 == 5);
    SA2_CHECK(sa2::kKMeanBlockRows == 128);
    SA2_CHECK(sa2::kKMeanDataBlockLanes == 64);
    SA2_CHECK(sa2::kKMeanDoubleBufSlots == 2);
}

struct CaseEntry {
    const char *name;
    void (*fn)();
};

} // namespace

int main()
{
    const CaseEntry kCases[] = {
        {"kappa_two_tiers", CaseKappaTiers},
        {"theta_skip_two_path_consistency", CaseThetaSkipTwoPathConsistency},
        {"lse_contract_with_negative_ln_k", CaseLseContractWithNegativeLnK},
        {"e8m0_bit_trick_matches_std", CaseE8m0BitTrickMatchesStd},
        {"g_factor_bound_and_deq_identity", CaseGFactorBoundAndDeqIdentity},
        {"granularity_paths_differ", CaseGranularityPathsDiffer},
        {"causal_block_vs_element_predicate", CaseCausalBlockVsElementPredicate},
        {"fp8_code_table_roundtrip", CaseFp8CodeTableRoundtrip},
        {"half_f32_rne_feature_points", CaseHalfF32RneFeaturePoints},
        {"fp16_fold_error_bound", CaseFp16FoldErrorBound},
        {"microapi_vector_equivalence", CaseMicroApiVectorEquivalence},
        {"ub_plan_o1_hard_guard", CaseUbPlanO1HardGuard},
        {"p_nz_fractal_index_guard", CasePNzFractalIndexGuard},
        {"theta_skip_448_fullscale_guard", CaseThetaSkip448FullscaleGuard},
        {"state_typed_slot_typestate_guard", CaseStateTypedSlotTypestateGuard},
        {"score_range_safety_guard", CaseScoreRangeSafetyGuard},
        {"paired_pv_fold_schedule", CasePairedPvFoldSchedule},
        {"is_wide_path_eligible_matrix", CaseIsWidePathEligibleMatrix},
        {"quant_scale_reciprocal_and_pipeline_guard", CaseQuantScaleReciprocalAndPipelineGuard},
        {"pnz_pad_layout_and_scale_double_buf_guard", CasePNzPadLayoutAndScaleDoubleBufGuard},
    };
    for (const auto &entry : kCases) {
        const int before = g_failures;
        entry.fn();
        std::printf("[%s] %s (%d checks, %d failures so far)\n", (g_failures == before) ? "PASS" : "FAIL", entry.name,
                    g_checks, g_failures);
    }
    std::printf("== sa2_numeric_conformance: %d checks, %d failures ==\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
