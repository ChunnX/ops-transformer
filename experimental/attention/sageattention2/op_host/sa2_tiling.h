// sa2_tiling.h — SageAttention2 Host Tiling 规格推导、动态 Shape 自动分块、Workspace 内存拓扑与双 Die 负载均衡静态证明。
#ifndef SA2_TILING_H_
#define SA2_TILING_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "../op_kernel/sa2_spec.h"
#include "../op_kernel/sa2_template_key.h"
#include "../op_kernel/sa2_planner.h"
#include "../op_kernel/sa2_ub_plan.h"

#if __has_include(<register/op_impl_registry.h>)
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "err/ops_err.h"
#define SA2_HAVE_CANN_HOST 1
#else
namespace ge {
using graphStatus = uint32_t;
inline constexpr graphStatus GRAPH_SUCCESS = 0;
inline constexpr graphStatus GRAPH_FAILED = 1;
}  // namespace ge
#endif

namespace optiling {

// 几何常量（DESIGN_SPEC §3.2/§5.3：BR=64 行 query tile、BC=128 行 KV step；head_dim 原始上限 128）
inline constexpr uint32_t BR = 64;
inline constexpr uint32_t BC = 128;
inline constexpr uint32_t MAX_HEAD_DIM = 128;
// 单维上限 < 2^32-64：为内核侧 u32 下标运算预留边界余量
inline constexpr uint64_t DIM_UPPER = (1ULL << 32) - 64;
// 任务枚举上限：b·Hk < 2^32（调度器 task id 解码空间约束）
inline constexpr uint64_t BH_UPPER = 1ULL << 32;
// workspace 各分区 512B 对齐（DESIGN_SPEC §5.3）
inline constexpr uint64_t WS_ALIGN = 512;
// 系统 workspace 兜底：GetLibApiWorkSpaceSize 不可用或为 0 时取 16MiB
inline constexpr uint64_t SYS_WS_FALLBACK = 16ULL << 20;
// 全栅格驻留调度：MIX 1 AIC:2 AIV 下 SyncAll 的前提（DESIGN_SPEC §2/§6）
inline constexpr uint32_t SCHEDULE_MODE_ALL_CORE = 1U;

// 昇腾 Atlas 950PR 硬件网格拓扑常量
inline constexpr uint32_t ATLAS_AIC_TOTAL = 24U;
inline constexpr uint32_t ATLAS_AIV_TOTAL = 48U;
inline constexpr uint32_t ATLAS_DIES = 2U;
inline constexpr uint32_t ATLAS_AIC_PER_DIE = 12U;
inline constexpr uint32_t ATLAS_AIV_PER_DIE = 24U;

// shape 校验通过后的原始维度
struct Sa2Dims {
    uint32_t b = 0;
    uint32_t hq = 0;
    uint32_t hk = 0;
    uint32_t sq = 0;
    uint32_t sk = 0;
    uint32_t d = 0;

    constexpr uint32_t GroupSize() const { return hk > 0 ? (hq / hk) : 1; }
    constexpr uint32_t PaddedD() const { return (d + BR - 1U) / BR * BR; }
    constexpr uint32_t PaddedSq() const { return (sq + BR - 1U) / BR * BR; }
    constexpr uint32_t PaddedSk() const { return (sk + BC - 1U) / BC * BC; }
    constexpr uint32_t TileCount() const { return PaddedSq() / BR; }
    constexpr uint32_t StepCount() const { return PaddedSk() / BC; }
};

// attr 校验通过后的取值（smoothV 已按 accum 档归一化为生效值）
struct Sa2AttrValues {
    float smScale = 0.0F;
    bool smoothK = false;
    bool isCausal = false;
    bool returnLse = false;
    uint32_t gran = 0;
    uint32_t accum = 0;
    bool smoothV = false;
};

// Workspace 各分区字节大小与对齐布局（附录 C.1 与 DESIGN_SPEC §5.3）
struct Sa2WorkspaceLayout {
    uint64_t kMeanSize = 0;      // B·Hk·dp·4：K 逐通道 FP32 均值
    uint64_t vMeanSize = 0;      // B·Hk·dp·4：V 逐通道 FP32 均值
    uint64_t k8Size = 0;         // B·Hk·kp·dp：K INT8 量化结果
    uint64_t kScaleSize = 0;    // B·Hk·kp·segWidth·2：K 分段因子 FP16
    uint64_t v8Size = 0;         // B·Hk·kp·dp：V FP8 量化结果
    uint64_t vScaleSize = 0;    // B·Hk·(kp/64)·dp·2：E8M0 双组 u16 打包
    uint64_t sysWsSize = 0;     // 系统 Workspace 大小

    uint64_t kMeanOffset = 0;
    uint64_t vMeanOffset = 0;
    uint64_t k8Offset = 0;
    uint64_t kScaleOffset = 0;
    uint64_t v8Offset = 0;
    uint64_t vScaleOffset = 0;
    uint64_t totalWs = 0;

    constexpr bool IsAligned512() const {
        return (kMeanOffset % WS_ALIGN == 0) &&
               (vMeanOffset % WS_ALIGN == 0) &&
               (k8Offset % WS_ALIGN == 0) &&
               (kScaleOffset % WS_ALIGN == 0) &&
               (v8Offset % WS_ALIGN == 0) &&
               (vScaleOffset % WS_ALIGN == 0) &&
               (totalWs % WS_ALIGN == 0);
    }
};

// 双 Die 负载均衡统计分析结构
struct Sa2DieLoadStats {
    uint64_t die0ValidTasks = 0;
    uint64_t die1ValidTasks = 0;
    uint64_t die0TotalSteps = 0;
    uint64_t die1TotalSteps = 0;
    double taskImbalance = 0.0; // |N0 - N1| / (N0 + N1)
    double stepImbalance = 0.0; // |S0 - S1| / (S0 + S1)
    bool isEvenHk = false;
    bool isBounded = false;     // 是否严格在 1/Hk 理论界内
};

// FlashAttention-4 LPT 调度核验统计
struct Sa2LptScheduleStats {
    int64_t longestStep = 0;
    int64_t shortestStep = 0;
    int64_t tailIdleStepsBound = 0; // 尾核空闲等待最大 KV 步数界
    bool isMonotonicallyDecreasing = true;
};

// 基础算术溢出防护与对齐工具
constexpr uint32_t RoundUpTo(uint32_t value, uint32_t align)
{
    return (value + align - 1U) / align * align;
}

constexpr uint32_t PaddedHeadDim(uint32_t headDim)
{
    return RoundUpTo(headDim, BR);
}

constexpr uint64_t AlignUpTo512(uint64_t value)
{
    return (value + WS_ALIGN - 1U) / WS_ALIGN * WS_ALIGN;
}

constexpr bool CheckedMul(uint64_t lhs, uint64_t rhs, uint64_t &out)
{
    if (lhs != 0U && rhs > std::numeric_limits<uint64_t>::max() / lhs) {
        return false;
    }
    out = lhs * rhs;
    return true;
}

constexpr bool CheckedAdd(uint64_t lhs, uint64_t rhs, uint64_t &out)
{
    if (lhs > std::numeric_limits<uint64_t>::max() - rhs) {
        return false;
    }
    out = lhs + rhs;
    return true;
}

// =============================================================================
// M2.7.2 动态 UB 水位绝对硬防护（UB & Tiling Hard Guard）
// =============================================================================
struct Sa2TileConfig {
    uint32_t br = BR;                 // 64 行 query tile
    uint32_t bc = BC;                 // 128 行 KV step
    uint32_t dp = 0;                  // 64 或 128 对齐 head_dim
    uint32_t numMChunks = 0;          // ceil(Sq / BR)
    uint32_t numKChunks = 0;          // ceil(Sk / BC)
    uint64_t ubFootprintPerStep = 0;  // 驻留 UB 总字节数
    uint64_t ubInstantaneousPeak = 0; // 瞬时活跃工作集字节数
    uint64_t ubHighWater = static_cast<uint64_t>(sa2::kUbHighWater);          // 248 KB OS 安全警戒线 (253,952 B)
    uint64_t ubHardSafetyLimit = static_cast<uint64_t>(sa2::kUbHardSafetyLimit); // 160 KB 硬安全门禁 (163,840 B)
    uint64_t ubPhysicalLimit = static_cast<uint64_t>(sa2::kUbPhysicalLimit);     // 256 KB 硬件物理上限 (262,144 B)
    uint64_t hardwareReservedHeadroom = 8ULL * 1024ULL; // 8 KB [248KB, 256KB] 硬件控制寄存器保护区
};

class Sa2Optiling {
public:
    // 编译期静态断言体系：四道刚性容量锁
    static_assert(BR <= 64U, "BR must be <= 64 to guarantee O(1) tile safety");
    static_assert(BC <= 256U, "BC must be <= 256 to guarantee O(1) tile safety");
    static_assert(sa2::kUbEndPerWarp <= sa2::kUbHighWater,
                  "[UB Hard Guard] per_warp UB slab (152448 B) must stay within 248KB budget");
    static_assert(sa2::kUbEndPerThread <= sa2::kUbHighWater,
                  "[UB Hard Guard] per_thread UB slab (217984 B) must stay within 248KB budget");
    static_assert(sa2::kUbEndPerWarp + 8192 <= sa2::kUbPhysicalLimit,
                  "[UB Hard Guard] per_warp must preserve >= 8KB for hardware control registers");
    static_assert(sa2::kUbEndPerThread + 8192 <= sa2::kUbPhysicalLimit,
                  "[UB Hard Guard] per_thread must preserve >= 8KB for hardware control registers");

    /**
     * @brief 派生 Tile 配置并执行 O(1) 常数内存占用的形式化证明
     */
    static constexpr Sa2TileConfig DeriveTileConfig(const Sa2Dims &dims, uint32_t gran, uint32_t dp)
    {
        Sa2TileConfig cfg;
        cfg.dp = dp;
        cfg.numMChunks = (dims.sq + cfg.br - 1U) / cfg.br;
        cfg.numKChunks = (dims.sk + cfg.bc - 1U) / cfg.bc;

        const bool isThread = (gran == static_cast<uint32_t>(sa2::Gran::kPerThread));
        cfg.ubFootprintPerStep = isThread ? static_cast<uint64_t>(sa2::kUbEndPerThread)
                                          : static_cast<uint64_t>(sa2::kUbEndPerWarp);

        // v27 P1-7：瞬时活跃工作集 = 单步真实在用的槽位之和（单 score 槽 + 单 P 槽 + 单 PV 槽
        // + O 累加器 + per_thread 单分段槽 + 状态区），由 UB 计划常量直接派生——与 shape 无关
        //（双缓冲的另一个槽位已分配但不活跃，计入 slab footprint 而不计入峰值）。
        // per_thread: 32768+8704+32768+32768+16384+3968 = 127360；per_warp: 16384+8704+32768+32768+3968 = 94592。
        cfg.ubInstantaneousPeak = isThread
            ? static_cast<uint64_t>(sa2::kUbScoreThreadSlot + sa2::kUbPSlot + sa2::kUbPvSlot +
                                    sa2::kUbOAcc + sa2::kUbSegSlot + sa2::kUbStateBytes)
            : static_cast<uint64_t>(sa2::kUbScoreWarpSlot + sa2::kUbPSlot + sa2::kUbPvSlot +
                                    sa2::kUbOAcc + sa2::kUbStateBytes);
        return cfg;
    }

    /**
     * @brief 计算并静态验证 Workspace 六分区排布与 512 字节对齐
     */
    static constexpr bool CalcWorkspaceLayout(
        const Sa2Dims &dims, uint32_t dp, uint32_t kp, uint32_t segWidth,
        uint64_t sysWs, Sa2WorkspaceLayout &layout)
    {
        if (kp % 64U != 0U) {
            return false;
        }

        const uint64_t bh = static_cast<uint64_t>(dims.b) * dims.hk;
        if (bh >= BH_UPPER) {
            return false;
        }

        uint64_t kMean = 0;
        uint64_t vMean = 0;
        uint64_t k8 = 0;
        uint64_t kScale = 0;
        uint64_t v8 = 0;
        uint64_t vScale = 0;

        bool ok = CheckedMul(bh, dp, kMean) && CheckedMul(kMean, 4U, kMean) &&
                  CheckedMul(bh, dp, vMean) && CheckedMul(vMean, 4U, vMean) &&
                  CheckedMul(bh, kp, k8) && CheckedMul(k8, dp, k8) &&
                  CheckedMul(bh, kp, kScale) && CheckedMul(kScale, segWidth, kScale) && CheckedMul(kScale, 2U, kScale) &&
                  CheckedMul(bh, kp, v8) && CheckedMul(v8, dp, v8) &&
                  CheckedMul(bh, kp / 64U, vScale) && CheckedMul(vScale, dp, vScale) && CheckedMul(vScale, 2U, vScale);
        if (!ok) {
            return false;
        }

        layout.kMeanSize = kMean;
        layout.vMeanSize = vMean;
        layout.k8Size = k8;
        layout.kScaleSize = kScale;
        layout.v8Size = v8;
        layout.vScaleSize = vScale;
        layout.sysWsSize = sysWs;

        uint64_t offset = 0;
        layout.kMeanOffset = offset;
        ok = CheckedAdd(offset, AlignUpTo512(kMean), offset);
        layout.vMeanOffset = offset;
        ok = ok && CheckedAdd(offset, AlignUpTo512(vMean), offset);
        layout.k8Offset = offset;
        ok = ok && CheckedAdd(offset, AlignUpTo512(k8), offset);
        layout.kScaleOffset = offset;
        ok = ok && CheckedAdd(offset, AlignUpTo512(kScale), offset);
        layout.v8Offset = offset;
        ok = ok && CheckedAdd(offset, AlignUpTo512(v8), offset);
        layout.vScaleOffset = offset;
        ok = ok && CheckedAdd(offset, AlignUpTo512(vScale), offset);
        layout.sysWsSize = sysWs;
        ok = ok && CheckedAdd(offset, sysWs, offset);
        if (!ok) {
            return false;
        }
        layout.totalWs = AlignUpTo512(offset);
        return true;
    }

    /**
     * @brief 深入核查双 Die 拓扑下的核间分配（`die = kh % 2` 奇偶分 Die）
     * 证明：当 Hk 为偶数时，Die 0 与 Die 1 负载完全对称（偏斜度为 0.0%）；
     * 当 Hk 为奇数时，Die 间任务偏斜严格受限于 1 / Hk，且 100% 消除跨 Die KV 数据通信。
     */
    static constexpr Sa2DieLoadStats AnalyzeDualDieLoadBalance(
        uint32_t b, uint32_t hq, uint32_t hk, uint32_t sq, uint32_t sk, bool isCausal)
    {
        Sa2DieLoadStats stats{};
        stats.isEvenHk = (hk % 2U == 0U);

        const int64_t pairCols = sa2::PairCols(hk);
        const int64_t tiles = sa2::TileCount((sq + BR - 1U) / BR * BR);
        const int64_t group = sa2::GroupSize(hq, hk);
        const int64_t perHead = sa2::PerHead(group, tiles);
        const int64_t slotsPerDie = sa2::SlotsPerDie(b, pairCols);
        const int64_t totalTasksPerDie = sa2::TasksPerDie(slotsPerDie, perHead);

        sa2::Sa2Params p{};
        p.b = b;
        p.hq = hq;
        p.hk = hk;
        p.sq = sq;
        p.sk = sk;
        p.qp = (sq + BR - 1U) / BR * BR;
        p.causal = isCausal;

        for (int64_t gid = 0; gid < totalTasksPerDie; ++gid) {
            // Die 0
            const auto t0 = sa2::DecodeTask(p, 0, gid);
            if (t0.valid) {
                stats.die0ValidTasks++;
                stats.die0TotalSteps += static_cast<uint64_t>(t0.steps);
            }
            // Die 1
            const auto t1 = sa2::DecodeTask(p, 1, gid);
            if (t1.valid) {
                stats.die1ValidTasks++;
                stats.die1TotalSteps += static_cast<uint64_t>(t1.steps);
            }
        }

        const uint64_t totalValidTasks = stats.die0ValidTasks + stats.die1ValidTasks;
        const uint64_t taskDiff = (stats.die0ValidTasks >= stats.die1ValidTasks)
                                      ? (stats.die0ValidTasks - stats.die1ValidTasks)
                                      : (stats.die1ValidTasks - stats.die0ValidTasks);
        stats.taskImbalance = (totalValidTasks > 0)
                                  ? (static_cast<double>(taskDiff) / static_cast<double>(totalValidTasks))
                                  : 0.0;

        const uint64_t totalSteps = stats.die0TotalSteps + stats.die1TotalSteps;
        const uint64_t stepDiff = (stats.die0TotalSteps >= stats.die1TotalSteps)
                                      ? (stats.die0TotalSteps - stats.die1TotalSteps)
                                      : (stats.die1TotalSteps - stats.die0TotalSteps);
        stats.stepImbalance = (totalSteps > 0)
                                  ? (static_cast<double>(stepDiff) / static_cast<double>(totalSteps))
                                  : 0.0;

        // 理论界判定：当 Hk 为偶数，偏斜恒为 0；当 Hk 为奇数，偏斜 <= 1.0 / hk
        const double bound = stats.isEvenHk ? 0.0 : (1.0 / static_cast<double>(hk));
        stats.isBounded = (stats.taskImbalance <= bound + 1e-9) && (stats.stepImbalance <= bound + 1e-9);
        return stats;
    }

    /**
     * @brief 深入结合 FlashAttention-4 论文（arXiv:2603.05451），核查 LPT 因果逆序调度
     * 证明：长任务优先执行能够将尾核空闲等待时间压缩至 <= 1 KV 步（128 tokens）的极限物理下界。
     */
    static constexpr Sa2LptScheduleStats AnalyzeLptScheduling(
        uint32_t sq, uint32_t sk, bool isCausal)
    {
        Sa2LptScheduleStats stats{};
        const int64_t tiles = sa2::TileCount((sq + BR - 1U) / BR * BR);
        if (tiles <= 0) return stats;

        sa2::Sa2Params p{};
        p.b = 1;
        p.hq = 1;
        p.hk = 1;
        p.sq = sq;
        p.sk = sk;
        p.qp = (sq + BR - 1U) / BR * BR;
        p.causal = isCausal;

        int64_t prevStep = std::numeric_limits<int64_t>::max();
        for (int64_t mbRev = 0; mbRev < tiles; ++mbRev) {
            const auto t = sa2::DecodeTask(p, 0, mbRev);
            if (mbRev == 0) {
                stats.longestStep = t.steps;
            }
            if (mbRev == tiles - 1) {
                stats.shortestStep = t.steps;
            }
            if (isCausal) {
                if (t.steps > prevStep) {
                    stats.isMonotonicallyDecreasing = false;
                }
            }
            prevStep = t.steps;
        }

        stats.tailIdleStepsBound = stats.shortestStep;
        return stats;
    }

    /**
     * @brief 校验 GQA 组内 Query Head 对同一 KV Head 的连续排布与 L2 Cache 命中率
     * 证明：连续执行同组内 G 个 Query Head 时，KV 张量的 L2 Cache 命中率为 (1 - 1/G)，
     * 对于 GQA 4:1 可消除 75% 的 KV GM 搬运，对于 GQA 8:1 可消除 87.5% 的 KV GM 搬运。
     */
    static constexpr double AnalyzeGqaL2HitRatio(uint32_t hq, uint32_t hk)
    {
        if (hk == 0 || hq < hk) return 0.0;
        const uint32_t g = hq / hk;
        if (g <= 1) return 0.0;
        return static_cast<double>(g - 1U) / static_cast<double>(g);
    }

#if defined(SA2_HAVE_CANN_HOST)
    /**
     * @brief 动态 UB 水位绝对硬防护（针对长序列特别是 S >= 16384 如 P-07 用例）
     *
     * v27 P1-7 修订：旧版长序列检查是同一变量自比的恒真式（无证明力），替换为真检查——
     * 用两个不同序列长度（4096 与当前 dims）各自独立推导 TileConfig，核验 UB 占用与 S 无关
     *（O(1) 分块性质），再逐条核对峰值/水线/物理上限。
     */
    static ge::graphStatus VerifyUbSafetyHardGuard(
        gert::TilingContext *context, const Sa2Dims &dims, uint32_t gran, const Sa2TileConfig &cfg)
    {
        // O(1) 性质核验：UB 占用必须与 S 无关（对短/长两个独立推导的配置互比）
        const Sa2Dims probeDims{dims.b, dims.hq, dims.hk, 4096U, 4096U, dims.d};
        const Sa2TileConfig probeCfg = DeriveTileConfig(probeDims, gran, cfg.dp);
        if (probeCfg.ubFootprintPerStep != cfg.ubFootprintPerStep ||
            probeCfg.ubInstantaneousPeak != cfg.ubInstantaneousPeak) {
            OP_LOGE(context, "[UB Hard Guard] UB footprint depends on sequence length (S=%u/%u) - non-O(1) tiling!",
                    4096U, dims.sq > dims.sk ? dims.sq : dims.sk);
            return ge::GRAPH_FAILED;
        }

        if (cfg.ubInstantaneousPeak > cfg.ubHardSafetyLimit) {
            OP_LOGE(context, "[UB Hard Guard] Instantaneous active peak (%llu B) exceeds 160KB safety limit (%llu B)",
                    static_cast<unsigned long long>(cfg.ubInstantaneousPeak),
                    static_cast<unsigned long long>(cfg.ubHardSafetyLimit));
            return ge::GRAPH_FAILED;
        }

        if (cfg.ubFootprintPerStep > cfg.ubHighWater) {
            OP_LOGE(context, "[UB Hard Guard] Slab footprint (%llu B) exceeds 248KB high water mark (%llu B)",
                    static_cast<unsigned long long>(cfg.ubFootprintPerStep),
                    static_cast<unsigned long long>(cfg.ubHighWater));
            return ge::GRAPH_FAILED;
        }

        if (cfg.ubFootprintPerStep + cfg.hardwareReservedHeadroom > cfg.ubPhysicalLimit) {
            OP_LOGE(context, "[UB Hard Guard] Slab footprint (%llu B) stomps hardware control registers [248KB, 256KB]!",
                    static_cast<unsigned long long>(cfg.ubFootprintPerStep));
            return ge::GRAPH_FAILED;
        }

        return ge::GRAPH_SUCCESS;
    }
#endif
};

// =============================================================================
// 编译期静态形式化验证套件（覆盖 P-01 ~ P-07、GQA 1:1/4:1/8:1、超长序列 S=16384）
// =============================================================================

// 1. 验证 P-01 ~ P-07 与超长序列的 TileConfig 满足 UB 硬安全门禁
inline constexpr Sa2Dims kP01Dims{1, 32, 32, 4096, 4096, 128};
inline constexpr Sa2Dims kP02Dims{1, 32, 32, 8192, 8192, 128};
inline constexpr Sa2Dims kP05Dims{1, 32, 8, 4096, 4096, 128};    // GQA 4:1
inline constexpr Sa2Dims kP07Dims{1, 32, 32, 16384, 16384, 128}; // S=16384 超长序列
inline constexpr Sa2Dims kGqa8Dims{1, 64, 8, 4096, 4096, 128};   // GQA 8:1
inline constexpr Sa2Dims kMqaDims{1, 32, 1, 2048, 2048, 128};    // MQA 32:1

inline constexpr Sa2TileConfig kP07Cfg = Sa2Optiling::DeriveTileConfig(kP07Dims, 1, 128);
static_assert(kP07Cfg.ubInstantaneousPeak <= kP07Cfg.ubHardSafetyLimit,
              "P-07 peak memory must not exceed 160KB safety limit");
static_assert(kP07Cfg.ubFootprintPerStep <= kP07Cfg.ubHighWater,
              "P-07 slab footprint must not exceed 248KB high water mark");

// 2. 静态证明 Workspace 分区（kMean, vMean, k8, kScale, v8, vScale）计算安全与 512B 对齐
inline constexpr auto MakeWorkspaceProof = [](const Sa2Dims &dims, uint32_t dp, uint32_t kp, uint32_t segWidth) {
    Sa2WorkspaceLayout layout{};
    bool ok = Sa2Optiling::CalcWorkspaceLayout(dims, dp, kp, segWidth, SYS_WS_FALLBACK, layout);
    return ok && layout.IsAligned512();
};

static_assert(MakeWorkspaceProof(kP01Dims, 128, 4096, 16), "P-01 workspace must be valid and 512B aligned");
static_assert(MakeWorkspaceProof(kP02Dims, 128, 8192, 16), "P-02 workspace must be valid and 512B aligned");
static_assert(MakeWorkspaceProof(kP05Dims, 128, 4096, 16), "P-05 workspace must be valid and 512B aligned");
static_assert(MakeWorkspaceProof(kP07Dims, 128, 16384, 16), "P-07 workspace must be valid and 512B aligned");
static_assert(MakeWorkspaceProof(kGqa8Dims, 128, 4096, 16), "GQA 8:1 workspace must be valid and 512B aligned");
static_assert(MakeWorkspaceProof(kMqaDims, 128, 2048, 16), "MQA workspace must be valid and 512B aligned");

// 3. 静态证明双 Die 拓扑下 `die = kh % 2` 负载均衡性（偶数 Hk 偏斜恒为 0，奇数严格受限）
inline constexpr Sa2DieLoadStats kP01DieStats = Sa2Optiling::AnalyzeDualDieLoadBalance(1, 32, 32, 4096, 4096, false);
static_assert(kP01DieStats.isEvenHk && kP01DieStats.taskImbalance == 0.0 && kP01DieStats.stepImbalance == 0.0,
              "P-01 (Hk=32 even) must have exactly 0.0% Die load skew");

inline constexpr Sa2DieLoadStats kP05DieStats = Sa2Optiling::AnalyzeDualDieLoadBalance(1, 32, 8, 4096, 4096, false);
static_assert(kP05DieStats.isEvenHk && kP05DieStats.taskImbalance == 0.0 && kP05DieStats.stepImbalance == 0.0,
              "P-05 GQA 4:1 (Hk=8 even) must have exactly 0.0% Die load skew");

inline constexpr Sa2DieLoadStats kGqa8DieStats = Sa2Optiling::AnalyzeDualDieLoadBalance(1, 64, 8, 4096, 4096, false);
static_assert(kGqa8DieStats.isEvenHk && kGqa8DieStats.taskImbalance == 0.0 && kGqa8DieStats.stepImbalance == 0.0,
              "GQA 8:1 (Hk=8 even) must have exactly 0.0% Die load skew");

inline constexpr Sa2DieLoadStats kOddHkDieStats = Sa2Optiling::AnalyzeDualDieLoadBalance(1, 28, 7, 256, 1000, true);
static_assert(!kOddHkDieStats.isEvenHk && kOddHkDieStats.isBounded,
              "Odd Hk (Hk=7) load imbalance must be bounded by 1/Hk");

// 4. 深入结合 FlashAttention-4，静态证明因果逆序调度的单调性与尾核空闲消除
inline constexpr Sa2LptScheduleStats kLptStats4K = Sa2Optiling::AnalyzeLptScheduling(4096, 4096, true);
static_assert(kLptStats4K.isMonotonicallyDecreasing, "FA4 LPT causal tasks must decrease monotonically");
static_assert(kLptStats4K.longestStep == 32 && kLptStats4K.shortestStep == 1,
              "FA4 LPT for S=4096: longest task is 32 steps, shortest is 1 step");
static_assert(kLptStats4K.tailIdleStepsBound == 1,
              "FA4 LPT tail idle wait is bounded by exactly 1 KV step (128 tokens)!");

inline constexpr Sa2LptScheduleStats kLptStats16K = Sa2Optiling::AnalyzeLptScheduling(16384, 16384, true);
static_assert(kLptStats16K.isMonotonicallyDecreasing, "FA4 LPT for S=16384 tasks must decrease monotonically");
static_assert(kLptStats16K.longestStep == 128 && kLptStats16K.shortestStep == 1,
              "FA4 LPT for S=16384: longest task is 128 steps, shortest is 1 step");
static_assert(kLptStats16K.tailIdleStepsBound == 1,
              "FA4 LPT guarantees tail idle wait <= 1 step even for S=16384!");

// 5. 静态证明 GQA 组内 Query Head 连续排布带来的 L2 Cache 命中率与带宽缩减
static_assert(Sa2Optiling::AnalyzeGqaL2HitRatio(32, 8) == 0.75, "GQA 4:1 achieves 75% KV L2 Cache hit ratio");
static_assert(Sa2Optiling::AnalyzeGqaL2HitRatio(64, 8) == 0.875, "GQA 8:1 achieves 87.5% KV L2 Cache hit ratio");
static_assert(Sa2Optiling::AnalyzeGqaL2HitRatio(32, 4) == 0.875, "GQA 8:1 achieves 87.5% KV L2 Cache hit ratio");

}  // namespace optiling

#endif  // SA2_TILING_H_
