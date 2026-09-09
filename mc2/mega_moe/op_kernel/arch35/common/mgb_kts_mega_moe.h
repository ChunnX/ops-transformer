/**
 * mgb_kts_mega_moe.h — mega_moe (arch35 MTE wave) 的 KTS 打点适配层
 *
 * 点位编号一经发布永不复用（host/points_mega_moe.py 与本表同源，解析器按 pointCount 互证）。
 * 边界质量申明（v1 不注入任何新 fence，宁可诚实标 ISSUE 也不扰动被测流水）：
 *   CAUSAL = begin/end 均落在既有完成边界（X_S / 排空等待）之后，含义是「工作真完成」；
 *   ISSUE  = end 是 Scalar 观察/下发时刻，异步流水（MMAD/MTE3）可能尚未收尾。
 *
 * 使用：诊断构建加 -DMGB_KTS_ENABLE=1；默认构建宏未定义，本文件所有宏展开为空、零残留
 * （verify/verify_off.sh 用 cpp 预处理逐 token 证明与基线一致）。
 */

#ifndef MGB_KTS_MEGA_MOE_H
#define MGB_KTS_MEGA_MOE_H

#include "mgb_kts.h"

#if MGB_KTS_ENABLE

namespace MgbKts {

// ---- 点位表（枚举冻结；wave/expert/arg 语义见 host/points_mega_moe.py 同名表）----
enum MmKtsPoint : uint32_t {
    // 固定阶段（每核至多一次）
    MM_INPUT_QUANTIZE = 1,      // AIV：本卡 token 量化写窗口
    MM_ROUTE_SEND = 2,          // AIV：compact route 收集与发送
    MM_SYNC_RESET = 3,          // AIV：flag 区清零
    MM_CROSS_RANK_SYNC_IN = 4,  // AIV：输入侧全卡握手（wait 类）
    MM_COUNT_TABLE_PREP = 5,    // AIV1：count 表搬入+epoch 校验+前缀和
    MM_DEDUP_PREPASS = 6,       // AIV1：去重 rowDesc 建表（本核工作段）
    MM_DEDUP_PREPASS_BARRIER = 7, // AIV1：建表后等全体 AIV1 到齐（wait 类）
    MM_CROSS_RANK_SYNC_OUT = 8, // AIV：输出侧全卡握手（wait 类）
    MM_UNPERMUTE = 9,           // AIV：归属卡槽位累加出 y
    MM_MOE_TAIL_SYNC = 10,      // AIV：MoE 主流水结束后的全核同步（wait 类）
    MM_SHARED_GMM1 = 11,        // 共享专家 GMM1+激活
    MM_SHARED_GMM2 = 12,        // 共享专家 GMM2
    // 分段任务（多次，携带 wave/expert/args）
    MM_DISPATCH_WAVE = 16,      // AIV1：一个 wave 的 DispatchTokenRange；arg0=rangeRows
    MM_GMM1_PROBLEM = 17,       // AIC：一个 wave-slice 的 GMM1 problem（end=ISSUE：MMAD 异步）
    MM_GMM2_PROBLEM = 18,       // AIC：一个 wave-slice 的 GMM2 problem（end=ISSUE：MMAD 异步）
    MM_COMBINE_TILE_RANGE = 19, // AIV1：非量化 per-tile combine 的一个 tile；arg0=mLoc arg1=nLoc
    MM_COMBINE_TILE_WAIT = 20,  // AIV1：等配对 AIC 的 gmmToEpilogueFlag（wait 类）
    MM_DEDUP_MERGE_WAIT = 21,   // AIV1：合并驱动行等成员专家 gmm2Ready（wait 类，去重专属）
    MM_DEDUP_DESC_WAIT = 22,    // AIV1：dispatch 批装载 desc 前等所覆盖 prepass 工作项 ready（wait 类）
};

// 计数器槽位语义（record 头 counters[4]；主要在 AIV1 有效）
// c0 = dispatch 实际拉取行数（PLAIN+FIRST）  c1 = 扇出额外写行数（去重节省的拉取数）
// c2 = dispatch 跳过行数（SKIP）             c3 = combine 合并单发数 / 被合并省掉的发送数低 16 高 16
constexpr uint32_t MM_KTS_CTR_FETCHED = 0U;
constexpr uint32_t MM_KTS_CTR_FANOUT = 1U;
constexpr uint32_t MM_KTS_CTR_SKIPPED = 2U;
constexpr uint32_t MM_KTS_CTR_COMBINE = 3U;

// 每核状态在 mgb_kts.h 的 g_ktsState/g_ktsRecordBase 全局数组里（aicore 禁类类型全局）。

// FNV-1a：配置指纹，防「kernel 与解析脚本不同配置混用」。
__aicore__ inline uint32_t MmKtsConfigHash(const MegaMoeTilingData *t)
{
    uint32_t h = 2166136261U;
    uint32_t vals[8] = {t->bs, t->h, t->hiddenDim, t->topK, t->epWorldSize, t->moeExpertPerRank,
                        static_cast<uint32_t>(t->dedupMode), static_cast<uint32_t>(t->combineQuantMode)};
    for (uint32_t i = 0U; i < 8U; ++i) {
        for (uint32_t b = 0U; b < 4U; ++b) {
            h ^= (vals[i] >> (b * 8U)) & 0xFFU;
            h *= 16777619U;
        }
    }
    return h;
}

// 入口初始化：scratch = workspace 基址 + 布局占用（dispatchRevDataOffset 恒为 0，
// 故 dispatchRevDataPtr 即 workspace 基址；其后是 RESERVED_WORKSPACE_SIZE 50MiB 保留区）。
__aicore__ inline void MmKtsInit(const MegaMoeImpl::Params &params)
{
    const MegaMoeTilingData *tilingData = params.tilingData;
    MegaMoeImpl::WorkspaceLayout layout(tilingData);
    GM_ADDR scratchBase = params.workspaceInfo.dispatchRevDataPtr + layout.workspaceSize;
    uint64_t carrierBytes = static_cast<uint64_t>(tilingData->bs) * tilingData->h * 2U; // y 为 BF16
    uint64_t scratchBytes = 48ULL * 1024U * 1024U; // 保留区 50MiB，留 2MiB 余量
    KtsInit(scratchBase, tilingData->aicNum, carrierBytes, scratchBytes);
}

/*
 * 收尾发布（kernel 最末尾、业务 y 已写完之后调用）：
 * 各核封 record 头 → 全核同步 → 拷入 y → 全核同步 → AIC0 写头 → PipeBarrier → magic。
 * 注意：本轮 y 已被 trace 覆盖，该轮输出不可用于精度比对。
 */
__aicore__ inline void MmKtsFinish(const MegaMoeImpl::Params &params)
{
    const MegaMoeTilingData *tilingData = params.tilingData;
    KtsSealRecord();
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::SyncAll<false>(); // 全核（AIC+AIV）到齐后再动 y
    KtsCopyRecordToCarrier(params.y2GmAddr);
    AscendC::PipeBarrier<PIPE_ALL>();
    AscendC::SyncAll<false>();
    uint32_t opMeta[8] = {tilingData->bs, tilingData->h, tilingData->topK, tilingData->epWorldSize,
                          tilingData->moeExpertPerRank, static_cast<uint32_t>(tilingData->dedupMode),
                          static_cast<uint32_t>(tilingData->combineQuantMode), tilingData->aicNum};
    KtsPublishHeader(params.y2GmAddr, MmKtsConfigHash(tilingData), opMeta,
                     1000U /* A5: 1000 cycles/us */);
}

} // namespace MgbKts

#define MM_KTS_INIT(params) MgbKts::MmKtsInit(params)
#define MM_KTS_FINISH(params) MgbKts::MmKtsFinish(params)
#define MM_KTS_T(var) uint32_t var = MgbKts::KtsNow()
#define MM_KTS_SPAN(point, wave, expert, a0, a1, t0) \
    MgbKts::KtsSpan((point), (wave), (expert), (a0), (a1), (t0), MgbKts::KtsNow())
#define MM_KTS_SPAN_ISSUE(point, wave, expert, a0, a1, t0) \
    MgbKts::KtsSpan((point), (wave), (expert), (a0), (a1), (t0), MgbKts::KtsNow(), \
                    MgbKts::KTS_FLAG_BOUNDARY_ISSUE)
#define MM_KTS_CTR(idx, delta) MgbKts::KtsCounterAdd((idx), (delta))

#else // !MGB_KTS_ENABLE：全部展开为空，零残留

#define MM_KTS_INIT(params)
#define MM_KTS_FINISH(params)
#define MM_KTS_T(var)
#define MM_KTS_SPAN(point, wave, expert, a0, a1, t0)
#define MM_KTS_SPAN_ISSUE(point, wave, expert, a0, a1, t0)
#define MM_KTS_CTR(idx, delta)

#endif // MGB_KTS_ENABLE
#endif // MGB_KTS_MEGA_MOE_H
