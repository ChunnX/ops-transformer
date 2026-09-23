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
 * \file mega_moe_token_dedup.h
 * \brief MTE 通信去重：rowDesc prepass 建表 + 去重版 Token Dispatch。
 *
 * 背景：MTE dispatch 为 pull 模式，接收卡按源 tokenIndex 从远端窗口拉取记录。topK 路由下同一
 * 源 token 常被本卡多个专家选中，基线会把同一远端记录跨片拉 K 次。去重后每个 (srcRank, token)
 * 只拉一次，由组首行从 UB 直接扇出写到组内全部目的行（数据/scale/metaInfo 三区），数值 bit 恒等。
 */

#ifndef MEGA_MOE_TOKEN_DEDUP_H
#define MEGA_MOE_TOKEN_DEDUP_H

#include "../common/mega_moe_utils.h"
#include "../common/mgb_kts_mega_moe.h" // KTS 打点（默认关）

namespace MegaMoeImpl {

using namespace AscendC;

/*
 * 去重 UB 暂存从 dispatch 固定区尾（scratch.dedupUbBaseAddr，512B 对齐）动态起址，总占用 ≤ 64KiB；
 * host 在 dedupMode!=0 时已从 dispatch 自适应预算中预留 DEDUP_DISPATCH_UB_RESERVE_BYTES（72KiB，
 * 含对齐余量），保证任意合法规格下不越 UB。prepass 与 dispatch 批内暂存分时复用同一段
 * （prepass 先于任何 dispatch；dispatch 批内暂存为调用内瞬态、每批重新装载）。
 */
// prepass 布局（相对 dedupUbBaseAddr 的偏移）：route 装载 + 逐 token 计数 + desc 拼装 + 成员表。
constexpr uint32_t DEDUP_DESC_ASM_UB_BYTES = 2U * 1024U;
// dispatch 去重批内布局（相对 dedupUbBaseAddr）：desc 装载区 + 扇出 meta 拼装区 + 活跃条目索引区。
constexpr uint32_t DEDUP_DESC_BATCH_UB_BYTES = 48U * 1024U;
constexpr uint32_t DEDUP_FAN_META_UB_BYTES = 8U * 1024U; // bufferCount(≤6) × topK(≤32) × 32B 上界
constexpr uint32_t DEDUP_ACTIVE_LIST_UB_BYTES = 8U * 1024U;

/*
 * prepass 工作项的 token 切片宽度（prepass 建表与 dispatch 按需等待两侧必须用同一公式）。
 * 三重约束：①UB 容量硬上限（member/count/route 三区装得下）②负载目标 = 工作项数 ≈ 2×AIV1 数
 * （S3 实测 512-token 粗切片时 28 核只有 4 个工作项、24 核围观，prepass max/mean = 7 倍）
 * ③chunkCount ≤ DEDUP_PREPASS_MAX_CHUNKS（flag 区按此上界分配）。
 */
__aicore__ inline uint32_t CalcDedupPrepassChunkTokens(uint32_t numTokens, uint32_t worldSize, uint32_t topK,
                                                       uint32_t aivJobs)
{
    uint32_t chunk = DEDUP_MEMBER_UB_BYTES / (topK * 3U * static_cast<uint32_t>(sizeof(int32_t)));
    chunk = chunk > 512U ? 512U : chunk;
    uint32_t countCapacity = DEDUP_COUNT_UB_BYTES / static_cast<uint32_t>(sizeof(int32_t));
    chunk = chunk > countCapacity ? countCapacity : chunk;
    uint32_t routeCapacity = DEDUP_ROUTE_LOAD_UB_BYTES / static_cast<uint32_t>(sizeof(int32_t));
    chunk = chunk > routeCapacity ? routeCapacity : chunk;
    uint32_t targetItems = aivJobs * 2U;
    if (targetItems > 0U) {
        uint32_t balanced = (numTokens * worldSize + targetItems - 1U) / targetItems;
        chunk = balanced < chunk ? balanced : chunk;
    }
    uint32_t floorChunk = (numTokens + DEDUP_PREPASS_MAX_CHUNKS - 1U) / DEDUP_PREPASS_MAX_CHUNKS;
    chunk = chunk < floorChunk ? floorChunk : chunk;
    return chunk == 0U ? 1U : chunk;
}
// prepass 逐 token 打包状态：低 8 位幸存成员数（<clamp）、次 8 位全量出现数（含被 clamp 裁掉的）、
// 高 8 位全量最小 topkIdx。minK 必须按全量成员算：clamp 裁掉最小 k 成员时，发送槽仍要与归属卡
// 按完整 topkIds 推导的 canonical 槽一致（退化语义与基线截断对齐——被裁行的贡献丢失、幸存行保留）。
constexpr int32_t DEDUP_PACK_INIT = 0x00FF0000;

struct DedupConfig {
    GM_ADDR rowDescPtr = nullptr;      // workspace rowDesc 表基址
    GM_ADDR prepassReadyPtr = nullptr; // prepass 完成标记区（槽0=每核done计数，槽1+item=项ready；随 launch 清零）
    int32_t dedupMode = 0;             // DEDUP_MODE_* 位掩码，0 = 关闭
    uint32_t rowDescStrideInt32 = 0;   // 每行描述符 int32 数（CalcDedupRowDescStrideInt32）
    uint32_t prepassChunkTokens = 1;   // 工作项 token 切片宽（CalcDedupPrepassChunkTokens 单源）
    uint32_t prepassChunkCount = 1;    // 每 srcRank 的 chunk 数 = ceil(numTokens/chunkTokens)
};

__aicore__ inline DedupConfig CreateDedupConfig(const Params &params)
{
    DedupConfig config;
    config.dedupMode = params.tilingData->dedupMode;
    if (config.dedupMode == 0) {
        return config;
    }
    config.rowDescPtr = params.workspaceInfo.dedupRowDescPtr;
    config.prepassReadyPtr = params.workspaceInfo.dedupPrepassReadyPtr;
    config.rowDescStrideInt32 =
        static_cast<uint32_t>(CalcDedupRowDescStrideInt32(static_cast<int64_t>(params.tilingData->topK)));
    const uint32_t numTokens = params.tilingData->numMaxTokensPerRank;
    config.prepassChunkTokens = CalcDedupPrepassChunkTokens(numTokens, params.tilingData->epWorldSize,
                                                            params.tilingData->topK, params.tilingData->aicNum);
    config.prepassChunkCount = (numTokens + config.prepassChunkTokens - 1U) / config.prepassChunkTokens;
    return config;
}

/*
 * 去重 prepass：扫描 route 槽（[expert][srcRank]，槽内按 topkIndex 升序），为每个接收行写一条
 * rowDesc（布局见 mega_moe_constants.h DEDUP_ROW_*）。
 * 分工：工作项 = (srcRank, token 段)，全体 AIV1+AIV0 平铺认领（AIV0 的本地前缀表与 UB 布局由
 * PrepareDedupPrepassAiv0Local 先行就位；工人互斥无交叠，done 计数期望 2×blockNum）。
 * （历史：v3 曾把 prepass 挪到 AIV0——它在等首 wave GMM1 输出，可与 AIV1 dispatch 完全并行；
 * 但实测大 bs 下 AIV0 早期密集访问 route 窗口触发 aicore trap（机制待与驱动侧对齐，2026-09-09
 * 通宵二分 F 系探针铁证：不调 prepass 不崩/调即崩且崩点在公共段），已回退 AIV1 保稳。）
 * 同一 (srcRank, token) 的所有出现恒落在同一工作项内，分段装载流式判界下工作项之间零交叠，无需原子。
 * 依赖：PrepareMoeExpertTokenCountTable 已完成（cumsumInfoTensor 就绪；count epoch 校验到达
 * 即 route 槽到达，是 DispatchRankTokens 沿用的既有契约）。
 * 函数尾部含 AIV1 间栅栏：所有 AIV1 返回后整张 rowDesc 表可读。
 */

/*
 * prepass 单槽扫描体（从 BuildDedupRowDescTable 主循环提取，2026-09-17 双缓冲流水改造）：
 * 对已装载进 UB 半区的一个 route 槽段做判界扫描并更新 count/member 表。语义与提取前逐字等价。
 */
template <typename TopkIndexType>
__aicore__ inline void DedupPrepassScanSlot(LocalTensor<TopkIndexType> &routeSeg, LocalTensor<int32_t> &countTensor,
                                            LocalTensor<int32_t> &memberTensor, int32_t segCount, int32_t rowBase,
                                            int32_t segBase, int32_t tokenLo, int32_t targetLo, int32_t targetHi,
                                            uint32_t topK, uint32_t maxOutputSize, uint32_t expertIdx,
                                            int32_t expertRowBegin, int32_t dispatchFlagSlotCountPerExpert)
{
    for (int32_t entryIdx = 0; entryIdx < segCount; ++entryIdx) {
        const int32_t topkIndex = static_cast<int32_t>(routeSeg.GetValue(entryIdx));
        if (topkIndex < targetLo) {
            continue;
        }
        if (topkIndex >= targetHi) {
            break;
        }
        const int32_t row = rowBase + segBase + entryIdx;
        const int32_t localToken = topkIndex / static_cast<int32_t>(topK) - tokenLo;
        const int32_t k = topkIndex % static_cast<int32_t>(topK);
        int32_t packed = countTensor.GetValue(localToken);
        int32_t survivorCount = packed & 0xFF;
        int32_t fullCount = (packed >> 8) & 0xFF;
        int32_t fullMinK = (packed >> 16) & 0xFF;
        ++fullCount;
        fullMinK = k < fullMinK ? k : fullMinK;
        if (row < static_cast<int32_t>(maxOutputSize)) {
            const int32_t tileIdx = (row - expertRowBegin) / static_cast<int32_t>(L1_TILE_M_256);
            const int32_t flagSlotOffset =
                static_cast<int32_t>(expertIdx) * dispatchFlagSlotCountPerExpert + tileIdx * INT_CACHELINE;
            const uint32_t memberBase =
                static_cast<uint32_t>(localToken) * topK * 3U + static_cast<uint32_t>(survivorCount) * 3U;
            memberTensor.SetValue(memberBase, row);
            memberTensor.SetValue(memberBase + 1U, topkIndex);
            memberTensor.SetValue(memberBase + 2U, flagSlotOffset);
            ++survivorCount;
        }
        countTensor.SetValue(localToken, survivorCount | (fullCount << 8) | (fullMinK << 16));
    }
}

template <typename TopkIndexType, typename DispatchScratch>
__aicore__ inline void BuildDedupRowDescTable(const DedupConfig &dedup, const MoeStageCommonConfig &common,
                                              const BlockJobContext &blockJob,
                                              const MoeSyncWorkspaceLayout &syncLayout, const Params &params,
                                              DispatchScratch &scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (blockJob.totalJobs == 0U) {
        return;
    }
    // 建表工人集按档切换（判据见 DEDUP_AIV0_PREPASS_MAX_HIDDEN_DIM）：AIV0 参与时按
    // AIV1=jobIndex / AIV0=jobIndex+totalJobs 平铺；不参与档 AIV0 直接返回，保持 AIV1 单方建表。
    const bool aiv0Joins = params.tilingData->hiddenDim <= DEDUP_AIV0_PREPASS_MAX_HIDDEN_DIM;
    if (GetSubBlockIdx() != 1U && !aiv0Joins) {
        return;
    }
    const uint32_t workerIndex =
        GetSubBlockIdx() == 1U ? blockJob.jobIndex : blockJob.jobIndex + blockJob.totalJobs;
    const uint32_t totalWorkers = aiv0Joins ? blockJob.totalJobs * 2U : blockJob.totalJobs;

    const uint32_t topK = params.tilingData->topK;
    const uint32_t strideInt32 = dedup.rowDescStrideInt32;
    const uint32_t maxOutputSize = params.tilingData->maxOutputSize;
    const uint64_t routeIndexAlignSize = static_cast<uint64_t>(CalcDispatchRouteIndexAlignSize(params.tilingData));

    // 切片宽/项数以 DedupConfig 缓存值为准（与 dispatch 消费侧同一 CalcDedupPrepassChunkTokens 单源）。
    const uint32_t chunkTokens = dedup.prepassChunkTokens;
    const uint32_t numTokens = params.tilingData->numMaxTokensPerRank;
    const uint32_t chunkCount = dedup.prepassChunkCount;
    const uint32_t totalWorkItems = common.worldSize * chunkCount;
    __gm__ int32_t *readyBase = reinterpret_cast<__gm__ int32_t *>(dedup.prepassReadyPtr);

    const uint32_t routeLoadAddr = scratch.dedupUbBaseAddr;
    const uint32_t countAddr = routeLoadAddr + DEDUP_ROUTE_LOAD_UB_BYTES;
    const uint32_t descAsmAddr = countAddr + DEDUP_COUNT_UB_BYTES;
    const uint32_t memberAddr = descAsmAddr + DEDUP_DESC_ASM_UB_BYTES;
    // route 槽元素类型与基线 dispatch 同源（int16/int32 由 UseInt16TopkIndex 选定）；UB 字节预算不变，条数随元素宽度变。
    LocalTensor<TopkIndexType> routeLoadTensor(TPosition::VECCALC, routeLoadAddr,
                                               DEDUP_ROUTE_LOAD_UB_BYTES / sizeof(TopkIndexType));
    LocalTensor<int32_t> memberTensor(TPosition::VECCALC, memberAddr, DEDUP_MEMBER_UB_BYTES / sizeof(int32_t));
    LocalTensor<int32_t> countTensor(TPosition::VECCALC, countAddr, DEDUP_COUNT_UB_BYTES / sizeof(int32_t));
    LocalTensor<int32_t> descAsmTensor(TPosition::VECCALC, descAsmAddr, DEDUP_DESC_ASM_UB_BYTES / sizeof(int32_t));

    GlobalTensor<int32_t> rowDescGm;
    rowDescGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(dedup.rowDescPtr));

    /*
     * desc 落盘乒乓（2026-09-11，KTS 细分实测 DESC 段 33us 占 prepass 52.6us 的 63%——每行一次
     * MTE3_S 全等待暴露 GM 写延迟）：descAsm 区分两半轮转，行 i 的 GM 写与行 i+1 的标量拼装重叠，
     * MTE3_S 等待推迟到该半区下次复用前。半区 1KB ≥ 最宽 desc 行（topK≤32 → 104 int32 = 416B）。
     * 事件配平：半区0 用 SYNC_EVENT_ID3、半区1 用 SYNC_EVENT_ID5；函数头武装两半的 MTE3_S，
     * 每 workItem 尾 drain 两半（AtomicAdd ready 前 desc 必须全落盘——dispatch 靠它）再重新武装，
     * 全部工作项结束后最终 drain 消费掉武装，不留悬挂 flag。
     */
    constexpr uint32_t DESC_ASM_HALF_INT32 = DEDUP_DESC_ASM_UB_BYTES / 2U / sizeof(int32_t);
    SetFlag<AscendC::HardEvent::MTE3_S>(static_cast<TEventID>(SYNC_EVENT_ID3));
    SetFlag<AscendC::HardEvent::MTE3_S>(static_cast<TEventID>(SYNC_EVENT_ID5));
    uint32_t descHalf = 0U;

    MM_KTS_T(ktsPrepassT0);
    for (uint32_t workItem = workerIndex; workItem < totalWorkItems; workItem += totalWorkers) {
        const uint32_t srcRank = workItem / chunkCount;
        const uint32_t chunkIdx = workItem % chunkCount;
        const int32_t tokenLo = static_cast<int32_t>(chunkIdx * chunkTokens);
        int32_t tokenHi = tokenLo + static_cast<int32_t>(chunkTokens);
        if (tokenHi > static_cast<int32_t>(numTokens)) {
            tokenHi = static_cast<int32_t>(numTokens);
        }

        // 上一工作项的标量 count 写先于本项向量清零（S→V），再保证清零结果对标量可见（V→S）。
        SyncFuncStatic<AscendC::HardEvent::S_V, SYNC_EVENT_ID2>();
        Duplicate<int32_t>(countTensor, DEDUP_PACK_INIT, static_cast<int32_t>(chunkTokens));
        SyncFuncStatic<AscendC::HardEvent::V_S, SYNC_EVENT_ID2>();

        MM_KTS_T(ktsScanT0);
        /*
         * 槽间 MTE2 双缓冲流水（2026-09-17，KTS 定位：每工作项对 moeExpertPerRank 个小槽的
         * MTE2 装载与标量扫描完全串行 = prepass 主耗时，B256 实测 73us/核）。
         * routeLoad 分两半（各 ROUTE_HALF_ITEMS）：单段槽（slotCount<=半区）装载槽 e+1 与
         * 扫描槽 e 重叠；大槽（罕见，大 bs）flush 流水后退回原地分段串行。
         * 事件配平：half0=SYNC_EVENT_ID4、half1=SYNC_EVENT_ID6，每半区两方向
         * （S_MTE2=扫描完可复写、MTE2_S=装载完可扫描）；函数头武装两半 S_MTE2，
         * 主循环结束全部回到已武装 S_MTE2 态后统一消费，零悬挂。
         */
        constexpr int32_t ROUTE_HALF_ITEMS =
            static_cast<int32_t>(DEDUP_ROUTE_LOAD_UB_BYTES / 2U / sizeof(TopkIndexType));
        const int32_t targetLo = tokenLo * static_cast<int32_t>(topK);
        const int32_t targetHi = tokenHi * static_cast<int32_t>(topK);
        SetFlag<AscendC::HardEvent::S_MTE2>(static_cast<TEventID>(SYNC_EVENT_ID4));
        SetFlag<AscendC::HardEvent::S_MTE2>(static_cast<TEventID>(SYNC_EVENT_ID6));
        uint32_t loadHalf = 0U;
        int32_t pendExpert = -1;
        int32_t pendRowBase = 0;
        int32_t pendSlotCount = 0;
        int32_t pendExpertRowBegin = 0;
        uint32_t pendHalf = 0U;
        for (uint32_t expertIdx = 0U; expertIdx < common.moeExpertPerRank; ++expertIdx) {
            const uint32_t prefixIdx = expertIdx * common.worldSize + srcRank;
            const int32_t rowBase =
                prefixIdx == 0U ? 0 : scratch.cumsumInfoTensor.GetValue(prefixIdx - 1U);
            const int32_t slotCount = scratch.cumsumInfoTensor.GetValue(prefixIdx) - rowBase;
            if (slotCount <= 0) {
                continue;
            }
            const int32_t expertRowBegin =
                expertIdx == 0U ? 0 : scratch.cumsumInfoTensor.GetValue(expertIdx * common.worldSize - 1U);
            __gm__ TopkIndexType *slotPtr = reinterpret_cast<__gm__ TopkIndexType *>(
                params.peermemInfo.maskRecvPtr + static_cast<uint64_t>(prefixIdx) * routeIndexAlignSize);

            if (slotCount > ROUTE_HALF_ITEMS) {
                // 大槽：flush 流水中的 pending，再原地分段串行（复用 half loadHalf）。
                if (pendExpert >= 0) {
                    const TEventID pevt = static_cast<TEventID>(pendHalf == 0U ? SYNC_EVENT_ID4 : SYNC_EVENT_ID6);
                    WaitFlag<AscendC::HardEvent::MTE2_S>(pevt);
                    LocalTensor<TopkIndexType> pendSeg = routeLoadTensor[pendHalf * static_cast<uint32_t>(ROUTE_HALF_ITEMS)];
                    DedupPrepassScanSlot(pendSeg, countTensor, memberTensor, pendSlotCount, pendRowBase, 0, tokenLo,
                                         targetLo, targetHi, topK, maxOutputSize,
                                         static_cast<uint32_t>(pendExpert), pendExpertRowBegin,
                                         syncLayout.dispatchFlagSlotCountPerExpert);
                    SetFlag<AscendC::HardEvent::S_MTE2>(pevt);
                    pendExpert = -1;
                }
                const TEventID evt = static_cast<TEventID>(loadHalf == 0U ? SYNC_EVENT_ID4 : SYNC_EVENT_ID6);
                LocalTensor<TopkIndexType> seg = routeLoadTensor[loadHalf * static_cast<uint32_t>(ROUTE_HALF_ITEMS)];
                bool reachedEnd = false;
                for (int32_t segBase = 0; segBase < slotCount && !reachedEnd; segBase += ROUTE_HALF_ITEMS) {
                    const int32_t segCount =
                        (slotCount - segBase) < ROUTE_HALF_ITEMS ? (slotCount - segBase) : ROUTE_HALF_ITEMS;
                    GlobalTensor<TopkIndexType> slotGm;
                    slotGm.SetGlobalBuffer(slotPtr + segBase);
                    WaitFlag<AscendC::HardEvent::S_MTE2>(evt);
                    DataCopyExtParams segCopyParams{1U, static_cast<uint32_t>(segCount * sizeof(TopkIndexType)), 0U, 0U,
                                                    0U};
                    DataCopyPadExtParams<TopkIndexType> segCopyPad{false, 0U, 0U, 0U};
                    DataCopyPad(seg, slotGm, segCopyParams, segCopyPad);
                    SetFlag<AscendC::HardEvent::MTE2_S>(evt);
                    WaitFlag<AscendC::HardEvent::MTE2_S>(evt);
                    if (seg.GetValue(static_cast<uint32_t>(segCount - 1)) >= targetLo) {
                        const int32_t lastVal = seg.GetValue(static_cast<uint32_t>(segCount - 1));
                        DedupPrepassScanSlot(seg, countTensor, memberTensor, segCount, rowBase, segBase, tokenLo,
                                             targetLo, targetHi, topK, maxOutputSize, expertIdx, expertRowBegin,
                                             syncLayout.dispatchFlagSlotCountPerExpert);
                        if (lastVal >= targetHi) {
                            reachedEnd = true;
                        }
                    }
                    SetFlag<AscendC::HardEvent::S_MTE2>(evt);
                }
                continue;
            }

            // 单段槽：发起本槽装载（与 pending 槽的扫描重叠）。
            const TEventID levt = static_cast<TEventID>(loadHalf == 0U ? SYNC_EVENT_ID4 : SYNC_EVENT_ID6);
            LocalTensor<TopkIndexType> loadSeg = routeLoadTensor[loadHalf * static_cast<uint32_t>(ROUTE_HALF_ITEMS)];
            GlobalTensor<TopkIndexType> slotGm;
            slotGm.SetGlobalBuffer(slotPtr);
            WaitFlag<AscendC::HardEvent::S_MTE2>(levt);
            DataCopyExtParams slotCopyParams{1U, static_cast<uint32_t>(slotCount * sizeof(TopkIndexType)), 0U, 0U, 0U};
            DataCopyPadExtParams<TopkIndexType> slotCopyPad{false, 0U, 0U, 0U};
            DataCopyPad(loadSeg, slotGm, slotCopyParams, slotCopyPad);
            SetFlag<AscendC::HardEvent::MTE2_S>(levt);
            if (pendExpert >= 0) {
                const TEventID pevt = static_cast<TEventID>(pendHalf == 0U ? SYNC_EVENT_ID4 : SYNC_EVENT_ID6);
                WaitFlag<AscendC::HardEvent::MTE2_S>(pevt);
                LocalTensor<TopkIndexType> pendSeg = routeLoadTensor[pendHalf * static_cast<uint32_t>(ROUTE_HALF_ITEMS)];
                DedupPrepassScanSlot(pendSeg, countTensor, memberTensor, pendSlotCount, pendRowBase, 0, tokenLo,
                                     targetLo, targetHi, topK, maxOutputSize,
                                     static_cast<uint32_t>(pendExpert), pendExpertRowBegin,
                                     syncLayout.dispatchFlagSlotCountPerExpert);
                SetFlag<AscendC::HardEvent::S_MTE2>(pevt);
            }
            pendExpert = static_cast<int32_t>(expertIdx);
            pendRowBase = rowBase;
            pendSlotCount = slotCount;
            pendExpertRowBegin = expertRowBegin;
            pendHalf = loadHalf;
            loadHalf ^= 1U;
        }
        if (pendExpert >= 0) {
            const TEventID pevt = static_cast<TEventID>(pendHalf == 0U ? SYNC_EVENT_ID4 : SYNC_EVENT_ID6);
            WaitFlag<AscendC::HardEvent::MTE2_S>(pevt);
            LocalTensor<TopkIndexType> pendSeg = routeLoadTensor[pendHalf * static_cast<uint32_t>(ROUTE_HALF_ITEMS)];
            DedupPrepassScanSlot(pendSeg, countTensor, memberTensor, pendSlotCount, pendRowBase, 0, tokenLo,
                                 targetLo, targetHi, topK, maxOutputSize,
                                 static_cast<uint32_t>(pendExpert), pendExpertRowBegin,
                                 syncLayout.dispatchFlagSlotCountPerExpert);
            SetFlag<AscendC::HardEvent::S_MTE2>(pevt);
            pendExpert = -1;
        }
        WaitFlag<AscendC::HardEvent::S_MTE2>(static_cast<TEventID>(SYNC_EVENT_ID4));
        WaitFlag<AscendC::HardEvent::S_MTE2>(static_cast<TEventID>(SYNC_EVENT_ID6));

        MM_KTS_SPAN(MgbKts::MM_DEDUP_PREPASS_SCAN, 0U, srcRank, chunkIdx, 0U, ktsScanT0);
        MM_KTS_T(ktsDescT0);
        // 逐 token 落 rowDesc。组成员按扫描序 = 专家升序 = 行号升序，天然满足 FIRST/LAST 语义。
        // fullCount>=2 时即使幸存成员只剩 1 个也按组发（minK 取全量口径），保证发送槽与归属卡
        // canonical 槽一致；fullCount==1 才走 PLAIN。
        for (int32_t localToken = 0; localToken < tokenHi - tokenLo; ++localToken) {
            const int32_t packed = countTensor.GetValue(localToken);
            const int32_t survivorCount = packed & 0xFF;
            const int32_t fullCount = (packed >> 8) & 0xFF;
            const int32_t fullMinK = (packed >> 16) & 0xFF;
            if (survivorCount <= 0) {
                continue;
            }
            const uint32_t memberBase = static_cast<uint32_t>(localToken) * topK * 3U;
            if (fullCount == 1) {
                const int32_t row = memberTensor.GetValue(memberBase);
                const TEventID evt = static_cast<TEventID>(descHalf == 0U ? SYNC_EVENT_ID3 : SYNC_EVENT_ID5);
                LocalTensor<int32_t> asmHalf = descAsmTensor[descHalf * DESC_ASM_HALF_INT32];
                WaitFlag<AscendC::HardEvent::MTE3_S>(evt); // 该半区上次 GM 写已完成，可复写
                asmHalf.SetValue(0U, 0);
                SetFlag<AscendC::HardEvent::S_MTE3>(evt);
                WaitFlag<AscendC::HardEvent::S_MTE3>(evt);
                DataCopy(rowDescGm[static_cast<uint64_t>(row) * strideInt32], asmHalf, INT32_PER_256B);
                SetFlag<AscendC::HardEvent::MTE3_S>(evt); // 写完武装，下次复用本半区时消费
                descHalf ^= 1U;
                continue;
            }
            for (int32_t m = 0; m < survivorCount; ++m) {
                int32_t flags = DEDUP_ROW_FLAG_GROUP | survivorCount;
                flags |= (m == 0 ? DEDUP_ROW_FLAG_FIRST : 0);
                flags |= (m == survivorCount - 1 ? DEDUP_ROW_FLAG_LAST : 0);
                const TEventID evt = static_cast<TEventID>(descHalf == 0U ? SYNC_EVENT_ID3 : SYNC_EVENT_ID5);
                LocalTensor<int32_t> asmHalf = descAsmTensor[descHalf * DESC_ASM_HALF_INT32];
                WaitFlag<AscendC::HardEvent::MTE3_S>(evt);
                asmHalf.SetValue(0U, flags);
                // 同一 token 的各成员行共享同一份 [minK | 成员三元组]，只有 word0 的 FIRST/LAST 位不同。
                // 两个半区严格交替，故本 token 内每个半区只在首次使用（m<2）时拼整表，之后复用只改 word0；
                // 改写前的 WaitFlag 已保证该半区上一次 GM 写完成。标量拼装从每 token s 次降到至多 2 次
                // （KTS 实测 k16/bs3359：rowDesc 落盘段 796us 占 prepass 989us 的 80%，主要耗在这里）。
                if (m < 2) {
                    asmHalf.SetValue(1U, fullMinK);
                    for (int32_t j = 0; j < survivorCount; ++j) {
                        const uint32_t srcBase = memberBase + static_cast<uint32_t>(j) * 3U;
                        const uint32_t dstBase = DEDUP_ROW_DESC_HEADER_INT32 + static_cast<uint32_t>(j) * 3U;
                        asmHalf.SetValue(dstBase, memberTensor.GetValue(srcBase));
                        asmHalf.SetValue(dstBase + 1U, memberTensor.GetValue(srcBase + 1U));
                        asmHalf.SetValue(dstBase + 2U, memberTensor.GetValue(srcBase + 2U));
                    }
                }
                const int32_t row = memberTensor.GetValue(memberBase + static_cast<uint32_t>(m) * 3U);
                SetFlag<AscendC::HardEvent::S_MTE3>(evt);
                WaitFlag<AscendC::HardEvent::S_MTE3>(evt);
                DataCopy(rowDescGm[static_cast<uint64_t>(row) * strideInt32], asmHalf, strideInt32);
                SetFlag<AscendC::HardEvent::MTE3_S>(evt);
                descHalf ^= 1U;
            }
        }

        MM_KTS_SPAN(MgbKts::MM_DEDUP_PREPASS_DESC, 0U, srcRank, chunkIdx, 0U, ktsDescT0);
        // ready 置位前 drain 两半区（乒乓让末尾 1-2 行的 GM 写还在飞，而 ready 语义=本项 desc
        // 全部落盘、dispatch 消费侧靠它），随即重新武装供下一工作项。
        WaitFlag<AscendC::HardEvent::MTE3_S>(static_cast<TEventID>(SYNC_EVENT_ID3));
        WaitFlag<AscendC::HardEvent::MTE3_S>(static_cast<TEventID>(SYNC_EVENT_ID5));
        SetFlag<AscendC::HardEvent::MTE3_S>(static_cast<TEventID>(SYNC_EVENT_ID3));
        SetFlag<AscendC::HardEvent::MTE3_S>(static_cast<TEventID>(SYNC_EVENT_ID5));
        AtomicAdd(readyBase + (1U + workItem) * INT_CACHELINE, static_cast<int32_t>(1));
    }
    // 消费掉最后一次武装，不留悬挂 flag 毒化后续同 id 同步。
    WaitFlag<AscendC::HardEvent::MTE3_S>(static_cast<TEventID>(SYNC_EVENT_ID3));
    WaitFlag<AscendC::HardEvent::MTE3_S>(static_cast<TEventID>(SYNC_EVENT_ID5));

    MM_KTS_SPAN(MgbKts::MM_DEDUP_PREPASS, 0U, 0U, totalWorkItems, chunkTokens, ktsPrepassT0);
    // 槽 0 = done 计数，AIV1 计低 16 位、AIV0 计高 16 位（dispatch 发布 gate 只等高位=全体 AIV0，
    // AIV1 之间不互等——每个 AIV1 发布前自己必然已建完，等全体 2N 会把 AIV1 拴死、打断
    // prepass 与 dispatch 的流水交叠，B8192 实测劣化 ~200us）；combine 兜底等全值。本核不等任何人。
    AtomicAdd(readyBase, GetSubBlockIdx() == 1U ? static_cast<int32_t>(1) : static_cast<int32_t>(1) << 16);
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_TOKEN_DEDUP_H
