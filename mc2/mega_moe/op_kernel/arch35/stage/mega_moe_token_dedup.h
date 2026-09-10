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
constexpr uint32_t DEDUP_ROUTE_LOAD_UB_BYTES = 4U * 1024U;
constexpr uint32_t DEDUP_COUNT_UB_BYTES = 4U * 1024U;
constexpr uint32_t DEDUP_DESC_ASM_UB_BYTES = 2U * 1024U;
constexpr uint32_t DEDUP_MEMBER_UB_BYTES = 54U * 1024U;
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

// GM 上有序 int32 数组的 lower_bound：返回首个 >= target 的下标。
__aicore__ inline int32_t DedupLowerBoundGm(__gm__ int32_t *base, int32_t count, int32_t target)
{
    int32_t lo = 0;
    int32_t hi = count;
    while (lo < hi) {
        int32_t mid = lo + (hi - lo) / 2;
        if (ReadGmByPassDCache(base + mid) < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// 在专家级前缀（cumsum 的 (e+1)*worldSize-1 位置）上二分定位 globalRow 所属专家。
__aicore__ inline uint32_t FindDedupExpertByGlobalRow(const LocalTensor<int32_t> &cumsumInfoTensor,
                                                      uint32_t expertCount, uint32_t worldSize, uint32_t globalRow)
{
    uint32_t lo = 0U;
    uint32_t hi = expertCount;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2U;
        uint32_t expertRowEnd = static_cast<uint32_t>(cumsumInfoTensor.GetValue((mid + 1U) * worldSize - 1U));
        if (expertRowEnd <= globalRow) {
            lo = mid + 1U;
        } else {
            hi = mid;
        }
    }
    return lo;
}

/*
 * 去重 prepass：扫描 route 槽（[expert][srcRank]，槽内按 topkIndex 升序），为每个接收行写一条
 * rowDesc（布局见 mega_moe_constants.h DEDUP_ROW_*）。
 * 分工：工作项 = (srcRank, token 段)，全体 AIV1 平铺认领。
 * （历史：v3 曾把 prepass 挪到 AIV0——它在等首 wave GMM1 输出，可与 AIV1 dispatch 完全并行；
 * 但实测大 bs 下 AIV0 早期密集访问 route 窗口触发 aicore trap（机制待与驱动侧对齐，2026-09-09
 * 通宵二分 F 系探针铁证：不调 prepass 不崩/调即崩且崩点在公共段），已回退 AIV1 保稳。）
 * 同一 (srcRank, token) 的所有出现恒落在同一工作项内，槽内二分定界后工作项之间零交叠，无需原子。
 * 依赖：PrepareMoeExpertTokenCountTable 已完成（cumsumInfoTensor 就绪；count epoch 校验到达
 * 即 route 槽到达，是 DispatchRankTokens 沿用的既有契约）。
 * 函数尾部含 AIV1 间栅栏：所有 AIV1 返回后整张 rowDesc 表可读。
 */
template <typename DispatchScratch>
__aicore__ inline void BuildDedupRowDescTable(const DedupConfig &dedup, const MoeStageCommonConfig &common,
                                              const BlockJobContext &blockJob,
                                              const MoeSyncWorkspaceLayout &syncLayout, const Params &params,
                                              DispatchScratch &scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (GetSubBlockIdx() != 1U || blockJob.totalJobs == 0U) {
        return;
    }

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
    LocalTensor<int32_t> routeLoadTensor(TPosition::VECCALC, routeLoadAddr,
                                         DEDUP_ROUTE_LOAD_UB_BYTES / sizeof(int32_t));
    LocalTensor<int32_t> memberTensor(TPosition::VECCALC, memberAddr, DEDUP_MEMBER_UB_BYTES / sizeof(int32_t));
    LocalTensor<int32_t> countTensor(TPosition::VECCALC, countAddr, DEDUP_COUNT_UB_BYTES / sizeof(int32_t));
    LocalTensor<int32_t> descAsmTensor(TPosition::VECCALC, descAsmAddr, DEDUP_DESC_ASM_UB_BYTES / sizeof(int32_t));

    GlobalTensor<int32_t> rowDescGm;
    rowDescGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(dedup.rowDescPtr));

    MM_KTS_T(ktsPrepassT0);
    for (uint32_t workItem = blockJob.jobIndex; workItem < totalWorkItems; workItem += blockJob.totalJobs) {
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
            __gm__ int32_t *slotPtr = reinterpret_cast<__gm__ int32_t *>(
                params.peermemInfo.maskRecvPtr + static_cast<uint64_t>(prefixIdx) * routeIndexAlignSize);

            /*
             * 定界与装载一体化：对槽做分段 MTE2 装载（与基线 dispatch 同款访问方式），UB 内标量
             * 流式判界。**禁对 route 槽做 GM 标量读**——实测（2026-09-08 通宵二分 F2/F3 铁证）
             * 标量密集读大 route 窗口区在大 bs 下触发 aicore 访问异常（count 小区标量读无恙，
             * 基线对 route 槽从来只走 MTE2）；改 MTE2 后与基线访问方式同构。
             * 槽内 topkIndex 升序：段内见 token>=tokenHi 即可早停。
             */
            const int32_t segCap = static_cast<int32_t>(DEDUP_ROUTE_LOAD_UB_BYTES / sizeof(int32_t));
            const int32_t targetLo = tokenLo * static_cast<int32_t>(topK);
            const int32_t targetHi = tokenHi * static_cast<int32_t>(topK);
            bool reachedEnd = false;
            for (int32_t segBase = 0; segBase < slotCount && !reachedEnd; segBase += segCap) {
                const int32_t segCount = (slotCount - segBase) < segCap ? (slotCount - segBase) : segCap;
                GlobalTensor<int32_t> slotGm;
                slotGm.SetGlobalBuffer(slotPtr + segBase);
                SyncFuncStatic<AscendC::HardEvent::S_MTE2, SYNC_EVENT_ID2>();
                DataCopyPad(routeLoadTensor, slotGm,
                            {1U, static_cast<uint32_t>(segCount * sizeof(int32_t)), 0U, 0U, 0U},
                            {false, 0U, 0U, 0U});
                SyncFuncStatic<AscendC::HardEvent::MTE2_S, SYNC_EVENT_ID2>();
                // 段级快跳：段尾 < targetLo 则整段在区间前，跳过（保持升序扫描的低成本）。
                if (routeLoadTensor.GetValue(static_cast<uint32_t>(segCount - 1)) < targetLo) {
                    continue;
                }
            for (int32_t entryIdx = 0; entryIdx < segCount; ++entryIdx) {
                const int32_t topkIndex = routeLoadTensor.GetValue(entryIdx);
                if (topkIndex < targetLo) {
                    continue;
                }
                if (topkIndex >= targetHi) {
                    reachedEnd = true;
                    break;
                }
                const int32_t row = rowBase + segBase + entryIdx;
                const int32_t localToken = topkIndex / static_cast<int32_t>(topK) - tokenLo;
                const int32_t k = topkIndex % static_cast<int32_t>(topK);
                int32_t packed = countTensor.GetValue(localToken);
                int32_t survivorCount = packed & 0xFF;
                int32_t fullCount = (packed >> 8) & 0xFF;
                int32_t fullMinK = (packed >> 16) & 0xFF;
                // 全量口径先更新（含将被 clamp 裁掉的成员），保证 minK 与归属卡 canonical 约定一致。
                ++fullCount;
                fullMinK = k < fullMinK ? k : fullMinK;
                if (row < static_cast<int32_t>(maxOutputSize)) {
                    // dispatch→GMM1 flag 槽偏移在此预计算，扇出发布方无需再依赖 cumsum。
                    const int32_t tileIdx = (row - expertRowBegin) / static_cast<int32_t>(L1_TILE_M_256);
                    const int32_t flagSlotOffset =
                        static_cast<int32_t>(expertIdx) * syncLayout.dispatchFlagSlotCountPerExpert +
                        tileIdx * INT_CACHELINE;
                    const uint32_t memberBase =
                        static_cast<uint32_t>(localToken) * topK * 3U + static_cast<uint32_t>(survivorCount) * 3U;
                    memberTensor.SetValue(memberBase, row);
                    memberTensor.SetValue(memberBase + 1U, topkIndex);
                    memberTensor.SetValue(memberBase + 2U, flagSlotOffset);
                    ++survivorCount;
                }
                countTensor.SetValue(localToken, survivorCount | (fullCount << 8) | (fullMinK << 16));
            }
            } // 段循环收尾
        }

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
                descAsmTensor.SetValue(0U, 0);
                SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();
                DataCopy(rowDescGm[static_cast<uint64_t>(row) * strideInt32], descAsmTensor, INT32_PER_256B);
                SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID3>();
                continue;
            }
            for (int32_t m = 0; m < survivorCount; ++m) {
                int32_t flags = DEDUP_ROW_FLAG_GROUP | survivorCount;
                flags |= (m == 0 ? DEDUP_ROW_FLAG_FIRST : 0);
                flags |= (m == survivorCount - 1 ? DEDUP_ROW_FLAG_LAST : 0);
                descAsmTensor.SetValue(0U, flags);
                descAsmTensor.SetValue(1U, fullMinK);
                for (int32_t j = 0; j < survivorCount; ++j) {
                    const uint32_t srcBase = memberBase + static_cast<uint32_t>(j) * 3U;
                    const uint32_t dstBase = DEDUP_ROW_DESC_HEADER_INT32 + static_cast<uint32_t>(j) * 3U;
                    descAsmTensor.SetValue(dstBase, memberTensor.GetValue(srcBase));
                    descAsmTensor.SetValue(dstBase + 1U, memberTensor.GetValue(srcBase + 1U));
                    descAsmTensor.SetValue(dstBase + 2U, memberTensor.GetValue(srcBase + 2U));
                }
                const int32_t row = memberTensor.GetValue(memberBase + static_cast<uint32_t>(m) * 3U);
                SyncFuncStatic<AscendC::HardEvent::S_MTE3, SYNC_EVENT_ID3>();
                DataCopy(rowDescGm[static_cast<uint64_t>(row) * strideInt32], descAsmTensor, strideInt32);
                SyncFuncStatic<AscendC::HardEvent::MTE3_S, SYNC_EVENT_ID3>();
            }
        }

        // 本工作项 desc 全部落盘（每次 DataCopy 后均有 MTE3_S），置该项 ready：
        // dispatch 消费侧按 (srcRank,chunk) 只等自己覆盖的项，不再有全局 barrier。
        AtomicAdd(readyBase + (1U + workItem) * INT_CACHELINE, static_cast<int32_t>(1));
    }

    MM_KTS_SPAN(MgbKts::MM_DEDUP_PREPASS, 0U, 0U, totalWorkItems, chunkTokens, ktsPrepassT0);
    // 槽 0 = 每核 done 计数：combine 消费 desc 前（远晚于 prepass）等其到齐；本核不等任何人。
    AtomicAdd(readyBase, static_cast<int32_t>(1));
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_TOKEN_DEDUP_H
