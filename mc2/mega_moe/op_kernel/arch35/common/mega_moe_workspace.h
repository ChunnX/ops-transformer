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
 * \file mega_moe_workspace.h
 * \brief
 */

#ifndef MEGA_MOE_WORKSPACE_H
#define MEGA_MOE_WORKSPACE_H

// peermem 窗口的布局与尺寸（含 HOST_DEVICE / GM_ADDR 宏与基础 include）。
#include "mega_moe_peermem.h"

namespace MegaMoeImpl {

constexpr int64_t EXCEPTION_DUMP_REGION_SIZE = 60 * 1024LL; // 异常dump区
constexpr int64_t SIZE_INT_8 = 1U;
constexpr int64_t SIZE_INT_32 = 4U;
constexpr int64_t SIZE_BF_16 = 2U;

struct WorkspaceInfo {
    GM_ADDR dispatchRevDataPtr;
    GM_ADDR dispatchRevScalePtr;
    GM_ADDR dispatchRevWeightsPtr{nullptr};
    GM_ADDR activationQuantDataPtr;
    GM_ADDR activationQuantScalePtr;
    GM_ADDR expertRevTokenNumsPtr;
    GM_ADDR metaInfoPtr;
    GM_ADDR flagActivationToGmm2Ptr;
    GM_ADDR flagDispatchToGmm1Ptr;
    GM_ADDR flagSendCntCalToUpdParamsPtr;
    GM_ADDR flagGmmToEpiloguePtr{nullptr};
    GM_ADDR gmm2ReadyPtr{nullptr};
    GM_ADDR gmm2CombineSyncCounterPtr{nullptr};
    GM_ADDR cumsumInfoPtr{nullptr};
    GM_ADDR gmm1MmadResPtr{nullptr};
    GM_ADDR gmm2MmadResPtr{nullptr};
    GM_ADDR sharedExpertResultPtr{nullptr};
    GM_ADDR sharedExpertGmm1OutPtr{nullptr};
    GM_ADDR sharedExpertInputDataPtr{nullptr};
    GM_ADDR sharedExpertInputScalePtr{nullptr};
    GM_ADDR sharedExpertActivationDataPtr{nullptr};
    GM_ADDR sharedExpertActivationScalePtr{nullptr};
    GM_ADDR gmm1TileStatusPtr{nullptr}; // GMM1 tile 就绪状态位区（仅 prefetch 软同步分配）
    GM_ADDR sharedExpertGmm2TileCounterPtr{nullptr};

    GM_ADDR maskSlotPtr{nullptr};       // urma发送mask临时GM
    GM_ADDR dispatchRelaySendQueuePtr{nullptr};       // 按目标 Server 划分的一级中继发送队列
    GM_ADDR dispatchRemoteReadyFlagSnapshotPtr{nullptr}; // 各逻辑核的固定 256-token 远端就绪标志窗口

    // 连续 flag 通知区（自 flagActivationToGmm2Ptr 起）的 int32 元素总数。
    // 在分配处顺手记账，保证 ResetSyncStatus 的清零范围与分配范围恒同源。
    int64_t flagResetElementCount{0};
    // prefetch 软同步 GMM1 tile 状态区的 int32 元素总数（未分配时为 0）。
    int64_t gmm1TileStatusElementCount{0};

    int64_t workspaceSize;
    HOST_DEVICE WorkspaceInfo() = default;
    HOST_DEVICE WorkspaceInfo(GM_ADDR base, const MegaMoeTilingData *tilingData)
    {
        uint32_t rankNumPerServer = tilingData->rankNumPerServer == 0U ?
                                        tilingData->epWorldSize : tilingData->rankNumPerServer;
        uint32_t serverNum = Ops::Base::CeilDiv(tilingData->epWorldSize, rankNumPerServer);
        InitializeWorkspaceLayout(base, tilingData, serverNum);
    }

private:
    HOST_DEVICE void InitializeWorkspaceLayout(GM_ADDR base, const MegaMoeTilingData *tilingData, uint32_t serverNum)
    {
        workspaceSize = 0;
        dispatchRevDataPtr = base;
        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->maxOutputSize * tilingData->h, ALIGN_512);
        dispatchRevScalePtr = base + workspaceSize;

        int64_t dispatchScaleElementsPerToken =
            Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->h), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
            MXFP_MULTI_BASE_SIZE;
        workspaceSize +=
            Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->maxOutputSize * dispatchScaleElementsPerToken, ALIGN_512);

        if (tilingData->topkWeightsPrefetch == 1 && tilingData->topoType == TOPO_TYPE_URMA) {
            dispatchRevWeightsPtr = base + workspaceSize;
            uint32_t weightAlignBytes = Ops::Base::CeilAlign(static_cast<uint32_t>(tilingData->topK * sizeof(float)),
                                                             static_cast<uint32_t>(ALIGN_32));
            workspaceSize += Ops::Base::CeilAlign(
                static_cast<int64_t>(tilingData->maxOutputSize) * static_cast<int64_t>(weightAlignBytes), ALIGN_512);
        }

        activationQuantDataPtr = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(
            SIZE_INT_8 * tilingData->maxOutputSize * tilingData->hiddenDim / ACTIVATION_N_HALF, ALIGN_512);

        activationQuantScalePtr = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(
            SIZE_INT_8 * tilingData->maxOutputSize *
                Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->hiddenDim / ACTIVATION_N_HALF),
                                   static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
                MXFP_MULTI_BASE_SIZE,
            ALIGN_512);

        expertRevTokenNumsPtr = base + workspaceSize;
        int64_t expertMajorPaddedCountBytes =
            static_cast<int64_t>(tilingData->moeExpertPerRank) * ALIGN_32 * tilingData->aicNum;
        int64_t blockMajorCountStride = static_cast<int64_t>(
            Ops::Base::CeilAlign(tilingData->moeExpertPerRank, static_cast<uint32_t>(INT_CACHELINE)));
        int64_t blockMajorCountBytes =
            blockMajorCountStride * static_cast<int64_t>(sizeof(int32_t)) * tilingData->aicNum;
        int64_t expertCountWorkspaceBytes =
            tilingData->topoType == TOPO_TYPE_MTE ? blockMajorCountBytes : expertMajorPaddedCountBytes;
        workspaceSize += Ops::Base::CeilAlign(expertCountWorkspaceBytes, ALIGN_512);

        metaInfoPtr = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(tilingData->maxOutputSize * ALIGN_32, ALIGN_512);

        // 以下三组 flag 仅由 MoE 专家使用；共享专家路径不使用这些 flag。
        const bool useMteWaveCombine = tilingData->topoType == TOPO_TYPE_MTE;
        int64_t maxWavesPerExpert =
            Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->maxOutputSize), static_cast<int64_t>(L1_TILE_M_256));
        int64_t waveFlagSlotsPerExpert = maxWavesPerExpert * static_cast<int64_t>(INT_CACHELINE);
        // MTE GMM2 按 256-token M group 消费 activation；URMA 只使用每个专家的首槽。
        int64_t activationFlagSlotsPerExpert =
            tilingData->topoType == TOPO_TYPE_MTE ? waveFlagSlotsPerExpert : static_cast<int64_t>(INT_CACHELINE);
        int64_t moeExpertCount = static_cast<int64_t>(tilingData->moeExpertPerRank);

        // 所有 Scalar 通知都放在同一段连续 workspace 中。每个逻辑槽独占一个 64B cache line，
        // 因而 ResetFlagList 可以按任务分区一次清理完整区域。
        int64_t flagRegionBeginOffset = workspaceSize;
        flagActivationToGmm2Ptr = base + workspaceSize;
        workspaceSize += SIZE_INT_32 * moeExpertCount * activationFlagSlotsPerExpert;
        flagDispatchToGmm1Ptr = base + workspaceSize;
        workspaceSize += SIZE_INT_32 * moeExpertCount * waveFlagSlotsPerExpert;

        // 每(expert, aiCore)单独占一个cache_line
        flagSendCntCalToUpdParamsPtr = base + workspaceSize;
        workspaceSize += SIZE_INT_32 * INT_CACHELINE * moeExpertCount * tilingData->aicNum;

        // 每个 AIC 的就绪序列与前面的 flag 连续存放，使 ResetFlagList 能用同一次 MTE reset 清理；
        // 每个序列独占一个 64B cache line。
        // W4 GMM1 activation 始终使用该序号；非量化 GMM2/Combine 也复用它做 tile 一对一通知。
        bool isW4Mode = tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4 ||
                        tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4 ||
                        tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ;
        if (isW4Mode || (tilingData->topoType == TOPO_TYPE_MTE && tilingData->combineQuantMode == COMBINE_NO_QUANT)) {
            flagGmmToEpiloguePtr = base + workspaceSize;
            workspaceSize += static_cast<int64_t>(tilingData->aicNum) * INT_CACHELINE * SIZE_INT_32;
        }

        if (tilingData->topoType == TOPO_TYPE_MTE && tilingData->combineQuantMode != COMBINE_NO_QUANT) {
            // 量化 Combine 在完整专家结束后等待每个 AIC 的完成标记。
            gmm2ReadyPtr = base + workspaceSize;
            workspaceSize += SIZE_INT_32 * moeExpertCount * tilingData->aicNum * INT_CACHELINE;
        }

        if (tilingData->topoType == TOPO_TYPE_URMA) {
            gmm2CombineSyncCounterPtr = base + workspaceSize;
            workspaceSize += static_cast<int64_t>(tilingData->combineSyncSlotCountPerExpert) * moeExpertCount *
                             INT_CACHELINE * SIZE_INT_32;
        }

        // URMA Layered 共享专家由全核同步收口；仅 MTE 共享专家需要 tile counter。
        if (tilingData->sharedExpertNum > 0 && tilingData->topoType == TOPO_TYPE_MTE) {
            sharedExpertGmm2TileCounterPtr = base + workspaceSize;
            int64_t tokenGroupCount =
                Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->bs), static_cast<int64_t>(L1_TILE_M_256));
            workspaceSize +=
                SIZE_INT_32 * tokenGroupCount * static_cast<int64_t>(tilingData->sharedExpertNum) * INT_CACHELINE;
        }
        flagResetElementCount = (workspaceSize - flagRegionBeginOffset) / SIZE_INT_32;

        // A8W4 / Combine 量化路径的条件 workspace 分配。
        // 以下条件分配与 mega_moe.h 编译期守卫 (ENABLE_A8W4 / ENABLE_A4W4 / CombineQuantMode) 一致，
        // 由 TilingKey 保证同步。
        // W4 Wave-ahead Dispatch 与 layered A8W4 都会跨 Activation 保留 cumsum；Activation 会覆盖对应 UB，
        // 因此需要逐物理 block 的 GM 备份；URMA Layered 的 A8W8/A8W4/A4W4 均使用该 Wave 状态机。
        cumsumInfoPtr = nullptr;
        gmm1MmadResPtr = nullptr;
        gmm2MmadResPtr = nullptr;
        bool activationOverwritesDispatchCumsum =
            tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4 ||
            tilingData->topoType == TOPO_TYPE_URMA ||
            (tilingData->topoType == TOPO_TYPE_MTE && (tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4 ||
                                                       tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ));
        if (activationOverwritesDispatchCumsum) {
            // cumsumInfo：逐核备份 cumsum 状态，每核 moeExpertPerRank × epWorldSize 个 int32。
            cumsumInfoPtr = base + workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(static_cast<int64_t>(SIZE_INT_32 * tilingData->moeExpertPerRank *
                                                                       tilingData->epWorldSize),
                                                  ALIGN_32) *
                             tilingData->aicNum;
        }
        if (tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4 || tilingData->topkWeightsPrefetch == 1) {
            // gmm1MmadRes：GMM1 matmul 输出，布局为 maxOutputSize × hiddenDim 个 BF16。
            gmm1MmadResPtr = base + workspaceSize;
            workspaceSize += SIZE_BF_16 * tilingData->maxOutputSize * tilingData->hiddenDim;
        }
        // 合法拓扑只有 MTE/URMA，两条 Wave 路径都先将 GMM2 输出落到 GM，再由 Combine 消费。
        if (useMteWaveCombine) {
            workspaceSize = Ops::Base::CeilAlign(workspaceSize, static_cast<int64_t>(ALIGN_512));
        }
        gmm2MmadResPtr = base + workspaceSize;
        int64_t gmm2OutputBytes = static_cast<int64_t>(SIZE_BF_16) *
                                  static_cast<int64_t>(tilingData->maxOutputSize) *
                                  static_cast<int64_t>(tilingData->h);
        workspaceSize += useMteWaveCombine ?
                             Ops::Base::CeilAlign(gmm2OutputBytes, static_cast<int64_t>(ALIGN_512)) :
                             gmm2OutputBytes;

        // GMM1 tile 状态位区（仅 prefetch 路径分配，用于 AIC→AIV 软同步）。
        // 每个 tile 状态和末尾 allDone 状态都独占一条 64B cache line。
        gmm1TileStatusPtr = nullptr;
        if (tilingData->topkWeightsPrefetch == 1) {
            gmm1TileStatusPtr = base + workspaceSize;
            int64_t statusSlots =
                static_cast<int64_t>(tilingData->moeExpertPerRank) * tilingData->maxTilesPerExpert + 1;
            gmm1TileStatusElementCount = statusSlots * INT_CACHELINE;
            int64_t statusBytes = SIZE_INT_32 * statusSlots * INT_CACHELINE;
            workspaceSize += Ops::Base::CeilAlign(statusBytes, ALIGN_512);
        }
        if (tilingData->topoType == TOPO_TYPE_URMA) {
            maskSlotPtr = base + workspaceSize;
            // 远端 mask 槽的 count 位于全卡共用的容量尾部，本地发送暂存必须使用相同槽宽。
            const int64_t maskAlignSize = CalcDispatchMaskAlignSize(tilingData);
            int64_t maskSlotSize = maskAlignSize + static_cast<int64_t>(ALIGN_32); // mask + 32B count

            workspaceSize += Ops::Base::CeilAlign(
                static_cast<int64_t>(tilingData->moeExpertPerRank) * tilingData->epWorldSize * maskSlotSize,
                static_cast<int64_t>(ALIGN_512));

            dispatchRelaySendQueuePtr = base + workspaceSize;
            int64_t serverWorkspaceBytes =
                static_cast<int64_t>(ALIGN_32) + static_cast<int64_t>(tilingData->bs) * ALIGN_32;
            workspaceSize += Ops::Base::CeilAlign(static_cast<int64_t>(serverNum) * serverWorkspaceBytes,
                                                  static_cast<int64_t>(ALIGN_512));

            dispatchRemoteReadyFlagSnapshotPtr = base + workspaceSize;
            // 每个逻辑核只缓存当前 256-token 窗口；稀疏 token 按命中的窗口分块读取。
            int64_t flagSnapshotBytes = static_cast<int64_t>(L1_TILE_M_256) * sizeof(uint64_t);
            int64_t relayFlagSnapshotBytesPerBlock =
                Ops::Base::CeilAlign(flagSnapshotBytes, static_cast<int64_t>(ALIGN_512));
            workspaceSize += static_cast<int64_t>(tilingData->aicNum) * relayFlagSnapshotBytesPerBlock;
        }

        // 共享专家 workspace buffer。
        if (tilingData->sharedExpertNum > 0) {
            // sharedExpertResult：共享专家 GMM2 输出 [sharedExpertNum × bs × h]。
            sharedExpertResultPtr = base + workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(
                SIZE_BF_16 * tilingData->bs * tilingData->sharedExpertNum * tilingData->h, ALIGN_512);
            // sharedExpertGmm1Out：共享专家 GMM1 输出 [sharedExpertNum × bs × hiddenDim]。
            sharedExpertGmm1OutPtr = base + workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(
                SIZE_BF_16 * tilingData->bs * tilingData->sharedExpertNum * tilingData->hiddenDim, ALIGN_512);
            // sharedExpertInputData：GMM1 输入数据 [bs × h]。
            sharedExpertInputDataPtr = base + workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->bs * tilingData->h, ALIGN_512);
            // sharedExpertInputScale：GMM1 输入 scale [bs × CeilDiv(h, 32) × 2]。
            sharedExpertInputScalePtr = base + workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->bs *
                                                      Ops::Base::CeilDiv(static_cast<uint32_t>(tilingData->h),
                                                                         static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM)) *
                                                      MXFP_MULTI_BASE_SIZE,
                                                  ALIGN_512);
            // sharedExpertActivationData：SwiGLU 量化输出 [sharedExpertNum × bs × hiddenDim / 2] FP8。
            sharedExpertActivationDataPtr = base + workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(
                SIZE_INT_8 * tilingData->bs * tilingData->sharedExpertNum * tilingData->hiddenDim / ACTIVATION_N_HALF,
                ALIGN_512);
            // sharedExpertActivationScale：每个 MXFP scale group 保存两个 multi-base 分量。
            sharedExpertActivationScalePtr = base + workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(
                SIZE_INT_8 * tilingData->bs * tilingData->sharedExpertNum *
                    Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->hiddenDim / ACTIVATION_N_HALF),
                                       static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
                    MXFP_MULTI_BASE_SIZE,
                ALIGN_512);
        }

    }
};

} // namespace MegaMoeImpl

#endif // MEGA_MOE_WORKSPACE_H
