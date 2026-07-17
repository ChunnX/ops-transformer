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
 * \file mega_moe_workspace_info.h
 * \brief
 */

#ifndef MEGA_MOE_WORKSPACE_INFO_H
#define MEGA_MOE_WORKSPACE_INFO_H

#if defined(__DAV_C310_CUBE__) || defined(__DAV_C310_VEC__)
#include "kernel_operator.h"
#define HOST_DEVICE __forceinline__[aicore]
#else
#define GM_ADDR uint8_t *
#define HOST_DEVICE
#endif
#include "mega_moe_impl_base.h"

// GMM2 direct-GM readiness A/B modes.  Keep the selection in this common
// host/device header so workspace sizing and kernel addressing cannot drift.
#define MEGA_MOE_GMM2_FANIN_SHARED_ATOMIC 0
#define MEGA_MOE_GMM2_FANIN_PRIVATE_COMPACT 1
#define MEGA_MOE_GMM2_FANIN_PRIVATE_STRIDED 2
#define MEGA_MOE_GMM2_FANIN_ALL_AIV_REDUCE 3
#ifndef MEGA_MOE_GMM2_FANIN_MODE
#define MEGA_MOE_GMM2_FANIN_MODE MEGA_MOE_GMM2_FANIN_ALL_AIV_REDUCE
#endif
#if MEGA_MOE_GMM2_FANIN_MODE != MEGA_MOE_GMM2_FANIN_SHARED_ATOMIC && \
    MEGA_MOE_GMM2_FANIN_MODE != MEGA_MOE_GMM2_FANIN_PRIVATE_COMPACT && \
    MEGA_MOE_GMM2_FANIN_MODE != MEGA_MOE_GMM2_FANIN_PRIVATE_STRIDED && \
    MEGA_MOE_GMM2_FANIN_MODE != MEGA_MOE_GMM2_FANIN_ALL_AIV_REDUCE
#error "invalid MEGA_MOE_GMM2_FANIN_MODE"
#endif

// Number of int32 elements reserved per AIC arrival.  The default 16 keeps
// writers on distinct 64B cache lines.  Set this one value to 1 for the
// compact-layout A/B; the GM layout, reset size and AIV1 reduction adapt
// together without extra compiler options.
#ifndef MEGA_MOE_GMM2_READY_CORE_STRIDE
#define MEGA_MOE_GMM2_READY_CORE_STRIDE 16
#endif
#if MEGA_MOE_GMM2_READY_CORE_STRIDE < 1
#error "MEGA_MOE_GMM2_READY_CORE_STRIDE must be positive"
#endif

namespace {
constexpr uint64_t M_VALUE = 0UL;
constexpr uint64_t N_VALUE = 1UL;
constexpr uint64_t K_VALUE = 2UL;
constexpr uint64_t IDX_A_OFFSET = 0UL;
constexpr uint64_t IDX_B_OFFSET = 1UL;
constexpr uint64_t IDX_A_SCALE_OFFSET = 2UL;
constexpr uint64_t IDX_B_SCALE_OFFSET = 3UL;
constexpr uint64_t IDX_C_OFFSET = 5UL;
constexpr uint64_t IDX_C_SCALE_OFFSET = 6UL;
constexpr uint64_t IDX_FLAG_OFFSET = 7UL;
constexpr uint64_t IDX_B2_OFFSET = 8UL;
constexpr uint64_t IDX_B2_SCALE_OFFSET = 9UL;
constexpr uint64_t IDX_Y2_OFFSET = 10UL;
constexpr uint64_t IDX_M_OFFSET = 11UL;
constexpr uint64_t IDX_GMM1_OFFSET = 12UL;
constexpr uint64_t IDX_GMM2_OFFSET = 13UL;
constexpr uint64_t INT32_PER_256B = 8U;
constexpr uint8_t SYNC_AIC_AIV_MODE = 4;
constexpr uint16_t AIC_SYNC_AIV_FLAG = 4;
constexpr uint16_t AIV_SYNC_AIC_FLAG = 6;
constexpr uint16_t AIC_SYNC_AIV_EPILOGUE_FLAG = 8;
constexpr uint16_t AIV1_SYNC_AIC_EPILOGUE_ACK_FLAG = 9;
constexpr uint16_t FLAG_ID_MAX_PER_V = 16;
constexpr uint32_t SWIGLU_N_HALF = 2U;
constexpr int32_t MXFP_DIVISOR_SIZE = 64;
constexpr int32_t MXFP_SCALE_GROUP_NUM = 32;
constexpr int32_t MXFP_MULTI_BASE_SIZE = 2;
constexpr int64_t PEERMEM_DATA_OFFSET = 1024 * 60LL;
constexpr int64_t SIMT_THREAD_NUM = 2048;
constexpr int64_t ALIGN_32 = 32LL;
constexpr int64_t ALIGN_128 = 128LL;
constexpr int64_t ALIGN_256 = 256LL;
constexpr int64_t ALIGN_512 = 512LL;
// Keep independent GM synchronization words 512B apart.  Several producers
// may update one counter, while unrelated counters must not share the same
// HBM/L2 sector.
constexpr int32_t INT_CACHELINE = ALIGN_512 / static_cast<int32_t>(sizeof(int32_t));
static_assert(INT_CACHELINE * static_cast<int32_t>(sizeof(int32_t)) == ALIGN_512,
              "GM flag stride must be 512B");
constexpr int32_t MAX_AICORE_NUM = 36;
// wave-grain dispatch flag: must match MegaMoeImpl::L1_TILE_M
constexpr int64_t DISPATCH_WAVE_TILE_M = 256LL;
// GMM2 tile size for token group calculation
constexpr int64_t L1_TILE_M_128 = 128LL;
constexpr uint32_t BUFFER_ALIGN = 96U * 1024U * 2U;
constexpr uint32_t HCCL_MAX_RANK_SIZE = 1024U;
constexpr uint32_t UNPERMUTE_LIST_NUM = 3U;
constexpr uint32_t DOUBLE_BUFFER = 2U;
constexpr uint32_t TWO_FLAG = 2U;
constexpr uint32_t RANK_ID = 0U;
constexpr uint32_t TOKEN_ID = 1U;
constexpr uint32_t TOPK_INDEX = 2U;
constexpr uint32_t SYNC_EVENT_ID1 = 1;
constexpr uint32_t SYNC_EVENT_ID2 = 2;
constexpr uint32_t SYNC_EVENT_ID3 = 3;
constexpr uint32_t SYNC_EVENT_ID4 = 4;
constexpr uint32_t SYNC_EVENT_ID5 = 5;
constexpr int64_t SIZE_INT_8 = 1U;
constexpr int64_t SIZE_INT_32 = 4U;
constexpr int64_t SIZE_BF_16 = 2U;
constexpr int64_t E5M2_QUANT = 3U;
constexpr int64_t E4M3_QUANT = 4U;
constexpr int64_t E2M1_QUANT = 5U;
constexpr int64_t HALF_TO_FP32 = 2U;
constexpr int64_t OVERFLOW_MODE_CTRL = 60U;
// Combine quantization modes
constexpr uint8_t COMBINE_NO_QUANT = 0;
constexpr uint8_t MXFP8_E5M2_COMM_QUANT = 3;
constexpr uint8_t MXFP8_E4M3_COMM_QUANT = 4;
// Combine buffer constants
constexpr uint32_t TRIPLE_SIZE = 8U; // 每个 token 的三元组大小（8 个 int32）
// GroupedMatmul modes
constexpr uint8_t GROUPED_MATMUL_MODE_GENERAL = 0U;
constexpr uint8_t GROUPED_MATMUL_MODE_A8W4 = 1U;
constexpr uint8_t GROUPED_MATMUL_MODE_A8W8_NZ = 2U;
// a4w4 混合场景：GMM1 走 generic a4w4，GMM2 走 A8W4。GMM2 需要 gmm2MmadResPtr workspace。
constexpr uint8_t GROUPED_MATMUL_MODE_A4W4 = 3U;
// a4w4 NZ 场景：weight1/weight2 均为 fp4 NZ 格式。
constexpr uint8_t GROUPED_MATMUL_MODE_A4W4_NZ = 4U;
constexpr int64_t TOPO_TYPE_MTE = 0U;  // mte
constexpr int64_t TOPO_TYPE_URMA = 1U; // urma

} // namespace

struct WorkspaceInfo {
    GM_ADDR dispatchRevDataPtr;
    GM_ADDR dispatchRevScalePtr;
    GM_ADDR swigluQuantDataPtr;
    GM_ADDR swigluQuantScalePtr;
    GM_ADDR expertRevTokenNumsPtr;
    GM_ADDR tripleInfoPtr;
    GM_ADDR flagSwiGluToGmm2Ptr;
    GM_ADDR flagDispatchToGmm1Ptr;
    GM_ADDR flagSendCntCalToUpdParamsPtr;
    GM_ADDR cumsumInfoPtr;
    GM_ADDR gmm1MmadResPtr;
    GM_ADDR gmm2MmadResPtr;
    GM_ADDR gmm2CombineSyncCounterPtr;
    GM_ADDR gmm2NoQuantReadyPtr;
    GM_ADDR gmm2NoQuantReleasePtr;

    GM_ADDR maskSlotPtr;          // urma发送mask临时GM
    GM_ADDR dispatchL1CommPtr;    // dispatch L1 communication workspace
    GM_ADDR dispatchCursorPtr;    // dispatch cnt for each expert
    GM_ADDR dispatchDonePtr;      // dispatch done
    GM_ADDR dispatchL2CommPtr;    // dispatch l2 communication workspace
    GM_ADDR combineCommNotifyPtr; // 合并发送通知临时GM
    GM_ADDR combineCommDataPtr;   // combine communication workspace

    int64_t workspaceSize;
    HOST_DEVICE WorkspaceInfo() = default;
    HOST_DEVICE WorkspaceInfo(GM_ADDR base, const MegaMoeTilingData *tilingData, uint32_t serverNum = 1)
    {
        workspaceSize = 0;
        dispatchRevDataPtr = base;

        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->maxOutputSize * tilingData->h, ALIGN_512);
        dispatchRevScalePtr = base + workspaceSize;

        workspaceSize += Ops::Base::CeilAlign(
            SIZE_INT_8 * tilingData->maxOutputSize * tilingData->h / MXFP_SCALE_GROUP_NUM, ALIGN_512);

        swigluQuantDataPtr = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(
            SIZE_INT_8 * tilingData->maxOutputSize * tilingData->hiddenDim / SWIGLU_N_HALF, ALIGN_512);

        swigluQuantScalePtr = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(SIZE_INT_8 * tilingData->maxOutputSize * tilingData->hiddenDim /
                                                  SWIGLU_N_HALF / MXFP_SCALE_GROUP_NUM,
                                              ALIGN_512);

        expertRevTokenNumsPtr = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(tilingData->expertPerRank * ALIGN_32 * tilingData->aicNum, ALIGN_512);

        tripleInfoPtr = base + workspaceSize;
        workspaceSize += Ops::Base::CeilAlign(tilingData->maxOutputSize * ALIGN_32, ALIGN_512);

        flagSwiGluToGmm2Ptr = base + workspaceSize;
        int64_t maxWavesPerExpert =
            Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->maxOutputSize), DISPATCH_WAVE_TILE_M);
        // Every wave owns one 512B-spaced counter.  The value is an int32
        // element count, not a count of logical flags.
        int64_t waveFlagSlotsPerExpert = maxWavesPerExpert * static_cast<int64_t>(INT_CACHELINE);
        // Interleaved GMM1 lets GMM2 consume each completed 256-row SwiGLU
        // wave. Reserve one isolated counter per wave.
        workspaceSize += SIZE_INT_32 * tilingData->expertPerRank * waveFlagSlotsPerExpert;

        flagDispatchToGmm1Ptr = base + workspaceSize;
        // Dispatch readiness already uses the same wave-grain layout.
        workspaceSize += SIZE_INT_32 * tilingData->expertPerRank * waveFlagSlotsPerExpert;

        // 每(expert, aiCore)单独占一个cache_line
        flagSendCntCalToUpdParamsPtr = base + workspaceSize;
        workspaceSize += SIZE_INT_32 * INT_CACHELINE * tilingData->expertPerRank * tilingData->aicNum;

        // Conditional allocation for A8W4 / combine-quant paths.
        // 以下条件分配与 mega_moe.h 编译期守卫 (ENABLE_A8W4 / ENABLE_A4W4 / CombineQuantMode) 一致，
        // 由 TilingKey 保证同步。
        // A8W4-only: cumsum GM backup and GMM1 intermediate result.
        cumsumInfoPtr = nullptr;
        gmm1MmadResPtr = nullptr;
        gmm2MmadResPtr = nullptr;
        gmm2NoQuantReadyPtr = nullptr;
        gmm2NoQuantReleasePtr = nullptr;
        if (tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4) {
            // cumsumInfo: per-core backup of cumsum state (expertPerRank × epWorldSize int32 per core).
            cumsumInfoPtr = base + workspaceSize;
            workspaceSize +=
                Ops::Base::CeilAlign(
                    static_cast<int64_t>(SIZE_INT_32 * tilingData->expertPerRank * tilingData->epWorldSize), ALIGN_32) *
                tilingData->aicNum;
            // gmm1MmadRes: GMM1 matmul output (maxOutputSize × hiddenDim bf16).
            gmm1MmadResPtr = base + workspaceSize;
            workspaceSize += SIZE_BF_16 * tilingData->maxOutputSize * tilingData->hiddenDim;
        }
        // gmm2MmadRes: GMM2 matmul output (maxOutputSize × h bf16), needed by A8W4 GMM1 path,
        // A4W4 hybrid path (GMM2 uses A8W4), A4W4_NZ, and combine-quant.
        bool isGenericGmm = tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_GENERAL ||
                            tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W8_NZ;
        bool useMteNoQuantGmm2Gm = tilingData->topoType == TOPO_TYPE_MTE && isGenericGmm &&
                                   tilingData->combineQuantMode == COMBINE_NO_QUANT;
        if (tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A8W4 ||
            tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4 ||
            tilingData->groupedMatmulMode == GROUPED_MATMUL_MODE_A4W4_NZ ||
            tilingData->combineQuantMode != COMBINE_NO_QUANT || useMteNoQuantGmm2Gm) {
            if (useMteNoQuantGmm2Gm) {
                workspaceSize = Ops::Base::CeilAlign(workspaceSize, static_cast<int64_t>(ALIGN_512));
            }
            gmm2MmadResPtr = base + workspaceSize;
            // The MTE no-quant path gives every routed row a unique [H] BF16
            // region for the complete kernel lifetime.  It is not a batch or
            // ring slot, so AIC never needs a consumer ACK before advancing.
            int64_t gmm2OutputBytes = static_cast<int64_t>(SIZE_BF_16) *
                                      static_cast<int64_t>(tilingData->maxOutputSize) *
                                      static_cast<int64_t>(tilingData->h);
            workspaceSize += useMteNoQuantGmm2Gm ?
                                 Ops::Base::CeilAlign(gmm2OutputBytes, static_cast<int64_t>(ALIGN_512)) :
                                 gmm2OutputBytes;
        }

        if (useMteNoQuantGmm2Gm) {
            gmm2NoQuantReadyPtr = base + workspaceSize;
            // slotIdx is expertIdx in PER_EXPERT mode and batchIdx in
            // PER_BATCH mode.  The all-AIV reduction A/B uses a configurable
            // int32 stride per AIC and keeps each generation 64B aligned.
            int64_t readyStrideBytes = ALIGN_512;
#if MEGA_MOE_GMM2_FANIN_MODE == MEGA_MOE_GMM2_FANIN_PRIVATE_STRIDED
            readyStrideBytes *= static_cast<int64_t>(tilingData->aicNum);
#elif MEGA_MOE_GMM2_FANIN_MODE == MEGA_MOE_GMM2_FANIN_ALL_AIV_REDUCE
            readyStrideBytes = Ops::Base::CeilAlign(static_cast<int64_t>(tilingData->aicNum) *
                                                        static_cast<int64_t>(MEGA_MOE_GMM2_READY_CORE_STRIDE) *
                                                        static_cast<int64_t>(SIZE_INT_32),
                                                    static_cast<int64_t>(64));
#endif
            int64_t readyBytes = static_cast<int64_t>(tilingData->expertPerRank) * readyStrideBytes;
            workspaceSize += Ops::Base::CeilAlign(readyBytes, static_cast<int64_t>(ALIGN_512));

#if MEGA_MOE_GMM2_FANIN_MODE != MEGA_MOE_GMM2_FANIN_ALL_AIV_REDUCE
            // One private 512B release line for every (slot, logical AIV1).
            // The coordinator publishes the whole slot with one MTE3 copy.
            gmm2NoQuantReleasePtr = base + workspaceSize;
            int64_t releaseBytes = static_cast<int64_t>(tilingData->expertPerRank) *
                                   static_cast<int64_t>(tilingData->aicNum) * ALIGN_512;
            workspaceSize += Ops::Base::CeilAlign(releaseBytes, static_cast<int64_t>(ALIGN_512));
#endif
        }

        // Combine-quant-only workspace buffers
        gmm2CombineSyncCounterPtr = nullptr;
        if (tilingData->combineQuantMode != COMBINE_NO_QUANT) {
            // Token group completion counters; every independently polled
            // counter occupies one 512B-spaced slot.
            gmm2CombineSyncCounterPtr = base + workspaceSize;
            // 每个 expert 分配 blockAivNum 个 512B slot。
            int64_t stridePerExpert = Ops::Base::CeilAlign(
                static_cast<int64_t>(tilingData->blockAivNum) * INT_CACHELINE * SIZE_INT_32, ALIGN_128);
            workspaceSize += stridePerExpert * tilingData->expertPerRank;
        }

        if (tilingData->topoType == TOPO_TYPE_URMA) {
            maskSlotPtr = base + workspaceSize;
            int64_t sendTotalNum = static_cast<int64_t>(tilingData->bs) * tilingData->topK;
            int64_t compareCount = Ops::Base::CeilAlign(sendTotalNum * (int64_t)sizeof(int32_t), (int64_t)ALIGN_256) /
                                   (int64_t)sizeof(int32_t);
            int64_t maskAlignSize = Ops::Base::CeilAlign(compareCount / 8, (int64_t)ALIGN_32);
            int64_t maskSlotSize = maskAlignSize + (int64_t)ALIGN_32; // mask + 32B count

            workspaceSize += Ops::Base::CeilAlign(
                (int64_t)tilingData->expertPerRank * tilingData->epWorldSize * maskSlotSize, (int64_t)ALIGN_512);

            dispatchL1CommPtr = base + workspaceSize;
            uint32_t mxScaleNum = Ops::Base::CeilDiv(tilingData->h, static_cast<uint32_t>(MXFP_SCALE_GROUP_NUM));
            uint32_t dataBytes = Ops::Base::CeilAlign(tilingData->h, static_cast<uint32_t>(ALIGN_256)) * SIZE_INT_8;
            uint32_t scaleBytes = mxScaleNum * SIZE_INT_8;
            uint32_t tokenScaleBytes = Ops::Base::CeilAlign(dataBytes + scaleBytes, static_cast<uint32_t>(ALIGN_32));
            uint32_t recordBytes = Ops::Base::CeilAlign(tokenScaleBytes + static_cast<uint32_t>(ALIGN_32),
                                                        static_cast<uint32_t>(ALIGN_512));
            int64_t serverWorkspaceBytes =
                static_cast<int64_t>(ALIGN_32) + static_cast<int64_t>(tilingData->bs) * recordBytes;
            workspaceSize += Ops::Base::CeilAlign(static_cast<int64_t>(serverNum) * serverWorkspaceBytes,
                                                  static_cast<int64_t>(ALIGN_512));

            dispatchCursorPtr = base + workspaceSize;
            workspaceSize +=
                Ops::Base::CeilAlign(static_cast<int64_t>(serverNum * SIZE_INT_32), static_cast<int64_t>(ALIGN_512));

            dispatchDonePtr = base + workspaceSize;
            workspaceSize += Ops::Base::CeilAlign(static_cast<int64_t>(tilingData->aicNum * SIZE_INT_32),
                                                  static_cast<int64_t>(ALIGN_512));

            dispatchL2CommPtr = base + workspaceSize;
            workspaceSize +=
                Ops::Base::CeilAlign(static_cast<int64_t>(tilingData->aicNum) * 6 * static_cast<int64_t>(recordBytes),
                                     static_cast<int64_t>(ALIGN_512));

            combineCommDataPtr = base + workspaceSize;
            int64_t maxDataLengthPerBlock = Ops::Base::CeilAlign(
                static_cast<int64_t>(MegaMoeImpl::L1_TILE_N * sizeof(uint32_t) / 2), (int64_t)ALIGN_32);
            // 每个 token 含数据+三元组(32B)
            int64_t maxDataSizePerToken =
                Ops::Base::CeilDiv(static_cast<int64_t>(tilingData->h), (int64_t)MegaMoeImpl::L1_TILE_N) *
                (ALIGN_32 + maxDataLengthPerBlock);
            int64_t maxDataSize = Ops::Base::CeilAlign(static_cast<int64_t>(tilingData->bs) * maxDataSizePerToken *
                                                           static_cast<int64_t>(tilingData->expertPerRank) *
                                                           static_cast<int64_t>(tilingData->epWorldSize),
                                                       (int64_t)ALIGN_512);
            workspaceSize += maxDataSize;

            combineCommNotifyPtr = base + workspaceSize;
            // 每个 expert 分配一个 int32 通知
            uint32_t maxNotifySize =
                Ops::Base::CeilAlign((int64_t)(tilingData->epWorldSize * sizeof(int32_t)), (int64_t)ALIGN_512);
            workspaceSize += maxNotifySize;
        }
    }
};
#endif // MEGA_MOE_WORKSPACE_INFO_H
