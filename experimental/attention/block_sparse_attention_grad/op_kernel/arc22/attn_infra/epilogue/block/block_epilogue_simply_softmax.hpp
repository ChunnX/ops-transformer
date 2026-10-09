/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file block_epliogue_simply_softmax.h
 * \brief Block Epliogue Simply Softmax Kernel Implementation
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_SIMPLY_SOFTMAX_HPP
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_SIMPLY_SOFTMAX_HPP

#include "../../../attn_infra/arch/bsag_resource.hpp"
#include "../../../attn_infra/epilogue/bsag_epilogue_dispatch_policy.hpp"
#include "kernel_operator.h"

using namespace AscendC;

template <typename InDtype>
struct SimplySoftMaxInfo {
    LocalTensor<float> sTensor;
    LocalTensor<float> lseTensor;
    LocalTensor<float> lseBrocTensor;
    LocalTensor<float> pFp32Tensor;
    LocalTensor<InDtype> pFp16Tensor;

    GlobalTensor<float> sGm;
    GlobalTensor<float> lseGm;
    GlobalTensor<InDtype> pGm;
};

template <typename InDtype>
struct CalDsInfo {
    LocalTensor<float> dpFp32Tensor;
    LocalTensor<float> softmaxGradTensor;
    LocalTensor<float> pFp32Tensor;
    LocalTensor<InDtype> dsFp16Tensor;

    GlobalTensor<float> dpGm;
    GlobalTensor<float> softmaxGradGm;
    GlobalTensor<InDtype> dsGm;
};

namespace NpuArch::Epilogue::Block {
template <typename InputDType, typename OutputDtype, uint32_t INPUT_LAYOUT>
class SimpltSoftmax {
public:
    using DispatchPolicy = EpilogueAtlasA2FAGPre;
    using ArchTag = typename DispatchPolicy::ArchTag;

    struct Params {
        GM_ADDR s;          // 连续
        GM_ADDR softmaxLse; // 需要跳着搬运
        GM_ADDR dp;         // 连续
        GM_ADDR actualQSeqlen;
        GM_ADDR actualKvSeqlen;
        GM_ADDR softGradworkspace; // 需要跳着搬运
        GM_ADDR pWorkspace;        // 连续
        GM_ADDR dsWorkspace;       // 连续
        GM_ADDR tilingData;
        uint64_t actualRow = 0;
        uint64_t actualCol = 0;
        uint64_t processNums = 0;
        uint64_t curCoreBatch = 0;
        uint64_t curCoreN1Idx = 0;
        uint64_t curCoreS1Idx = 0;
        uint64_t curT1Idx = 0;
        uint64_t validCol = 0;
        uint64_t rawMatrixRows = 0;
        uint64_t rawRowOffset = 0;
        bool rawNz = false;

        __aicore__ inline Params() {}

        __aicore__ inline Params(GM_ADDR s_, GM_ADDR softmaxLse_, GM_ADDR dp_, GM_ADDR actualQSeqlen_,
                                 GM_ADDR actualKvSeqlen_, GM_ADDR softGradworkspace_, GM_ADDR pWorkspace_,
                                 GM_ADDR dsWorkspace_, GM_ADDR tilingData_, uint64_t acutualRow_, uint64_t actualCol_,
                                 uint64_t processNums_, uint64_t curBatch_, uint64_t curN1_, uint64_t curS1_,
                                 uint64_t curT1_, uint64_t validCol_ = 0, uint64_t rawMatrixRows_ = 0,
                                 uint64_t rawRowOffset_ = 0, bool rawNz_ = false)
            : s(s_),
              softmaxLse(softmaxLse_),
              dp(dp_),
              actualQSeqlen(actualQSeqlen_),
              actualKvSeqlen(actualKvSeqlen_),
              softGradworkspace(softGradworkspace_),
              pWorkspace(pWorkspace_),
              dsWorkspace(dsWorkspace_),
              tilingData(tilingData_),
              actualRow(acutualRow_),
              actualCol(actualCol_),
              processNums(processNums_),
              curCoreBatch(curBatch_),
              curCoreN1Idx(curN1_),
              curCoreS1Idx(curS1_),
              curT1Idx(curT1_),
              validCol(validCol_ == 0 ? actualCol_ : validCol_),
              rawMatrixRows(rawMatrixRows_),
              rawRowOffset(rawRowOffset_),
              rawNz(rawNz_)
        {}
    };

    NpuArch::Arch::Resource<ArchTag> resource;

    constexpr static uint64_t STAGES = 1;
    constexpr static uint64_t GROUPED_INPUT_STAGES = 2;
    constexpr static uint64_t GROUPED_Q_METADATA_SLOTS = 4;
    constexpr static uint64_t INPUT_NUM = 2;
    constexpr static uint64_t DOUBLE_BUFFER = 2;
    constexpr static uint64_t BNSD = 1;
    constexpr static uint64_t TND = 0;
    constexpr static uint64_t proceeM = 128;
    constexpr static uint64_t proceeK = 128;
    constexpr static uint64_t BRCB_BASE_NUM = 8;
    constexpr static uint64_t REAPTE_BYTE = 256;

    constexpr static uint64_t BLOCK_BYTE_SIZE = 32;
    constexpr static uint64_t BLOCK_FP32_NUM = 8;
    constexpr static uint64_t BLOCK_16_NUM = 16;
    constexpr static uint64_t SFMG_HIGH_PERF_N_FACTOR = 8;
    constexpr static uint64_t SFMG_HIGH_PERF_D_FACTOR = 64;
    constexpr static uint64_t baseM = 64;

    // The ABI reserves a 33-KiB raw-stage stride. Wide tiles pack 16-column
    // NZ slabs into at most 32 KiB for 64x128, then reuse the consumed raw-S
    // stage as two 16-KiB P/dS buffers after forming P in FP32 scratch.
    constexpr static uint32_t GROUPED_RAW_STAGE_BYTES = 33 * 1024;
    constexpr static uint32_t GROUPED_S_STAGE_BASE = 0;
    constexpr static uint32_t GROUPED_DP_STAGE_BASE = 66 * 1024;
    constexpr static uint32_t GROUPED_NZ_SCRATCH_OFFSET = 132 * 1024;
    // Shared outputs and the first two row-metadata slots end below 174 KiB;
    // the final two metadata slots use the remaining UB tail.
    constexpr static uint32_t GROUPED_EXTRA_LSE_OFFSET = 174 * 1024;
    constexpr static uint32_t GROUPED_EXTRA_D_OFFSET =
        GROUPED_EXTRA_LSE_OFFSET + 2 * baseM * BRCB_BASE_NUM * sizeof(float);

    uint64_t cBlockIdx = 0;
    uint64_t cubeCoreIdx = 0;
    uint64_t vecCoreIdx = 0;
    uint64_t row = 0; // 当前core需要处理q方向的s数
    uint64_t col = 0; // 当前core需要处理kv方向的s数
    uint64_t align32Col = 0;
    uint64_t align16Col = 0;
    uint64_t alignCol = 0;
    uint64_t alignRow = 0;
    uint64_t curCoreBatch = 0;
    uint64_t curCoreN1Idx = 0; // q_n
    uint64_t curT1Idx = 0;     // q_t
    uint64_t curCoreS1Idx = 0; // q_s
    uint64_t maxQSeqlen = 0;
    uint64_t maxKvSeqlen = 0;
    uint64_t n1 = 0; // q_n
    uint64_t transpseStride = 0;

    uint64_t usedVecCoreNums = 0;
    uint64_t p16BaseBufLen = 0;
    uint64_t p32BaseBufLen = 0;
    float scaleValue = 0.0f;
    uint64_t validCol = 0;
    uint64_t rawMatrixRows = 0;
    uint64_t rawRowOffset = 0;
    bool groupedPipeline = false;
    bool groupedInputPrefetched = false;
    bool groupedRawNz = false;
    bool groupedOutputBuffersReady = false;
    uint32_t groupedQMetadataMask = 0;

    GlobalTensor<float> sGm;                 // (N s1 s2)
    GlobalTensor<float> softmaxLseGm;        // (N s1 1)
    GlobalTensor<float> dpGm;                // (N s1 s2)
    GlobalTensor<InputDType> pWorkspaceGm;   // (N s1 s2)
    GlobalTensor<float> softGradworkspaceGm; // (N s1 8)
    GlobalTensor<InputDType> dsWorkspaceGm;  // (N s1 s2)

    GM_ADDR actualQSeqlen;
    GM_ADDR actualKvSeqlen;

    LocalTensor<float> sTensor[GROUPED_INPUT_STAGES];
    LocalTensor<float> lseTensor[GROUPED_INPUT_STAGES];
    LocalTensor<float> lseBrocTensor[GROUPED_INPUT_STAGES];
    LocalTensor<float> pFp32Tensor[GROUPED_INPUT_STAGES];
    LocalTensor<InputDType> p16Tensor[GROUPED_INPUT_STAGES];
    LocalTensor<float> dpFp32Tensor[GROUPED_INPUT_STAGES];
    LocalTensor<float> softmaxGradTensor[GROUPED_INPUT_STAGES];
    LocalTensor<float> groupedDCache[GROUPED_Q_METADATA_SLOTS];
    LocalTensor<float> groupedLseCache[GROUPED_Q_METADATA_SLOTS];
    LocalTensor<float> dsTensor[GROUPED_INPUT_STAGES];
    LocalTensor<InputDType> ds16Tensor[GROUPED_INPUT_STAGES];

    __aicore__ inline SimpltSoftmax()
    {
        // 分核 一个core 最大 128 * 128 一个vec 64 * 128
        uint64_t sBufferLen = GROUPED_RAW_STAGE_BYTES;
        uint64_t lBufferLen = baseM * sizeof(float);
        uint64_t lBrobBufferLen = BRCB_BASE_NUM * baseM * sizeof(float);
        uint64_t p32BufferLen = baseM * 128 * sizeof(float);
        uint64_t dpBufLen = GROUPED_RAW_STAGE_BYTES;
        uint64_t p16BufLen = baseM * 128 * sizeof(InputDType);
        uint64_t ds16BufLen = p16BufLen;
        uint64_t dBufLen = BRCB_BASE_NUM * baseM * sizeof(float);
        p16BaseBufLen = p16BufLen;
        p32BaseBufLen = p32BufferLen;

        constexpr uint64_t P16_BASE = GROUPED_NZ_SCRATCH_OFFSET;
        constexpr uint64_t DS16_BASE = 148 * 1024;
        constexpr uint64_t LSE_BROADCAST_BASE = 164 * 1024;
        constexpr uint64_t LSE_BASE = 168 * 1024;
        constexpr uint64_t SOFTMAX_GRAD_BASE = 169 * 1024;

        for (uint64_t i = 0; i < GROUPED_INPUT_STAGES; i++) {
            // Two full-height S/dP stages feed the grouped vector pipeline.
            // P/dS outputs are shared because MTE3->V events guard reuse.
            sTensor[i] = resource.ubBuf.template GetBufferByByte<float>(GROUPED_S_STAGE_BASE + sBufferLen * i);
            pFp32Tensor[i] = sTensor[i]; // 复用s
            p16Tensor[i] = resource.ubBuf.template GetBufferByByte<InputDType>(P16_BASE);
            lseBrocTensor[i] = resource.ubBuf.template GetBufferByByte<float>(LSE_BROADCAST_BASE + lBrobBufferLen * i);
            lseTensor[i] = resource.ubBuf.template GetBufferByByte<float>(LSE_BASE + lBufferLen * i);
            dpFp32Tensor[i] = resource.ubBuf.template GetBufferByByte<float>(GROUPED_DP_STAGE_BASE + dpBufLen * i);
            softmaxGradTensor[i] = resource.ubBuf.template GetBufferByByte<float>(SOFTMAX_GRAD_BASE + dBufLen * i);
            // dsTensor is otherwise unused by the legacy path; retain the
            // original raw-S address here so the wide path can repurpose the
            // public tensor descriptors without growing the AIV stack frame.
            dsTensor[i] = sTensor[i];
            ds16Tensor[i] = resource.ubBuf.template GetBufferByByte<InputDType>(DS16_BASE);
        }
        groupedDCache[0] = softmaxGradTensor[0];
        groupedDCache[1] = softmaxGradTensor[1];
        if constexpr (INPUT_LAYOUT == TND) {
            // TND's strided copy expands each scalar directly to eight lanes.
            // Keep that broadcast form in the two Q-major metadata slots.
            groupedLseCache[0] = lseBrocTensor[0];
            groupedLseCache[1] = lseBrocTensor[1];
        } else {
            groupedLseCache[0] = lseTensor[0];
            groupedLseCache[1] = lseTensor[1];
        }
        // The two additional q slots live in the UB tail. This makes the
        // metadata lifetime span the full 4-Q group.
        constexpr uint64_t metadataBytes = baseM * BRCB_BASE_NUM * sizeof(float);
        groupedLseCache[2] = resource.ubBuf.template GetBufferByByte<float>(GROUPED_EXTRA_LSE_OFFSET);
        groupedLseCache[3] = resource.ubBuf.template GetBufferByByte<float>(GROUPED_EXTRA_LSE_OFFSET + metadataBytes);
        groupedDCache[2] = resource.ubBuf.template GetBufferByByte<float>(GROUPED_EXTRA_D_OFFSET);
        groupedDCache[3] = resource.ubBuf.template GetBufferByByte<float>(GROUPED_EXTRA_D_OFFSET + metadataBytes);
    }

    __aicore__ inline ~SimpltSoftmax() {}

    __aicore__ inline void BeginGroupedPipeline(GM_ADDR softmaxLse_, GM_ADDR actualQSeqlen_, GM_ADDR actualKvSeqlen_,
                                                GM_ADDR softGradworkspace_, GM_ADDR tilingData_)
    {
        groupedPipeline = true;
        // Cache core-invariant values once; pipeline stages update only tile coordinates.
        vecCoreIdx = GetBlockIdx();
        cBlockIdx = vecCoreIdx;
        __gm__ BlockSparseAttentionGradTilingData *tilingData =
            reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData *>(tilingData_);
        usedVecCoreNums = tilingData->usedVecCoreNum;
        maxQSeqlen = tilingData->maxQSeqlen;
        maxKvSeqlen = tilingData->maxKvSeqlen;
        n1 = tilingData->numHeads;
        scaleValue = tilingData->scaleValue;
        actualQSeqlen = actualQSeqlen_;
        actualKvSeqlen = actualKvSeqlen_;
        if constexpr (INPUT_LAYOUT == TND) {
            transpseStride = (n1 - 1) * sizeof(float);
        }
        softmaxLseGm.SetGlobalBuffer((__gm__ float *)softmaxLse_);
        softGradworkspaceGm.SetGlobalBuffer((__gm__ float *)softGradworkspace_);
        // P and dS have distinct UB output buffers.  Guard their reuse from
        // Vector while allowing the next tile's independent GM->UB reads to
        // overlap the previous tile's MTE3 writes.
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
    }

    __aicore__ inline void ResetGroupedQMetadata()
    {
        groupedQMetadataMask = 0;
    }

    __aicore__ inline void InvalidateGroupedQMetadataSlot(uint32_t slot)
    {
        // Wide packets bind each metadata cache slot to the matching raw
        // input stage.  PrefetchGrouped waits for that stage's V->MTE2 token
        // before refilling it, so invalidating only the selected slot lets a
        // different Q's metadata overlap the current stage's vector work.
        const uint32_t qSlot = slot & (GROUPED_Q_METADATA_SLOTS - 1U);
        groupedQMetadataMask &= ~(1U << qSlot);
    }

    __aicore__ inline void BeginWideRawStageReuse()
    {
        // Events 0/1 of MTE3->MTE2 are free after the per-work D workspace
        // fence.  Seed one token per physical raw stage; every reuse consumes
        // the previous output-store completion and every computed tile
        // returns it after both P and dS have reached GM.
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    }

    __aicore__ inline void EndWideRawStageReuse()
    {
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    }

    __aicore__ inline void EndGroupedPipeline()
    {
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID6);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID7);
        groupedPipeline = false;
    }

    // Copy one logical row range from an FP32 NZ tile in GM.  Fixpipe's NZ
    // ABI uses C0=16 even for FP32; DataCopy lengths/strides are expressed in
    // 32-byte blocks (eight floats).  C1 slabs are packed tightly in UB so
    // layout-independent elementwise operations can cover the whole tile.
    __aicore__ inline void CopyNzRawToUb(LocalTensor<float> dst, GlobalTensor<float> src) const
    {
        constexpr uint32_t C0_SIZE = 16;
        constexpr uint32_t FP32_PER_BLOCK = 8;
        const uint32_t matrixRowsAlign = (rawMatrixRows + C0_SIZE - 1) / C0_SIZE * C0_SIZE;
        DataCopyParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(alignCol / C0_SIZE);
        copyParams.blockLen = static_cast<uint16_t>(row * C0_SIZE / FP32_PER_BLOCK);
        copyParams.srcStride = static_cast<uint16_t>((matrixRowsAlign - row) * C0_SIZE / FP32_PER_BLOCK);
        copyParams.dstStride = 0;
        DataCopy(dst, src[rawRowOffset * C0_SIZE], copyParams);
    }

    __aicore__ inline void PrepareGroupedNz(uint32_t stage)
    {
        if (!groupedRawNz) {
            return;
        }
        // Preserve S/dP in their native FP32 NZ slab order.  P32 uses the
        // shared scratch, while the consumed raw-S stage is split into two
        // non-overlapping low-precision outputs after every S slab has been
        // consumed.  The per-stage MTE3->MTE2 token protects this alias.
        pFp32Tensor[stage] = resource.ubBuf.template GetBufferByByte<float>(GROUPED_NZ_SCRATCH_OFFSET);
        p16Tensor[stage] =
            resource.ubBuf.template GetBufferByByte<InputDType>(GROUPED_S_STAGE_BASE + GROUPED_RAW_STAGE_BYTES * stage);
        ds16Tensor[stage] = resource.ubBuf.template GetBufferByByte<InputDType>(
            GROUPED_S_STAGE_BASE + GROUPED_RAW_STAGE_BYTES * stage + p16BaseBufLen);
        groupedOutputBuffersReady = true;
    }

    __aicore__ inline void PrefetchGrouped(Params const &params, uint32_t stage)
    {
        SetGroupedTileState(params);
        sGm.SetGlobalBuffer((__gm__ float *)params.s);
        dpGm.SetGlobalBuffer((__gm__ float *)params.dp);
        const auto eventId = stage == 0 ? EVENT_ID6 : EVENT_ID7;
        wait_flag(PIPE_V, PIPE_MTE2, eventId);
        if (groupedRawNz) {
            const auto outputEventId = stage == 0 ? EVENT_ID0 : EVENT_ID1;
            // The preceding output stores alias this stage's raw-S storage.
            wait_flag(PIPE_MTE3, PIPE_MTE2, outputEventId);
            // Restore canonical raw input descriptors after the previous
            // compute overlaid its low-precision outputs on the S storage.
            sTensor[stage] =
                resource.ubBuf.template GetBufferByByte<float>(GROUPED_S_STAGE_BASE + GROUPED_RAW_STAGE_BYTES * stage);
            dpFp32Tensor[stage] =
                resource.ubBuf.template GetBufferByByte<float>(GROUPED_DP_STAGE_BASE + GROUPED_RAW_STAGE_BYTES * stage);
            CopyNzRawToUb(sTensor[stage], sGm);
            CopyNzRawToUb(dpFp32Tensor[stage], dpGm);
        } else {
            if (align32Col * sizeof(float) % BLOCK_BYTE_SIZE == 0) {
                DataCopyPad(sTensor[stage], sGm,
                            {static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(float)), 0, 0, 0},
                            {true, 0, static_cast<uint8_t>(align32Col - col), 0});
                DataCopyPad(dpFp32Tensor[stage], dpGm,
                            {static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(float)), 0, 0, 0},
                            {true, 0, static_cast<uint8_t>(align32Col - col), 0});
            } else {
                DataCopyPad(sTensor[stage], sGm,
                            {static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(float)), 0, 1, 0},
                            {true, 0, static_cast<uint8_t>(align32Col - col), 0});
                DataCopyPad(dpFp32Tensor[stage], dpGm,
                            {static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(float)), 0, 1, 0},
                            {true, 0, static_cast<uint8_t>(align32Col - col), 0});
            }
        }
        const uint32_t qSlot = static_cast<uint32_t>(curT1Idx);
        const uint32_t metadataSlot = qSlot & (GROUPED_Q_METADATA_SLOTS - 1U);
        if (((groupedQMetadataMask >> qSlot) & 1U) == 0) {
            LseCopy(softmaxLseGm, groupedLseCache[metadataSlot], groupedLseCache[metadataSlot], row, curCoreS1Idx);
            CopyDIn(softGradworkspaceGm, groupedDCache[metadataSlot], row, curCoreS1Idx);
            groupedQMetadataMask |= 1U << qSlot;
        }
        set_flag(PIPE_MTE2, PIPE_V, eventId);
    }

    __aicore__ inline void ComputeGrouped(Params const &params, uint32_t stage)
    {
        SetGroupedTileState(params);
        pWorkspaceGm.SetGlobalBuffer((__gm__ InputDType *)params.pWorkspace);
        dsWorkspaceGm.SetGlobalBuffer((__gm__ InputDType *)params.dsWorkspace);
        const auto eventId = stage == 0 ? EVENT_ID6 : EVENT_ID7;
        wait_flag(PIPE_MTE2, PIPE_V, eventId);
        PrepareGroupedNz(stage);
        groupedInputPrefetched = true;
        if (groupedRawNz) {
            ComputeGroupedNz(stage);
        } else {
            compute(0, row, col, curCoreS1Idx, stage);
        }
        groupedInputPrefetched = false;
        groupedOutputBuffersReady = false;
        set_flag(PIPE_V, PIPE_MTE2, eventId);
    }

    // Store one AIV-owned row range into a standard BF16 zN tile.  Each
    // 16-column slab is contiguous in UB, while GM retains the full rounded
    // Q stride shared by both vector subcores.
    __aicore__ inline void CopyNzLowpToGm(GlobalTensor<InputDType> dst, LocalTensor<InputDType> src,
                                          uint32_t storeRows) const
    {
        constexpr uint32_t C0_SIZE = 16;
        const uint32_t matrixRowsAlign = (rawMatrixRows + C0_SIZE - 1) / C0_SIZE * C0_SIZE;
        DataCopyParams copyParams;
        copyParams.blockCount = static_cast<uint16_t>(alignCol / C0_SIZE);
        // One BF16 row of a C0 slab is exactly one 32-byte DMA block.
        copyParams.blockLen = static_cast<uint16_t>(storeRows);
        copyParams.srcStride = 0;
        copyParams.dstStride = static_cast<uint16_t>(matrixRowsAlign - storeRows);
        DataCopy(dst[rawRowOffset * C0_SIZE], src, copyParams);
    }

    // Q padding must not be appended to a 64-row AIV output slab: e.g. a
    // 126-row tile gives AIV1 63 real rows plus two padding rows and would
    // exceed its 16-KiB P16 half.  Instead, reuse one tiny zero run from the
    // unused tail of P16 and clear each GM C1 slab directly.
    __aicore__ inline void ClearNzLowpQPadding(GlobalTensor<InputDType> dst, LocalTensor<InputDType> zero,
                                               uint32_t qPadding) const
    {
        constexpr uint32_t C0_SIZE = 16;
        if (qPadding == 0) {
            return;
        }
        const uint32_t matrixRowsAlign = (rawMatrixRows + C0_SIZE - 1) / C0_SIZE * C0_SIZE;
        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        // One BF16 row of a C0 slab is one 32-byte block.
        copyParams.blockLen = static_cast<uint16_t>(qPadding);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        const uint32_t paddingRow = static_cast<uint32_t>(rawRowOffset + row);
        const uint32_t c1Count = static_cast<uint32_t>(alignCol / C0_SIZE);
        for (uint32_t c1 = 0; c1 < c1Count; ++c1) {
            const uint64_t destinationOffset =
                static_cast<uint64_t>(c1) * matrixRowsAlign * C0_SIZE + static_cast<uint64_t>(paddingRow) * C0_SIZE;
            DataCopy(dst[destinationOffset], zero, copyParams);
        }
    }

    // Native-NZ vector epilogue used only by the wide schedule.  Arithmetic
    // remains FP32 and in the same scale/subtract/exp and subtract/multiply
    // order as the ND path; only the physical traversal changes.  P/dS are
    // rounded once to the input dtype and remain zN through their Cube reads.
    // NZ slabs have sixteen columns and tightly packed rows. Clear alignment
    // tails with one masked vector instruction per affected slab.
    __aicore__ inline void FillNzInvalidColumns(LocalTensor<float> tensor, float value) const
    {
        constexpr uint32_t C0_SIZE = 16;
        const uint32_t firstSlab = static_cast<uint32_t>(validCol / C0_SIZE);
        const uint32_t slabCount = static_cast<uint32_t>(alignCol / C0_SIZE);
        for (uint32_t slab = firstSlab; slab < slabCount; ++slab) {
            const uint32_t validLanes = slab == firstSlab ? validCol % C0_SIZE : 0;
            uint64_t mask[2] = {(0xffffULL << validLanes) & 0xffffULL, 0};
            Duplicate(tensor[slab * row * C0_SIZE], value, mask, static_cast<uint8_t>(row), 1,
                      C0_SIZE / BLOCK_FP32_NUM);
        }
        SetVectorMask<int8_t>(~0ULL, ~0ULL);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeGroupedNz(uint32_t stage)
    {
        constexpr uint32_t C0_SIZE = 16;
        constexpr float NEGATIVE_MAX = -3.402823466e+38F;
        const uint32_t metadataSlot = static_cast<uint32_t>(curT1Idx) & (GROUPED_Q_METADATA_SLOTS - 1U);
        LocalTensor<float> currentLse = groupedLseCache[metadataSlot];
        LocalTensor<float> currentLseBroadcast =
            INPUT_LAYOUT == TND ? groupedLseCache[metadataSlot] : lseBrocTensor[stage];
        LocalTensor<float> currentD = groupedDCache[metadataSlot];
        LocalTensor<float> rawS = sTensor[stage];
        LocalTensor<float> rawDp = dpFp32Tensor[stage];
        LocalTensor<float> p32 = pFp32Tensor[stage];
        LocalTensor<InputDType> p16 = p16Tensor[stage];
        LocalTensor<InputDType> ds16 = ds16Tensor[stage];

        const uint32_t qRound = (rawMatrixRows + C0_SIZE - 1) / C0_SIZE * C0_SIZE;
        // Exactly one non-empty subcore owns the bottom of the Q tile.  It
        // clears the rounded GM tail separately after storing its real rows;
        // padding is never appended to a capacity-64 UB slab.
        const uint32_t qPadding = rawRowOffset + row == rawMatrixRows ? qRound - rawMatrixRows : 0;
        const uint32_t storeRows = static_cast<uint32_t>(row);
        const uint32_t c1Count = static_cast<uint32_t>(alignCol / C0_SIZE);
        const uint32_t inputSlabElements = static_cast<uint32_t>(row) * C0_SIZE;
        const uint32_t outputSlabElements = storeRows * C0_SIZE;
        const uint32_t storeElements = storeRows * static_cast<uint32_t>(alignCol);
        const bool hasInvalidColumns = validCol < alignCol;

        if constexpr (INPUT_LAYOUT == BNSD) {
            const uint8_t repeatTimes = static_cast<uint8_t>(CeilDiv(row, BRCB_BASE_NUM));
            Brcb(currentLseBroadcast, currentLse, repeatTimes, {1, 8});
            AscendC::PipeBarrier<PIPE_V>();
        }

        // Raw S slabs are tightly packed by CopyNzRawToUb, so the scalar
        // scale can cover them with one vector launch.
        Muls(rawS, rawS, static_cast<float>(scaleValue), row * alignCol);
        AscendC::PipeBarrier<PIPE_V>();

        if (hasInvalidColumns) {
            FillNzInvalidColumns(rawS, NEGATIVE_MAX);
        }

        for (uint32_t c1 = 0; c1 < c1Count; ++c1) {
            const uint32_t inputOffset = c1 * inputSlabElements;
            const uint32_t outputOffset = c1 * outputSlabElements;
            SubBrcb(p32[outputOffset], rawS[inputOffset], currentLseBroadcast, row, C0_SIZE);
        }
        AscendC::PipeBarrier<PIPE_V>();
        // All C1 slabs are packed consecutively in UB.  Exp and Mul do not
        // depend on row metadata, so issue one flat vector operation instead
        // of one launch per 16-column slab.
        Exp(p32, p32, storeElements);
        AscendC::PipeBarrier<PIPE_V>();

        // Exp(-FLT_MAX) is already zero on this target; write the padding
        // explicitly as well so correctness is independent of exp underflow
        // and of prior contents in the shared scratch.
        if (hasInvalidColumns) {
            FillNzInvalidColumns(p32, 0.0F);
        }
        Cast(p16, p32, AscendC::RoundMode::CAST_ROUND, storeElements);
        // Avoid constructing a one-past-end LocalTensor descriptor for the
        // common full-height case.  No padding access is made in that case,
        // but keeping the descriptor in bounds also satisfies stricter
        // AscendC bounds instrumentation.
        const uint32_t paddingZeroOffset = qPadding == 0 ? 0 : storeElements;
        LocalTensor<InputDType> paddingZero = p16[paddingZeroOffset];
        if (qPadding != 0) {
            Duplicate(paddingZero, static_cast<InputDType>(0), qPadding * C0_SIZE);
        }

        const auto eventId = stage == 0 ? EVENT_ID5 : EVENT_ID4;
        set_flag(PIPE_V, PIPE_MTE3, eventId);
        wait_flag(PIPE_V, PIPE_MTE3, eventId);
        CopyNzLowpToGm(pWorkspaceGm, p16, storeRows);

        for (uint32_t c1 = 0; c1 < c1Count; ++c1) {
            const uint32_t inputOffset = c1 * inputSlabElements;
            SubBrcb(rawDp[inputOffset], rawDp[inputOffset], currentD, row, C0_SIZE);
        }
        AscendC::PipeBarrier<PIPE_V>();
        // Reuse P32 as the tightly packed dS FP32 result.  P16 is disjoint,
        // so its preceding MTE3 write can overlap these vector operations.
        Mul(p32, p32, rawDp, storeElements);
        AscendC::PipeBarrier<PIPE_V>();
        if (hasInvalidColumns) {
            FillNzInvalidColumns(p32, 0.0F);
        }
        Cast(ds16, p32, AscendC::RoundMode::CAST_ROUND, storeElements);

        set_flag(PIPE_V, PIPE_MTE3, eventId);
        wait_flag(PIPE_V, PIPE_MTE3, eventId);
        CopyNzLowpToGm(dsWorkspaceGm, ds16, storeRows);
        if (qPadding != 0) {
            ClearNzLowpQPadding(pWorkspaceGm, paddingZero, qPadding);
            ClearNzLowpQPadding(dsWorkspaceGm, paddingZero, qPadding);
        }
        const auto outputEventId = stage == 0 ? EVENT_ID0 : EVENT_ID1;
        set_flag(PIPE_MTE3, PIPE_MTE2, outputEventId);
    }

    __aicore__ inline void SetGroupedTileState(Params const &params)
    {
        col = params.actualCol;
        align32Col = (col + BLOCK_FP32_NUM - 1) / BLOCK_FP32_NUM * BLOCK_FP32_NUM;
        align16Col = (col + BLOCK_16_NUM - 1) / BLOCK_16_NUM * BLOCK_16_NUM;
        alignCol = (align32Col % BLOCK_16_NUM != 0) ? align16Col : align32Col;
        row = params.actualRow;
        curCoreBatch = params.curCoreBatch;
        curCoreN1Idx = params.curCoreN1Idx;
        curCoreS1Idx = params.curCoreS1Idx;
        curT1Idx = params.curT1Idx;
        validCol = params.validCol == 0 ? params.actualCol : params.validCol;
        rawMatrixRows = params.rawMatrixRows;
        rawRowOffset = params.rawRowOffset;
        groupedRawNz = params.rawNz;
    }

    template <int32_t CORE_TYPE = g_coreType>
    __aicore__ inline void operator()(Params const &params);

    template <>
    __aicore__ inline void operator()<AscendC::AIC>(Params const &params)
    {}

    __aicore__ inline void Init(Params const &params)
    {
        vecCoreIdx = GetBlockIdx();
        cBlockIdx = vecCoreIdx;
        __gm__ BlockSparseAttentionGradTilingData *tilingData =
            reinterpret_cast<__gm__ BlockSparseAttentionGradTilingData *>(params.tilingData);
        usedVecCoreNums = tilingData->usedVecCoreNum;

        if (cBlockIdx >= usedVecCoreNums) {
            return;
        }

        maxQSeqlen = tilingData->maxQSeqlen;
        maxKvSeqlen = tilingData->maxKvSeqlen;
        n1 = tilingData->numHeads; // q_n
        col = params.actualCol;
        align32Col = (col + BLOCK_FP32_NUM - 1) / BLOCK_FP32_NUM * BLOCK_FP32_NUM; // fp32 对齐后的列数
        align16Col = (col + BLOCK_16_NUM - 1) / BLOCK_16_NUM * BLOCK_16_NUM;
        alignCol = (align32Col % BLOCK_16_NUM != 0) ? align16Col : align32Col;
        curCoreBatch = params.curCoreBatch;
        curCoreN1Idx = params.curCoreN1Idx;
        curT1Idx = params.curT1Idx;
        curCoreS1Idx = params.curCoreS1Idx;
        actualQSeqlen = params.actualQSeqlen;
        actualKvSeqlen = params.actualKvSeqlen;
        scaleValue = tilingData->scaleValue;
        validCol = params.validCol == 0 ? params.actualCol : params.validCol;

        row = params.actualRow;
        if (row <= 0) {
            return;
        }

        if constexpr (INPUT_LAYOUT == TND) {
            transpseStride = (n1 * 1 - 1) * sizeof(float);
        }

        // 初始化 GM
        sGm.SetGlobalBuffer((__gm__ float *)params.s);
        softmaxLseGm.SetGlobalBuffer((__gm__ float *)params.softmaxLse);
        dpGm.SetGlobalBuffer((__gm__ float *)params.dp);
        pWorkspaceGm.SetGlobalBuffer((__gm__ InputDType *)params.pWorkspace);
        softGradworkspaceGm.SetGlobalBuffer((__gm__ float *)params.softGradworkspace);
        dsWorkspaceGm.SetGlobalBuffer((__gm__ InputDType *)params.dsWorkspace);
    }

    template <>
    __aicore__ inline void operator()<AscendC::AIV>(Params const &params)
    {
        Init(params);

        if (cBlockIdx >= usedVecCoreNums || row <= 0) {
            return;
        }

        // col <= 128
        // 计算单loop的计算量及loop次数
        uint64_t eleBaseBuffNum = p32BaseBufLen / STAGES / sizeof(float); // 基本buffer块的元素数量
        uint64_t bufferRows = baseM / STAGES;                             // 一次lopp可以执行的最多row行数
        uint64_t rowLoopTimes = row / bufferRows;
        uint64_t tailRowNum = row - rowLoopTimes * bufferRows;

        uint64_t ping = 0;

        // One UB stage shares a reuse event across row chunks; drain it after the last chunk.
        if (!groupedPipeline) {
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
        }
        for (uint64_t i = 0; i < rowLoopTimes; i++) {
            uint64_t curS1Idx = curCoreS1Idx + i * bufferRows;
            int32_t gmRowOffset = i * bufferRows * col;
            compute(gmRowOffset, bufferRows, col, curS1Idx, ping);
        }

        if (tailRowNum > 0) {
            uint64_t curS1Idx = curCoreS1Idx + rowLoopTimes * bufferRows;
            int32_t gmOffset = rowLoopTimes * bufferRows * col;
            compute(gmOffset, tailRowNum, col, curS1Idx, ping);
        }
        if (!groupedPipeline) {
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID5);
        }
    }

    __aicore__ inline void compute(int32_t gmOffset, uint64_t row, uint64_t col, uint64_t curS1Idx, uint64_t ping)
    {
        const uint32_t metadataSlot = static_cast<uint32_t>(curT1Idx) & (GROUPED_Q_METADATA_SLOTS - 1U);
        LocalTensor<float> currentLse = groupedPipeline ? groupedLseCache[metadataSlot] : lseTensor[ping];
        LocalTensor<float> currentLseBroadcast =
            groupedPipeline && INPUT_LAYOUT == TND ? groupedLseCache[metadataSlot] : lseBrocTensor[ping];
        LocalTensor<float> currentD = groupedPipeline ? groupedDCache[metadataSlot] : softmaxGradTensor[ping];
        struct SimplySoftMaxInfo<InputDType> runSftInfo = {
            sTensor[ping], currentLse, currentLseBroadcast, pFp32Tensor[ping], p16Tensor[ping], sGm[gmOffset],
                softmaxLseGm, pWorkspaceGm[gmOffset]
        };
        struct CalDsInfo<InputDType> runDsInfo = {
            dpFp32Tensor[ping], currentD, pFp32Tensor[ping], ds16Tensor[ping], dpGm[gmOffset], softGradworkspaceGm,
                dsWorkspaceGm[gmOffset]
        };

        CalSimplySoft(runSftInfo, row, col, curS1Idx, ping);
        CalDs(runDsInfo, row, col, curS1Idx, ping);
    }

    // Reconstruct P = exp(scale * S - LSE) in FP32 before casting to the input dtype.
    __aicore__ inline void CalSimplySoft(struct SimplySoftMaxInfo<InputDType> runSftInfo, uint64_t row, uint64_t col,
                                         uint64_t curS1Idx, uint64_t ping)
    {
        LocalTensor<float> &sLocal = runSftInfo.sTensor;
        LocalTensor<float> &lse = runSftInfo.lseTensor;
        LocalTensor<float> &lseFp32Brc = runSftInfo.lseBrocTensor;
        LocalTensor<float> &p32Local = runSftInfo.pFp32Tensor;
        LocalTensor<InputDType> &p16Local = runSftInfo.pFp16Tensor;

        GlobalTensor<float> s = runSftInfo.sGm;
        GlobalTensor<float> lseGm = runSftInfo.lseGm;
        GlobalTensor<InputDType> pGm = runSftInfo.pGm;

        uint64_t countAlign = row * alignCol;

        auto eventId = ping ? EVENT_ID4 : EVENT_ID5;

        if (!groupedInputPrefetched) {
            if (!groupedPipeline) {
                wait_flag(PIPE_MTE3, PIPE_MTE2, eventId);
            }

            if (align32Col * sizeof(InputDType) % BLOCK_BYTE_SIZE == 0) {
                DataCopyPad(sLocal, s,
                            {static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(float)), 0, 0, 0},
                            {true, 0, static_cast<uint8_t>(align32Col - col), 0});
            } else {
                DataCopyPad(sLocal, s,
                            {static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(float)), 0, 1, 0},
                            {true, 0, static_cast<uint8_t>(align32Col - col), 0});
            }

            LseCopy(lseGm, lse, lseFp32Brc, row, curS1Idx);

            set_flag(PIPE_MTE2, PIPE_V, eventId);
            wait_flag(PIPE_MTE2, PIPE_V, eventId);
        }

        uint8_t repeatimes = CeilDiv(row, BRCB_BASE_NUM);
        if constexpr (INPUT_LAYOUT == BNSD) {
            Brcb(lseFp32Brc, lse, repeatimes, {1, 8});
            AscendC::PipeBarrier<PIPE_V>();
        }

        Muls(sLocal, sLocal, (float)scaleValue, countAlign);
        AscendC::PipeBarrier<PIPE_V>();

        if (validCol < col) {
            for (uint64_t r = 0; r < row; ++r) {
                for (uint64_t c = validCol; c < col; ++c) {
                    sLocal.SetValue(r * alignCol + c, -3.402823466e+38F);
                }
            }
            AscendC::PipeBarrier<PIPE_V>();
        }

        SubBrcb(p32Local, sLocal, lseFp32Brc, row, alignCol);
        AscendC::PipeBarrier<PIPE_V>();

        Exp(p32Local, p32Local, countAlign);
        AscendC::PipeBarrier<PIPE_V>();

        if (groupedPipeline && !groupedOutputBuffersReady) {
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
        }
        Cast(p16Local, p32Local, AscendC::RoundMode::CAST_ROUND, countAlign);

        set_flag(PIPE_V, PIPE_MTE3, eventId);
        wait_flag(PIPE_V, PIPE_MTE3, eventId);

        DataCopyPad(pGm, p16Local,
                    {static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(InputDType)), 0, 0, 0});
        if (groupedPipeline && !groupedRawNz) {
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID4);
        }
    }

    // Compute dS = P * (dP - D), broadcasting the row-wise D term.
    __aicore__ inline void CalDs(struct CalDsInfo<InputDType> runDsInfo, uint64_t row, uint64_t col, uint64_t curS1Idx,
                                 uint64_t ping)
    {
        LocalTensor<float> dpLocal = runDsInfo.dpFp32Tensor;
        LocalTensor<float> dLocal = runDsInfo.softmaxGradTensor;
        LocalTensor<InputDType> ds16Tensor = runDsInfo.dsFp16Tensor;
        LocalTensor<float> &p32Local = runDsInfo.pFp32Tensor;

        GlobalTensor<float> dp = runDsInfo.dpGm;
        GlobalTensor<float> d = runDsInfo.softmaxGradGm;
        GlobalTensor<InputDType> ds = runDsInfo.dsGm;

        uint64_t countAlign = row * alignCol;

        auto eventId = ping ? EVENT_ID4 : EVENT_ID5;

        if (!groupedInputPrefetched) {
            if (align32Col * sizeof(InputDType) % BLOCK_BYTE_SIZE == 0) {
                DataCopyPad(dpLocal, dp,
                            {static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(float)), 0, 0, 0},
                            {true, 0, static_cast<uint8_t>(align32Col - col), 0});
            } else {
                DataCopyPad(dpLocal, dp,
                            {static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(float)), 0, 1, 0},
                            {true, 0, static_cast<uint8_t>(align32Col - col), 0});
            }
            CopyDIn(d, dLocal, row, curS1Idx);

            set_flag(PIPE_MTE2, PIPE_V, eventId);
            wait_flag(PIPE_MTE2, PIPE_V, eventId);
        }

        SubBrcb(dpLocal, dpLocal, dLocal, row, alignCol);
        AscendC::PipeBarrier<PIPE_V>();

        Mul(dpLocal, p32Local, dpLocal, countAlign);
        AscendC::PipeBarrier<PIPE_V>();

        if (groupedPipeline && !groupedOutputBuffersReady) {
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
        }
        Cast(ds16Tensor, dpLocal, AscendC::RoundMode::CAST_ROUND, countAlign);

        set_flag(PIPE_V, PIPE_MTE3, eventId);
        wait_flag(PIPE_V, PIPE_MTE3, eventId);

        DataCopyPad(ds, ds16Tensor,
                    {static_cast<uint16_t>(row), static_cast<uint32_t>(col * sizeof(InputDType)), 0, 0, 0});
        if (groupedPipeline && groupedRawNz) {
            const auto outputEventId = ping == 0 ? EVENT_ID0 : EVENT_ID1;
            set_flag(PIPE_MTE3, PIPE_MTE2, outputEventId);
        } else if (groupedPipeline) {
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID5);
        } else {
            set_flag(PIPE_MTE3, PIPE_MTE2, eventId);
        }
    }

    // TND gathers strided LSE values into eight-lane rows; BNSD loads contiguous values.
    __aicore__ inline void LseCopy(GlobalTensor<float> &LseGm, LocalTensor<float> &lse, LocalTensor<float> &lseFp32Brc,
                                   uint64_t count, uint64_t curS1Idx)
    {
        uint64_t startOffset = 0;
        auto eventId = EVENT_ID5;
        if constexpr (INPUT_LAYOUT == TND) {
            uint64_t prefixSum = 0;
            for (uint64_t b = 0; b < curCoreBatch; b++) {
                prefixSum += ((__gm__ int64_t *)actualQSeqlen)[b];
            }
            uint64_t bOffset = n1 * prefixSum;
            startOffset = bOffset + curS1Idx * n1 + curCoreN1Idx;
            // 对于TND 格式来说， 会进行类似 (s n) -> (n s) 的transpose转换
            DataCopyPad(lseFp32Brc, LseGm[startOffset],
                        {static_cast<uint16_t>(count), static_cast<uint32_t>(1 * sizeof(float)),
                         static_cast<uint32_t>(transpseStride), 0, 0},
                        {false, 0, 0, 0});
        } else {
            startOffset = curCoreBatch * (n1 * maxQSeqlen) + curCoreN1Idx * maxQSeqlen + curS1Idx;
            DataCopyPad(lse, LseGm[startOffset],
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(count * sizeof(float)),
                         static_cast<uint32_t>(0), 0, 0},
                        {false, 0, 0, 0});
        }
    }

    // Subtract row-broadcast ubIn1 [row, 8] from ubIn0 [row, col].
    __aicore__ inline void SubBrcb(LocalTensor<float> const &ubOut, LocalTensor<float> const &ubIn0,
                                   LocalTensor<float> const &ubIn1, uint64_t row, uint64_t col)
    {
        // Each vector repeat handles 64 FP32 columns; repeats advance across rows.
        uint32_t countEachRepeat = REAPTE_BYTE / sizeof(float);
        uint32_t colLoop = (col + countEachRepeat - 1) / countEachRepeat;
        uint32_t remain = col % countEachRepeat;
        uint64_t mask = countEachRepeat;
        uint8_t repeatTimes = row;
        AscendC::BinaryRepeatParams repeatParams;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = col / BLOCK_FP32_NUM;
        repeatParams.src0RepStride = col / BLOCK_FP32_NUM;
        repeatParams.src1RepStride = 1;

        for (uint32_t i = 0; i < colLoop; i++) {
            if (i == colLoop - 1 && remain != 0) {
                mask = remain;
            }
            AscendC::Sub(ubOut[i * countEachRepeat], ubIn0[i * countEachRepeat], ubIn1[0], mask, repeatTimes,
                         repeatParams);
        }
    }

    // D stores eight FP32 lanes per logical query row.
    __aicore__ inline void CopyDIn(GlobalTensor<float> &d, LocalTensor<float> &dLocal, uint64_t count,
                                   uint64_t curS1Idx)
    {
        uint64_t startOffset = 0;
        uint32_t srcStride = 0;
        if constexpr (INPUT_LAYOUT == TND) {
            uint64_t prefixSum = 0;
            for (uint64_t b = 0; b < curCoreBatch; b++) {
                prefixSum += ((__gm__ int64_t *)actualQSeqlen)[b];
            }
            uint64_t bOffset = prefixSum * n1 * BRCB_BASE_NUM;
            startOffset = bOffset + curS1Idx * n1 * BRCB_BASE_NUM + curCoreN1Idx * BRCB_BASE_NUM;
            srcStride = static_cast<uint32_t>((n1 - 1) * BRCB_BASE_NUM * sizeof(float));
        } else {
            startOffset = curCoreBatch * (n1 * maxQSeqlen * BRCB_BASE_NUM) + curCoreN1Idx * maxQSeqlen * BRCB_BASE_NUM +
                          curS1Idx * BRCB_BASE_NUM;
        }
        DataCopyPad(
            dLocal, d[startOffset],
            {static_cast<uint16_t>(count), static_cast<uint32_t>(BRCB_BASE_NUM * sizeof(float)), srcStride, 0, 0},
            {false, 0, 0, 0});
    }
};

} // namespace NpuArch::Epilogue::Block

#endif // CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_SIMPLY_SOFTMAX_HPP
