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
 * \file bsa_select_block_mask_base.h
 * \brief
 */
#ifndef BSA_SELECT_BLOCK_MASK_BASE_H
#define BSA_SELECT_BLOCK_MASK_BASE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "bsa_select_block_mask_common.h"
#include "bsa_select_block_mask_tiling_data.h"
#include "bsa_select_block_mask_vector.h"
#include "bsa_select_block_mask_service_cube.h"
#include "bsa_select_block_mask_radix_topk.h"

using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

#ifndef BSA_TILING_VERIFY_PRINT
#define BSA_TILING_VERIFY_PRINT 0
#endif

#define BSA_TILING_VERIFY_PRINTF(...) do {} while (0)

template <typename BSAT> class BSASelectBlockMaskBase {
public:
    using T = float;
    using IN_T = typename BSAT::inputT;
    using OUT_T = typename BSAT::outputT;
    using MM_OUT_T = T;

    static constexpr BSALayout LAYOUT_Q = BSAT::layoutQ;
    static constexpr BSALayout LAYOUT_KV = BSAT::layoutKV;

    __aicore__ inline BSASelectBlockMaskBase() {};
    __aicore__ inline void Init(__gm__ uint8_t *query, __gm__ uint8_t *key,
                                __gm__ uint8_t *blockShape, __gm__ uint8_t *postBlockShape,
                                __gm__ uint8_t *actualSeqLensQ, __gm__ uint8_t *actualSeqLensKV,
                                __gm__ uint8_t *actualBlockLenQ, __gm__ uint8_t *actualBlockLenKV,
                                __gm__ uint8_t *maskOut,
                                __gm__ uint8_t *qCompOut, __gm__ uint8_t *kCompOut,
                                __gm__ uint8_t *scoreOut, __gm__ uint8_t *softmaxOut,
                                __gm__ uint8_t *workspace,
                                const optiling::BSASelectBlockMaskTilingData *__restrict tiling,
                                TPipe *tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitConstInfo();
    __aicore__ inline void InitWorkspace(__gm__ uint8_t *workspace);
    __aicore__ inline void CalcMultiCoreOffset(uint32_t &startRow, uint32_t &endRow);
    __aicore__ inline void CalcKPoolingRange(uint32_t &startKBlock, uint32_t &endKBlock);
    __aicore__ inline void ProcessPoolingK(uint32_t batchIdx, uint32_t headIdx);
    __aicore__ inline void ProcessPoolingQ(uint32_t batchIdx, uint32_t headIdx,
                                           uint32_t tokenStart, uint32_t tokenEnd);
    __aicore__ inline void ProcessMatmulSoftmax(uint32_t batchIdx, uint32_t headIdx,
                                                uint32_t qChunkStart, uint32_t curQChunkSize);
    __aicore__ inline void ProcessSoftmaxSecondPass(uint32_t qChunkStart, uint32_t curQChunkSize, uint32_t batchIdx, uint32_t headIdx);

    TPipe *pipe = nullptr;
    const optiling::BSASelectBlockMaskTilingData *__restrict tilingData = nullptr;
    BSAConstInfo constInfo;

    BSAMatmulService<BSAT> matmulService;
    BSAVectorService<BSAT> vectorService;
    BSARadixTopKService<BSAT> radixTopKService;

    GlobalTensor<IN_T> queryGm, keyGm;
    GlobalTensor<int64_t> blockShapeGm;
    GlobalTensor<int64_t> actualSeqLensQGm, actualSeqLensKVGm;
    GlobalTensor<int64_t> actualBlockLenQGm, actualBlockLenKVGm;
    GlobalTensor<uint8_t> maskOutGmU8;

    __gm__ uint8_t *qCompOutPtr = nullptr;
    __gm__ uint8_t *kCompOutPtr = nullptr;
    __gm__ uint8_t *scoreOutPtr = nullptr;
    __gm__ uint8_t *softmaxOutPtr = nullptr;

    // 调试用
    GlobalTensor<T> qCmpOutGm, kCmpOutGm;
    GlobalTensor<T> socreFp32OutGm;
    GlobalTensor<half> softmaxOutGm;

    GlobalTensor<IN_T> qCmpGm, kCmpGm;
    GlobalTensor<half> attnScoreFp16Gm;
    GlobalTensor<T> scoreFp32Gm;
    GlobalTensor<uint8_t> maskTmpGm;
    GlobalTensor<int32_t> topkWorkspaceGm;

    // Output tensor pointers for the 4 new outputs

};

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::InitConstInfo()
{
    if ASCEND_IS_AIV {
        constInfo.aivIdx = GetBlockIdx();
        constInfo.aicIdx = constInfo.aivIdx / 2;
        constInfo.subBlockIdx = constInfo.aivIdx % 2;
    } else {
        constInfo.aicIdx = GetBlockIdx();
    }
    constInfo.aivNum = GetBlockNum() * AIC_AIV_RATIO;

    auto &baseInfo = tilingData->baseParams;
    constInfo.batchSize = baseInfo.batchSize;
    constInfo.numHeads = baseInfo.numHeads;
    constInfo.maxQSeqlen = baseInfo.maxQSeqlen;
    constInfo.maxKvSeqlen = baseInfo.maxKvSeqlen;
    constInfo.dSize = baseInfo.dSize;
    constInfo.blockShapeX = baseInfo.blockShapeX;
    constInfo.blockShapeY = baseInfo.blockShapeY;
    constInfo.xBlocks = baseInfo.xBlocks;
    constInfo.yBlocks = baseInfo.yBlocks;
    constInfo.scaleValue = baseInfo.scaleValue;
    constInfo.sparsity = baseInfo.sparsity;
    constInfo.topKValue = baseInfo.topKValue;
    constInfo.sparsityMode = static_cast<BSASparseMode>(baseInfo.sparsityMode);

    auto &mcInfo = tilingData->multiCoreParams;
    constInfo.coreNum = mcInfo.coreNum;
    constInfo.activeCoreNum = mcInfo.activeCoreNum;
    constInfo.rowsPerCore = mcInfo.rowsPerCore;
    constInfo.extraCores = mcInfo.extraCores;
    constInfo.totalRows = mcInfo.totalRows;
    constInfo.yBlocksPerCore = mcInfo.yBlocksPerCore;
    constInfo.extraYCores = mcInfo.extraYCores;
    constInfo.qChunkSize = baseInfo.qChunkSize;
    constInfo.kChunkSize = baseInfo.kChunkSize;
    constInfo.activeYVecCoreNum = mcInfo.activeYVecCoreNum;

    auto &outInfo = tilingData->outputParams;
    constInfo.qCmpOffset = 0;
    constInfo.kCmpOffset = constInfo.qCmpOffset + outInfo.qCmpSize;
    constInfo.attnScoreOffset = constInfo.kCmpOffset + outInfo.kCmpSize;
    constInfo.softmaxTmpOffset = constInfo.attnScoreOffset + outInfo.attnScoreSize;
    constInfo.topkWorkspaceOffset = constInfo.softmaxTmpOffset + outInfo.softmaxTmpSize;
    constInfo.maskOffset = constInfo.topkWorkspaceOffset + outInfo.topkWorkspaceSize;

    if ASCEND_IS_AIV {
        if (BSA_TILING_VERIFY_PRINT && GetBlockIdx() == 0) {
            uint32_t errCount = 0;
            auto &baseInfo = tilingData->baseParams;
            auto &outInfo = tilingData->outputParams;

            uint32_t expXBlocks = (constInfo.maxQSeqlen + constInfo.blockShapeX - 1) / constInfo.blockShapeX;
            uint32_t expYBlocks = (constInfo.maxKvSeqlen + constInfo.blockShapeY - 1) / constInfo.blockShapeY;
            uint32_t expActiveCoreNum = (constInfo.xBlocks < constInfo.coreNum) ? constInfo.xBlocks : constInfo.coreNum;
            uint32_t expExtraCores = constInfo.xBlocks % expActiveCoreNum;
            uint32_t expRowsPerCore = constInfo.xBlocks / expActiveCoreNum + (expExtraCores > 0 ? 1 : 0);
            uint32_t expActiveYCoreNum = (constInfo.yBlocks < constInfo.coreNum) ? constInfo.yBlocks : constInfo.coreNum;
            uint32_t expExtraYCores = constInfo.yBlocks % expActiveYCoreNum;
            uint32_t expYBlocksPerCore = constInfo.yBlocks / expActiveYCoreNum + (expExtraYCores > 0 ? 1 : 0);
            uint64_t expQCmpSize = static_cast<uint64_t>(constInfo.xBlocks) * constInfo.dSize * sizeof(float);
            uint64_t expKCmpSize = static_cast<uint64_t>(constInfo.yBlocks) * constInfo.dSize * sizeof(float);
            uint64_t expAttnScoreSize = static_cast<uint64_t>(constInfo.xBlocks) * constInfo.yBlocks * sizeof(half);
            BSA_TILING_VERIFY_PRINTF("DEBUG: xBlocks=%u, yBlocks=%u, sizeof(IN_T)=%lu, expAttnScoreSize=%lu\n",
                            constInfo.xBlocks, constInfo.yBlocks, sizeof(IN_T), expAttnScoreSize);
            uint64_t expSoftmaxTmpSize = static_cast<uint64_t>(constInfo.xBlocks) * constInfo.yBlocks * sizeof(float);
            uint64_t expMaskSize = static_cast<uint64_t>(constInfo.xBlocks) * constInfo.yBlocks * sizeof(uint8_t);
            uint64_t expSortLen = static_cast<uint64_t>(constInfo.xBlocks) * constInfo.yBlocks;
            uint64_t expTileLen = BSAMin(static_cast<uint64_t>(7168), BSAMax(static_cast<uint64_t>(128), expSortLen));
            uint64_t expTotalTileNum = (expSortLen + expTileLen - 1) / expTileLen;
            uint64_t expTopkHeaderSize = static_cast<uint64_t>(RADIX_NUM_BINS * constInfo.aivNum + constInfo.aivNum + constInfo.aivNum) * sizeof(int32_t);
            uint64_t expTopkTileDataSize = expTotalTileNum * (1 + RADIX_NUM_BINS) * sizeof(int32_t);
            uint64_t expTopkWsSize = expTopkHeaderSize + expTopkTileDataSize;

            BSA_TILING_VERIFY_PRINTF("=== Kernel Tiling Verification (core 0) ===\n");
            BSA_TILING_VERIFY_PRINTF("[baseParams]\n");
            BSA_TILING_VERIFY_PRINTF("batchSize=%u (exp=1) %s\n", constInfo.batchSize,
                            (constInfo.batchSize == 1) ? "OK" : "ERR");
            if (constInfo.batchSize != 1) errCount++;
            BSA_TILING_VERIFY_PRINTF("numHeads=%u (nonzero=%s)\n", constInfo.numHeads,
                            (constInfo.numHeads > 0) ? "OK" : "ERR");
            if (constInfo.numHeads == 0) errCount++;
            BSA_TILING_VERIFY_PRINTF("maxQSeqlen=%u (exp=4096) %s\n", constInfo.maxQSeqlen,
                            (constInfo.maxQSeqlen == 4096) ? "OK" : "ERR");
            if (constInfo.maxQSeqlen != 4096) errCount++;
            BSA_TILING_VERIFY_PRINTF("maxKvSeqlen=%u (exp=4096) %s\n", constInfo.maxKvSeqlen,
                            (constInfo.maxKvSeqlen == 4096) ? "OK" : "ERR");
            if (constInfo.maxKvSeqlen != 4096) errCount++;
            BSA_TILING_VERIFY_PRINTF("dSize=%u (exp=128) %s\n", constInfo.dSize,
                            (constInfo.dSize == 128) ? "OK" : "ERR");
            if (constInfo.dSize != 128) errCount++;
            BSA_TILING_VERIFY_PRINTF("blockShapeX=%lu (exp=128) %s\n", constInfo.blockShapeX,
                            (constInfo.blockShapeX == 128) ? "OK" : "ERR");
            if (constInfo.blockShapeX != 128) errCount++;
            BSA_TILING_VERIFY_PRINTF("blockShapeY=%lu (exp=128) %s\n", constInfo.blockShapeY,
                            (constInfo.blockShapeY == 128) ? "OK" : "ERR");
            if (constInfo.blockShapeY != 128) errCount++;
            BSA_TILING_VERIFY_PRINTF("xBlocks=%u (exp=%u) %s\n", constInfo.xBlocks, expXBlocks,
                            (constInfo.xBlocks == expXBlocks) ? "OK" : "ERR");
            if (constInfo.xBlocks != expXBlocks) errCount++;
            BSA_TILING_VERIFY_PRINTF("yBlocks=%u (exp=%u) %s\n", constInfo.yBlocks, expYBlocks,
                            (constInfo.yBlocks == expYBlocks) ? "OK" : "ERR");
            if (constInfo.yBlocks != expYBlocks) errCount++;
            BSA_TILING_VERIFY_PRINTF("scaleValue=%f\n", constInfo.scaleValue);
            BSA_TILING_VERIFY_PRINTF("sparsity=%f\n", constInfo.sparsity);
            BSA_TILING_VERIFY_PRINTF("topKValue=%lu (nonzero=%s)\n", constInfo.topKValue,
                            (constInfo.topKValue > 0) ? "OK" : "ERR");
            if (constInfo.topKValue == 0) errCount++;
            BSA_TILING_VERIFY_PRINTF("sparsityMode=%u (exp=0) %s\n",
                            static_cast<uint32_t>(constInfo.sparsityMode),
                            (static_cast<uint32_t>(constInfo.sparsityMode) == 0) ? "OK" : "ERR");
            if (static_cast<uint32_t>(constInfo.sparsityMode) != 0) errCount++;
            BSA_TILING_VERIFY_PRINTF("queryLayout=%u (exp=1=BNSD) %s\n",
                            static_cast<uint32_t>(baseInfo.queryLayout),
                            (static_cast<uint32_t>(baseInfo.queryLayout) == 1) ? "OK" : "ERR");
            if (static_cast<uint32_t>(baseInfo.queryLayout) != 1) errCount++;
            BSA_TILING_VERIFY_PRINTF("kvLayout=%u (exp=1=BNSD) %s\n",
                            static_cast<uint32_t>(baseInfo.kvLayout),
                            (static_cast<uint32_t>(baseInfo.kvLayout) == 1) ? "OK" : "ERR");
            if (static_cast<uint32_t>(baseInfo.kvLayout) != 1) errCount++;
            BSA_TILING_VERIFY_PRINTF("useActualBlockLenQ=%u (exp=0) %s\n",
                            static_cast<uint32_t>(baseInfo.useActualBlockLenQ),
                            (static_cast<uint32_t>(baseInfo.useActualBlockLenQ) == 0) ? "OK" : "ERR");
            if (static_cast<uint32_t>(baseInfo.useActualBlockLenQ) != 0) errCount++;
            BSA_TILING_VERIFY_PRINTF("useActualBlockLenK=%u (exp=0) %s\n",
                            static_cast<uint32_t>(baseInfo.useActualBlockLenK),
                            (static_cast<uint32_t>(baseInfo.useActualBlockLenK) == 0) ? "OK" : "ERR");
            if (static_cast<uint32_t>(baseInfo.useActualBlockLenK) != 0) errCount++;
            BSA_TILING_VERIFY_PRINTF("qChunkSize=%u (exp=128) %s\n", constInfo.qChunkSize,
                            (constInfo.qChunkSize == 128) ? "OK" : "ERR");
            if (constInfo.qChunkSize != 128) errCount++;
            BSA_TILING_VERIFY_PRINTF("kChunkSize=%u (exp=128) %s\n", constInfo.kChunkSize,
                            (constInfo.kChunkSize == 128) ? "OK" : "ERR");
            if (constInfo.kChunkSize != 128) errCount++;

            BSA_TILING_VERIFY_PRINTF("[multiCoreParams]\n");
            BSA_TILING_VERIFY_PRINTF("coreNum=%u (exp=20) %s\n", constInfo.coreNum,
                            (constInfo.coreNum == 20) ? "OK" : "ERR");
            if (constInfo.coreNum != 20) errCount++;
            BSA_TILING_VERIFY_PRINTF("activeCoreNum=%u (exp=%u) %s\n",
                            constInfo.activeCoreNum, expActiveCoreNum,
                            (constInfo.activeCoreNum == expActiveCoreNum) ? "OK" : "ERR");
            if (constInfo.activeCoreNum != expActiveCoreNum) errCount++;
            BSA_TILING_VERIFY_PRINTF("rowsPerCore=%u (exp=%u) %s\n", constInfo.rowsPerCore, expRowsPerCore,
                            (constInfo.rowsPerCore == expRowsPerCore) ? "OK" : "ERR");
            if (constInfo.rowsPerCore != expRowsPerCore) errCount++;
            BSA_TILING_VERIFY_PRINTF("extraCores=%u (exp=%u) %s\n", constInfo.extraCores, expExtraCores,
                            (constInfo.extraCores == expExtraCores) ? "OK" : "ERR");
            if (constInfo.extraCores != expExtraCores) errCount++;
            BSA_TILING_VERIFY_PRINTF("totalRows=%u (exp=%u) %s\n", constInfo.totalRows, constInfo.xBlocks,
                            (constInfo.totalRows == constInfo.xBlocks) ? "OK" : "ERR");
            if (constInfo.totalRows != constInfo.xBlocks) errCount++;
            BSA_TILING_VERIFY_PRINTF("yBlocksPerCore=%u (exp=%u) %s\n", constInfo.yBlocksPerCore, expYBlocksPerCore,
                            (constInfo.yBlocksPerCore == expYBlocksPerCore) ? "OK" : "ERR");
            if (constInfo.yBlocksPerCore != expYBlocksPerCore) errCount++;
            BSA_TILING_VERIFY_PRINTF("extraYCores=%u (exp=%u) %s\n", constInfo.extraYCores, expExtraYCores,
                            (constInfo.extraYCores == expExtraYCores) ? "OK" : "ERR");
            if (constInfo.extraYCores != expExtraYCores) errCount++;

            BSA_TILING_VERIFY_PRINTF("[outputParams]\n");
            BSA_TILING_VERIFY_PRINTF("qCmpSize=%lu (exp=%lu) %s\n", outInfo.qCmpSize, expQCmpSize,
                            (outInfo.qCmpSize == expQCmpSize) ? "OK" : "ERR");
            if (outInfo.qCmpSize != expQCmpSize) errCount++;
            BSA_TILING_VERIFY_PRINTF("kCmpSize=%lu (exp=%lu) %s\n", outInfo.kCmpSize, expKCmpSize,
                            (outInfo.kCmpSize == expKCmpSize) ? "OK" : "ERR");
            if (outInfo.kCmpSize != expKCmpSize) errCount++;
            BSA_TILING_VERIFY_PRINTF("attnScoreSize=%lu (exp=%lu) %s\n", outInfo.attnScoreSize, expAttnScoreSize,
                            (outInfo.attnScoreSize == expAttnScoreSize) ? "OK" : "ERR");
            if (outInfo.attnScoreSize != expAttnScoreSize) errCount++;
            BSA_TILING_VERIFY_PRINTF("softmaxTmpSize=%lu (exp=%lu) %s\n", outInfo.softmaxTmpSize, expSoftmaxTmpSize,
                            (outInfo.softmaxTmpSize == expSoftmaxTmpSize) ? "OK" : "ERR");
            if (outInfo.softmaxTmpSize != expSoftmaxTmpSize) errCount++;
            BSA_TILING_VERIFY_PRINTF("topkWorkspaceSize=%lu (exp=%lu) %s\n", outInfo.topkWorkspaceSize, expTopkWsSize,
                            (outInfo.topkWorkspaceSize == expTopkWsSize) ? "OK" : "ERR");
            if (outInfo.topkWorkspaceSize != expTopkWsSize) errCount++;
            BSA_TILING_VERIFY_PRINTF("maskSize=%lu (exp=%lu) %s\n", outInfo.maskSize, expMaskSize,
                            (outInfo.maskSize == expMaskSize) ? "OK" : "ERR");
            if (outInfo.maskSize != expMaskSize) errCount++;
            uint64_t expTotalWs = expQCmpSize + expKCmpSize + expAttnScoreSize + expSoftmaxTmpSize + expTopkWsSize + expMaskSize;
            BSA_TILING_VERIFY_PRINTF("totalWorkspaceSize=%lu (exp=%lu) %s\n", outInfo.totalWorkspaceSize, expTotalWs,
                            (outInfo.totalWorkspaceSize == expTotalWs) ? "OK" : "ERR");
            if (outInfo.totalWorkspaceSize != expTotalWs) errCount++;

            BSA_TILING_VERIFY_PRINTF("[workspace offsets]\n");
            BSA_TILING_VERIFY_PRINTF("qCmpOffset=%lu (exp=0) %s\n", constInfo.qCmpOffset,
                            (constInfo.qCmpOffset == 0) ? "OK" : "ERR");
            if (constInfo.qCmpOffset != 0) errCount++;
            BSA_TILING_VERIFY_PRINTF("kCmpOffset=%lu (exp=%lu) %s\n", constInfo.kCmpOffset, expQCmpSize,
                            (constInfo.kCmpOffset == expQCmpSize) ? "OK" : "ERR");
            if (constInfo.kCmpOffset != expQCmpSize) errCount++;
            BSA_TILING_VERIFY_PRINTF("attnScoreOffset=%lu (exp=%lu) %s\n", constInfo.attnScoreOffset,
                            expQCmpSize + expKCmpSize,
                            (constInfo.attnScoreOffset == expQCmpSize + expKCmpSize) ? "OK" : "ERR");
            if (constInfo.attnScoreOffset != expQCmpSize + expKCmpSize) errCount++;
            BSA_TILING_VERIFY_PRINTF("softmaxTmpOffset=%lu (exp=%lu) %s\n", constInfo.softmaxTmpOffset,
                            expQCmpSize + expKCmpSize + expAttnScoreSize,
                            (constInfo.softmaxTmpOffset == expQCmpSize + expKCmpSize + expAttnScoreSize) ? "OK" : "ERR");
            if (constInfo.softmaxTmpOffset != expQCmpSize + expKCmpSize + expAttnScoreSize) errCount++;
            BSA_TILING_VERIFY_PRINTF("topkWorkspaceOffset=%lu (exp=%lu) %s\n", constInfo.topkWorkspaceOffset,
                            expQCmpSize + expKCmpSize + expAttnScoreSize + expSoftmaxTmpSize,
                            (constInfo.topkWorkspaceOffset == expQCmpSize + expKCmpSize + expAttnScoreSize + expSoftmaxTmpSize) ? "OK" : "ERR");
            if (constInfo.topkWorkspaceOffset != expQCmpSize + expKCmpSize + expAttnScoreSize + expSoftmaxTmpSize) errCount++;
            BSA_TILING_VERIFY_PRINTF("maskOffset=%lu (exp=%lu) %s\n", constInfo.maskOffset,
                            expQCmpSize + expKCmpSize + expAttnScoreSize + expSoftmaxTmpSize + outInfo.topkWorkspaceSize,
                            (constInfo.maskOffset == expQCmpSize + expKCmpSize + expAttnScoreSize + expSoftmaxTmpSize + outInfo.topkWorkspaceSize) ? "OK" : "ERR");
            if (constInfo.maskOffset != expQCmpSize + expKCmpSize + expAttnScoreSize + expSoftmaxTmpSize + outInfo.topkWorkspaceSize) errCount++;
            BSA_TILING_VERIFY_PRINTF("=== Verification: %u errors ===\n", errCount);
        }
    }
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::InitWorkspace(__gm__ uint8_t *workspace)
{
    auto &outInfo = tilingData->outputParams;

    qCmpGm.SetGlobalBuffer((__gm__ IN_T *)(workspace + constInfo.qCmpOffset),
                           outInfo.qCmpSize / sizeof(IN_T));
    kCmpGm.SetGlobalBuffer((__gm__ IN_T *)(workspace + constInfo.kCmpOffset),
                           outInfo.kCmpSize / sizeof(IN_T));
    attnScoreFp16Gm.SetGlobalBuffer((__gm__ half *)(workspace + constInfo.attnScoreOffset),
                                 outInfo.attnScoreSize / sizeof(half));
    scoreFp32Gm.SetGlobalBuffer((__gm__ T *)(workspace + constInfo.softmaxTmpOffset),
                                 outInfo.softmaxTmpSize / sizeof(T));
    topkWorkspaceGm.SetGlobalBuffer((__gm__ int32_t *)(workspace + constInfo.topkWorkspaceOffset),
                                    outInfo.topkWorkspaceSize / sizeof(int32_t));
    maskTmpGm.SetGlobalBuffer((__gm__ uint8_t *)(workspace + constInfo.maskOffset),
                              outInfo.maskSize / sizeof(uint8_t));

    SyncAll();
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::CalcMultiCoreOffset(uint32_t &startRow, uint32_t &endRow)
{
    uint32_t coreIdx = constInfo.aicIdx;
    uint32_t extraCores = constInfo.extraCores;
    uint32_t rowsPerCore = constInfo.rowsPerCore; // blocks

    if (coreIdx < extraCores || extraCores == 0) {
        startRow = coreIdx * rowsPerCore;
        endRow = startRow + rowsPerCore;
    } else {
        uint32_t baseRows = rowsPerCore - 1;
        startRow = extraCores * rowsPerCore + (coreIdx - extraCores) * baseRows;
        endRow = startRow + baseRows;
    }
    if (startRow >= constInfo.totalRows) {
        startRow = constInfo.totalRows;
        endRow = constInfo.totalRows;
    }
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::CalcKPoolingRange(uint32_t &startKBlock, uint32_t &endKBlock)
{
    uint32_t coreIdx = constInfo.aivIdx; // vec idx
    uint32_t extraYCores = constInfo.extraYCores;
    uint32_t yBlocksPerCore = constInfo.yBlocksPerCore;


    if (coreIdx < extraYCores || extraYCores == 0) {
        startKBlock = coreIdx * yBlocksPerCore;
        endKBlock = startKBlock + yBlocksPerCore;
    } else {
        uint32_t baseYRows = yBlocksPerCore - 1;
        startKBlock = extraYCores * yBlocksPerCore + (coreIdx - extraYCores) * baseYRows;
        endKBlock = startKBlock + baseYRows;
    }
    if (startKBlock >= constInfo.yBlocks) {
        startKBlock = constInfo.yBlocks;
        endKBlock = constInfo.yBlocks;
    }
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::Init(
    __gm__ uint8_t *query, __gm__ uint8_t *key,
    __gm__ uint8_t *blockShape, __gm__ uint8_t *postBlockShape,
    __gm__ uint8_t *actualSeqLensQ, __gm__ uint8_t *actualSeqLensKV,
    __gm__ uint8_t *actualBlockLenQ, __gm__ uint8_t *actualBlockLenKV,
    __gm__ uint8_t *maskOut,
    __gm__ uint8_t *qCompOut, __gm__ uint8_t *kCompOut,
    __gm__ uint8_t *scoreOut, __gm__ uint8_t *softmaxOut,
    __gm__ uint8_t *workspace,
    const optiling::BSASelectBlockMaskTilingData *__restrict tiling,
    TPipe *tPipe)
{
    pipe = tPipe;
    tilingData = tiling;
    InitConstInfo();

    queryGm.SetGlobalBuffer((__gm__ IN_T *)query);
    keyGm.SetGlobalBuffer((__gm__ IN_T *)key);
    if (blockShape != nullptr) {
        blockShapeGm.SetGlobalBuffer((__gm__ int64_t *)blockShape, 2);
    }
    if (actualSeqLensQ != nullptr) {
        actualSeqLensQGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLensQ, constInfo.batchSize);
    } else {
        actualSeqLensQGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLensQ, 0);
    }
    if (actualSeqLensKV != nullptr) {
        actualSeqLensKVGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLensKV, constInfo.batchSize);
    } else {
        actualSeqLensKVGm.SetGlobalBuffer((__gm__ int64_t *)actualSeqLensKV, 0);
    }
    if (actualBlockLenQ != nullptr) {
        uint32_t totalQBlocks = constInfo.batchSize * constInfo.xBlocks;
        actualBlockLenQGm.SetGlobalBuffer((__gm__ int64_t *)actualBlockLenQ, totalQBlocks);
    } else {
        actualBlockLenQGm.SetGlobalBuffer((__gm__ int64_t *)actualBlockLenQ, 0);
    }
    if (actualBlockLenKV != nullptr) {
        uint32_t totalKVBlocks = constInfo.batchSize * constInfo.yBlocks;
        actualBlockLenKVGm.SetGlobalBuffer((__gm__ int64_t *)actualBlockLenKV, totalKVBlocks);
    } else {
        actualBlockLenKVGm.SetGlobalBuffer((__gm__ int64_t *)actualBlockLenKV, 0);
    }
    maskOutGmU8.SetGlobalBuffer((__gm__ uint8_t *)maskOut);

    qCompOutPtr = qCompOut;
    kCompOutPtr = kCompOut;
    scoreOutPtr = scoreOut;
    softmaxOutPtr = softmaxOut;

    // Set up output GlobalTensors from pointers (for pool/softmax to write to)
    auto &outInfo = tilingData->outputParams;
    qCmpOutGm.SetGlobalBuffer((__gm__ T *)qCompOut);
    kCmpOutGm.SetGlobalBuffer((__gm__ T *)kCompOut);
    softmaxOutGm.SetGlobalBuffer((__gm__ half *)softmaxOut);
    socreFp32OutGm.SetGlobalBuffer((__gm__ T *)scoreOutPtr);

    InitWorkspace(workspace);

    if ASCEND_IS_AIV {
        vectorService.InitParams(constInfo, tilingData);
        vectorService.InitBuffers(pipe);
        vectorService.InitGM(qCmpGm, kCmpGm, attnScoreFp16Gm, scoreFp32Gm,
                             queryGm, keyGm, actualBlockLenQGm, actualBlockLenKVGm,
                             qCmpOutGm, kCmpOutGm, softmaxOutGm);
        radixTopKService.InitParams(constInfo, tilingData);
        radixTopKService.InitBuffers(pipe);
        radixTopKService.InitGM(attnScoreFp16Gm, topkWorkspaceGm, maskTmpGm, maskOutGmU8);
    } else if ASCEND_IS_AIC {
        matmulService.InitParams(constInfo);
        matmulService.InitBuffers(pipe);
        matmulService.InitGM(qCmpGm, kCmpGm, scoreFp32Gm, socreFp32OutGm);
    }
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::Process()
{
    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
        radixTopKService.AllocEventID();
    } else {
        matmulService.AllocEventID();
    }

    // 1行等于1 block
    // startRow: q_comp 的start行数
    // endRow: q_comp 的end行数
    uint32_t startRow, endRow;
    CalcMultiCoreOffset(startRow, endRow);

    // uint32_t startKBlock, endKBlock;
    // CalcKPoolingRange(startKBlock, endKBlock);

    for (uint32_t batchIdx = 0; batchIdx < constInfo.batchSize; batchIdx++) {
        for (uint32_t headIdx = 0; headIdx < constInfo.numHeads; headIdx++) {

            ProcessPoolingK(batchIdx, headIdx);
            AscendC::PipeBarrier<PIPE_ALL>();
            SyncAll<false>();

            for (uint32_t qChunkStart = startRow; qChunkStart < endRow;
                 qChunkStart += constInfo.qChunkSize) {

                uint32_t qChunkEnd = BSAMin(qChunkStart + constInfo.qChunkSize, endRow);
                uint32_t curQChunkSize = qChunkEnd - qChunkStart;

                // q 的行数
                uint32_t tokenStart = qChunkStart * constInfo.blockShapeX;
                uint32_t tokenEnd = BSAMin(qChunkEnd * constInfo.blockShapeX, constInfo.maxQSeqlen);
                ProcessPoolingQ(batchIdx, headIdx, tokenStart, tokenEnd);

                if ASCEND_IS_AIV {
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V1_TO_C1_FLAG[qChunkStart & 1]);
                }
                if ASCEND_IS_AIC {
                    CrossCoreWaitFlag(SYNC_V1_TO_C1_FLAG[qChunkStart & 1]);
                }

                ProcessMatmulSoftmax(batchIdx, headIdx, qChunkStart, curQChunkSize);

                ProcessSoftmaxSecondPass(qChunkStart, curQChunkSize, batchIdx, headIdx);
            }

            AscendC::PipeBarrier<PIPE_ALL>();
            SyncAll<false>();

            if ASCEND_IS_AIV {
                radixTopKService.ProcessRadixTopKAndWriteMask(batchIdx, headIdx);
            }
            SyncAll();
        }
    }

    if ASCEND_IS_AIV {
        vectorService.FreeEventID();
        radixTopKService.FreeEventID();
    } else {
        matmulService.FreeEventID();
    }
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::ProcessPoolingK(uint32_t batchIdx, uint32_t headIdx)
{
    if ASCEND_IS_AIV {
        if (constInfo.aivIdx >= constInfo.activeYVecCoreNum) {
            return;
        }
        uint32_t startKBlock, endKBlock;
        CalcKPoolingRange(startKBlock, endKBlock);

        for (uint32_t kBlockIdx = startKBlock; kBlockIdx < endKBlock; kBlockIdx++) {
            vectorService.PoolingSingleKBlock(batchIdx, headIdx, kBlockIdx,
                                              keyGm, actualBlockLenKVGm, kCmpGm, kCmpOutGm);
        }
    }
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::ProcessPoolingQ(
    uint32_t batchIdx, uint32_t headIdx, uint32_t tokenStart, uint32_t tokenEnd)
{
    if ASCEND_IS_AIV {
        uint32_t coreIdx = constInfo.aivIdx; // vec idx
        uint32_t subVecIdx = constInfo.aivIdx % AIC_AIV_RATIO;
        uint32_t qRows = tokenEnd - tokenStart;
        uint32_t qBlocks = (qRows + constInfo.blockShapeX - 1) / constInfo.blockShapeX; // blockShapeX行为1个block
        uint32_t excuteBlock = 0;
        uint32_t startQBlockIdx = 0;
        uint32_t endQBlockIdx = 0;

        // vec 0 and 1 进行分核
        if (subVecIdx == 0) {
            excuteBlock = qBlocks / AIC_AIV_RATIO + qBlocks % AIC_AIV_RATIO;
            startQBlockIdx = tokenStart / constInfo.blockShapeX;
            endQBlockIdx = startQBlockIdx + excuteBlock;
        } else {
            excuteBlock = qBlocks / AIC_AIV_RATIO;
            if (excuteBlock == 0) {
                // sub vec 1 没有分到数据
                return;
            }
            uint32_t subvec0ExcuteBlocks = qBlocks / AIC_AIV_RATIO + qBlocks % AIC_AIV_RATIO;
            startQBlockIdx = tokenStart / constInfo.blockShapeX + subvec0ExcuteBlocks;
            endQBlockIdx = startQBlockIdx + excuteBlock;
        }

        for (uint32_t qBlockIdx = startQBlockIdx; qBlockIdx < endQBlockIdx; qBlockIdx++) {
            vectorService.PoolingSingleQBlock(batchIdx, headIdx, qBlockIdx,
                                                queryGm, actualBlockLenQGm, qCmpGm, qCmpOutGm);
        }
    }
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::ProcessMatmulSoftmax(
    uint32_t batchIdx, uint32_t headIdx,
    uint32_t qChunkStart, uint32_t curQChunkSize)
{
    // printf("ProcessMatmulSoftmax\n");
    uint32_t loopChunkSize = CV_EXEC_RATIO * constInfo.kChunkSize;
    uint32_t flagId = 0;
    for (uint32_t kChunkStart = 0; kChunkStart < constInfo.yBlocks;
         kChunkStart += loopChunkSize) {

        uint32_t kChunkEnd = BSAMin(kChunkStart + loopChunkSize, constInfo.yBlocks);
        uint32_t curKChunkSize = kChunkEnd - kChunkStart; // 实际执行k的blocks, 即k_comp的行数

        if ASCEND_IS_AIC {
            matmulService.ComputeMatmulChunk(qChunkStart, curQChunkSize,
                                             kChunkStart, curKChunkSize, batchIdx, headIdx);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C1_TO_V1_FLAG[flagId % SYNC_C1_TO_V1_FLAG_NUMS]);
        }

        if ASCEND_IS_AIV {
            CrossCoreWaitFlag(SYNC_C1_TO_V1_FLAG[flagId % SYNC_C1_TO_V1_FLAG_NUMS]);
            // printf("qChunkStart: %d, curQChunkSize:  %d, kChunkStart: %d, curKChunkSize: %d\n",
            //     qChunkStart, curQChunkSize, kChunkStart, curKChunkSize);
            vectorService.OnlineSoftmaxFirstPassChunk(
                qChunkStart, curQChunkSize, kChunkStart, curKChunkSize);
        }
        flagId++;
    }
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::ProcessSoftmaxSecondPass(
    uint32_t qChunkStart, uint32_t curQChunkSize, uint32_t batchIdx, uint32_t headIdx)
{
    // printf("ProcessSoftmaxSecondPass\n");
    if ASCEND_IS_AIV {
        // 等上个步骤结束
        AscendC::PipeBarrier<PIPE_ALL>();
        uint32_t loopChunkSize = CV_EXEC_RATIO * constInfo.kChunkSize;
        for (uint32_t kChunkStart = 0; kChunkStart < constInfo.yBlocks; kChunkStart += loopChunkSize) {

            uint32_t kChunkEnd = BSAMin(kChunkStart + loopChunkSize, constInfo.yBlocks);
            uint32_t curKChunkSize = kChunkEnd - kChunkStart; // 实际执行k的blocks, 即k_comp的行数

            // printf("qChunkStart: %d, curQChunkSize:  %d, kChunkStart: %d, curKChunkSize: %d, batchIdx: %d, headIdx:  %d\n",
            //         qChunkStart, curQChunkSize, kChunkStart, curKChunkSize, batchIdx, headIdx);
            vectorService.SoftmaxSecondPassAndCast(qChunkStart, curQChunkSize, kChunkStart, curKChunkSize, batchIdx, headIdx);
        }
    }
}

#undef BSA_TILING_VERIFY_PRINTF

#endif // BSA_SELECT_BLOCK_MASK_BASE_H
