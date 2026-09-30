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
#include "bsa_vector_service.h"
#include "bsa_matmul_service.h"
#include "bsa_radix_topk_service.h"

using namespace matmul;
using AscendC::CacheMode;
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename BSAT>
class BSASelectBlockMaskBase {
public:
    using T = float;
    using IN_T = typename BSAT::inputT;
    using OUT_T = typename BSAT::outputT;
    using MM_OUT_T = T;
    using POOL_OUT_T = half;
    using SFTMAX_OUT_T = half;
    static constexpr BSALayout LAYOUT_Q = BSAT::layoutQ;
    static constexpr BSALayout LAYOUT_KV = BSAT::layoutKV;

    __aicore__ inline BSASelectBlockMaskBase(){};
    __aicore__ inline void Init(__gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* blockShape,
                                __gm__ uint8_t* postBlockShape, __gm__ uint8_t* actualSeqLensQ,
                                __gm__ uint8_t* actualSeqLensKV, __gm__ uint8_t* actualBlockLenQ,
                                __gm__ uint8_t* actualBlockLenKV, __gm__ uint8_t* maskOut, __gm__ uint8_t* workspace,
                                const optiling::BSASelectBlockMaskTilingData* __restrict tiling, TPipe* tPipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitConstInfo();
    __aicore__ inline void InitWorkspace(__gm__ uint8_t* workspace);
    __aicore__ inline void CalcMultiCoreOffset(uint32_t& startRow, uint32_t& endRow);
    __aicore__ inline void CalcKPoolingRange(uint32_t& startKBlock, uint32_t& endKBlock);
    // D2: dynamic partition by validYBlocks
    __aicore__ inline void CalcKPoolingRangeValid(uint32_t& startKBlock, uint32_t& endKBlock, uint32_t validYBlocks);
    // D2: dynamic partition by validXBlocks
    __aicore__ inline void CalcMultiCoreOffsetValid(uint32_t& startRow, uint32_t& endRow, uint32_t validXBlocks);
    __aicore__ inline void ProcessPoolingK(uint32_t batchIdx, uint32_t headIdx, uint64_t seqPrefixSumKV,
                                           uint64_t blockPrefixSumKV, uint32_t validYBlocks);
    __aicore__ inline void ProcessPoolingQ(uint32_t batchIdx, uint32_t headIdx, uint32_t tokenStart, uint32_t tokenEnd,
                                           uint64_t seqPrefixSumQ, uint64_t blockPrefixSumQ);
    __aicore__ inline void ProcessMatmulSoftmax(uint32_t batchIdx, uint32_t headIdx, uint32_t qChunkStart,
                                                uint32_t curQChunkSize, uint32_t validXBlocks, uint32_t validYBlocks);
    __aicore__ inline void ProcessSoftmaxSecondPass(uint32_t qChunkStart, uint32_t curQChunkSize, uint32_t batchIdx,
                                                    uint32_t headIdx, uint32_t validXBlocks, uint32_t validYBlocks);

    TPipe* pipe = nullptr;
    const optiling::BSASelectBlockMaskTilingData* __restrict tilingData = nullptr;
    BSAConstInfo constInfo;

    BSAMatmulService<BSAT> matmulService;
    BSAVectorService<BSAT> vectorService;
    BSARadixTopKService<BSAT> radixTopKService;

    GlobalTensor<IN_T> queryGm, keyGm;
    GlobalTensor<int64_t> blockShapeGm;
    GlobalTensor<int64_t> actualSeqLensQGm, actualSeqLensKVGm;
    GlobalTensor<int64_t> actualBlockLenQGm, actualBlockLenKVGm;
    GlobalTensor<uint8_t> maskOutGmU8;

    GlobalTensor<POOL_OUT_T> qCmpGm, kCmpGm;
    GlobalTensor<SFTMAX_OUT_T> attnScoreFp16Gm;
    GlobalTensor<T> postStagingGm;
    GlobalTensor<T> scoreFp32Gm;
    GlobalTensor<T> partSumGm;
    GlobalTensor<T> partMaxGm;
    GlobalTensor<int32_t> topkWorkspaceGm;
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

    auto& baseInfo = tilingData->baseParams;
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

    auto& mcInfo = tilingData->multiCoreParams;
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

    auto& outInfo = tilingData->outputParams;
    constInfo.qCmpOffset = 0;
    constInfo.kCmpOffset = constInfo.qCmpOffset + outInfo.qCmpSize;
    constInfo.attnScoreOffset = constInfo.kCmpOffset + outInfo.kCmpSize;
    // 统一单槽：post 模式 post-pool 折叠分写入 attnScore 槽（B/N 顺序外循环复用），postStaging 紧随其后
    constInfo.postStagingOffset = (constInfo.attnScoreOffset + outInfo.attnScoreSize + 511) / 512 * 512;
    constInfo.softmaxTmpOffset = (constInfo.postStagingOffset + outInfo.postStagingSize + 511) / 512 * 512;
    constInfo.topkWorkspaceOffset = constInfo.softmaxTmpOffset + outInfo.softmaxTmpSize;
    constInfo.partSumOffset = constInfo.topkWorkspaceOffset + outInfo.topkWorkspaceSize;
    constInfo.partMaxOffset = constInfo.partSumOffset + outInfo.partSumSize;

    constInfo.usePostBlockShape = baseInfo.usePostBlockShape;
    constInfo.debugOldPass2 = 0; // 诊断开关：置 1 走旧两趟（对照精度基线）
    constInfo.postBlockShapeX = baseInfo.postBlockShapeX;
    constInfo.postBlockShapeY = baseInfo.postBlockShapeY;
    constInfo.postXBlocks = baseInfo.postXBlocks;
    constInfo.postYBlocks = baseInfo.postYBlocks;
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::InitWorkspace(__gm__ uint8_t* workspace)
{
    auto& outInfo = tilingData->outputParams;

    qCmpGm.SetGlobalBuffer((__gm__ POOL_OUT_T*)(workspace + constInfo.qCmpOffset),
                           outInfo.qCmpSize / sizeof(POOL_OUT_T));
    kCmpGm.SetGlobalBuffer((__gm__ POOL_OUT_T*)(workspace + constInfo.kCmpOffset),
                           outInfo.kCmpSize / sizeof(POOL_OUT_T));
    attnScoreFp16Gm.SetGlobalBuffer((__gm__ SFTMAX_OUT_T*)(workspace + constInfo.attnScoreOffset),
                                    outInfo.attnScoreSize / sizeof(SFTMAX_OUT_T));
    if (constInfo.usePostBlockShape) {
        postStagingGm.SetGlobalBuffer((__gm__ T*)(workspace + constInfo.postStagingOffset),
                                      outInfo.postStagingSize / sizeof(T));
    }
    scoreFp32Gm.SetGlobalBuffer((__gm__ T*)(workspace + constInfo.softmaxTmpOffset),
                                outInfo.softmaxTmpSize / sizeof(T));
    partSumGm.SetGlobalBuffer((__gm__ T*)(workspace + constInfo.partSumOffset), outInfo.partSumSize / sizeof(T));
    partMaxGm.SetGlobalBuffer((__gm__ T*)(workspace + constInfo.partMaxOffset), outInfo.partMaxSize / sizeof(T));
    topkWorkspaceGm.SetGlobalBuffer((__gm__ int32_t*)(workspace + constInfo.topkWorkspaceOffset),
                                    outInfo.topkWorkspaceSize / sizeof(int32_t));

    SyncAll();
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::CalcMultiCoreOffset(uint32_t& startRow, uint32_t& endRow)
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
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::CalcKPoolingRange(uint32_t& startKBlock, uint32_t& endKBlock)
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

// D2: dynamic partition by validYBlocks (early-return style)
template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::CalcKPoolingRangeValid(uint32_t& startKBlock, uint32_t& endKBlock,
                                                                            uint32_t validYBlocks)
{
    if (validYBlocks == 0) {
        startKBlock = 0;
        endKBlock = 0;
        return;
    }

    uint32_t coreIdx = constInfo.aivIdx;
    uint32_t activeYVecCoreNum = constInfo.activeYVecCoreNum;
    uint32_t actualActiveCores = BSAMin(validYBlocks, activeYVecCoreNum);

    if (coreIdx >= actualActiveCores) {
        startKBlock = 0;
        endKBlock = 0;
        return;
    }

    uint32_t baseRows = validYBlocks / actualActiveCores;
    uint32_t extraCores = validYBlocks % actualActiveCores;
    uint32_t rowsPerCore = baseRows + (extraCores > 0 ? 1 : 0);

    if (extraCores == 0) {
        startKBlock = coreIdx * rowsPerCore;
        endKBlock = startKBlock + rowsPerCore;
        return;
    }

    if (coreIdx < extraCores) {
        startKBlock = coreIdx * rowsPerCore;
        endKBlock = startKBlock + rowsPerCore;
        return;
    }

    uint32_t baseYRows = rowsPerCore - 1;
    startKBlock = extraCores * rowsPerCore + (coreIdx - extraCores) * baseYRows;
    endKBlock = startKBlock + baseYRows;
}

// D2: dynamic partition by validXBlocks (early-return style)
// post 模式下按 postXBlock 组对齐切分：保证跨 chunk 的 post 行组片元不跨核（staging 同核合并）
template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::CalcMultiCoreOffsetValid(uint32_t& startRow, uint32_t& endRow,
                                                                              uint32_t validXBlocks)
{
    if (validXBlocks == 0) {
        startRow = 0;
        endRow = 0;
        return;
    }

    uint32_t coreIdx = constInfo.aicIdx;
    uint32_t activeCoreNum = constInfo.activeCoreNum;
    uint32_t actualActiveCores = BSAMin(validXBlocks, activeCoreNum);

    if (coreIdx >= actualActiveCores) {
        startRow = 0;
        endRow = 0;
        return;
    }

    if (constInfo.usePostBlockShape) {
        // 以 postXBlock 组为最小分配单位：组不跨核，负载差 ≤ postXBlock-1 行
        uint32_t postXBlock = constInfo.postBlockShapeX;
        uint32_t totalGroups = BSACeilDiv(validXBlocks, postXBlock);
        uint32_t groupsPerCore = BSACeilDiv(totalGroups, actualActiveCores);
        startRow = coreIdx * groupsPerCore * postXBlock;
        endRow = BSAMin(startRow + groupsPerCore * postXBlock, validXBlocks);
        return;
    }

    uint32_t baseRows = validXBlocks / actualActiveCores;
    uint32_t extraCores = validXBlocks % actualActiveCores;
    uint32_t rowsPerCore = baseRows + (extraCores > 0 ? 1 : 0);

    if (extraCores == 0) {
        startRow = coreIdx * rowsPerCore;
        endRow = startRow + rowsPerCore;
        return;
    }

    if (coreIdx < extraCores) {
        startRow = coreIdx * rowsPerCore;
        endRow = startRow + rowsPerCore;
        return;
    }

    uint32_t baseYRows = rowsPerCore - 1;
    startRow = extraCores * rowsPerCore + (coreIdx - extraCores) * baseYRows;
    endRow = startRow + baseYRows;
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::Init(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* blockShape, __gm__ uint8_t* postBlockShape,
    __gm__ uint8_t* actualSeqLensQ, __gm__ uint8_t* actualSeqLensKV, __gm__ uint8_t* actualBlockLenQ,
    __gm__ uint8_t* actualBlockLenKV, __gm__ uint8_t* maskOut, __gm__ uint8_t* workspace,
    const optiling::BSASelectBlockMaskTilingData* __restrict tiling, TPipe* tPipe)
{
    pipe = tPipe;
    tilingData = tiling;
    InitConstInfo();

    queryGm.SetGlobalBuffer((__gm__ IN_T*)query);
    keyGm.SetGlobalBuffer((__gm__ IN_T*)key);
    if (blockShape != nullptr) {
        blockShapeGm.SetGlobalBuffer((__gm__ int64_t*)blockShape, 2);
    }
    if (actualSeqLensQ != nullptr) {
        actualSeqLensQGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLensQ, constInfo.batchSize);
    } else {
        actualSeqLensQGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLensQ, 0);
    }
    if (actualSeqLensKV != nullptr) {
        actualSeqLensKVGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLensKV, constInfo.batchSize);
    } else {
        actualSeqLensKVGm.SetGlobalBuffer((__gm__ int64_t*)actualSeqLensKV, 0);
    }
    if (actualBlockLenQ != nullptr) {
        uint32_t totalQBlocks = constInfo.batchSize * constInfo.xBlocks;
        actualBlockLenQGm.SetGlobalBuffer((__gm__ int64_t*)actualBlockLenQ, totalQBlocks);
    } else {
        actualBlockLenQGm.SetGlobalBuffer((__gm__ int64_t*)actualBlockLenQ, 0);
    }
    if (actualBlockLenKV != nullptr) {
        uint32_t totalKVBlocks = constInfo.batchSize * constInfo.yBlocks;
        actualBlockLenKVGm.SetGlobalBuffer((__gm__ int64_t*)actualBlockLenKV, totalKVBlocks);
    } else {
        actualBlockLenKVGm.SetGlobalBuffer((__gm__ int64_t*)actualBlockLenKV, 0);
    }
    maskOutGmU8.SetGlobalBuffer((__gm__ uint8_t*)maskOut);

    InitWorkspace(workspace);

    if ASCEND_IS_AIV {
        vectorService.InitParams(constInfo, tilingData);
        vectorService.InitBuffers(pipe);
        vectorService.InitGM(qCmpGm, kCmpGm, attnScoreFp16Gm, scoreFp32Gm, attnScoreFp16Gm, postStagingGm, queryGm,
                             keyGm, actualBlockLenQGm, actualBlockLenKVGm, actualSeqLensQGm, actualSeqLensKVGm,
                             partSumGm, partMaxGm);

        radixTopKService.InitParams(constInfo, tilingData);
        radixTopKService.InitBuffers(pipe);
        // post 启用时 TopK 在 attnScore 槽的粗粒度折叠分上选择并直接输出粗粒度 mask [B,N,postX,postY]
        if (constInfo.usePostBlockShape) {
            radixTopKService.InitGM(attnScoreFp16Gm, topkWorkspaceGm, maskOutGmU8);
        } else {
            radixTopKService.InitGM(attnScoreFp16Gm, topkWorkspaceGm, maskOutGmU8);
        }
    } else if ASCEND_IS_AIC {
        matmulService.InitParams(constInfo);
        matmulService.InitBuffers(pipe);
        matmulService.InitGM(qCmpGm, kCmpGm, scoreFp32Gm);
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
    // D2: startRow/endRow moved to batch loop (per-batch dynamic partition)

    // TND 布局下各 batch 的 token 在 GM 中变长拼接，需用前缀和计算 batch 偏移。
    // 在 batch 循环外初始化为 0，每个 batch 结束后累加该 batch 的实际序列长度，
    // 避免在每次 PoolingSingleQ/KBlock 调用中重复从 GM 读取 actualSeqLens 做 O(B²) 前缀和。
    // BNSD 布局下 seqPrefixSum 不被使用（pool 的 BNSD 分支忽略），保持为 0 不影响行为。
    uint64_t seqPrefixSumQ = 0;
    uint64_t seqPrefixSumKV = 0;
    uint64_t blockPrefixSumQ = 0;
    uint64_t blockPrefixSumKV = 0;

    for (uint32_t batchIdx = 0; batchIdx < constInfo.batchSize; batchIdx++) {
        // D1: compute valid block counts for this batch
        uint32_t curBatchSq, curBatchSkv;
        if (tilingData->baseParams.useActualSeqLenQ) {
            curBatchSq = static_cast<uint32_t>(actualSeqLensQGm.GetValue(batchIdx));
        } else {
            curBatchSq = constInfo.maxQSeqlen;
        }
        if (tilingData->baseParams.useActualSeqLenK) {
            curBatchSkv = static_cast<uint32_t>(actualSeqLensKVGm.GetValue(batchIdx));
        } else {
            curBatchSkv = constInfo.maxKvSeqlen;
        }
        uint32_t validXBlocks = BSACeilDiv(curBatchSq, static_cast<uint32_t>(constInfo.blockShapeX));
        uint32_t validYBlocks = BSACeilDiv(curBatchSkv, static_cast<uint32_t>(constInfo.blockShapeY));

        if ASCEND_IS_AIV {
            radixTopKService.UpdateRuntimeParams(validXBlocks, validYBlocks);
        }

        // D2: dynamic partition by validXBlocks
        uint32_t startRow, endRow;
        CalcMultiCoreOffsetValid(startRow, endRow, validXBlocks);

        for (uint32_t headIdx = 0; headIdx < constInfo.numHeads; headIdx++) {
#if 1 // TODO(perf-test): 跳过 poolK，测完恢复
            ProcessPoolingK(batchIdx, headIdx, seqPrefixSumKV, blockPrefixSumKV, validYBlocks);
            AscendC::PipeBarrier<PIPE_ALL>();
            SyncAll<false>();
#endif

            // D2: Q chunk loop with dynamic partition (no truncation needed)
            // chunk 边界重排：poolQ(c)+V1C1(c) 提到 finalize(c-1) 之前——AIC 在 AIV 收尾期间
            // 完成 Q/K 装载并预产出 t0/t1（填充期无需回执），消除 AIV 每 chunk 的 ramp-up 空转。
            // UB overlay 安全：poolQ→finalize 靠 ProcessSoftmaxSecondPass 入口 PipeBarrier<PIPE_ALL>；
            // finalize→tiles 靠 pass1 入口 MTE3_MTE2 / 子带 V_MTE2 屏障。
            bool hasPrevChunk = false;
            uint32_t prevQChunkStart = 0;
            uint32_t prevCurQChunkSize = 0;
            for (uint32_t qChunkStart = startRow; qChunkStart < endRow; qChunkStart += constInfo.qChunkSize) {
                uint32_t qChunkEnd = BSAMin(qChunkStart + constInfo.qChunkSize, endRow);
                uint32_t curQChunkSize = qChunkEnd - qChunkStart;

                uint32_t tokenStart = qChunkStart * constInfo.blockShapeX;
                uint32_t tokenEnd = BSAMin(qChunkEnd * constInfo.blockShapeX, curBatchSq);
#if 1 // TODO(perf-test): 纯 matmul+softmax+post pool 形态，poolQ 计算跳过、旗标保留，测完恢复
                ProcessPoolingQ(batchIdx, headIdx, tokenStart, tokenEnd, seqPrefixSumQ, blockPrefixSumQ);
#endif

                if ASCEND_IS_AIV {
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V1_TO_C1_FLAG[qChunkStart & 1]);
                }
                if ASCEND_IS_AIC {
                    CrossCoreWaitFlag(SYNC_V1_TO_C1_FLAG[qChunkStart & 1]);
                }

                if (constInfo.usePostBlockShape) {
                    // 原始交错调度：finalize(c-1) 与 pass1(c) 重叠
                    if (hasPrevChunk) {
                        ProcessSoftmaxSecondPass(prevQChunkStart, prevCurQChunkSize, batchIdx, headIdx, validXBlocks,
                                                 validYBlocks);
                    }
                    ProcessMatmulSoftmax(batchIdx, headIdx, qChunkStart, curQChunkSize, validXBlocks, validYBlocks);
                } else {
                    ProcessMatmulSoftmax(batchIdx, headIdx, qChunkStart, curQChunkSize, validXBlocks, validYBlocks);
                    ProcessSoftmaxSecondPass(qChunkStart, curQChunkSize, batchIdx, headIdx, validXBlocks, validYBlocks);
                }

                prevQChunkStart = qChunkStart;
                prevCurQChunkSize = curQChunkSize;
                hasPrevChunk = true;
            }

            // 末 chunk 收尾（必须在 TopK 之前：attnScore 槽内折叠分是其输入）。
            // 仅 post 交错调度需要（循环内只跑了 prev chunk 的 finalize，末 chunk 的留到此处）；
            // 非 post 循环内已对每个 chunk 跑过 pass2，重复调用是幂等重写，徒增开销，
            // 且会多发一次延迟 receipt 破坏 Set/Wait 1:1（残留置位旗标 → 下轮背压提前放行）
            if (hasPrevChunk && constInfo.usePostBlockShape) {
                ProcessSoftmaxSecondPass(prevQChunkStart, prevCurQChunkSize, batchIdx, headIdx, validXBlocks,
                                         validYBlocks);
            }

#if 1 // TODO(perf-test): 跳过 TopK + 前置排空，测完恢复
            AscendC::PipeBarrier<PIPE_ALL>();
            SyncAll<false>(); // vector 全核同步（含 MTE3 排空语义）

            // TopK 前 dump attnScore（原始直读 GM）；最终移除
            if ASCEND_IS_AIV {
                if (GetBlockIdx() == 0) {
                    const uint64_t hOff = constInfo.usePostBlockShape ?
                                              (static_cast<uint64_t>(batchIdx) * constInfo.numHeads + headIdx) *
                                                  constInfo.postXBlocks * constInfo.postYBlocks :
                                              0;
                    const uint32_t dSize = constInfo.usePostBlockShape ?
                                               static_cast<uint32_t>(constInfo.postXBlocks * constInfo.postYBlocks) :
                                               static_cast<uint32_t>(constInfo.xBlocks) * validYBlocks;
                    DumpTensor(attnScoreFp16Gm[hOff],
                               400 + static_cast<uint32_t>(batchIdx) * constInfo.numHeads + headIdx, dSize);
                }
            }

            if ASCEND_IS_AIV {
                radixTopKService.ProcessRadixTopKAndWriteMask(batchIdx, headIdx);
            }
            SyncAll();
#endif
        }

        // 当前 batch 处理完毕，累加其序列长度为下一个 batch 准备前缀和。
        // 仅 TND 布局需要；BNSD 分支由 constexpr 消除，不会访问 actualSeqLens GM（BNSD 下可能为空）。
        if (batchIdx + 1 < constInfo.batchSize) {
            if constexpr (LAYOUT_Q == BSALayout::TND) {
                seqPrefixSumQ += static_cast<uint64_t>(actualSeqLensQGm.GetValue(batchIdx));
                blockPrefixSumQ += validXBlocks;
            }
            if constexpr (LAYOUT_KV == BSALayout::TND) {
                seqPrefixSumKV += static_cast<uint64_t>(actualSeqLensKVGm.GetValue(batchIdx));
                blockPrefixSumKV += validYBlocks;
            }
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
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::ProcessPoolingK(uint32_t batchIdx, uint32_t headIdx,
                                                                     uint64_t seqPrefixSumKV, uint64_t blockPrefixSumKV,
                                                                     uint32_t validYBlocks)
{
    if ASCEND_IS_AIV {
        if (constInfo.aivIdx >= constInfo.activeYVecCoreNum) {
            return;
        }
        uint32_t startKBlock, endKBlock;
        CalcKPoolingRangeValid(startKBlock, endKBlock, validYBlocks);

        // 批量快速路径：满块且无变长标记时整段一次批处理（转置读+单次ReduceMean）
        if (!vectorService.FastPoolingKRange(batchIdx, headIdx, startKBlock, endKBlock, seqPrefixSumKV)) {
            for (uint32_t kBlockIdx = startKBlock; kBlockIdx < endKBlock; kBlockIdx++) {
                vectorService.PoolingSingleKBlock(batchIdx, headIdx, kBlockIdx, keyGm, actualBlockLenKVGm, kCmpGm,
                                                  seqPrefixSumKV, blockPrefixSumKV);
            }
        }
    }
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::ProcessPoolingQ(uint32_t batchIdx, uint32_t headIdx,
                                                                     uint32_t tokenStart, uint32_t tokenEnd,
                                                                     uint64_t seqPrefixSumQ, uint64_t blockPrefixSumQ)
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

        // 批量快速路径：满块且无变长标记时本 sub-vec 段一次批处理（转置读+单次ReduceMean）
        if (!vectorService.FastPoolingQRange(batchIdx, headIdx, startQBlockIdx, endQBlockIdx, seqPrefixSumQ)) {
            for (uint32_t qBlockIdx = startQBlockIdx; qBlockIdx < endQBlockIdx; qBlockIdx++) {
                vectorService.PoolingSingleQBlock(batchIdx, headIdx, qBlockIdx, queryGm, actualBlockLenQGm, qCmpGm,
                                                  seqPrefixSumQ, blockPrefixSumQ);
            }
        }
    }
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::ProcessMatmulSoftmax(uint32_t batchIdx, uint32_t headIdx,
                                                                          uint32_t qChunkStart, uint32_t curQChunkSize,
                                                                          uint32_t validXBlocks, uint32_t validYBlocks)
{
    uint32_t loopChunkSize = CV_EXEC_RATIO * constInfo.kChunkSize;
    uint32_t flagId = 0;
    uint32_t totalKChunks = (validYBlocks + loopChunkSize - 1) / loopChunkSize;
    // D1: K loop truncated to validYBlocks
    for (uint32_t kChunkStart = 0; kChunkStart < validYBlocks; kChunkStart += loopChunkSize) {
        uint32_t kChunkEnd = BSAMin(kChunkStart + loopChunkSize, validYBlocks);
        uint32_t curKChunkSize = kChunkEnd - kChunkStart; // 实际执行k的blocks, 即k_comp的行数

        if ASCEND_IS_AIC {
            // 背压：领先满一个复用周期(9)前，等 AIV 第 flagId-9 个消费回执，
            // 保证同槽 C1_TO_V1 的上一次 Set 已被 Wait 消费清零，Set 永不合并丢通知
            if (flagId >= SYNC_C1_TO_V1_FLAG_NUMS) {
                CrossCoreWaitFlag(SYNC_V2_TO_C2_FLAG[flagId % SYNC_C1_TO_V1_FLAG_NUMS]);
            }
            matmulService.ComputeMatmulChunk(qChunkStart, curQChunkSize, kChunkStart, curKChunkSize, batchIdx, headIdx);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C1_TO_V1_FLAG[flagId % SYNC_C1_TO_V1_FLAG_NUMS]);
        }

        if ASCEND_IS_AIV {
            // （L2''）pass1 band 末预读时已提前 Wait 掉本 tile 的 ready（保证 AIC 写完才预读），
            // 此处跳过避免二次 Wait 挂死；Set/Wait 保持严格 1:1
            if (!vectorService.TakePreWaitedNextReady()) {
                CrossCoreWaitFlag(SYNC_C1_TO_V1_FLAG[flagId % SYNC_C1_TO_V1_FLAG_NUMS]);
            }
#if 1 // pass1 消费
            vectorService.OnlineSoftmaxFirstPassChunk(qChunkStart, curQChunkSize, kChunkStart, curKChunkSize,
                                                      validXBlocks, validYBlocks);
#endif
            // （receipt 前移）pass1 band 末已发 receipt（MTE2 队列空、零延迟），
            // 此处补发未发的情况（pass1 早退），保证 Set/Wait 严格 1:1。
            // 非 post / debugOldPass2 路径 pass2 还要重读本 tile：receipt 延迟到
            // ProcessSoftmaxSecondPass 的 kChunk 循环末尾发（槽的最后消费者读完才放行 AIC 覆写）
            if (!vectorService.TakeReceiptSent() && constInfo.usePostBlockShape && !constInfo.debugOldPass2) {
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(SYNC_V2_TO_C2_FLAG[flagId % SYNC_C1_TO_V1_FLAG_NUMS]);
            }
        }
        flagId++;
    }

    // qChunk 末尾排空回执环：AIC 循环内只消费了 C_0..C_{T-10}（flagId>=9 才 Wait），
    // 残留 C_{T-9}..C_{T-1}（T>=9 时 9 个，T<9 时 T 个）；flagId 跨 qChunk 重置为 0，
    // 若不清零，下个 qChunk 的 R 会错位吃到残留回执。j 取 [max(9,T), T+9) 恰好逐槽清零。
    if ASCEND_IS_AIC {
        for (uint32_t j = totalKChunks; j < totalKChunks + SYNC_C1_TO_V1_FLAG_NUMS; j++) {
            if (j >= SYNC_C1_TO_V1_FLAG_NUMS) {
                CrossCoreWaitFlag(SYNC_V2_TO_C2_FLAG[j % SYNC_C1_TO_V1_FLAG_NUMS]);
            }
        }
    }
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::ProcessSoftmaxSecondPass(uint32_t qChunkStart,
                                                                              uint32_t curQChunkSize, uint32_t batchIdx,
                                                                              uint32_t headIdx, uint32_t validXBlocks,
                                                                              uint32_t validYBlocks)
{
    if ASCEND_IS_AIV {
        AscendC::PipeBarrier<PIPE_ALL>();
        if (constInfo.usePostBlockShape && !constInfo.debugOldPass2) {
            // softmax 单趟化收尾：per-tile 部分和（pass1 已写出）× 修正因子 → 行折叠 → attnScore 槽
            vectorService.SoftmaxFinalize(qChunkStart, curQChunkSize, batchIdx, headIdx, validXBlocks, validYBlocks);
        } else {
            uint32_t loopChunkSize = CV_EXEC_RATIO * constInfo.kChunkSize;
            uint32_t flagId2 = 0;
            for (uint32_t kChunkStart = 0; kChunkStart < validYBlocks; kChunkStart += loopChunkSize) {
                uint32_t kChunkEnd = BSAMin(kChunkStart + loopChunkSize, validYBlocks);
                uint32_t curKChunkSize = kChunkEnd - kChunkStart;

                vectorService.SoftmaxSecondPassAndCast(qChunkStart, curQChunkSize, kChunkStart, curKChunkSize, batchIdx,
                                                       headIdx, validXBlocks, validYBlocks);
                // 延迟 receipt：老 pass2 是本 tile 的最后消费者，读完后 AIC 方可覆写环槽。
                // flagId2 与 ProcessMatmulSoftmax 的 flagId 同序（kChunk 主序），Set/Wait 保持 1:1；
                // pass2 早退核（无 slice）同样发满 T 个，防同核 AIC 排空 Wait 挂死
                CrossCoreSetFlag<SYNC_MODE, PIPE_MTE2>(SYNC_V2_TO_C2_FLAG[flagId2 % SYNC_C1_TO_V1_FLAG_NUMS]);
                flagId2++;
            }
        }
    }
}

#undef BSA_TILING_VERIFY_PRINTF

#endif // BSA_SELECT_BLOCK_MASK_BASE_H
