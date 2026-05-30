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
                                                uint32_t qChunkStart, uint32_t curQChunkSize,
                                                T &mFinal, T &lFinal);
    __aicore__ inline void ProcessSoftmaxSecondPass(uint32_t qChunkStart, uint32_t qChunkEnd,
                                                     T mFinal, T lFinal);

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
    GlobalTensor<OUT_T> maskOutGm;

    GlobalTensor<IN_T> qCmpGm, kCmpGm;
    GlobalTensor<MM_OUT_T> attnScoreGm;
    GlobalTensor<T> softmaxTmpGm;
    GlobalTensor<uint8_t> maskTmpGm;
    GlobalTensor<int32_t> topkWorkspaceGm;
};

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::InitConstInfo()
{
    AscendC::printf("[InitConstInfo] Enter, BlockIdx=%u\n", GetBlockIdx());
    
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

    auto &outInfo = tilingData->outputParams;
    constInfo.qCmpOffset = 0;
    constInfo.kCmpOffset = constInfo.qCmpOffset + outInfo.qCmpSize;
    constInfo.attnScoreOffset = constInfo.kCmpOffset + outInfo.kCmpSize;
    constInfo.softmaxTmpOffset = constInfo.attnScoreOffset + outInfo.attnScoreSize;
    constInfo.topkWorkspaceOffset = constInfo.softmaxTmpOffset + outInfo.softmaxTmpSize;
    constInfo.maskOffset = constInfo.topkWorkspaceOffset + outInfo.topkWorkspaceSize;

    if ASCEND_IS_AIV {
        if (GetBlockIdx() == 0) {
            AscendC::printf("=== BSA Tiling Parameters (core 0) ===\n");
            AscendC::printf("batchSize=%u, numHeads=%u, maxQSeqlen=%u, maxKvSeqlen=%u\n",
                            constInfo.batchSize, constInfo.numHeads, constInfo.maxQSeqlen, constInfo.maxKvSeqlen);
            AscendC::printf("dSize=%u, blockShapeX=%lu, blockShapeY=%lu\n",
                            constInfo.dSize, constInfo.blockShapeX, constInfo.blockShapeY);
            AscendC::printf("xBlocks=%u, yBlocks=%u, topKValue=%u, sparsityMode=%u\n",
                            constInfo.xBlocks, constInfo.yBlocks, constInfo.topKValue,
                            static_cast<uint32_t>(constInfo.sparsityMode));
            AscendC::printf("scaleValue=%f, sparsity=%f\n",
                            constInfo.scaleValue, constInfo.sparsity);
            AscendC::printf("coreNum=%u, activeCoreNum=%u, rowsPerCore=%u, extraCores=%u, totalRows=%u\n",
                            constInfo.coreNum, constInfo.activeCoreNum, constInfo.rowsPerCore,
                            constInfo.extraCores, constInfo.totalRows);
            AscendC::printf("yBlocksPerCore=%u, extraYCores=%u, qChunkSize=%u, kChunkSize=%u\n",
                            constInfo.yBlocksPerCore, constInfo.extraYCores,
                            constInfo.qChunkSize, constInfo.kChunkSize);
            AscendC::printf("qCmpOffset=%lu, kCmpOffset=%lu, attnScoreOffset=%lu\n",
                            constInfo.qCmpOffset, constInfo.kCmpOffset, constInfo.attnScoreOffset);
            AscendC::printf("softmaxTmpOffset=%lu, topkWorkspaceOffset=%lu, maskOffset=%lu\n",
                            constInfo.softmaxTmpOffset, constInfo.topkWorkspaceOffset, constInfo.maskOffset);
            AscendC::printf("=== End Tiling Parameters ===\n");
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
    attnScoreGm.SetGlobalBuffer((__gm__ MM_OUT_T *)(workspace + constInfo.attnScoreOffset),
                                outInfo.attnScoreSize / sizeof(MM_OUT_T));
    softmaxTmpGm.SetGlobalBuffer((__gm__ T *)(workspace + constInfo.softmaxTmpOffset),
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
    uint32_t rowsPerCore = constInfo.rowsPerCore;

    if (coreIdx < extraCores) {
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
    uint32_t coreIdx = constInfo.aicIdx;
    uint32_t extraYCores = constInfo.extraYCores;
    uint32_t yBlocksPerCore = constInfo.yBlocksPerCore;

    if (coreIdx < extraYCores) {
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
    maskOutGm.SetGlobalBuffer((__gm__ OUT_T *)maskOut);

    InitWorkspace(workspace);

    if ASCEND_IS_AIV {
        vectorService.InitParams(constInfo, tilingData);
        vectorService.InitBuffers(pipe);
        vectorService.InitGM(qCmpGm, kCmpGm, attnScoreGm, softmaxTmpGm, maskTmpGm,
                             queryGm, keyGm, actualBlockLenQGm, actualBlockLenKVGm, maskOutGm);
        radixTopKService.InitParams(constInfo, tilingData);
        radixTopKService.InitBuffers(pipe);
        radixTopKService.InitGM(attnScoreGm, topkWorkspaceGm, maskTmpGm);
    } else if ASCEND_IS_AIC {
        matmulService.InitParams(constInfo);
        matmulService.InitBuffers(pipe);
        matmulService.InitGM(qCmpGm, kCmpGm, attnScoreGm);
    }
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::Process()
{
    AscendC::printf("[Process] Enter, BlockIdx=%u\n", GetBlockIdx());
    
    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
        radixTopKService.AllocEventID();
    } else {
        matmulService.AllocEventID();
    }

    uint32_t startRow, endRow;
    CalcMultiCoreOffset(startRow, endRow);

    uint32_t startKBlock, endKBlock;
    CalcKPoolingRange(startKBlock, endKBlock);

    if ASCEND_IS_AIV {
        AscendC::printf("[Core %u] startRow=%u, endRow=%u, startKBlock=%u, endKBlock=%u\n",
                        GetBlockIdx(), startRow, endRow, startKBlock, endKBlock);
    }

    for (uint32_t batchIdx = 0; batchIdx < constInfo.batchSize; batchIdx++) {
        for (uint32_t headIdx = 0; headIdx < constInfo.numHeads; headIdx++) {

            ProcessPoolingK(batchIdx, headIdx);
            SyncAll<false>();

            for (uint32_t qChunkStart = startRow; qChunkStart < endRow;
                 qChunkStart += constInfo.qChunkSize) {

                uint32_t qChunkEnd = BSAMin(qChunkStart + constInfo.qChunkSize, endRow);
                uint32_t curQChunkSize = qChunkEnd - qChunkStart;

                uint32_t tokenStart = qChunkStart * constInfo.blockShapeX;
                uint32_t tokenEnd = BSAMin(qChunkEnd * constInfo.blockShapeX, constInfo.maxQSeqlen);
                ProcessPoolingQ(batchIdx, headIdx, tokenStart, tokenEnd);

                if ASCEND_IS_AIV {
                    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V1_TO_C1_FLAG[qChunkStart & 1]);
                }
                if ASCEND_IS_AIC {
                    CrossCoreWaitFlag(SYNC_V1_TO_C1_FLAG[qChunkStart & 1]);
                }

                T mFinal = SOFTMAX_NEG_INF;
                T lFinal = 0.0f;
                ProcessMatmulSoftmax(batchIdx, headIdx, qChunkStart, curQChunkSize, mFinal, lFinal);

                ProcessSoftmaxSecondPass(qChunkStart, qChunkEnd, mFinal, lFinal);
            }
            SyncAll<false>();

            if ASCEND_IS_AIV {
                radixTopKService.ProcessRadixTopK(batchIdx, headIdx);
            }
            SyncAll();

            if ASCEND_IS_AIV {
                vectorService.ProcessMaskGen(batchIdx, headIdx);
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
        uint32_t startKBlock, endKBlock;
        CalcKPoolingRange(startKBlock, endKBlock);
        for (uint32_t kBlockIdx = startKBlock; kBlockIdx < endKBlock; kBlockIdx++) {
            vectorService.PoolingSingleKBlock(batchIdx, headIdx, kBlockIdx,
                                              keyGm, actualBlockLenKVGm, kCmpGm);
        }
    }
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::ProcessPoolingQ(
    uint32_t batchIdx, uint32_t headIdx, uint32_t tokenStart, uint32_t tokenEnd)
{
    if ASCEND_IS_AIV {
        for (uint32_t tokenIdx = tokenStart; tokenIdx < tokenEnd;
             tokenIdx += constInfo.blockShapeX) {
            uint32_t qBlockIdx = tokenIdx / constInfo.blockShapeX;
            vectorService.PoolingSingleQBlock(batchIdx, headIdx, qBlockIdx,
                                              queryGm, actualBlockLenQGm, qCmpGm);
        }
    }
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::ProcessMatmulSoftmax(
    uint32_t batchIdx, uint32_t headIdx,
    uint32_t qChunkStart, uint32_t curQChunkSize,
    T &mFinal, T &lFinal)
{
    for (uint32_t kChunkStart = 0; kChunkStart < constInfo.yBlocks;
         kChunkStart += constInfo.kChunkSize) {

        uint32_t kChunkEnd = BSAMin(kChunkStart + constInfo.kChunkSize, constInfo.yBlocks);
        uint32_t curKChunkSize = kChunkEnd - kChunkStart;

        if ASCEND_IS_AIC {
            matmulService.ComputeMatmulChunk(qChunkStart, curQChunkSize,
                                             kChunkStart, curKChunkSize);
            CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C1_TO_V1_FLAG[kChunkStart & 1]);
        }

        if ASCEND_IS_AIV {
            CrossCoreWaitFlag(SYNC_C1_TO_V1_FLAG[kChunkStart & 1]);
            vectorService.OnlineSoftmaxFirstPassChunk(
                qChunkStart, curQChunkSize, kChunkStart, curKChunkSize,
                mFinal, lFinal);
        }
    }
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::ProcessSoftmaxSecondPass(
    uint32_t qChunkStart, uint32_t qChunkEnd, T mFinal, T lFinal)
{
    if ASCEND_IS_AIV {
        for (uint32_t qBlockIdx = qChunkStart; qBlockIdx < qChunkEnd; qBlockIdx++) {
            vectorService.SoftmaxSecondPassAndCast(qBlockIdx, mFinal, lFinal);
        }
    }
}

#endif // BSA_SELECT_BLOCK_MASK_BASE_H
