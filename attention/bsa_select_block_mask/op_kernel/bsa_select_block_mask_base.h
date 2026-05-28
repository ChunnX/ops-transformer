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
    GlobalTensor<half> attnScoreGm;
    GlobalTensor<T> softmaxTmpGm;
    GlobalTensor<uint8_t> maskTmpGm;
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
    constInfo.totalRows = mcInfo.totalRows;
    constInfo.qChunkSize = baseInfo.qChunkSize;
    constInfo.kChunkSize = baseInfo.kChunkSize;

    auto &outInfo = tilingData->outputParams;
    constInfo.qCmpOffset = 0;
    constInfo.kCmpOffset = constInfo.qCmpOffset + outInfo.qCmpSize;
    constInfo.attnScoreOffset = constInfo.kCmpOffset + outInfo.kCmpSize;
    constInfo.softmaxTmpOffset = constInfo.attnScoreOffset + outInfo.attnScoreSize;
    constInfo.topkWorkspaceOffset = constInfo.softmaxTmpOffset + outInfo.softmaxTmpSize;
    constInfo.maskOffset = constInfo.topkWorkspaceOffset + outInfo.topkWorkspaceSize;
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::InitWorkspace(__gm__ uint8_t *workspace)
{
    auto &outInfo = tilingData->outputParams;

    qCmpGm.SetGlobalBuffer((__gm__ IN_T *)(workspace + constInfo.qCmpOffset));
    kCmpGm.SetGlobalBuffer((__gm__ IN_T *)(workspace + constInfo.kCmpOffset));
    attnScoreGm.SetGlobalBuffer((__gm__ half *)(workspace + constInfo.attnScoreOffset));
    softmaxTmpGm.SetGlobalBuffer((__gm__ T *)(workspace + constInfo.softmaxTmpOffset));
    topkWorkspaceGm.SetGlobalBuffer((__gm__ int32_t *)(workspace + constInfo.topkWorkspaceOffset));
    maskTmpGm.SetGlobalBuffer((__gm__ uint8_t *)(workspace + constInfo.maskOffset));

    SyncAll();
}

template <typename BSAT>
__aicore__ inline void BSASelectBlockMaskBase<BSAT>::CalcMultiCoreOffset(uint32_t &startRow, uint32_t &endRow)
{
    uint32_t coreIdx = constInfo.aicIdx;
    startRow = coreIdx * constInfo.rowsPerCore;
    endRow = BSAMin(startRow + constInfo.rowsPerCore, constInfo.totalRows);
    if (startRow >= constInfo.totalRows) {
        startRow = constInfo.totalRows;
        endRow = constInfo.totalRows;
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
    if ASCEND_IS_AIV {
        vectorService.AllocEventID();
        radixTopKService.AllocEventID();
    } else {
        matmulService.AllocEventID();
    }

    uint32_t startRow, endRow;
    CalcMultiCoreOffset(startRow, endRow);

    for (uint32_t batchIdx = 0; batchIdx < constInfo.batchSize; batchIdx++) {
        for (uint32_t headIdx = 0; headIdx < constInfo.numHeads; headIdx++) {

            // ========== Stage 1a: Pooling Key (全核参与) ==========
            if ASCEND_IS_AIV {
                for (uint32_t kBlockIdx = 0; kBlockIdx < constInfo.yBlocks; kBlockIdx++) {
                    vectorService.PoolingSingleKBlock(batchIdx, headIdx, kBlockIdx,
                                                      keyGm, actualBlockLenKVGm, kCmpGm);
                }
            }
            SyncAll<false>();

            // ========== Stage 1b+2: Pooling Query + Matmul + OnlineSoftmax ==========
            for (uint32_t qChunkStart = startRow; qChunkStart < endRow;
                 qChunkStart += constInfo.qChunkSize) {

                uint32_t qChunkEnd = BSAMin(qChunkStart + constInfo.qChunkSize, endRow);
                uint32_t curQChunkSize = qChunkEnd - qChunkStart;

                // Pooling Query (仅AIV, 无需SyncAll)
                if ASCEND_IS_AIV {
                    for (uint32_t qBlockIdx = qChunkStart; qBlockIdx < qChunkEnd; qBlockIdx++) {
                        vectorService.PoolingSingleQBlock(batchIdx, headIdx, qBlockIdx,
                                                          queryGm, actualBlockLenQGm, qCmpGm);
                    }
                }

                // Matmul + OnlineSoftmax第一遍: 循环kBlock, 每次算qChunk×kChunk
                T mFinal = -1e38f;
                T lFinal = 0.0f;
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

                // OnlineSoftmax第二遍: 归一化 + Cast FP32→FP16
                if ASCEND_IS_AIV {
                    for (uint32_t qBlockIdx = qChunkStart; qBlockIdx < qChunkEnd; qBlockIdx++) {
                        vectorService.SoftmaxSecondPassAndCast(
                            qBlockIdx, mFinal, lFinal);
                    }
                }
            }
            SyncAll<false>();

            // ========== Stage 3: Radix TopK/BottomK ==========
            if ASCEND_IS_AIV {
                radixTopKService.ProcessRadixTopK(batchIdx, headIdx);
            }
            SyncAll<false>();

            // ========== Stage 4: Mask生成 ==========
            if ASCEND_IS_AIV {
                vectorService.ProcessMaskGen(batchIdx, headIdx);
            }
            SyncAll<false>();
        }
    }

    if ASCEND_IS_AIV {
        vectorService.FreeEventID();
        radixTopKService.FreeEventID();
    } else {
        matmulService.FreeEventID();
    }
}

#endif // BSA_SELECT_BLOCK_MASK_BASE_H
