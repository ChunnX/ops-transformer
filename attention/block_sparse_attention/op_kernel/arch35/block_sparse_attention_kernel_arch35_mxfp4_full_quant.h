/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../arch35/kernel_utils.hpp"

using namespace NpuArch;
using namespace tla;

namespace BsaKernelArch35 {

template <
    class EpilogueMask2Idx,
    class BlockMmadQK,
    class EpilogueOnlineSoftmax,
    class BlockMmadPV,
    class EpilogueRescaleO,
    class EpilogueComputePScale,
    class EpilogueCopyGlobalMaxUbToL1,
    class BlockMmadCopyGlobalMaxL1ToUB,
    Format qFormat,
    Format kvFormat,
    bool transposedMm1 = false>
class BsaMXFP4FullQuantKernelArch35 {
public:
    using ArchTag = typename BlockMmadPV::ArchTag;

    using ElementQ = typename BlockMmadQK::ElementA;
    using ElementK = typename BlockMmadQK::ElementB;
    using ElementS = typename EpilogueOnlineSoftmax::ElementInput;
    using ElementP = typename BlockMmadPV::ElementA;
    using ElementV = typename BlockMmadPV::ElementB;
    using ElementOTmp = typename BlockMmadPV::ElementC;
    using ElementO = typename EpilogueOnlineSoftmax::ElementInput;
    using ElementLse = typename EpilogueRescaleO::ElementLse;
    using ElementSparseMask = typename EpilogueMask2Idx::ElementSparseMask;
    using ElementSparseIdx = typename EpilogueMask2Idx::ElementSparseIdx;
    using ElementSparseCount = typename EpilogueMask2Idx::ElementSparseCount;
    using ElementGroupMax = typename EpilogueComputePScale::ElementGroupMax;
    using ElementScale = typename EpilogueComputePScale::ElementScale;
    using ElementLocalGlobalMax = typename EpilogueCopyGlobalMaxUbToL1::ElementLocalGlobalMax;

    using LayoutQ = layout::RowMajor;
    using LayoutK = layout::ColumnMajor;
    using LayoutS = layout::RowMajor;
    using LayoutP = layout::RowMajor;
    using LayoutV = layout::RowMajor;
    using LayoutO = layout::RowMajor;
    using LayoutLse = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;
    using LayoutLocalGlobalMax = layout::RowMajor;
    using LayoutSparseIdx = layout::RowMajor;
    using LayoutSparseCount = layout::RowMajor;

    using LayoutTagL1P = typename BlockMmadPV::LayoutTagL1A;

    static constexpr uint8_t SYNC_MODE_2 = 2;
    static constexpr uint8_t SYNC_MODE_4 = 4;
    // static constexpr uint16_t CROSS_CORE_SYNC_C1_V1[2] = {0, 1};
    static constexpr uint16_t CROSS_CORE_SYNC_V1_C1[2] = {9, 10};
    static constexpr uint16_t CROSS_CORE_SYNC_GMAX_UB_TO_L1 = 8;
    static constexpr uint16_t CROSS_CORE_SYNC_GMAX_L1_TO_UB = 7;
    static constexpr uint16_t CROSS_CORE_SYNC_PSCALE_C2 = 6;
    static constexpr uint16_t CROSS_CORE_SYNC_C2_V2 = 5;
    static constexpr uint16_t CROSS_CORE_SYNC_V2_C2 = 4;

    static constexpr uint32_t PV_RE_DELAY_N = 20;
    static constexpr uint32_t DELAY_PSCALE_N = 3;

    static constexpr uint32_t PRELOAD_TASK_CACHE_SIZE = PV_RE_DELAY_N + 1;
    static constexpr uint32_t TILE_GROUP_N = 16;

    static constexpr uint32_t PRE_LAUNCH = 2;
    static constexpr uint32_t MAX_CROSS_CORE_BUF_STAGES = PRE_LAUNCH + 1;
    static constexpr uint32_t UB_S_OTMP_BUF_STAGES = 2;

    // Methods
    __aicore__ inline
    BsaMXFP4FullQuantKernelArch35() {}

    __aicore__ inline
    void operator()(BsaFullQuantKernelParamsArch35 const &params)
    {
        __gm__ BlockSparseAttentionTilingData *bsaTilingData =
            reinterpret_cast<__gm__ BlockSparseAttentionTilingData *>(params.tiling);
        FetchBaseShapeInfo(bsaTilingData);
        CalcOnChipBufTileInfo(bsaTilingData);
        // global buffers
        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ *)params.query);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK *)params.key);
        AscendC::GlobalTensor<ElementK> gV;
        gV.SetGlobalBuffer((__gm__ ElementK *)params.value);
        AscendC::GlobalTensor<int64_t> gActualQseqlen;
        gActualQseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualSeqLengths);
        AscendC::GlobalTensor<int64_t> gActualKvseqlen;
        gActualKvseqlen.SetGlobalBuffer((__gm__ int64_t *)params.actualSeqLengthsKv);
        AscendC::GlobalTensor<uint8_t> gBlockSparseMask;
        gBlockSparseMask.SetGlobalBuffer((__gm__ uint8_t *)params.blockSparseMask);
        AscendC::GlobalTensor<float> gQDequantScale;
        gQDequantScale.SetGlobalBuffer((__gm__ float *)params.qDequantScale);
        AscendC::GlobalTensor<float> gKDequantScale;
        gKDequantScale.SetGlobalBuffer((__gm__ float *)params.kDequantScale);
        AscendC::GlobalTensor<float> gVDequantScale;
        gVDequantScale.SetGlobalBuffer((__gm__ float *)params.vDequantScale);
        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO *)params.attentionOut);
        AscendC::GlobalTensor<ElementLse> gLse;
        gLse.SetGlobalBuffer((__gm__ ElementLse *)params.lse);
        AscendC::GlobalTensor<ElementSparseIdx> gSparseIdx;
        gSparseIdx.SetGlobalBuffer((__gm__ ElementSparseIdx *)params.workSpace);
        AscendC::GlobalTensor<ElementSparseCount> gSparseCount;
        gSparseCount.SetGlobalBuffer((__gm__ ElementSparseCount *)(params.workSpace + sparseIdxSize_));
        // cross core data move dst buffers
        AscendC::LocalTensor<ElementP> l1PTensor[MAX_CROSS_CORE_BUF_STAGES];
        AscendC::LocalTensor<ElementS> ubSTensor[UB_S_OTMP_BUF_STAGES];
        AscendC::LocalTensor<ElementOTmp> ubOTmpTensor[UB_S_OTMP_BUF_STAGES];
        InitCrossCoreDstBuf(l1PTensor, ubSTensor, ubOTmpTensor);
        // core idx
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        // set reverse sync flags
        InitSyncFlags();
        // mask preprocess
        if ASCEND_IS_AIV {
            EpilogueMask2Idx epilogueMask2Idx(resource);
            uint32_t totalRowNumBlockMask = batch_ * qHeads_ * xBlockNumAligned_;
            epilogueMask2Idx(
                gBlockSparseMask, gSparseIdx, gSparseCount,
                totalRowNumBlockMask, yBlockNumAligned_, avgRowPerSubCore_, preActiveSubCoreNum_);
        }
        AscendC::SyncAll<false>();

        if ASCEND_IS_AIC {
            coreIdx = AscendC::GetBlockIdx();
            // BlockMmadQK blockMmadQK(resource, mm1L1TileHelper_);
            // BlockMmadPV blockMmadPV(resource, mm2L1AddrStart_, mm2L1TileHelper_);
        } else {
            coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
            // EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue_);
            // EpilogueRescaleO epilogueRescaleO(resource);
        }

        // runMainLoop
        uint64_t coreTileLoop = 0;

        uint64_t curTaskTileIdx = 0;
        uint64_t delay20TaskTileIdx = 0;
        uint64_t delay3TaskTileIdx = 0;

        uint32_t coreTaskId = 0;
        uint32_t delay20CoreTaskId = 0;
        uint32_t delay3CoreTaskId = 0;

        uint64_t preTaskTileSum = 0;
        uint64_t delay20preTaskTileSum = 0;
        uint64_t delay3preTaskTileSum = 0;

        uint64_t coreTaskNum = coreTaskNum_;
        coreTaskNum = (coreIdx < totalTaskNum_ % coreNum) ? (coreTaskNum + 1) : coreTaskNum;

        TaskInfo taskInfos[PRELOAD_TASK_CACHE_SIZE] = {};
        TileInfo tileInfos[PRELOAD_TASK_CACHE_SIZE] = {};

        while (coreTaskId < coreTaskNum || delay20CoreTaskId < coreTaskNum || delay3CoreTaskId < coreTaskNum)
        {
            // AscendC::PRINTF("loop:%llu \n", coreTileLoop);
            curTaskTileIdx = coreTileLoop - preTaskTileSum;
            delay20TaskTileIdx = coreTileLoop - PV_RE_DELAY_N - delay20preTaskTileSum;
            delay3TaskTileIdx = coreTileLoop - DELAY_PSCALE_N - delay3preTaskTileSum;

            TaskInfo &curTaskInfo = taskInfos[coreTaskId % PRELOAD_TASK_CACHE_SIZE];
            TaskInfo &delay20TaskInfo = taskInfos[delay20CoreTaskId % PRELOAD_TASK_CACHE_SIZE];
            TaskInfo &delay3TaskInfo = taskInfos[delay3CoreTaskId % PRELOAD_TASK_CACHE_SIZE];

            TileInfo &curTileInfo = tileInfos[coreTileLoop % PRELOAD_TASK_CACHE_SIZE];
            TileInfo &delay20TileInfo = tileInfos[(coreTileLoop - PV_RE_DELAY_N) % PRELOAD_TASK_CACHE_SIZE];
            TileInfo &delay3TileInfo = tileInfos[(coreTileLoop - DELAY_PSCALE_N) % PRELOAD_TASK_CACHE_SIZE];
            
            // 创建qs任务
            if (curTaskTileIdx == 0 && coreTaskId < coreTaskNum) {
                CreateTaskInfo(curTaskInfo, coreTaskId, coreIdx, coreNum,
                               gActualQseqlen, gActualKvseqlen, gSparseCount, gSparseIdx);
            }

            // main process
            if (coreTileLoop >= PV_RE_DELAY_N && delay20CoreTaskId < coreTaskNum) {
                // delay20TileInfo = tileInfos[(coreTileLoop - PV_RE_DELAY_N) % PRELOAD_TASK_CACHE_SIZE];
                if ASCEND_IS_AIC {
                    if (delay20TileInfo.isTileGoupFirstTile) {
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_PSCALE_C2);
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_PSCALE_C2 + 16);
                    }
                    if (delay20TileInfo.isUpdatePScale) {
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V2_C2);
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V2_C2 + 16);
                    }
                    ComputeMm2(delay20TileInfo);
                    if (delay20TileInfo.isUpdatePScale) {
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_C2_V2);
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_C2_V2 + 16);
                    }
                } else {
                    if (delay20TileInfo.isUpdatePScale) {
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_C2_V2);
                        ComputeVec2(delay20TileInfo);
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V2_C2);
                    }
                }
            }
            if (coreTaskId < coreTaskNum) {
                // curTileInfo = tileInfos[coreTileLoop % PRELOAD_TASK_CACHE_SIZE];
                CreateTileInfo(curTileInfo, coreTileLoop,
                              static_cast<uint32_t>(curTaskTileIdx), curTaskInfo);// 创建任务

                uint32_t mm1ResBufId = (curTileInfo.loop / 2) % 2;
                uint32_t subBlockIdx = curTileInfo.loop % 2;
                if ASCEND_IS_AIC {
                    AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[mm1ResBufId] + subBlockIdx * 16);
                    ComputeMm1(curTileInfo);
                    AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[mm1ResBufId] + subBlockIdx * 16);
                } else {
                    if (subBlockIdx == AscendC::GetSubBlockIdx()) {
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V1_C1[mm1ResBufId]);
                        ComputeVec1(curTileInfo);
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V1_C1[mm1ResBufId]);
                    }
                    if (curTileInfo.isUpdatePScale) {
                        CopyGMaxUbToL1(curTileInfo);
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE3>(CROSS_CORE_SYNC_GMAX_UB_TO_L1);
                    }
                }
            }
            if (coreTileLoop >= DELAY_PSCALE_N && delay3CoreTaskId < coreTaskNum) {
                // delay3TileInfo = tileInfos[(coreTileLoop - DELAY_PSCALE_N) % PRELOAD_TASK_CACHE_SIZE];
                if (delay3TileInfo.isUpdatePScale) {
                    if ASCEND_IS_AIC {
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_GMAX_UB_TO_L1);
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_GMAX_UB_TO_L1 + 16);
                        CopyGMaxL1ToUb(delay3TileInfo);
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_GMAX_L1_TO_UB);
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE1>(CROSS_CORE_SYNC_GMAX_L1_TO_UB + 16);
                    } else {
                        AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_GMAX_L1_TO_UB);
                        UpdatePScale(delay3TileInfo);
                        AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_MTE3>(CROSS_CORE_SYNC_PSCALE_C2);
                    }
                }
            }

            // 是否要切换到下一个qs任务, 新的kvs
            if (curTaskTileIdx == curTaskInfo.taskTileNum - 1 && coreTaskId < coreTaskNum) {
                preTaskTileSum += curTaskInfo.taskTileNum;
                coreTaskId++;
            }
            if (coreTileLoop >= PV_RE_DELAY_N && delay20TaskTileIdx == delay20TaskInfo.taskTileNum - 1) {
                delay20preTaskTileSum += delay20TaskInfo.taskTileNum;
                delay20CoreTaskId++;
            }
            if (coreTileLoop >= DELAY_PSCALE_N && delay3TaskTileIdx == delay3TaskInfo.taskTileNum - 1  && delay3CoreTaskId < coreTaskNum) {
                delay3preTaskTileSum += delay3TaskInfo.taskTileNum;
                delay3CoreTaskId++;
            }
            coreTileLoop++;
            // AscendC::PRINTF("-----------------------\n");
        }

        // release reverse sync flags
        ReleaseSyncFlags();
    }

    __aicore__ inline void ComputeMm1(TileInfo &runInfo)
    {
        AscendC::PRINTF("C1 \n");
        // cubeBlock.ComputeMm1(runInfo);
    }

    __aicore__ inline void ComputeMm2(TileInfo &runInfo)
    {
        AscendC::PRINTF("C2 \n");
        // cubeBlock.ComputeMm2(runInfo);
    }

    __aicore__ inline void ComputeVec1(TileInfo &runInfo)
    {
        AscendC::PRINTF("V1 \n");
        // vectorBlock.ComputeVec1(runInfo);
    }

    __aicore__ inline void CopyGMaxUbToL1(TileInfo &runInfo)
    {
        AscendC::PRINTF("CopyGMaxUbToL1 \n");
        // vectorBlock.CopyGMaxUbToL1(runInfo);
    }

    __aicore__ inline void CopyGMaxL1ToUb(TileInfo &runInfo)
    {
        AscendC::PRINTF("CopyGMaxL1ToUb \n");
        // cubeBlock.CopyGMaxL1ToUb(runInfo);
    }

    __aicore__ inline void UpdatePScale(TileInfo &runInfo)
    {
        AscendC::PRINTF("UpdatePScale \n");
        // vectorBlock.UpdatePScale(runInfo);
    }

    __aicore__ inline void ComputeVec2(TileInfo &runInfo)
    {
        AscendC::PRINTF("V2  \n");
        // vectorBlock.ComputeVec2(runInfo);
    }

    __aicore__ inline
    void FetchBaseShapeInfo(__gm__ BlockSparseAttentionTilingData *bsaTilingData)
    {
        batch_ = bsaTilingData->batch;
        qHeads_ = bsaTilingData->numHeads;
        kvHeads_ = bsaTilingData->kvHeads;
        embed_ = bsaTilingData->embeddingSize;
        firstBatchTaskNum_ = bsaTilingData->firstBatchTaskNum;
        totalTaskNum_ = bsaTilingData->totalTaskNum;
        coreTaskNum_ = bsaTilingData->coreTaskNum;
        blockShapeX_ = bsaTilingData->blockShapeX;
        blockShapeY_ = bsaTilingData->blockShapeY;
        scaleValue_ = bsaTilingData->scaleValue;
        // mask2idx tile info
        xBlockNumAligned_ = bsaTilingData->BsaMask2IdxTileInfo.xBlockNumAligned;
        yBlockNumAligned_ = bsaTilingData->BsaMask2IdxTileInfo.yBlockNumAligned;
        avgRowPerSubCore_ = bsaTilingData->BsaMask2IdxTileInfo.avgRowPerSubCore;
        preActiveSubCoreNum_ = bsaTilingData->BsaMask2IdxTileInfo.preActiveSubCoreNum;
        // base tile info
        qBaseTile_ = bsaTilingData->BsaBaseTileInfo.qBaseTile;
        kvBaseTile_ = bsaTilingData->BsaBaseTileInfo.kvBaseTile;
        // whether actual seqlen is provided
        actSeqAval_ = (!bsaTilingData->useUniformQSeqlen) && (!bsaTilingData->useUniformKvSeqlen);
        sparseIdxSize_ = bsaTilingData->selectIdxSize;
        // aligned seqlen q & kv
        qSeqlenAligned_ = bsaTilingData->maxQSeqlen;
        kvSeqlenAligned_ = bsaTilingData->maxKvSeqlen;
    }

    __aicore__ inline
    void CalcOnChipBufTileInfo(__gm__ BlockSparseAttentionTilingData *bsaTilingData)
    {
        mm1L1TileM_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm1L1TileM;
        mm1L1TileN_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm1L1TileN;
        mm1L1TileKLeft_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm1L1TileKLeft;
        mm1L1TileKRight_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm1L1TileKRight;
        mm2L1TileM_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm2L1TileM;
        mm2L1TileN_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm2L1TileN;
        mm2L1TileKLeft_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm2L1TileKLeft;
        mm2L1TileKRight_ = bsaTilingData->BsaMmPhaseL1TileInfo.mm2L1TileKRight;
        qL1BufNum_ = bsaTilingData->BsaMmPhaseL1TileInfo.qL1BufNum;
        kL1BufNum_ = bsaTilingData->BsaMmPhaseL1TileInfo.kL1BufNum;
        vL1BufNum_ = bsaTilingData->BsaMmPhaseL1TileInfo.vL1BufNum;
        pL1BufNum_ = bsaTilingData->BsaMmPhaseL1TileInfo.pL1BufNum;
        Gemm::Block::Mm1L1TileHelper mm1L1TileHelper(mm1L1TileM_, mm1L1TileN_, mm1L1TileKLeft_, mm1L1TileKRight_,
            qL1BufNum_, kL1BufNum_);
        mm1L1TileHelper_ = mm1L1TileHelper;
        Gemm::Block::Mm2L1TileHelper mm2L1TileHelper(mm2L1TileM_, mm2L1TileN_, mm2L1TileKLeft_, mm2L1TileKRight_,
            pL1BufNum_, vL1BufNum_);
        mm2L1TileHelper_ = mm2L1TileHelper;
        mm2L1AddrStart_ = mm1L1TileM_ * mm1L1TileKLeft_ * qL1BufNum_ * sizeof(ElementQ) +
                          mm1L1TileKRight_ * mm1L1TileN_ * kL1BufNum_ * sizeof(ElementK);
        mm1L0ATotalStages_ = (qBaseTile_ / BlockMmadQK::L0_TILE_M) * (embed_ / BlockMmadQK::L0_TILE_K);
        mm1L0BTotalStages_ = (kvBaseTile_ / BlockMmadQK::L0_TILE_N) * (embed_ / BlockMmadQK::L0_TILE_K);
        mm2L0ATotalStages_ = (qBaseTile_ / BlockMmadPV::L0_TILE_M) * (kvBaseTile_ / BlockMmadPV::L0_TILE_K);
        mm2L0BTotalStages_ = (kvBaseTile_ / BlockMmadPV::L0_TILE_K) * (embed_ / BlockMmadPV::L0_TILE_N);
    }

    __aicore__ inline
    uint64_t CalcCrossMm1Mm2PrefixSumL0ABStages(uint32_t gatheredKvSTileIdx, uint32_t singleMm1L0Stages,
                                                uint32_t singleMm2L0Stages, uint32_t kvSLoopNum, bool isCurPhaseMm1)
    {
        // uint64_t prefixSumStages;
        // if (isCurPhaseMm1) {
        //     prefixSumStages =
        //         (gatheredKvSTileIdx <= PRE_LAUNCH) ?
        //             gatheredKvSTileIdx * singleMm1L0Stages :
        //             gatheredKvSTileIdx * singleMm1L0Stages + (gatheredKvSTileIdx - PRE_LAUNCH) * singleMm2L0Stages;
        // } else {
        //     prefixSumStages =
        //         (gatheredKvSTileIdx < kvSLoopNum - PRE_LAUNCH) ?
        //             (gatheredKvSTileIdx + 1 + PRE_LAUNCH) * singleMm1L0Stages + gatheredKvSTileIdx * singleMm2L0Stages :
        //             kvSLoopNum * singleMm1L0Stages + gatheredKvSTileIdx * singleMm2L0Stages;
        // }
        // return prefixSumStages;
    }

    __aicore__ inline
    void InitCrossCoreDstBuf(
        AscendC::LocalTensor<ElementP> (&l1PTensor)[MAX_CROSS_CORE_BUF_STAGES],
        AscendC::LocalTensor<ElementS> (&ubSTensor)[UB_S_OTMP_BUF_STAGES],
        AscendC::LocalTensor<ElementOTmp> (&ubOTmpTensor)[UB_S_OTMP_BUF_STAGES])
    {
        // for (uint32_t i = 0; i < pL1BufNum_; i++) {
        //     l1PTensor[i] = resource.l1Buf.template GetBufferByByte<ElementP>(
        //         mm2L1AddrStart_ + mm2L1TileM_ * mm2L1TileKLeft_ * sizeof(ElementP) * i);
        // }
        // uint32_t rowNumPerSubCore = EpilogueOnlineSoftmax::SM_ROW_MAX_ELEM_NUM;
        // uint32_t colNumPerSubCore = EpilogueOnlineSoftmax::SM_COL_MAX_ELEM_NUM;
        // uint32_t rescaleCol = EpilogueRescaleO::RESCALE_COL_MAX_ELEM_NUM;
        // for (uint32_t i = 0; i < UB_S_OTMP_BUF_STAGES; i++) {
        //     ubSTensor[i] = resource.ubBuf.template GetBufferByByte<ElementS>(
        //         rowNumPerSubCore * colNumPerSubCore * sizeof(ElementS) * i);
        //     ubOTmpTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(
        //         rowNumPerSubCore * colNumPerSubCore * sizeof(ElementS) * UB_S_OTMP_BUF_STAGES +
        //         rowNumPerSubCore * colNumPerSubCore * sizeof(ElementS) * UB_S_OTMP_BUF_STAGES +
        //         rowNumPerSubCore * rescaleCol * sizeof(ElementOTmp) * i);
        // }
    }


    // 由 coreTaskId (本核上的 task 序号) 反推 globalTaskId 并填充 TaskInfo
    // 仅在 task 起始 tile (taskTileId == 0) 调用，避免每个 tile 重复计算
    // coreIdx / coreNum 由调用方 Process 传入, 避免重复调用 GetBlockIdx/GetBlockNum
    __aicore__ inline
    void CreateTaskInfo(
        TaskInfo& info,
        uint32_t coreTaskId,
        uint32_t coreIdx,
        uint32_t coreNum,
        AscendC::GlobalTensor<int64_t>& gActualQseqlen,
        AscendC::GlobalTensor<int64_t>& gActualKvseqlen,
        AscendC::GlobalTensor<ElementSparseCount>& gSparseCount,
        AscendC::GlobalTensor<ElementSparseIdx>& gSparseIdx)
    {
        uint32_t taskIdx = coreTaskId * coreNum + coreIdx;  // 反推全局 task 序号

        uint32_t groupSize = qHeads_ / kvHeads_;
        uint32_t qSTileNumPerFullXBlock = CeilDiv(blockShapeX_, qBaseTile_);

        // 反推 taskIdx 所在的 batch
        uint32_t curBatch = 0;
        uint32_t preTotalTaskNum = 0;
        int64_t qSeqlen = actSeqAval_ ? gActualQseqlen.GetValue(curBatch) : qSeqlenAligned_;
        int64_t kvSeqlen = actSeqAval_ ? gActualKvseqlen.GetValue(curBatch) : kvSeqlenAligned_;
        uint32_t curQSTileNum = GetCurQSTileNum(qSeqlen, blockShapeX_, qBaseTile_);
        uint32_t curTotalTaskNum = firstBatchTaskNum_;
        while (taskIdx >= curTotalTaskNum) {
            ++curBatch;
            preTotalTaskNum = curTotalTaskNum;
            qSeqlen = actSeqAval_ ? gActualQseqlen.GetValue(curBatch) : qSeqlenAligned_;
            kvSeqlen = actSeqAval_ ? gActualKvseqlen.GetValue(curBatch) : kvSeqlenAligned_;
            curQSTileNum = GetCurQSTileNum(qSeqlen, blockShapeX_, qBaseTile_);
            curTotalTaskNum += curQSTileNum * qHeads_;
        }

        // 拆解 taskIdx 得到 head / qSTile / xBlock 索引
        uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNum;
        uint32_t qHeadIdx = taskIdxCurBatch / curQSTileNum;
        uint32_t kvHeadIdx = qHeadIdx / groupSize;
        uint32_t qSTileIdx = taskIdxCurBatch - qHeadIdx * curQSTileNum;
        uint32_t xBlockIdx = qSTileIdx / qSTileNumPerFullXBlock;
        uint32_t qSTileIdxCurXBlock = qSTileIdx - xBlockIdx * qSTileNumPerFullXBlock;

        // 实际 Q tile 大小 (最后一个 tile 可能不足 qBaseTile_)
        uint32_t xBlockNumAval = static_cast<uint32_t>(CeilDiv(qSeqlen, static_cast<int64_t>(blockShapeX_)));
        uint32_t xBlockSize = (xBlockIdx == xBlockNumAval - 1) ?
            static_cast<uint32_t>(qSeqlen - xBlockIdx * blockShapeX_) : blockShapeX_;
        uint32_t qSTileNumCurXBlock = CeilDiv(xBlockSize, qBaseTile_);
        uint32_t qsActBaseTile = (qSTileIdxCurXBlock == qSTileNumCurXBlock - 1) ?
            (xBlockSize - qSTileIdxCurXBlock * qBaseTile_) : qBaseTile_;

        // 从 sparse mask 计算 gatheredKvSeqlen
        uint32_t gmOffsetSparseCount =
            curBatch * qHeads_ * xBlockNumAligned_ + qHeadIdx * xBlockNumAligned_ + xBlockIdx;
        uint32_t yBlockNumRsvd = gSparseCount.GetValue(gmOffsetSparseCount);
        uint32_t gmOffsetSparseIdx = gmOffsetSparseCount * yBlockNumAligned_;
        uint32_t lastIdxOffset = gmOffsetSparseIdx + yBlockNumRsvd - 1;
        uint32_t lastSparseIdx = gSparseIdx.GetValue(lastIdxOffset);
        uint32_t yBlockNumAval = static_cast<uint32_t>(CeilDiv(kvSeqlen, static_cast<int64_t>(blockShapeY_)));
        uint32_t lastYBlockSize = (lastSparseIdx == yBlockNumAval - 1) ?
            static_cast<uint32_t>(kvSeqlen - lastSparseIdx * blockShapeY_) : blockShapeY_;
        int64_t gatheredKvSeqlen = (yBlockNumRsvd - 1) * blockShapeY_ + lastYBlockSize;

        // taskTileNum = ceil(gatheredKvSeqlen / kvBaseTile_)
        uint32_t taskTileNum = static_cast<uint32_t>(
            CeilDiv(gatheredKvSeqlen, static_cast<int64_t>(kvBaseTile_)));

        info.batchIdx = curBatch;
        info.qHeadIdx = qHeadIdx;
        info.kvHeadIdx = kvHeadIdx;
        info.qSTileIdx = qSTileIdx;
        info.xBlockIdx = xBlockIdx;
        info.gatheredKvSeqlen = gatheredKvSeqlen;
        info.qsActBaseTile = qsActBaseTile;
        info.qsBaseTile = qBaseTile_;
        info.kvsBaseTile = kvBaseTile_;
        info.taskTileNum = taskTileNum;
    }

    // 每个 tile 调用一次，根据 coreTileLoop / curTaskTileIdx / taskInfo 填充 TileInfo
    __aicore__ inline
    void CreateTileInfo(
        TileInfo& info,
        uint64_t coreTileLoop,
        uint32_t curTaskTileIdx,
        const TaskInfo& taskInfo)
    {

        info.loop = static_cast<uint32_t>(coreTileLoop);
        info.curKvsTileLoopIdx = curTaskTileIdx;
        info.isFirstKvsTile = (curTaskTileIdx == 0);
        info.isLastKvsTile = (curTaskTileIdx + 1 == taskInfo.taskTileNum);
        info.isLastSecondKvsTile = (curTaskTileIdx + 2 == taskInfo.taskTileNum);

        info.isUpdatePScale = (info.isLastKvsTile ||
                               ((info.curKvsTileLoopIdx + 1) % TILE_GROUP_N == 0));
        info.isTileGoupFirstTile = (info.curKvsTileLoopIdx % TILE_GROUP_N == 0);
        if (info.isFirstKvsTile) {
            kvsFirstTileStartVecCore_ = static_cast<uint32_t>(coreTileLoop % 2);
        }
        info.kvsFirstTileStartVecCore = kvsFirstTileStartVecCore_;

        if (info.isTileGoupFirstTile) {
            updateScaleIdx_ = (updateScaleIdx_ + 1) % 4;
        }

        // task 第一个 tile 或 TileGroup 第一个 tile 时切换 buff (双缓冲 / 四缓冲)
        if (info.isFirstKvsTile || info.isTileGoupFirstTile) {
            tileLoopIdx_ = (tileLoopIdx_ + 1) % 2;
            tileMaxIdx_ = (tileMaxIdx_ + 1) % 4;
        }
        info.tileBuffIdx = tileLoopIdx_;
        info.tileMaxIdx = tileMaxIdx_;
        info.updateScaleIdx = updateScaleIdx_;
        info.isKvsFirstTilePerCore = (info.curKvsTileLoopIdx % TILE_GROUP_N / 2 == 0);

        info.kvsActBaseTile = info.isLastKvsTile ?
                (taskInfo.gatheredKvSeqlen - (taskInfo.taskTileNum - 1) * kvBaseTile_) : kvBaseTile_;
    }

    
    __aicore__ inline
    void InitSyncFlags()
    {
        if ASCEND_IS_AIC {

        } else {
            // 核间
            AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V1_C1[0]);
            AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V1_C1[1]);
            AscendC::CrossCoreSetFlag<SYNC_MODE_4, PIPE_V>(CROSS_CORE_SYNC_V2_C2);
            // mask2index
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);

            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
        }
    }

    __aicore__ inline
    void ReleaseSyncFlags()
    {
        if ASCEND_IS_AIC {
            // 核间
            AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V2_C2);
            AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V2_C2 + 16);
            AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[0]);
            AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[0] + 16);
            AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[1]);
            AscendC::CrossCoreWaitFlag<SYNC_MODE_4, PIPE_FIX>(CROSS_CORE_SYNC_V1_C1[1] + 16);
        } else {
            // mask2index
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);

        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    Arch::Resource<ArchTag> resource;
    /*
    tiling info, which are const in each kernel launch
    */
    // basic shape info
    uint32_t batch_;
    uint32_t qHeads_;
    uint32_t kvHeads_;
    uint32_t embed_;
    uint32_t firstBatchTaskNum_;
    uint32_t totalTaskNum_;
    uint32_t coreTaskNum_;  // 向下取整: totalTaskNum_ / blockDim
    uint32_t blockShapeX_;
    uint32_t blockShapeY_;
    float scaleValue_;
    // mask2idx tile info
    uint32_t xBlockNumAligned_;
    uint32_t yBlockNumAligned_;
    uint32_t avgRowPerSubCore_;
    uint32_t preActiveSubCoreNum_;
    // base tile info
    uint32_t qBaseTile_;
    uint32_t kvBaseTile_;
    // whether actual seqlen is provided
    uint32_t actSeqAval_;
    // workspace size
    uint64_t sparseIdxSize_;
    // aligned seqlen q & kv
    int64_t qSeqlenAligned_;
    int64_t kvSeqlenAligned_;
    // L1 tile info
    uint32_t mm1L1TileM_;
    uint32_t mm1L1TileN_;
    uint32_t mm1L1TileKLeft_;
    uint32_t mm1L1TileKRight_;
    uint32_t mm2L1TileM_;
    uint32_t mm2L1TileN_;
    uint32_t mm2L1TileKLeft_;
    uint32_t mm2L1TileKRight_;
    uint32_t qL1BufNum_;
    uint32_t kL1BufNum_;
    uint32_t vL1BufNum_;
    uint32_t pL1BufNum_;
    uint32_t mm1L0ATotalStages_;
    uint32_t mm1L0BTotalStages_;
    uint32_t mm2L0ATotalStages_;
    uint32_t mm2L0BTotalStages_;
    uint32_t mm2L1AddrStart_ = 0;
    Gemm::Block::Mm1L1TileHelper mm1L1TileHelper_;
    Gemm::Block::Mm2L1TileHelper mm2L1TileHelper_;

    // ---- CreateTileInfo 持久状态 (跨 tile 保持, 参考 qfa.cpp CalcParams) ----
    uint32_t kvsFirstTileStartVecCore_ = 0; // task 首 tile 时切换 (loop % 2)
    uint32_t updateScaleIdx_ = 3;       // C2 Sync 时切换 (循环 0~3)
    uint32_t tileLoopIdx_ = 1;          // 双缓冲索引 (0/1)
    uint32_t tileMaxIdx_ = 3;           // 四缓冲索引 (0~3)
};

}