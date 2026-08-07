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
    Format qFormat,
    Format kvFormat,
    bool transposedMm1>
class BsaFullQuantKernelArch35 {
public:
    using ArchTag = typename BlockMmadPV::ArchTag;
    
    using ElementQ = typename BlockMmadQK::ElementQ; // fp8_e4m3fn_t
    using ElementK = typename BlockMmadQK::ElementK; // fp8_e4m3fn_t
    using ElementS = typename EpilogueOnlineSoftmax::ElementInput; // half/bloat16_t/float
    using ElementP = typename BlockMmadPV::ElementA; // fp8_e4m3fn_t
    using ElementV = typename BlockMmadPV::ElementB; // fp8_e4m3fn_t
    using ElementOTmp = typename BlockMmadPV::ElementC; // float
    using ElementO = typename EpilogueRescaleO::ElementO; // half/bloat16_t
    using ElementLse = typename EpilogueRescaleO::ElementLse;
    using ElementSparseMask = typename EpilogueMask2Idx::ElementSparseMask;
    using ElementSparseIdx = typename EpilogueMask2Idx::ElementSparseIdx;
    using ElementSparseCount = typename EpilogueMask2Idx::ElementSparseCount;

    // rescaleOnCube on cube 路径开关：仅 DN + FP8（新 PV 特化 <true,true>）生效；
    // 其余（含 FP8 ND 12 键）回落旧路径（VEC rescaleO 逐 tile + lo/dm UB 缓冲）
    static constexpr bool RESCALE_ON_CUBE = transposedMm1 && BlockMmadPV::rescaleOnCube;

    using LayoutQ = std::conditional_t<transposedMm1, layout::ColumnMajor, layout::RowMajor>;
    using LayoutK = std::conditional_t<transposedMm1, layout::RowMajor, layout::ColumnMajor>;
    using LayoutS = layout::RowMajor;
    using LayoutP = layout::RowMajor;
    // V：rescaleOnCube 时以 ColumnMajor [D, kvS] 进 L1（A=(V)^T 直接读，与 Q 的 DN 加载同构）
    using LayoutV = std::conditional_t<RESCALE_ON_CUBE, layout::ColumnMajor, layout::RowMajor>;
    using LayoutO = layout::RowMajor;
    using LayoutLse = layout::RowMajor;
    using LayoutOTmp = layout::RowMajor;
    using LayoutSparseIdx = layout::RowMajor;
    using LayoutSparseCount = layout::RowMajor;

    using LayoutTagL1P = typename BlockMmadPV::LayoutTagL1A;

    static constexpr uint32_t PRE_LAUNCH = 2;
    static constexpr uint32_t MAX_CROSS_CORE_BUF_STAGES = PRE_LAUNCH + 1;
    static constexpr uint32_t UB_S_OTMP_BUF_STAGES = 2;

    // Methods
    __aicore__ inline
    BsaFullQuantKernelArch35() {}

    __aicore__ inline
    void operator()(BsaFullQuantKernelParamsArch35 const &params)
    {
        __gm__ BlockSparseAttentionTilingData *bsaTilingData =
            reinterpret_cast<__gm__ BlockSparseAttentionTilingData *>(params.tiling);
        FetchBaseShapeInfo(bsaTilingData);
        CalcOnChipBufTileInfo(bsaTilingData);
        CalcUBufTileInfo();
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
        AscendC::LocalTensor<ElementOTmp> ubGoTensor;      // rescaleOnCube on cube：仅末块（NZ2DN 搬入）
        AscendC::LocalTensor<ElementOTmp> ubOTmpTensor[UB_S_OTMP_BUF_STAGES];  // 旧路径：lo 逐 tile 2 份轮转
        AscendC::LocalTensor<float> l1DiagTensor[3];
        InitCrossCoreDstBuf(l1PTensor, ubSTensor, ubGoTensor, ubOTmpTensor, l1DiagTensor);
        // core idx
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t coreNum = AscendC::GetBlockNum();
        // set reverse sync flags
        InitSyncFlags<4, 4, 4>();
#ifdef __DAV_VEC__
        EpilogueMask2Idx epilogueMask2Idx(resource);
        uint32_t totalRowNumBlockMask = batch_ * qHeads_ * xBlockNumAligned_;
        epilogueMask2Idx(
            gBlockSparseMask, gSparseIdx, gSparseCount,
            totalRowNumBlockMask, yBlockNumAligned_, avgRowPerSubCore_, preActiveSubCoreNum_);
#endif
        AscendC::SyncAll<false>();
#ifdef __DAV_CUBE__
        coreIdx = AscendC::GetBlockIdx();
        BlockMmadQK blockMmadQK(resource, mm1L1TileHelper_);
        // rescaleOnCube on cube：4 参（多 diag 块 L1 起始地址）；旧路径 3 参（l1DiagAddrStart_ 默认 0 被忽略）
        BlockMmadPV blockMmadPV(resource, mm2L1AddrStart_, l1DiagAddrStart_, mm2L1TileHelper_);
#endif
#ifdef __DAV_VEC__
        coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, scaleValue_, uBufTileHelper_);
        EpilogueRescaleO epilogueRescaleO(resource, uBufTileHelper_);
#endif
        uint32_t qSTileNumPerFullXBlock = CeilDiv(blockShapeX_, qBaseTile_);
        // Calculate strides based on layout
        // For TND: [T, N, D], stride = N * D
        // For BNSD: [B, N, S, D], strideB = N * S * D, strideN = S * D, strideS = D
        // For BSND: [B, S, N, D], strideB = S * B * D, strideS = N * D, strideN = D
        int64_t strideQO = 0;
        int64_t strideKV = 0;
        int64_t strideLse = 0;
        int64_t strideQOB = 0;  // BNSD/BSND batch_ stride for Q
        int64_t strideQON = 0;  // BNSD/BSND head stride for Q
        int64_t strideQOS = 0;  // BNSD/BSND seq stride for Q
        int64_t strideKVB = 0;  // BNSD/BSND batch_ stride for KV
        int64_t strideKVN = 0;  // BNSD/BSND head stride for KV
        int64_t strideKVS = 0;  // BNSD/BSND seq stride for KV
        int64_t strideLseB = 0;
        int64_t strideLseN = 0;
        int64_t strideLseS = 0;

        if constexpr (qFormat == Format::TND) {
            strideQO = qHeads_ * embed_;
            strideLse = qHeads_;
        } else if constexpr (qFormat == Format::BNSD) {
            strideQOB = qHeads_ * qSeqlenAligned_ * embed_;  // batch_ stride
            strideQON = qSeqlenAligned_ * embed_;  // head stride
            strideQOS = embed_;  // seq stride
            strideLseB = qHeads_ * qSeqlenAligned_;
            strideLseN = qSeqlenAligned_;
            strideLseS = 1;
        } else if constexpr (qFormat == Format::BSND) {
            strideQOB = qSeqlenAligned_ * qHeads_ * embed_; // batch_ stride
            strideQOS = qHeads_ * embed_;                   // seq stride
            strideQON = embed_;                             // head stride
            strideLseB = qSeqlenAligned_ * qHeads_;
            strideLseS = qHeads_;
            strideLseN = 1;
        }

        if constexpr (kvFormat == Format::TND) {
            strideKV = kvHeads_ * embed_;
        } else if constexpr (kvFormat == Format::BNSD) {
            strideKVB = kvHeads_ * kvSeqlenAligned_ * embed_; // batch_ stride
            strideKVN = kvSeqlenAligned_ * embed_;            // head stride
            strideKVS = embed_;                               // seq stride
        } else if constexpr (kvFormat == Format::BSND) {
            strideKVB = kvSeqlenAligned_ * kvHeads_ * embed_; // batch_ stride
            strideKVS = kvHeads_ * embed_;                    // seq stride
            strideKVN = embed_;                               // head stride
        }

        uint32_t embedRound = RoundUp(embed_, 16);
        uint32_t groupSize = qHeads_ / kvHeads_;
        int64_t qBOffset = 0;
        int64_t kBOffset = 0;
        int64_t vBOffset = 0;
        int64_t oBOffset = 0;
        int64_t lseBOffset = 0;
        uint32_t preTotalTaskNum = 0;
        uint32_t curBatch = 0;
        int64_t qSeqlen = actSeqAval_ ? gActualQseqlen.GetValue(curBatch) : qSeqlenAligned_;
        int64_t kvSeqlen = actSeqAval_ ? gActualKvseqlen.GetValue(curBatch) : kvSeqlenAligned_;
        uint32_t curQSTileNum = GetCurQSTileNum(qSeqlen, blockShapeX_, qBaseTile_);
        uint32_t curTotalTaskNum = firstBatchTaskNum_;
        // Go through each task
        for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum_; taskIdx += coreNum) {
            while (taskIdx >= curTotalTaskNum) {
                ++curBatch;
                preTotalTaskNum = curTotalTaskNum;
                if constexpr (qFormat == Format::TND) {
                    qBOffset += qSeqlen * strideQO;
                    oBOffset += qSeqlen * strideQO;
                    lseBOffset += qSeqlen * strideLse;
                }
                if constexpr (kvFormat == Format::TND) {
                    kBOffset += kvSeqlen * strideKV;
                    vBOffset += kvSeqlen * strideKV;
                }
                qSeqlen = actSeqAval_ ? gActualQseqlen.GetValue(curBatch) : qSeqlenAligned_;
                kvSeqlen = actSeqAval_ ? gActualKvseqlen.GetValue(curBatch) : kvSeqlenAligned_;
                curQSTileNum = GetCurQSTileNum(qSeqlen, blockShapeX_, qBaseTile_);
                curTotalTaskNum += curQSTileNum * qHeads_;
            }
            uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNum;
            uint32_t qHeadIdx = taskIdxCurBatch / curQSTileNum;
            uint32_t kvHeadIdx = qHeadIdx / groupSize;
            uint32_t qSTileIdx = taskIdxCurBatch - qHeadIdx * curQSTileNum;
            // corresponding xBlock index of cur task
            uint32_t xBlockIdx = qSTileIdx / qSTileNumPerFullXBlock;
            // the q base tile index within the corrsponding xBlock of cur task
            uint32_t qSTileIdxCurXBlock = qSTileIdx - xBlockIdx * qSTileNumPerFullXBlock;
            // corresponding head index of cur task
            // corresponding blockSparseMask gm offset of cur task
            // gmBlockSparseMask has the shape [B, qN, xBlockNumAligned, yBlockNumAligned]
            int64_t sparseMaskBOffset = curBatch * qHeads_ * xBlockNumAligned_ * yBlockNumAligned_;
            int64_t sparseMaskNOffset = qHeadIdx * xBlockNumAligned_ * yBlockNumAligned_;
            int64_t sparseMaskXOffset = xBlockIdx * yBlockNumAligned_;
            int64_t gmOffsetSparseMask = sparseMaskBOffset + sparseMaskNOffset + sparseMaskXOffset;
            // corresponding Q/K/V/O/Lse gm offset of cur task
            int64_t gmOffsetQ = 0;
            int64_t gmOffsetK = 0;
            int64_t gmOffsetV = 0;
            int64_t gmOffsetO = 0;
            int64_t gmOffsetLse = 0;
            int64_t qSOffset = xBlockIdx * blockShapeX_ + qSTileIdxCurXBlock * qBaseTile_;

            if constexpr (qFormat == Format::TND) {
                gmOffsetQ = qBOffset + qSOffset * strideQO + qHeadIdx * embed_;
                gmOffsetO = oBOffset + qSOffset * strideQO + qHeadIdx * embed_;
                gmOffsetLse = lseBOffset + qSOffset * strideLse + qHeadIdx;
            } else if constexpr (qFormat == Format::BNSD) {
                qBOffset = curBatch * strideQOB;
                oBOffset = curBatch * strideQOB;
                lseBOffset = curBatch * strideLseB;
                gmOffsetQ = qBOffset + qHeadIdx * strideQON + qSOffset * strideQOS;
                gmOffsetO = oBOffset + qHeadIdx * strideQON + qSOffset * strideQOS;
                gmOffsetLse = lseBOffset + qHeadIdx * strideLseN + qSOffset * strideLseS;
            } else if constexpr (qFormat == Format::BSND) {
                qBOffset = curBatch * strideQOB;
                oBOffset = curBatch * strideQOB;
                lseBOffset = curBatch * strideLseB;
                gmOffsetQ = qBOffset + qSOffset * strideQOS + qHeadIdx * strideQON;
                gmOffsetO = oBOffset + qSOffset * strideQOS + qHeadIdx * strideQON;
                gmOffsetLse = lseBOffset + qSOffset * strideLseS + qHeadIdx * strideLseN;
            }

            if constexpr (kvFormat == Format::TND) {
                gmOffsetK = kBOffset + kvHeadIdx * embed_;
                gmOffsetV = vBOffset + kvHeadIdx * embed_;
            } else if constexpr (kvFormat == Format::BNSD) {
                kBOffset = curBatch * strideKVB;
                vBOffset = curBatch * strideKVB;
                gmOffsetK = kBOffset + kvHeadIdx * strideKVN;
                gmOffsetV = vBOffset + kvHeadIdx * strideKVN;
            } else if constexpr (kvFormat == Format::BSND) {
                kBOffset = curBatch * strideKVB;
                vBOffset = curBatch * strideKVB;
                gmOffsetK = kBOffset + kvHeadIdx * strideKVN;
                gmOffsetV = vBOffset + kvHeadIdx * strideKVN;
            }

            // the actual x block num of cur batch_, calc by actual qseqlen
            uint32_t xBlockNumAval = static_cast<uint32_t>(CeilDiv(qSeqlen, static_cast<int64_t>(blockShapeX_)));
            uint32_t xBlockSize = (xBlockIdx == xBlockNumAval - 1) ?
                (qSeqlen - xBlockIdx * blockShapeX_) : blockShapeX_;
            uint32_t qSTileNumCurXBlock = CeilDiv(xBlockSize, qBaseTile_);
            uint32_t qSTileSizeAct = (qSTileIdxCurXBlock == qSTileNumCurXBlock - 1) ?
                (xBlockSize - qSTileIdxCurXBlock * qBaseTile_) : qBaseTile_;
            // calc the gathered kvS from sparse mask
            uint32_t gmOffsetSparseCount =
                curBatch * qHeads_ * xBlockNumAligned_ + qHeadIdx * xBlockNumAligned_ + xBlockIdx;
            uint32_t yBlockNumRsvd = gSparseCount.GetValue(gmOffsetSparseCount);

            uint32_t gmOffsetSparseIdx = gmOffsetSparseCount * yBlockNumAligned_;
            uint32_t lastIdxOffset = gmOffsetSparseIdx + yBlockNumRsvd - 1;
            uint32_t lastSparseIdx = gSparseIdx.GetValue(lastIdxOffset);

            uint32_t yBlockNumAval = static_cast<uint32_t>(CeilDiv(kvSeqlen, static_cast<int64_t>(blockShapeY_)));
            uint32_t lastYBlockSize = (lastSparseIdx == yBlockNumAval - 1) ?
                kvSeqlen - lastSparseIdx * blockShapeY_ : blockShapeY_;
            int64_t gatheredKvSeqlen = (yBlockNumRsvd - 1) * blockShapeY_ + lastYBlockSize;
            // the rowNum of cur task
            // no qS*qN combination even in GQA/MQA senario, since each qN has a different sparse pattern
            uint32_t rowNum = qSTileSizeAct;
            uint32_t rowNumRound = RoundUp(rowNum, 16);
            uint32_t kvSLoopNum = static_cast<uint32_t>(CeilDiv(gatheredKvSeqlen, static_cast<int64_t>(kvBaseTile_)));
            uint32_t kvSTileSizeAct = kvBaseTile_;
#ifdef __DAV_CUBE__
            uint32_t qShapeCol = 0;
            uint32_t qShapeRow = 0;
            uint32_t kShapeCol = 0;
            uint32_t kShapeRow = 0;
            uint32_t vShapeCol = 0;
            uint32_t vShapeRow = 0;

            if constexpr (qFormat == Format::TND) {
                qShapeCol = strideQO;
            } else if constexpr (qFormat == Format::BNSD) {
                qShapeCol = strideQOS;
            } else if constexpr (qFormat == Format::BSND) {
                qShapeCol = strideQOS;
            }
            if constexpr (kvFormat == Format::TND) {
                kShapeRow = strideKV;
                vShapeCol = strideKV;
            } else if constexpr (kvFormat == Format::BNSD) {
                kShapeRow = strideKVS;
                vShapeCol = strideKVS;
            } else if constexpr (qFormat == Format::BSND) {
                kShapeRow = strideKVS;
                vShapeCol = strideKVS;
            }
            if constexpr (RESCALE_ON_CUBE) {
                // V 以 ColumnMajor [D, kvS] 进 L1（A=(V)^T 直接读，与 Q 的 DN 加载同构）
                vShapeRow = strideKV;
                vShapeCol = kvBaseTile_;
            } else {
                vShapeRow = kvBaseTile_;
            }
            // if doing K * Q^t, switch Q/K row&col
            if constexpr (transposedMm1) {
                qShapeRow = qShapeCol;
                qShapeCol = qBaseTile_;
                kShapeCol = kShapeRow;
                kShapeRow = kvBaseTile_;
            } else {
                qShapeRow = qBaseTile_;
                kShapeCol = kvBaseTile_;
            }

            auto gmQLayoutTla = tla::MakeLayout<ElementQ, LayoutQ>(qShapeRow, qShapeCol);
            auto gmQTensorTla = tla::MakeTensor(gQ[gmOffsetQ], gmQLayoutTla, Arch::PositionGM{});
            GemmCoord actualBlockShapeQ{rowNum, embed_, 0};
            blockMmadQK.loadQGM(gmQTensorTla, actualBlockShapeQ);

            auto gmKLayoutTla = tla::MakeLayout<ElementK, LayoutK>(kShapeRow, kShapeCol);
            auto gmKTensorTla = tla::MakeTensor(gK[gmOffsetK], gmKLayoutTla, Arch::PositionGM{});

            auto gmVLayoutTla = tla::MakeLayout<ElementV, LayoutV>(vShapeRow, vShapeCol);
            auto gmVTensorTla = tla::MakeTensor(gV[gmOffsetV], gmVLayoutTla, Arch::PositionGM{});

            uint32_t qDequantScaleOffset =
                curBatch * qHeads_ * xBlockNumAligned_ + qHeadIdx * xBlockNumAligned_ + xBlockIdx;
            float qDequantScaleValue = gQDequantScale.GetValue(qDequantScaleOffset);
#endif
#ifdef __DAV_VEC__
            uint32_t oShapeCol = 0;
            uint32_t lseShapeCol = 0;
            if constexpr (qFormat == Format::TND) {
                oShapeCol = strideQO;
                lseShapeCol = strideLse;
            } else if constexpr (qFormat == Format::BNSD) {
                oShapeCol = strideQOS;
                lseShapeCol = strideLseS;
            } else if constexpr (qFormat == Format::BSND) {
                oShapeCol = strideQOS;
                lseShapeCol = strideLseS;
            }

            auto gmOLayoutTla = tla::MakeLayout<ElementO, LayoutO>(qBaseTile_, oShapeCol);
            auto gmOTensorTla = tla::MakeTensor(gO[gmOffsetO], gmOLayoutTla, Arch::PositionGM{});
            auto gmLseLayoutTla = tla::MakeLayout<ElementLse, LayoutLse>(qBaseTile_, lseShapeCol);
            auto gmLseTensorTla = tla::MakeTensor(gLse[gmOffsetLse], gmLseLayoutTla, Arch::PositionGM{});
#endif
            for (uint32_t gatheredKvSTileIdx = 0; gatheredKvSTileIdx < kvSLoopNum + PRE_LAUNCH; gatheredKvSTileIdx++) {
                if (gatheredKvSTileIdx < kvSLoopNum) {
                    if (gatheredKvSTileIdx == kvSLoopNum - 1) {
                        kvSTileSizeAct = gatheredKvSeqlen - gatheredKvSTileIdx * kvBaseTile_;
                    } else {
                        kvSTileSizeAct = kvBaseTile_;
                    }
                    // QK
                    GemmCoord actualBlockShapeQK{rowNum, kvSTileSizeAct, embed_};
                    uint32_t ubSBufId = gatheredKvSTileIdx % UB_S_OTMP_BUF_STAGES;
                    uint32_t sShapeRow = 0;
                    uint32_t sShapeCol = 0;
                    if constexpr (transposedMm1) {
                        // S has shape [kvSTileSizeAct, RoundUp(rowNum, 32) / 2], rowmajor in UB, has stride [64, 1].
                        // It is a spectial design for transposed mm1
                        sShapeRow = RoundUp(kvSTileSizeAct, 16);
                        sShapeCol = 64;
                    } else {
                        sShapeRow = rowNumRound;
                        sShapeCol = RoundUp(kvSTileSizeAct, 16);
                    }
                    auto ubSLayoutTla = tla::MakeLayout<ElementS, LayoutS>(sShapeRow, sShapeCol);
                    auto ubSTensorTla = tla::MakeTensor(ubSTensor[ubSBufId],
                    ubSLayoutTla, Arch::PositionUB{});
                    uint32_t Mm1ToSmFlagId = ubSBufId;
                    Arch::CrossCoreFlag mm1ToSmFlag(Mm1ToSmFlagId);
#ifdef __DAV_CUBE__
                    uint64_t prefixSumL0AStages = CalcCrossMm1Mm2PrefixSumL0ABStages(
                        gatheredKvSTileIdx, mm1L0ATotalStages_, mm2L0ATotalStages_, kvSLoopNum, true);
                    uint64_t prefixSumL0BStages = CalcCrossMm1Mm2PrefixSumL0ABStages(
                        gatheredKvSTileIdx, mm1L0BTotalStages_, mm2L0BTotalStages_, kvSLoopNum, true);
                    uint32_t yBlockIndexOffset = (gatheredKvSTileIdx * kvBaseTile_) / blockShapeY_;
                    uint32_t yBlockIndex = gSparseIdx[gmOffsetSparseIdx].GetValue(yBlockIndexOffset);
                    uint32_t kvDequantScaleOffset =
                        curBatch * kvHeads_ * yBlockNumAligned_ + kvHeadIdx * yBlockNumAligned_ + yBlockIndex;
                    float kDequantScaleValue = gKDequantScale.GetValue(kvDequantScaleOffset);
                    float combinedDeqScalar = qDequantScaleValue * kDequantScaleValue;
                    uint64_t deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&combinedDeqScalar));
                    blockMmadQK(gmKTensorTla, ubSTensorTla, gSparseIdx[gmOffsetSparseIdx], actualBlockShapeQK,
                                gatheredKvSTileIdx, kvSeqlen, kvBaseTile_, blockShapeY_, yBlockNumAval, yBlockNumRsvd,
                                prefixSumL0AStages, prefixSumL0BStages, mm1ToSmFlag, deqScalar);
                    if (gatheredKvSTileIdx == kvSLoopNum - 1) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
                    }
#endif
                    // SM
                    uint32_t l1PBufId = gatheredKvSTileIdx % pL1BufNum_;
                    uint32_t smToMm2FlagId = l1PBufId + UB_S_OTMP_BUF_STAGES;
                    Arch::CrossCoreFlag smToMm2Flag(smToMm2FlagId);
                    auto l1PLayoutTla = tla::MakeLayout<ElementP, NpuArch::layout::zN>(rowNum, kvSTileSizeAct);
                    auto l1PTensorTla = tla::MakeTensor(l1PTensor[l1PBufId],
                    l1PLayoutTla, Arch::PositionL1{});
#ifdef __DAV_VEC__
                    // Stage 2: Online softmax (computed on VECTOR core)
                    // online softmax
                    if constexpr (RESCALE_ON_CUBE) {
                        // vScale（当前索引，无 De）：生成 diag(dm/vScale) 使用
                        uint32_t yBlockIndexOffsetSm = (gatheredKvSTileIdx * kvBaseTile_) / blockShapeY_;
                        uint32_t yBlockIndexSm = gSparseIdx[gmOffsetSparseIdx].GetValue(yBlockIndexOffsetSm);
                        uint32_t kvDequantScaleOffsetSm =
                            curBatch * kvHeads_ * yBlockNumAligned_ + kvHeadIdx * yBlockNumAligned_ + yBlockIndexSm;
                        float vDequantScaleValueSm = gVDequantScale.GetValue(kvDequantScaleOffsetSm);
                        epilogueOnlineSoftmax(
                            l1PTensorTla,
                            actualBlockShapeQK,
                            (gatheredKvSTileIdx == 0),
                            ubSBufId,
                            l1PBufId,
                            mm1ToSmFlag,
                            smToMm2Flag,
                            l1DiagTensor[l1PBufId],
                            vDequantScaleValueSm);
                    } else {
                        // 旧路径（FP8 ND）：ND softmax 7 参签名，无 diag/vScale
                        epilogueOnlineSoftmax(
                            l1PTensorTla,
                            actualBlockShapeQK,
                            (gatheredKvSTileIdx == 0),
                            ubSBufId,
                            l1PBufId,
                            mm1ToSmFlag,
                            smToMm2Flag);
                    }
#endif
                }
                if (gatheredKvSTileIdx >= PRE_LAUNCH) {
                    uint32_t gatheredKvSTileIdxDe = gatheredKvSTileIdx - PRE_LAUNCH;
                    if (gatheredKvSTileIdxDe == kvSLoopNum - 1) {
                        kvSTileSizeAct = gatheredKvSeqlen - gatheredKvSTileIdxDe * kvBaseTile_;
                    } else {
                        kvSTileSizeAct = kvBaseTile_;
                    }
                    // PV (transposed, rescale fused on cube)
                    GemmCoord actualBlockShapePV{rowNum, embed_, kvSTileSizeAct};
                    // 核间同步flagId规律：每份dst对应一个id，从小到大顺序依次为qk->sm, sm->pv, pv->re
                    uint32_t Mm2ToReFlagId = UB_S_OTMP_BUF_STAGES + pL1BufNum_;
                    uint32_t ubOTmpBufId = gatheredKvSTileIdxDe % UB_S_OTMP_BUF_STAGES;
                    if constexpr (!RESCALE_ON_CUBE) {
                        // 旧路径：lo 逐 tile 2 份轮转 -> flag id 随 ubOTmpBufId 交替（5/6）
                        Mm2ToReFlagId += ubOTmpBufId;
                    }
#ifdef __DAV_CUBE__
                    uint32_t l1PBufId = gatheredKvSTileIdxDe % pL1BufNum_;
                    uint32_t smToMm2FlagId = l1PBufId + UB_S_OTMP_BUF_STAGES;

                    Arch::CrossCoreFlag smToMm2Flag(smToMm2FlagId);
                    Arch::CrossCoreFlag mm2ToReFlag(Mm2ToReFlagId);
                    uint64_t prefixSumL0AStages = CalcCrossMm1Mm2PrefixSumL0ABStages(
                        gatheredKvSTileIdxDe, mm1L0ATotalStages_, mm2L0ATotalStages_, kvSLoopNum, false);
                    uint64_t prefixSumL0BStages = CalcCrossMm1Mm2PrefixSumL0ABStages(
                        gatheredKvSTileIdxDe, mm1L0BTotalStages_, mm2L0BTotalStages_, kvSLoopNum, false);
                    uint32_t yBlockIndexOffsetDe = (gatheredKvSTileIdxDe * kvBaseTile_) / blockShapeY_;
                    uint32_t yBlockIndexDe = gSparseIdx[gmOffsetSparseIdx].GetValue(yBlockIndexOffsetDe);
                    uint32_t kvDequantScaleOffsetDe =
                    curBatch * kvHeads_ * yBlockNumAligned_ + kvHeadIdx * yBlockNumAligned_ + yBlockIndexDe;
                    float vDequantScaleValue = gVDequantScale.GetValue(kvDequantScaleOffsetDe);
                    uint64_t deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&vDequantScaleValue));
                    if constexpr (RESCALE_ON_CUBE) {
                        // go 的 UB TLA：NZ2DN 输出为 ColumnMajor（DN）布局（仅末块实际使用，供 CopyL0CToDst<TensorC> 匹配特化）
                        // 半区固定 64（与 S 的 sShapeCol=64 一致）：FixPipe 的 nSize 需 16 倍数，
                        // 尾块实际列 < 64 时多搬的部分为 L0C dummy / go 缓冲固定分配区，无害
                        auto ubGoLayoutTla = tla::MakeLayout<float, NpuArch::layout::ColumnMajor>(embed_, 64);
                        auto ubGoTensorTla = tla::MakeTensor(ubGoTensor, ubGoLayoutTla, Arch::PositionUB{});
                        // GO^T 的 L1 TLA：zN [D, qS]，固定 qBaseTile_ 保证 FixPipe 的 nSize 为 16 倍数（尾块实际列
                        // < qBaseTile_ 时多搬的部分为 L0C dummy / GO^T 缓冲固定分配区，无害）；仅新 PV 使用，
                        // 旧路径（FP8 ND）不构造（避免未使用告警与无谓寻址）
                        AscendC::LocalTensor<float> l1GoTensor =
                            resource.l1Buf.template GetBufferByByte<float>(l1GoAddrStart_);
                        auto l1GoLayoutTla = tla::MakeLayout<float, NpuArch::layout::zN>(embed_, qBaseTile_);
                        auto l1GoTensorTla = tla::MakeTensor(l1GoTensor, l1GoLayoutTla, Arch::PositionL1{});
                        blockMmadPV(gmVTensorTla, ubGoTensorTla, l1GoTensorTla, gSparseIdx[gmOffsetSparseIdx], actualBlockShapePV,
                                    gatheredKvSTileIdxDe, kvSeqlen, kvBaseTile_, blockShapeY_, yBlockNumAval, yBlockNumRsvd,
                                    prefixSumL0AStages, prefixSumL0BStages, smToMm2Flag, mm2ToReFlag,
                                    (gatheredKvSTileIdxDe == 0), (gatheredKvSTileIdxDe == kvSLoopNum - 1), deqScalar);
                    } else {
                        // 旧路径（FP8 ND）：PV 结果逐 tile 搬 UB lo（2 份轮转），VEC 侧逐 tile rescale
                        auto ubOTmpLayoutTla = tla::MakeLayout<ElementOTmp, LayoutOTmp>(rowNumRound, embedRound);
                        auto ubOTmpTensorTla = tla::MakeTensor(ubOTmpTensor[ubOTmpBufId],
                            ubOTmpLayoutTla, Arch::PositionUB{});
                        blockMmadPV(gmVTensorTla, ubOTmpTensorTla, gSparseIdx[gmOffsetSparseIdx], actualBlockShapePV,
                                    gatheredKvSTileIdxDe, kvSeqlen, kvBaseTile_, blockShapeY_, yBlockNumAval, yBlockNumRsvd,
                                    prefixSumL0AStages, prefixSumL0BStages, smToMm2Flag, mm2ToReFlag, deqScalar);
                    }
#endif
#ifdef __DAV_VEC__
                    Arch::CrossCoreFlag mm2ToReFlag(Mm2ToReFlagId);
                    uint32_t curTileMod = gatheredKvSTileIdxDe % (PRE_LAUNCH + 1);
                    if constexpr (RESCALE_ON_CUBE) {
                        // rescale O：仅末块（中间块的 rescale 已在 CUBE 完成）
                        if (gatheredKvSTileIdxDe == kvSLoopNum - 1) {
                            epilogueRescaleO(
                                gmOTensorTla, gmLseTensorTla, actualBlockShapePV,
                                curTileMod, gatheredKvSTileIdxDe,
                                (gatheredKvSTileIdxDe == 0),
                                true,
                                mm2ToReFlag);
                        }
                    } else {
                        // rescale O：每个 tile 都执行（VEC 维护 go 递推 + dm 系数乘）
                        epilogueRescaleO(
                            gmOTensorTla, gmLseTensorTla, actualBlockShapePV,
                            curTileMod, gatheredKvSTileIdxDe,
                            (gatheredKvSTileIdxDe == 0),
                            (gatheredKvSTileIdxDe == kvSLoopNum - 1),
                            mm2ToReFlag);
                    }
#endif
                }
            }
        }
        // release reverse sync flags
        ReleaseSyncFlags<4, 4, 4>();
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
        if constexpr (RESCALE_ON_CUBE) {
            // GO^T [qS, D] fp32 1 份 + diag 块 [qS/2, qS/2] fp32 3 份，追加在 V 段之后（旧路径不需要，不计算）
            uint32_t mm2PTotalSize = mm2L1TileM_ * mm2L1TileKLeft_ * pL1BufNum_ * sizeof(ElementP);
            uint32_t mm2VTotalSize = mm2L1TileKRight_ * mm2L1TileN_ * vL1BufNum_ * sizeof(ElementV);
            l1GoAddrStart_ = mm2L1AddrStart_ + mm2PTotalSize + mm2VTotalSize;
            l1DiagAddrStart_ = l1GoAddrStart_ + qBaseTile_ * embed_ * sizeof(float);
        }
        if constexpr (transposedMm1) {
            mm1L0ATotalStages_ = (kvBaseTile_ / BlockMmadQK::L0_TILE_N) * (embed_ / BlockMmadQK::L0_TILE_K);
            mm1L0BTotalStages_ = (qBaseTile_ / BlockMmadQK::L0_TILE_N) * (embed_ / BlockMmadQK::L0_TILE_K);
        } else {
            mm1L0ATotalStages_ = (qBaseTile_ / BlockMmadQK::L0_TILE_M) * (embed_ / BlockMmadQK::L0_TILE_K);
            mm1L0BTotalStages_ = (kvBaseTile_ / BlockMmadQK::L0_TILE_N) * (embed_ / BlockMmadQK::L0_TILE_K);
        }
        if constexpr (RESCALE_ON_CUBE) {
            // 新 PV：每 tile L0A/L0B 用量 = PV^T 分块数 + RE 的 16 列块数（GO 沿 D 按 L0_TILE_M 切分、
            // diag 不切；与 PV 内部 counter 结构（mL0×kL0 / nL0×kL0 + fr）严格对应）
            mm2L0ATotalStages_ = (embed_ / BlockMmadPV::L0_TILE_M) * (kvBaseTile_ / BlockMmadPV::L0_TILE_K) +
                CeilDiv(qBaseTile_, 16) * (embed_ / BlockMmadPV::L0_TILE_M);
            mm2L0BTotalStages_ = (kvBaseTile_ / BlockMmadPV::L0_TILE_K) * (qBaseTile_ / BlockMmadPV::L0_TILE_N) +
                CeilDiv(qBaseTile_, 16);
        } else {
            mm2L0ATotalStages_ = (qBaseTile_ / BlockMmadPV::L0_TILE_M) * (kvBaseTile_ / BlockMmadPV::L0_TILE_K);
            mm2L0BTotalStages_ = (kvBaseTile_ / BlockMmadPV::L0_TILE_K) * (embed_ / BlockMmadPV::L0_TILE_N);
        }
    }

    __aicore__ inline
    void CalcUBufTileInfo()
    {
        // Prerequisites: qBaseTile_ <= 128, and kvBaseTile_ is a multiple of 128, guaranteed by tiling strategy.
        uint32_t qBaseTilePerSubCore = 64;
        uint32_t kvBaseTilePerSubCore = RoundUp(kvBaseTile_, 32);
        uint32_t embedPerSubCore = RoundUp(embed_, 32);
        uint32_t pExtraElemNum = 0;
        if constexpr (transposedMm1) {
            pExtraElemNum = Max(qBaseTilePerSubCore, kvBaseTilePerSubCore);
        }
        uint32_t sStartOffset = 0;
        uint32_t pStartOffset = sStartOffset + qBaseTilePerSubCore * kvBaseTilePerSubCore * sizeof(ElementS) * UB_S_OTMP_BUF_STAGES;
        uint32_t pTotalSize = (qBaseTilePerSubCore * kvBaseTilePerSubCore + pExtraElemNum) *
            sizeof(ElementP) * UB_S_OTMP_BUF_STAGES;
        uint32_t loStartOffset;
        uint32_t diagStartOffset;
        uint32_t goStartOffset;
        uint32_t dmStartOffset;
        uint32_t llStartOffset;
        if constexpr (RESCALE_ON_CUBE) {
            // rescaleOnCube on cube：OTmp(lo) 不再驻留 UB（偏移传 0）；
            // diag 块中转 [qS/2, qS/2] fp32 3 份轮转（与 L1 diag 3 份、dm 3 份同节奏，索引 l1PBufId）
            loStartOffset = 0;
            diagStartOffset = pStartOffset + pTotalSize;
            goStartOffset = diagStartOffset +
                qBaseTilePerSubCore * qBaseTilePerSubCore * sizeof(float) * (PRE_LAUNCH + 1);
        } else {
            // 旧路径：lo 2 份轮转、dm 3 份轮转（偏移与 HEAD 一致）
            loStartOffset = pStartOffset + pTotalSize;
            diagStartOffset = 0;
            goStartOffset = loStartOffset + qBaseTilePerSubCore * embedPerSubCore * sizeof(ElementOTmp) * UB_S_OTMP_BUF_STAGES;
        }
        uint32_t lmStartOffset = goStartOffset + qBaseTilePerSubCore * embedPerSubCore * sizeof(ElementOTmp);
        uint32_t gmStartOffset = lmStartOffset + qBaseTilePerSubCore * sizeof(float);
        // dm 3 份轮转（rescaleOnCube 同样保留）：UpdateDeltaMax 先算出 dm 存 dmUb，
        // SetDiagonalDeltaMax 再据此生成 diag 块，不再借用 diag 中转缓冲暂存 dm
        dmStartOffset = gmStartOffset + qBaseTilePerSubCore * sizeof(float);
        llStartOffset = dmStartOffset + qBaseTilePerSubCore * (PRE_LAUNCH + 1) * sizeof(float);
        uint32_t glStartOffset = llStartOffset + qBaseTilePerSubCore * sizeof(float);
        uint32_t lseStartOffset = glStartOffset + qBaseTilePerSubCore * sizeof(float);
        uint32_t maskStartOffset = lseStartOffset + qBaseTilePerSubCore * (32 / sizeof(float)) * sizeof(float);

        Epilogue::Block::UBufTileHelper uBufTileHelper(
            qBaseTilePerSubCore, kvBaseTilePerSubCore, embedPerSubCore,
            sStartOffset, pStartOffset, loStartOffset, goStartOffset,
            lmStartOffset, gmStartOffset, dmStartOffset, llStartOffset, glStartOffset,
            lseStartOffset, maskStartOffset, diagStartOffset);
        uBufTileHelper_ = uBufTileHelper;
    }

    __aicore__ inline
    uint64_t CalcCrossMm1Mm2PrefixSumL0ABStages(uint32_t gatheredKvSTileIdx, uint32_t singleMm1L0Stages,
                                                uint32_t singleMm2L0Stages, uint32_t kvSLoopNum, bool isCurPhaseMm1)
    {
        uint64_t prefixSumStages;
        if (isCurPhaseMm1) {
            prefixSumStages =
                (gatheredKvSTileIdx <= PRE_LAUNCH) ?
                    gatheredKvSTileIdx * singleMm1L0Stages :
                    gatheredKvSTileIdx * singleMm1L0Stages + (gatheredKvSTileIdx - PRE_LAUNCH) * singleMm2L0Stages;
        } else {
            prefixSumStages =
                (gatheredKvSTileIdx < kvSLoopNum - PRE_LAUNCH) ?
                    (gatheredKvSTileIdx + 1 + PRE_LAUNCH) * singleMm1L0Stages + gatheredKvSTileIdx * singleMm2L0Stages :
                    kvSLoopNum * singleMm1L0Stages + gatheredKvSTileIdx * singleMm2L0Stages;
        }
        return prefixSumStages;
    }

    __aicore__ inline
    void InitCrossCoreDstBuf(
        AscendC::LocalTensor<ElementP> (&l1PTensor)[MAX_CROSS_CORE_BUF_STAGES],
        AscendC::LocalTensor<ElementS> (&ubSTensor)[UB_S_OTMP_BUF_STAGES],
        AscendC::LocalTensor<ElementOTmp> &ubGoTensor,
        AscendC::LocalTensor<ElementOTmp> (&ubOTmpTensor)[UB_S_OTMP_BUF_STAGES],
        AscendC::LocalTensor<float> (&l1DiagTensor)[3])
    {
        for (uint32_t i = 0; i < pL1BufNum_; i++) {
            l1PTensor[i] = resource.l1Buf.template GetBufferByByte<ElementP>(
                mm2L1AddrStart_ + mm2L1TileM_ * mm2L1TileKLeft_ * sizeof(ElementP) * i);
        }
        for (uint32_t i = 0; i < UB_S_OTMP_BUF_STAGES; i++) {
            ubSTensor[i] = resource.ubBuf.template GetBufferByByte<ElementS>(
                uBufTileHelper_.sStartOffset +
                uBufTileHelper_.qBaseTilePerSubCore * uBufTileHelper_.kvBaseTilePerSubCore * sizeof(ElementS) * i);
        }
        if constexpr (RESCALE_ON_CUBE) {
            // go 仅末块使用（NZ2DN 搬入），1 份
            ubGoTensor = resource.ubBuf.template GetBufferByByte<ElementOTmp>(
                uBufTileHelper_.goStartOffset);
            // diag 块 L1 缓冲（每 stage 双子核各一块 16KB，3 份轮转，VEC 写 / CUBE 读）
            for (uint32_t i = 0; i < 3; i++) {
                l1DiagTensor[i] = resource.l1Buf.template GetBufferByByte<float>(
                    l1DiagAddrStart_ + 2 * 64 * 64 * sizeof(float) * i);
            }
        } else {
            // 旧路径：lo 逐 tile 2 份轮转（VEC rescaleO 读取）
            for (uint32_t i = 0; i < UB_S_OTMP_BUF_STAGES; i++) {
                ubOTmpTensor[i] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(
                    uBufTileHelper_.loStartOffset +
                    uBufTileHelper_.qBaseTilePerSubCore * uBufTileHelper_.embedPerSubCore * sizeof(ElementOTmp) * i);
            }
        }
    }

    template <uint32_t MM1_SM_MODE, uint32_t MM2_RE_MODE, uint32_t SM_MM2_MODE>
    __aicore__ inline
    void InitSyncFlags()
    {
#ifdef __DAV_CUBE__
        // same core sync between pipes
        // Query
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        // Key
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        // Value
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        // L0A
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        // L0B
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        // L0C
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID3);
        // cross core sync
        if constexpr (SM_MM2_MODE == Arch::CROSS_CORE_SYNC_MODE_4) {
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(Arch::FLAG_ID2);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(Arch::FLAG_ID18);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(Arch::FLAG_ID3);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(Arch::FLAG_ID19);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(Arch::FLAG_ID4);
            AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(Arch::FLAG_ID20);
            if constexpr (RESCALE_ON_CUBE) {
                // diag 块消费完成预置（CUBE 消费方，RE 读 diag 后 set / softmax 覆写前 wait）：6/7/8 + V1 侧 22/23/24
                // 必须与 Release VEC 的 6/7/8 收尾成对守卫：旧路径 flag 6/22 是 MM2_RE lo 协议的 flag，不能预置占用
                AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(Arch::FLAG_ID6);
                AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(Arch::FLAG_ID22);
                AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(Arch::FLAG_ID7);
                AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(Arch::FLAG_ID23);
                AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(Arch::FLAG_ID8);
                AscendC::CrossCoreSetFlag<SM_MM2_MODE, PIPE_MTE1>(Arch::FLAG_ID24);
            }
        }
#endif
#ifdef __DAV_VEC__
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        // mask2index
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        // softmax
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        // rescale
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        if constexpr (RESCALE_ON_CUBE) {
            // diag 中转缓冲 3 份轮转的 MTE3 搬运围栏（softmax 每 tile 按 l1PBufId 索引 wait/set 一次，
            // 与 Release VEC 的 EVENT_ID5~7 收尾成对；P 的 P→L1 围栏同模式）
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
        }

        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
        if constexpr (MM1_SM_MODE == Arch::CROSS_CORE_SYNC_MODE_4) {
            AscendC::CrossCoreSetFlag<MM1_SM_MODE, PIPE_V>(Arch::FLAG_ID0);
            AscendC::CrossCoreSetFlag<MM1_SM_MODE, PIPE_V>(Arch::FLAG_ID1);
        }
        if constexpr (MM2_RE_MODE == Arch::CROSS_CORE_SYNC_MODE_4) {
            AscendC::CrossCoreSetFlag<MM2_RE_MODE, PIPE_V>(Arch::FLAG_ID5);
            if constexpr (!RESCALE_ON_CUBE) {
                // 旧路径：lo 2 份轮转（flag 5/6 交替）
                AscendC::CrossCoreSetFlag<MM2_RE_MODE, PIPE_V>(Arch::FLAG_ID6);
            }
        }
#endif
    }

    template <uint32_t MM1_SM_MODE, uint32_t MM2_RE_MODE, uint32_t SM_MM2_MODE>
    __aicore__ inline
    void ReleaseSyncFlags()
    {
#ifdef __DAV_CUBE__
        // same core sync between pipes
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID3);
        if constexpr (MM1_SM_MODE == Arch::CROSS_CORE_SYNC_MODE_4) {
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(Arch::FLAG_ID0);
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(Arch::FLAG_ID1);
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(Arch::FLAG_ID16);
            AscendC::CrossCoreWaitFlag<MM1_SM_MODE, PIPE_FIX>(Arch::FLAG_ID17);
        }
        if constexpr (MM2_RE_MODE == Arch::CROSS_CORE_SYNC_MODE_4) {
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(Arch::FLAG_ID5);
            AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(Arch::FLAG_ID21);
            if constexpr (!RESCALE_ON_CUBE) {
                // 旧路径：lo 2 份轮转（flag 6/22 收尾）
                AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(Arch::FLAG_ID6);
                AscendC::CrossCoreWaitFlag<MM2_RE_MODE, PIPE_FIX>(Arch::FLAG_ID22);
            }
        }
#endif
#ifdef __DAV_VEC__
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
        if constexpr (RESCALE_ON_CUBE) {
            // diag 中转缓冲 3 份围栏收尾（与 Init VEC 的 EVENT_ID5~7 预置成对）
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID7);
        }
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(EVENT_ID1);
        if constexpr (SM_MM2_MODE == Arch::CROSS_CORE_SYNC_MODE_4) {
            AscendC::CrossCoreWaitFlag<SM_MM2_MODE, PIPE_MTE3>(Arch::FLAG_ID2);
            AscendC::CrossCoreWaitFlag<SM_MM2_MODE, PIPE_MTE3>(Arch::FLAG_ID3);
            AscendC::CrossCoreWaitFlag<SM_MM2_MODE, PIPE_MTE3>(Arch::FLAG_ID4);
            if constexpr (RESCALE_ON_CUBE) {
                // diag 块消费完成收尾（与 Init CUBE 的 6/22/7/23/8/24 预置成对；
                // 旧路径 flag 6 属 MM2_RE lo 协议，不在 SM_MM2 组收尾）
                AscendC::CrossCoreWaitFlag<SM_MM2_MODE, PIPE_MTE3>(Arch::FLAG_ID6);
                AscendC::CrossCoreWaitFlag<SM_MM2_MODE, PIPE_MTE3>(Arch::FLAG_ID7);
                AscendC::CrossCoreWaitFlag<SM_MM2_MODE, PIPE_MTE3>(Arch::FLAG_ID8);
            }
        }
#endif
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
    uint32_t l1GoAddrStart_ = 0;      // rescaleOnCube on cube：GO^T [D, qS] 在 L1 的起始地址
    uint32_t l1DiagAddrStart_ = 0;    // rescaleOnCube on cube：diag 块 [qS/2, qS/2] 在 L1 的起始地址
    Gemm::Block::Mm1L1TileHelper mm1L1TileHelper_;
    Gemm::Block::Mm2L1TileHelper mm2L1TileHelper_;
    Epilogue::Block::UBufTileHelper uBufTileHelper_;
};

}
