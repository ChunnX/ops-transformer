/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * SageAttention2 FA pipeline kernel (read S from GM, no MM1).
 *
 * Based on ascend950_fp8_mx_flash_attention_infer/fai_kernel.h, but:
 *   - MM1 (QK^T) is REMOVED → S is read directly from GM (pre-computed by mm1_int8_qk)
 *   - AIV reads S from GM → UB → online softmax → P in L1 (FP8 MX)
 *   - AIC MM2: FP8 MX PV matmul (MmadFAIPVMx)
 *   - AIV rescale O with LSE
 */

#ifndef SAGEATTN_FAI_PIPELINE_KERNEL_H
#define SAGEATTN_FAI_PIPELINE_KERNEL_H

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "tla/tensor.hpp"
#include "tla/layout.hpp"
#include "fai_kernel_utils.h"
#include "tiling_data_def.h"

using namespace Catlass;
using namespace tla;

// ===================== Pipeline kernel params =====================
struct FAIPipelineParams {
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR sBuffer;    // S (fp16) from mm1_int8_qk [Sq, Sk]
    GM_ADDR mask;
    GM_ADDR blockTables;
    GM_ADDR actualQSeqlen;
    GM_ADDR actualKvSeqlen;
    GM_ADDR o;
    GM_ADDR mxScaleQ;
    GM_ADDR mxScaleK;
    GM_ADDR mxScaleV;
    GM_ADDR pScale;
    GM_ADDR tiling;
};

// ===================== FA Pipeline Kernel =====================
// NOTE: BlockMmadQK is kept as a template parameter even though MM1 is removed,
//       because the original FAInferKernel derives types (ArchTag, L1TileShape, etc.)
//       from it. We keep the same interface for compatibility.
template <class BlockMmadQK, class BlockMmadPV, class EpilogueOnlineSoftmax, class EpilogueRescaleO,
          bool PAGED_CACHE_FLAG = false, bool ENABLE_P_SCALE = false>
class FAIPipelineKernel {
public:
    using ArchTag = typename BlockMmadQK::ArchTag;
    using L1TileShape = typename BlockMmadQK::L1TileShape;
    using ElementQ = typename BlockMmadQK::ElementA;
    using LayoutTagQ = typename BlockMmadQK::LayoutTagA;
    using ElementK = typename BlockMmadQK::ElementB;
    using LayoutTagK = typename BlockMmadQK::LayoutTagB;
    using ElementS = typename BlockMmadQK::ElementC;
    using LayoutTagS = typename BlockMmadQK::LayoutTagC;

    using ElementP = typename BlockMmadPV::ElementA;
    using LayoutTagP = typename BlockMmadPV::LayoutTagA;
    using ElementV = typename BlockMmadPV::ElementB;
    using LayoutTagV = typename BlockMmadPV::LayoutTagB;

    using ElementMask = typename EpilogueOnlineSoftmax::ElementMask;
    using LayoutTagMask = typename EpilogueOnlineSoftmax::LayoutTagMask;

    using ElementOTmp = typename EpilogueRescaleO::ElementOTmp;
    using LayoutTagOTmp = typename EpilogueRescaleO::LayoutTagOTmp;
    using ElementO = typename EpilogueRescaleO::ElementO;
    using LayoutTagO = typename EpilogueRescaleO::LayoutTagO;

    static constexpr uint32_t qSeqlenTemplateType = tla::get<0>(L1TileShape{});
    static constexpr uint32_t kvSeqlenTemplateType = tla::get<1>(L1TileShape{});
    static constexpr uint32_t embedTemplateType = tla::get<2>(L1TileShape{});

    static constexpr uint32_t MM2_LEFT_SIZE = qSeqlenTemplateType * kvSeqlenTemplateType * sizeof(ElementP);

    // S buffer type: fp16 from MM1 kernel
    using ElementSIn = half;

    // Methods
    CATLASS_DEVICE
    FAIPipelineKernel()
    {}

    CATLASS_DEVICE void Init(FAIPipelineParams const& params)
    {
        if ASCEND_IS_AIC {
            this->blockIdx = AscendC::GetBlockIdx();
        } else {
            this->blockIdx = AscendC::GetBlockIdx() >> 1;
        }
        this->subBlockIdx = AscendC::GetSubBlockIdx();
        constInfo.subBlockIdx = this->subBlockIdx;

        // Read tiling
        auto faTilingStruct = (__gm__ FATilingData*)params.tiling;
        auto &inputParamsRegbase = faTilingStruct->inputParamsRegbase;
        this->constInfo.scaleValue = static_cast<float>(inputParamsRegbase.scaleValue);
        this->constInfo.batch = inputParamsRegbase.batch;
        this->constInfo.qHeads = inputParamsRegbase.qHeads;
        this->constInfo.kvHeads = inputParamsRegbase.kvHeads;
        this->constInfo.groupSize = inputParamsRegbase.groupSize;
        this->constInfo.qSeqlen = inputParamsRegbase.qSeqlen;
        this->constInfo.kvSeqlen = inputParamsRegbase.kvSeqlen;
        this->constInfo.embed = inputParamsRegbase.embed;
        this->constInfo.attenMaskQSeqlen = inputParamsRegbase.attenMaskQSeqlen;
        this->constInfo.attenMaskKvSeqlen = inputParamsRegbase.attenMaskKvSeqlen;
        this->constInfo.headNumRatio = inputParamsRegbase.headNumRatio;
        this->constInfo.actualSeqLengthsSize = inputParamsRegbase.actualSeqLengthsSize;
        this->constInfo.actualSeqLengthsKVSize = inputParamsRegbase.actualSeqLengthsKVSize;
        this->constInfo.isActualSeqLengthsNull = inputParamsRegbase.isActualSeqLengthsNull;
        this->constInfo.isActualSeqLengthsKVNull = inputParamsRegbase.isActualSeqLengthsKVNull;

        if constexpr (PAGED_CACHE_FLAG) {
            this->constInfo.blockTableDim2 = inputParamsRegbase.blockTableDim2;
            this->constInfo.blockSize = inputParamsRegbase.blockSize;
            this->constInfo.paBlockNumSum = inputParamsRegbase.paBlockNumSum;
        }

        auto &multiCoreParamsRegbase = faTilingStruct->multiCoreParamsRegbase;
        this->constInfo.qSeqlenOuterSize = multiCoreParamsRegbase.qSeqlenOuterSize;
        this->constInfo.coreNum = multiCoreParamsRegbase.coreNum;
        this->constInfo.multiCoreInnerOffset = multiCoreParamsRegbase.sparseStartIdx[this->blockIdx];
        this->constInfo.multiCoreInnerLimit = multiCoreParamsRegbase.sparseStartIdx[this->blockIdx + 1];
        this->constInfo.bnAxisStartIdx = multiCoreParamsRegbase.bnAxisStartIdx[this->blockIdx];
        this->constInfo.bnAxisEndIdx = multiCoreParamsRegbase.bnAxisStartIdx[this->blockIdx + 1];

        CrossCoreSetFlag<4, PIPE_V>(MM2_RES_INTRA_EVENT[0]);
        CrossCoreSetFlag<4, PIPE_V>(MM2_RES_INTRA_EVENT[1]);
        CrossCoreSetFlag<4, PIPE_V>(MM1_RES_INTRA_EVENT[0]);
        CrossCoreSetFlag<4, PIPE_V>(MM1_RES_INTRA_EVENT[1]);

        this->constInfo.qSeqlenBase = qSeqlenTemplateType;
        this->constInfo.kvSeqlenBase = kvSeqlenTemplateType;

        if constexpr (ENABLE_P_SCALE) {
            AscendC::GlobalTensor<float> gmPscale;
            gmPscale.SetGlobalBuffer((__gm__ float *)params.pScale);
            this->constInfo.pScaleValue = gmPscale.GetValue(0);
        }

        // UB buffers
        for (int i = 0; i < NUM2; i++) {
            bmm1TensorList[i] = resource.ubBuf.template GetBufferByByte<ElementS>(ubBufAddrStart);
            ubBufAddrStart += MM1_RESULT_SIZE;
            bmm2TensorList[i] = resource.ubBuf.template GetBufferByByte<ElementOTmp>(ubBufAddrStart);
            ubBufAddrStart += MM2_RESULT_SIZE;
        }

        if ASCEND_IS_AIV {
            for (int i = 0; i < KERNEL_TASK_NUM; i++) {
                sumUb[i] = resource.ubBuf.template GetBufferByByte<ElementS>(ubBufAddrStart);
                ubBufAddrStart += SHARE_UB_SIZE;
                expUb[i] = resource.ubBuf.template GetBufferByByte<ElementS>(ubBufAddrStart);
                ubBufAddrStart += SHARE_UB_SIZE;
                maxUb[i] = resource.ubBuf.template GetBufferByByte<ElementS>(ubBufAddrStart);
                ubBufAddrStart += SHARE_UB_SIZE;
            }
        }

        // L1 buffers for MM2
        for (int i = 0; i < KERNEL_TASK_NUM; i++) {
            mm2AL1TensorList[i] = resource.l1Buf.template GetBufferByByte<ElementP>(l1BufAddrStart + i * MM2_LEFT_SIZE);
        }
        l1BufAddrStart += KERNEL_TASK_NUM * MM2_LEFT_SIZE;
    }

    CATLASS_DEVICE void operator()(FAIPipelineParams const &params) {
        Init(params);
        uint32_t l0CBufAddrStart = 0;
        // Create BlockMmadQK to initialize MMA hardware (dummy, not used for computation)
        BlockMmadQK blockMmadMmadQK(resource, l1BufAddrStart, l0CBufAddrStart);
        BlockMmadPV blockMmadMmadPV(resource, l1BufAddrStart, l0CBufAddrStart);
        EpilogueOnlineSoftmax epilogueOnlineSoftmax(resource, constInfo.scaleValue, ubBufAddrStart, constInfo.scaleValue);
        EpilogueRescaleO epilogueRescaleO(resource, ubBufAddrStart);

        int32_t blockNum = this->constInfo.coreNum;
        if (this->blockIdx >= blockNum) return;

        int64_t batch = this->constInfo.batch;
        int64_t qHeads = this->constInfo.qHeads;
        int64_t qSeqlen = this->constInfo.qSeqlen;
        int64_t kvHeads = this->constInfo.kvHeads;
        int64_t kvSeqlen = this->constInfo.kvSeqlen;
        int64_t groupSize = this->constInfo.groupSize;
        int64_t embed = this->constInfo.embed;
        int64_t blockSize = this->constInfo.blockSize;
        if constexpr (PAGED_CACHE_FLAG) {
            kvSeqlen = RoundUp(kvSeqlen, blockSize);
        }
        int64_t kvSeqlenMask = kvSeqlen;

        // Init tensor layouts
        using ElementMxScale = float8_e8m0_t;

        // V (FP8 MX)
        AscendC::GlobalTensor<ElementV> gmV;
        gmV.SetGlobalBuffer((__gm__ ElementV *)params.v);
        auto layoutV = MakeLayout<ElementV, LayoutTagV>(batch * kvSeqlen, kvHeads * embed);
        auto tensorVWithLayout = tla::MakeTensor(gmV, layoutV, Arch::PositionGM{});

        AscendC::GlobalTensor<ElementMxScale> gmMxScaleV;
        gmMxScaleV.SetGlobalBuffer((__gm__ ElementMxScale *)params.mxScaleV);
        auto layoutMxScaleV = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagV, true>(
            CeilDiv<MX_SCALE_GROUP_NUM>(batch * kvSeqlen), kvHeads * embed);
        auto tensorMxScaleVWithLayout = tla::MakeTensor(gmMxScaleV, layoutMxScaleV, Arch::PositionGM{});

        // S buffer from GM (fp16)
        AscendC::GlobalTensor<ElementSIn> gmS;
        gmS.SetGlobalBuffer((__gm__ ElementSIn *)params.sBuffer);
        auto layoutS = MakeLayout<ElementSIn, LayoutTagS>(qSeqlen, kvSeqlen);
        auto tensorSWithLayout = tla::MakeTensor(gmS, layoutS, Arch::PositionGM{});

        // Mask
        AscendC::GlobalTensor<ElementMask> gmMask;
        gmMask.SetGlobalBuffer((__gm__ ElementMask *)params.mask);
        auto layoutMask = MakeLayout<ElementMask, LayoutTagMask>(batch * qSeqlen, kvSeqlenMask);
        auto tensorMaskWithLayout = tla::MakeTensor(gmMask, layoutMask, Arch::PositionGM{});

        // BlockTable
        AscendC::GlobalTensor<int32_t> tensorTable;
        tensorTable.SetGlobalBuffer((__gm__ int32_t*)params.blockTables);

        // Output O
        AscendC::GlobalTensor<ElementO> attentionOutGm;
        attentionOutGm.SetGlobalBuffer((__gm__ ElementO *)params.o);
        auto layoutO = MakeLayout<ElementO, LayoutTagO>(batch * qSeqlen, qHeads * embed);
        auto attentionOutGmWithLayout = tla::MakeTensor(attentionOutGm, layoutO, Arch::PositionGM{});

        uint32_t maxBlockNumPerBatch = this->constInfo.blockTableDim2;

        // ===================== Main loop =====================
        int64_t qSeqAxisStartIdx;
        uint32_t bnAxisStartIdx, bnAxisEndIdx;
        int64_t kvSeqLoopLimit;
        int64_t nextQSeqAxisIdx = this->constInfo.multiCoreInnerLimit;
        bnAxisStartIdx = this->constInfo.bnAxisStartIdx;
        qSeqAxisStartIdx = this->constInfo.multiCoreInnerOffset;
        if (likely((this->constInfo.coreNum - 1) > this->blockIdx)) {
            bnAxisEndIdx = this->constInfo.bnAxisEndIdx;
            if (nextQSeqAxisIdx != 0) bnAxisEndIdx++;
        } else {
            bnAxisEndIdx = this->constInfo.batch * this->constInfo.kvHeads * this->constInfo.headNumRatio;
        }

        int64_t taskId = 0;
        bool notLast = true;
        bool isLastBmm1 = false;
        int64_t multiCoreInnerIdx = 1;

        for (uint32_t bnIdx = bnAxisStartIdx; bnIdx < bnAxisEndIdx; ++bnIdx) {
            bool lastBN = (bnIdx == bnAxisEndIdx - 1);
            runParam.batchOuterIdx = bnIdx / (this->constInfo.kvHeads * this->constInfo.headNumRatio);
            runParam.kvHeadsOuterIdx = (bnIdx / this->constInfo.headNumRatio) % this->constInfo.kvHeads;
            ComputeParamBatch(runParam, this->constInfo, this->attenMaskInfo);
            ComputeQseqLoopInfo<qSeqlenTemplateType>(runParam, this->constInfo, lastBN, nextQSeqAxisIdx);
            int64_t tempQSeqAxisEnd = lastBN ? (runParam.qSeqLoopTimes + 3) : runParam.qSeqLoopTimes;

            for (int64_t qSeqAxisIndex = qSeqAxisStartIdx; qSeqAxisIndex < tempQSeqAxisEnd; ++qSeqAxisIndex) {
                bool notLastThreeLoop = true;
                bool notLastTwoLoop = true;
                if (lastBN) {
                    int32_t extraQSeqAxis = qSeqAxisIndex - runParam.qSeqLoopTimes;
                    switch (extraQSeqAxis) {
                        case -1: isLastBmm1 = true; break;
                        case 0:  notLastThreeLoop = false; break;
                        case 1:  notLastThreeLoop = false; notLastTwoLoop = false; break;
                        case 2:  notLast = false; notLastThreeLoop = false; notLastTwoLoop = false; break;
                    }
                }

                if (notLastThreeLoop) {
                    runParam.groupIdx = bnIdx % this->constInfo.headNumRatio;
                    runParam.qSeqOuterAxisIdx = qSeqAxisIndex % this->constInfo.qSeqlenOuterSize;
                    ComputeParamQSeq<qSeqlenTemplateType>(runParam, this->constInfo, qSeqAxisIndex);
                    ComputeKvSeqLoopInfo<kvSeqlenTemplateType>(runParam, this->constInfo);
                    kvSeqLoopLimit = runParam.kvSeqLoopEndIdx - 1;
                } else {
                    runParam.kvSeqLoopStartIdx = 0;
                    kvSeqLoopLimit = 0;
                }

                // ---- kvSeq loop ----
                for (int64_t kvSeqLoopCount = runParam.kvSeqLoopStartIdx; kvSeqLoopCount <= kvSeqLoopLimit; ++kvSeqLoopCount) {
                    if (notLastThreeLoop) {
                        RunInfo &runInfo1 = runInfo[taskId & 3];
                        this->SetRunInfo(runInfo1, runParam, taskId, kvSeqLoopCount, kvSeqLoopLimit, multiCoreInnerIdx);
                        if ASCEND_IS_AIC {
                            CalcKvSeqCoord(runInfo1, this->constInfo);
                            CalcQSeqCoord(runInfo1, this->constInfo);
                            // --- NO MM1 (QK^T) on AIC ---
                            // S is already in GM from mm1_int8_qk kernel.
                            CrossCoreSetFlag<4, PIPE_FIX>(SYNC_C1_V1_FLAG[runInfo1.taskIdMod2]);
                            CrossCoreSetFlag<4, PIPE_FIX>(16 + SYNC_C1_V1_FLAG[runInfo1.taskIdMod2]);
                        }
                    }

                    // ---- AIV: online softmax (read S from GM) ----
                    if (taskId > 0 && notLastTwoLoop) {
                        if ASCEND_IS_AIV {
                            RunInfo &runInfo3 = runInfo[(taskId + 3) & 3];
                            auto &taskIdMod2 = runInfo3.taskIdMod2;
                            auto &taskIdMod3 = runInfo3.taskIdMod3;
                            auto &multiCoreIdxMod3 = runInfo3.multiCoreIdxMod3;

                            CrossCoreWaitFlag<4, PIPE_V>(SYNC_C1_V1_FLAG[taskIdMod2]);

                            // ---- Read S from GM (fp16), convert to float in UB ----
                            int64_t qSeqOffset = runInfo3.qSeqOuterAxisIdx * qSeqlenTemplateType + runInfo3.firstHalfQSeqRealSize * constInfo.subBlockIdx;
                            int64_t kvSeqOffset = runInfo3.kvSeqLoopCount * kvSeqlenTemplateType;
                            int64_t bOffset = runInfo3.batchOuterIdx * qSeqlen;

                            auto gmSTile = GetTile(
                                tensorSWithLayout,
                                tla::MakeCoord(qSeqOffset, kvSeqOffset),
                                tla::MakeShape(runInfo3.halfQSeqRealSize, runInfo3.kvSeqRealSize)
                            );

                            // Copy S from GM to UB (fp16 temporary) — FIXED address, do NOT advance
                                                        uint32_t sTileSize = runInfo3.halfQSeqRealSize * runInfo3.kvSeqRealSize;
                                                        auto sUbFp16 = resource.ubBuf.template GetBufferByByte<ElementSIn>(ubBufAddrStart);

                                                                                    {   // DataCopy GM→UB (fp16)
                                                            uint32_t blockLen = (sTileSize * sizeof(ElementSIn) + 31) / 32;
                                                            auto gmData = gmSTile.data();
                                                            auto ubData = sUbFp16;
                                                            AscendC::DataCopy(ubData, gmData, {static_cast<uint16_t>(1), static_cast<uint16_t>(blockLen), 0, 0});
                                                        }

                            // Convert fp16→float in bmm1TensorList
                            auto bmm1Layout = tla::MakeLayout<ElementS, LayoutTagS>(runInfo3.halfQSeqRealSize, runInfo3.kvSeqRealSize);
                            auto bmm1Tensor = tla::MakeTensor(bmm1TensorList[taskIdMod2], bmm1Layout, Arch::PositionUB{});
                            AscendC::Cast<ElementS, ElementSIn, false>(
                                bmm1TensorList[taskIdMod2], sUbFp16,
                                AscendC::RoundMode::CAST_NONE, (uint64_t)0, sTileSize,
                                AscendC::UnaryRepeatParams(1, 1, 8, 8));

                            // L1 output for MM2
                            auto l1Vf1OutLayout = tla::MakeLayout<ElementP, LayoutTagP>(qSeqlenTemplateType, kvSeqlenTemplateType);
                            auto l1Vf1OutTensor = tla::MakeTensor(mm2AL1TensorList[taskIdMod3], l1Vf1OutLayout, Arch::PositionL1{});
                            auto l1Vf1OutTile = GetTile(
                                l1Vf1OutTensor,
                                tla::MakeCoord(constInfo.subBlockIdx * runInfo3.firstHalfQSeqRealSize, 0),
                                tla::MakeShape(runInfo3.halfQSeqRealSize, kvSeqlenTemplateType)
                            );

                            auto gmMaskTile = GetTile(
                                tensorMaskWithLayout,
                                tla::MakeCoord(bOffset + qSeqOffset, kvSeqOffset),
                                tla::MakeShape(runInfo3.halfQSeqRealSize, runInfo3.kvSeqRealSize)
                            );

                            bool isFirstLoop = (runInfo3.kvSeqLoopCount == runInfo3.kvSeqLoopStartIdx);

                            epilogueOnlineSoftmax(
                                l1Vf1OutTile,
                                sumUb[multiCoreIdxMod3], maxUb[multiCoreIdxMod3], expUb[taskIdMod3],
                                bmm1Tensor, gmMaskTile,
                                !isFirstLoop,
                                taskIdMod2,
                                taskIdMod3,
                                MM1_RES_INTRA_EVENT[taskIdMod2], SYNC_V1_C2_FLAG[taskIdMod3]
                            );
                        }
                    }

                    // ---- AIC: MM2 (P × V with FP8 MX) ----
                    if (taskId > 1 && notLast) {
                        if ASCEND_IS_AIC {
                            RunInfo &runInfo2 = runInfo[(taskId + 2) & 3];
                            auto &taskIdMod2 = runInfo2.taskIdMod2;
                            auto &taskIdMod3 = runInfo2.taskIdMod3;

                            CrossCoreWaitFlag<4, PIPE_MTE1>(SYNC_V1_C2_FLAG[taskIdMod3]);
                            CrossCoreWaitFlag<4, PIPE_MTE1>(16 + SYNC_V1_C2_FLAG[taskIdMod3]);

                            auto layoutMM2O = tla::MakeLayout<ElementOTmp, LayoutTagOTmp>(runInfo2.qSeqRealSize, embedTemplateType);
                            auto mm2OutTensor = tla::MakeTensor(bmm2TensorList[taskIdMod2], layoutMM2O, Arch::PositionUB{});

                            auto layoutVec1O = tla::MakeLayout<ElementP, LayoutTagP>(qSeqlenTemplateType, kvSeqlenTemplateType);
                            auto mm2AL1Tensor = tla::MakeTensor(mm2AL1TensorList[taskIdMod3], layoutVec1O, Arch::PositionL1{});

                            auto kCoord = 0;
                            auto nCoord = runInfo2.kvHeadsOuterIdx * embed;
                            auto kShape = runInfo2.kvSeqRealSize;
                            if constexpr (PAGED_CACHE_FLAG) {
                                kShape = batch * kvSeqlen;
                            } else {
                                kCoord = coordInfo[runInfo2.taskIdMod3].curBIdx * kvSeqlen + coordInfo[runInfo2.taskIdMod3].kvSeqCoord;
                            }

                            auto tensorInV = GetTile(
                                tensorVWithLayout,
                                tla::MakeCoord(kCoord, nCoord),
                                tla::MakeShape(kShape, this->constInfo.embed)
                            );
                            auto tensorInMxScaleV = GetTile(
                                tensorMxScaleVWithLayout,
                                tla::MakeCoord(kCoord / MX_SCALE_GROUP_NUM, nCoord),
                                tla::MakeShape(CeilDiv<MX_SCALE_GROUP_NUM>(kShape), this->constInfo.embed)
                            );

                            auto actualShape = tla::MakeShape(runInfo2.qSeqRealSize, embedTemplateType, runInfo2.kvSeqRealSize);
                            auto tensorInTableV = tensorTable[runInfo2.blockTableOffset];

                            blockMmadMmadPV(
                                mm2AL1Tensor, tensorInV, mm2OutTensor, tensorInMxScaleV,
                                tensorInTableV, actualShape,
                                taskIdMod2, this->constInfo.blockSize
                            );

                            CrossCoreSetFlag<4, PIPE_FIX>(SYNC_C2_V2_FLAG[runInfo2.taskIdMod2]);
                            CrossCoreSetFlag<4, PIPE_FIX>(16 + SYNC_C2_V2_FLAG[runInfo2.taskIdMod2]);
                        }
                    }

                    // ---- AIV: rescale O ----
                    if (taskId > 2) {
                        if ASCEND_IS_AIV {
                            RunInfo &runInfo3 = runInfo[(taskId + 1) & 3];
                            auto &taskIdMod2 = runInfo3.taskIdMod2;
                            auto &taskIdMod3 = runInfo3.taskIdMod3;
                            auto &multiCoreIdxMod3 = runInfo3.multiCoreIdxMod3;

                            bool isFirstLoop = (runInfo3.kvSeqLoopCount == runInfo3.kvSeqLoopStartIdx);
                            bool isLastUpdate = (runInfo3.kvSeqLoopCount == runInfo3.kvSeqLoopLimit);

                            CrossCoreWaitFlag<4, PIPE_V>(SYNC_C2_V2_FLAG[taskIdMod2]);

                            auto bmm2Layout = MakeLayout<ElementOTmp, LayoutTagOTmp>(runInfo3.halfQSeqRealSize, embedTemplateType);
                            auto bmm2Tensor = tla::MakeTensor(bmm2TensorList[taskIdMod2], bmm2Layout, Arch::PositionUB{});

                            int64_t bOffset = runInfo3.batchOuterIdx * qSeqlen;
                            int64_t qSeqOffset = runInfo3.qSeqOuterAxisIdx * qSeqlenTemplateType + runInfo3.firstHalfQSeqRealSize * constInfo.subBlockIdx;
                            int64_t kvHeadsOffset = runInfo3.kvHeadsOuterIdx * groupSize * embed;
                            int64_t embedOffset = runInfo3.groupIdx * embed;

                            auto attenOutGmTile = GetTile(
                                attentionOutGmWithLayout,
                                tla::MakeCoord(bOffset + qSeqOffset, kvHeadsOffset + embedOffset),
                                tla::MakeShape(runInfo3.halfQSeqRealSize, embedTemplateType)
                            );

                            epilogueRescaleO(
                                attenOutGmTile,
                                expUb[taskIdMod3], sumUb[multiCoreIdxMod3], bmm2Tensor,
                                isFirstLoop, isLastUpdate,
                                MM2_RES_INTRA_EVENT[taskIdMod2]
                            );
                        }
                    }
                    ++taskId;
                }
                ++multiCoreInnerIdx;
            }
            qSeqAxisStartIdx = 0;
        }
    }

private:
    static constexpr uint32_t CV_RATIO = 2;
    static constexpr uint32_t NUM2 = 2;
    static constexpr uint32_t KERNEL_TASK_NUM = 3;

    static constexpr uint32_t embedTemplateAlign64 = Align64Func((uint16_t)embedTemplateType);
    static constexpr uint32_t MM1_RESULT_SIZE = qSeqlenTemplateType / CV_RATIO * kvSeqlenTemplateType * sizeof(ElementS);
    static constexpr uint32_t MM2_RESULT_SIZE = qSeqlenTemplateType / CV_RATIO * embedTemplateAlign64 * sizeof(ElementOTmp);
    static constexpr uint32_t SHARE_UB_SIZE = CeilDiv(qSeqlenTemplateType, NUM2) * sizeof(ElementS);

    // Sync event IDs (same as MX FA reference)
    static constexpr uint64_t SYNC_C1_V1_FLAG[2] = {0, 1};
    static constexpr uint64_t SYNC_V1_C2_FLAG[3] = {2, 3, 4};
    static constexpr uint64_t SYNC_C2_V2_FLAG[2] = {5, 6};
    static constexpr uint64_t MM2_RES_INTRA_EVENT[2] = {7, 8};
    static constexpr uint64_t MM1_RES_INTRA_EVENT[2] = {9, 10};

    AscendC::LocalTensor<ElementS> bmm1TensorList[NUM2];
    AscendC::LocalTensor<ElementP> mm2AL1TensorList[KERNEL_TASK_NUM];
    AscendC::LocalTensor<ElementOTmp> bmm2TensorList[NUM2];
    AscendC::LocalTensor<ElementS> expUb[KERNEL_TASK_NUM];
    AscendC::LocalTensor<ElementS> sumUb[KERNEL_TASK_NUM];
    AscendC::LocalTensor<ElementS> maxUb[KERNEL_TASK_NUM];
    ConstInfo constInfo;
    AttenMaskInfo attenMaskInfo;
    uint32_t blockIdx;
    uint32_t subBlockIdx;

    RunInfo runInfo[4];
    RunParamStr runParam;
    uint32_t l1BufAddrStart = 0;
    uint32_t ubBufAddrStart = 0;

    Arch::Resource<ArchTag> resource;
    CubeCoordInfo coordInfo[3];

    // ============ private functions ============
    CATLASS_DEVICE inline void SetRunInfo(
        RunInfo &runInfo, RunParamStr &runParam, int64_t taskId, int64_t kvSeqLoopCount,
        int64_t kvSeqLoopLimit, int64_t multiCoreInnerIdx)
    {
        runInfo.kvSeqAxisStartIdx = runParam.kvSeqAxisLineStartIdx;
        runInfo.kvSeqLoopStartIdx = runParam.kvSeqLoopStartIdx;
        runInfo.kvSeqAxisEndIdx = runParam.kvSeqAxisLineEndIdx;
        runInfo.kvSeqLoopCount = kvSeqLoopCount;
        if (runInfo.multiCoreInnerIdx != multiCoreInnerIdx) {
            runInfo.qSeqOuterAxisIdx = runParam.qSeqOuterAxisIdx;
            runInfo.batchOuterIdx = runParam.batchOuterIdx;
            runInfo.kvHeadsOuterIdx = runParam.kvHeadsOuterIdx;
            runInfo.groupIdx = runParam.groupIdx;
            runInfo.multiCoreInnerIdx = multiCoreInnerIdx;
            runInfo.multiCoreIdxMod2 = multiCoreInnerIdx & 1;
            runInfo.multiCoreIdxMod3 = multiCoreInnerIdx % 3;
        }
        runInfo.taskId = taskId;
        runInfo.taskIdMod2 = taskId & 1;
        runInfo.taskIdMod3 = taskId % 3;
        runInfo.kvSeqLoopLimit = kvSeqLoopLimit;
        runInfo.actualQSeqSize = runParam.actualQSeqSize;
        runInfo.actualKvSeqSize = runParam.actualKvSeqSize;
        this->ComputeBmm1Tail(runInfo, runParam);
        runInfo.batchOuterIdx = runParam.batchOuterIdx;
    }

    CATLASS_DEVICE void ComputeBmm1Tail(RunInfo &runInfo, RunParamStr &runParam) {
        runInfo.qSeqRealSize = runParam.qSeqRealSize;
        runInfo.halfQSeqRealSize = runParam.halfQSeqRealSize;
        runInfo.firstHalfQSeqRealSize = runParam.firstHalfQSeqRealSize;
        runInfo.kvSeqRealSize = this->constInfo.kvSeqlenBase;
        if ((runInfo.kvSeqLoopCount + 1) * runInfo.kvSeqRealSize > runInfo.kvSeqAxisEndIdx) {
            runInfo.kvSeqRealSize = runInfo.kvSeqAxisEndIdx - runInfo.kvSeqLoopCount * runInfo.kvSeqRealSize;
        }
    }

    CATLASS_DEVICE void CalcQSeqCoord(RunInfo &runInfo, ConstInfo &constInfo) {
        coordInfo[runInfo.taskIdMod3].qSeqCoord = runInfo.qSeqOuterAxisIdx * this->constInfo.qSeqlenBase;
    }

    CATLASS_DEVICE void CalcKvSeqCoord(RunInfo &runInfo, ConstInfo &constInfo) {
        coordInfo[runInfo.taskIdMod3].kvSeqCoord = runInfo.kvSeqAxisStartIdx +
            (runInfo.kvSeqLoopCount - runInfo.kvSeqLoopStartIdx) * this->constInfo.kvSeqlenBase;
        coordInfo[runInfo.taskIdMod3].curBIdx = runInfo.batchOuterIdx;
    }
};

// ===================== Entry point =====================
// NOTE: BlockMmadQK is kept as a dummy to satisfy the template interface.
//       The actual MM1 computation is done by the separate mm1_int8_qk kernel.
template <class InDtype, class OutDtype, bool enableMaskFlag = false, bool enablePaFlag = false, bool enablePscale = false>
CATLASS_GLOBAL void FAInferPipeline(
    GM_ADDR q,
    GM_ADDR k,
    GM_ADDR v,
    GM_ADDR sBuffer,     // S from mm1_int8_qk (fp16)
    GM_ADDR mask,
    GM_ADDR blockTables,
    GM_ADDR o,
    GM_ADDR actualQSeqlen,
    GM_ADDR actualKvSeqlen,
    GM_ADDR mxScaleQ,
    GM_ADDR mxScaleK,
    GM_ADDR mxScaleV,
    GM_ADDR pScale,
    GM_ADDR tiling
) {
    using ArchTag = Arch::Ascend950;
    // Dummy QK types (for BlockMmadQK template params — MM1 is not computed here)
    // Must use MXFP8 types (float8_e4m3_t) to satisfy MmadFAIQKMx Cube MMA validation
    using ElementQ = float8_e4m3_t;
    using LayoutTagQ = layout::RowMajor;
    using ElementK = float8_e4m3_t;
    using LayoutTagK = layout::ColumnMajor;
    using ElementS = float;
    using LayoutTagS = layout::RowMajor;

    // PV types (actual MM2 computation)
    using ElementP = InDtype;
    using LayoutTagP = layout::zN;
    using ElementV = InDtype;
    using LayoutTagV = layout::RowMajor;
    using ElementO = OutDtype;
    using LayoutTagO = layout::RowMajor;
    using ElementMask = uint8_t;
    using LayoutTagMask = layout::RowMajor;
    using ElementOTmp = float;
    using LayoutTagOTmp = layout::RowMajor;
    using ElementMxScale = float8_e8m0_t;

    // Dummy MX scale layouts (for template instantiation)
    auto layoutMxScaleQ = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagQ, false>(128, 128);
    auto layoutMxScaleK = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagK, true>(128, 128);
    auto layoutMxScaleP = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagQ, false>(128, 128);
    auto layoutMxScaleV = tla::MakeMxScaleLayout<ElementMxScale, LayoutTagV, true>(128, 128);

    using L1TileShape = tla::Shape<_128, _128, _128>;
    using L0TileShape = L1TileShape;

    // Dummy MM1 block (QK^T) — not used for computation, only for template compatibility
    using DispatchPolicyQK = Gemm::MmadFAIQKMx<ArchTag, enablePaFlag>;
    using TileCopyQK = Gemm::Tile::PackedMxTileCopyTlaToUB<
        ArchTag, ElementQ, LayoutTagQ, ElementK, LayoutTagK, ElementMxScale, decltype(layoutMxScaleQ),
        ElementMxScale, decltype(layoutMxScaleK),
        ElementS, LayoutTagS, void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using TileMmadQK = Gemm::Tile::TileMmadTla<ArchTag, ElementQ, typename TileCopyQK::LayoutTagL1A>;
    using BlockMmadQK = Gemm::Block::BlockMmadTla<
        DispatchPolicyQK, L1TileShape, L0TileShape, ElementQ, ElementK, ElementS, void, TileCopyQK, TileMmadQK>;

    // MM2: P × V (FP8 MX)
    using DispatchPolicyPV = Gemm::MmadFAIPVMx<ArchTag, enablePaFlag>;
    using TileCopyPV = Gemm::Tile::PackedMxTileCopyTlaToUB<
        ArchTag, ElementP, LayoutTagP, ElementV, LayoutTagV, ElementMxScale, decltype(layoutMxScaleP),
        ElementMxScale, decltype(layoutMxScaleV),
        ElementOTmp, LayoutTagV, void, Gemm::Tile::CopyL0CToUBMode::SPLIT_M>;
    using TileMmadPV = Gemm::Tile::TileMmadTla<ArchTag, ElementP, typename TileCopyPV::LayoutTagL1A>;
    using BlockMmadPV = Gemm::Block::BlockMmadTla<
        DispatchPolicyPV, L1TileShape, L0TileShape, ElementP, ElementV, ElementOTmp, void, TileCopyPV, TileMmadPV>;

    // Online softmax
    using DispatchPolicySoftmax = Epilogue::EpilogueAscend950FASoftmax<enableMaskFlag, enablePscale>;
    using PType = Gemm::GemmType<ElementP, LayoutTagP>;
    using SType = Gemm::GemmType<ElementS, LayoutTagS>;
    using maskType = Gemm::GemmType<ElementMask, LayoutTagMask>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<
        DispatchPolicySoftmax, L1TileShape, PType, SType, maskType>;

    // Rescale O
    using DispatchPolicyRescaleO = Epilogue::EpilogueAscend950FARescaleO;
    using OType = Gemm::GemmType<ElementO, LayoutTagO>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutTagOTmp>;
    using EpilogueRescaleO = Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, L1TileShape, OType, OTmpType>;

    using FAIKernel = FAIPipelineKernel<
        BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO, enablePaFlag, enablePscale>;

    FAIPipelineParams params{q, k, v, sBuffer, mask, blockTables, actualQSeqlen, actualKvSeqlen,
                             o, mxScaleQ, mxScaleK, mxScaleV, pScale, tiling};
    FAIKernel flashAttnInfer;
    flashAttnInfer(params);
}

#endif // SAGEATTN_FAI_PIPELINE_KERNEL_H