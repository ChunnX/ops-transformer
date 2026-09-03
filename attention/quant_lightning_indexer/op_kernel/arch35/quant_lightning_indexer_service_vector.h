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
 * \file quant_lightning_indexer_service_vector.h
 * \brief
 */
#ifndef QUANT_LIGHTNING_INDEXER_SERVICE_VECTOR_H
#define QUANT_LIGHTNING_INDEXER_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "../quant_lightning_indexer_common.h"
#include "../arch35/vf/quant_lightning_indexer_vector1.h"
#include "../arch35/vf/quant_lightning_indexer_topk.h"

namespace QLIKernel {
using namespace QLICommon;
constexpr uint32_t TRUNK_LEN_32K = 32768;
constexpr uint32_t TRUNK_LEN_16K = 16384;
constexpr uint32_t TRUNK_LEN_6K = 6144;
constexpr uint64_t AIV_UB_LIMIT = 256ULL * 1024ULL;
template <typename QLIT>
class QLIVector {
public:
    // =================================类型定义区=================================
    static constexpr LI_LAYOUT Q_LAYOUT_T = QLIT::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = QLIT::keyLayout;
    static constexpr bool PAGE_ATTENTION = QLIT::pageAttention;
    using W_T = typename QLIT::weightType;
    using SCALE_T = typename QLIT::scaleType;
    using QK_T = typename QLIT::qkType;
    using SCORE_T = typename QLIT::scoreType;

    __aicore__ inline QLIVector(){};
    __aicore__ inline void ProcessVec1(const QLICommon::RunInfo &info);
    __aicore__ inline void ProcessTopK(const QLICommon::RunInfo &info);
    __aicore__ inline void ProcessLD();
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct QLICommon::ConstInfo &constInfo,
                                      const QLITilingData *__restrict tilingData);
    __aicore__ inline void InitVecWorkspaceTensor(GlobalTensor<SCORE_T> scoreGm);
    __aicore__ inline void InitLDTensor(GlobalTensor<SCORE_T> ldValGm, GlobalTensor<uint32_t> ldIdxGm,
                                        GlobalTensor<int64_t> ldParamGm);
    __aicore__ inline void InitVecInputTensor(GlobalTensor<W_T> weightsGm, GlobalTensor<SCALE_T> qScaleGm,
                                              GlobalTensor<SCALE_T> kScaleGm, GlobalTensor<int32_t> indiceOutGm,
                                              GlobalTensor<int32_t> blockTableGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void FlushLDWrites();  // L-fix

protected:
    GlobalTensor<SCORE_T> scoreGm;
    GlobalTensor<SCORE_T> ldValGm;
    GlobalTensor<uint32_t> ldIdxGm;
    GlobalTensor<int64_t> ldParamGm;
    GlobalTensor<W_T> weightsGm;
    GlobalTensor<SCALE_T> qScaleGm;
    GlobalTensor<SCALE_T> kScaleGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> blockTableGm;
    // =================================常量区=================================
    // P8 event ID rearrangement: Vec1 gets IDs 0-3, TopK gets IDs 4-7 (fully disjoint
    // for ITK interleaving). Uses shared-cycle (binary semaphore ping-pong) technique:
    // one ID serves both directions (V Set -> MTE2 Wait -> MTE2 Set -> V Wait -> ...).
    // Each Set is consumed by exactly one Wait on the other pipe.
    // Vec1 events (IDs 0-3):
    static constexpr uint32_t VEC1_SCALE_CYCLE = EVENT_ID0;     // ID 0: V<->MTE2 cycle for kScale+qScale (merged)
    static constexpr uint32_t VEC1_KSCALE_SCALAR = EVENT_ID1;   // ID 1: S->MTE2 for blockTable GM reads
    static constexpr uint32_t VEC1_OUT_CYCLE = EVENT_ID2;       // ID 2+pp: V<->MTE3 shared cycle for out pingpong
    // TopK events (IDs 4-7):
    static constexpr uint32_t TOPK_MRG_CYCLE = EVENT_ID4;       // ID 4: V<->MTE2 cycle for mrgValue (free/ready)
    static constexpr uint32_t TOPK_FILL_GUARD = EVENT_ID5;      // ID 5: V->MTE2 one-shot fill guard (zero/align fill)
    static constexpr uint32_t TOPK_IDX_CYCLE = EVENT_ID6;       // ID 6: V<->MTE3 shared cycle for indicesOut
    static constexpr uint32_t TOPK_MTE3_MTE2 = EVENT_ID7;       // ID 7: MTE3->MTE2 ordering (ProcessTopK entry)

private:
    __aicore__ inline void GetKeyScale(const QLICommon::RunInfo &runInfo, LocalTensor<SCALE_T> &kScaleUB,
                                       int64_t batchId, int64_t startS2, int64_t getLen);
    // ============================LD(s2 split two-phase TopK)=============================
    __aicore__ inline uint32_t GetRowBlockNum(const QLICommon::RunInfo &info);
    __aicore__ inline void SegmentTopK(const QLICommon::RunInfo &info, uint32_t rowIdx, uint32_t s1RowLocal,
                                       uint32_t vecOffset, int32_t validGlobalLen, int64_t segStartCol,
                                       int64_t segEndCol, bool isSegStart);
    __aicore__ inline void WriteLDList(uint32_t slot, uint32_t s1RowLocal, const LocalTensor<SCORE_T> &valLocal,
                                       const LocalTensor<uint32_t> &idxLocal);
    __aicore__ inline void WriteLDParam(uint32_t slot, const QLICommon::RunInfo &info, int64_t segStartCol,
                                        int64_t segEndCol);
    __aicore__ inline int64_t LDValElemOffset(uint32_t aic, uint32_t aiv, uint32_t slot, uint32_t s1RowLocal);
    __aicore__ inline int64_t LDIdxElemOffset(uint32_t aic, uint32_t aiv, uint32_t slot, uint32_t s1RowLocal);
    __aicore__ inline int64_t LDParamElemOffset(uint32_t aic, uint32_t aiv, uint32_t slot);
    QLICommon::RunInfo ldRowInfo_{};
    bool ldRowValid_ = false;
    QLICommon::RunInfo ldRowInfoArr_[4]{};  // v2LD: all LD tasks in this core's run
    uint32_t ldRowCount_ = 0;
    // ================================Local Buffer区====================================

    // tmp buff for vector
    TBuf<TPosition::VECCALC> resMm1Buf_;
    LocalTensor<QK_T> resMm1UB_;
    // tmp buff for weight
    TBuf<TPosition::VECCALC> weightBuf_;
    LocalTensor<W_T> weightUB_; 
    // tmp buff for weight cast float
    TBuf<TPosition::VECCALC> weightFloatBuf_;
    LocalTensor<float> weightFloatUB_;  
    // tmp buff for kScale
    TBuf<TPosition::VECCALC> kScaleBuf_;
    LocalTensor<SCALE_T> kScaleUB_;
    // tmp buff for qScale
    TBuf<TPosition::VECCALC> qScaleBuf_;
    LocalTensor<SCALE_T> qScaleUB_;

    // tmp buff for out
    TBuf<TPosition::VECCALC> outBuf_;
    LocalTensor<SCORE_T> vec1OutUB_;
    // tmp buff for LD

    // tmp buff for topk
    TBuf<TPosition::VECCALC> mrgValueBuf_;
    LocalTensor<SCORE_T> mrgValueLocal_;

    TBuf<TPosition::VECCALC> indicesOutBuf_;
    LocalTensor<uint32_t> indicesOutLocal_;

    TBuf<TPosition::VECCALC> scoreOutBuf_;
    LocalTensor<SCORE_T> scoreOutLocal_;

    TBuf<TPosition::VECCALC> topkSharedTmpBuf_;
    LocalTensor<uint32_t> topkSharedTmpLocal_;

    LocalTensor<int32_t> outInvalidLocal_;

    int32_t blockId_ = -1;
    // para for vector
    int32_t groupInner_ = 0;
    int32_t globalTopkNum_ = 0;
    int64_t blockS2StartIdx_ = 0;
    int32_t gSize_ = 0;
    int32_t kSeqSize_ = 0;
    int32_t kHeadNum_ = 0;
    int32_t qHeadNum_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;
    int32_t kCacheBlockSize_ = 0;
    int32_t maxBlockNumPerBatch_ = 0;
    uint32_t topkCount_ = 0;
    uint32_t topkCountAlign256_ = 0; // topkCount对齐到256(直方图需要)，支持topk泛化
    uint32_t trunkLen_ = 0;
    struct QLICommon::ConstInfo constInfo_;
    topk::LITopk<SCORE_T> topkOp_;
};

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(resMm1Buf_, 2 * CeilDiv(constInfo_.mBaseSize, 2) * s2BaseSize_ * sizeof(QK_T));
    resMm1UB_ = resMm1Buf_.Get<QK_T>();
    pipe->InitBuffer(weightBuf_, 2 * CeilDiv(s1BaseSize_, 2) * UB_BANK_DEPTH_STRIDE);
    weightUB_ = weightBuf_.Get<W_T>();
    pipe->InitBuffer(weightFloatBuf_, 2 * CeilDiv(s1BaseSize_, 2) * UB_BANK_DEPTH_STRIDE);
    weightFloatUB_ = weightFloatBuf_.Get<float>();
    pipe->InitBuffer(kScaleBuf_, 2 * s2BaseSize_ * 16 * sizeof(SCALE_T));
    kScaleUB_ = kScaleBuf_.Get<SCALE_T>();
    pipe->InitBuffer(qScaleBuf_, 2 * CeilDiv(s1BaseSize_, 2) * UB_BANK_DEPTH_STRIDE);
    qScaleUB_ = qScaleBuf_.Get<SCALE_T>();

    pipe->InitBuffer(outBuf_, 2 * CeilDiv(s1BaseSize_, 2) * s2BaseSize_ * sizeof(SCORE_T));
    vec1OutUB_ = outBuf_.Get<SCORE_T>();

    // Topk
    pipe->InitBuffer(mrgValueBuf_, (topkCountAlign256_ + trunkLen_) * sizeof(SCORE_T));
    mrgValueLocal_ = mrgValueBuf_.Get<SCORE_T>();
    outInvalidLocal_ = mrgValueBuf_.Get<int32_t>();

    pipe->InitBuffer(indicesOutBuf_, (topkCountAlign256_ + 64) * sizeof(uint32_t));         // 大小：(topkCountAlign256_ + 64) * 4  64:duplicate刷-1需要额外空间
    indicesOutLocal_ = indicesOutBuf_.Get<uint32_t>();

    pipe->InitBuffer(scoreOutBuf_, topkCountAlign256_ * sizeof(SCORE_T));
    scoreOutLocal_ = scoreOutBuf_.Get<SCORE_T>();

    uint64_t topkSharedTmpSize = topkOp_.GetSharedTmpBufferSize();
    pipe->InitBuffer(topkSharedTmpBuf_, topkSharedTmpSize);
    topkSharedTmpLocal_ = topkSharedTmpBuf_.Get<uint32_t>();
    topkOp_.InitBuffers(topkSharedTmpLocal_, indicesOutLocal_);

    //刷-1
    Duplicate(kScaleUB_, static_cast<SCALE_T>(0), 2 * s2BaseSize_ * 16);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitParams(const struct QLICommon::ConstInfo &constInfo,
                                                   const QLITilingData *__restrict tilingData)
{
    this->constInfo_ = constInfo;
    blockS2StartIdx_ = 0;
    gSize_ = constInfo.gSize;
    kSeqSize_ = constInfo.kSeqSize;
    // define N2 para
    kHeadNum_ = constInfo.kHeadNum;
    qHeadNum_ = constInfo.qHeadNum;
    // define MMBase para
    s1BaseSize_ = constInfo.s1BaseSize;  // 4
    s2BaseSize_ = constInfo.s2BaseSize;  // 128
    kCacheBlockSize_ = constInfo.kCacheBlockSize;
    maxBlockNumPerBatch_ = constInfo.maxBlockNumPerBatch;
    blockId_ = GetBlockIdx();
    topkCount_ = constInfo.sparseCount;
    topkCountAlign256_ = QLICommon::Align(constInfo.sparseCount, (uint64_t)256); // topkCount对齐到256
    if constexpr (std::is_same_v<SCORE_T, uint32_t>) {
        trunkLen_ = TRUNK_LEN_6K;
    } else {
        // C4: 32K trunk halves the per-row trunk count (each trunk pays fixed histogram
        // init/kth/gather passes); take it only if the full AIV UB account below
        // (mirror of the InitBuffer sizes, keep in sync) still fits in 256KB,
        // otherwise fall back to the 16K trunk (e.g. gSize=64 && tSize>64 shapes)
        uint64_t ubBytes = 2ULL * CeilDiv(constInfo.mBaseSize, 2) * s2BaseSize_ * sizeof(QK_T)
            + 3ULL * (2ULL * CeilDiv(s1BaseSize_, 2) * UB_BANK_DEPTH_STRIDE)
            + 2ULL * s2BaseSize_ * 16 * sizeof(SCALE_T)
            + 2ULL * CeilDiv(s1BaseSize_, 2) * s2BaseSize_ * sizeof(SCORE_T)
            + (topkCountAlign256_ + 64) * sizeof(uint32_t)
            + topkCountAlign256_ * sizeof(SCORE_T)
            + (2ULL * topkCountAlign256_ + 3 * 256 + 64) * sizeof(uint32_t)
            + 2ULL * (topkCountAlign256_ + TRUNK_LEN_32K) * sizeof(SCORE_T)
            - topkCountAlign256_ * sizeof(uint32_t);
        trunkLen_ = (ubBytes <= AIV_UB_LIMIT) ? TRUNK_LEN_32K : TRUNK_LEN_16K;
    }
    topkOp_.Init(topkCount_, trunkLen_);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitVecInputTensor(GlobalTensor<W_T> weightsGm, GlobalTensor<SCALE_T> qScaleGm,
                                                           GlobalTensor<SCALE_T> kScaleGm,
                                                           GlobalTensor<int32_t> indiceOutGm,
                                                           GlobalTensor<int32_t> blockTableGm)
{
    this->weightsGm = weightsGm;
    this->qScaleGm = qScaleGm;
    this->kScaleGm = kScaleGm;
    this->indiceOutGm = indiceOutGm;
    this->blockTableGm = blockTableGm;
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitVecWorkspaceTensor(GlobalTensor<SCORE_T> scoreGm)
{
    this->scoreGm = scoreGm; // resucesum*k
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitLDTensor(GlobalTensor<SCORE_T> ldValGm, GlobalTensor<uint32_t> ldIdxGm,
                                                     GlobalTensor<int64_t> ldParamGm)
{
    this->ldValGm = ldValGm;
    this->ldIdxGm = ldIdxGm;
    this->ldParamGm = ldParamGm;
}

// LD candidate list layout: [aic][aiv][slot][s1Row]{topkCount val | topkCount idx}, keep in sync with host
template <typename QLIT>
__aicore__ inline int64_t QLIVector<QLIT>::LDValElemOffset(uint32_t aic, uint32_t aiv, uint32_t slot,
                                                           uint32_t s1RowLocal)
{
    constexpr uint32_t ldAiv = QLICommon::ConstInfo::LD_AIV_NUM;
    constexpr uint32_t ldSlot = QLICommon::ConstInfo::LD_SLOT_NUM;
    constexpr uint32_t ldRows = QLICommon::ConstInfo::LD_S1_ROWS_PER_AIV;
    int64_t rowBytes = (int64_t)topkCount_ * (sizeof(SCORE_T) + sizeof(uint32_t));
    int64_t base = (((int64_t)aic * ldAiv + aiv) * ldSlot + slot) * ldRows * rowBytes + (int64_t)s1RowLocal * rowBytes;
    return base / (int64_t)sizeof(SCORE_T);
}

template <typename QLIT>
__aicore__ inline int64_t QLIVector<QLIT>::LDIdxElemOffset(uint32_t aic, uint32_t aiv, uint32_t slot,
                                                           uint32_t s1RowLocal)
{
    constexpr uint32_t ldAiv = QLICommon::ConstInfo::LD_AIV_NUM;
    constexpr uint32_t ldSlot = QLICommon::ConstInfo::LD_SLOT_NUM;
    constexpr uint32_t ldRows = QLICommon::ConstInfo::LD_S1_ROWS_PER_AIV;
    int64_t rowBytes = (int64_t)topkCount_ * (sizeof(SCORE_T) + sizeof(uint32_t));
    int64_t base = (((int64_t)aic * ldAiv + aiv) * ldSlot + slot) * ldRows * rowBytes + (int64_t)s1RowLocal * rowBytes;
    return (base + (int64_t)topkCount_ * sizeof(SCORE_T)) / (int64_t)sizeof(uint32_t);
}

template <typename QLIT>
__aicore__ inline int64_t QLIVector<QLIT>::LDParamElemOffset(uint32_t aic, uint32_t aiv, uint32_t slot)
{
    constexpr uint32_t ldAiv = QLICommon::ConstInfo::LD_AIV_NUM;
    constexpr uint32_t ldSlot = QLICommon::ConstInfo::LD_SLOT_NUM;
    return (((int64_t)aic * ldAiv + aiv) * ldSlot + slot) * QLICommon::ConstInfo::LD_PARAM_NUM;
}

template <typename QLIT>
__aicore__ inline uint32_t QLIVector<QLIT>::GetRowBlockNum(const QLICommon::RunInfo &info)
{
    if (!constInfo_.attenMaskFlag) {
        return (uint32_t)((info.actS2Size + s2BaseSize_ - 1) / s2BaseSize_);
    }
    uint32_t s1Offset = s1BaseSize_ * info.gS1Idx;
    int32_t validS2LenBase = (int32_t)info.actS2SizeOrig - (int32_t)info.actS1Size;
    int32_t validS2Len = (int32_t)s1Offset + validS2LenBase + (int32_t)s1BaseSize_;
    validS2Len = validS2Len > (int32_t)info.actS2SizeOrig ? (int32_t)info.actS2SizeOrig : validS2Len;
    validS2Len = validS2Len < 1 ? 1 : validS2Len;
    return (uint32_t)((validS2Len + s2BaseSize_ - 1) / s2BaseSize_);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::AllocEventID()
{
    // Vec1 events (IDs 0-3): arm initial states
    SetFlag<HardEvent::V_MTE2>(VEC1_SCALE_CYCLE);              // ID 0: scale buffer initially free
    SetFlag<HardEvent::MTE3_V>(VEC1_OUT_CYCLE + 0);            // ID 2: out buf 0 initially done
    SetFlag<HardEvent::MTE3_V>(VEC1_OUT_CYCLE + 1);            // ID 3: out buf 1 initially done
    // TopK events (IDs 4-6): arm initial states
    SetFlag<HardEvent::V_MTE2>(TOPK_MRG_CYCLE);                // ID 4: mrgValue initially free
    SetFlag<HardEvent::MTE3_V>(TOPK_IDX_CYCLE);                // ID 6: indicesOut initially done
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(VEC1_SCALE_CYCLE);             // ID 0: consume leftover scale free
    WaitFlag<HardEvent::MTE3_V>(VEC1_OUT_CYCLE + 0);           // ID 2: consume leftover out done
    WaitFlag<HardEvent::MTE3_V>(VEC1_OUT_CYCLE + 1);           // ID 3: consume leftover out done
    WaitFlag<HardEvent::V_MTE2>(TOPK_MRG_CYCLE);               // ID 4: consume leftover mrg free
    WaitFlag<HardEvent::MTE3_V>(TOPK_IDX_CYCLE);               // ID 6: consume leftover idx done
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::CleanInvalidOutput(int64_t invalidS1Offset)
{
    // init -1 and copy to output
    Duplicate(outInvalidLocal_, constInfo_.INVALID_IDX, constInfo_.sparseCount);

    // BUG-4 fix: the row-cycle invariant is "ID6 armed between rows" (every row ends
    // with Set<MTE3_V>). The V_MTE3 pair below fired on an already-armed flag: Set
    // no-oped and Wait consumed the row-cycle arm, leaving ID6 disarmed -> the next
    // row's Wait<MTE3_V> blocked forever (prefill multi-task hang, >=9 tasks).
    // Normalize first (deterministically drains the arm), then the pair, then re-arm
    // after the copy to restore the invariant.
    SetFlag<HardEvent::MTE3_V>(TOPK_IDX_CYCLE);
    WaitFlag<HardEvent::MTE3_V>(TOPK_IDX_CYCLE);
    SetFlag<HardEvent::V_MTE3>(TOPK_IDX_CYCLE);
    WaitFlag<HardEvent::V_MTE3>(TOPK_IDX_CYCLE);

    AscendC::DataCopyParams dataCopyOutParams;
    dataCopyOutParams.blockCount = 1;
    dataCopyOutParams.blockLen = constInfo_.sparseCount * sizeof(int32_t);
    dataCopyOutParams.srcStride = 0;
    dataCopyOutParams.dstStride = 0;
    AscendC::DataCopyPad(indiceOutGm[invalidS1Offset], outInvalidLocal_, dataCopyOutParams);
    SetFlag<HardEvent::MTE3_V>(TOPK_IDX_CYCLE);  // re-arm row-cycle token (BUG-4 invariant)
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::GetKeyScale(const QLICommon::RunInfo &runInfo, LocalTensor<SCALE_T> &kScaleUB,
                                                    int64_t batchId, int64_t startS2, int64_t getLen)
{
    // startS2一定能整除kCacheBlockSize_
    AscendC::DataCopyPadExtParams<SCALE_T> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams copyInParams;
    if constexpr (PAGE_ATTENTION) {
        int32_t startBlockTableIdx = startS2 / kCacheBlockSize_;
        int32_t startBlockTableOffset = startS2 % kCacheBlockSize_;
        int32_t blockTableBatchOffset = batchId * maxBlockNumPerBatch_;
        copyInParams.blockCount = 1;
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        int32_t resUbBaseOffset = 0;
        if (startBlockTableOffset > 0) {
            int32_t firstPartLen =
                kCacheBlockSize_ - startBlockTableOffset > getLen ? getLen : kCacheBlockSize_ - startBlockTableOffset;
            copyInParams.blockLen = firstPartLen * sizeof(SCALE_T);
            int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx);
            SetFlag<HardEvent::S_MTE2>(VEC1_KSCALE_SCALAR);
            WaitFlag<HardEvent::S_MTE2>(VEC1_KSCALE_SCALAR);
            AscendC::DataCopyPad(kScaleUB[16 * (runInfo.kScaleLoop % 2) * s2BaseSize_], kScaleGm[blockId *
                                constInfo_.keyDequantScaleStride0 + startBlockTableOffset], copyInParams, padParams);
            startBlockTableIdx++;
            getLen = getLen - firstPartLen;
            resUbBaseOffset = firstPartLen;
        }
        int32_t getLoopNum = CeilDiv(getLen, kCacheBlockSize_);
        copyInParams.blockLen = kCacheBlockSize_ * sizeof(SCALE_T);
        for (int32_t i = 0; i < getLoopNum; i++) {
            if (i == getLoopNum - 1) {
                copyInParams.blockLen = (getLen - i * kCacheBlockSize_) * sizeof(SCALE_T);
            }
            int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx + i);
            SetFlag<HardEvent::S_MTE2>(VEC1_KSCALE_SCALAR);
            WaitFlag<HardEvent::S_MTE2>(VEC1_KSCALE_SCALAR);
            AscendC::DataCopyPad(kScaleUB[16 * (runInfo.kScaleLoop % 2) * s2BaseSize_ +
                                               resUbBaseOffset + i * kCacheBlockSize_],
                                 kScaleGm[blockId * constInfo_.keyDequantScaleStride0],
                                 copyInParams, padParams);
        }
    } else {
        copyInParams.blockCount = 1;
        copyInParams.blockLen = getLen * sizeof(SCALE_T);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        AscendC::DataCopyPad(kScaleUB[16 * (runInfo.kScaleLoop % 2) * s2BaseSize_],
                                            kScaleGm[runInfo.tensorKeyScaleOffset],
                                            copyInParams, padParams);
    }
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ProcessVec1(const QLICommon::RunInfo &info)
{
    auto pingpong = (info.loop % 2);
    auto qScalepingpong = (info.qScaleLoop % 2);
    auto kScalepingpong = (info.kScaleLoop % 2);
    auto s1BaseSizePerAIV = CeilDiv(s1BaseSize_, 2);
    int64_t curS1Idx = info.gS1Idx * s1BaseSize_;	 
    int64_t curS2Idx = info.s2Idx * s2BaseSize_;	 
    int64_t curS1ProcNum = curS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;	 
    int64_t curAivS1Idx = curS1Idx + (blockId_ % 2) * CeilDiv(curS1ProcNum, 2);
    int64_t curAivS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curS1ProcNum, 2) : curS1ProcNum / 2; 
    if (curAivS1ProcNum == 0) {
        CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_CV_EVENT + pingpong);  // V核等C核计算完mm1，mm1Res已搬运到UB
        CrossCoreSetFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_VC_EVENT + pingpong);   // V核处理完，通知C核可以把mm1Res搬运到UB
        return;
    }
    
    // P8: merged scale cycle — qScale and kScale copies share one event (ID 0).
    // At the first block of a row, BOTH copies happen in one critical section.
    // At 16-block boundaries, only kScale copy happens.
    const bool needQScale = info.isFirstS2InnerLoop;
    const bool needKScale = ((info.s2Idx - info.s2Start) % 16 == 0);
    if (needQScale) {
        WaitFlag<HardEvent::V_MTE2>(VEC1_SCALE_CYCLE);
        //weightsGm --> weightUB_
        int64_t weightGmOffset = info.tensorWeightsOffset + curAivS1Idx * kHeadNum_ * gSize_;
        DataCopyPadExtParams<W_T> padWeightsParams{false, 0, 0, 0};
        DataCopyExtParams wDataCopyExtParams;
        wDataCopyExtParams.blockCount = curAivS1ProcNum;
        wDataCopyExtParams.blockLen = gSize_ * sizeof(W_T);
        wDataCopyExtParams.srcStride = 0;
        wDataCopyExtParams.dstStride = (UB_BANK_DEPTH_STRIDE - wDataCopyExtParams.blockLen) / 32;
        DataCopyPad(weightUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(W_T))],
                    weightsGm[weightGmOffset], wDataCopyExtParams, padWeightsParams);

        //qScaleGm  -->  qScaleUB_
        DataCopyPadExtParams<SCALE_T> padQScaleParams{false, 0, 0, 0};
        DataCopyExtParams qDataCopyExtParams;
        qDataCopyExtParams.blockCount = curAivS1ProcNum;
        qDataCopyExtParams.blockLen = gSize_ * sizeof(SCALE_T);
        qDataCopyExtParams.srcStride = 0;
        qDataCopyExtParams.dstStride = (UB_BANK_DEPTH_STRIDE - qDataCopyExtParams.blockLen) / 32;
        DataCopyPad(qScaleUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(SCALE_T))],
                    qScaleGm[weightGmOffset], qDataCopyExtParams, padQScaleParams);
    }

    if (needKScale) {
        if (!needQScale) {
            // standalone kScale copy (16-block boundary, not first block): enter the cycle
            WaitFlag<HardEvent::V_MTE2>(VEC1_SCALE_CYCLE);
        }
        // if needQScale, we're already inside the critical section (same cycle)
        uint32_t getLen = 16 * s2BaseSize_ > (info.validS2Len - info.s2Idx * s2BaseSize_)
                                           ? info.validS2Len - info.s2Idx * s2BaseSize_
                                           : 16 * s2BaseSize_;
        //kScaleGm  -->  kScaleUB_
        GetKeyScale(info, kScaleUB_, info.bIdx, curS2Idx, getLen);
    }
    if (needQScale || needKScale) {
        SetFlag<HardEvent::MTE2_V>(VEC1_SCALE_CYCLE);
        WaitFlag<HardEvent::MTE2_V>(VEC1_SCALE_CYCLE);
    }
    WaitFlag<HardEvent::MTE3_V>(VEC1_OUT_CYCLE + pingpong);

    //CV同步
    CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_CV_EVENT + info.loop % 2);   //V核等C核计算完mm1，mm1Res已搬运到UB
    LocalTensor<SCORE_T> outBase;
    if (std::is_same_v<SCORE_T, uint16_t>) {
        outBase = vec1OutUB_[pingpong * (UB_BANK_STRIDE / sizeof(uint16_t))];
    } else {
        outBase = vec1OutUB_[pingpong * CeilDiv(s1BaseSize_, 2) * s2BaseSize_];
    }
    auto weightBase = weightUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(W_T))];
    auto weightFloatBase = weightFloatUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(float))];
    auto qScaleBase = qScaleUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(SCALE_T))];
    auto kScaleBase = kScaleUB_[kScalepingpong * 16 * s2BaseSize_ + ((info.s2Idx - info.s2Start) % 16) * s2BaseSize_];
    auto qkBase = resMm1UB_[pingpong * (UB_BANK_STRIDE / sizeof(float))];
    auto qkVLstride = (UB_BANK_DEPTH_STRIDE / sizeof(float)) / 2 * constInfo_.mBaseSize;

    // batch pairs: BatchMulWeightAndReduceSum supports batch 1/2 per call; s1Base=8 (mBase 256 prefill)
    // gives curAivS1ProcNum up to 4 -> iterate row pairs with row-relative offsets (out/qk/weight/qScale
    // share the 512B row stride; weightFloat scratch pairs live 256B apart per call, disjoint per slot)
    const uint32_t outRowStride = UB_BANK_DEPTH_STRIDE / sizeof(SCORE_T);
    const uint32_t qkRowStride = (uint32_t)(gSize_ * UB_BANK_DEPTH_STRIDE / sizeof(float));
    const uint32_t wRowStride = UB_BANK_DEPTH_STRIDE / sizeof(W_T);
    const uint32_t qsRowStride = UB_BANK_DEPTH_STRIDE / sizeof(SCALE_T);
    const uint32_t wfRowStride = UB_BANK_STRIDE / sizeof(float);
    for (uint32_t rowBase = 0; rowBase < static_cast<uint32_t>(curAivS1ProcNum); rowBase += 2U) {
        const uint32_t batch = Min(2U, static_cast<uint32_t>(curAivS1ProcNum) - rowBase);
        vector1::BatchMulWeightAndReduceSum(outBase[rowBase * outRowStride], outRowStride,
                                            qkBase[rowBase * qkRowStride], qkVLstride, qkRowStride,
                                            weightBase[rowBase * wRowStride], wRowStride,
                                            weightFloatBase[rowBase * wfRowStride],
                                            kScaleBase, (uint32_t)0,
                                            qScaleBase[rowBase * qsRowStride], qsRowStride,
                                            gSize_, static_cast<int>(batch));
    }
    if (needQScale || needKScale) {
        SetFlag<HardEvent::V_MTE2>(VEC1_SCALE_CYCLE);
    }
    SetFlag<HardEvent::V_MTE3>(VEC1_OUT_CYCLE + pingpong);
    WaitFlag<HardEvent::V_MTE3>(VEC1_OUT_CYCLE + pingpong);
    //outUB_ --->  scoreGm
    int64_t vec1OutGmOffset = blockId_ % 2 == 0 ? curS2Idx :
                            s1BaseSizePerAIV * QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) + curS2Idx;
    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = curAivS1ProcNum;
    copyOutParams.blockLen = s2BaseSize_ * sizeof(SCORE_T);
    copyOutParams.srcStride = (UB_BANK_DEPTH_STRIDE - copyOutParams.blockLen) / 32;
    copyOutParams.dstStride = (QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) - s2BaseSize_) * sizeof(SCORE_T);
    DataCopyPad(scoreGm[vec1OutGmOffset], outBase, copyOutParams);
    SetFlag<HardEvent::MTE3_V>(VEC1_OUT_CYCLE + pingpong);
    CrossCoreSetFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_VC_EVENT + pingpong);   //V核处理完，通知C核可以把mm1Res搬运到UB
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ProcessTopK(const QLICommon::RunInfo &info)
{
    SetFlag<HardEvent::MTE3_MTE2>(TOPK_MTE3_MTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(TOPK_MTE3_MTE2);

    int64_t curS1Idx = info.gS1Idx * s1BaseSize_;
    int64_t curS2Idx = info.s2Idx * s2BaseSize_;	 
    int64_t curS1ProcNum = curS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;	 
    int64_t curAivS1Idx = curS1Idx + (blockId_ % 2) * CeilDiv(curS1ProcNum, 2);
    int64_t curAivS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curS1ProcNum, 2) : curS1ProcNum / 2; 

    AscendC::DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    copyInParams.rsv = 0;

    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = topkCount_ * sizeof(uint32_t); // bytes
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    int32_t cuRealAcSeq = info.actS2Size;
    if (constInfo_.attenMaskFlag) {
        cuRealAcSeq = info.actS2SizeOrig - info.actS1Size + curAivS1Idx + 1;
    } 

    int32_t validS2Len = cuRealAcSeq;
    for (uint32_t i = 0; i < curAivS1ProcNum; i++) {
        uint32_t rowIdx = blockId_ % 2 * CeilDiv(curS1ProcNum, 2) + i;
        uint32_t vecOffset = blockId_ % 2 * CeilDiv(s1BaseSize_, 2) + i;
        
        SCORE_T zero = 0;
        int32_t neg = -1;
        if (constInfo_.attenMaskFlag) {
            validS2Len = (int32_t)i + cuRealAcSeq;
        }
        if (constInfo_.isLDOpen) {
            // s2 split rows: this core owns only a segment of the row -> segment-local topK + LD list writeout
            uint32_t rowBlockNum = GetRowBlockNum(info);
            bool isSegTail = info.s2Start > 0;  // this core owns the tail segment of the row
            bool isSegStart = (info.s2Start == 0) && (info.s2Idx + 1 < rowBlockNum);  // owns row head, not whole row
            if (isSegStart || isSegTail) {
                int64_t segStartCol = isSegTail ? (int64_t)info.s2Start * s2BaseSize_ : 0;
                int64_t segEndCol = (isSegStart ? (int64_t)(info.s2Idx + 1) : (int64_t)rowBlockNum) * s2BaseSize_;
                SegmentTopK(info, rowIdx, i, vecOffset, validS2Len, segStartCol, segEndCol, isSegStart);
                if (isSegStart) {
                    ldRowInfo_ = info;  // saved for ProcessLD (row merge owner)
                    ldRowValid_ = true;
                    if (ldRowCount_ == 0 ||
                        ldRowInfoArr_[ldRowCount_ - 1].gS1Idx != info.gS1Idx ||
                        ldRowInfoArr_[ldRowCount_ - 1].bN2Idx != info.bN2Idx) {
                        if (ldRowCount_ < 4) {
                            ldRowInfoArr_[ldRowCount_++] = info;
                        }
                    }
                }
                continue;
            }
        }
        if (validS2Len <= 0) {
            WaitFlag<HardEvent::MTE3_V>(TOPK_IDX_CYCLE);
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>(), neg, topkCount_);
            SetFlag<HardEvent::V_MTE3>(TOPK_IDX_CYCLE);
            WaitFlag<HardEvent::V_MTE3>(TOPK_IDX_CYCLE);
            AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_], indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_IDX_CYCLE);
            continue;
        }

        WaitFlag<HardEvent::V_MTE2>(TOPK_MRG_CYCLE);
        WaitFlag<HardEvent::MTE3_V>(TOPK_IDX_CYCLE);

        AscendC::DataCopyPadExtParams<SCORE_T> padParams{true, 0, 0, 0};
        if (validS2Len >= topkCount_) {
            uint32_t s2LoopNum = (validS2Len + trunkLen_ - 1) / trunkLen_;
            if (s2LoopNum == 1) {
                uint32_t validS2LenAlign = QLICommon::Align(validS2Len, (int32_t)256);
                Duplicate(mrgValueLocal_[validS2Len / 256 * 256], zero, validS2LenAlign - validS2Len / 256 * 256);
                SetFlag<HardEvent::V_MTE2>(TOPK_FILL_GUARD);
                WaitFlag<HardEvent::V_MTE2>(TOPK_FILL_GUARD);
                copyInParams.blockLen = validS2Len * sizeof(SCORE_T); // byte
                AscendC::DataCopyPadExtParams<SCORE_T> padParams{true, 0, 0, 0};
                AscendC::DataCopyPad(mrgValueLocal_, scoreGm[vecOffset * QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_)], copyInParams, padParams);
                SetFlag<HardEvent::MTE2_V>(TOPK_MRG_CYCLE);
                WaitFlag<HardEvent::MTE2_V>(TOPK_MRG_CYCLE);
                topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, validS2LenAlign, 0, 1);
            } else {
                for (uint32_t loopIdx = 0; loopIdx < s2LoopNum; loopIdx++) {
                    if (loopIdx == 0) {
                        copyInParams.blockLen = trunkLen_ * sizeof(SCORE_T); // byte
                        AscendC::DataCopyPad(mrgValueLocal_, scoreGm[vecOffset * QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_)], copyInParams, padParams);
                        SetFlag<HardEvent::MTE2_V>(TOPK_MRG_CYCLE);
                        WaitFlag<HardEvent::MTE2_V>(TOPK_MRG_CYCLE);
                        topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, trunkLen_, loopIdx, s2LoopNum);
                        continue;
                    }
                    // P8: mrgValue free cycle on ID 4; fill guard on ID 5 (separate to
                    // avoid double-Set deadlock when V runs ahead of MTE2)
                    SetFlag<HardEvent::V_MTE2>(TOPK_MRG_CYCLE);
                    WaitFlag<HardEvent::V_MTE2>(TOPK_MRG_CYCLE);
                    uint32_t validTrunkLen = (loopIdx * trunkLen_ + trunkLen_) > validS2Len ? validS2Len % trunkLen_ : trunkLen_;
                    uint32_t offset = vecOffset * QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) + loopIdx * trunkLen_;
                    AscendC::DataCopy(mrgValueLocal_, scoreOutLocal_, topkCountAlign256_);
                    // P8: combine zero-fill and align-fill into ONE Set/Wait pair on ID 5
                    // (original code had two separate EVENT3 pairs — latent double-Set
                    // deadlock for non-256-aligned topk, now fixed)
                    const bool needZeroFill = (topkCountAlign256_ != topkCount_);
                    const bool needAlignFill = ((topkCountAlign256_ + validTrunkLen) % 256 != 0);
                    if (needZeroFill) {
                        uint64_t mask[1];
                        mask[0] = ~0;
                        mask[0] = mask[0] << (topkCount_ % 64);
                        PipeBarrier<PIPE_V>();
                        // 把topkCount_对齐到64刷0，此处由于duplicate的限制mask[0]刷64个数
                        Duplicate(mrgValueLocal_[topkCount_ / 64 * 64], zero, mask, 1, 1, 0);
                        PipeBarrier<PIPE_V>();
                        // 把topk剩余对齐到256的部分刷0
                        Duplicate(mrgValueLocal_[topkCount_ / 64 * 64 + 64], zero, topkCountAlign256_ - (topkCount_ / 64 * 64 + 64));
                    }
                    copyInParams.blockLen = validTrunkLen * sizeof(SCORE_T); // byte
                    // TOPK 直方图一次必须计算256，输入处理数据需要和256对齐
                    if (needAlignFill) {
                        Duplicate(mrgValueLocal_[topkCountAlign256_ + validTrunkLen / 256 * 256], zero, QLICommon::Align(validTrunkLen, (uint32_t)256) - validTrunkLen / 256 * 256);
                    }
                    if (needZeroFill || needAlignFill) {
                        SetFlag<HardEvent::V_MTE2>(TOPK_FILL_GUARD);
                        WaitFlag<HardEvent::V_MTE2>(TOPK_FILL_GUARD);
                    }
                    AscendC::DataCopyPad(mrgValueLocal_[topkCountAlign256_], scoreGm[offset], copyInParams, padParams);
                    SetFlag<HardEvent::MTE2_V>(TOPK_MRG_CYCLE);
                    WaitFlag<HardEvent::MTE2_V>(TOPK_MRG_CYCLE);
                    topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, QLICommon::Align(topkCountAlign256_ + validTrunkLen, (uint32_t)256), loopIdx, s2LoopNum);
                }
            }
        } else {
            AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(), (int32_t)zero, validS2Len);
        }
        
        if (validS2Len < topkCount_) {
            uint64_t mask[1];
            mask[0] = ~0;
            mask[0] = mask[0] << (validS2Len % 8);
            PipeBarrier<PIPE_V>();
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8], neg, mask, 1, 1, 0);
        }
        
        if (validS2Len / 8 * 8 + 64 < topkCount_) {
            PipeBarrier<PIPE_V>();
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[validS2Len / 8 * 8 + 64], neg, topkCount_ - (validS2Len / 8 * 8 + 64));
        }

        SetFlag<HardEvent::V_MTE2>(TOPK_MRG_CYCLE);
        SetFlag<HardEvent::V_MTE3>(TOPK_IDX_CYCLE);
        WaitFlag<HardEvent::V_MTE3>(TOPK_IDX_CYCLE);
        AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_], indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
        SetFlag<HardEvent::MTE3_V>(TOPK_IDX_CYCLE);
    }
}

// ============================LD(s2 split two-phase TopK)=============================
// Segment-local topK over [segStartCol, segEndCol) of this s1 row, write candidate list {value,idx} + param.
// short segment (< topkCount): raw scores as candidates; long segment: trunked topk, indices offset by segStartCol.
template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::SegmentTopK(const QLICommon::RunInfo &info, uint32_t rowIdx,
                                                    uint32_t s1RowLocal, uint32_t vecOffset, int32_t validGlobalLen,
                                                    int64_t segStartCol, int64_t segEndCol, bool isSegStart)
{
    SCORE_T zero = 0;
    int32_t neg = -1;
    bool singleTrunkIdxInSharedTmp = false;  // single-trunk path leaves final indices in shared tmp
    int64_t rowBase = (int64_t)vecOffset * QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_);
    int64_t segValid = (validGlobalLen < segEndCol ? (int64_t)validGlobalLen : segEndCol) - segStartCol;
    if (segValid < 0) {
        segValid = 0;
    }
    uint32_t slot = isSegStart ? 1U : 0U;  // slot1: tail slot (LD core carry), slot0: head slot (tail core list)

    AscendC::DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    copyInParams.rsv = 0;
    AscendC::DataCopyPadExtParams<SCORE_T> padParams{true, 0, 0, 0};

    if (segValid == 0) {
        WaitFlag<HardEvent::V_MTE2>(TOPK_MRG_CYCLE);
        WaitFlag<HardEvent::MTE3_V>(TOPK_IDX_CYCLE);
        Duplicate(mrgValueLocal_, zero, topkCount_);
        PipeBarrier<PIPE_V>();
        // two-step -1 fill (masked + 64-aligned), plain Duplicate at arbitrary offset is unsafe
        uint64_t maskAll[1];
        maskAll[0] = ~0;
        Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[0], neg, maskAll, 1, 1, 0);
        PipeBarrier<PIPE_V>();
        Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[64], neg, topkCount_ - 64);
        SetFlag<HardEvent::V_MTE2>(TOPK_MRG_CYCLE);
    } else if (segValid < (int64_t)topkCount_) {
        // short segment: all elements survive; values = raw scores, indices = global col numbers
        uint32_t segValidAlign256 = QLICommon::Align((uint32_t)segValid, (uint32_t)256);
        WaitFlag<HardEvent::V_MTE2>(TOPK_MRG_CYCLE);
        WaitFlag<HardEvent::MTE3_V>(TOPK_IDX_CYCLE);
        // zero the FULL value tail [segValidAlign256, topkCountAlign256_): the LD list stores
        // topkCount_ values wholesale, and any stale UB left there pairs with the -1 index
        // padding and displaces real candidates in the merge (raw 0x0000 is the weakest
        // sortable key, same neutral fill the segValid==0 path uses).
        Duplicate(mrgValueLocal_[segValid / 256 * 256], zero, segValidAlign256 - segValid / 256 * 256);
        if (segValidAlign256 < topkCountAlign256_) {
            Duplicate(mrgValueLocal_[segValidAlign256], zero, topkCountAlign256_ - segValidAlign256);
        }
        SetFlag<HardEvent::V_MTE2>(TOPK_MRG_CYCLE);
        WaitFlag<HardEvent::V_MTE2>(TOPK_MRG_CYCLE);
        copyInParams.blockLen = (uint32_t)segValid * sizeof(SCORE_T);
        AscendC::DataCopyPad(mrgValueLocal_, scoreGm[rowBase + segStartCol], copyInParams, padParams);
        SetFlag<HardEvent::MTE2_V>(TOPK_MRG_CYCLE);
        WaitFlag<HardEvent::MTE2_V>(TOPK_MRG_CYCLE);
        // indices: 0..segValid-1 (+segStartCol), then -1 padding (two-step fill, same pattern as direct path)
        AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(), (int32_t)zero, (uint32_t)segValid);
        PipeBarrier<PIPE_V>();
        topkb32gather::IndicesAddOffset(indicesOutLocal_, (uint32_t)segStartCol, topkCount_);
        PipeBarrier<PIPE_V>();
        if ((uint32_t)segValid < topkCount_) {
            uint64_t mask[1];
            mask[0] = ~0;
            mask[0] = mask[0] << ((uint32_t)segValid % 8);
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[segValid / 8 * 8], neg, mask, 1, 1, 0);
            PipeBarrier<PIPE_V>();
        }
        if ((uint32_t)segValid / 8 * 8 + 64 < topkCount_) {
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[segValid / 8 * 8 + 64], neg,
                      topkCount_ - ((uint32_t)segValid / 8 * 8 + 64));
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_MTE2>(TOPK_MRG_CYCLE);
    } else {
        // long segment: trunked histogram topk over the segment stream, then add segStartCol to indices
        uint32_t segValidU = (uint32_t)segValid;
        uint32_t s2LoopNum = (segValidU + trunkLen_ - 1) / trunkLen_;
        if (s2LoopNum == 1) {
            uint32_t segValidAlign256 = QLICommon::Align(segValidU, (uint32_t)256);
            WaitFlag<HardEvent::V_MTE2>(TOPK_MRG_CYCLE);
            WaitFlag<HardEvent::MTE3_V>(TOPK_IDX_CYCLE);
            Duplicate(mrgValueLocal_[segValidU / 256 * 256], zero, segValidAlign256 - segValidU / 256 * 256);
            SetFlag<HardEvent::V_MTE2>(TOPK_MRG_CYCLE);
            WaitFlag<HardEvent::V_MTE2>(TOPK_MRG_CYCLE);
            copyInParams.blockLen = segValidU * sizeof(SCORE_T);
            AscendC::DataCopyPad(mrgValueLocal_, scoreGm[rowBase + segStartCol], copyInParams, padParams);
            SetFlag<HardEvent::MTE2_V>(TOPK_MRG_CYCLE);
            WaitFlag<HardEvent::MTE2_V>(TOPK_MRG_CYCLE);
            // use the (loopIdx=0, s2LoopNum=2) protocol so LiTopKVF<true> emits values into scoreOutLocal_
            // and positions into shared tmp (hisIndex ping slot); add segStartCol in place (no UB->UB copy)
            topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, segValidAlign256, 0, 2);
            PipeBarrier<PIPE_V>();
            topkb32gather::IndicesAddOffset(topkSharedTmpLocal_, (uint32_t)segStartCol, topkCount_);
            PipeBarrier<PIPE_V>();
            singleTrunkIdxInSharedTmp = true;
        } else {
            for (uint32_t loopIdx = 0; loopIdx < s2LoopNum; loopIdx++) {
                if (loopIdx == 0) {
                    WaitFlag<HardEvent::V_MTE2>(TOPK_MRG_CYCLE);
                    WaitFlag<HardEvent::MTE3_V>(TOPK_IDX_CYCLE);
                    copyInParams.blockLen = trunkLen_ * sizeof(SCORE_T);
                    AscendC::DataCopyPad(mrgValueLocal_, scoreGm[rowBase + segStartCol], copyInParams, padParams);
                    SetFlag<HardEvent::MTE2_V>(TOPK_MRG_CYCLE);
                    WaitFlag<HardEvent::MTE2_V>(TOPK_MRG_CYCLE);
                    topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, trunkLen_, loopIdx, s2LoopNum);
                    continue;
                }
                SetFlag<HardEvent::V_MTE2>(TOPK_MRG_CYCLE);
                WaitFlag<HardEvent::V_MTE2>(TOPK_MRG_CYCLE);
                uint32_t validTrunkLen = (loopIdx * trunkLen_ + trunkLen_) > segValidU ? segValidU % trunkLen_ : trunkLen_;
                int64_t offset = rowBase + segStartCol + (int64_t)loopIdx * trunkLen_;
                AscendC::DataCopy(mrgValueLocal_, scoreOutLocal_, topkCountAlign256_);
                // P8: combined fill guard on ID 5 (same fix as ProcessTopK)
                const bool needZeroFill = (topkCountAlign256_ != topkCount_);
                const bool needAlignFill = ((topkCountAlign256_ + validTrunkLen) % 256 != 0);
                if (needZeroFill) {
                    uint64_t mask[1];
                    mask[0] = ~0;
                    mask[0] = mask[0] << (topkCount_ % 64);
                    PipeBarrier<PIPE_V>();
                    Duplicate(mrgValueLocal_[topkCount_ / 64 * 64], zero, mask, 1, 1, 0);
                    PipeBarrier<PIPE_V>();
                    Duplicate(mrgValueLocal_[topkCount_ / 64 * 64 + 64], zero,
                              topkCountAlign256_ - (topkCount_ / 64 * 64 + 64));
                }
                copyInParams.blockLen = validTrunkLen * sizeof(SCORE_T);
                if (needAlignFill) {
                    Duplicate(mrgValueLocal_[topkCountAlign256_ + validTrunkLen / 256 * 256], zero,
                              QLICommon::Align(validTrunkLen, (uint32_t)256) - validTrunkLen / 256 * 256);
                }
                if (needZeroFill || needAlignFill) {
                    SetFlag<HardEvent::V_MTE2>(TOPK_FILL_GUARD);
                    WaitFlag<HardEvent::V_MTE2>(TOPK_FILL_GUARD);
                }
                AscendC::DataCopyPad(mrgValueLocal_[topkCountAlign256_], scoreGm[offset], copyInParams, padParams);
                SetFlag<HardEvent::MTE2_V>(TOPK_MRG_CYCLE);
                WaitFlag<HardEvent::MTE2_V>(TOPK_MRG_CYCLE);
                topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_,
                        QLICommon::Align(topkCountAlign256_ + validTrunkLen, (uint32_t)256), loopIdx, s2LoopNum);
            }
        }
        if (!singleTrunkIdxInSharedTmp) {
            PipeBarrier<PIPE_V>();
            topkb32gather::IndicesAddOffset(indicesOutLocal_, (uint32_t)segStartCol, topkCount_);
            PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_MTE2>(TOPK_MRG_CYCLE);
    }

    // write candidate list {value | idx} then param (param written last as completion marker)
    const LocalTensor<SCORE_T> &valLocal =
        (segValid >= (int64_t)topkCount_) ? scoreOutLocal_ : mrgValueLocal_;
    const LocalTensor<uint32_t> &idxLocal = singleTrunkIdxInSharedTmp ? topkSharedTmpLocal_ :
                                                                            indicesOutLocal_;
    WriteLDList(slot, s1RowLocal, valLocal, idxLocal);
    WriteLDParam(slot, info, segStartCol, segEndCol);
    (void)rowIdx;
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::WriteLDList(uint32_t slot, uint32_t s1RowLocal,
                                                    const LocalTensor<SCORE_T> &valLocal,
                                                    const LocalTensor<uint32_t> &idxLocal)
{
    uint32_t aic = (uint32_t)(blockId_ / 2);
    uint32_t aiv = (uint32_t)(blockId_ % 2);
    int64_t valOff = LDValElemOffset(aic, aiv, slot, s1RowLocal);
    int64_t idxOff = LDIdxElemOffset(aic, aiv, slot, s1RowLocal);
    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = topkCount_ * sizeof(SCORE_T);
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;
    SetFlag<HardEvent::V_MTE3>(TOPK_IDX_CYCLE);
    WaitFlag<HardEvent::V_MTE3>(TOPK_IDX_CYCLE);
    AscendC::DataCopyPad(ldValGm[valOff], valLocal, copyOutParams);
    copyOutParams.blockLen = topkCount_ * sizeof(uint32_t);
    AscendC::DataCopyPad(ldIdxGm[idxOff], idxLocal, copyOutParams);
    SetFlag<HardEvent::MTE3_V>(TOPK_IDX_CYCLE);  // arm only, no wait: next row's Wait consumes it
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::WriteLDParam(uint32_t slot, const QLICommon::RunInfo &info,
                                                     int64_t segStartCol, int64_t segEndCol)
{
    uint32_t aic = (uint32_t)(blockId_ / 2);
    uint32_t aiv = (uint32_t)(blockId_ % 2);
    int64_t pOff = LDParamElemOffset(aic, aiv, slot);
    ldParamGm.SetValue(pOff + 0, (int64_t)1);  // valid
    ldParamGm.SetValue(pOff + 1, (int64_t)(slot == 0 ? 1 : 0));  // isS2End: tail segment reaches row end
    ldParamGm.SetValue(pOff + 2, (int64_t)info.bN2Idx);
    ldParamGm.SetValue(pOff + 3, (int64_t)info.gS1Idx);
    ldParamGm.SetValue(pOff + 4, segStartCol);
    ldParamGm.SetValue(pOff + 5, segEndCol);
}

// L-fix: flush MTE3 pipe before cross-core SyncAll
// Wait<MTE3_V> consumes the last LD write arm; the flag only arms after the MTE3
// copies complete (in-order MTE3 queue), so passing this Wait proves every GM list
// is fully written before SyncAll releases the merge cores (Bug1 torn-read fix).
// Re-arm afterwards so the FreeEventID drain stays balanced.
template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::FlushLDWrites()
{
    WaitFlag<HardEvent::MTE3_V>(TOPK_IDX_CYCLE);
    SetFlag<HardEvent::MTE3_V>(TOPK_IDX_CYCLE);
}

// LD phase (after AIV-only SyncAll): the row-start core of each split row merges
// carry(own slot1) + remote(core+1 slot0) candidate lists into the final topK output.
template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ProcessLD()
{
    if (!ldRowValid_ || ldRowCount_ == 0) {
        return;
    }
    for (uint32_t t = 0; t < ldRowCount_; t++) {
    const QLICommon::RunInfo &info = ldRowInfoArr_[t];
    int64_t curS1Idx = info.gS1Idx * s1BaseSize_;
    int64_t curS1ProcNum = curS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int64_t curAivS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curS1ProcNum, 2) : curS1ProcNum / 2;
    if (curAivS1ProcNum == 0) {
        continue;  // zero rows for this AIV in this task; a return here silently
                   // skipped every remaining task's merge rows
    }
    uint32_t aic = (uint32_t)(blockId_ / 2);
    uint32_t aiv = (uint32_t)(blockId_ % 2);
    SCORE_T zero = 0;

    // prime events for LD loads/stores (AllocEventID/FreeEventID already paired in ProcessMain)
    SetFlag<HardEvent::V_MTE2>(TOPK_MRG_CYCLE);
    SetFlag<HardEvent::MTE3_V>(TOPK_IDX_CYCLE);


    AscendC::DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    copyInParams.rsv = 0;
    AscendC::DataCopyPadExtParams<SCORE_T> padValParams{true, 0, 0, 0};
    AscendC::DataCopyPadExtParams<uint32_t> padIdxParams{true, 0, 0, 0};
    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = topkCount_ * sizeof(uint32_t);
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    // fullIdx mirrors mrgValue: [carry 2048 | remote 2048]
    LocalTensor<uint32_t> fullIdx;
    if constexpr (sizeof(SCALE_T) == 4) {
        fullIdx = kScaleUB_.template ReinterpretCast<uint32_t>();
    } else {
        fullIdx = resMm1UB_.template ReinterpretCast<uint32_t>();
    }

    for (uint32_t i = 0; i < (uint32_t)curAivS1ProcNum; i++) {
        uint32_t rowIdx = blockId_ % 2 * CeilDiv(curS1ProcNum, 2) + i;

        // pre-zero the merge + index streams (V), then release to MTE2
        WaitFlag<HardEvent::MTE3_V>(TOPK_IDX_CYCLE);
        WaitFlag<HardEvent::V_MTE2>(TOPK_MRG_CYCLE);
        Duplicate(mrgValueLocal_, zero, 2 * topkCountAlign256_);
        Duplicate(fullIdx, (uint32_t)0, 2 * topkCountAlign256_);
        // BUG-4 fix: Dups -> MTE2 loads ordering via the fill-guard pair (ID5, same
        // shape as direct rows). The old S/W<V_MTE2>(ID4) pair here made rows
        // 3W+2S on ID4 (net -1): the ring starved from row 2 of a task.
        SetFlag<HardEvent::V_MTE2>(TOPK_FILL_GUARD);
        WaitFlag<HardEvent::V_MTE2>(TOPK_FILL_GUARD);

        // load carry (own slot1) + remote (next core slot0): vals into mrgValue, GLOBAL idx into fullIdx
        copyInParams.blockLen = topkCount_ * sizeof(SCORE_T);
        AscendC::DataCopyPad(mrgValueLocal_, ldValGm[LDValElemOffset(aic, aiv, 1, i)], copyInParams, padValParams);
        copyInParams.blockLen = topkCount_ * sizeof(uint32_t);
        AscendC::DataCopyPad(fullIdx, ldIdxGm[LDIdxElemOffset(aic, aiv, 1, i)], copyInParams, padIdxParams);
        copyInParams.blockLen = topkCount_ * sizeof(SCORE_T);
        AscendC::DataCopyPad(mrgValueLocal_[topkCountAlign256_],
                             ldValGm[LDValElemOffset(aic + 1, aiv, 0, i)], copyInParams, padValParams);
        copyInParams.blockLen = topkCount_ * sizeof(uint32_t);
        AscendC::DataCopyPad(fullIdx[topkCountAlign256_],
                             ldIdxGm[LDIdxElemOffset(aic + 1, aiv, 0, i)], copyInParams, padIdxParams);
        SetFlag<HardEvent::MTE2_V>(TOPK_MRG_CYCLE);
        WaitFlag<HardEvent::MTE2_V>(TOPK_MRG_CYCLE);

        // v2 merge: single-shot topk over 4096, positions -> gather GLOBAL indices
        topkOp_.LdTopK(mrgValueLocal_, fullIdx, indicesOutLocal_, scoreOutLocal_,
                       2 * topkCountAlign256_);

        // BUG-4 fix: release mrgValue to MTE2 for the next row (direct-row protocol:
        // ID4 rows are 2W+2S state-preserving; the missing end-Set starved the ring)
        SetFlag<HardEvent::V_MTE2>(TOPK_MRG_CYCLE);

        // final output: proven pair + arm (WriteLDList-style, no extra choreography)
        SetFlag<HardEvent::V_MTE3>(TOPK_IDX_CYCLE);
        WaitFlag<HardEvent::V_MTE3>(TOPK_IDX_CYCLE);
        AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_],
                             indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
        SetFlag<HardEvent::MTE3_V>(TOPK_IDX_CYCLE);  // arm-only: next head Wait consumes
    }
    WaitFlag<HardEvent::MTE3_V>(TOPK_IDX_CYCLE);
    SetFlag<HardEvent::MTE3_V>(TOPK_IDX_CYCLE);
}
    }  // end per-task merge loop
}  // namespace QLIKernel
#endif