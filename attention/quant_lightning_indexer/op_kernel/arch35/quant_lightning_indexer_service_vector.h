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
constexpr uint32_t TRUNK_LEN_16K = 16384;
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

    __aicore__ inline QLIVector(){};
    __aicore__ inline void ProcessVec1(const QLICommon::RunInfo &info);
    __aicore__ inline void ProcessTopK(const QLICommon::RunInfo &info);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct QLICommon::ConstInfo &constInfo,
                                      const QLITilingData *__restrict tilingData);
    __aicore__ inline void InitVecWorkspaceTensor(GlobalTensor<uint16_t> scoreGm);
    __aicore__ inline void InitVecInputTensor(GlobalTensor<W_T> weightsGm, GlobalTensor<SCALE_T> qScaleGm,
                                              GlobalTensor<SCALE_T> kScaleGm, GlobalTensor<int32_t> indiceOutGm,
                                              GlobalTensor<int32_t> blockTableGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

protected:
    GlobalTensor<uint16_t> scoreGm;
    GlobalTensor<W_T> weightsGm;
    GlobalTensor<SCALE_T> qScaleGm;
    GlobalTensor<SCALE_T> kScaleGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> blockTableGm;
    // =================================常量区=================================
    static constexpr uint32_t VEC1_V_MTE2_EVENT_KSCALE = EVENT_ID0;
    static constexpr uint32_t VEC1_MTE2_V_EVENT_KSCALE = EVENT_ID1;
    static constexpr uint32_t VEC1_V_MTE3_EVENT = EVENT_ID2;
    static constexpr uint32_t VEC1_MTE3_V_EVENT = EVENT_ID3;
    static constexpr uint32_t VEC1_V_MTE2_EVENT_QSCALE = EVENT_ID6;
    static constexpr uint32_t VEC1_MTE2_V_EVENT_QSCALE = EVENT_ID3;
    static constexpr uint32_t TOPK_V_MTE2_EVENT = EVENT_ID4;
    static constexpr uint32_t TOPK_MTE2_V_EVENT = EVENT_ID5;
    static constexpr uint32_t TOPK_V_MTE3_EVENT = EVENT_ID6;
    static constexpr uint32_t TOPK_MTE3_V_EVENT = EVENT_ID7;

    static constexpr uint32_t KSCALE_S_MTE2_EVENT = EVENT_ID7;
    static constexpr uint32_t MTE3_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t V_MTE2_EVENT1 = EVENT_ID2;
    static constexpr uint32_t V_MTE2_EVENT2 = EVENT_ID3;
    static constexpr uint32_t V_MTE2_EVENT3 = EVENT_ID5;

private:
    __aicore__ inline void GetKeyScale(const QLICommon::RunInfo &runInfo, LocalTensor<SCALE_T> &kScaleUB,
                                       int64_t batchId, int64_t startS2, int64_t getLen);
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
    LocalTensor<uint16_t> vec1OutUB_;
    // tmp buff for LD

    // tmp buff for topk
    TBuf<TPosition::VECCALC> mrgValueBuf_;
    LocalTensor<uint16_t> mrgValueLocal_;

    TBuf<TPosition::VECCALC> indicesOutBuf_;
    LocalTensor<uint32_t> indicesOutLocal_;

    TBuf<TPosition::VECCALC> scoreOutBuf_;
    LocalTensor<uint16_t> scoreOutLocal_;

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
    topk::LITopk<uint16_t> topkOp_;
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
    
    pipe->InitBuffer(outBuf_, 2 * CeilDiv(s1BaseSize_, 2) * s2BaseSize_ * sizeof(uint16_t));
    vec1OutUB_ = outBuf_.Get<uint16_t>();

    // Topk
    pipe->InitBuffer(mrgValueBuf_, (topkCountAlign256_ + trunkLen_) * sizeof(uint16_t));     // 大小：(topkCountAlign256_ + 每次排序长度) * sizeof(uint16_t)
    mrgValueLocal_ = mrgValueBuf_.Get<uint16_t>();
    outInvalidLocal_ = mrgValueBuf_.Get<int32_t>();

    pipe->InitBuffer(indicesOutBuf_, (topkCountAlign256_ + 64) * sizeof(uint32_t));         // 大小：(topkCountAlign256_ + 64) * 4  64:duplicate刷-1需要额外空间
    indicesOutLocal_ = indicesOutBuf_.Get<uint32_t>();

    pipe->InitBuffer(scoreOutBuf_, topkCountAlign256_ * sizeof(uint16_t));                   // 大小：topkCountAlign256_ * sizeof(uint16_t)
    scoreOutLocal_ = scoreOutBuf_.Get<uint16_t>();

    uint64_t topkSharedTmpSize = topkOp_.GetSharedTmpBufferSize();
    pipe->InitBuffer(topkSharedTmpBuf_, topkSharedTmpSize);
    topkSharedTmpLocal_ = topkSharedTmpBuf_.Get<uint32_t>();
    topkOp_.InitBuffers(topkSharedTmpLocal_);

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
    trunkLen_ = TRUNK_LEN_16K;
    topkCount_ = constInfo.sparseCount;
    topkOp_.Init(topkCount_, trunkLen_);
    topkCountAlign256_ = QLICommon::Align(constInfo.sparseCount, (uint64_t)256); // topkCount对齐到256
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
__aicore__ inline void QLIVector<QLIT>::InitVecWorkspaceTensor(GlobalTensor<uint16_t> scoreGm)
{
    this->scoreGm = scoreGm; // resucesum*k
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 0);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 1);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 0);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 1);

    SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 0);
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + 1);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 0);
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + 1);

    WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::CleanInvalidOutput(int64_t invalidS1Offset)
{
    // init -1 and copy to output
    Duplicate(outInvalidLocal_, constInfo_.INVALID_IDX, constInfo_.sparseCount);

    SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
    WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

    AscendC::DataCopyParams dataCopyOutParams;
    dataCopyOutParams.blockCount = 1;
    dataCopyOutParams.blockLen = constInfo_.sparseCount * sizeof(int32_t);
    dataCopyOutParams.srcStride = 0;
    dataCopyOutParams.dstStride = 0;
    AscendC::DataCopyPad(indiceOutGm[invalidS1Offset], outInvalidLocal_, dataCopyOutParams);
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
            SetFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            WaitFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
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
            SetFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            WaitFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
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
    
    if (info.isFirstS2InnerLoop) {
        WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + qScalepingpong);
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
        SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_QSCALE + qScalepingpong);
        WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_QSCALE + qScalepingpong);
    }

    if ((info.s2Idx - info.s2Start) % 16 == 0) {
        WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + kScalepingpong);
        uint32_t getLen = 16 * s2BaseSize_ > (info.validS2Len - info.s2Idx * s2BaseSize_)
                                           ? info.validS2Len - info.s2Idx * s2BaseSize_
                                           : 16 * s2BaseSize_;
        //kScaleGm  -->  kScaleUB_
        GetKeyScale(info, kScaleUB_, info.bIdx, curS2Idx, getLen);
        SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_KSCALE + kScalepingpong);
        WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT_KSCALE + kScalepingpong);
    }
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);

    //CV同步
    CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_CV_EVENT + info.loop % 2);   //V核等C核计算完mm1，mm1Res已搬运到UB

    static_assert(std::is_same_v<uint16_t, uint16_t>);
    auto outBase = vec1OutUB_[pingpong * (UB_BANK_STRIDE / sizeof(uint16_t))];
    auto weightBase = weightUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(W_T))];
    auto weightFloatBase = weightFloatUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(float))];
    auto qScaleBase = qScaleUB_[qScalepingpong * (UB_BANK_STRIDE / sizeof(SCALE_T))];
    auto kScaleBase = kScaleUB_[kScalepingpong * 16 * s2BaseSize_ + ((info.s2Idx - info.s2Start) % 16) * s2BaseSize_];
    auto qkBase = resMm1UB_[pingpong * (UB_BANK_STRIDE / sizeof(float))];
    auto qkVLstride = (UB_BANK_DEPTH_STRIDE / sizeof(float)) / 2 * constInfo_.mBaseSize;

    vector1::BatchMulWeightAndReduceSum(outBase, UB_BANK_DEPTH_STRIDE / sizeof(uint16_t),
                                        qkBase, qkVLstride, (uint32_t)(gSize_ * UB_BANK_DEPTH_STRIDE / sizeof(float)), 
                                        weightBase, UB_BANK_DEPTH_STRIDE / sizeof(W_T), weightFloatBase,
                                        kScaleBase, (uint32_t)0,
                                        qScaleBase, UB_BANK_DEPTH_STRIDE / sizeof(SCALE_T),
                                        gSize_, curAivS1ProcNum);
    if (info.isFirstS2InnerLoop) {
        SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_QSCALE + qScalepingpong);
    }
    if ((info.s2Idx - info.s2Start) % 16 == 0) {
        SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT_KSCALE + kScalepingpong);
    }
    SetFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + pingpong);
    WaitFlag<HardEvent::V_MTE3>(VEC1_V_MTE3_EVENT + pingpong);
    //outUB_ --->  scoreGm
    int64_t vec1OutGmOffset = blockId_ % 2 == 0 ? curS2Idx :
                            s1BaseSizePerAIV * QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) + curS2Idx;
    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = curAivS1ProcNum;
    copyOutParams.blockLen = s2BaseSize_ * sizeof(uint16_t);
    copyOutParams.srcStride = (UB_BANK_DEPTH_STRIDE - UB_BANK_STRIDE) / 32;
    copyOutParams.dstStride = (QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_) - s2BaseSize_) * sizeof(uint16_t);
    DataCopyPad(scoreGm[vec1OutGmOffset], outBase, copyOutParams);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);
    CrossCoreSetFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_VC_EVENT + pingpong);   //V核处理完，通知C核可以把mm1Res搬运到UB
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ProcessTopK(const QLICommon::RunInfo &info)
{
    // 等待MTE3到MTE2的同步事件，确保之前的内存搬运完成
    SetFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_MTE2>(MTE3_MTE2_EVENT);

    // init_blocks: K方向前INIT_BLOCKS个块必选（索引0 ~ INIT_BLOCKS-1）
    const int32_t INIT_BLOCKS = 4;
    // window_blocks: K方向后WINDOW_BLOCKS个块必选（索引validS2Len-WINDOW_BLOCKS ~ validS2Len-1）
    const int32_t WINDOW_BLOCKS = 4;

    // 当前处理的S1行起始索引（全局位置）
    int64_t curS1Idx = info.gS1Idx * s1BaseSize_;
    // 当前S1行需要处理的数量（可能小于s1BaseSize_，处理尾块）
    int64_t curS1ProcNum = curS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    // 当前AIV核处理的S1起始索引（双核分工）
    int64_t curAivS1Idx = curS1Idx + (blockId_ % 2) * CeilDiv(curS1ProcNum, 2);
    // 当前AIV核需要处理的S1行数量
    int64_t curAivS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curS1ProcNum, 2) : curS1ProcNum / 2;

    // 输入搬运参数（从GM搬入UB）
    AscendC::DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    copyInParams.rsv = 0;

    // 输出搬运参数（从UB搬出到GM）
    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = topkCount_ * sizeof(uint32_t);
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    // 计算当前S2方向的有效长度
    int32_t cuRealAcSeq = info.actS2Size;
    // 如果开启attenMask（causal mask），有效长度需要动态计算
    if (constInfo_.attenMaskFlag) {
        cuRealAcSeq = info.actS2SizeOrig - info.actS1Size + curAivS1Idx + 1;
    }

    // 有效S2总长度（用于计算必选块位置）
    int32_t validS2Len = cuRealAcSeq;
    // 遍历当前AIV核需要处理的每个S1行
    for (uint32_t i = 0; i < curAivS1ProcNum; i++) {
        // 当前S1行在核内的相对索引（用于计算输出偏移）
        uint32_t rowIdx = blockId_ % 2 * CeilDiv(curS1ProcNum, 2) + i;
        // 当前S1行在scoreGm中的偏移（score存储按行组织）
        uint32_t vecOffset = blockId_ % 2 * CeilDiv(s1BaseSize_, 2) + i;

        // 常量定义：用于填充零值和无效索引
        uint16_t zero = 0;
        int32_t neg = -1;

        // 如果开启attenMask，每行的有效S2长度需要动态计算
        if (constInfo_.attenMaskFlag) {
            validS2Len = (int32_t)i + cuRealAcSeq;
        }

        // ========== 重叠检测：当validS2Len < INIT_BLOCKS + WINDOW_BLOCKS时，前后必选块范围重叠 ==========
        // actualInit: 实际前部必选块数（不超过validS2Len）
        // actualWindow: 实际后部必选块数（不超过validS2Len - actualInit，避免与前部重叠）
        // actualMust: 实际必选块总数 = actualInit + actualWindow
        int32_t actualInit = Min((int32_t)INIT_BLOCKS, validS2Len);
        int32_t actualWindow = Min((int32_t)WINDOW_BLOCKS, validS2Len - actualInit);
        int32_t actualMust = actualInit + actualWindow;

        // TopK计算的起始位置：跳过前actualInit个必选块
        int32_t topkValidStart = actualInit;
        // TopK计算的结束位置：排除后actualWindow个必选块
        int32_t topkValidEnd = validS2Len - actualWindow;
        // TopK需要处理的中间有效长度
        int32_t topkValidLen = topkValidEnd - topkValidStart;
        // TopK需要从中间选取的数量（总topK减去实际必选块数）
        int32_t topkSelectNum = (int32_t)topkCount_ - actualMust;

        // score行在GM中的基地址
        uint32_t scoreRowBase = vecOffset *
            QLICommon::Align((uint64_t)constInfo_.kSeqSize, (uint64_t)s2BaseSize_);

        // 等待上轮MTE2和MTE3完成，确保buffer可安全覆盖
        WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

        // 边界检查：有效长度不足，输出无效索引
        if (validS2Len <= 0) {
            Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>(), neg, topkCount_);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_],
                                 indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            continue;
        }

        // 定义score搬运的填充参数
        AscendC::DataCopyPadExtParams<uint16_t> padParams{true, 0, 0, 0};

        // ========== 无需TopK的分支：必选块已覆盖全部topK名额，或有效元素不超过topK名额 ==========
        // 条件1: actualMust >= topkCount_ → 必选块已经够topK名额
        // 条件2: validS2Len <= topkCount_ → 所有有效元素都不够topK，全选+填-1
        if (actualMust >= (int32_t)topkCount_ || validS2Len <= (int32_t)topkCount_) {
            // 输出索引 = [0, 1, ..., actualMust-1, -1, -1, ...]
            // 生成必选块索引序列：[0, 1, ..., actualMust-1]
            AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(),
                                     (int32_t)zero, actualMust);

            // 如果actualMust < topkCount_，填充剩余位置为-1（无效索引）
            if (actualMust < (int32_t)topkCount_) {
                uint64_t mask[1];
                mask[0] = ~0;
                mask[0] = mask[0] << (actualMust % 8);
                PipeBarrier<PIPE_V>();
                Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[actualMust / 8 * 8],
                          neg, mask, 1, 1, 0);
            }
            if (actualMust / 8 * 8 + 64 < (int32_t)topkCount_) {
                PipeBarrier<PIPE_V>();
                Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>()[actualMust / 8 * 8 + 64],
                          neg, topkCount_ - (actualMust / 8 * 8 + 64));
            }

            // 设置同步并搬出索引
            SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_],
                                 indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            continue;
        }

        // ========== 执行TopK计算（仅对中间部分，跳过前后必选块）==========
        // topkSelectNum: 需要从中间选取的数量（已在上方计算）
        // topkValidLen: 中间有效长度
        // 注意：topkValidLen 可能 < topkSelectNum（中间元素不够选），此时TopK会选出全部中间元素，
        //       剩余位置由后续填充-1处理
        uint32_t effectiveTopkInputLen = (uint32_t)Max(topkValidLen, topkSelectNum);
        uint32_t s2LoopNum = (effectiveTopkInputLen + trunkLen_ - 1) / trunkLen_;

        // 单轮TopK场景：有效长度较小，一次计算完成
        if (s2LoopNum == 1) {
            // 对齐计算：TopK直方图需要256对齐
            uint32_t inputLenAlign = QLICommon::Align(effectiveTopkInputLen, (uint32_t)256);
            // 在UB中填充尾部零值（超出有效长度的部分，包括topkValidLen不足topkSelectNum的零填充）
            Duplicate(mrgValueLocal_[effectiveTopkInputLen / 256 * 256],
                      zero,
                      inputLenAlign - effectiveTopkInputLen / 256 * 256);

            SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
            WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);

            // 从GM搬入中间部分score（跳过前actualInit个必选块）
            uint32_t scoreGmOffset = scoreRowBase + topkValidStart;
            copyInParams.blockLen = (uint32_t)topkValidLen * sizeof(uint16_t);
            AscendC::DataCopyPad(mrgValueLocal_, scoreGm[scoreGmOffset], copyInParams, padParams);

            SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
            WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);

            // TopK输入：中间部分score（长度effectiveTopkInputLen，含零填充）
            // TopK输出：indicesOutLocal_（相对中间部分的索引）、scoreOutLocal_（选出的score值）
            topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_,
                    inputLenAlign, 0, 1);
        } else {
            // 多轮TopK场景：有效长度很大，需要多轮归约
            // topkSelectNumAlign256: 实际选取数量（减去必选块）的256对齐值
            // scoreOutLocal_中只有topkSelectNum个有效值，归约时只需搬入这么多
            uint32_t topkSelectNumAlign256 = QLICommon::Align((uint32_t)topkSelectNum, (uint32_t)256);

            for (uint32_t loopIdx = 0; loopIdx < s2LoopNum; loopIdx++) {
                if (loopIdx == 0) {
                    copyInParams.blockLen = trunkLen_ * sizeof(uint16_t);
                    uint32_t scoreGmOffset = scoreRowBase + topkValidStart;
                    AscendC::DataCopyPad(mrgValueLocal_, scoreGm[scoreGmOffset], copyInParams, padParams);

                    SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                    WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);

                    topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_,
                            trunkLen_, loopIdx, s2LoopNum);
                    continue;
                }

                SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);
                WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT2);

                // 当前轮有效长度（最后一轮可能不足trunkLen_）
                uint32_t validTrunkLen = (loopIdx * trunkLen_ + trunkLen_) > effectiveTopkInputLen
                                         ? effectiveTopkInputLen % trunkLen_
                                         : trunkLen_;

                uint32_t offset = scoreRowBase + topkValidStart + loopIdx * trunkLen_;

                // 将上一轮TopK结果（topkSelectNum个score值）搬入mrgValueLocal_开头，用于归约合并
                AscendC::DataCopy(mrgValueLocal_, scoreOutLocal_, topkSelectNumAlign256);

                // 如果topkSelectNumAlign256 != topkSelectNum，填充尾部零值
                if (topkSelectNumAlign256 != (uint32_t)topkSelectNum) {
                    uint64_t mask[1];
                    mask[0] = ~0;
                    mask[0] = mask[0] << (topkSelectNum % 64);
                    PipeBarrier<PIPE_V>();
                    Duplicate(mrgValueLocal_[topkSelectNum / 64 * 64], zero, mask, 1, 1, 0);
                    PipeBarrier<PIPE_V>();
                    Duplicate(mrgValueLocal_[topkSelectNum / 64 * 64 + 64], zero,
                              topkSelectNumAlign256 - (topkSelectNum / 64 * 64 + 64));
                    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                }

                copyInParams.blockLen = validTrunkLen * sizeof(uint16_t);

                // 新chunk追加到topkSelectNumAlign256位置，检查合并后是否需要256对齐补齐
                if ((topkSelectNumAlign256 + validTrunkLen) % 256 != 0) {
                    Duplicate(
                        mrgValueLocal_[topkSelectNumAlign256 + validTrunkLen / 256 * 256],
                        zero,
                        QLICommon::Align(validTrunkLen, (uint32_t)256) - validTrunkLen / 256 * 256);
                    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT3);
                }

                WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);

                // 新chunk搬入mrgValueLocal_的topkSelectNumAlign256位置
                AscendC::DataCopyPad(mrgValueLocal_[topkSelectNumAlign256], scoreGm[offset], copyInParams, padParams);

                SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);

                // TopK输入长度 = 上轮topK结果 + 新chunk，256对齐
                topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_,
                        QLICommon::Align(topkSelectNumAlign256 + validTrunkLen, (uint32_t)256),
                        loopIdx, s2LoopNum);

                SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
            }
        }

        // ========== 索引重组：分段写入GM，避免UB非32B对齐偏移 ==========
        // 核心思路：所有UB操作都在buffer起始位置（32B对齐）完成，
        //          通过3次DataCopyPad分段写入GM（GM地址无对齐要求，blockLen支持非对齐）
        // 最终GM布局：[前必选(actualInit) | TopK选出(topkSelectNum) | 后必选(actualWindow)]
        PipeBarrier<PIPE_V>();

        // --- Step 1: TopK索引加偏移，映射回原始S2位置 ---
        // TopK输出的索引是相对于中间部分的（0 ~ topkValidLen-1），
        // 需要加上topkValidStart（= actualInit）得到原始位置
        Duplicate(mrgValueLocal_.ReinterpretCast<int32_t>(),
                  (int32_t)topkValidStart, topkSelectNum);
        PipeBarrier<PIPE_V>();

        // 对前topkSelectNum个元素做加法，calCount版本内部自动处理repeat和mask
        Add(indicesOutLocal_.ReinterpretCast<int32_t>(),
            indicesOutLocal_.ReinterpretCast<int32_t>(),
            mrgValueLocal_.ReinterpretCast<int32_t>(),
            (uint32_t)topkSelectNum);
        PipeBarrier<PIPE_V>();

        // --- Step 2: 保存TopK索引到mrgValueLocal_（起始位置，32B对齐）---
        AscendC::DataCopy(mrgValueLocal_.ReinterpretCast<int32_t>(),
                          indicesOutLocal_.ReinterpretCast<int32_t>(),
                          (uint32_t)topkSelectNum);
        PipeBarrier<PIPE_V>();

        // --- Step 3: 在indicesOutLocal_起始位置生成前必选索引 [0, 1, ..., actualInit-1] ---
        AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(),
                                 (int32_t)zero, actualInit);
        PipeBarrier<PIPE_V>();

        // === 写入段1+段2到GM ===
        // 此时：indicesOutLocal_ = 前必选索引，mrgValueLocal_ = TopK索引
        int64_t gmBase = info.indiceOutOffset + (curS1Idx + rowIdx) * topkCount_;

        SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

        // 段1: 前必选索引 → GM[base, base+actualInit)
        copyOutParams.blockLen = (uint16_t)((uint32_t)actualInit * sizeof(int32_t));
        AscendC::DataCopyPad(indiceOutGm[gmBase],
                             indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);

        // 段2: TopK索引 → GM[base+actualInit, base+actualInit+topkSelectNum)
        copyOutParams.blockLen = (uint16_t)((uint32_t)topkSelectNum * sizeof(int32_t));
        AscendC::DataCopyPad(indiceOutGm[gmBase + actualInit],
                             mrgValueLocal_.ReinterpretCast<int32_t>(), copyOutParams);

        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

        // === 生成后必选索引并写入段3到GM ===
        if (actualWindow > 0) {
            // 等待MTE3完成，避免buffer被覆盖
            WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

            // 在indicesOutLocal_起始位置生成 [0, 1, ..., actualWindow-1]
            AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(),
                                     (int32_t)zero, actualWindow);
            PipeBarrier<PIPE_V>();

            // 在mrgValueLocal_起始位置填充基址 windowBase
            int32_t windowBase = validS2Len - actualWindow;
            Duplicate(mrgValueLocal_.ReinterpretCast<int32_t>(),
                      (int32_t)windowBase, actualWindow);
            PipeBarrier<PIPE_V>();

            // 基址 + [0,1,...,actualWindow-1] = [windowBase, windowBase+1, ..., validS2Len-1]
            Add(indicesOutLocal_.ReinterpretCast<int32_t>(),
                indicesOutLocal_.ReinterpretCast<int32_t>(),
                mrgValueLocal_.ReinterpretCast<int32_t>(),
                (uint32_t)actualWindow);
            PipeBarrier<PIPE_V>();

            // 段3: 后必选索引 → GM[base+actualInit+topkSelectNum, base+topkCount_)
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

            copyOutParams.blockLen = (uint16_t)((uint32_t)actualWindow * sizeof(int32_t));
            AscendC::DataCopyPad(indiceOutGm[gmBase + actualInit + topkSelectNum],
                                 indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);

            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        }
    }
}
}  // namespace QLIKernel
#endif