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
    __aicore__ inline void ProcessTopK(const QLICommon::RunInfo &info, bool isFirstS2InnerLoop, bool isLastS2InnerLoop);
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
    __aicore__ inline uint32_t CalcDynamicTopkCount(uint32_t s1Idx, int32_t kb_offset, int32_t num_prompt_k);
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
    // score 保留在 vec1OutUB_ 中，不搬出到 GM，供 ProcessTopK 直接使用
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);
    CrossCoreSetFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_VC_EVENT + pingpong);
}

template <typename QLIT>
__aicore__ inline uint32_t QLIVector<QLIT>::CalcDynamicTopkCount(
    uint32_t s1Idx, int32_t kb_offset, int32_t num_prompt_k)
{
    // 常量定义
    const int32_t K_SMALL_SEQ_MAX = 56;
    const int32_t K_MEDIUM_SEQ_MAX = 160;
    const float K_BLOCK_NUM_RATE_MEDIUM = 0.2f;
    const float K_BLOCK_NUM_RATE_LARGE = 0.1f;
    const int32_t K_BLOCK_NUM_BIAS_MEDIUM = 30;
    const int32_t K_BLOCK_NUM_BIAS_LARGE = 30;
    const float ALPHA = 1.0f;
    
    // block级别的s1索引转换成token级别的s1索引
    s1Idx = s1Idx * 8;
    
    // 计算 k_start（分段函数）
    int32_t k_start;
    if (num_prompt_k < K_SMALL_SEQ_MAX) {
        k_start = K_SMALL_SEQ_MAX;
    } else if (num_prompt_k < K_MEDIUM_SEQ_MAX) {
        k_start = (int32_t)(num_prompt_k * K_BLOCK_NUM_RATE_MEDIUM + K_BLOCK_NUM_BIAS_MEDIUM);
    } else {
        k_start = (int32_t)(num_prompt_k * K_BLOCK_NUM_RATE_LARGE + K_BLOCK_NUM_BIAS_LARGE);
    }
    
    // 计算 s1Pos 和 decay_len
    int32_t s1Pos = (int32_t)s1Idx + kb_offset;
    int32_t decay_len = num_prompt_k - k_start;
    
    // 边界条件：s1Pos < k_start 或 decay_len < 1 时，直接返回 k_start
    if (s1Pos < k_start || decay_len < 1) {
        return (uint32_t)k_start;
    }
    
    // 计算 k_end
    int32_t k_end = (int32_t)(k_start * ALPHA);
    
    // 计算插值系数 t
    float t = (float)(s1Pos - k_start) / (float)(decay_len - 1);
    t = Max(0.0f, Min(1.0f, t));
    
    // 计算 dynamicTopkCount
    uint32_t dynamicTopkCount = (uint32_t)floorf((float)k_start + t * (float)(k_end - k_start));
    
    // 限制范围：下限为 1，上限为 k_start
    dynamicTopkCount = dynamicTopkCount < 1 ? 1 : (dynamicTopkCount > (uint32_t)k_start ? (uint32_t)k_start : dynamicTopkCount);
    
    return dynamicTopkCount;
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ProcessTopK(
    const QLICommon::RunInfo &info, bool isFirstS2InnerLoop, bool isLastS2InnerLoop)
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

    // S2 基本块信息
    uint32_t s2BlockStart = info.s2Idx * s2BaseSize_;
    uint32_t pingpong = info.loop % 2;

    // 输出搬运参数（从UB搬出到GM，仅最后一个基本块使用）
    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    // 计算当前S2方向的有效长度
    int32_t cuRealAcSeq = info.actS2Size;
    // 如果开启attenMask（causal mask），有效长度需要动态计算
    if (constInfo_.attenMaskFlag) {
        cuRealAcSeq = info.actS2Size - info.actS1Size + curAivS1Idx + 1;
    }

    // 有效S2总长度（用于计算必选块位置）
    int32_t validS2Len = cuRealAcSeq;
    
    // 计算动态 topkCount 所需的块级常量
    uint32_t s2Size = info.actS2Size;
    uint32_t s1Size = info.actS1Size;
    int32_t kb_offset = (int32_t)s2Size - (int32_t)s1Size;
    int32_t num_prompt_k = (int32_t)s2Size;  // 穿刺测试：假设 num_prompt_k = s2Size
    
    // 累计输出偏移（用于计算每行的 GM 输出地址，仅最后一个基本块使用）
    int64_t cumulativeOutputOffset = 0;
    
    // 常量定义：用于填充零值和无效索引
    uint16_t zero = 0;
    int32_t neg = -1;

    // 外层循环：按 group 遍历（同一个 group 内 s1Idx/validS2Len/topkCount 相同）
    uint32_t numGroups = ((uint32_t)curAivS1ProcNum + gSize_ - 1) / gSize_;
    for (uint32_t gIdx = 0; gIdx < numGroups; gIdx++) {
        // group 第一行在当前 AIV 核处理行中的索引（相对于当前核处理的起始行）
        uint32_t groupFirstI = gIdx * gSize_;
        // 全局 S1 索引
        uint32_t s1Idx = curAivS1Idx + groupFirstI;

        // 计算动态 topkCount（同一个 group 内相同）
        uint32_t dynamicTopkCount = CalcDynamicTopkCount(s1Idx, kb_offset, num_prompt_k);
        int32_t topkSelectNum = (int32_t)dynamicTopkCount;
        uint32_t topkSelectNumAlign256 = QLICommon::Align((uint32_t)topkSelectNum, (uint32_t)256);

        // 计算 validS2Len（同一个 group 内相同）
        int32_t groupValidS2Len = cuRealAcSeq;
        if (constInfo_.attenMaskFlag) {
            groupValidS2Len = (int32_t)groupFirstI + cuRealAcSeq;
        }

        // 重叠检测
        int32_t actualInit = Min((int32_t)INIT_BLOCKS, groupValidS2Len);
        int32_t actualWindow = (int32_t)WINDOW_BLOCKS;
        int32_t totalOutput = actualInit + topkSelectNum + actualWindow;

        // 当前基本块的S2范围
        uint32_t s2BlockEnd = Min(s2BlockStart + s2BaseSize_, (uint32_t)Max(groupValidS2Len, 0));
        uint32_t s2BlockValidLen = (s2BlockEnd > s2BlockStart) ? (s2BlockEnd - s2BlockStart) : 0;

        // 边界检查：有效长度不足
        if (groupValidS2Len <= 0) {
            if (isLastS2InnerLoop) {
                // 外层循环按 group 迭代，每个 group 包含 gSize_ 行
                // 即使边界条件也需要为 group 内每行生成输出，否则 GM 数据不完整
                uint32_t groupSize = Min((uint32_t)gSize_, (uint32_t)curAivS1ProcNum - groupFirstI);
                for (uint32_t j = 0; j < groupSize; j++) {
                    Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>(), neg, totalOutput);
                    SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                    WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                    copyOutParams.blockLen = (uint16_t)((uint32_t)totalOutput * sizeof(int32_t));
                    AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + cumulativeOutputOffset],
                                         indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
                    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
                    cumulativeOutputOffset += totalOutput;
                }
            }
            continue;
        }

        // 内层循环：遍历 group 内每行
        uint32_t groupSize = Min((uint32_t)gSize_, (uint32_t)curAivS1ProcNum - groupFirstI);
        for (uint32_t j = 0; j < groupSize; j++) {
            uint32_t i = groupFirstI + j;
            uint32_t rowIdx = blockId_ % 2 * CeilDiv(curS1ProcNum, 2) + i;

            // vec1OutUB_中当前行的基地址
            uint32_t vec1OutRowOffset = pingpong * (UB_BANK_STRIDE / sizeof(uint16_t))
                                        + rowIdx * (UB_BANK_DEPTH_STRIDE / sizeof(uint16_t));

            // 等待上轮MTE2和MTE3完成，确保buffer可安全覆盖
            WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

            // ========== TopK计算 ==========
            if (isFirstS2InnerLoop) {
                // ===== 第一个基本块：独立TopK =====
                AscendC::DataCopy(mrgValueLocal_, vec1OutUB_[vec1OutRowOffset], s2BlockValidLen);

                if (actualInit > 0) {
                    Duplicate(mrgValueLocal_, zero, (uint32_t)actualInit);
                }

                uint32_t effectiveInputLen = (uint32_t)Max((int32_t)s2BlockValidLen, topkSelectNum);
                uint32_t inputLenAlign = QLICommon::Align(effectiveInputLen, (uint32_t)256);
                Duplicate(mrgValueLocal_[effectiveInputLen / 256 * 256], zero,
                          inputLenAlign - effectiveInputLen / 256 * 256);
                PipeBarrier<PIPE_V>();

                topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, inputLenAlign, 0, 1);

            } else {
                // ===== 非第一个基本块：合并上一轮TopK结果 + 当前基本块score =====
                AscendC::DataCopy(topkSharedTmpLocal_, indicesOutLocal_.ReinterpretCast<int32_t>(), (uint32_t)topkSelectNum);
                PipeBarrier<PIPE_V>();

                AscendC::DataCopy(mrgValueLocal_, scoreOutLocal_, topkSelectNumAlign256);

                if (topkSelectNumAlign256 != (uint32_t)topkSelectNum) {
                    uint64_t mask[1];
                    mask[0] = ~0;
                    mask[0] = mask[0] << (topkSelectNum % 64);
                    PipeBarrier<PIPE_V>();
                    Duplicate(mrgValueLocal_[topkSelectNum / 64 * 64], zero, mask, 1, 1, 0);
                    PipeBarrier<PIPE_V>();
                    Duplicate(mrgValueLocal_[topkSelectNum / 64 * 64 + 64], zero,
                              topkSelectNumAlign256 - (topkSelectNum / 64 * 64 + 64));
                }

                AscendC::DataCopy(mrgValueLocal_[topkSelectNumAlign256],
                                 vec1OutUB_[vec1OutRowOffset], s2BlockValidLen);

                uint32_t mergedInputLen = topkSelectNumAlign256 + s2BlockValidLen;
                uint32_t mergedInputAlign = QLICommon::Align(mergedInputLen, (uint32_t)256);
                if (mergedInputLen % 256 != 0) {
                    Duplicate(mrgValueLocal_[mergedInputLen / 256 * 256], zero,
                              mergedInputAlign - mergedInputLen / 256 * 256);
                }
                PipeBarrier<PIPE_V>();

                topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, mergedInputAlign, 0, 1);

                for (uint32_t k = 0; k < (uint32_t)topkSelectNum; k++) {
                    uint32_t idx = indicesOutLocal_.GetValue(k);
                    uint32_t globalIdx;
                    if (idx < (uint32_t)topkSelectNum) {
                        globalIdx = topkSharedTmpLocal_.GetValue(idx);
                    } else {
                        globalIdx = s2BlockStart + (idx - (uint32_t)topkSelectNum);
                    }
                    indicesOutLocal_.SetValue(k, globalIdx);
                }
            }

            // ========== 索引重组和GM搬出（仅最后一个基本块）==========
            if (isLastS2InnerLoop) {
                int64_t gmBase = info.indiceOutOffset + cumulativeOutputOffset;

                AscendC::DataCopy(mrgValueLocal_.ReinterpretCast<int32_t>(),
                                  indicesOutLocal_.ReinterpretCast<int32_t>(),
                                  (uint32_t)topkSelectNum);
                PipeBarrier<PIPE_V>();

                AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(),
                                         (int32_t)zero, actualInit);
                PipeBarrier<PIPE_V>();

                SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
                SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

                copyOutParams.blockLen = (uint16_t)((uint32_t)actualInit * sizeof(int32_t));
                AscendC::DataCopyPad(indiceOutGm[gmBase],
                                     indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);

                copyOutParams.blockLen = (uint16_t)((uint32_t)topkSelectNum * sizeof(int32_t));
                AscendC::DataCopyPad(indiceOutGm[gmBase + actualInit],
                                     mrgValueLocal_.ReinterpretCast<int32_t>(), copyOutParams);

                SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

                if (actualWindow > 0) {
                    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

                    AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(),
                                             (int32_t)zero, actualWindow);
                    PipeBarrier<PIPE_V>();

                    int32_t windowBase = groupValidS2Len;
                    Duplicate(mrgValueLocal_.ReinterpretCast<int32_t>(),
                              (int32_t)windowBase, actualWindow);
                    PipeBarrier<PIPE_V>();

                    Add(indicesOutLocal_.ReinterpretCast<int32_t>(),
                        indicesOutLocal_.ReinterpretCast<int32_t>(),
                        mrgValueLocal_.ReinterpretCast<int32_t>(),
                        (uint32_t)actualWindow);
                    PipeBarrier<PIPE_V>();

                    SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                    WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

                    copyOutParams.blockLen = (uint16_t)((uint32_t)actualWindow * sizeof(int32_t));
                    AscendC::DataCopyPad(indiceOutGm[gmBase + actualInit + topkSelectNum],
                                         indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);

                    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
                }

                cumulativeOutputOffset += totalOutput;
            }
        }
    }
}
}  // namespace QLIKernel
#endif