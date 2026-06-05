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
 * \file stem_indexer_service_vector.h
 * \brief
 */
#ifndef stem_indexer_SERVICE_VECTOR_H
#define stem_indexer_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "stem_indexer_common.h"
#include "../arch35/vf/stem_indexer_vector1.h"
#include "../arch35/vf/stem_indexer_topk.h"

namespace QLIKernel {
using namespace QLICommon;
constexpr uint32_t TRUNK_LEN_16K = 16384;
constexpr uint32_t TRUNK_LEN_256 = 256;
template <typename QLIT>
class QLIVector {
public:
    // =================================类型定义区=================================
    static constexpr LI_LAYOUT Q_LAYOUT_T = QLIT::layout;
    static constexpr LI_LAYOUT K_LAYOUT_T = QLIT::keyLayout;
    static constexpr bool PAGE_ATTENTION = QLIT::pageAttention;

    using QK_T = typename QLIT::queryKeyType;
    using SCORE_T = typename QLIT::scoreType;

    __aicore__ inline QLIVector(){};
    __aicore__ inline void ProcessVec1(const QLICommon::RunInfo &info);
    __aicore__ inline void ProcessTopK(const QLICommon::RunInfo &info, bool isFirstS2InnerLoop, bool isLastS2InnerLoop);
    __aicore__ inline void ProcessDirectOutput(const QLICommon::RunInfo &info, uint32_t validS2Len);
    __aicore__ inline void ProcessLD();
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct QLICommon::ConstInfo &constInfo,
                                      const struct QLICommon::LdSplitCoreInfo &ldInfo,
                                      const QLITilingData *__restrict tilingData);
    __aicore__ inline void InitVecWorkspaceTensor(GlobalTensor<SCORE_T> scoreGm,
                                                    GlobalTensor<SCORE_T> ldScoreGm,
                                                    GlobalTensor<int32_t> ldIndexGm);
    __aicore__ inline void InitVecInputTensor(GlobalTensor<float> weightsGm, GlobalTensor<float> qScaleGm,
                                              GlobalTensor<float> kScaleGm, GlobalTensor<int32_t> indiceOutGm,
                                              GlobalTensor<int32_t> indiceLenGm,
                                              GlobalTensor<int32_t> blockTableGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitLDBuffers(TPipe *pipe, const struct QLICommon::LdSplitCoreInfo &ldInfo);

protected:
    GlobalTensor<SCORE_T> scoreGm;
    GlobalTensor<float> weightsGm;
    GlobalTensor<SCORE_T> ldScoreGm;
    GlobalTensor<int32_t> ldIndexGm;
    GlobalTensor<float> qScaleGm;
    GlobalTensor<float> kScaleGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<int32_t> indiceLenGm;
    GlobalTensor<int32_t> blockTableGm;
    // =================================常量区=================================
    static constexpr uint32_t VEC1_V_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t VEC1_MTE2_V_EVENT = EVENT_ID1;
    static constexpr uint32_t VEC1_V_MTE3_EVENT = EVENT_ID2;
    static constexpr uint32_t VEC1_MTE3_V_EVENT = EVENT_ID3;

    static constexpr uint32_t TOPK_V_MTE2_EVENT = EVENT_ID4;
    static constexpr uint32_t TOPK_MTE2_V_EVENT = EVENT_ID5;
    static constexpr uint32_t TOPK_V_MTE3_EVENT = EVENT_ID6;
    static constexpr uint32_t TOPK_MTE3_V_EVENT = EVENT_ID7;

    static constexpr uint32_t KSCALE_S_MTE2_EVENT = EVENT_ID7;
    static constexpr uint32_t MTE3_MTE2_EVENT = EVENT_ID0;
    static constexpr uint32_t V_MTE2_EVENT = EVENT_ID7;
    static constexpr uint32_t V_MTE2_EVENT1 = EVENT_ID2;
    static constexpr uint32_t V_MTE2_EVENT2 = EVENT_ID3;
    static constexpr uint32_t V_MTE2_EVENT3 = EVENT_ID5;

private:
    __aicore__ inline void GetKeyScale(const QLICommon::RunInfo &runInfo, LocalTensor<float> &kScaleUB,
                                       int64_t batchId, int64_t startS2, int64_t getLen);
    __aicore__ inline uint32_t CalcDynamicTopkCount(uint32_t s1Idx, int32_t kb_offset, int32_t num_prompt_k);
    // ================================Local Buffer区====================================

    // tmp buff for vector
    TBuf<TPosition::VECCALC> resMm1Buf_;
    LocalTensor<QK_T> resMm1UB_;
    //tmp buff for weight
    TBuf<TPosition::VECCALC> weightBuf_;
    LocalTensor<float> weightUB_;
    //tmp buff for kScale
    TBuf<TPosition::VECCALC> kScaleBuf_;
    LocalTensor<float> kScaleUB_;
    //tmp buff for qScale
    TBuf<TPosition::VECCALC> qScaleBuf_;
    LocalTensor<float> qScaleUB_;
    //tmp buff for out
    TBuf<TPosition::VECCALC> outBuf_;
    LocalTensor<SCORE_T> vec1OutUB_;
    // tmp buff for LD
    TBuf<TPosition::VECCALC> ldValueBuf_;
    LocalTensor<uint32_t> ldValueLocal_;  // SCORE_T

    TBuf<TPosition::VECCALC> topkIndexBuf_;
    LocalTensor<uint32_t> topkIndexLocal_;

    TBuf<TPosition::VECCALC> topkValueBuf_;
    LocalTensor<uint32_t> topkValueLocal_;

    // tmp buff for topk
    TBuf<TPosition::VECCALC> mrgValueBuf_;
    LocalTensor<SCORE_T> mrgValueLocal_;

    TBuf<TPosition::VECCALC> indicesOutBuf_;
    LocalTensor<uint32_t> indicesOutLocal_;

    TBuf<TPosition::VECCALC> scoreOutBuf_;
    LocalTensor<SCORE_T> scoreOutLocal_;

    TBuf<TPosition::VECCALC> topkSharedTmpBuf_;
    LocalTensor<uint32_t> topkSharedTmpLocal_;

    TBuf<TPosition::VECCALC> outInvalidBuf_;
    LocalTensor<int32_t> outInvalidLocal_;

    TBuf<TPosition::VECCALC> indiceLenBuf_;
    LocalTensor<int32_t> indiceLenLocal_;

    LocalTensor<uint32_t> ldIndexLocal_;

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
    static constexpr uint32_t MAX_TOPK_COUNT = 256;
    uint32_t trunkLen_ = 0;

    struct QLICommon::ConstInfo constInfo_;
    struct QLICommon::LdSplitCoreInfo ldInfo_;
    topk::LITopk<SCORE_T> topkOp_;
};

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(resMm1Buf_, 2 * CeilDiv(constInfo_.mBaseSize, 2) * s2BaseSize_ * sizeof(QK_T));   //大小：2(开dB) * 2 * 64 * 128 * 4 = 128KB
    resMm1UB_ = resMm1Buf_.Get<QK_T>();//qk
    pipe->InitBuffer(weightBuf_, 2 * CeilDiv(s1BaseSize_, 2) * gSize_* sizeof(float));   // 大小：2(开dB) * 2 * 64 * 2 = 0.5KB 
    weightUB_ = weightBuf_.Get<float>();//weight
    pipe->InitBuffer(kScaleBuf_, 2 * s2BaseSize_ * sizeof(float));                   // 大小：2(开dB) * 128 * 4 = 1KB
    kScaleUB_ = kScaleBuf_.Get<float>();//kScale
    pipe->InitBuffer(qScaleBuf_, 2 * CeilDiv(s1BaseSize_, 2) * gSize_* sizeof(float));      // 大小：2(开dB) * 2 * 64 * 4 = 1KB
    qScaleUB_ = qScaleBuf_.Get<float>();//qScale
    pipe->InitBuffer(outBuf_, 2 * CeilDiv(s1BaseSize_, 2) * s2BaseSize_ * sizeof(SCORE_T));      // 大小：2(开dB) * 2 * 128 * 4 = 2KB
    vec1OutUB_ = outBuf_.Get<SCORE_T>();//out

    // Topk
    pipe->InitBuffer(mrgValueBuf_, (MAX_TOPK_COUNT + trunkLen_) * sizeof(SCORE_T));     // 大小：(topkCountAlign256_ + 每次排序长度) * sizeof(SCORE_T)
    mrgValueLocal_ = mrgValueBuf_.Get<SCORE_T>();
    
    pipe->InitBuffer(indicesOutBuf_, MAX_TOPK_COUNT * sizeof(uint32_t));         // 大小：(topkCountAlign256_ + 64) * 4  64:duplicate刷-1需要额外空间
    indicesOutLocal_ = indicesOutBuf_.Get<uint32_t>();

    pipe->InitBuffer(scoreOutBuf_, MAX_TOPK_COUNT * sizeof(SCORE_T));            // (topkCountAlign256_ + 64) * sizeof(SCORE_T) 64:duplicate刷-1额外空间
    scoreOutLocal_ = scoreOutBuf_.Get<SCORE_T>();

    uint64_t topkSharedTmpSize = topkOp_.GetSharedTmpBufferSize();
    pipe->InitBuffer(topkSharedTmpBuf_, topkSharedTmpSize);
    topkSharedTmpLocal_ = topkSharedTmpBuf_.Get<uint32_t>();
    topkOp_.InitBuffers(topkSharedTmpLocal_);

    //刷-1
    pipe->InitBuffer(outInvalidBuf_, MAX_TOPK_COUNT * sizeof(int32_t));
    outInvalidLocal_ = outInvalidBuf_.Get<int32_t>();

    // indice_len
    pipe->InitBuffer(indiceLenBuf_, mBaseSize_ * sizeof(int32_t));
    indiceLenLocal_ = indiceLenBuf_.Get<int32_t>();

    Duplicate(kScaleUB_, float(0), 2 * s2BaseSize_);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitLDBuffers(TPipe *pipe, const struct QLICommon::LdSplitCoreInfo &ldInfo)
{
    ldIndexLocal_ = resMm1UB_.template ReinterpretCast<uint32_t>();
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitParams(const struct QLICommon::ConstInfo &constInfo,
                                                   const struct QLICommon::LdSplitCoreInfo &ldInfo,
                                                   const QLITilingData *__restrict tilingData)
{
    this->constInfo_ = constInfo;
    this->ldInfo_ = ldInfo;
    blockS2StartIdx_ = 0;
    gSize_ = constInfo.gSize;
    kSeqSize_ = constInfo.kSeqSize;
    // define N2 para
    kHeadNum_ = constInfo.kHeadNum;
    qHeadNum_ = constInfo.qHeadNum;
    // define MMBase para
    s1BaseSize_ = constInfo.s1BaseSize;  // 4
    s2BaseSize_ = constInfo.s2BaseSize;  // 128
    mBaseSize_ = constInfo.mBaseSize;
    kCacheBlockSize_ = constInfo.kCacheBlockSize;
    maxBlockNumPerBatch_ = constInfo.maxBlockNumPerBatch;
    blockId_ = GetBlockIdx();
    trunkLen_ = TRUNK_LEN_256;
    topkOp_.Init(MAX_TOPK_COUNT, trunkLen_);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitVecInputTensor(GlobalTensor<float> weightsGm, GlobalTensor<float> qScaleGm,
                                                           GlobalTensor<float> kScaleGm,
                                                           GlobalTensor<int32_t> indiceOutGm,
                                                           GlobalTensor<int32_t> indiceLenGm,
                                                           GlobalTensor<int32_t> blockTableGm)
{
    this->weightsGm = weightsGm;
    this->qScaleGm = qScaleGm;
    this->kScaleGm = kScaleGm;
    this->indiceOutGm = indiceOutGm;
    this->indiceLenGm = indiceLenGm;
    this->blockTableGm = blockTableGm;
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::InitVecWorkspaceTensor(GlobalTensor<SCORE_T> scoreGm,
                                                            GlobalTensor<SCORE_T> ldScoreGm,
                                                            GlobalTensor<int32_t> ldIndexGm)
{
    this->scoreGm = scoreGm;//resucesum*k
    this->ldScoreGm = ldScoreGm;
    this->ldIndexGm = ldIndexGm;
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 0);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);

    SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 0);
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);

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
__aicore__ inline void QLIVector<QLIT>::GetKeyScale(const QLICommon::RunInfo &runInfo, LocalTensor<float> &kScaleUB,
                                                    int64_t batchId, int64_t startS2, int64_t getLen)
{
    // startS2一定能整除kCacheBlockSize_
    AscendC::DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
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
            copyInParams.blockLen = firstPartLen * sizeof(float);
            int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx);
            SetFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            WaitFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            AscendC::DataCopyPad(kScaleUB[(runInfo.loop % 2) * s2BaseSize_], kScaleGm[blockId * constInfo_.keyDequantScaleStride0 + startBlockTableOffset],
                                 copyInParams, padParams);
            startBlockTableIdx++;
            getLen = getLen - firstPartLen;
            resUbBaseOffset = firstPartLen;
        }
        int32_t getLoopNum = CeilDiv(getLen, kCacheBlockSize_);
        copyInParams.blockLen = kCacheBlockSize_ * sizeof(float);
        for (int32_t i = 0; i < getLoopNum; i++) {
            if (i == getLoopNum - 1) {
                copyInParams.blockLen = (getLen - i * kCacheBlockSize_) * sizeof(float);
            }
            int32_t blockId = blockTableGm.GetValue(blockTableBatchOffset + startBlockTableIdx + i);
            SetFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            WaitFlag<HardEvent::S_MTE2>(KSCALE_S_MTE2_EVENT);
            AscendC::DataCopyPad(kScaleUB[(runInfo.loop % 2) * s2BaseSize_ + resUbBaseOffset + i * kCacheBlockSize_], 
                                 kScaleGm[blockId * constInfo_.keyDequantScaleStride0],
                                 copyInParams, padParams);
        }
    } else {
        copyInParams.blockCount = 1;
        copyInParams.blockLen = getLen * sizeof(float);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        copyInParams.rsv = 0;
        AscendC::DataCopyPad(kScaleUB[(runInfo.loop % 2) * s2BaseSize_], kScaleGm[runInfo.tensorKeyScaleOffset], copyInParams, padParams);
    }
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ProcessVec1(const QLICommon::RunInfo &info)
{
    auto pingpong = (info.loop % 2);
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
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + pingpong);
    //weightsGm --> weightUB_ 
    int64_t weightGmOffset = info.tensorWeightsOffset + curAivS1Idx * kHeadNum_ * gSize_;
    DataCopyPadExtParams<float> padWeightsParams{false, 0, 0, 0};
    DataCopyExtParams qwDataCopyExtParams;
    qwDataCopyExtParams.blockCount = curAivS1ProcNum;
    qwDataCopyExtParams.blockLen = gSize_ * sizeof(float);
    qwDataCopyExtParams.srcStride = 0;
    qwDataCopyExtParams.dstStride = (UB_BANK_DEPTH_STRIDE - UB_BANK_STRIDE) / 32;
    DataCopyPad(weightUB_[pingpong * (UB_BANK_STRIDE / sizeof(float))], 
                weightsGm[weightGmOffset], qwDataCopyExtParams, padWeightsParams);

    //qScaleGm  -->  qScaleUB_
    DataCopyPadExtParams<float> padQScaleParams{false, 0, 0, 0};
    DataCopyPad(qScaleUB_[pingpong * (UB_BANK_STRIDE / sizeof(float))],
                qScaleGm[weightGmOffset], qwDataCopyExtParams, padQScaleParams);

    //kScaleGm  -->  kScaleUB_
    GetKeyScale(info, kScaleUB_, info.bIdx, curS2Idx, info.actualSingleProcessSInnerSize);
    SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT + pingpong);
    WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT + pingpong);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + pingpong);

    //CV同步
    CrossCoreWaitFlag<QLICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(QLICommon::ConstInfo::CROSS_CV_EVENT + info.loop % 2);   //V核等C核计算完mm1，mm1Res已搬运到UB

    static_assert(std::is_same_v<SCORE_T, uint16_t>);
    auto outBase = vec1OutUB_[pingpong * (UB_BANK_STRIDE / sizeof(SCORE_T))];
    auto weightBase = weightUB_[pingpong * (UB_BANK_STRIDE / sizeof(float))];
    auto qScaleBase = qScaleUB_[pingpong * (UB_BANK_STRIDE / sizeof(float))];
    auto kScaleBase = kScaleUB_[pingpong * s2BaseSize_];

    auto qkBase = resMm1UB_[pingpong * (UB_BANK_STRIDE / sizeof(QK_T))];
    auto qkVLstride = (UB_BANK_DEPTH_STRIDE / sizeof(QK_T)) / 2 * constInfo_.mBaseSize;
    vector1::BatchMulWeightAndReduceSum(outBase, UB_BANK_DEPTH_STRIDE / sizeof(SCORE_T),
                                        qkBase, qkVLstride, (uint32_t)(gSize_ * UB_BANK_DEPTH_STRIDE / sizeof(QK_T)), 
                                        weightBase, UB_BANK_DEPTH_STRIDE / sizeof(float),
                                        kScaleBase, (uint32_t)0,
                                        qScaleBase, UB_BANK_DEPTH_STRIDE / sizeof(float),
                                        gSize_, curAivS1ProcNum);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + pingpong);
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
    
    //gS1轴转s1轴，是否要向上对齐？
    s1Idx = s1Idx / gSize_;

    // block级别的s1索引转换成token级别的s1索引，向上取整确保block数不小于实际值
    num_prompt_k = CeilDiv(num_prompt_k, 128);

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
    const uint32_t INIT_BLOCKS = 4;
    // window_blocks: K方向后WINDOW_BLOCKS个块必选（索引validS2Len-WINDOW_BLOCKS ~ validS2Len-1）
    const uint32_t WINDOW_BLOCKS = 4;

    // 当前处理的GS1行起始索引（全局位置）
    int64_t curGS1Idx = info.gS1Idx * mBaseSize_;
    // 当前GS1行需要处理的数量（可能小于mBaseSize_，处理尾块）
    int64_t curGS1ProcNum = curGS1Idx + mBaseSize_ > info.actMBaseSize ? info.actMBaseSize % mBaseSize_ : mBaseSize_;
    // 当前AIV核处理的GS1起始索引（双核分工）
    int64_t curAivGS1Idx = curGS1Idx + (blockId_ % 2) * CeilDiv(curGS1ProcNum, 2);
    // 当前AIV核需要处理的GS1行数量
    int64_t curAivGS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curGS1ProcNum, 2) : curGS1ProcNum / 2;

    // S2 基本块信息
    uint32_t s2BlockStart = info.s2Idx * s2BaseSize_;
    uint32_t pingpong = info.loop % 2;
    uint32_t origActS2Size = info.actS2Size + WINDOW_BLOCKS;
    AscendC::DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    copyInParams.rsv = 0;
    // 输出搬运参数（从UB搬出到GM，仅最后一个基本块使用）
    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;

    // 计算当前S2方向的有效长度（基于第一个S1位置）
    uint32_t firstS2Len;
    if (constInfo_.attenMaskFlag) {
        // causal 场景：当前AIV核处理的第一个S1位置对应的s2合法长度
        // 算causal每行的s2实际长度actS2Size需要加上WINDOW_BLOCKS，最后又需要减去WINDOW_BLOCKS，可以抵消
        firstS2Len = origActS2Size - info.actS1Size + curAivGS1Idx / gSize_ + 1 - WINDOW_BLOCKS;
    } else {
        // 非 causal 场景：使用调度层传入的 actS2Size，已经减去了WINDOW_BLOCKS
        firstS2Len = info.actS2Size;
    }
    
    // 计算动态 topkCount 所需的块级常量（使用原始 actS2Size）
    int32_t kb_offset = (int32_t)origActS2Size - (int32_t)info.actS1Size;
    int32_t num_prompt_k = (int32_t)origActS2Size;  // 穿刺测试：假设 num_prompt_k = s2Size
    
    // 累计输出偏移（用于计算每行的 GM 输出地址，仅最后一个基本块使用）
    int64_t cumulativeOutputOffset = 0;
    
    // 常量定义：用于填充零值和无效索引
    uint16_t zero = 0;
    int32_t neg = -1;

    // 当前核处理的 S1 轴大小
    uint32_t curAivS1Size = CeilDiv(curAivGS1ProcNum, gSize_);
    uint32_t curS1RealS2Len = firstS2Len;
    // 外层循环：按 S1 位置遍历（同一个 S1 位置下的 G 个行共享相同的 s2ValidLen和topkCount）
    for (uint32_t s1Pos = 0; s1Pos < curAivS1Size; s1Pos++) {
        // 当前 S1 位置在 group 中的起始行索引
        uint32_t groupStartRowIdx = s1Pos * gSize_;

        // causal 场景：基于 S1 位置索引更新
        if (constInfo_.attenMaskFlag) {
            curS1RealS2Len = s1Pos + firstS2Len;
        }
        // 如果causal 场景的s2有效长度小于init_block + windows_block的长度，直接输出连续索引，跳过 TopK
        if (constInfo_.attenMaskFlag && (curS1RealS2Len <= INIT_BLOCKS)) {
            uint32_t curS1TotalS2Len = curS1RealS2Len + WINDOW_BLOCKS;
            uint32_t groupSize = Min((uint32_t)gSize_, (uint32_t)curAivGS1ProcNum - groupStartRowIdx);
            WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            // 循环外一次性生成索引
            AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(), 
                                    0, curS1TotalS2Len);
            // todo:这个同步需要确认是循环外还是循环内
            PipeBarrier<PIPE_V>();
            // 循环内只做 DataCopyPad
            for (uint32_t j = 0; j < groupSize; j++) {
                indiceLenLocal_.SetValue(s1Pos * gSize_ + j, curS1TotalS2Len);
                SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                copyOutParams.blockLen = (uint16_t)(curS1TotalS2Len * sizeof(int32_t));
                AscendC::DataCopyPad(indiceOutGm[info.indiceOutOffset + cumulativeOutputOffset],
                                        indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
                SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
                cumulativeOutputOffset += curS1TotalS2Len;
            }
            continue;
        }

        // 计算动态 topkCount（同一个 S1 位置下相同）
        uint32_t dynamicTopkCount = CalcDynamicTopkCount((uint32_t)(curAivGS1Idx / gSize_) + s1Pos, kb_offset, num_prompt_k);

        // 这个操作是否必要？
        int32_t topkSelectNum = Min(dynamicTopkCount, MAX_TOPK_COUNT);

        int32_t actualInit = INIT_BLOCKS;
        int32_t actualWindow =  WINDOW_BLOCKS;
        int32_t totalOutput = actualInit + topkSelectNum + actualWindow;

        // 参与 TopK 的有效数据不足 topkSelectNum，直接输出连续索引
        if (curS1RealS2Len - actualInit < topkSelectNum) {
            uint32_t groupSize = Min((uint32_t)gSize_, (uint32_t)curAivGS1ProcNum - groupStartRowIdx);
            for (uint32_t j = 0; j < groupSize; j++) {
                indiceLenLocal_.SetValue(s1Pos * gSize_ + j, totalOutput);

                int64_t gmBase = info.indiceOutOffset + cumulativeOutputOffset;
                // init 段：[0, actualInit-1]
                if (actualInit > 0) {
                    AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(), 0, (uint32_t)actualInit);
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                    WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                    copyOutParams.blockLen = (uint16_t)((uint32_t)actualInit * sizeof(int32_t));
                    AscendC::DataCopyPad(indiceOutGm[gmBase], indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
                    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
                }
                
                //  [actualInit, ..., curS1RealS2Len-1]
                AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(), actualInit, curS1RealS2Len - actualInit);
                PipeBarrier<PIPE_V>();
                WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
                SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                copyOutParams.blockLen = (uint16_t)((curS1RealS2Len - actualInit) * sizeof(int32_t));
                AscendC::DataCopyPad(indiceOutGm[gmBase + actualInit], indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
                SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
                
                // [curS1RealS2Len, topkSelectNum-1]
                int32_t topkPadLen = topkSelectNum - (curS1RealS2Len - actualInit);
                if (topkPadLen > 0) {
                    Duplicate(indicesOutLocal_.ReinterpretCast<int32_t>(), neg, (uint32_t)topkPadLen);
                    PipeBarrier<PIPE_V>();
                    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
                    SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                    WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                    copyOutParams.blockLen = (uint16_t)((uint32_t)topkPadLen * sizeof(int32_t));
                    AscendC::DataCopyPad(indiceOutGm[gmBase + curS1RealS2Len], indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
                    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
                }
                
                // window 段：[topkSelectNum, topkSelectNum+actualWindow-1]
                if (actualWindow > 0) {
                    AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(), topkSelectNum, (uint32_t)actualWindow);
                    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
                    SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                    WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                    copyOutParams.blockLen = (uint16_t)((uint32_t)actualWindow * sizeof(int32_t));
                    AscendC::DataCopyPad(indiceOutGm[gmBase + actualInit + topkSelectNum],
                                         indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
                    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
                }
                cumulativeOutputOffset += totalOutput;
            }
            continue;
        }
        // 动态更新 topK（buffer 布局已在 InitBuffers 按最大 topkCount_ 固定，此处仅更新标量）
        topkOp_.SetTopK(topkSelectNum);
        uint32_t topkSelectNumAlign256 = QLICommon::Align((uint32_t)topkSelectNum, (uint32_t)256);
        
        // 当前基本块的s2范围
        uint32_t s2BlockEnd = Min(s2BlockStart + s2BaseSize_, Max(curS1RealS2Len, 0));
        // 当前基本块中s2的有效长度
        uint32_t s2BlockValidLen = (s2BlockEnd > s2BlockStart) ? (s2BlockEnd - s2BlockStart) : 0;

        // 正常 TopK 流程
        uint32_t groupSize = Min((uint32_t)gSize_, (uint32_t)curAivGS1ProcNum - groupStartRowIdx);

        // 内层循环：遍历同一 S1 位置下的 G 个行
        for (uint32_t j = 0; j < groupSize; j++) {
            indiceLenLocal_.SetValue(s1Pos * gSize_ + j, totalOutput);

            uint32_t rowIdx = blockId_ % 2 * CeilDiv(curGS1ProcNum, 2) + groupStartRowIdx + j;

            // vec1OutUB_中当前行的基地址
            uint32_t vec1OutRowOffset = pingpong * (UB_BANK_STRIDE / sizeof(uint16_t))
                                        + rowIdx * (UB_BANK_DEPTH_STRIDE / sizeof(uint16_t));

            // todo:这两个同步的作用需要确认
            WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
            WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

            // ========== TopK计算 ==========
            if (isFirstS2InnerLoop) {
                // ===== 第一个基本块：独立TopK =====
                uint32_t inputLenAlign = QLICommon::Align(s2BlockValidLen, (uint32_t)256);
                Duplicate(mrgValueLocal_[s2BlockValidLen / 256 * 256], zero,
                          inputLenAlign - s2BlockValidLen / 256 * 256);
                SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);
                WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT);

                copyInParams.blockLen = s2BlockValidLen * sizeof(uint16_t); // byte
                AscendC::DataCopyPadExtParams<uint16_t> padParams{true, 0, 0, 0};
                // todo::这里vec1OutRowOffset需要确认是否是32B对齐
                AscendC::DataCopyPad(mrgValueLocal_, vec1OutUB_[vec1OutRowOffset], copyInParams, padParams);
                SetFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                WaitFlag<HardEvent::MTE2_V>(TOPK_MTE2_V_EVENT);
                if (actualInit > 0) {
                    Duplicate(mrgValueLocal_, zero, (uint32_t)actualInit);
                    PipeBarrier<PIPE_V>();
                }

                topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, inputLenAlign, 0, 1);

            } else {
                // ===== 非第一个基本块：合并上一轮TopK结果 + 当前基本块score =====

                AscendC::DataCopy(mrgValueLocal_, scoreOutLocal_, topkSelectNumAlign256);
                // 这里是否需要个同步
                PipeBarrier<PIPE_V>();
                copyInParams.blockLen = s2BlockValidLen * sizeof(uint16_t); // byte
                AscendC::DataCopyPadExtParams<uint16_t> padParams{true, 0, 0, 0};
                // todo::这里vec1OutRowOffset需要确认是否是32B对齐
                AscendC::DataCopyPad(mrgValueLocal_[topkSelectNumAlign256], vec1OutUB_[vec1OutRowOffset],
                                     copyInParams, padParams);
                
                // 两个dulicate实现topkSelectNum - topkSelectNumAlign256 非对齐补0
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

                uint32_t mergedInputLen = topkSelectNumAlign256 + s2BlockValidLen;
                uint32_t mergedInputAlign = QLICommon::Align(mergedInputLen, (uint32_t)256);
                if (mergedInputLen % 256 != 0) {
                    Duplicate(mrgValueLocal_[mergedInputLen / 256 * 256], zero,
                              mergedInputAlign - mergedInputLen / 256 * 256);
                }
                PipeBarrier<PIPE_V>();

                topkOp_(mrgValueLocal_, indicesOutLocal_, scoreOutLocal_, mergedInputAlign, 0, 1);

            }

            // ========== 索引重组和GM搬出（仅最后一个基本块）==========
            if (isLastS2InnerLoop) {
                int64_t gmBase = info.indiceOutOffset + cumulativeOutputOffset;

                AscendC::DataCopy(mrgValueLocal_.ReinterpretCast<int32_t>(),
                                  indicesOutLocal_.ReinterpretCast<int32_t>(),
                                  (uint32_t)topkSelectNum);
                PipeBarrier<PIPE_V>();

                if (actualInit > 0) {
                    AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(),
                                             (int32_t)zero, actualInit);
                    PipeBarrier<PIPE_V>();
                }

                SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
                SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);

                if (actualInit > 0) {
                    copyOutParams.blockLen = (uint16_t)((uint32_t)actualInit * sizeof(int32_t));
                    AscendC::DataCopyPad(indiceOutGm[gmBase],
                                         indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
                }

                copyOutParams.blockLen = (uint16_t)((uint32_t)topkSelectNum * sizeof(int32_t));
                AscendC::DataCopyPad(indiceOutGm[gmBase + actualInit],
                                     mrgValueLocal_.ReinterpretCast<int32_t>(), copyOutParams);

                SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

                if (actualWindow > 0) {
                    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

                    int32_t windowBase = curS1RealS2Len;
                    AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(),
                                             (int32_t)windowBase, actualWindow);
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
    // 批量写出 indice_len 到 GM
    if (curAivGS1ProcNum > 0) {
        AscendC::DataCopyParams indiceLenCopyParams;
        indiceLenCopyParams.blockCount = 1;
        indiceLenCopyParams.srcStride = 0;
        indiceLenCopyParams.dstStride = 0;
        indiceLenCopyParams.blockLen = (uint16_t)((uint32_t)curAivGS1ProcNum * sizeof(int32_t));
        AscendC::DataCopyPad(indiceLenGm[info.indiceLenOffset], indiceLenLocal_, indiceLenCopyParams);
    }
}

template <typename QLIT>
__aicore__ inline void QLIVector<QLIT>::ProcessDirectOutput(const QLICommon::RunInfo &info, uint32_t validS2Len)
{
    // 当前处理的gS1行起始索引（全局位置）
    int64_t curGS1Idx = info.gS1Idx * mBaseSize_;
    // 当前gS1行需要处理的数量（可能小于mBaseSize_，处理尾块）
    int64_t curGS1ProcNum = curGS1Idx + mBaseSize_ > info.actMBaseSize ? info.actMBaseSize % mBaseSize_ : mBaseSize_;
    // 当前AIV核处理的gS1起始索引（双核分工）
    int64_t curAivGS1Idx = curGS1Idx + (blockId_ % 2) * CeilDiv(curGS1ProcNum, 2);
    // 当前AIV核需要处理的gS1行数量
    int64_t curAivGS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curGS1ProcNum, 2) : curGS1ProcNum / 2;
    
    // 输出搬运参数
    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;
    
    // 累计输出偏移（causal 场景下每行长度可能不同）
    int64_t cumulativeOutputOffset = 0;
    
    // 外层循环：按 S1 位置遍历（同一个 S1 位置下的 G 个行共享相同的 rowOutputLen）
    uint32_t curAivS1Size = ((uint32_t)curAivGS1ProcNum + gSize_ - 1) / gSize_;
    for (uint32_t s1Pos = 0; s1Pos < curAivS1Size; s1Pos++) {
        uint32_t groupStartRowIdx = s1Pos * gSize_;
        uint32_t groupSize = Min((uint32_t)gSize_, (uint32_t)curAivGS1ProcNum - groupStartRowIdx);
        
        // 计算当前 S1 位置的输出长度（同一 S1 位置下共享）
        uint32_t rowOutputLen = validS2Len;
        if (constInfo_.attenMaskFlag) {
            int32_t originalRowValidS2Len = info.actS2Size - info.actS1Size + (int32_t)(curAivGS1Idx / gSize_) + s1Pos + 1;
            rowOutputLen = (uint32_t)Max(originalRowValidS2Len, 0);
        }
        
        // 循环外一次性生成索引
        if (rowOutputLen > 0) {
            AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(), 0, rowOutputLen);
            PipeBarrier<PIPE_V>();
            for (uint32_t j = 0; j < groupSize; j++) {
                indiceLenLocal_.SetValue(s1Pos * gSize_ + j, (int32_t)rowOutputLen);
            }
        }
        
        // 内层循环：按 G 行 DataCopyPad
        for (uint32_t j = 0; j < groupSize; j++) {
            if (rowOutputLen > 0) {
                int64_t gmBase = info.indiceOutOffset + cumulativeOutputOffset;
                copyOutParams.blockLen = (uint16_t)(rowOutputLen * sizeof(int32_t));
                SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                AscendC::DataCopyPad(indiceOutGm[gmBase], indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
                SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
                cumulativeOutputOffset += rowOutputLen;
            }
        }
    }
}
}  // namespace QLIKernel
#endif