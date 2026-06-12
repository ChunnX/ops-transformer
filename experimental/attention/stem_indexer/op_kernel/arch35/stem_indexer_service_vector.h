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

namespace SIKernel {
using namespace SICommon;
constexpr uint32_t TRUNK_LEN_16K = 16384;
constexpr uint32_t TRUNK_LEN_256 = 256;
template <typename SIT>
class SIVector {
public:
    // =================================类型定义区=================================
    static constexpr SI_LAYOUT Q_LAYOUT_T = SIT::layout;
    static constexpr SI_LAYOUT K_LAYOUT_T = SIT::keyLayout;

    using QK_T = float32_t;
    using SCORE_T = uint32_t;

    __aicore__ inline SIVector(){};
    __aicore__ inline void ProcessVec1(const SICommon::RunInfo &info);
    __aicore__ inline void ProcessTopK(const SICommon::RunInfo &info, bool isFirstS2InnerLoop, bool isLastS2InnerLoop);
    __aicore__ inline void ProcessDirectOutput(const SICommon::TempLoopInfo &tempLoopInfo, uint32_t gS1Idx);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct SICommon::ConstInfo &constInfo,
                                      const StemIndexerTilingData *__restrict tilingData);
    __aicore__ inline void InitVecInputTensor(GlobalTensor<float> vbiasGm, GlobalTensor<int32_t> numPromptTokensGm,
                                              GlobalTensor<int32_t> sparseIndicesGm, GlobalTensor<int32_t> sparseSeqLenGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void initS2LenToNeg(uint32_t actMBaseSize, uint64_t indiceOutOffset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();

protected:
    GlobalTensor<float> vBiasGm;
    GlobalTensor<int32_t> numPromptTokensGm;
    GlobalTensor<int32_t> sparseIndicesGm;
    GlobalTensor<int32_t> sparseSeqLenGm;
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
    static constexpr uint32_t MAX_TOPK_COUNT = 256;

private:
    __aicore__ inline uint32_t CalcDynamicTopkCount(uint32_t s1Idx, int32_t kbOffset, int32_t numPromptK);
    // ================================Local Buffer区====================================
    // tmp buff for vector
    TBuf<TPosition::VECCALC> resMm1Buf_;
    LocalTensor<QK_T> resMm1UB_;
    // tmp buff for vbias
    TBuf<TPosition::VECCALC> vBiasBuf_;
    LocalTensor<float> vBiasLocal_;
    //tmp buff for out
    TBuf<TPosition::VECCALC> outBuf_;
    LocalTensor<SCORE_T> vec1OutUB_;

    TBuf<TPosition::VECCALC> topkIndexBuf_;
    LocalTensor<uint32_t> topkIndexLocal_;

    TBuf<TPosition::VECCALC> topkValueBuf_;
    LocalTensor<uint32_t> topkValueLocal_;

    // tmp buff for topk
    TBuf<TPosition::VECCALC> mrgValueBuf_;
    LocalTensor<uint32_t> mrgValueLocal_;

    TBuf<TPosition::VECCALC> globalIndexBuf_;
    LocalTensor<uint32_t> globalIndexLocal_;

    TBuf<TPosition::VECCALC> indicesOutBuf_;
    LocalTensor<uint32_t> indicesOutLocal_;

    TBuf<TPosition::VECCALC> scoreOutBuf_;
    LocalTensor<SCORE_T> scoreOutLocal_;

    TBuf<TPosition::VECCALC> topkSharedTmpBuf_;
    LocalTensor<uint32_t> topkSharedTmpLocal_;

    TBuf<TPosition::VECCALC> outInvalidBuf_;
    LocalTensor<int32_t> outInvalidLocal_;

    TBuf<TPosition::VECCALC> indiceLenBuf_;
    LocalTensor<uint32_t> indiceLenLocal_;

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
    uint32_t trunkLen_ = 0;
    int32_t mBaseSize_ = 0;
    float alpha_ = 1.0f;
    float kBlockNumRateMedium_ = 0.2f;
    uint32_t kBlockNumBiasMedium_ = 30U;
    float kBlockNumRateLarge_ = 0.1f;
    uint32_t kBlockNumBiasLarge_ = 30U;
    uint32_t initialBlocks_ = 4U;
    uint32_t windowSize_ = 4U;
    struct SICommon::ConstInfo constInfo_;
    SIKernel::SITopk<SCORE_T> topkOp_;
};

template <typename SIT>
__aicore__ inline void SIVector<SIT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(resMm1Buf_, 2 * CeilDiv(constInfo_.mBaseSize, 2) * s2BaseSize_ * sizeof(QK_T));   //大小：2(开dB) * 2 * 64 * 128 * 4 = 128KB
    resMm1UB_ = resMm1Buf_.Get<QK_T>();// qk
    // TODO kbSize获取
    pipe->InitBuffer(vBiasBuf_, 2 * s2BaseSize_ * sizeof(float));   // 大小：2(开dB) * kbSize(256) * 4 = 4KB 
    vBiasLocal_ = vBiasBuf_.Get<float>();                      // vBias
    pipe->InitBuffer(outBuf_, 2 * CeilDiv(constInfo_.mBaseSize, 2) * s2BaseSize_ * sizeof(SCORE_T));      // 大小：2(开dB) * 2 * 128 * 4 = 2KB
    vec1OutUB_ = outBuf_.Get<SCORE_T>();// out

    // Topk
    pipe->InitBuffer(mrgValueBuf_, constInfo_.mBaseSize / 2 * (MAX_TOPK_COUNT + trunkLen_) * sizeof(uint32_t));     // 大小：(topkCountAlign256_ + 每次排序长度) * sizeof(SCORE_T)
    mrgValueLocal_ = mrgValueBuf_.Get<uint32_t>();

    pipe->InitBuffer(globalIndexBuf_, constInfo_.mBaseSize / 2 * (MAX_TOPK_COUNT + trunkLen_) * sizeof(uint32_t));     // 大小：(topkCountAlign256_ + 每次排序长度) * sizeof(SCORE_T)
    globalIndexLocal_ = globalIndexBuf_.Get<uint32_t>();

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
    pipe->InitBuffer(indiceLenBuf_, constInfo_.mBaseSize * sizeof(uint32_t));
    indiceLenLocal_ = indiceLenBuf_.Get<uint32_t>();
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::InitParams(const struct SICommon::ConstInfo &constInfo,
                                                   const StemIndexerTilingData *__restrict tilingData)
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
    mBaseSize_ = constInfo.mBaseSize;
    kCacheBlockSize_ = constInfo.kCacheBlockSize;
    maxBlockNumPerBatch_ = constInfo.maxBlockNumPerBatch;
    alpha_ = constInfo.alpha;
    kBlockNumRateMedium_ = constInfo.kBlockNumRateMedium;
    kBlockNumBiasMedium_ = constInfo.kBlockNumBiasMedium;
    kBlockNumRateLarge_ = constInfo.kBlockNumRateLarge;
    kBlockNumBiasLarge_ = constInfo.kBlockNumBiasLarge;
    initialBlocks_ = constInfo.initialBlocks;
    windowSize_ = constInfo.windowSize;
    blockId_ = GetBlockIdx();
    trunkLen_ = TRUNK_LEN_256;
    topkOp_.Init(MAX_TOPK_COUNT, trunkLen_);
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::InitVecInputTensor(GlobalTensor<float> vBiasGm, GlobalTensor<int32_t> numPromptTokensGm,
                                              GlobalTensor<int32_t> sparseIndicesGm, GlobalTensor<int32_t> sparseSeqLenGm)
{
    this->vBiasGm = vBiasGm;
    this->numPromptTokensGm = numPromptTokensGm;
    this->sparseIndicesGm = sparseIndicesGm;
    this->sparseSeqLenGm = sparseSeqLenGm;
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 0);
    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 1);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    SetFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);

    SetFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    SetFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 0);
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + 1);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 0);
    WaitFlag<HardEvent::MTE3_V>(VEC1_MTE3_V_EVENT + 1);

    WaitFlag<HardEvent::V_MTE2>(TOPK_V_MTE2_EVENT);
    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
    WaitFlag<HardEvent::V_MTE2>(V_MTE2_EVENT1);
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::CleanInvalidOutput(int64_t invalidS1Offset)
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
    AscendC::DataCopyPad(sparseIndicesGm[invalidS1Offset], outInvalidLocal_, dataCopyOutParams);
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::initS2LenToNeg(uint32_t actMBaseSize, uint64_t indiceOutOffset)
{
    int32_t neg = -1;

    GlobalTensor<int32_t> indiceOut = sparseIndicesGm[indiceOutOffset];
    AscendC::InitGlobalMemory(indiceOut, actMBaseSize * constInfo_.kSeqSize, neg);

    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::ProcessVec1(const SICommon::RunInfo &info)
{
    auto pingpong = (info.loop % 2);
    int64_t curS1Idx = info.gS1Idx * s1BaseSize_;
    int64_t gS1BasePerVecSize_ = CeilDiv(constInfo_.mBaseSize, 2); // 当前每个vec核上分到的mbase/2行
    int64_t curS2Idx = info.s2Idx * s2BaseSize_;	 
    int64_t curS1ProcNum = curS1Idx + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;	 
    int64_t curAivS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curS1ProcNum, 2) : curS1ProcNum / 2; 

    if (curAivS1ProcNum == 0) {
        CrossCoreWaitFlag<SICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(SICommon::ConstInfo::CROSS_CV_EVENT + pingpong);  // V核等C核计算完mm1，mm1Res已搬运到UB
        CrossCoreSetFlag<SICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(SICommon::ConstInfo::CROSS_VC_EVENT + pingpong);   // V核处理完，通知C核可以把mm1Res搬运到UB
        return;
    }
    WaitFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + pingpong);
    // vBiasGm --> vBiasLocal_ 搬运vBias
    int64_t vBiasGmOffset = info.tensorVBiasOffset + curS2Idx;
    DataCopyPadExtParams<float> padVBiasParams{false, 0, 0, 0};
    DataCopyExtParams vBiasDataCopyExtParams;
    vBiasDataCopyExtParams.blockCount = 1;
    vBiasDataCopyExtParams.blockLen = s2BaseSize_ * sizeof(float);
    vBiasDataCopyExtParams.srcStride = 0;
    vBiasDataCopyExtParams.dstStride = 0;
    DataCopyPad(vBiasLocal_[pingpong * s2BaseSize_], vBiasGm[vBiasGmOffset], vBiasDataCopyExtParams, padVBiasParams);

    SetFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT + pingpong);
    WaitFlag<HardEvent::MTE2_V>(VEC1_MTE2_V_EVENT + pingpong);

    //CV同步
    CrossCoreWaitFlag<SICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(SICommon::ConstInfo::CROSS_CV_EVENT + pingpong);   //V核等C核计算完mm1，mm1Res已搬运到U

    auto qkBase = resMm1UB_[pingpong * (gS1BasePerVecSize_ * s2BaseSize_)];
    auto outBase = mrgValueLocal_[MAX_TOPK_COUNT];
    auto vBiasBase = vBiasLocal_[pingpong * s2BaseSize_];

    vector1::MulRSquareAndAddVBiasVF(outBase, qkBase, vBiasBase, constInfo_.rSquare, gS1BasePerVecSize_, s2BaseSize_, (trunkLen_ + MAX_TOPK_COUNT));

    SetFlag<HardEvent::V_MTE2>(VEC1_V_MTE2_EVENT + pingpong);
    
    CrossCoreSetFlag<SICommon::ConstInfo::QLI_SYNC_MODE4, PIPE_V>(SICommon::ConstInfo::CROSS_VC_EVENT + pingpong);   //V核处理完，通知C核可以把mm1Res搬运到UB
}

template <typename SIT>
__aicore__ inline uint32_t SIVector<SIT>::CalcDynamicTopkCount(
    uint32_t s1Idx, int32_t kbOffset, int32_t numPromptK)
{
    // 常量定义
    const int32_t K_SMALL_SEQ_MAX = 56;
    const int32_t K_MEDIUM_SEQ_MAX = 160;

    // block级别的s1索引转换成token级别的s1索引，向上取整确保block数不小于实际值
    numPromptK = CeilDiv(numPromptK, 128);

    // 计算 kStart（分段函数）
    int32_t kStart;
    if (numPromptK < K_SMALL_SEQ_MAX) {
        kStart = K_SMALL_SEQ_MAX;
    } else if (numPromptK < K_MEDIUM_SEQ_MAX) {
        kStart = (int32_t)(numPromptK * kBlockNumRateMedium_ + kBlockNumBiasMedium_);
    } else {
        kStart = (int32_t)(numPromptK * kBlockNumRateLarge_ + kBlockNumBiasLarge_);
    }
    
    // 计算 s1Pos 和 decayLen
    int32_t s1Pos = (int32_t)s1Idx + kbOffset;
    int32_t decayLen = numPromptK - kStart;
    
    // 边界条件：s1Pos < kStart 或 decayLen < 1 时，直接返回 kStart
    if (s1Pos < kStart || decayLen < 1) {
        return (uint32_t)kStart;
    }
    
    // 计算 k_end
    int32_t k_end = (int32_t)(kStart * alpha_);
    
    // 计算插值系数 t
    float t = (float)(s1Pos - kStart) / (float)(decayLen - 1);
    
    // 计算 dynamicTopkCount
    uint32_t dynamicTopkCount = (uint32_t)((float)kStart + t * (float)(k_end - kStart));
    
    // 限制范围：下限为 1，上限为 kStart
    dynamicTopkCount = dynamicTopkCount < 1 ? 1 : (dynamicTopkCount > (uint32_t)kStart ? (uint32_t)kStart : dynamicTopkCount);
    
    return dynamicTopkCount;
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::ProcessTopK(
    const SICommon::RunInfo &info, bool isFirstS2InnerLoop, bool isLastS2InnerLoop)
{
    // 当前处理的GS1行起始索引（全局位置）
    int64_t curGS1Idx = info.gS1Idx * mBaseSize_;
    // 当前GS1行需要处理的数量（可能小于mBaseSize_，处理尾块）
    int64_t curGS1ProcNum = info.actMBaseSize;
    // 当前AIV核处理的GS1起始索引（双核分工）
    int64_t curAivGS1Idx = curGS1Idx + (blockId_ % 2) * CeilDiv(curGS1ProcNum, 2);
    // 当前AIV核需要处理的GS1行数量
    int64_t curAivGS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curGS1ProcNum, 2) : curGS1ProcNum / 2;

    // S2 基本块信息
    uint32_t s2BlockStart = info.s2Idx * s2BaseSize_;

    AscendC::DataCopyExtParams copyInParams;
    copyInParams.blockCount = 1;
    copyInParams.srcStride = 0;
    copyInParams.dstStride = 0;
    copyInParams.rsv = 0;
    // indice输出搬运参数
    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;
    // indiceLen搬出参数
    AscendC::DataCopyParams indiceLenCopyParams;

    // 计算当前S2方向的有效长度（基于第一个S1位置）
    int32_t firstS2Len = static_cast<int32_t>(info.actS2Size) - static_cast<int32_t>(windowSize_);
    if (constInfo_.attenMaskFlag) {
        // causal 场景：当前AIV核处理的第一个S1位置对应的s2合法长度
        firstS2Len = static_cast<int32_t>(info.actS2Size) - static_cast<int32_t>(info.actS1Size) +
                     static_cast<int32_t>(curAivGS1Idx % info.actS1Size) + 1 -
                     static_cast<int32_t>(windowSize_);
    }
    firstS2Len = firstS2Len > 0 ? firstS2Len : 0;
    
    // 计算动态 topkCount 所需的块级常量（使用原始 actS2Size）
    int32_t kbOffset = (int32_t)info.actS2Size - (int32_t)info.actS1Size;
    int32_t numPromptK = (int32_t)info.numPromptTokens;
    
    // 常量定义：用于填充零值和无效索引
    uint16_t zero = 0;
    int32_t neg = -1;

    // 当前核处理的 S1 轴大小
    uint32_t curAivS1Size = CeilDiv(curAivGS1ProcNum, gSize_); // 48 / 32 向上取整 = 2
    uint32_t curS1RealS2Len = static_cast<uint32_t>(firstS2Len);
    // 外层循环：按 S1 位置遍历（同一个 S1 位置下的 G 个行共享相同的 s2Len和topkCount）
    for (uint32_t s1Pos = 0; s1Pos < curAivS1Size; s1Pos++) {

        uint32_t globalS1Pos = curAivGS1Idx % info.actS1Size + s1Pos;

        // causal 场景：基于 S1 位置索引更新
        if (constInfo_.attenMaskFlag) {
            int32_t curS1RealS2LenTmp = firstS2Len + static_cast<int32_t>(s1Pos);
            curS1RealS2Len = static_cast<uint32_t>(curS1RealS2LenTmp > 0 ? curS1RealS2LenTmp : 0);
        }

        // 短路1：如果causal 场景的s2有效长度小于init_block + windows_block的长度，直接输出连续索引，跳过 TopK
        if (constInfo_.attenMaskFlag && (curS1RealS2Len <= initialBlocks_)) {
            if (!isFirstS2InnerLoop) continue;  // 非首块跳过，避免重复写入GM
            uint32_t curS1TotalS2Len = curS1RealS2Len + windowSize_;
            WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

            // 搬出indiceLen到gm
            Duplicate(indiceLenLocal_, curS1TotalS2Len, gSize_);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            indiceLenCopyParams.blockCount = gSize_;
            indiceLenCopyParams.blockLen = (uint16_t)(sizeof(uint32_t));
            indiceLenCopyParams.srcStride = 0;
            indiceLenCopyParams.dstStride = (constInfo_.qSeqSize - 1) * sizeof(uint32_t);
            AscendC::DataCopyPad(sparseSeqLenGm[info.indiceLenOffset + globalS1Pos], indiceLenLocal_.ReinterpretCast<int32_t>(), indiceLenCopyParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

            // 搬出indice到gm
            AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(), 
                                    0, curS1TotalS2Len);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            copyOutParams.blockCount = gSize_;
            copyOutParams.blockLen = (uint16_t)(curS1TotalS2Len * sizeof(int32_t));
            copyOutParams.srcStride = 0;
            copyOutParams.dstStride = (uint16_t)((constInfo_.qSeqSize * constInfo_.kSeqSize - curS1TotalS2Len) * sizeof(int32_t));
            AscendC::DataCopyPad(sparseIndicesGm[info.indiceOutOffset + globalS1Pos * constInfo_.kSeqSize],
                                indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

            continue;
        }

        // 计算动态 topkCount（同一个 S1 位置下相同）
        uint32_t dynamicTopkCount = CalcDynamicTopkCount(globalS1Pos, kbOffset, numPromptK);

        uint32_t topkSelectNum = Min(dynamicTopkCount, MAX_TOPK_COUNT);

        uint32_t actualInit = initialBlocks_;
        uint32_t actualWindow =  windowSize_;
        uint32_t totalOutput = actualInit + topkSelectNum + actualWindow;

        // 短路2：参与 TopK 的有效数据不足 topkSelectNum，直接输出连续索引
        if (curS1RealS2Len - actualInit < topkSelectNum) {
            if (!isFirstS2InnerLoop) continue;  // 非首块跳过，避免重复写入GM
            WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

            // 搬出indiceLen到gm
            Duplicate(indiceLenLocal_, totalOutput, gSize_);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            indiceLenCopyParams.blockLen = (uint16_t)(sizeof(uint32_t));
            indiceLenCopyParams.blockCount = gSize_;
            indiceLenCopyParams.srcStride = 0;
            indiceLenCopyParams.dstStride = (constInfo_.qSeqSize - 1) * sizeof(uint32_t);
            AscendC::DataCopyPad(sparseSeqLenGm[info.indiceLenOffset + globalS1Pos], indiceLenLocal_.ReinterpretCast<int32_t>(), indiceLenCopyParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_V_MTE3_EVENT);
            // 计算 baseOffset
            int64_t baseOffset = info.indiceOutOffset + globalS1Pos * constInfo_.kSeqSize;
            uint32_t fullStride = constInfo_.qSeqSize * constInfo_.kSeqSize;

            // 第一次copyout，[0 ~ curS1RealS2Len-1]
            AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(), (int32_t)0, (uint32_t)curS1RealS2Len);
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            copyOutParams.blockCount = gSize_;
            copyOutParams.blockLen = (uint16_t)((uint32_t)curS1RealS2Len * sizeof(int32_t));
            copyOutParams.srcStride = 0;
            copyOutParams.dstStride = (uint16_t)((fullStride - (uint32_t)curS1RealS2Len) * sizeof(int32_t));
            AscendC::DataCopyPad(sparseIndicesGm[baseOffset], indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            // 第二次copyout， [topkSelectNum, topkSelectNum+actualWindow-1]
            if (actualWindow > 0) {
                AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(), (int32_t)topkSelectNum, (uint32_t)actualWindow);
                SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                copyOutParams.blockCount = gSize_;
                copyOutParams.blockLen = (uint16_t)((uint32_t)actualWindow * sizeof(int32_t));
                copyOutParams.srcStride = 0;
                copyOutParams.dstStride = (uint16_t)((fullStride - (uint32_t)actualWindow) * sizeof(int32_t));
                AscendC::DataCopyPad(sparseIndicesGm[baseOffset + actualInit + topkSelectNum], indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
                SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
            }
            continue;
        }
        
        // 动态更新 topK（buffer 布局已在 InitBuffers 按最大 topkCount_ 固定，此处仅更新标量）
        topkOp_.SetTopK(topkSelectNum);
        uint32_t topkSelectNumAlign256 = SICommon::Align((uint32_t)topkSelectNum, (uint32_t)256);
        
        // 当前基本块的s2范围
        uint32_t s2BlockEnd = Min(s2BlockStart + s2BaseSize_, Max(curS1RealS2Len, 0));
        // 当前基本块中s2的有效长度
        uint32_t s2BlockValidLen = (s2BlockEnd > s2BlockStart) ? (s2BlockEnd - s2BlockStart) : 0;

        // 正常 TopK 流程
        // 如果mBaseSize 无法整除 g，后一部分g处理的s1Num比前一部分g少一个，所以这里搬出g的数量需要根据s1位置变化
        int32_t curGsize = gSize_;
        if (curAivGS1ProcNum % gSize_ != 0) {
            if (blockId_ % 2 == 0) {
                curGsize = (s1Pos == curAivS1Size - 1) ? curAivGS1ProcNum % gSize_ : gSize_;
            } else {
                curGsize = (s1Pos == 0) ? (gSize_ - curAivGS1ProcNum % gSize_) : gSize_;
            }
        }
        // 正常topk的indiceLen搬出
        Duplicate(indiceLenLocal_, totalOutput, curGsize);
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        indiceLenCopyParams.blockLen = (uint16_t)(sizeof(uint32_t));
        indiceLenCopyParams.blockCount = curGsize;
        indiceLenCopyParams.srcStride = 0;
        indiceLenCopyParams.dstStride = (constInfo_.qSeqSize - 1) * sizeof(uint32_t);
        AscendC::DataCopyPad(sparseSeqLenGm[info.indiceLenOffset + globalS1Pos], indiceLenLocal_.ReinterpretCast<int32_t>(), indiceLenCopyParams);
        SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        uint32_t copyCount = s2BlockValidLen;
        uint64_t copyMask = (copyCount <= 64) ? copyCount : 64;
        uint8_t copyRepeat = (copyCount + 63) / 64;
        AscendC::CopyRepeatParams copyRepeatParams{1, 1, 8, 8};
        // 内层循环：遍历同一 S1 位置下的 G 个行
        for (uint32_t gIdx = 0; gIdx < curGsize; gIdx++) {

            uint32_t mInnerIdx = gIdx * s1BaseSize_ + s1Pos;
            uint32_t mInnerOffset = mInnerIdx * (MAX_TOPK_COUNT + trunkLen_);

            // todo:确认这个同步的必要性
            WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);

            uint32_t s2BlockValidLenAlign256 = SICommon::Align(s2BlockValidLen, trunkLen_);

            if (isFirstS2InnerLoop) {
                // ===== 第一个基本块：独立TopK =====
                // mrgValueLocal_的topk的位置在首块需要刷0
                Duplicate(mrgValueLocal_[mInnerOffset], (uint32_t)zero, topkSelectNumAlign256);
            } else {
                // ===== 非第一个基本块：合并上一轮TopK结果的score + 当前基本块score =====
                // 两个dulicate实现topkSelectNum - topkSelectNumAlign256 非对齐补0
                if (topkSelectNumAlign256 != (uint32_t)topkSelectNum) {
                    uint64_t mask[1];
                    mask[0] = ~0;
                    mask[0] = mask[0] << (topkSelectNum % 64);
                    Duplicate(mrgValueLocal_[mInnerOffset + topkSelectNum / 64 * 64], (uint32_t)zero, mask, 1, 1, 0);
                    PipeBarrier<PIPE_V>();
                    Duplicate(mrgValueLocal_[mInnerOffset + topkSelectNum / 64 * 64 + 64],(uint32_t)zero,
                              topkSelectNumAlign256 - (topkSelectNum / 64 * 64 + 64));
                }
            }
            if (s2BlockValidLenAlign256 != (uint32_t)s2BlockValidLen) {
                uint64_t mask[1];
                mask[0] = ~0;
                mask[0] = mask[0] << (s2BlockValidLen % 64);
                PipeBarrier<PIPE_V>();
                Duplicate(mrgValueLocal_[mInnerOffset + topkSelectNumAlign256 + s2BlockValidLen / 64 * 64], (uint32_t)zero, mask, 1, 1, 0);
                PipeBarrier<PIPE_V>();
                Duplicate(mrgValueLocal_[mInnerOffset + topkSelectNumAlign256 + s2BlockValidLen / 64 * 64 + 64], (uint32_t)zero,
                            s2BlockValidLenAlign256 - (s2BlockValidLen / 64 * 64 + 64));
            }
            if (isFirstS2InnerLoop) {
                // 如果是首块，需要把initBlock的位置刷0
                if (actualInit > 0) {
                    PipeBarrier<PIPE_V>();
                    Duplicate(mrgValueLocal_[mInnerOffset + topkSelectNumAlign256], (uint32_t)zero, (uint32_t)actualInit);
                }
            }
            PipeBarrier<PIPE_V>();
            topkOp_(mrgValueLocal_[mInnerOffset], indicesOutLocal_, scoreOutLocal_, topkSelectNumAlign256 + s2BlockValidLenAlign256, info.s2Idx, info.s2LoopEnd);

            copyCount = topkSelectNumAlign256;
            copyMask = (copyCount <= 64) ? copyCount : 64;
            copyRepeat = (copyCount + 63) / 64;
            AscendC::Copy(mrgValueLocal_[mInnerOffset], scoreOutLocal_, copyMask, copyRepeat, copyRepeatParams);
            AscendC::Copy(globalIndexLocal_[mInnerOffset], indicesOutLocal_, copyMask, copyRepeat, copyRepeatParams);
            // ========== 索引重组和GM搬出（仅最后一个基本块）==========
            if (isLastS2InnerLoop) {
                int64_t rowOffset = ((int64_t)gIdx * constInfo_.qSeqSize + globalS1Pos) * constInfo_.kSeqSize;
                int64_t gmBase = info.indiceOutOffset + rowOffset;

                if (actualInit > 0) {
                    PipeBarrier<PIPE_V>();
                    AscendC::CreateVecIndex(mrgValueLocal_[mInnerOffset].ReinterpretCast<int32_t>(),
                                             (int32_t)zero, actualInit);
                    SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                    WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                    copyOutParams.blockLen = (uint16_t)((uint32_t)actualInit * sizeof(int32_t));
                    AscendC::DataCopyPad(sparseIndicesGm[gmBase],
                                         mrgValueLocal_[mInnerOffset].ReinterpretCast<int32_t>(), copyOutParams);
                }

                SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                copyOutParams.blockLen = (uint16_t)((uint32_t)topkSelectNum * sizeof(int32_t));
                AscendC::DataCopyPad(sparseIndicesGm[gmBase + actualInit],
                                     globalIndexLocal_[mInnerOffset].ReinterpretCast<int32_t>(), copyOutParams);
                SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
                
                if (actualWindow > 0) {
                    WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
                    // 把之前在s2Len尾部减去的windowsSize拼回来
                    int32_t windowBase = curS1RealS2Len;
                    AscendC::CreateVecIndex(mrgValueLocal_[mInnerOffset].ReinterpretCast<int32_t>(),
                                             (int32_t)windowBase, actualWindow);
                    SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                    WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
                    copyOutParams.blockLen = (uint16_t)((uint32_t)actualWindow * sizeof(int32_t));
                    AscendC::DataCopyPad(sparseIndicesGm[gmBase + actualInit + topkSelectNum],
                                         mrgValueLocal_[mInnerOffset].ReinterpretCast<int32_t>(), copyOutParams);
                    SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
                }
            }
        }
    }
}

template <typename SIT>
__aicore__ inline void SIVector<SIT>::ProcessDirectOutput(const SICommon::TempLoopInfo &tempLoopInfo, uint32_t gS1Idx)
{
    uint32_t actMBaseSize = tempLoopInfo.actMBaseSize;
    uint32_t actS1Size = tempLoopInfo.actS1Size;
    uint32_t actS2Size = tempLoopInfo.actS2Size;
    uint64_t qFlatBase = (uint64_t)tempLoopInfo.bIdx * constInfo_.qSeqSize * constInfo_.qHeadNum
                       + (uint64_t)tempLoopInfo.n2Idx * constInfo_.gSize * constInfo_.qSeqSize;
    uint32_t indiceLenOffset = (uint32_t)qFlatBase;
    uint64_t indiceOutOffset = qFlatBase * constInfo_.kSeqSize;

    initS2LenToNeg(actMBaseSize, indiceOutOffset);

    // 当前处理的gS1行起始索引（全局位置）
    int64_t curGS1Idx = gS1Idx * mBaseSize_;
    // 当前gS1行需要处理的数量（可能小于mBaseSize_，处理尾块）
    int64_t curGS1ProcNum = actMBaseSize;
    // 当前AIV核处理的gS1起始索引（双核分工）
    int64_t curAivGS1Idx = curGS1Idx + (blockId_ % 2) * CeilDiv(curGS1ProcNum, 2);
    // 当前AIV核需要处理的gS1行数量
    int64_t curAivGS1ProcNum = (blockId_ % 2 == 0) ? CeilDiv(curGS1ProcNum, 2) : curGS1ProcNum / 2;
    
    // 输出搬运参数
    AscendC::DataCopyParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;
    
    uint32_t curS1RealS2Len = actS2Size;
    // 外层循环：按 S1 位置遍历（同一个 S1 位置下的 G 个行共享相同的 S2Len）
    uint32_t curAivS1Size = CeilDiv(curAivGS1ProcNum, gSize_);
    for (uint32_t s1Pos = 0; s1Pos < curAivS1Size; s1Pos++) {
        uint32_t globalS1Pos = gS1Idx * s1BaseSize_ + s1Pos;
        
        if (constInfo_.attenMaskFlag) {
            int32_t curS1RealS2LenTmp = static_cast<int32_t>(actS2Size) - static_cast<int32_t>(actS1Size) +
                                        static_cast<int32_t>(curAivGS1Idx % actS1Size) +
                                        static_cast<int32_t>(s1Pos) + 1;
            curS1RealS2Len = static_cast<uint32_t>(curS1RealS2LenTmp > 0 ? curS1RealS2LenTmp : 0);
        }
        // 如果mBaseSize 无法整除 g，后一部分g处理的s1Num比前一部分g少一个，所以这里搬出g的数量需要根据s1位置变化
        int32_t curGsize = gSize_;
        if (curAivGS1ProcNum % gSize_ != 0) {
            if (blockId_ % 2 == 0) {
                curGsize = s1Pos == curAivS1Size - 1 ? curAivGS1ProcNum % gSize_ : gSize_;
            } else {
                curGsize = s1Pos == 0 ? (gSize_ - curAivGS1ProcNum % gSize_) : gSize_;
            }
        }
        WaitFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        Duplicate(indiceLenLocal_, curS1RealS2Len, curGsize);
        SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
        AscendC::DataCopyParams indiceLenCopyParams;
        indiceLenCopyParams.blockLen = (uint16_t)(sizeof(uint32_t));
        indiceLenCopyParams.blockCount = curGsize;
        indiceLenCopyParams.srcStride = 0;
        indiceLenCopyParams.dstStride = (constInfo_.qSeqSize - 1) * sizeof(uint32_t);
        AscendC::DataCopyPad(sparseSeqLenGm[indiceLenOffset + globalS1Pos], indiceLenLocal_.ReinterpretCast<int32_t>(), indiceLenCopyParams);
        
        uint32_t fullStride = constInfo_.qSeqSize * constInfo_.kSeqSize;
        int64_t baseOffset = indiceOutOffset + globalS1Pos * constInfo_.kSeqSize;
        
        if (curS1RealS2Len > 0) {
            AscendC::CreateVecIndex(indicesOutLocal_.ReinterpretCast<int32_t>(), (int32_t)0, curS1RealS2Len);
            // 有效数据：一次搬出 curGsize 个块
            SetFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            WaitFlag<HardEvent::V_MTE3>(TOPK_V_MTE3_EVENT);
            copyOutParams.blockCount = curGsize;
            copyOutParams.blockLen = (uint16_t)(curS1RealS2Len * sizeof(int32_t));
            copyOutParams.srcStride = 0;
            copyOutParams.dstStride = (uint16_t)((fullStride - curS1RealS2Len) * sizeof(int32_t));
            AscendC::DataCopyPad(sparseIndicesGm[baseOffset], indicesOutLocal_.ReinterpretCast<int32_t>(), copyOutParams);
            SetFlag<HardEvent::MTE3_V>(TOPK_MTE3_V_EVENT);
        }
    }
}
}  // namespace SIKernel
#endif
