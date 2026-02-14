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
 * \file compressor_block_vec_perf.h
 * \brief
 */

#ifndef COMPRESSOR_BLOCK_VEC_PERF_H
#define COMPRESSOR_BLOCK_VEC_PERF_H

#include "../compressor_comm.h"
#include "../compressor_tools.h"
#include "arch32.h"

using namespace AscendC;

namespace Compressor 
{
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;

template <typename COMP> class CompressorBlockVectorPerf {
public:
    static constexpr bool X_DTYPE = COMP::xDtype == X_DTYPE::BF16;
    static constexpr uint64_t BLOCK_VEC_BASE_BUFFER_SIZE = 32 * 1024; // 32k
    static constexpr uint32_t DATABLOCK_BYTES = 32;
    static constexpr float FLOAT_ZERO = 0;
    // =================================类型定义区=================================
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using X_T = typename AscendC::Conditional<X_DTYPE, bfloat16_t, half>::type;

    __aicore__ inline CompressorBlockVectorPerf(){};
    // =================================设置参数=================================
    __aicore__ inline void InitParams(const ConstInfo &constInfo, const CompressorTools<COMP> &tools);
    __aicore__ inline void Init( 
        __gm__ uint8_t *x,
        __gm__ uint8_t *wKv,
        __gm__ uint8_t *wGate,
        __gm__ uint8_t *kvState,
        __gm__ uint8_t *scoreState,
        __gm__ uint8_t *ape,
        __gm__ uint8_t *normWeight,
        __gm__ uint8_t *ropeSin,
        __gm__ uint8_t *ropeCos,
        __gm__ uint8_t *kvBlockTable,
        __gm__ uint8_t *scoreBlockTable,
        __gm__ uint8_t *cuSeqlens,
        __gm__ uint8_t *seqUsed,
        __gm__ uint8_t *startPos,
        __gm__ uint8_t *cmpKvOut);
    // =================================资源管理=================================
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    // =================================执行计算=================================
    __aicore__ inline void ComputeVec1(const Compressor::RunInfo &info);
    __aicore__ inline uint32_t GetBasicNum();
    __aicore__ inline uint32_t GetScSize();
    __aicore__ inline void GetScIdxInfo(uint32_t bStart, uint32_t scStart, uint32_t dealScSize, uint32_t v2TcStart, uint32_t v2TcEnd, 
                                                            uint32_t &outputBStart, uint32_t &outputSStart, uint32_t &outputScSize);
    __aicore__ inline void CalcScEndIdx(uint32_t bStart, uint32_t scStart, uint32_t dealScSize, uint32_t &bEnd, uint32_t &scEnd);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<T> preMm1ResGm, GlobalTensor<T> curMm1ResGm, GlobalTensor<T> vec1ResGm, GlobalTensor<T> vec2InputGm);
    __aicore__ inline void ComputeVec2(const Compressor::Vec2RunInfo &info);
    __aicore__ inline void WriteToCacheState(const GlobalTensor<T> &state, const GlobalTensor<int32_t> &blockTableGm,
        const LocalTensor<T> &input, uint32_t batchIdx, uint32_t startSeqIdx, uint32_t endSeqIdx, uint32_t dStart, uint32_t dDealSize);
    __aicore__ inline void ReadFromCacheState(const LocalTensor<T> &output, const GlobalTensor<T> &state, const GlobalTensor<int32_t> &blockTableGm,
        uint32_t batchIdx, uint32_t startSeqIdx, uint32_t endSeqIdx, uint32_t dStart, uint32_t dDealSize);

protected:
    GlobalTensor<T> vec1ResGm_;
    GlobalTensor<T> vec2InputGm_;
    GlobalTensor<T> preMm1ResGm_;
    GlobalTensor<T> curMm1ResGm_;

private:
    __aicore__ inline uint32_t GetSeqUsed(uint32_t bIdx);
    __aicore__ inline uint32_t GetStartPos(uint32_t bIdx);
    __aicore__ inline uint32_t GetSeqLength(uint32_t bIdx);
    __aicore__ inline uint32_t GetBsLength(uint32_t index);
    __aicore__ inline void CalcGlobalScStart(uint32_t bStart, uint32_t scStart, uint32_t bEnd, uint32_t scEnd, uint64_t &globalScStart);
    __aicore__ inline void UpdateOutputIdx(uint32_t &outputBStart, uint32_t &outputSStart, uint32_t &dealScSize, uint32_t &curDealScSize);
    __aicore__ inline void DealVec1BaseBlock(const RunInfo &info, CompressorVec1SliceIterator<COMP, false> &sliceIterator, uint32_t dStartIdx, uint32_t dDealSize);
    __aicore__ inline void CopyInApe(const LocalTensor<T> &apeUb,uint32_t dStartIdx, uint32_t dDealSize);
    __aicore__ inline void AddApeToScore(const LocalTensor<T> &scoreLocal, const LocalTensor<T> &apeUb, uint32_t tcDealSize, uint32_t dDealSize);
    __aicore__ inline void DataCopyAlignUbToUb(const LocalTensor<T> dstLocal, const LocalTensor<T> srcLocal,
        uint32_t copyRowCount, uint32_t copyColCount, uint32_t srcSingleRowCount, uint32_t dstSingleRowCount);
    __aicore__ inline void DataCopyAlignGmToUb(const LocalTensor<T> dstLocal, const GlobalTensor<T> srcGm,
        uint32_t copyRowCount, uint32_t copyColCount, uint32_t srcSingleRowCount, uint32_t dstSingleRowCount);
    __aicore__ inline void OverLap(const LocalTensor<T> dstLocal, const LocalTensor<T> srcLocal, const Vec1SliceInfo &sliceInfo,
                                        uint32_t dStartIdx, uint32_t dDealSize);
    __aicore__ inline void FromWokrSpaceToUb(const LocalTensor<T> dstLocal,
        const Vec1SliceInfo &sliceInfo, const StatisticInfo &statisticInfo, uint32_t dStartIdx, uint32_t dDealSize);
    __aicore__ inline void UpdateState(const LocalTensor<T> kvLocal, const LocalTensor<T> scoreLocal, 
        const Vec1SliceInfo &sliceInfo, uint32_t dStartIdx, uint32_t dDealSize);
    __aicore__ inline void SoftmaxDN(const LocalTensor<T> &scoreLocal, const LocalTensor<T> &tmpUb, uint32_t tcDealSize, uint32_t dDealSize);
    __aicore__ inline void KvMulReduceScore(const LocalTensor<T> &kvLocal, const LocalTensor<T>& scoreLocal, const LocalTensor<T> &dstLocal, const LocalTensor<T> &tmpUb, uint32_t tcDealSize, uint32_t dDealSize);
    __aicore__ inline void CopyOutVec1Res(const RunInfo &info, const LocalTensor<T> comperssoredUb, uint32_t compressTcSize, uint32_t dStartIdx, uint32_t dDealSize);

    __aicore__ inline void SplitCoreV2(const Compressor::Vec2RunInfo& info);
    __aicore__ inline void CopyFinalResultOut(const Compressor::Vec2RunInfo& info, const LocalTensor<X_T> &cmpKvOutUb,
        uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void DealVec2BaseBlock(const Compressor::Vec2RunInfo& info, uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void MultRowRmsNorm(const LocalTensor<T> &normResUb, const LocalTensor<T> &vec1ResUb, const LocalTensor<T> &normWeightUb,
    const LocalTensor<T> &tempLocal, const Compressor::Vec2RunInfo& info, uint32_t dealRowCount);
    __aicore__ inline void CalRope(const Compressor::Vec2RunInfo& info, const LocalTensor<X_T> &outputUb, const LocalTensor<T> &normResUb,
        uint32_t startRow, uint32_t dealRowCount);
    __aicore__ inline void SaveState(const LocalTensor<T> kvLocal, const LocalTensor<T> scoreLocal,
        const Vec1SliceInfo &sliceInfo, uint32_t dStartIdx, uint32_t dDealSize);
    __aicore__ inline void ReadState(const LocalTensor<T> kvLocal, const LocalTensor<T> scoreLocal,
        const Vec1SliceInfo &sliceInfo, uint32_t dStartIdx, uint32_t dDealSize);
    uint32_t cmpRatio_ = 0U;
    uint32_t coff_ = 0U;
    uint32_t curStartPos_ = 0;
    uint32_t curActSeqLength_ = 0;
    uint32_t compressedCnt_ = 0;
    uint32_t v1SplitSize_ = 0;
    uint32_t v1ScLoopTimes_ = 0;
    uint32_t v1DLoopTimes_ = 0;
    uint32_t dealTcNum_ = 0;
    bool apeIsLoad_ = false;
    bool isExistSeqUsed = false;
    bool isExistStartPos = false;
    // vec2
    uint32_t v2MBaseSize = 16; // Tc块数量：32 * 1024 / (512 * 4)
    uint32_t v2TcStartIdx = 0U;
    uint32_t v2TcEndIdx = 0U;
    uint32_t mmResColSize_ = 128;
    int64_t vec1ResGmStart = 0U;
    uint32_t usedCoreNum = 16;
    uint32_t OutputBStartIdx, OutputSStartIdx, OutputSize;
    uint32_t mmResBaseOffset_ = 0;
    CompressorTools<COMP> tools_;
    ConstInfo constInfo_ = {};
    MSplitInfo mSplitInfo = {};
    GlobalTensor<int32_t> startPosGm_;
    GlobalTensor<int32_t> cuSeqlensGm_;
    GlobalTensor<int32_t> sequsedGm_;
    GlobalTensor<int32_t> kvBlockTableGm_;
    GlobalTensor<int32_t> scoreBlockTableGm_;
    GlobalTensor<T> kvStateGm_;
    GlobalTensor<T> scoreStateGm_;
    GlobalTensor<T> apeGm_;
    GlobalTensor<X_T> normWeightGm_;
    GlobalTensor<X_T> ropeSinGm_;
    GlobalTensor<X_T> ropeCosGm_;
    GlobalTensor<X_T> cmpKvOutGm_;

    // ================================Local Buffer区====================================
    // TBuf<TPosition::VECIN> mm1ResUb;
    LocalTensor<T> mm1ResTensor;
    LocalTensor<T> leftStateTensor;
    LocalTensor<T> rightStateTensor;
    LocalTensor<T> normWeightUb;
    LocalTensor<T> apeUb;
    LocalTensor<uint32_t> gatherOffsetCastUb;
    // 临时tbuf
    TBuf<TPosition::VECCALC> tmpBuff1;
    TBuf<TPosition::VECCALC> tmpBuff2;
    TBuf<TPosition::VECCALC> gatherOffsetBuf;
    TBuf<TPosition::VECCALC> apeBuf;
    // in queue
    TQue<QuePosition::VECIN, 1> inputQue1;
    TBuf<TPosition::VECIN> normWeightBuf;
    // out queue
    TQue<QuePosition::VECOUT, 1> outputQue1;
};

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::InitParams(const ConstInfo &constInfo, const CompressorTools<COMP> &tools)
{
    this->constInfo_ = constInfo;
    this->tools_ = tools;
    v2MBaseSize = BLOCK_VEC_BASE_BUFFER_SIZE / (constInfo_.headDim * sizeof(float));
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVectorPerf<COMP>::Init(
        __gm__ uint8_t *x,
        __gm__ uint8_t *wKv,
        __gm__ uint8_t *wGate,
        __gm__ uint8_t *kvState,
        __gm__ uint8_t *scoreState,
        __gm__ uint8_t *ape,
        __gm__ uint8_t *normWeight,
        __gm__ uint8_t *ropeSin,
        __gm__ uint8_t *ropeCos,
        __gm__ uint8_t *kvBlockTable,
        __gm__ uint8_t *scoreBlockTable,
        __gm__ uint8_t *cuSeqlens,
        __gm__ uint8_t *seqUsed,
        __gm__ uint8_t *startPos,
        __gm__ uint8_t *cmpKvOut)
{
    kvBlockTableGm_.SetGlobalBuffer((__gm__ int32_t *)kvBlockTable);
    scoreBlockTableGm_.SetGlobalBuffer((__gm__ int32_t *)scoreBlockTable);
    kvStateGm_.SetGlobalBuffer((__gm__ T *)kvState);
    scoreStateGm_.SetGlobalBuffer((__gm__ T *)scoreState);
    apeGm_.SetGlobalBuffer((__gm__ T *)ape);
    normWeightGm_.SetGlobalBuffer((__gm__ X_T *)normWeight);
    ropeSinGm_.SetGlobalBuffer((__gm__ X_T *)ropeSin);
    ropeCosGm_.SetGlobalBuffer((__gm__ X_T *)ropeCos);
    cmpKvOutGm_.SetGlobalBuffer((__gm__ X_T *)cmpKvOut);
    isExistSeqUsed = (seqUsed != nullptr);
    isExistStartPos = (startPos != nullptr);
    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        cuSeqlensGm_.SetGlobalBuffer((__gm__ int32_t *)cuSeqlens);
    }
    if (isExistSeqUsed) {
        sequsedGm_.SetGlobalBuffer((__gm__ int32_t *)seqUsed);
    }
    if (isExistStartPos) {
        startPosGm_.SetGlobalBuffer((__gm__ int32_t *)startPos);
    }
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVectorPerf<COMP>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(inputQue1, 1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(tmpBuff1, BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(tmpBuff2, BUFFER_SIZE_BYTE_64K);
    pipe->InitBuffer(outputQue1, 1, BUFFER_SIZE_BYTE_16K);
    pipe->InitBuffer(normWeightBuf, BUFFER_SIZE_BYTE_4K);
    pipe->InitBuffer(gatherOffsetBuf, BUFFER_SIZE_BYTE_1K);
    pipe->InitBuffer(apeBuf, BUFFER_SIZE_BYTE_32K);
    normWeightUb = normWeightBuf.Get<T>();
    apeUb = apeBuf.Get<T>();
    LocalTensor<X_T> normweightInUb = inputQue1.AllocTensor<X_T>();
    LocalTensor<int32_t> gatherOffsetUb = gatherOffsetBuf.Get<int32_t>();
    DataCopy(normweightInUb, normWeightGm_, constInfo_.headDim); // 获取normWeight，常驻
    inputQue1.EnQue(normweightInUb);
    inputQue1.DeQue<X_T>();
    Cast(normWeightUb, normweightInUb, RoundMode::CAST_NONE, constInfo_.headDim);
    inputQue1.FreeTensor(normweightInUb);
    if constexpr (COMP::rotaryMode == Compressor::ROTARY_MODE::INTERLEAVE) {
        SetGatherSrcOffset<float>(gatherOffsetUb, constInfo_.ropeHeadDim);
    }
    gatherOffsetCastUb = gatherOffsetUb.ReinterpretCast<uint32_t>();
    PipeBarrier<PIPE_V>();
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVectorPerf<COMP>::AllocEventID()
{
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVectorPerf<COMP>::FreeEventID()
{
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVectorPerf<COMP>::InitVec1GlobalTensor(GlobalTensor<T> preMm1ResGm,
                                                                         GlobalTensor<T> curMm1ResGm,
                                                                         GlobalTensor<T> vec1ResGm,
                                                                         GlobalTensor<T> vec2InputGm) {
    this->preMm1ResGm_ = preMm1ResGm;
    this->curMm1ResGm_ = curMm1ResGm;
    this->vec1ResGm_ = vec1ResGm;
    this->vec2InputGm_ = vec2InputGm;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVectorPerf<COMP>::GetSeqUsed(uint32_t bIdx)
{
    if (isExistSeqUsed) {
        return (uint32_t)sequsedGm_.GetValue(bIdx);
    } else {
        if constexpr (COMP::xLayout == X_LAYOUT::TH) {
            return (uint32_t)(cuSeqlensGm_.GetValue(bIdx + 1) - cuSeqlensGm_.GetValue(bIdx));
        } else {
            return constInfo_.sSize;
        }
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVectorPerf<COMP>::GetStartPos(uint32_t bIdx)
{
    if (isExistStartPos) {
        return startPosGm_.GetValue(bIdx);
    }
    return 0;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVectorPerf<COMP>::GetSeqLength(uint32_t bIdx)
{
    if (isExistSeqUsed) {
        return sequsedGm_.GetValue(bIdx);
    } else if (COMP::xLayout == X_LAYOUT::TH) {
        return cuSeqlensGm_.GetValue(bIdx + 1) - cuSeqlensGm_.GetValue(bIdx);
    } else {
        return constInfo_.sSize;
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVectorPerf<COMP>::GetBsLength(uint32_t index)
{
    if (COMP::xLayout == X_LAYOUT::TH) {
        return cuSeqlensGm_.GetValue(index);
    } else {
        return index * constInfo_.sSize;
    }
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVectorPerf<COMP>::GetBasicNum() {
    // 获取 m方向上对应基本单元Tc的个数
    uint32_t curBasicNum = 0;
    uint32_t headSize = 0;
    if (curStartPos_ % constInfo_.cmpRatio != 0) {
        headSize = constInfo_.cmpRatio - curStartPos_ % constInfo_.cmpRatio;
        headSize = headSize > curActSeqLength_ ? curActSeqLength_ : headSize;
        curBasicNum++;
    }
    // 加上中间整块及尾块
    curBasicNum += CeilDivT(curActSeqLength_ - headSize, constInfo_.cmpRatio);
    return curBasicNum;
}

template <typename COMP>
__aicore__ inline uint32_t CompressorBlockVectorPerf<COMP>::GetScSize() {
    uint32_t curBasicNum = (curStartPos_ + curActSeqLength_) / constInfo_.cmpRatio - curStartPos_ / constInfo_.cmpRatio;
    return curBasicNum;
}

// 根据计算Tc开始结束索引
template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::CalcScEndIdx(uint32_t bStart, uint32_t scStart, uint32_t dealScSize, uint32_t &bEnd, uint32_t &scEnd) {
    uint32_t accScSize = 0;
    for (int bIdx = bStart; bIdx < constInfo_.batchSize; ++bIdx) {
        bEnd = bIdx;
        // 计算起始batch的剩余块
        if (bIdx == bStart) {
            curActSeqLength_ = GetSeqLength(bIdx);
            curStartPos_ = GetStartPos(bIdx);
            accScSize += GetScSize() - scStart;
            if (accScSize >= dealScSize) {
                scEnd = scStart + dealScSize;
                return;
            }
        } else {
            curActSeqLength_ = GetSeqLength(bIdx);
            curStartPos_ = GetStartPos(bIdx);
            uint32_t curBasicNum = GetScSize();
            uint32_t curBasicNumEnd = dealScSize - accScSize;
            
            if (accScSize + curBasicNum >= dealScSize) {
                scEnd = curBasicNumEnd;
                return;
            }
            accScSize += curBasicNum;
        }
    }
}

// 根据sc的开始索引计算vec输出时的b、sc的索引
template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::GetScIdxInfo(uint32_t bStart, uint32_t scStart, uint32_t dealScSize, uint32_t v2TcStart, uint32_t v2TcEnd, 
                                                            uint32_t &outputBStart, uint32_t &outputSStart, uint32_t &outputScSize)
{
    outputScSize = v2TcEnd - v2TcStart;
    uint32_t scEnd = 0;
    uint32_t bEnd = 0;
    CalcScEndIdx(bStart, scStart, v2TcStart, bEnd, scEnd);
    outputSStart = scEnd;
    outputBStart = bEnd;
    // 处理跳batch
    curActSeqLength_ = GetSeqLength(bEnd);
    curStartPos_ = GetStartPos(bEnd);
    uint32_t curScSize = GetScSize();
    if (curScSize == scEnd) {
        outputSStart = 0;
        outputBStart++;
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::CopyInApe(const LocalTensor<T> &apeUb, uint32_t dStartIdx, uint32_t dDealSize)
{
    LocalTensor<T> apeUbTmp = inputQue1.AllocTensor<T>();

    uint32_t copyRowCount = constInfo_.cmpRatio;
    uint32_t copyColCount = dDealSize;
    uint32_t dstSingleRowCount = ((uint32_t)COMP::coff) * dDealSize;
    uint32_t srcSingleRowCount = ((uint32_t)COMP::coff) * constInfo_.headDim;

    uint32_t gmOffset = (constInfo_.aiCoreIdx % constInfo_.dBasicBlockNum) * constInfo_.dBaseSize + dStartIdx;
    uint32_t dstUbOffset = 0;
    if constexpr (COMP::coff == COFF::OVERLAP) {
        DataCopyAlignGmToUb(apeUbTmp[dstUbOffset], apeGm_[gmOffset], copyRowCount, copyColCount, srcSingleRowCount, dstSingleRowCount);

        gmOffset += constInfo_.headDim;
        dstUbOffset += dDealSize;
    }
    DataCopyAlignGmToUb(apeUbTmp[dstUbOffset], apeGm_[gmOffset], copyRowCount, copyColCount, srcSingleRowCount, dstSingleRowCount);
    inputQue1.EnQue(apeUbTmp);
    inputQue1.DeQue<T>();
    DataCopy(apeUb, apeUbTmp, ((uint32_t)COMP::coff) * dDealSize * constInfo_.cmpRatio);
    inputQue1.FreeTensor(apeUbTmp);
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::AddApeToScore(const LocalTensor<T> &scoreLocal, const LocalTensor<T> &apeUb,
    uint32_t tcDealSize, uint32_t dDealSize)
{
    uint32_t ReduceSize = COMP::coff == COFF:: OVERLAP ? 2 * constInfo_.cmpRatio :
        constInfo_.cmpRatio;
    uint32_t rCnt = ReduceSize * dDealSize;
    for (uint32_t r = 0; r < tcDealSize; r++) {
        Add(scoreLocal[r * rCnt], scoreLocal[r * rCnt], apeUb, rCnt);
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::DataCopyAlignUbToUb(const LocalTensor<T> dstLocal, const LocalTensor<T> srcLocal,
    uint32_t copyRowCount, uint32_t copyColCount, uint32_t srcSingleRowCount, uint32_t dstSingleRowCount)
{
    if (copyRowCount == 0) {
        return ;
    }
    DataCopyParams intriParams;
    intriParams.blockCount = copyRowCount;
    intriParams.blockLen = copyColCount / FP32_BLOCK_ELEMENT_NUM;
    intriParams.dstStride = (dstSingleRowCount - copyColCount) / FP32_BLOCK_ELEMENT_NUM;
    intriParams.srcStride = (srcSingleRowCount - copyColCount) / FP32_BLOCK_ELEMENT_NUM;
    DataCopy(dstLocal, srcLocal, intriParams);
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::DataCopyAlignGmToUb(const LocalTensor<T> dstLocal, const GlobalTensor<T> srcGm,
    uint32_t copyRowCount, uint32_t copyColCount, uint32_t srcSingleRowCount, uint32_t dstSingleRowCount)
{
    if (copyRowCount == 0) {
        return ;
    }
    DataCopyParams intriParams;
    intriParams.blockCount = copyRowCount;
    intriParams.blockLen = copyColCount / FP32_BLOCK_ELEMENT_NUM;
    intriParams.dstStride = (dstSingleRowCount - copyColCount) / FP32_BLOCK_ELEMENT_NUM;
    intriParams.srcStride = (srcSingleRowCount - copyColCount) / FP32_BLOCK_ELEMENT_NUM;
    DataCopy(dstLocal, srcGm, intriParams);
}


template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::OverLap(const LocalTensor<T> dstLocal, const LocalTensor<T> srcLocal,
    const Vec1SliceInfo &sliceInfo, uint32_t dStartIdx, uint32_t dDealSize)
{
    // Ub data layout after overlap when r = 4 and coff = 2:
    //  Tc0_seq01: |--- --D_L--- -|------D_R-----|
    //  Tc0_seq02: |--- --D_L--- -|------D_R-----|
    //  Tc0_seq03: |--- --D_L--- -|------D_R-----|
    //  Tc0_seq04: |--- --D_L--- -|------D_R-----|
    //  Tc1_seq01: |--- --D_L--- -|------D_R-----|
    //  Tc1_seq02: |--- --D_L--- -|------D_R-----|
    //  Tc1_seq03: |--- --D_L--- -|------D_R-----|
    //  Tc1_seq04: |--- --D_L--- -|------D_R-----|
    uint32_t srcSingleRowElemNum = dDealSize; // 2: kv和score各一份
    uint32_t copyRowCount = sliceInfo.dealTcSize * constInfo_.cmpRatio - sliceInfo.headHolderSeqCnt - sliceInfo.tailHolderSeqCnt;
    uint32_t copyColCount = dDealSize;
    uint32_t srcSingleRowCount = srcSingleRowElemNum;
    uint32_t dstSingleRowCount = ((uint32_t)COMP::coff) * dDealSize; // left和right在seq方向是交错存储的
    uint32_t srcLocalOffset = sliceInfo.dealedSeqCnt * srcSingleRowElemNum;

    uint32_t dstUbOffset = sliceInfo.dealedTcCnt * constInfo_.cmpRatio * dstSingleRowCount;
    if constexpr (COMP::coff == COFF::OVERLAP) {
        // 左侧首块
        uint32_t preSrcLocalOffset = sliceInfo.preDealedSeqCnt * srcSingleRowElemNum;
        uint32_t preDstUbOffset = (sliceInfo.dealedTcCnt * constInfo_.cmpRatio + sliceInfo.preHeadHolderSeqCnt) * dstSingleRowCount;
        DataCopyAlignUbToUb(dstLocal[preDstUbOffset], srcLocal[preSrcLocalOffset],
            sliceInfo.preValidSeqCnt, copyColCount, srcSingleRowCount, dstSingleRowCount);

        // 左侧剩余块
        preSrcLocalOffset += sliceInfo.preValidSeqCnt * srcSingleRowElemNum;
        preDstUbOffset += (sliceInfo.preValidSeqCnt + sliceInfo.preTailHolderSeqCnt + sliceInfo.headHolderSeqCnt) * dstSingleRowCount;
        DataCopyAlignUbToUb(dstLocal[preDstUbOffset], srcLocal[preSrcLocalOffset],
            copyRowCount - sliceInfo.lastTcSeqCnt, copyColCount, srcSingleRowCount, dstSingleRowCount);
        dstUbOffset += dDealSize;
        srcLocalOffset += BUFFER_SIZE_BYTE_16K / sizeof(T);
    }
    dstUbOffset += sliceInfo.headHolderSeqCnt * dstSingleRowCount;
    DataCopyAlignUbToUb(dstLocal[dstUbOffset], srcLocal[srcLocalOffset],
        copyRowCount, copyColCount, srcSingleRowCount, dstSingleRowCount);
}


template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::FromWokrSpaceToUb(const LocalTensor<T> dstLocal,
    const Vec1SliceInfo &sliceInfo, const StatisticInfo &statisticInfo, uint32_t dStartIdx, uint32_t dDealSize)
{
    uint32_t srcSingleRowElemNum = constInfo_.dBaseSize * 2; // 2: kv和score各一份
    uint32_t copyRowCount = sliceInfo.dealTcSize * constInfo_.cmpRatio - sliceInfo.headHolderSeqCnt - sliceInfo.tailHolderSeqCnt;
    uint32_t copyColCount = dDealSize;
    uint32_t srcSingleRowCount = srcSingleRowElemNum;
    uint32_t dstSingleRowCount = dDealSize; // left和right在seq方向是交错存储的
    uint32_t srcGmOffset = sliceInfo.dealedSeqCnt * srcSingleRowElemNum +  dStartIdx;

    uint32_t dstUbOffset = 0;
    if constexpr (COMP::coff == COFF::OVERLAP) {
        uint32_t preSrcGmOffset = sliceInfo.preDealedSeqCnt * srcSingleRowElemNum + dStartIdx;
        DataCopyAlignGmToUb(dstLocal[dstUbOffset], preMm1ResGm_[preSrcGmOffset],
            statisticInfo.preDealSeqCnt, copyColCount, srcSingleRowCount, dstSingleRowCount);
        dstUbOffset += BUFFER_SIZE_BYTE_16K / sizeof(T);
    }
    DataCopyAlignGmToUb(dstLocal[dstUbOffset], curMm1ResGm_[srcGmOffset],
        statisticInfo.dealSeqCnt, copyColCount, srcSingleRowCount, dstSingleRowCount);
}


template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::ReadFromCacheState(const LocalTensor<T> &output,
    const GlobalTensor<T> &state, const GlobalTensor<int32_t> &blockTableGm, uint32_t batchIdx, uint32_t startSeqIdx, uint32_t endSeqIdx,
    uint32_t dStart, uint32_t dDealSize)
{
    uint32_t coff = static_cast<uint32_t>(COMP::coff);
    uint64_t blockTablebaseOffset = batchIdx * constInfo_.maxBlockNumPerBatch;
    uint32_t curSeqIdx = startSeqIdx;
    uint32_t copyFinishRowCnt = 0;
    uint32_t seqCnt = endSeqIdx - startSeqIdx;
    while (copyFinishRowCnt < seqCnt) {
        uint64_t blockIdOffset = curSeqIdx / constInfo_.blockSize;
        uint64_t remainRowCnt = curSeqIdx % constInfo_.blockSize;
        uint64_t idInBlockTable = blockTableGm.GetValue(blockTablebaseOffset + blockIdOffset);
        uint32_t copyRowCnt = constInfo_.blockSize - remainRowCnt;
        if (copyFinishRowCnt + copyRowCnt > seqCnt) {
            copyRowCnt = seqCnt - copyFinishRowCnt;
        }
        uint64_t stateOffset = idInBlockTable * constInfo_.blockSize * coff * constInfo_.headDim +
            remainRowCnt * coff * constInfo_.headDim +
            (constInfo_.aiCoreIdx % constInfo_.dBasicBlockNum) * constInfo_.dBaseSize +
            dStart;

        DataCopyParams copyParams;
        copyParams.blockCount = copyRowCnt;
        copyParams.blockLen = dDealSize / FP32_BLOCK_ELEMENT_NUM;
        copyParams.dstStride = (coff * dDealSize - dDealSize) / FP32_BLOCK_ELEMENT_NUM;
        copyParams.srcStride = (coff * constInfo_.headDim - dDealSize) / FP32_BLOCK_ELEMENT_NUM;
        DataCopy(output[copyFinishRowCnt * coff * dDealSize], state[stateOffset], copyParams);
        copyFinishRowCnt += copyRowCnt;
        curSeqIdx += copyRowCnt;
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::WriteToCacheState(const GlobalTensor<T> &state, const GlobalTensor<int32_t> &blockTableGm,
    const LocalTensor<T> &input, uint32_t batchIdx, uint32_t startSeqIdx, uint32_t endSeqIdx,
    uint32_t dStart, uint32_t dDealSize)
{
    uint32_t coff = static_cast<uint32_t>(COMP::coff);
    uint64_t blockTablebaseOffset = batchIdx * constInfo_.maxBlockNumPerBatch;
    uint32_t curSeqIdx = startSeqIdx;
    uint32_t copyFinishRowCnt = 0;
    uint32_t seqCnt = endSeqIdx - startSeqIdx;
    while (copyFinishRowCnt < seqCnt) {
        uint64_t blockIdOffset = curSeqIdx / constInfo_.blockSize;
        uint64_t remainRowCnt = curSeqIdx % constInfo_.blockSize;
        uint64_t idInBlockTable = blockTableGm.GetValue(blockTablebaseOffset + blockIdOffset);
        uint32_t copyRowCnt = constInfo_.blockSize - remainRowCnt;
        if (copyFinishRowCnt + copyRowCnt > seqCnt) {
            copyRowCnt = seqCnt - copyFinishRowCnt;
        }
        if (idInBlockTable != 0) { // 32
            uint64_t stateOffset = idInBlockTable * constInfo_.blockSize * coff * constInfo_.headDim +
                remainRowCnt * coff * constInfo_.headDim +
                (constInfo_.aiCoreIdx % constInfo_.dBasicBlockNum) * constInfo_.dBaseSize + dStart;

            DataCopyParams copyParams;
            copyParams.blockCount = copyRowCnt;
            copyParams.blockLen = dDealSize / FP32_BLOCK_ELEMENT_NUM;
            copyParams.dstStride = (coff * constInfo_.headDim - dDealSize) / FP32_BLOCK_ELEMENT_NUM;
            copyParams.srcStride = (coff * dDealSize - dDealSize) / FP32_BLOCK_ELEMENT_NUM;
            DataCopy(state[stateOffset], input[copyFinishRowCnt * coff * dDealSize], copyParams);
        }

        copyFinishRowCnt += copyRowCnt;
        curSeqIdx += copyRowCnt;
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::SaveState(const LocalTensor<T> kvLocal, const LocalTensor<T> scoreLocal,
    const Vec1SliceInfo &sliceInfo, uint32_t dStartIdx, uint32_t dDealSize)
{
    uint32_t coff = static_cast<uint32_t>(COMP::coff);
    {
        uint32_t startSeqIdx = sliceInfo.bStartPos + sliceInfo.sIdx;
        uint32_t endSeqIdx = startSeqIdx + sliceInfo.validSeqCnt;
        uint64_t srcBaseOffset = sliceInfo.headHolderSeqCnt * coff * dDealSize + (coff - 1) * dDealSize;
        WriteToCacheState(kvStateGm_, kvBlockTableGm_, kvLocal[srcBaseOffset], sliceInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx + (coff - 1) * constInfo_.headDim, dDealSize);
        WriteToCacheState(scoreStateGm_, scoreBlockTableGm_, scoreLocal[srcBaseOffset], sliceInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx + (coff - 1) * constInfo_.headDim, dDealSize);
    }
    if constexpr (COMP::coff == COFF::OVERLAP) {
        uint32_t startSeqIdx = sliceInfo.bStartPos + sliceInfo.sIdx;
        uint32_t endSeqIdx = startSeqIdx + sliceInfo.validSeqCnt;
        if (sliceInfo.isLast) {
            endSeqIdx -= sliceInfo.lastTcSeqCnt;
        }
        uint64_t srcBaseOffset = (constInfo_.cmpRatio + sliceInfo.headHolderSeqCnt) * coff * dDealSize;
                    WriteToCacheState(kvStateGm_, kvBlockTableGm_, kvLocal[srcBaseOffset], sliceInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx, dDealSize);
                    WriteToCacheState(scoreStateGm_, scoreBlockTableGm_, scoreLocal[srcBaseOffset], sliceInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx, dDealSize);

        if (sliceInfo.isFirst && sliceInfo.preSIdx < sliceInfo.preBSeqUsed) {
            uint32_t startSeqIdx = sliceInfo.preBStartPos + sliceInfo.preSIdx;
            uint32_t endSeqIdx = min(sliceInfo.preBStartPos + sliceInfo.preBSeqUsed, startSeqIdx + sliceInfo.preValidSeqCnt);
            uint64_t srcBaseOffset = sliceInfo.preHeadHolderSeqCnt * coff * dDealSize;
            WriteToCacheState(kvStateGm_, kvBlockTableGm_, kvLocal[srcBaseOffset], sliceInfo.preBIdx, startSeqIdx, endSeqIdx, dStartIdx, dDealSize);
            WriteToCacheState(scoreStateGm_, scoreBlockTableGm_, scoreLocal[srcBaseOffset], sliceInfo.preBIdx, startSeqIdx, endSeqIdx, dStartIdx, dDealSize);
        }
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::ReadState(const LocalTensor<T> kvLocal, const LocalTensor<T> scoreLocal,
    const Vec1SliceInfo &sliceInfo, uint32_t dStartIdx, uint32_t dDealSize)
{
    // 没有需要压缩的块时, 不需要读state的信息
    if (sliceInfo.compressTcSize == 0) {
        return;
    }
    uint32_t coff = static_cast<uint32_t>(COMP::coff);
        // 填充右边
        if (sliceInfo.headHolderSeqCnt > 0) {
            // 整个batch的第一块
        uint32_t startSeqIdx = Trunc(sliceInfo.bStartPos + sliceInfo.sIdx, constInfo_.cmpRatio);
        uint32_t endSeqIdx = sliceInfo.bStartPos;
        uint64_t dstBaseOffset = (coff - 1) * dDealSize;
        ReadFromCacheState(kvLocal[dstBaseOffset], kvStateGm_, kvBlockTableGm_, sliceInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx + (coff - 1) * constInfo_.headDim, dDealSize);
        ReadFromCacheState(scoreLocal[dstBaseOffset], scoreStateGm_, scoreBlockTableGm_, sliceInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx + (coff - 1) * constInfo_.headDim, dDealSize);
        }

        // 填充左边
    if constexpr (COMP::coff == Compressor::COFF::OVERLAP) {
        float SOFTMAX_MIN_NUM = static_cast<float>(-1.0/0.0);
        bool isFirst = sliceInfo.bStartPos + sliceInfo.sIdx < constInfo_.cmpRatio;
        if (isFirst) {
            // 无历史数据
            // dDealSize必须为64
            Duplicate(kvLocal, FLOAT_ZERO, dDealSize, constInfo_.cmpRatio, 1, coff * dDealSize / REPEAT_STRIDE_NUM);
            Duplicate(scoreLocal, SOFTMAX_MIN_NUM, dDealSize, constInfo_.cmpRatio, 1, coff * dDealSize / REPEAT_STRIDE_NUM);
        }
        if (sliceInfo.sIdx < constInfo_.cmpRatio && (!isFirst || sliceInfo.compressTcSize > 1)) {
            uint32_t startSeqIdx = sliceInfo.bStartPos < constInfo_.cmpRatio ? 0 : Trunc(sliceInfo.bStartPos + sliceInfo.sIdx, constInfo_.cmpRatio) - constInfo_.cmpRatio;
            uint32_t endSeqIdx = min(Trunc(sliceInfo.bStartPos + sliceInfo.sIdx + sliceInfo.validSeqCnt, constInfo_.cmpRatio) - constInfo_.cmpRatio, sliceInfo.bStartPos);
            uint64_t dstBaseOffset = isFirst ? constInfo_.cmpRatio * coff * dDealSize : 0;
            ReadFromCacheState(kvLocal[dstBaseOffset], kvStateGm_, kvBlockTableGm_, sliceInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx, dDealSize);
            ReadFromCacheState(scoreLocal[dstBaseOffset], scoreStateGm_, scoreBlockTableGm_, sliceInfo.bIdx, startSeqIdx, endSeqIdx, dStartIdx, dDealSize);
        }
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::UpdateState(const LocalTensor<T> kvLocal, const LocalTensor<T> scoreLocal,
    const Vec1SliceInfo &sliceInfo, uint32_t dStartIdx, uint32_t dDealSize)
{
    event_t eventId_V_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    event_t eventId_MTE3_MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    event_t eventId_MTE3_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    event_t eventId_MTE2_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));

    SetFlag<HardEvent::V_MTE3>(eventId_V_MTE3);
    WaitFlag<HardEvent::V_MTE3>(eventId_V_MTE3);
    SaveState(kvLocal, scoreLocal, sliceInfo, dStartIdx, dDealSize);
    SetFlag<HardEvent::MTE3_MTE2>(eventId_MTE3_MTE2);
    SetFlag<HardEvent::MTE3_V>(eventId_MTE3_V);
    WaitFlag<HardEvent::MTE3_MTE2>(eventId_MTE3_MTE2);
    WaitFlag<HardEvent::MTE3_V>(eventId_MTE3_V);
    ReadState(kvLocal, scoreLocal, sliceInfo, dStartIdx, dDealSize);
    SetFlag<HardEvent::MTE2_V>(eventId_MTE2_V);
    WaitFlag<HardEvent::MTE2_V>(eventId_MTE2_V);
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::SoftmaxDN(
    const LocalTensor<T> &scoreLocal, const LocalTensor<T> &tmpUb, uint32_t tcDealSize, uint32_t dDealSize)
{
    float minValue = -2e38;
    uint32_t ReduceSize = COMP::coff == COFF:: OVERLAP ? 2 * constInfo_.cmpRatio :
        constInfo_.cmpRatio;
    uint32_t rCnt = ReduceSize * dDealSize;
    for (uint32_t r = 0; r < tcDealSize; r++) {
        ColumnSoftMax(scoreLocal[r * rCnt], scoreLocal[r * rCnt],
                      tmpUb[r * rCnt], ReduceSize, dDealSize);
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::KvMulReduceScore(
    const LocalTensor<T> &kvLocal, const LocalTensor<T>& scoreLocal, const LocalTensor<T> &dstLocal, const LocalTensor<T> &tmpUb, uint32_t tcDealSize, uint32_t dDealSize)
{
    uint32_t ReduceSize = COMP::coff == COFF:: OVERLAP ? 2 * constInfo_.cmpRatio : constInfo_.cmpRatio;
    uint32_t rCnt = ReduceSize * dDealSize;
    Mul(kvLocal, kvLocal, scoreLocal, tcDealSize * rCnt);
    PipeBarrier<PIPE_V>();
    for (uint32_t r = 0; r < tcDealSize; r++) {
        ColumnSum(dstLocal[r * dDealSize], kvLocal[r * rCnt], tmpUb[r  * rCnt], ReduceSize, dDealSize);
    }
}

template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::CopyOutVec1Res(const RunInfo &info, const LocalTensor<T> comperssoredUb,
    uint32_t compressTcSize, uint32_t dStartIdx, uint32_t dDealSize)
{
    uint64_t outGmOffset = info.vec1ResOffset + compressedCnt_ * constInfo_.headDim + dStartIdx;
    DataCopyParams copyParams;
    copyParams.blockCount = compressTcSize;
    copyParams.blockLen = dDealSize / FP32_BLOCK_ELEMENT_NUM;
    copyParams.dstStride = (constInfo_.headDim - dDealSize) / FP32_BLOCK_ELEMENT_NUM;
    copyParams.srcStride = 0;
    DataCopy(vec1ResGm_[outGmOffset], comperssoredUb, copyParams);
}


template <typename COMP>
__aicore__ inline void CompressorBlockVectorPerf<COMP>::DealVec1BaseBlock(const RunInfo &info,
                                                                      CompressorVec1SliceIterator<COMP, false> &sliceIterator,
                                                                      uint32_t dStartIdx, uint32_t dDealSize)
{
    Vec1SliceInfo originSliceInfo = sliceIterator.GetSlice();
    uint32_t needDealTcSize = sliceIterator.GetNeedDealTcSize();
    StatisticInfo& statisticInfo = sliceIterator.template FullIteratorSlice<true>();
    bool onlyPreLastTc = false;
    if (statisticInfo.actualTcCnt == 0) {
        if (statisticInfo.hasPreLastTc) {
            needDealTcSize = 1;
            onlyPreLastTc = true;
        } else {
            return ;
        }
    }

    CompressorVec1SliceIterator overLapSliceIterator(tools_);
    overLapSliceIterator.SetMaxBatchSize(constInfo_.batchSize);
    Vec1SliceInfo& overLapSliceInfo = overLapSliceIterator.GetSlice();

    LocalTensor<T> scoreUb = inputQue1.AllocTensor<T>();
    FromWokrSpaceToUb(scoreUb, originSliceInfo, statisticInfo, dStartIdx + constInfo_.dBaseSize, dDealSize);
    inputQue1.EnQue(scoreUb);
    inputQue1.DeQue<T>();
    LocalTensor<T> scoreLocal = tmpBuff1.Get<T>();
    overLapSliceIterator.template Reset<true>(originSliceInfo.bIdx, originSliceInfo.sIdx, 0U, 0U);
    overLapSliceIterator.SetNeedDealTcSize(needDealTcSize);
    while (!overLapSliceIterator.IsEnd()) {
        overLapSliceIterator.GetSlice();
        OverLap(scoreLocal, scoreUb, overLapSliceInfo, dStartIdx, dDealSize);
        overLapSliceIterator.IteratorSlice();
    }
    PipeBarrier<PIPE_V>();
    inputQue1.FreeTensor(scoreUb);

    AddApeToScore(scoreLocal, apeUb, needDealTcSize, dDealSize);

    LocalTensor<T> kvUb = inputQue1.AllocTensor<T>();
    FromWokrSpaceToUb(kvUb, originSliceInfo, statisticInfo, dStartIdx, dDealSize);
    inputQue1.EnQue(kvUb);
    inputQue1.DeQue<T>();
    LocalTensor<T> kvLocal = tmpBuff2.Get<T>();
    overLapSliceIterator.template Reset<true>(originSliceInfo.bIdx, originSliceInfo.sIdx, 0U, 0U);
    overLapSliceIterator.SetNeedDealTcSize(needDealTcSize);
    while (!overLapSliceIterator.IsEnd()) {
        overLapSliceIterator.GetSlice();
        OverLap(kvLocal, kvUb, overLapSliceInfo, dStartIdx, dDealSize);
        overLapSliceIterator.IteratorSlice();
    }
    PipeBarrier<PIPE_V>();
    inputQue1.FreeTensor(kvUb);

    CompressorVec1SliceIterator<COMP, true> computeSliceIterator(tools_);
    computeSliceIterator.SetMaxBatchSize(constInfo_.batchSize);
    computeSliceIterator.template Reset<true>(originSliceInfo.bIdx, originSliceInfo.sIdx, originSliceInfo.dealedSeqCnt, originSliceInfo.dealedTcCnt);
    computeSliceIterator.SetNeedDealTcSize(statisticInfo.actualTcCnt, needDealTcSize);
    Vec1SliceInfo& sliceInfo = computeSliceIterator.GetSlice();
    while (!computeSliceIterator.IsEnd() || onlyPreLastTc) {
        computeSliceIterator.GetSlice();
        uint32_t ubOffset = sliceInfo.dealedTcCnt * constInfo_.cmpRatio * ((uint32_t)COMP::coff) * dDealSize;

        UpdateState(kvLocal[ubOffset], scoreLocal[ubOffset], sliceInfo, dStartIdx, dDealSize);
        
        if (onlyPreLastTc) {
            break;
        }

        if (sliceInfo.compressTcSize > 0) {
            LocalTensor<T> tmpUb = kvLocal[BUFFER_SIZE_BYTE_32K / sizeof(T)];
            PipeBarrier<PIPE_V>();
            SoftmaxDN(scoreLocal[ubOffset], tmpUb, sliceInfo.compressTcSize, dDealSize);

            LocalTensor<T> comperssoredUb = outputQue1.AllocTensor<T>();
            PipeBarrier<PIPE_V>();
            KvMulReduceScore(kvLocal[ubOffset], scoreLocal[ubOffset], comperssoredUb, tmpUb, sliceInfo.compressTcSize, dDealSize);
            PipeBarrier<PIPE_V>();
            outputQue1.EnQue(comperssoredUb);
            outputQue1.DeQue<T>();
            CopyOutVec1Res(info, comperssoredUb, sliceInfo.compressTcSize, dStartIdx, dDealSize);
            outputQue1.FreeTensor(comperssoredUb);
        }
        computeSliceIterator.IteratorSlice();
        compressedCnt_ += sliceInfo.compressTcSize;
    }
}

template <typename COMP>
 __aicore__ inline void CompressorBlockVectorPerf<COMP>::ComputeVec1(const RunInfo &info)
{

    CompressorVec1SliceIterator sliceIterator(tools_);
    sliceIterator.SetMaxBatchSize(constInfo_.batchSize);
    sliceIterator.template Reset<true>(info.bStart, info.sStart, 0U, 0U);
    Vec1SliceInfo sliceInfo = sliceIterator.GetSlice();
    // 计算当前VecCore的任务量
    uint32_t dealSeqStartIdx = 0;
    uint32_t dealTcSize = CeilDivT(info.dealTcNum, 2U);
    uint32_t curBStart = info.bStart;
    uint32_t curSStart = info.sStart;
    uint32_t curCompressedCnt = 0;
    // 刷新当前VecCore的起始bIdx和sIdx
    if (GetBlockIdx() % 2 == 1) {
        sliceIterator.SetNeedDealTcSize(dealTcSize);
        dealTcSize = info.dealTcNum - dealTcSize;
        if (dealTcSize == 0) {
            return;
        }
        while (!sliceIterator.IsEnd()) {
            sliceInfo = sliceIterator.GetSlice();
            curCompressedCnt += sliceInfo.compressTcSize;
            sliceIterator.IteratorSlice();
        }
        sliceInfo = sliceIterator.GetSlice();
        dealSeqStartIdx = sliceInfo.dealedSeqCnt;
        curBStart = sliceInfo.bIdx;
        curSStart = sliceInfo.sIdx;
    }
    // 计算headDim和Tc方向切分大小
    constexpr uint32_t BASE_BLOCK_ELEMENT_NUM = BUFFER_SIZE_BYTE_32K / sizeof(T);
    uint32_t maxDealColNum = BASE_BLOCK_ELEMENT_NUM / (constInfo_.cmpRatio * (uint32_t)COMP::coff);

    uint32_t tcSplitSize = 0;
    uint32_t dSplitSize = 0;
    uint32_t dLoopCount = 0;
    
    // 切块逻辑
    {
        if (maxDealColNum < constInfo_.dBaseSize) {
            tcSplitSize = 1;
            dLoopCount = CeilDivT(constInfo_.dBaseSize, maxDealColNum);
            dSplitSize = constInfo_.dBaseSize / dLoopCount;
        } else {
            dSplitSize = constInfo_.dBaseSize;
            dLoopCount = constInfo_.dBaseSize / dSplitSize;
            tcSplitSize = maxDealColNum / constInfo_.dBaseSize;
        }    
    }

    // 切块循环
    for (uint32_t dLoopIdx = 0; dLoopIdx < dLoopCount; dLoopIdx++) {
        CopyInApe(apeUb, dLoopIdx * dSplitSize, dSplitSize);

        sliceIterator.Reset(curBStart, curSStart, dealSeqStartIdx, 0U);
        uint32_t actDealTcSize = tcSplitSize;
        compressedCnt_ = curCompressedCnt;
        for (uint32_t tcIdx = 0; tcIdx < dealTcSize; tcIdx += tcSplitSize) {
            if (tcIdx + tcSplitSize > dealTcSize) {
                actDealTcSize = dealTcSize - tcIdx;
            }
            // 处理单个切块
            sliceIterator.SetNeedDealTcSize(actDealTcSize);
            sliceIterator.SetDealedTcCnt(0U);
            DealVec1BaseBlock(info, sliceIterator, dLoopIdx * dSplitSize, dSplitSize);
        }
    }
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVectorPerf<COMP>::ComputeVec2(const Compressor::Vec2RunInfo &info)
{
    SplitCoreV2(info);
    uint32_t vec2DealM = v2TcEndIdx - v2TcStartIdx;
    uint32_t loopCount = CeilDivT(vec2DealM, v2MBaseSize);
    for (uint32_t v2LoopIdx = 0, dealSize = v2MBaseSize; v2LoopIdx < loopCount; ++v2LoopIdx) {
        if (v2LoopIdx == loopCount - 1) {
            dealSize = vec2DealM - v2LoopIdx * v2MBaseSize;
        }
        DealVec2BaseBlock(info, v2TcStartIdx + v2LoopIdx * v2MBaseSize, dealSize);
    }
    v2TcStartIdx = 0;
    v2TcEndIdx = 0;
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVectorPerf<COMP>::DealVec2BaseBlock(const Compressor::Vec2RunInfo& info, uint32_t startRow, uint32_t dealRowCount)
{
    uint32_t computeSize = dealRowCount * constInfo_.headDim;
    int64_t inGmOffset = startRow * constInfo_.headDim;
    // CopyIn
    LocalTensor<T> vec1ResUb = inputQue1.AllocTensor<T>();
    DataCopy(vec1ResUb, vec2InputGm_[inGmOffset], computeSize);
    inputQue1.EnQue(vec1ResUb);
    inputQue1.DeQue<T>();

    // RmsNorm
    LocalTensor<T> normResUb = tmpBuff1.Get<T>();
    LocalTensor<T> tempLocal = tmpBuff2.Get<T>();
    PipeBarrier<PIPE_V>();
    MultRowRmsNorm(normResUb, vec1ResUb, normWeightUb, tempLocal, info, dealRowCount);
    inputQue1.FreeTensor(vec1ResUb);



    // rope: 只对后RD进行rope; 将normResUb每行前headDim - ropeHeadDim个元素cast到X_T，然后再与rope后的结果组合存到outputUb
    LocalTensor<X_T> outputUb = outputQue1.AllocTensor<X_T>();
    PipeBarrier<PIPE_V>();
    CalRope(info, outputUb, normResUb, startRow - v2TcStartIdx, dealRowCount);
    PipeBarrier<PIPE_V>();
    // CopyOut
    outputQue1.EnQue(outputUb);
    outputQue1.DeQue<X_T>();
    CopyFinalResultOut(info, outputUb, startRow - v2TcStartIdx, dealRowCount);
    outputQue1.FreeTensor(outputUb);
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVectorPerf<COMP>::MultRowRmsNorm(const LocalTensor<T> &normResUb, const LocalTensor<T> &vec1ResUb, const LocalTensor<T> &normWeightUb,
    const LocalTensor<T> &tempLocal, const Compressor::Vec2RunInfo& info, uint32_t dealRowCount)
{
    RmsNormParam rmsNormParams = {
        constInfo_.reciprocalD,
        constInfo_.normEps,
        dealRowCount,
        constInfo_.headDim
    };
    RmsNorm(normResUb, vec1ResUb, normWeightUb, tempLocal, rmsNormParams);
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVectorPerf<COMP>::CalRope(const Compressor::Vec2RunInfo& info, const LocalTensor<X_T> &outputUb,
    const LocalTensor<T> &normResUb, uint32_t startRow, uint32_t dealRowCount)
{
    uint32_t bStartIdx = OutputBStartIdx;
    uint32_t sStartIdx = OutputSStartIdx;
    uint64_t globalScStart = 0;
    CalcGlobalScStart(0, 0, bStartIdx, sStartIdx, globalScStart);
    uint32_t totalSize = dealRowCount * constInfo_.headDim;
    uint32_t dealScSize = dealRowCount;
    uint32_t curDealScSize = 0;

    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        curDealScSize = dealRowCount;
        uint32_t computeSize = curDealScSize * constInfo_.ropeHeadDim;
        uint64_t SinCosOffset = globalScStart * constInfo_.ropeHeadDim;
        // sin与cos各占一半, 实际分别最多只会用8K,总占用16K
        LocalTensor<X_T> cosUb = inputQue1.AllocTensor<X_T>();
        LocalTensor<X_T> sinUb = cosUb[BUFFER_SIZE_BYTE_8K / sizeof(X_T)];
        DataCopy(cosUb, ropeCosGm_[SinCosOffset], computeSize);
        DataCopy(sinUb, ropeSinGm_[SinCosOffset], computeSize);
        inputQue1.EnQue(sinUb);
        inputQue1.DeQue<X_T>();

        LocalTensor<T> ropeCosFp32Local = tmpBuff2.Get<T>();
        LocalTensor<T> ropeSinFp32Local = ropeCosFp32Local[BUFFER_SIZE_BYTE_16K / sizeof(T)].template ReinterpretCast<T>();
        LocalTensor<T> tempLocal = ropeSinFp32Local[BUFFER_SIZE_BYTE_16K / sizeof(T)].template ReinterpretCast<T>();
        PipeBarrier<PIPE_V>();
        Cast(ropeCosFp32Local, cosUb, RoundMode::CAST_NONE, computeSize);
        Cast(ropeSinFp32Local, sinUb, RoundMode::CAST_NONE, computeSize);
        PipeBarrier<PIPE_V>();
        inputQue1.FreeTensor(sinUb);
        RotaryPosEmb<COMP::rotaryMode>(normResUb, normResUb, ropeCosFp32Local, ropeSinFp32Local, tempLocal, gatherOffsetCastUb, curDealScSize, 
                                        constInfo_.ropeHeadDim, constInfo_.headDim, constInfo_.headDim - constInfo_.ropeHeadDim);
        PipeBarrier<PIPE_V>();
        while (dealScSize > 0) {
            UpdateOutputIdx(bStartIdx, sStartIdx, dealScSize, curDealScSize);
        }
    } else {
        // 处理BSH有效数据在内存上不连续 （可能存在pad）
        uint32_t ubProcessedCount = 0;
        uint32_t preOutputBStartIdx = 0;
        uint32_t preOutputSStartIdx = 0;
        while (dealScSize > 0) {
            // 逐batch计算写出索引
            preOutputBStartIdx = bStartIdx;
            preOutputSStartIdx = sStartIdx;
            UpdateOutputIdx(bStartIdx, sStartIdx, dealScSize, curDealScSize);
            if (curDealScSize) {
                uint32_t computeSize = curDealScSize * constInfo_.ropeHeadDim;
                uint64_t SinCosOffset = globalScStart * constInfo_.ropeHeadDim;
                // sin与cos各占一半, 实际分别最多只会用8K,总占用16K
                LocalTensor<X_T> cosUb = inputQue1.AllocTensor<X_T>();
                LocalTensor<X_T> sinUb = cosUb[BUFFER_SIZE_BYTE_8K / sizeof(X_T)];
                DataCopy(cosUb, ropeCosGm_[SinCosOffset], computeSize);
                DataCopy(sinUb, ropeSinGm_[SinCosOffset], computeSize);
                inputQue1.EnQue(sinUb);
                inputQue1.DeQue<X_T>();

                LocalTensor<T> ropeCosFp32Local = tmpBuff2.Get<T>();
                LocalTensor<T> ropeSinFp32Local = ropeCosFp32Local[BUFFER_SIZE_BYTE_16K / sizeof(T)].template ReinterpretCast<T>();
                LocalTensor<T> tempLocal = ropeSinFp32Local[BUFFER_SIZE_BYTE_16K / sizeof(T)].template ReinterpretCast<T>();

                PipeBarrier<PIPE_V>();
                Cast(ropeCosFp32Local, cosUb, RoundMode::CAST_NONE, computeSize);
                Cast(ropeSinFp32Local, sinUb, RoundMode::CAST_NONE, computeSize);
                PipeBarrier<PIPE_V>();
                inputQue1.FreeTensor(sinUb);

                RotaryPosEmb<COMP::rotaryMode>(normResUb[(dealRowCount - dealScSize - curDealScSize) * constInfo_.headDim],
                                    normResUb[(dealRowCount - dealScSize - curDealScSize) * constInfo_.headDim],
                                    ropeCosFp32Local, ropeSinFp32Local, tempLocal, gatherOffsetCastUb, curDealScSize, 
                                    constInfo_.ropeHeadDim, constInfo_.headDim, constInfo_.headDim - constInfo_.ropeHeadDim);
                PipeBarrier<PIPE_V>();
            }
            CalcGlobalScStart(preOutputBStartIdx, preOutputSStartIdx, bStartIdx, sStartIdx, globalScStart);
            ubProcessedCount += curDealScSize;
        }
    }
    Cast(outputUb, normResUb, RoundMode::CAST_RINT, totalSize);
    PipeBarrier<PIPE_V>();
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVectorPerf<COMP>::SplitCoreV2(const Compressor::Vec2RunInfo& info)
{   
    // 累积N个基本块数据后做vec2，N=2，传入的RunInfo包含该组核处理的数据块的bStart、bEnd、sStart、sEnd以及dealTcCount；
    // 每组核切M方向将C1/V1后的数据分8 * 2个vec核上进行V2计算
    // 每次进行v2计算都会根据当前情况将workspace中的每组核处理的数据重新分到当前组的vec核

    // Input: syncAll前每组cube核处理的实际数据块在batch及s方向的起止idx及实际数据量(m方向)
    // Output: 每个vec核的处理数据块在m方向的起止位置及输出到Gm上的起始位置
    uint32_t coreNum = constInfo_.dBasicBlockNum * 2; // 组中有多少个vec核:16
    usedCoreNum = coreNum;
    uint32_t currCoreIdx = GetBlockIdx(); // 当前vec核ID
    uint32_t curVecCoreGroupIdx = currCoreIdx / coreNum; // 当前vec核所在组ID
    vec1ResGmStart = curVecCoreGroupIdx * constInfo_.nSize * constInfo_.tcBaseSize * constInfo_.headDim;
    // 1.计算总vec2基本块数量
    uint32_t totalBaseNum = info.dealScSize; // 当前组核累积的实际数据量
    // 2.每个vec核上分到的数据量
    uint32_t avgBaseNum = 1;
    if (totalBaseNum > coreNum) {
        avgBaseNum = CeilDivT(totalBaseNum, coreNum);
    } else {
        usedCoreNum = totalBaseNum;
    }
    if (currCoreIdx % coreNum >= usedCoreNum) {
        return;
    }
    // 3.计算每个vec核的起始结束位置
    uint32_t accumBaseNum = 0; // 当前累积的基本块数
    uint32_t targetBaseNum = (currCoreIdx % coreNum + 1) * avgBaseNum; // 当前vec核目标要达到的基本块数量
    uint32_t targetStartBaseNum = targetBaseNum - avgBaseNum; // 分当前vec核时前面已经完成分核的基本块数量
    bool setStart = false;
    for (uint32_t i = 0; i < totalBaseNum; ++i) {
        if (accumBaseNum >= totalBaseNum) {
            return;
        }
        accumBaseNum += 1;
        if (!setStart && (accumBaseNum > targetStartBaseNum)) {
            v2TcStartIdx = i;
            setStart = true;
        } 
        if (setStart && (accumBaseNum >= targetBaseNum || i == (totalBaseNum - 1))) {
            // 更新当前核的End分核信息
            v2TcEndIdx = i + 1;
            GetScIdxInfo(info.bStart, info.bCompressedId, info.dealScSize, v2TcStartIdx, v2TcEndIdx,
                OutputBStartIdx, OutputSStartIdx, OutputSize);
            return;
        }
    }
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVectorPerf<COMP>::CalcGlobalScStart(uint32_t bStart, uint32_t scStart, uint32_t bEnd,
                                                                            uint32_t scEnd, uint64_t &globalScStart)
{
    for (uint32_t bIdx = bStart; bIdx < bEnd; ++bIdx) {
        if constexpr (COMP::xLayout == X_LAYOUT::TH) {
            curActSeqLength_ = GetSeqLength(bIdx);
            curStartPos_ = GetStartPos(bIdx);
            globalScStart += GetScSize();
        } else {
            curActSeqLength_ = constInfo_.sSize;
            globalScStart += CeilDivT(curActSeqLength_, constInfo_.cmpRatio);
        }
    }
    globalScStart -= scStart;
    globalScStart += scEnd;
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVectorPerf<COMP>::UpdateOutputIdx(uint32_t &outputBStart, uint32_t &outputSStart,
                                                                        uint32_t &dealScSize, uint32_t &curDealScSize) 
{
    curActSeqLength_ = GetSeqLength(outputBStart);
    curStartPos_ = GetStartPos(outputBStart);
    uint32_t curBatchScSize = (curStartPos_ + curActSeqLength_) / constInfo_.cmpRatio - curStartPos_ / constInfo_.cmpRatio;
    uint32_t curBatchRemainScSize = curBatchScSize - outputSStart;
    curDealScSize = curBatchRemainScSize > dealScSize ? dealScSize : curBatchRemainScSize;
    dealScSize -= curDealScSize;
    outputSStart += curDealScSize;
    if (outputSStart == curBatchScSize) {
        outputBStart++;
        outputSStart = 0;
    }
}

template <typename COMP> 
__aicore__ inline void CompressorBlockVectorPerf<COMP>::CopyFinalResultOut(const Compressor::Vec2RunInfo& info, const LocalTensor<X_T> &cmpKvOutUb,
                                                                        uint32_t startRow, uint32_t dealRowCount)
{   
    uint64_t globalScStart = 0;
    CalcGlobalScStart(0, 0, OutputBStartIdx, OutputSStartIdx, globalScStart);
    uint64_t outOffset = globalScStart * constInfo_.headDim;
    uint32_t copySize = dealRowCount * constInfo_.headDim;

    uint32_t dealScSize = dealRowCount;
    uint32_t curDealScSize = 0;
    if constexpr (COMP::xLayout == X_LAYOUT::TH) {
        DataCopy(cmpKvOutGm_[outOffset], cmpKvOutUb, copySize);
        while (dealScSize > 0) {
            UpdateOutputIdx(OutputBStartIdx, OutputSStartIdx, dealScSize, curDealScSize);
        }
    } else {
        // 处理BSH有效数据在内存上不连续（可能存在pad）
        uint32_t ubProcessedCount = 0;
        uint32_t preOutputBStartIdx = 0;
        uint32_t preOutputSStartIdx = 0;
        while (dealScSize > 0) {
            // 逐batch计算写出索引
            preOutputBStartIdx = OutputBStartIdx;
            preOutputSStartIdx = OutputSStartIdx;
            UpdateOutputIdx(OutputBStartIdx, OutputSStartIdx, dealScSize, curDealScSize);
            DataCopy(cmpKvOutGm_[globalScStart * constInfo_.headDim], cmpKvOutUb[ubProcessedCount * constInfo_.headDim], curDealScSize * constInfo_.headDim);
            CalcGlobalScStart(preOutputBStartIdx, preOutputSStartIdx, OutputBStartIdx, OutputSStartIdx, globalScStart);
            ubProcessedCount += curDealScSize;
        }
    }
} 
} // namespace Compressor
#endif // COMPRESSOR_BLOCK_VECTOR_PREF_H