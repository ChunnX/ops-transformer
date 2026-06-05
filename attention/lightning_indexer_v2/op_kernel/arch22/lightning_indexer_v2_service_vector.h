/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 */

/*!
 * \file lightning_indexer_v2_service_vector.h
 * \brief
 */
#ifndef LIGHTNING_INDEXER_V2_SERVICE_VECTOR_H
#define LIGHTNING_INDEXER_V2_SERVICE_VECTOR_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "lightning_indexer_v2_common.h"
#include "lightning_indexer_v2_vector.h"

namespace LIV2Kernel {
using namespace LIV2Common;
using namespace LIV2ServiceVec;
constexpr uint32_t BASE_TOPK = 2048;
constexpr uint32_t SPARSE_COUNT_4K = 4096;
constexpr uint32_t LD_PARAM_NUM = 16;
constexpr uint32_t EVENTID_V_TO_MTE2_PING = 0;
constexpr uint32_t EVENTID_V_TO_MTE2_PONG = 1;
constexpr uint32_t EVENTID_V_TO_MTE2_TMPUB = 2;

template<typename Q_T, typename W_T = void>
struct LightningIndexerV2TypeTraits {
    using weightsType = Q_T;
};

template<typename Q_T>
struct LightningIndexerV2TypeTraits<Q_T, float> {
    using weightsType = float;
};

template <typename LIT>
class LightningIndexerV2ServiceVector {
public:
    // =================================类型定义区=================================
    static constexpr bool DT_W_FLAG = LIT::weightsTypeFlag;
    using Q_T = typename LIT::queryType;
    using K_T = typename LIT::keyType;
    static constexpr LI_V2_LAYOUT LAYOUT_T = LIT::layout;
    using QK_T = typename LIT::queryKeyType;
    using W_T = typename LightningIndexerV2TypeTraits<Q_T,
                                          typename std::conditional<DT_W_FLAG, float, void>::type>::weightsType;

    // MM输出数据类型, 当前只支持float
    using MM1_OUT_T = float;

    __aicore__ inline LightningIndexerV2ServiceVector(){};
    __aicore__ inline void ProcessVec1(const LIV2Common::RunInfo &info);
    __aicore__ inline void ProcessLD();
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct LIV2Common::ConstInfo &constInfo,
                                      const LIV2TilingData *__restrict tilingData);
    __aicore__ inline void InitVecInputTensor(GlobalTensor<W_T> weightsGm, GlobalTensor<int32_t> indiceOutGm,
                                              GlobalTensor<K_T> valueOutGm, GlobalTensor<int32_t> blockTableGm,
                                              GlobalTensor<int32_t> outputIdxOffsetGm);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<float> vec1ResGm,
                                                GlobalTensor<int64_t> vec1ParamGm, GlobalTensor<W_T> weightsGm,
                                                GlobalTensor<int32_t> indiceOutGm, GlobalTensor<K_T> valueOutGm);
    __aicore__ inline void CleanInvalidOutput(int64_t invalidS1offset);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitLDBuffers(TPipe *pipe);

protected:
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<float> vec1ResGm;
    GlobalTensor<int64_t> vec1ParamGm;
    GlobalTensor<W_T> weightsGm;
    GlobalTensor<int32_t> indiceOutGm;
    GlobalTensor<K_T> valueOutGm;
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> outputIdxOffsetGm;
    // =================================常量区=================================

private:
    // ================================Local Buffer区====================================
    // queue
    TQue<QuePosition::VECOUT, 1> outQueue_;

    // tmp buff for vector
    TBuf<TPosition::VECCALC> sortOutBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> indexBuf_;
    TBuf<TPosition::VECCALC> reduceOutBuf_;
    TBuf<TPosition::VECCALC> brcBuf_;
    TBuf<TPosition::VECCALC> paramBuf_;

    // tmp buff for LD
    TBuf<> ldToBeMrgBuf_;
    TBuf<> ldTmpBuf_;
    TBuf<> ldOutValueBuf_;
    TBuf<> ldOutIdxBuf_;

    LocalTensor<float> tmpUb_;
    LocalTensor<int32_t> globalTopkIndice_;
    LocalTensor<float> globalTopkUb_;
    LocalTensor<float> SortedBasicBlock_;

    int32_t blockId_ = -1;
    // para for vector
    int32_t groupInner_ = 0;
    int32_t globalTopkNum_ = 0;
    int64_t blockS2StartIdx_ = 0;
    int32_t gSize_ = 0;
    int32_t kHeadNum_ = 0;
    int32_t kSeqSize_ = 0;
    int32_t s1BaseSize_ = 0;
    int32_t s2BaseSize_ = 0;
    uint32_t topkCount_ = 0;
    uint32_t topkCountAlign256_ = 0;
    bool returnValueFlag = false;

    // para for LD
    uint32_t mrgListNum_ = 4;
    uint32_t paramNum_ = 16;
    int32_t virTopK = 0;

    constexpr static uint32_t REDUCE_BANK_CONFLICT_OFFSETS = 256;
    constexpr static uint32_t REDUCE_BANK_CONFLICT_NUM = REDUCE_BANK_CONFLICT_OFFSETS / sizeof(float);

    struct LIV2Common::ConstInfo constInfo_;
};

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::InitBuffers(TPipe *pipe)
{
    uint32_t outNeedBufSize = (BASE_TOPK * 2) * 2 * sizeof(float);
    uint32_t reduceCacheSize = REDUCE_BANK_CONFLICT_OFFSETS + groupInner_ * s2BaseSize_ * sizeof(float);
    outNeedBufSize = reduceCacheSize > outNeedBufSize ? reduceCacheSize : outNeedBufSize;
    virTopK = constInfo_.isSparseCountOver2K ? constInfo_.topk : BASE_TOPK;

    pipe->InitBuffer(outQueue_, 1, outNeedBufSize);
    pipe->InitBuffer(tmpBuf_, (groupInner_ * s2BaseSize_ + s2BaseSize_) * 2 * sizeof(float));
    pipe->InitBuffer(sortOutBuf_, CeilDiv(s1BaseSize_, 2) * virTopK * 2 * sizeof(float));
    pipe->InitBuffer(indexBuf_, s2BaseSize_ * sizeof(int32_t));
    pipe->InitBuffer(reduceOutBuf_, s2BaseSize_ * 2 * sizeof(float));
    pipe->InitBuffer(brcBuf_, groupInner_ * 8 * sizeof(float));
    pipe->InitBuffer(paramBuf_, LD_PARAM_NUM * sizeof(int64_t));

    tmpUb_ = tmpBuf_.Get<float>();
    globalTopkIndice_ = indexBuf_.Get<int32_t>();
    globalTopkUb_ = sortOutBuf_.Get<float>();
    SortedBasicBlock_ = globalTopkUb_[virTopK * 2 * 2];
    globalTopkNum_ = 0;

    ArithProgression<int32_t>(globalTopkIndice_, 0, 1, s2BaseSize_);
    InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * virTopK * 2);

    LocalTensor<float> tmpBuff = outQueue_.AllocTensor<float>();
    Duplicate(tmpBuff.template ReinterpretCast<int32_t>(), -1, 2 * (s1BaseSize_ / 2) * paramNum_ * 2);
    outQueue_.EnQue<float>(tmpBuff);
    tmpBuff = outQueue_.DeQue<float>();
    int64_t wsInfoOffset = (blockId_ / 2) * s1BaseSize_ * 2 * paramNum_ +
                           (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * paramNum_;
    DataCopyPad(vec1ParamGm[wsInfoOffset], tmpBuff.template ReinterpretCast<int64_t>(),
                {1, static_cast<uint16_t>((s1BaseSize_ / 2) * 2 * paramNum_ * sizeof(int64_t)), 0, 0});
    outQueue_.FreeTensor(tmpBuff);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::InitLDBuffers(TPipe *pipe)
{
    pipe->Reset();
    pipe->InitBuffer(ldToBeMrgBuf_, 2 * BASE_TOPK * mrgListNum_ * sizeof(float));
    pipe->InitBuffer(ldTmpBuf_, 2 * BASE_TOPK * mrgListNum_ * sizeof(float));
    pipe->InitBuffer(ldOutValueBuf_, BASE_TOPK * sizeof(float));
    pipe->InitBuffer(ldOutIdxBuf_, BASE_TOPK * sizeof(int32_t));
}

template <typename LIT>
__aicore__ inline void
LightningIndexerV2ServiceVector<LIT>::InitParams(const struct LIV2Common::ConstInfo &constInfo,
                                                 const LIV2TilingData *__restrict tilingData)
{
    this->constInfo_ = constInfo;
    blockS2StartIdx_ = 0;
    gSize_ = constInfo.gSize;
    kSeqSize_ = constInfo.kSeqSize;
    kHeadNum_ = constInfo.kHeadNum;
    s1BaseSize_ = constInfo.s1BaseSize;
    s2BaseSize_ = constInfo.s2BaseSize;
    returnValueFlag = constInfo.returnValueFlag;
    topkCount_ = constInfo.topk;
    topkCountAlign256_ = LIV2Common::Align(constInfo.topk, (uint64_t)256);

    groupInner_ = 16;

    blockId_ = GetBlockIdx();

    constInfo_.isSparseCountOver2K = (constInfo_.topk > BASE_TOPK);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::InitVecInputTensor(GlobalTensor<W_T> weightsGm,
                                                                                 GlobalTensor<int32_t> indiceOutGm,
                                                                                 GlobalTensor<K_T> valueOutGm,
                                                                                 GlobalTensor<int32_t> blockTableGm,
                                                                                 GlobalTensor<int32_t> outputIdxOffsetGm)
{
    this->weightsGm = weightsGm;
    this->indiceOutGm = indiceOutGm;
    this->valueOutGm = valueOutGm;
    this->blockTableGm = blockTableGm;
    this->outputIdxOffsetGm = outputIdxOffsetGm;
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm,
                                                                                   GlobalTensor<float> vec1ResGm,
                                                                                   GlobalTensor<int64_t> vec1ParamGm,
                                                                                   GlobalTensor<W_T> weightsGm,
                                                                                   GlobalTensor<int32_t> indiceOutGm,
                                                                                   GlobalTensor<K_T> valueOutGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->vec1ParamGm = vec1ParamGm;
    this->weightsGm = weightsGm;
    this->indiceOutGm = indiceOutGm;
    this->valueOutGm = valueOutGm;
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::AllocEventID()
{
    SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PING);
    SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PONG);
    SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::FreeEventID()
{
    WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PING);
    WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_PONG);
    WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::CleanInvalidOutput(int64_t invalidS1offset)
{
    LocalTensor<float> valueULocal = outQueue_.AllocTensor<float>();
    LocalTensor<int32_t> idxULocal1 = valueULocal.template ReinterpretCast<int32_t>();
    Duplicate(idxULocal1, constInfo_.INVALID_IDX, constInfo_.topk);
    outQueue_.EnQue<float>(valueULocal);
    valueULocal = outQueue_.DeQue<float>();
    LIV2ServiceVec::CopyOut(indiceOutGm[invalidS1offset], idxULocal1, constInfo_.topk);
    outQueue_.FreeTensor(valueULocal);

    if (returnValueFlag) {
        uint16_t negInf = 0;
        if constexpr (std::is_same_v<K_T, float16_t>) {
            negInf = 0xFC00;
        } else {
            negInf = 0xFF80;
        }
        LocalTensor<uint16_t> valueULocal = outQueue_.AllocTensor<uint16_t>();
        Duplicate(valueULocal, negInf, constInfo_.topk);
        outQueue_.EnQue<uint16_t>(valueULocal);
        valueULocal = outQueue_.DeQue<uint16_t>();
        GlobalTensor<uint16_t> valueOutGmTmp;
        valueOutGmTmp.SetGlobalBuffer((__gm__ uint16_t *)valueOutGm.GetPhyAddr());
        LIV2ServiceVec::CopyOut(valueOutGmTmp[invalidS1offset], valueULocal, constInfo_.topk);
        outQueue_.FreeTensor(valueULocal);
    }
}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::ProcessVec1(const LIV2Common::RunInfo &info)
{
    int32_t cuBaseS1Idx = info.gS1Idx * s1BaseSize_;
    int32_t cuBaseS2Idx = info.s2Idx * s2BaseSize_;

    int64_t mmGmOffset = (info.loop % 2) * (constInfo_.mBaseSizeAlign * s2BaseSize_);
    int64_t weightGmOffset = info.tensorWeightsOffset + cuBaseS1Idx * kHeadNum_ * gSize_;

    int32_t cuS1BeginIdxPerAiv = cuBaseS1Idx;
    int32_t cuS1ProcNum =
        cuS1BeginIdxPerAiv + s1BaseSize_ > info.actS1Size ? info.actS1Size % s1BaseSize_ : s1BaseSize_;
    int32_t cuS1ProcNumPerAiv = blockId_ % 2 == 0 ? CeilDiv(cuS1ProcNum, 2) : (cuS1ProcNum / 2);
    cuS1BeginIdxPerAiv += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2);

    weightGmOffset += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2) * kHeadNum_ * gSize_;
    mmGmOffset += (blockId_ % 2) * CeilDiv(cuS1ProcNum, 2) * gSize_ * info.actualSingleProcessSInnerSizeAlign;

    int32_t outerG = CeilDiv(gSize_, groupInner_);

    if (info.loop != 0 && info.s2Idx == 0) {
        InitSortOutBuf(globalTopkUb_, CeilDiv(s1BaseSize_, 2) * virTopK * 2);
        blockS2StartIdx_ = 0;
    } else if (info.loop == 0) {
        blockS2StartIdx_ = info.s2Idx;
    }
    int32_t cuRealAcSeq = info.actS2Size;
    int32_t cuRealAcSeqCount = 0;
    if (constInfo_.attenMaskFlag) {
        cuRealAcSeq = info.actS2SizeOrig - info.actS1Size + cuS1BeginIdxPerAiv;
    }
    int32_t cuRealAcSeqIni = cuRealAcSeq;
    LocalTensor<float> reduceOutBuff = reduceOutBuf_.Get<float>();
    LocalTensor<float> brcBuf = brcBuf_.Get<float>();
    uint32_t ldS1Offset = (blockId_ % 2 == 0) ? s1BaseSize_ / 2 - cuS1ProcNumPerAiv : 0;
    for (int innerS1Idx = 0; innerS1Idx < cuS1ProcNumPerAiv; innerS1Idx++) {
        if (constInfo_.attenMaskFlag) {
            cuRealAcSeqCount += 1;
            cuRealAcSeq = (cuRealAcSeqCount + cuRealAcSeqIni) / static_cast<int32_t>(constInfo_.cmpRatio);
        }
        int32_t cuS2Len = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq ? cuRealAcSeq - cuBaseS2Idx : s2BaseSize_;
        int32_t cuS1Idx = cuS1BeginIdxPerAiv + innerS1Idx;
        if (cuRealAcSeq > 0 && cuS2Len > 0) {
            int32_t cuS2LenVecAlign = CeilDiv(cuS2Len, s2BaseSize_) * s2BaseSize_;
            int32_t mmUbStride = (cuS2LenVecAlign - info.actualSingleProcessSInnerSizeAlign) / B32_BLOCK_ALIGN_NUM;
            LocalTensor<float> reduceOutInner = reduceOutBuff[s2BaseSize_];
            PipeBarrier<PIPE_V>();
            LocalTensor<float> reduceCacheBuf = outQueue_.AllocTensor<float>();
            if (constInfo_.isSparseCountOver2K) {
                WaitFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
            }
            for (int outerGidx = 0; outerGidx < outerG; outerGidx++) {
                int32_t procGnum = outerGidx != outerG - 1 ? groupInner_ : gSize_ - outerGidx * groupInner_;

                int32_t pingpong = outerGidx % 2;
                LocalTensor<float> dbTmpUb = tmpUb_[pingpong * (groupInner_ * s2BaseSize_ + s2BaseSize_)];
                LocalTensor<float> weightsInUb = dbTmpUb[procGnum * s2BaseSize_];
                WaitFlag<HardEvent::V_MTE2>(pingpong);
                LocalTensor<W_T> weightsInTUb = weightsInUb.template ReinterpretCast<W_T>();
                if constexpr (!IsSameType<W_T, float>::value) {
                    weightsInTUb = weightsInTUb[groupInner_];
                }
                int64_t mmGmAllOffet = mmGmOffset + innerS1Idx * gSize_ * info.actualSingleProcessSInnerSizeAlign +
                                       outerGidx * groupInner_ * info.actualSingleProcessSInnerSizeAlign;
                int64_t weightGmAllOffset = weightGmOffset + innerS1Idx * gSize_ + outerGidx * groupInner_;

                LIV2ServiceVec::CopyIn(dbTmpUb, weightsInTUb, mm1ResGm, weightsGm, mmGmAllOffet, weightGmAllOffset,
                                       procGnum, info.actualSingleProcessSInnerSizeAlign, mmUbStride);

                SetFlag<HardEvent::MTE2_V>(pingpong);
                WaitFlag<HardEvent::MTE2_V>(pingpong);
                LIV2ServiceVec::DoScale(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], dbTmpUb, weightsInUb, weightsInTUb,
                                        brcBuf, procGnum, s2BaseSize_, outerGidx);
                SetFlag<HardEvent::V_MTE2>(pingpong);
            }

            int32_t gRedCnt = groupInner_ > gSize_ ? gSize_ : groupInner_;
            bool isS2End = cuBaseS2Idx + s2BaseSize_ >= cuRealAcSeq;
            LIV2ServiceVec::DoReduce(reduceCacheBuf[REDUCE_BANK_CONFLICT_NUM], reduceOutInner, gRedCnt, s2BaseSize_);
            outQueue_.FreeTensor(reduceCacheBuf);

            int64_t outputIdxOffset = 0;
            if (info.isOutputIdxOffsetValid) {
                int32_t rowIdx = blockId_ % 2 * CeilDiv(cuS1ProcNum, 2) + innerS1Idx;
                outputIdxOffset = outputIdxOffsetGm.GetValue(info.outputIdxCoreOffset + rowIdx * kHeadNum_);
            }

            LocalTensor<float> sortScoreUb = reduceOutBuff;
            LocalTensor<float> sortIndiceUb = reduceOutBuff[cuS2LenVecAlign];
            PipeBarrier<PIPE_V>();
            Duplicate(sortScoreUb.template ReinterpretCast<int32_t>(), LIV2ServiceVec::NEG_INF, cuS2LenVecAlign);
            PipeBarrier<PIPE_V>();
            Adds(sortScoreUb, reduceOutInner, 0.0f, cuS2Len);
            PipeBarrier<PIPE_V>();
            LocalTensor<int32_t> sortIndiceUbInt = sortIndiceUb.template ReinterpretCast<int32_t>();
            if (cuS2LenVecAlign != cuS2Len) {
                Duplicate(sortIndiceUbInt, -1, cuS2LenVecAlign);
            }
            PipeBarrier<PIPE_V>();
            Adds(sortIndiceUbInt, globalTopkIndice_,
                 static_cast<int32_t>(info.isOutputIdxOffsetValid ? outputIdxOffset : cuBaseS2Idx), cuS2Len);
            PipeBarrier<PIPE_V>();

            LocalTensor<float> tmpSortBuf = outQueue_.AllocTensor<float>();
            if (info.actS1Size > 4 || constInfo_.isSparseCountOver2K) {
                LIV2ServiceVec::SortAll(reduceOutBuff, tmpSortBuf, cuS2LenVecAlign);
                PipeBarrier<PIPE_V>();
                LocalTensor<float> UbTmpSort = constInfo_.isSparseCountOver2K ? tmpUb_ : tmpSortBuf;
                LIV2ServiceVec::MergeSort(globalTopkUb_[innerS1Idx * virTopK * 2], virTopK, reduceOutBuff,
                                          cuS2LenVecAlign, UbTmpSort);
            } else {
                int64_t globalTopkUbCacheIdx = (info.s2Idx - blockS2StartIdx_) % 4;
                Sort<float, true>(
                    SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2 + globalTopkUbCacheIdx * s2BaseSize_ * 2],
                    reduceOutBuff, sortIndiceUbInt.template ReinterpretCast<uint32_t>(), tmpSortBuf,
                    cuS2LenVecAlign / 32);
                AscendC::PipeBarrier<PIPE_V>();
                if (globalTopkUbCacheIdx == 3 || isS2End || info.isAllLoopEnd) {
                    LocalTensor<float> tt = SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2];
                    if (info.s2Idx - blockS2StartIdx_ < 4) {
                        MrgBasicBlock(globalTopkUb_[innerS1Idx * BASE_TOPK * 2], tt,
                                      static_cast<int64_t>(globalTopkUbCacheIdx + 1), s2BaseSize_);
                    } else {
                        if (globalTopkUbCacheIdx > 0) {
                            MrgBasicBlock(tmpSortBuf, tt, static_cast<int64_t>(globalTopkUbCacheIdx + 1),
                                          s2BaseSize_);
                            PipeBarrier<PIPE_V>();
                            DataCopy(SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2], tmpSortBuf,
                                     (globalTopkUbCacheIdx + 1) * s2BaseSize_ * 2);
                        }
                        PipeBarrier<PIPE_V>();
                        SparseTopK(globalTopkUb_[innerS1Idx * BASE_TOPK * 2],
                                   SortedBasicBlock_[innerS1Idx * BASE_TOPK * 2], tmpSortBuf, BASE_TOPK,
                                   s2BaseSize_ * (globalTopkUbCacheIdx + 1));
                    }
                }
            }
            if (constInfo_.isSparseCountOver2K) {
                SetFlag<HardEvent::V_MTE2>(EVENTID_V_TO_MTE2_TMPUB);
            }
            PipeBarrier<PIPE_V>();
            outQueue_.FreeTensor(tmpSortBuf);

            bool needCopyOutGm = blockS2StartIdx_ == 0 && isS2End;
            bool needCopyWsGm = info.isLastS2InnerLoop || isS2End;

            if (needCopyOutGm) {
                int64_t offset = (constInfo_.topk <= SPARSE_COUNT_4K) ? virTopK : constInfo_.topk / 2;
                int64_t copyLen = (constInfo_.topk <= SPARSE_COUNT_4K) ? constInfo_.topk : constInfo_.topk / 2;
                int64_t copyNum = (constInfo_.topk <= SPARSE_COUNT_4K) ? 1 : 2;
                for (int64_t i = 0; i < copyNum; i++) {
                    LocalTensor<float> outValueUb = outQueue_.AllocTensor<float>();
                    LocalTensor<uint32_t> outIdxUb = outValueUb[offset].template ReinterpretCast<uint32_t>();
                    Extract(outValueUb, outIdxUb, globalTopkUb_[innerS1Idx * virTopK * 2 + 2 * i * offset],
                            (offset / 32));

                    LocalTensor<K_T> valueULocal1 = outValueUb.template ReinterpretCast<K_T>();
                    if (returnValueFlag) {
                        PipeBarrier<PIPE_V>();
                        Cast(valueULocal1, outValueUb, RoundMode::CAST_ROUND, copyLen);
                    }

                    LocalTensor<int32_t> idxULocal1 = outValueUb[offset].template ReinterpretCast<int32_t>();
                    outQueue_.EnQue<float>(outValueUb);
                    outValueUb = outQueue_.DeQue<float>();

                    LIV2ServiceVec::CopyOut(indiceOutGm[info.indiceOutOffset + cuS1Idx * constInfo_.topk + i * offset],
                                            idxULocal1, copyLen);
                    if (returnValueFlag) {
                        LIV2ServiceVec::CopyOut(valueOutGm[info.valueOutOffset + cuS1Idx * constInfo_.topk + i * offset],
                                                valueULocal1, copyLen);
                    }
                    outQueue_.FreeTensor(outValueUb);
                }
            } else if (needCopyWsGm) {
                int64_t wsOffset = (blockId_ / 2) * s1BaseSize_ * 2 * 2 * BASE_TOPK +
                                   (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * 2 * BASE_TOPK +
                                   (ldS1Offset + innerS1Idx) * 2 * 2 * BASE_TOPK;
                int64_t wsInfoOffset = (blockId_ / 2) * s1BaseSize_ * 2 * paramNum_ +
                                       (blockId_ % 2) * (s1BaseSize_ / 2) * 2 * paramNum_ +
                                       (ldS1Offset + innerS1Idx) * 2 * paramNum_;

                LocalTensor<int64_t> tmpiBuff = paramBuf_.Get<int64_t>();
                SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
                tmpiBuff.SetValue(0, static_cast<int64_t>(1));
                tmpiBuff.SetValue(1, static_cast<int64_t>(cuRealAcSeq));
                tmpiBuff.SetValue(2, static_cast<int64_t>(blockS2StartIdx_));
                tmpiBuff.SetValue(3, static_cast<int64_t>(cuBaseS2Idx + cuS2Len));
                tmpiBuff.SetValue(4, static_cast<int64_t>(isS2End));
                tmpiBuff.SetValue(5, static_cast<int64_t>(info.bN2Idx));
                tmpiBuff.SetValue(6, static_cast<int64_t>(cuS1Idx));
                tmpiBuff.SetValue(7, static_cast<int64_t>(cuS1ProcNum));
                tmpiBuff.SetValue(8, static_cast<int64_t>(info.indiceOutOffset + cuS1Idx * constInfo_.topk));
                bool isTailReduce = blockS2StartIdx_ == 0;
                if (isTailReduce) {
                    wsInfoOffset += paramNum_;
                    wsOffset += 2 * BASE_TOPK;
                }
                SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
                LIV2ServiceVec::CopyOut(vec1ParamGm[wsInfoOffset], tmpiBuff, 16);
                SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
                LIV2ServiceVec::CopyOut(vec1ResGm[wsOffset], globalTopkUb_[innerS1Idx * BASE_TOPK * 2],
                                        2 * BASE_TOPK);
                SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
            }
        } else if (cuRealAcSeq <= 0) {
            CleanInvalidOutput(info.indiceOutOffset + cuS1Idx * constInfo_.topk);
        }
    }

    if (LAYOUT_T == LI_V2_LAYOUT::BSND) {
        bool isS1LoopEnd = (cuBaseS1Idx + s1BaseSize_) >= info.actS1Size;
        int32_t invalidS1Num = constInfo_.qSeqSize - info.actS1Size;
        if (invalidS1Num > 0 && isS1LoopEnd && blockS2StartIdx_ == 0) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num, 2) : (invalidS1Num / 2);
            int32_t s1OffsetPerAiv = info.actS1Size + (blockId_ % 2) * CeilDiv(invalidS1Num, 2);
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput(info.indiceOutOffset + (s1OffsetPerAiv + innerS1Idx) * constInfo_.topk);
            }
        }

        int32_t invalidS1Num2 = (info.actS1Size - info.actS2SizeOrig) / static_cast<int32_t>(constInfo_.cmpRatio);
        if (invalidS1Num2 > 0 && isS1LoopEnd && blockS2StartIdx_ == 0 && constInfo_.attenMaskFlag) {
            int32_t s1NumPerAiv = blockId_ % 2 == 0 ? CeilDiv(invalidS1Num2, 2) : (invalidS1Num2 / 2);
            int32_t s1OffsetPerAiv = (blockId_ % 2) * CeilDiv(invalidS1Num2, 2);
            for (int innerS1Idx = 0; innerS1Idx < s1NumPerAiv; innerS1Idx++) {
                CleanInvalidOutput((info.bN2Idx * constInfo_.qSeqSize + s1OffsetPerAiv + innerS1Idx) *
                                   constInfo_.topk);
            }
        }
    }

    if (info.isLastS2InnerLoop) {
        blockS2StartIdx_ = 0;
    }

}

template <typename LIT>
__aicore__ inline void LightningIndexerV2ServiceVector<LIT>::ProcessLD()
{
    int32_t curCubeId = blockId_ / 2;
    int32_t tmpCubeId = curCubeId;

    int64_t s2ActSeq;
    int64_t s2Start;
    int64_t s2End;
    int64_t isS2End;
    int64_t bn2Idx;
    // [Fix P1-vec-3] s1Idx removed: written by producer (SetValue index 6) but never
    // read in ProcessLD. It was dead code consuming a local variable unnecessarily.
    uint32_t acc_list_num = 0;
    int64_t bIdx = 0;
    int64_t needFd;
    int64_t wsOffset;
    int64_t wsInfoOffset = 0;
    int64_t nextneedFd;
    int64_t valueOffset = 0;
    int64_t outOffset = 0;

    LocalTensor<float> curValueIdxUb = ldToBeMrgBuf_.Get<float>();
    LocalTensor<float> tmpUb = ldTmpBuf_.Get<float>();

    uint32_t s1LdStartIdx = 0;
    uint32_t s1ProcNum = 0;
    uint64_t paramGmCoreOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_;
    for (uint32_t innerS1Idx = 0; innerS1Idx < s1BaseSize_; innerS1Idx++) {
        needFd = vec1ParamGm.GetValue(paramGmCoreOffset + innerS1Idx * 2 * paramNum_ + paramNum_);
        if (needFd == 1) {
            s1LdStartIdx = (s1ProcNum == 0) ? innerS1Idx : s1LdStartIdx;
            s1ProcNum++;
        }
    }

    if (s1ProcNum == 0) {
        return;
    }

    uint32_t s1VecNum = CeilDiv(s1ProcNum, 2);
    if (blockId_ % 2 == 1) {
        s1LdStartIdx = s1LdStartIdx + s1VecNum;
        s1VecNum = s1ProcNum - s1VecNum;
    }
    for (uint32_t innerS1Idx = s1LdStartIdx; innerS1Idx < s1LdStartIdx + s1VecNum; innerS1Idx++) {
        tmpCubeId = curCubeId;
        acc_list_num = 0;
        valueOffset = 0;

        wsOffset = tmpCubeId * s1BaseSize_ * 2 * 2 * BASE_TOPK + innerS1Idx * 2 * 2 * BASE_TOPK + 2 * BASE_TOPK;
        SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
        SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
        DataCopyPad(curValueIdxUb, vec1ResGm[wsOffset],
                    {1, static_cast<uint16_t>(2 * BASE_TOPK * sizeof(int32_t)), 0, 0}, {true, 0, 0, 0});
        acc_list_num++;
        valueOffset += 2 * BASE_TOPK;

        tmpCubeId++;
        wsInfoOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_ + innerS1Idx * 2 * paramNum_;
        needFd = vec1ParamGm.GetValue(wsInfoOffset);
        isS2End = vec1ParamGm.GetValue(wsInfoOffset + 4);
        // s1Idx (offset 6) intentionally not read here: dead code removed [Fix P1-vec-3]
        outOffset = vec1ParamGm.GetValue(wsInfoOffset + 8);

        while (needFd == 1) {
            wsOffset = tmpCubeId * s1BaseSize_ * 2 * 2 * BASE_TOPK + innerS1Idx * 2 * 2 * BASE_TOPK;
            SetWaitFlag<HardEvent::V_MTE2>(HardEvent::V_MTE2);
            SetWaitFlag<HardEvent::S_MTE2>(HardEvent::S_MTE2);
            DataCopyPad(curValueIdxUb[valueOffset], vec1ResGm[wsOffset],
                        {1, static_cast<uint16_t>(2 * BASE_TOPK * sizeof(int32_t)), 0, 0}, {true, 0, 0, 0});
            valueOffset += 2 * BASE_TOPK;
            acc_list_num++;

            if (acc_list_num == mrgListNum_) {
                AscendC::MrgSort4Info params;
                params.elementLengths[0] = BASE_TOPK;
                params.elementLengths[1] = BASE_TOPK;
                params.elementLengths[2] = BASE_TOPK;
                params.elementLengths[3] = BASE_TOPK;
                params.ifExhaustedSuspension = true;
                params.validBit = 0b1111;
                params.repeatTimes = 1;

                AscendC::MrgSortSrcList<float> srcList;
                srcList.src1 = curValueIdxUb[0];
                srcList.src2 = curValueIdxUb[2 * BASE_TOPK];
                srcList.src3 = curValueIdxUb[4 * BASE_TOPK];
                srcList.src4 = curValueIdxUb[6 * BASE_TOPK];
                SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
                MrgSort(tmpUb, srcList, params);
                PipeBarrier<PIPE_V>();
                DataCopy(curValueIdxUb, tmpUb, 2 * BASE_TOPK);
                PipeBarrier<PIPE_V>();
                acc_list_num = 1;
                valueOffset = 2 * BASE_TOPK;
            }

            if (isS2End == 1) {
                break;
            }

            tmpCubeId++;
            wsInfoOffset = tmpCubeId * s1BaseSize_ * 2 * paramNum_ + innerS1Idx * 2 * paramNum_;
            needFd = vec1ParamGm.GetValue(wsInfoOffset);
            isS2End = vec1ParamGm.GetValue(wsInfoOffset + 4);
        }

        if (acc_list_num != 1) {
            AscendC::MrgSort4Info params;
            params.elementLengths[0] = BASE_TOPK;
            params.elementLengths[1] = BASE_TOPK;
            params.elementLengths[2] = BASE_TOPK;
            params.elementLengths[3] = BASE_TOPK;
            params.ifExhaustedSuspension = true;
            if (acc_list_num == 2) {
                params.validBit = 0b0011;
            } else if (acc_list_num == 3) {
                params.validBit = 0b0111;
            }
            params.repeatTimes = 1;

            AscendC::MrgSortSrcList<float> srcList;
            srcList.src1 = curValueIdxUb[0];
            srcList.src2 = curValueIdxUb[2 * BASE_TOPK];
            srcList.src3 = curValueIdxUb[4 * BASE_TOPK];
            srcList.src4 = curValueIdxUb[6 * BASE_TOPK];
            SetWaitFlag<HardEvent::MTE2_V>(HardEvent::MTE2_V);
            MrgSort(tmpUb, srcList, params);
            PipeBarrier<PIPE_V>();
            DataCopy(curValueIdxUb, tmpUb, 2 * BASE_TOPK);
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<float> outValueUb = ldOutValueBuf_.Get<float>();
        LocalTensor<uint32_t> outIdxUb = ldOutIdxBuf_.Get<uint32_t>();
        if (!returnValueFlag) {
            Extract(outValueUb, outIdxUb, curValueIdxUb, (BASE_TOPK / 32));
            LocalTensor<int32_t> idxULocal1 = outIdxUb.template ReinterpretCast<int32_t>();
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyPad(indiceOutGm[outOffset], idxULocal1,
                        {1, static_cast<uint16_t>(constInfo_.topk * sizeof(int32_t)), 0, 0});
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        } else {
            Extract(outValueUb, outIdxUb, curValueIdxUb, (BASE_TOPK / 32));
            PipeBarrier<PIPE_V>();
            LocalTensor<int32_t> idxULocal1 = outIdxUb.template ReinterpretCast<int32_t>();
            LocalTensor<K_T> valueULocal1 = outValueUb.template ReinterpretCast<K_T>();
            Cast(valueULocal1, outValueUb, RoundMode::CAST_ROUND, constInfo_.topk);
            PipeBarrier<PIPE_V>();
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyPad(indiceOutGm[outOffset], idxULocal1,
                        {1, static_cast<uint16_t>(constInfo_.topk * sizeof(int32_t)), 0, 0});
            DataCopyPad(valueOutGm[outOffset], valueULocal1,
                        {1, static_cast<uint16_t>(constInfo_.topk * sizeof(K_T)), 0, 0});
            SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
        }
    }
}
} // namespace LIV2Kernel
#endif // LIGHTNING_INDEXER_V2_SERVICE_VECTOR_H
