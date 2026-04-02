/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sparse_flash_attention_antiquant_service_vector_mla.h
 * \brief
 */
#ifndef SPARSE_FLASH_ATTENTION_ANTIQUANT_SERVICE_VECTOR_GQA_H
#define SPARSE_FLASH_ATTENTION_ANTIQUANT_SERVICE_VECTOR_GQA_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "sparse_flash_attention_antiquant_common.h"
#include "sparse_flash_attention_antiquant_metadata.h"

using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using namespace optiling::detail;
#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

template <typename SFAAT> class SFAAVectorServiceGqa {
public:
    // 中间计算数据类型为float，高精度模式
    using T = float;
    using KV_T = typename SFAAT::kvType;
    using KV_ORIGIN_T = typename SFAAT::queryType;
    using OUT_T = typename SFAAT::outputType;
    using UPDATE_T = T;
    using MM1_OUT_T = float;
    using MM2_OUT_T = float;

    __aicore__ inline SFAAVectorServiceGqa(){};
    __aicore__ inline void ProcessVec0L(const RunInfo &info);
    __aicore__ inline void ProcessVec1L(const RunInfo &info);
    __aicore__ inline void ProcessVec2L(const RunInfo &info);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitParams(const struct ConstInfo &constInfo,
                                      const SparseFlashAttentionAntiquantTilingDataMla *__restrict tilingData,
                                      SfaMetaData *metaDataPtr);
    __aicore__ inline void InitVec0GlobalTensor(const GlobalTensor<int32_t> &kvValidSizeGm, const GlobalTensor<KV_ORIGIN_T> &keyMergeGm,
                                                const GlobalTensor<KV_ORIGIN_T> &valueMergeGm, const GlobalTensor<KV_T> &keyGm, 
                                                const GlobalTensor<KV_T> &valueGm, const GlobalTensor<int32_t> &blkTableGm, 
                                                const GlobalTensor<T> &keyDequantScaleGm, const GlobalTensor<T> &valueDequantScaleGm);
    __aicore__ inline void InitVec1GlobalTensor(GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<KV_ORIGIN_T> vec1ResGm,
                                                GlobalTensor<int32_t> actualSeqLengthsQGm,
                                                GlobalTensor<int32_t> actualSeqLengthsKVGm, GlobalTensor<T> lseMaxFdGm,
                                                GlobalTensor<T> lseSumFdGm, GlobalTensor<int32_t> topKGm);
    __aicore__ inline void InitVec2GlobalTensor(GlobalTensor<T> accumOutGm, GlobalTensor<UPDATE_T> vec2ResGm,
                                                GlobalTensor<MM2_OUT_T> mm2ResGm, GlobalTensor<OUT_T> attentionOutGm);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void InitSoftmaxDefaultBuffer();
    // BLOCK和REPEAT的字节数
    static constexpr uint64_t BYTE_BLOCK = 32UL;
    static constexpr uint32_t REPEAT_BLOCK_BYTE = 256U;
    // BLOCK和REPEAT的FP32元素数
    static constexpr uint32_t FP32_BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(float);
    static constexpr uint32_t FP32_REPEAT_ELEMENT_NUM = REPEAT_BLOCK_BYTE / sizeof(float);
    // repeat stride不能超过256
    static constexpr uint32_t REPEATE_STRIDE_UP_BOUND = 256;

private:
    // ================================Base Vector==========================================
    __aicore__ inline void RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                   uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                                   uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    // ================================Vector0==========================================
    __aicore__ inline int64_t MergeKv(const RunInfo &runInfo, int64_t s2GmStartOffset, int64_t s2GmLimit,
                                    int64_t topkGmBaseOffset, bool isValue, int64_t mergeMte3Idx);
    __aicore__ inline int64_t GetKeyBNBOffset(int64_t realS2Idx, const RunInfo &runInfo, int64_t s2IdLimit);
    __aicore__ inline void GetRealS2Idx(int64_t s2GmOffset, int64_t &realS2Idx, int64_t topkGmBaseOffset,
                                        const RunInfo &runInfo);
    __aicore__ inline void CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx1,
                                    int64_t realS2Idx2, const RunInfo &runInfo, bool &mask);
    __aicore__ inline void CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size, int64_t s2StartGmOffset,
                                             int64_t mergeMte3Idx, const RunInfo &runInfo, bool isValue, bool &needWaitMte3ToMte2);
    __aicore__ inline void SetInfInBlk(const LocalTensor<T> &mmResUb, uint32_t dealRowCount, uint32_t columnCount,
                                       uint64_t startId, uint64_t endId);
    __aicore__ inline void SetInfInBlkHasTail(const LocalTensor<T> &mmResUb, uint32_t dealRowCount, uint32_t columnCount,
                                              uint64_t startId, uint64_t endId);
    __aicore__ inline void SetMidInf(const LocalTensor<T> &mmResUb, uint32_t dealRowCount, uint32_t columnCount,
                                     uint64_t startId, uint64_t endId);
    __aicore__ inline void CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx,
                                          int64_t keyBNBOffset, int64_t s2IdLimit, const RunInfo &runInfo, bool &mask);
    // ================================Vector1==========================================
    __aicore__ inline void ProcessVec1SingleBuf(const RunInfo &info, const MSplitInfo &mSplitInfo);
    __aicore__ inline void DealBmm1ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount, uint32_t loopId, bool &needMask,
                                                uint32_t &maskStart, uint32_t &maskEnd);
    __aicore__ inline void SoftmaxFlashV2Compute(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                 LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
                                                 uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount,
                                                 uint32_t actualColumnCount);
    __aicore__ inline void ElewiseCompute(const RunInfo &info, const LocalTensor<T> &mmResUb, uint32_t dealRowCount,
                                          uint32_t columnCount);
    __aicore__ inline void AttentionMaskCompute(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                const LocalTensor<T> &mmResUb, uint32_t dealRowCount,
                                                uint32_t columnCount, uint32_t startRow, bool &needMask,
                                                uint32_t &maskStart, uint32_t &maskEnd);
    __aicore__ inline void ComputeLogSumExpAndCopyToGm(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                       LocalTensor<T> &softmaxSumUb, LocalTensor<T> &softmaxMaxUb);
    // ================================Vecotr2==========================================
    __aicore__ inline void ProcessVec2SingleBuf(const RunInfo &info, const MSplitInfo &mSplitInfo);
    __aicore__ inline void DealBmm2ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow,
                                                uint32_t dealRowCount, uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void ProcessVec2Inner(const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t mStartRow,
                                            uint32_t mDealSize);
    __aicore__ inline void Bmm2DataCopyOutTrans(const RunInfo &info, LocalTensor<OUT_T> &attenOutUb, uint32_t wsMStart,
                                                uint32_t dealRowCount, uint32_t columnCount,
                                                uint32_t actualColumnCount);
    __aicore__ inline void Bmm2ResCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                          uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2CastAndCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                              uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline void Bmm2FDDataCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                             uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount);
    __aicore__ inline uint64_t CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx);
    __aicore__ inline void GetConfusionTransposeTiling(int64_t numR, int64_t numC, const uint32_t stackBufferSize,
                                                       const uint32_t typeSize, ConfusionTransposeTiling &tiling);

    static constexpr bool PAGE_ATTENTION = SFAAT::pageAttention;
    static constexpr int TEMPLATE_MODE = SFAAT::templateMode;
    // static constexpr bool FLASH_DECODE = SFAAT::flashDecode;
    bool FLASH_DECODE = SFAAT::flashDecode;
    // bool FLASH_DECODE = true;
    static constexpr SFAA_LAYOUT LAYOUT_T = SFAAT::layout;
    static constexpr SFAA_LAYOUT KV_LAYOUT_T = SFAAT::kvLayout;

    static constexpr uint64_t MERGE_CACHE_GM_BUF_NUM = 4;
    static constexpr uint64_t SYNC_INPUT_BUF1_FLAG = 2;
    static constexpr uint64_t SYNC_INPUT_BUF1_PONG_FLAG = 3;
    static constexpr uint64_t SYNC_INPUT_BUF2_FLAG = 4;
    static constexpr uint64_t SYNC_INPUT_DEQUANT_SCALE_FLAG = 6;
    static constexpr uint64_t SYNC_OUTPUT_BUF1_FLAG = 4;
    static constexpr uint64_t SYNC_OUTPUT_BUF2_FLAG = 5;
    static constexpr uint32_t INPUT1_BUFFER_OFFSET = ConstInfo::BUFFER_SIZE_BYTE_32K;
    static constexpr uint32_t SOFTMAX_TMP_BUFFER_OFFSET = ConstInfo::BUFFER_SIZE_BYTE_512B / sizeof(T);
    static constexpr uint32_t BASE_BLOCK_MAX_ELEMENT_NUM = ConstInfo::BUFFER_SIZE_BYTE_32K / sizeof(T);  // 32768/4=8096
    static constexpr uint32_t BLOCK_ELEMENT_NUM = BYTE_BLOCK / sizeof(T);                                // 32/4=8
    static constexpr T FLOAT_E_SCALAR = 8388608;
    static constexpr T LN2 = 0.6931471805599453094172;
    static constexpr T RECIP_OF_LN2 = 1 / LN2;
    static constexpr T SOFTMAX_MIN_NUM = -2e38;

    const SparseFlashAttentionAntiquantTilingDataMla *__restrict tilingData;
    SfaMetaData *metaDataPtr;

    uint32_t pingpongFlag = 0U;
    ConstInfo constInfo = {};
    uint32_t lastBN2Idx = 1 << 31;

    GlobalTensor<int32_t> mm2ResInt32Gm;
    GlobalTensor<MM1_OUT_T> mm1ResGm;
    GlobalTensor<KV_ORIGIN_T> vec1ResGm;
    GlobalTensor<T> lseSumFdGm;
    GlobalTensor<T> lseMaxFdGm;

    GlobalTensor<int32_t> actualSeqLengthsQGm;
    GlobalTensor<int32_t> actualSeqLengthsKVGm;
    GlobalTensor<T> vec2ResGm;
    GlobalTensor<MM2_OUT_T> mm2ResGm;
    GlobalTensor<T> accumOutGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<int32_t> blkTableGm_;

    GlobalTensor<KV_ORIGIN_T> keyMergeGm_;
    GlobalTensor<KV_ORIGIN_T> valueMergeGm_;
    GlobalTensor<KV_T> keyGm_;
    GlobalTensor<KV_T> valueGm_;
    GlobalTensor<T> keyDequantScaleGm_;
    GlobalTensor<T> valueDequantScaleGm_;
    GlobalTensor<int32_t> topkGm_;
    GlobalTensor<int32_t> kvValidSizeGm_;

    // ================================Local Buffer区====================================
    TBuf<> inputBuff1;  // 32K * 2
    TBuf<> inputBuff2;  // 32K
    TBuf<> outputBuff1; // 32K
    TBuf<> outputBuff2; // 4K

    TBuf<> tmpBuff1;         // 32K
    TBuf<> tmpBuff2;         // 4K
    TBuf<> v0ValidSizeBuff;  // 8K
    TBuf<> antiqauntScaleBuff;  // 1K KV共用

    TBuf<> softmaxMaxBuff;        // PRE_LOAD_NUM * 1K
    TBuf<> softmaxExpBuff;        // PRE_LOAD_NUM * 1K
    TBuf<> softmaxSumBuff;        // PRE_LOAD_NUM * 1K
    TBuf<> softmaxMaxDefaultBuff; // 1K
    TBuf<> softmaxSumDefaultBuff; // 1K

    LocalTensor<T> softmaxMaxDefaultUb;
    LocalTensor<T> softmaxSumDefaultUb;

    LocalTensor<T> softmaxMaxUb;
    LocalTensor<T> softmaxSumUb;
    LocalTensor<T> softmaxExpUb;
    LocalTensor<KV_T> kvMergUb_;
    LocalTensor<int32_t> v0ValidSizeUb_;
    LocalTensor<T> antiquantScaleUb_;
};

template <typename SFAAT> __aicore__ inline void SFAAVectorServiceGqa<SFAAT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(inputBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K * 2); // 2:pingpong
    pipe->InitBuffer(inputBuff2, ConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(outputBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K);
    pipe->InitBuffer(outputBuff2, ConstInfo::BUFFER_SIZE_BYTE_4K);

    pipe->InitBuffer(tmpBuff1, ConstInfo::BUFFER_SIZE_BYTE_32K + 128);
    pipe->InitBuffer(tmpBuff2, ConstInfo::BUFFER_SIZE_BYTE_4K);
    pipe->InitBuffer(v0ValidSizeBuff, ConstInfo::BUFFER_SIZE_BYTE_8K);
    pipe->InitBuffer(antiqauntScaleBuff, ConstInfo::BUFFER_SIZE_BYTE_1K);

    pipe->InitBuffer(softmaxMaxBuff, ConstInfo::BUFFER_SIZE_BYTE_512B * constInfo.preLoadNum);
    pipe->InitBuffer(softmaxExpBuff, ConstInfo::BUFFER_SIZE_BYTE_512B * constInfo.preLoadNum);
    pipe->InitBuffer(softmaxSumBuff, ConstInfo::BUFFER_SIZE_BYTE_512B * constInfo.preLoadNum);

    pipe->InitBuffer(softmaxMaxDefaultBuff, ConstInfo::BUFFER_SIZE_BYTE_512B);
    pipe->InitBuffer(softmaxSumDefaultBuff, ConstInfo::BUFFER_SIZE_BYTE_512B);

    softmaxMaxUb = softmaxMaxBuff.Get<T>();
    softmaxSumUb = softmaxSumBuff.Get<T>();
    softmaxExpUb = softmaxExpBuff.Get<T>();

    softmaxMaxDefaultUb = softmaxMaxDefaultBuff.Get<T>();
    softmaxSumDefaultUb = softmaxSumDefaultBuff.Get<T>();

    kvMergUb_ = inputBuff1.Get<KV_T>();

    v0ValidSizeUb_ = v0ValidSizeBuff.Get<int32_t>();
    antiquantScaleUb_ = antiqauntScaleBuff.Get<T>();
}

template <typename SFAAT>
__aicore__ inline void
SFAAVectorServiceGqa<SFAAT>::InitParams(const struct ConstInfo &constInfo,
                                     const SparseFlashAttentionAntiquantTilingDataMla *__restrict tilingData,
                                     SfaMetaData *metaDataPtr)
{
    this->constInfo = constInfo;
    this->tilingData = tilingData;
    this->metaDataPtr = metaDataPtr;
    if (metaDataPtr != nullptr) {
        FLASH_DECODE = metaDataPtr->numOfFdHead > 0U;
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqa<SFAAT>::InitVec0GlobalTensor(
    const GlobalTensor<int32_t> &kvValidSizeGm, const GlobalTensor<KV_ORIGIN_T> &keyMergeGm,
    const GlobalTensor<KV_ORIGIN_T> &valueMergeGm, const GlobalTensor<KV_T> &keyGm, const GlobalTensor<KV_T> &valueGm, const GlobalTensor<int32_t> &blkTableGm,
    const GlobalTensor<T> &keyDequantScaleGm, const GlobalTensor<T> &valueDequantScaleGm)
{
    this->keyMergeGm_ = keyMergeGm;
    this->valueMergeGm_ = valueMergeGm;
    this->keyGm_ = keyGm;
    this->valueGm_ = valueGm;
    this->blkTableGm_ = blkTableGm;
    this->kvValidSizeGm_ = kvValidSizeGm;
    this->keyDequantScaleGm_ = keyDequantScaleGm;
    this->valueDequantScaleGm_ = valueDequantScaleGm;
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqa<SFAAT>::InitVec1GlobalTensor(
    GlobalTensor<MM1_OUT_T> mm1ResGm, GlobalTensor<KV_ORIGIN_T> vec1ResGm,
    GlobalTensor<int32_t> actualSeqLengthsQGm, GlobalTensor<int32_t> actualSeqLengthsKVGm, GlobalTensor<T> lseMaxFdGm,
    GlobalTensor<T> lseSumFdGm, GlobalTensor<int32_t> topKGm)
{
    this->mm1ResGm = mm1ResGm;
    this->vec1ResGm = vec1ResGm;
    this->actualSeqLengthsQGm = actualSeqLengthsQGm;
    this->actualSeqLengthsKVGm = actualSeqLengthsKVGm;
    this->lseMaxFdGm = lseMaxFdGm;
    this->lseSumFdGm = lseSumFdGm;
    this->topkGm_ = topKGm;
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqa<SFAAT>::InitVec2GlobalTensor(GlobalTensor<T> accumOutGm,
                                                                      GlobalTensor<T> vec2ResGm,
                                                                      GlobalTensor<MM2_OUT_T> mm2ResGm,
                                                                      GlobalTensor<OUT_T> attentionOutGm)
{
    this->accumOutGm = accumOutGm;
    this->vec2ResGm = vec2ResGm;
    this->mm2ResGm = mm2ResGm;
    this->attentionOutGm = attentionOutGm;
}

template <typename SFAAT> __aicore__ inline void SFAAVectorServiceGqa<SFAAT>::AllocEventID()
{
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_DEQUANT_SCALE_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename SFAAT> __aicore__ inline void SFAAVectorServiceGqa<SFAAT>::FreeEventID()
{
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_PONG_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_DEQUANT_SCALE_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
}

template <typename SFAAT> __aicore__ inline void SFAAVectorServiceGqa<SFAAT>::InitSoftmaxDefaultBuffer()
{
    Duplicate(softmaxMaxDefaultUb, SOFTMAX_MIN_NUM, SOFTMAX_TMP_BUFFER_OFFSET);
    Duplicate(softmaxSumDefaultUb, ConstInfo::FLOAT_ZERO, SOFTMAX_TMP_BUFFER_OFFSET);
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqa<SFAAT>::ComputeLogSumExpAndCopyToGm(const RunInfo &info,
                                                                                         const MSplitInfo &mSplitInfo,
                                                                                         LocalTensor<T> &softmaxSumUb,
                                                                                         LocalTensor<T> &softmaxMaxUb)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint64_t baseOffset = mSplitInfo.nBufferStartM / 2;
    size_t size = mSplitInfo.vecDealM * FP32_BLOCK_ELEMENT_NUM;
    uint64_t accumTmpOutNum = CalcAccumOffset(info.bN2Idx, info.gS1Idx);
    uint64_t offset = (accumTmpOutNum * constInfo.mBaseSize +              // taskoffset
                       info.tndCoreStartKVSplitPos * constInfo.mBaseSize + // 份数offset
                       mSplitInfo.nBufferStartM + mSplitInfo.vecStartM) *
                       FP32_BLOCK_ELEMENT_NUM; // m轴offset
    if (info.actualSingleProcessSInnerSize != 0) {
        LocalTensor<T> tmp = outputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        Brcb(tmp, softmaxSumUb[baseOffset], (mSplitInfo.vecDealM + 7) / 8, {1, 8});
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(lseSumFdGm[offset], tmp, size);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);

        tmp = outputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
        Brcb(tmp, softmaxMaxUb[baseOffset], (mSplitInfo.vecDealM + 7) / 8, {1, 8});
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF2_FLAG);
        DataCopy(lseMaxFdGm[offset], tmp, size);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF2_FLAG);
    } else {
        matmul::InitOutput<T>(lseSumFdGm[offset], size, ConstInfo::FLOAT_ZERO);
        matmul::InitOutput<T>(lseMaxFdGm[offset], size, SOFTMAX_MIN_NUM);
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqa<SFAAT>::ElewiseCompute(const RunInfo &info,
                                                                const LocalTensor<T> &mmResUb,
                                                                uint32_t dealRowCount, uint32_t columnCount)
{
    Muls(mmResUb, mmResUb, static_cast<T>(tilingData->baseParams.scaleValue), dealRowCount * columnCount);
    // v0的无效值判断
    uint64_t s2ValidSizeFirstPart = v0ValidSizeUb_.GetValue(128 + info.loop % MERGE_CACHE_GM_BUF_NUM);
    uint64_t s2ValidSizeSecondPart = v0ValidSizeUb_.GetValue(256 + info.loop % MERGE_CACHE_GM_BUF_NUM);

    int64_t s2ProcessSize = info.actualSingleProcessSInnerSize;
    int64_t s2Pair = SfaaCeilDiv(s2ProcessSize, 2L * constInfo.sparseBlockSize);
    int64_t s2Mid = SfaaCeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize;
    if (s2Mid > s2ProcessSize) {
        s2Mid = s2ProcessSize;
    }
    if (unlikely(s2ValidSizeFirstPart < s2Mid)) {
        int64_t s2StartCeilAlign = SFAAAlign(s2ValidSizeFirstPart, 8);
        int64_t s2MidFloorAlign = s2Mid / 8 * 8;
        // 场景一 s2Mid > s2ValidSizeFirstPart + oneBlk
        // 可以推导出s2StartCeilAlign < s2Mid   第一阶段取到s2StartCeilAlign
        // s2StartCeilAlign <= s2MidFloorAlign 第二阶段取到s2MidFloorAlign
        // 场景二 s2Mid <= s2ValidSizeFirstPart + oneBlk
        // 可以推导出 s2StartCeilAlign >= s2Mid 第一阶段取到mid
        // s2StartCeilAlign > s2MidFloorAlign 第二阶段取到s2StartCeilAlign
        SetInfInBlk(mmResUb, dealRowCount, columnCount, s2ValidSizeFirstPart,
                    s2StartCeilAlign >= s2Mid ? s2Mid : s2StartCeilAlign);
        SetMidInf(mmResUb, dealRowCount, columnCount, s2StartCeilAlign, s2MidFloorAlign);
        SetInfInBlk(mmResUb, dealRowCount, columnCount,
                    s2StartCeilAlign <= s2MidFloorAlign ? s2MidFloorAlign : s2StartCeilAlign, s2Mid);
    }
    if (unlikely(s2ValidSizeSecondPart < s2ProcessSize - s2Mid)) {
        // 场景一 s2Mid + s2ValidSizeSecondPart > s2ProcessSize + oneBlk
        // 可以推导出 s2StartCeilAlign < s2ProcessSize 第一阶段取到s2StartCeilAlign
        // s2StartCeilAlign <= s2EndFloorAlign 第二阶段取到s2EndFloorAlign
        // 场景二 s2Mid + s2ValidSizeSecondPart <= s2ProcessSize + oneBlk
        // 可以推导出 s2StartCeilAlign >= s2ProcessSize 第一阶段取到s2ProcessSize
        // s2StartCeilAlign > s2EndFloorAlign 第二阶段取到s2StartCeilAlign
        int64_t s2StartCeilAlign = SFAAAlign(s2Mid + s2ValidSizeSecondPart, 8);
        int64_t s2EndFloorAlign = s2ProcessSize / 8 * 8;
        SetInfInBlk(mmResUb, dealRowCount, columnCount, s2Mid + s2ValidSizeSecondPart,
                    s2StartCeilAlign >= s2ProcessSize ? s2ProcessSize : s2StartCeilAlign);
        SetMidInf(mmResUb, dealRowCount, columnCount, s2StartCeilAlign, s2EndFloorAlign);
        SetInfInBlk(mmResUb, dealRowCount, columnCount,
                    s2StartCeilAlign <= s2EndFloorAlign ? s2EndFloorAlign : s2StartCeilAlign, s2ProcessSize);
    }
}
template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqa<SFAAT>::AttentionMaskCompute(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                            const LocalTensor<T> &mmResUb, uint32_t dealRowCount,
                                            uint32_t columnCount, uint32_t startRow, bool &needMask,
                                            uint32_t &maskStart, uint32_t &maskEnd)
{
    if (!(maskStart < info.actS2Size && needMask) || maskStart >= maskEnd) {
        return;
    }
    if (info.s2Idx * constInfo.s2BaseSize + constInfo.s2BaseSize <= maskStart) {
        return;
    }
    uint32_t gS1StartIdx = info.gS1Idx + mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow;
    uint32_t s1StartIdx = gS1StartIdx / constInfo.gSize;
    uint32_t gStartIdx = gS1StartIdx % constInfo.gSize;
    uint32_t gS1EndIdx = gS1StartIdx + dealRowCount - 1;
    uint32_t s1EndIdx = gS1EndIdx / constInfo.gSize;
    uint32_t s1Count = s1EndIdx - s1StartIdx + 1;
    uint32_t headGCount = s1Count > 1 ? (constInfo.gSize - gStartIdx) : dealRowCount;
    uint32_t dstMaskOffset = 0;
    int64_t s2StartCeilAlign = SFAAAlign(maskStart, 8);
    int64_t s2MidFloorAlign = maskEnd / 8 * 8;
    uint64_t maskEndTmp = s2StartCeilAlign >= maskEnd ? maskEnd : s2StartCeilAlign;
    uint64_t maskStartTmp = s2StartCeilAlign <= s2MidFloorAlign ? s2MidFloorAlign : s2StartCeilAlign;
    SetInfInBlkHasTail(mmResUb, headGCount, columnCount, maskStart, maskEndTmp);
    SetMidInf(mmResUb, headGCount, columnCount, s2StartCeilAlign, s2MidFloorAlign);
    SetInfInBlkHasTail(mmResUb, headGCount, columnCount, maskStartTmp, maskEnd);
    if (headGCount >= constInfo.gSize || (constInfo.gSize < 16 && headGCount < constInfo.gSize)) {
        maskStart++;
    }
    dstMaskOffset += headGCount * columnCount;
    uint32_t remainRowCount = dealRowCount - headGCount;
    uint32_t midS1Count = remainRowCount / constInfo.gSize;
    uint32_t tailGSize = remainRowCount % constInfo.gSize;
    for (uint32_t midIdx = 0; midIdx < midS1Count; ++midIdx) {
        s2StartCeilAlign = SFAAAlign(maskStart, 8);
        s2MidFloorAlign = maskEnd / 8 * 8;
        maskEndTmp = s2StartCeilAlign >= maskEnd ? maskEnd : s2StartCeilAlign;
        maskStartTmp = s2StartCeilAlign <= s2MidFloorAlign ? s2MidFloorAlign : s2StartCeilAlign;
        SetInfInBlkHasTail(mmResUb[dstMaskOffset], constInfo.gSize, columnCount, maskStart, maskEndTmp);
        SetMidInf(mmResUb[dstMaskOffset], constInfo.gSize, columnCount, s2StartCeilAlign, s2MidFloorAlign);
        SetInfInBlkHasTail(mmResUb[dstMaskOffset], constInfo.gSize, columnCount, maskStartTmp, maskEnd);
        dstMaskOffset += constInfo.gSize * columnCount;
        maskStart++;
    }
    if (tailGSize > 0) {
        s2StartCeilAlign = SFAAAlign(maskStart, 8);
        s2MidFloorAlign = maskEnd / 8 * 8;
        maskEndTmp = s2StartCeilAlign >= maskEnd ? maskEnd : s2StartCeilAlign;
        maskStartTmp = s2StartCeilAlign <= s2MidFloorAlign ? s2MidFloorAlign : s2StartCeilAlign;
        SetInfInBlkHasTail(mmResUb[dstMaskOffset], tailGSize, columnCount, maskStart, maskEndTmp);
        SetMidInf(mmResUb[dstMaskOffset], tailGSize, columnCount, s2StartCeilAlign, s2MidFloorAlign);
        SetInfInBlkHasTail(mmResUb[dstMaskOffset], tailGSize, columnCount, maskStartTmp, maskEnd);
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqa<SFAAT>::SetInfInBlk(const LocalTensor<T> &mmResUb,
                                                             uint32_t dealRowCount, uint32_t columnCount,
                                                             uint64_t startId, uint64_t endId)
{
    //       startId     endId
    // x x x   0      0   0     x x x
    // 从startId到endId部分置-inf, endId、startId为endId一个blk内部的下标
    if (startId >= endId) {
        return;
    }

    uint64_t startFloorAlignSize = startId / BLOCK_ELEMENT_NUM * BLOCK_ELEMENT_NUM;
    uint64_t notComputePreMaskOneBlk = (1 << (startId - startFloorAlignSize)) - 1;
    uint64_t notComputePostMaskOneBlk = ~((1 << (endId - startFloorAlignSize)) - 1);
    uint64_t notComputeMaskOneBlk = notComputePreMaskOneBlk ^ notComputePostMaskOneBlk;

    uint64_t maskOneBlk = ~notComputeMaskOneBlk;
    uint64_t mask[1] = {maskOneBlk};
    for (int i = 1; i < 8; i++) {
        mask[0] = mask[0] | (maskOneBlk << (i * 8));
    }
    for (uint64_t rowId = 0; rowId < dealRowCount; rowId += 8) {
        Duplicate(mmResUb[rowId * columnCount + startFloorAlignSize], SOFTMAX_MIN_NUM, mask,
                  1, SfaaCeilDiv(columnCount, 8), 0);
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqa<SFAAT>::SetInfInBlkHasTail(const LocalTensor<T> &mmResUb, uint32_t dealRowCount, uint32_t columnCount,
                                                                       uint64_t startId, uint64_t endId)
{
    if (startId >= endId) {
        return;
    }
    uint64_t startFloorAlignSize = startId / BLOCK_ELEMENT_NUM * BLOCK_ELEMENT_NUM;
    uint64_t notComputePreMaskOneBlk = ((1 << (startId - startFloorAlignSize)) - 1);
    uint64_t notComputePostMaskOneBlk = ~((1 << (endId - startFloorAlignSize)) - 1);
    uint64_t notComputeMaskOneBlk = notComputePreMaskOneBlk ^ notComputePostMaskOneBlk;
    uint64_t maskOneBlk = ~notComputeMaskOneBlk;
    uint64_t mask[1] = {maskOneBlk};
    for (int i = 1; i < 8; ++i) {
        mask[0] = mask[0] | (maskOneBlk << (i * 8));
    }
    uint32_t rowLoop = dealRowCount / 8;
    uint32_t rowTail = dealRowCount % 8;
    for (uint64_t rowId = 0; rowId < rowLoop; rowId += 8) {
        Duplicate(mmResUb[rowId * columnCount * 8 + startFloorAlignSize], SOFTMAX_MIN_NUM, mask,
            1, SfaaCeilDiv(columnCount, 8), 0);
    }

    if (rowTail > 0) {
        mask[0] = maskOneBlk;
        for (int i = 1; i < rowTail; ++i) {
            mask[0] = mask[0] | (maskOneBlk << (i * 8));
        }
        Duplicate(mmResUb[rowLoop * columnCount * 8 + startFloorAlignSize], SOFTMAX_MIN_NUM, mask,
            1, SfaaCeilDiv(columnCount, 8), 0);
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqa<SFAAT>::SetMidInf(const LocalTensor<T> &mmResUb,
                                                           uint32_t dealRowCount, uint32_t columnCount,
                                                           uint64_t startId, uint64_t endId)
{
    if (startId >= endId) {
        return;
    }
    // startId        endId
    //    0      ...    0
    // 从startId到endId部分置-inf, startId、endId为32B对齐的下标
    for (uint64_t rowId = 0; rowId < dealRowCount; rowId++) {
        Duplicate(mmResUb[rowId * columnCount + startId], SOFTMAX_MIN_NUM, endId - startId);
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqa<SFAAT>::SoftmaxFlashV2Compute(
    const RunInfo &info, const MSplitInfo &mSplitInfo, LocalTensor<T> &mmResUb, LocalTensor<uint8_t> &softmaxTmpUb,
    uint32_t startRow, uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    LocalTensor<T> inSumTensor;
    LocalTensor<T> inMaxTensor;
    uint32_t baseOffset = mSplitInfo.nBufferStartM / 2 + startRow;
    uint32_t outIdx = info.loop % (constInfo.preLoadNum);
    uint32_t softmaxOutOffset = outIdx * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset;
    if (info.isFirstSInnerLoop) {
        inMaxTensor = softmaxMaxDefaultUb;
        inSumTensor = softmaxSumDefaultUb;
    } else {
        uint32_t inIdx = (info.loop - 1) % (constInfo.preLoadNum);
        inMaxTensor = softmaxMaxUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset];
        inSumTensor = softmaxSumUb[inIdx * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset];
    }
    if (actualColumnCount !=0) {
        SoftMaxShapeInfo srcShape{dealRowCount, columnCount, dealRowCount, actualColumnCount};
        SoftMaxTiling newTiling =
            SoftMaxFlashV2TilingFunc(srcShape, sizeof(T), sizeof(T), softmaxTmpUb.GetSize(), true, false);
        SoftmaxFlashV2<T, true, true, false, false, SFAA_SOFTMAX_FLASHV2_CFG_WITHOUT_BRC>(
        mmResUb, softmaxSumUb[softmaxOutOffset], softmaxMaxUb[softmaxOutOffset], mmResUb,
        softmaxExpUb[softmaxOutOffset], inSumTensor, inMaxTensor, softmaxTmpUb, newTiling, srcShape);
    } else {
        DataCopy(softmaxSumUb[softmaxOutOffset], inSumTensor, dealRowCount);
        pipe_barrier(PIPE_V);
        DataCopy(softmaxMaxUb[softmaxOutOffset], inMaxTensor, dealRowCount);
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqa<SFAAT>::DealBmm1ResBaseBlock(
    const RunInfo &info, const MSplitInfo &mSplitInfo, uint32_t startRow, uint32_t dealRowCount,
    uint32_t columnCount, uint32_t loopId, bool &needMask, uint32_t &maskStart, uint32_t &maskEnd)
{
    uint32_t computeSize = dealRowCount * columnCount;
    uint64_t inOutGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize +
                             (mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow) * columnCount;
    LocalTensor<MM1_OUT_T> mmResUb = inputBuff1.Get<MM1_OUT_T>();
    mmResUb = mmResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM1_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    DataCopy(mmResUb, mm1ResGm[inOutGmOffset], computeSize);
    if (loopId == 0) {
        WaitFlag<HardEvent::MTE2_S>(0);
        needMask = v0ValidSizeUb_.GetValue(128 + info.loop % MERGE_CACHE_GM_BUF_NUM + 1) == info.s2Idx ||
                   v0ValidSizeUb_.GetValue(256 + info.loop % MERGE_CACHE_GM_BUF_NUM + 1) == info.s2Idx;
        if (v0ValidSizeUb_.GetValue(128 + info.loop % MERGE_CACHE_GM_BUF_NUM + 1) == info.s2Idx) {
            maskStart = v0ValidSizeUb_.GetValue(128 + info.loop % MERGE_CACHE_GM_BUF_NUM + 2);
        } else if (v0ValidSizeUb_.GetValue(256 + info.loop % MERGE_CACHE_GM_BUF_NUM + 1) == info.s2Idx) {
            maskStart = v0ValidSizeUb_.GetValue(256 + info.loop % MERGE_CACHE_GM_BUF_NUM + 2);
        } else {
            maskStart = info.actS2Size;
        }
        maskEnd = maskStart + constInfo.sparseShardSize - 1;
        maskStart += mSplitInfo.vecStartM / constInfo.gSize;
    }
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);

    ElewiseCompute(info, mmResUb, dealRowCount, columnCount);
    pipe_barrier(PIPE_V);
    AttentionMaskCompute(info, mSplitInfo, mmResUb, dealRowCount, columnCount,
        startRow, needMask, maskStart, maskEnd);
    pipe_barrier(PIPE_V);
    LocalTensor<T> tmpAFloorUb = tmpBuff1.Get<T>();
    LocalTensor<uint8_t> softmaxTmpUb = tmpAFloorUb.template ReinterpretCast<uint8_t>();

    SoftmaxFlashV2Compute(info, mSplitInfo, mmResUb, softmaxTmpUb, startRow, dealRowCount, columnCount,
                            info.actualSingleProcessSInnerSize);

    pipe_barrier(PIPE_V);
    LocalTensor<KV_ORIGIN_T> tmpMMResCastTensor = outputBuff1.Get<KV_ORIGIN_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);

    Cast(tmpMMResCastTensor, mmResUb, AscendC::RoundMode::CAST_ROUND, computeSize);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(vec1ResGm[inOutGmOffset], tmpMMResCastTensor, computeSize);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqa<SFAAT>::ProcessVec1SingleBuf(const RunInfo &info,
                                                                                  const MSplitInfo &mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }
    uint32_t mSplitSize = info.actualSingleProcessSInnerSize == 0 ?
        16 : BASE_BLOCK_MAX_ELEMENT_NUM / info.actualSingleProcessSInnerSizeAlign;
    // 1. 向下8对齐是因为UB操作至少32B
    // 2. info.actualSingleProcessSInnerSizeAlign最大512, mSplitSize可以确保最小为16
    mSplitSize = mSplitSize / 8 * 8;

    if (mSplitSize > mSplitInfo.vecDealM) {
        mSplitSize = mSplitInfo.vecDealM;
    }
    uint32_t loopCount = (mSplitInfo.vecDealM + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mSplitInfo.vecDealM - (loopCount - 1) * mSplitSize;
    uint32_t maskStart = 0;
    uint32_t maskEnd = 0;
    bool needMask = false;
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = 256 * sizeof(int32_t);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPadExtParams<int32_t> padParams;
    // 额外偏移128个元素，避免不同loop下v0和v1互相影响
    DataCopyPad(v0ValidSizeUb_[128], kvValidSizeGm_[info.loop % MERGE_CACHE_GM_BUF_NUM * (128 * 2)],
                dataCopyParams, padParams);
    SetFlag<HardEvent::MTE2_S>(0);
    if (unlikely(loopCount == 0)) {
        // scalar同步影响较大，挪到循环内部进行
        WaitFlag<HardEvent::MTE2_S>(0);
    }
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm1ResBaseBlock(info, mSplitInfo, i * mSplitSize, dealSize,
            info.actualSingleProcessSInnerSizeAlign, i, needMask, maskStart, maskEnd);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqa<SFAAT>::GetRealS2Idx(int64_t s2GmOffset, int64_t &realS2Idx,
                                                              int64_t topkGmBaseOffset, const RunInfo &runInfo)
{
    int64_t topkGmIdx = (s2GmOffset + runInfo.s2Idx * constInfo.s2BaseSize) / constInfo.sparseBlockSize;
    if (unlikely(topkGmIdx >= runInfo.sparseBlockCount)) {
        realS2Idx = -1;
        return;
    }
    realS2Idx = topkGm_.GetValue(topkGmBaseOffset + topkGmIdx) * static_cast<int64_t>(constInfo.sparseBlockSize) +
                static_cast<int64_t>((s2GmOffset + runInfo.s2Idx * constInfo.s2BaseSize) % constInfo.sparseBlockSize);
}

template <typename SFAAT>
__aicore__ inline int64_t SFAAVectorServiceGqa<SFAAT>::GetKeyBNBOffset(int64_t realS2Idx,
                                                                    const RunInfo &runInfo, int64_t s2IdLimit)
{
    if (realS2Idx < 0 || realS2Idx >= s2IdLimit) {
        return -1;
    }
    int64_t realKeyBNBOffset;
    if constexpr (PAGE_ATTENTION) {
        int64_t blkTableIdx = realS2Idx / constInfo.kvCacheBlockSize;
        int64_t blkTableOffset = realS2Idx % constInfo.kvCacheBlockSize;
        if constexpr (KV_LAYOUT_T == SFAA_LAYOUT::PA_BSND) {
            realKeyBNBOffset = (blkTableGm_.GetValue(runInfo.bIdx * constInfo.maxBlockNumPerBatch + blkTableIdx) *
                                static_cast<int64_t>(constInfo.kvCacheBlockSize) + blkTableOffset) *
                                static_cast<int64_t>(constInfo.kvHeadNum) + runInfo.n2Idx;
        } else { // PA_BNSD (blockNum n2 blockSize)
            realKeyBNBOffset = blkTableGm_.GetValue(runInfo.bIdx * constInfo.maxBlockNumPerBatch + blkTableIdx) *
                               static_cast<int64_t>(constInfo.kvHeadNum) * static_cast<int64_t>(constInfo.kvCacheBlockSize) +
                               + runInfo.n2Idx * static_cast<int64_t>(constInfo.kvCacheBlockSize) + blkTableOffset;
        }
    } else {
        realKeyBNBOffset = runInfo.tensorBOffset / constInfo.headDim + realS2Idx * static_cast<int64_t>(constInfo.kvHeadNum);
    }

    return realKeyBNBOffset;
}

template <typename SFAAT>
__aicore__ inline void
SFAAVectorServiceGqa<SFAAT>::CopyInSingleKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx, int64_t realS2Idx,
                                         int64_t keyBNBOffset, int64_t s2IdLimit, const RunInfo &runInfo, bool &mask)
{
    if (keyBNBOffset < 0) {
        return;
    }
    int64_t validS2Count =
        (realS2Idx + constInfo.sparseBlockSize > s2IdLimit ? s2IdLimit - realS2Idx : constInfo.sparseBlockSize);
    // 当前仅支持SEPARATE模式
    if (constInfo.quantScaleRepoMode == QUANT_SCALE_REPO_MODE::SEPARATE) {
        DataCopyExtParams intriParams;
        uint16_t srcStride = 0;
        if constexpr (KV_LAYOUT_T == SFAA_LAYOUT::PA_BNSD) {
            intriParams.blockCount = 1;
            intriParams.blockLen = validS2Count * constInfo.headDim * sizeof(KV_T);
        } else {
            srcStride = (constInfo.kvHeadNum - 1) * constInfo.headDim * sizeof(KV_T);
            if (unlikely(srcStride == 0)) {
                intriParams.blockCount = 1;
                intriParams.blockLen = validS2Count * constInfo.headDim * sizeof(KV_T);
            } else {
                intriParams.blockCount = validS2Count;
                intriParams.blockLen = constInfo.headDim * sizeof(KV_T); 
            }
        }
        intriParams.dstStride = 0;
        intriParams.srcStride = srcStride;
        DataCopyPadExtParams<KV_T> padParams;
        padParams.isPad = false;
        uint32_t ubOffset = mergeMte3Idx % 2 * INPUT1_BUFFER_OFFSET / sizeof(KV_T) + (mte2Size - mte3Size) * constInfo.headDim;
        DataCopyPad(kvMergUb_[ubOffset], keyGm_[keyBNBOffset * constInfo.headDim], intriParams, padParams);
        DataCopyPad(kvMergUb_[ubOffset + ConstInfo::BUFFER_SIZE_BYTE_4K], valueGm_[keyBNBOffset * constInfo.headDim], intriParams, padParams);
    }
    mte2Size += validS2Count;
    if (realS2Idx + constInfo.sparseBlockSize > s2IdLimit - constInfo.sparseShardSize + 1) {
        v0ValidSizeUb_.SetValue(runInfo.loop % MERGE_CACHE_GM_BUF_NUM + 1, runInfo.s2Idx);
        v0ValidSizeUb_.SetValue(runInfo.loop % MERGE_CACHE_GM_BUF_NUM + 2, mte2Size - constInfo.sparseShardSize + 1);
        mask = true;
    }
    if (!mask) {
        v0ValidSizeUb_.SetValue(runInfo.loop % MERGE_CACHE_GM_BUF_NUM + 1, -1);
        v0ValidSizeUb_.SetValue(runInfo.loop % MERGE_CACHE_GM_BUF_NUM + 2, runInfo.actS2Size);
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqa<SFAAT>::CopyInKv(int64_t &mte2Size, int64_t mte3Size, int64_t mergeMte3Idx,
                                                          int64_t realS2Idx1, int64_t realS2Idx2,
                                                          const RunInfo &runInfo, bool &mask)
{
    int64_t s2IdLimit = runInfo.threshold;

    int64_t keyBNBOffset1 = GetKeyBNBOffset(realS2Idx1, runInfo, s2IdLimit);
    int64_t keyBNBOffset2 = GetKeyBNBOffset(realS2Idx2, runInfo, s2IdLimit);
    if (unlikely(keyBNBOffset1 < 0 && keyBNBOffset2 < 0)) {
        return;
    }

    int64_t blkTableSrcStride = (keyBNBOffset1 > keyBNBOffset2 ? (keyBNBOffset1 - keyBNBOffset2) :
        (keyBNBOffset2 - keyBNBOffset1)) - constInfo.sparseBlockSize;
    int64_t keySrcStride = blkTableSrcStride * constInfo.headDim;
    if (likely(constInfo.kvHeadNum > 1 || keySrcStride >= INT32_MAX || keySrcStride < 0 || realS2Idx1 + constInfo.sparseBlockSize >= s2IdLimit ||
        realS2Idx2 + constInfo.sparseBlockSize >= s2IdLimit)) {
        // stride溢出、stride为负数、s2超长等异常场景，还原成2条搬运指令
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx1, keyBNBOffset1, s2IdLimit, runInfo, mask);
        CopyInSingleKv(mte2Size, mte3Size, mergeMte3Idx, realS2Idx2, keyBNBOffset2, s2IdLimit, runInfo, mask);
    } else {
        DataCopyExtParams intriParams;
        intriParams.blockCount = (keyBNBOffset1 >= 0) + (keyBNBOffset2 >= 0);
        intriParams.dstStride = 0;
        intriParams.srcStride = keySrcStride;
        DataCopyPadExtParams<KV_T> padParams;

        int64_t startGmOffset = keyBNBOffset1 > -1 ? keyBNBOffset1 : keyBNBOffset2;
        if (keyBNBOffset2 > -1 && keyBNBOffset2 < keyBNBOffset1) {
            startGmOffset = keyBNBOffset2;
        }

        // 当前仅支持SEPARATE模式
        if (constInfo.quantScaleRepoMode == QUANT_SCALE_REPO_MODE::SEPARATE) {
            intriParams.blockLen = constInfo.sparseBlockSize * constInfo.headDim;
            uint32_t headDimAlign = SFAAAlign(constInfo.headDim, ConstInfo::BUFFER_SIZE_BYTE_32B) / sizeof(KV_T);
            padParams.isPad = true;
            padParams.leftPadding = 0;
            padParams.rightPadding = headDimAlign - constInfo.headDim;
            padParams.paddingValue = 0;
            uint32_t ubOffset = mergeMte3Idx % 2 * INPUT1_BUFFER_OFFSET / sizeof(KV_T) + (mte2Size - mte3Size) * headDimAlign;
            DataCopyPad(kvMergUb_[ubOffset], keyGm_[startGmOffset * constInfo.headDim], intriParams, padParams);
            DataCopyPad(kvMergUb_[ubOffset + ConstInfo::BUFFER_SIZE_BYTE_4K], valueGm_[startGmOffset * constInfo.headDim], intriParams, padParams);
        }
        mte2Size += ((keyBNBOffset1 > -1) + (keyBNBOffset2 > -1)) * constInfo.sparseBlockSize;
        if ((realS2Idx1 > 0 && realS2Idx1 + constInfo.sparseBlockSize > s2IdLimit - constInfo.sparseShardSize + 1) ||
            (realS2Idx2 > 0 && realS2Idx2 + constInfo.sparseBlockSize > s2IdLimit - constInfo.sparseShardSize + 1)) {
            v0ValidSizeUb_.SetValue(runInfo.loop % MERGE_CACHE_GM_BUF_NUM + 1, runInfo.s2Idx);
            v0ValidSizeUb_.SetValue(runInfo.loop % MERGE_CACHE_GM_BUF_NUM + 2, mte2Size - constInfo.sparseShardSize + 1);
            mask = true;
        }
        if (!mask) {
            v0ValidSizeUb_.SetValue(runInfo.loop % MERGE_CACHE_GM_BUF_NUM + 1, -1);
            v0ValidSizeUb_.SetValue(runInfo.loop % MERGE_CACHE_GM_BUF_NUM + 2, runInfo.actS2Size);
        }
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqa<SFAAT>::CopyOutMrgeResult(int64_t mte2Size, int64_t mte3Size,
                                                                   int64_t s2GmStartOffset, int64_t mergeMte3Idx,
                                                                   const RunInfo &runInfo, bool isValue, bool &needWaitMte3ToMte2)
{
    if (mte2Size <= mte3Size) {
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + mergeMte3Idx % 2);
        return;
    }

    int32_t dealRow = mte2Size - mte3Size;
    int32_t dealRowAlign = SFAAAlign(dealRow, 32);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    LocalTensor<half> kvTensorAsFp16 = inputBuff1.Get<half>();
    kvTensorAsFp16 = kvTensorAsFp16[mergeMte3Idx % 2 * INPUT1_BUFFER_OFFSET / sizeof(KV_T) + ConstInfo::BUFFER_SIZE_BYTE_4K];
    LocalTensor<KV_T> srcTensor = kvMergUb_[mergeMte3Idx % 2 * INPUT1_BUFFER_OFFSET / sizeof(KV_T)];
    uint32_t elementNum = static_cast<uint32_t>(dealRowAlign * constInfo.headDim * 2);
    Cast(kvTensorAsFp16, srcTensor, RoundMode::CAST_NONE, elementNum);
    PipeBarrier<PIPE_V>();
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + mergeMte3Idx % 2);
    needWaitMte3ToMte2 = true; 
    LocalTensor<T> kvTensorAsFp32 = tmpBuff1.Get<T>();
    Cast(kvTensorAsFp32, kvTensorAsFp16, RoundMode::CAST_NONE, elementNum);
    PipeBarrier<PIPE_V>();
    LocalTensor<T> kvFp32 = inputBuff2.Get<T>();
    for (int i = 0; i < constInfo.headDim / FP32_REPEAT_ELEMENT_NUM; i++) {
        Mul(kvFp32[i * FP32_REPEAT_ELEMENT_NUM], kvTensorAsFp32[i * FP32_REPEAT_ELEMENT_NUM], antiquantScaleUb_[i * FP32_REPEAT_ELEMENT_NUM],
            FP32_REPEAT_ELEMENT_NUM, dealRow, {1, 1, 1, 16, 16, 0});
        Mul(kvFp32[i * FP32_REPEAT_ELEMENT_NUM + ConstInfo::BUFFER_SIZE_BYTE_4K], kvTensorAsFp32[i * FP32_REPEAT_ELEMENT_NUM + ConstInfo::BUFFER_SIZE_BYTE_4K],
            antiquantScaleUb_[constInfo.headDim + i * FP32_REPEAT_ELEMENT_NUM], FP32_REPEAT_ELEMENT_NUM, dealRow, {1, 1, 1, 16, 16, 0});
    }
    PipeBarrier<PIPE_V>();

    LocalTensor<KV_ORIGIN_T> antiKvTensorAsB16 = outputBuff1.Get<KV_ORIGIN_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    if constexpr (IsSameType<KV_ORIGIN_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
        Cast(antiKvTensorAsB16, kvFp32, RoundMode::CAST_RINT, elementNum);
    } else {
        Cast(antiKvTensorAsB16, kvFp32, RoundMode::CAST_ROUND, elementNum);
    }
    uint64_t blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = dealRow;
    dataCopyParams.blockLen = constInfo.headDim * sizeof(KV_ORIGIN_T);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopyPad(keyMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * constInfo.s2BaseSize * constInfo.headDim + (s2GmStartOffset + mte3Size) *
        constInfo.headDim], antiKvTensorAsB16, dataCopyParams);
    DataCopyPad(valueMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * constInfo.s2BaseSize * constInfo.headDim + (s2GmStartOffset + mte3Size) *
        constInfo.headDim], antiKvTensorAsB16[ConstInfo::BUFFER_SIZE_BYTE_4K], dataCopyParams);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

// b s1 k
template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqa<SFAAT>::ProcessVec0L(const RunInfo &info)
{
    int64_t s2ProcessSize = info.actualSingleProcessSInnerSize;
    int64_t s2Pair = SfaaCeilDiv(s2ProcessSize, 2L * constInfo.sparseBlockSize);
    int64_t topkGmBaseOffset = info.topkGmBaseOffset;

    int64_t s2GmStartOffset = GetSubBlockIdx() == 0 ? 0 : SfaaCeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize;
    int64_t s2GmLimit = GetSubBlockIdx() == 0 ? SfaaCeilDiv(s2Pair, 2L) * 2 * constInfo.sparseBlockSize: s2ProcessSize;
    if (s2GmLimit > s2ProcessSize) {
        s2GmLimit = s2ProcessSize;
    }
    
    if (info.bN2Idx != lastBN2Idx) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = constInfo.headDim * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        DataCopyPadExtParams<T> padParams;
        padParams.isPad = false;
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_DEQUANT_SCALE_FLAG);
        uint32_t srcOffset = info.n2Idx * constInfo.headDim;
        DataCopyPad(antiquantScaleUb_, keyDequantScaleGm_[srcOffset], dataCopyParams, padParams);
        DataCopyPad(antiquantScaleUb_[constInfo.headDim], valueDequantScaleGm_[srcOffset], dataCopyParams, padParams);
        SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_DEQUANT_SCALE_FLAG);
        WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_DEQUANT_SCALE_FLAG);
    }
    int64_t mergeMte3Idx = 0;
    int64_t mte2Size = MergeKv(info, s2GmStartOffset, s2GmLimit, topkGmBaseOffset, false, mergeMte3Idx);
    if (info.bN2Idx != lastBN2Idx) {
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_DEQUANT_SCALE_FLAG);
        lastBN2Idx = info.bN2Idx;
    }

    v0ValidSizeUb_.SetValue(info.loop % MERGE_CACHE_GM_BUF_NUM, mte2Size);
    SetFlag<AscendC::HardEvent::S_MTE3>(1);
    WaitFlag<AscendC::HardEvent::S_MTE3>(1);
    DataCopyExtParams dataCopyParams;
    dataCopyParams.blockCount = 1;
    dataCopyParams.blockLen = 128 * sizeof(int32_t);
    dataCopyParams.srcStride = 0;
    dataCopyParams.dstStride = 0;
    DataCopyPad(kvValidSizeGm_[info.loop % MERGE_CACHE_GM_BUF_NUM * (128 * 2) + GetSubBlockIdx() * 128],
                v0ValidSizeUb_, dataCopyParams);
    return;
}

template <typename SFAAT>
__aicore__ inline int64_t SFAAVectorServiceGqa<SFAAT>::MergeKv(const RunInfo &runInfo, int64_t s2GmStartOffset, int64_t s2GmLimit, int64_t topkGmBaseOffset, bool isValue, int64_t mergeMte3Idx)
{
    int64_t mte2Size = 0;
    int64_t mte3Size = 0;
    int64_t s2IdxArray0 = -1;
    int64_t s2IdxArray1 = -1;
    bool needWaitMte3ToMte2 = true;
    bool mask = false;

    for (int64_t s2GmOffsetArray = s2GmStartOffset; s2GmOffsetArray < s2GmLimit; s2GmOffsetArray += 2 *
        constInfo.sparseBlockSize) {
        if (needWaitMte3ToMte2) {
            WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + mergeMte3Idx % 2);
            needWaitMte3ToMte2 = false;
        }
        GetRealS2Idx(s2GmOffsetArray, s2IdxArray0, topkGmBaseOffset, runInfo);
        if (unlikely(s2IdxArray0 < 0)) {
            CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, runInfo, isValue, needWaitMte3ToMte2);
            mergeMte3Idx++;
            break;
        }
        GetRealS2Idx(s2GmOffsetArray + constInfo.sparseBlockSize, s2IdxArray1, topkGmBaseOffset, runInfo);
        CopyInKv(mte2Size, mte3Size, mergeMte3Idx, s2IdxArray0, s2IdxArray1, runInfo, mask);
        if ((mte2Size - mte3Size + 2 * constInfo.sparseBlockSize > 32) || // TODO，这里可以支持到128，但需要循环拷出
            s2GmOffsetArray + 2 * constInfo.sparseBlockSize >= s2GmLimit) {
            CopyOutMrgeResult(mte2Size, mte3Size, s2GmStartOffset, mergeMte3Idx, runInfo, isValue, needWaitMte3ToMte2);
            mte3Size = mte2Size;
            mergeMte3Idx++;
        }
    }

    if (unlikely(s2GmStartOffset + mte2Size < s2GmLimit)) {
        SetFlag<AscendC::HardEvent::MTE3_V>(0);
        WaitFlag<AscendC::HardEvent::MTE3_V>(0);
        LocalTensor<KV_ORIGIN_T> mergeUb = kvMergUb_.template ReinterpretCast<KV_ORIGIN_T>();
        Duplicate(mergeUb, static_cast<KV_ORIGIN_T>(0.0), constInfo.headDim);
        SetFlag<AscendC::HardEvent::V_MTE3>(0);
        WaitFlag<AscendC::HardEvent::V_MTE3>(0);

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = constInfo.headDim * sizeof(KV_ORIGIN_T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;
        for (int64_t s2GmOffset = s2GmStartOffset + mte2Size; s2GmOffset < s2GmLimit; s2GmOffset++) {
            DataCopyPad(keyMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * constInfo.s2BaseSize * constInfo.headDim + s2GmOffset * constInfo.headDim],
                        mergeUb, dataCopyParams);
            DataCopyPad(valueMergeGm_[runInfo.loop % MERGE_CACHE_GM_BUF_NUM * constInfo.s2BaseSize * constInfo.headDim + s2GmOffset * constInfo.headDim],
                        mergeUb, dataCopyParams);
        }
        mergeMte3Idx++;
    }
    return mte2Size;
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqa<SFAAT>::ProcessVec1L(const RunInfo &info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferIdx = i;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;

        mSplitInfo.vecDealM = (mSplitInfo.nBufferDealM <= 16) ? mSplitInfo.nBufferDealM :
                                                                (((mSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16);
        mSplitInfo.vecStartM = 0;
        if (GetBlockIdx() % 2 == 1) {
            mSplitInfo.vecStartM = mSplitInfo.vecDealM;
            mSplitInfo.vecDealM = mSplitInfo.nBufferDealM - mSplitInfo.vecDealM;
        }

        CrossCoreWaitFlag(constInfo.syncC1V1);
        // vec1 compute
        ProcessVec1SingleBuf(info, mSplitInfo);
        CrossCoreSetFlag<ConstInfo::SFAA_SYNC_MODE2, PIPE_MTE3>(constInfo.syncV1C2);
        // move lse for flash decode
        if (info.s2Idx == info.curSInnerLoopTimes - 1) {
            if (info.tndIsS2SplitCore) {
                if (FLASH_DECODE) {
                    uint32_t outIdx = info.loop % (constInfo.preLoadNum);
                    auto sumTensor = softmaxSumUb[outIdx * SOFTMAX_TMP_BUFFER_OFFSET];
                    auto maxTensor = softmaxMaxUb[outIdx * SOFTMAX_TMP_BUFFER_OFFSET];
                    ComputeLogSumExpAndCopyToGm(info, mSplitInfo, sumTensor, maxTensor);
                }
            }
        }
    }
}

template <typename SFAAT>
__aicore__ inline uint64_t SFAAVectorServiceGqa<SFAAT>::CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx)
{
    uint64_t accumTmpOutNum = 0;
    uint32_t taskId = 0;

    if (metaDataPtr != nullptr) {
        const uint32_t *bN2IdxOfFdHead = metaDataPtr->fdRes.bN2IdxOfFdHead;
        const uint32_t *gS1IdxOfFdHead = metaDataPtr->fdRes.gS1IdxOfFdHead;
        const uint32_t *s2SplitNumOfFdHead = metaDataPtr->fdRes.s2SplitNumOfFdHead;
        uint32_t usedCoreNum = metaDataPtr->usedCoreNum;
        while (taskId < usedCoreNum && (bN2IdxOfFdHead[taskId] != bN2Idx || gS1IdxOfFdHead[taskId] * constInfo.mBaseSize != gS1Idx)) {  // 考虑在tiling阶段直接算出accumOut的偏置，则可以省略CalcAccumOffset()
            accumTmpOutNum += s2SplitNumOfFdHead[taskId]; // 计算前面的workspace数
            taskId++;
        }
    } else {
        #ifdef ASCENDC_CPU_DEBUG
            const uint32_t *bN2IdxOfFdHead = tilingData->fdParams.bN2IdxOfFdHead;
            const uint32_t *gS1IdxOfFdHead = tilingData->fdParams.gS1IdxOfFdHead;
            const uint32_t *s2SplitNumOfFdHead = tilingData->fdParams.s2SplitNumOfFdHead;
        #else
            uint32_t bN2IdxOfFdHead[ARRAY_SIZE(tilingData->fdParams.bN2IdxOfFdHead)];
            uint32_t gS1IdxOfFdHead[ARRAY_SIZE(tilingData->fdParams.gS1IdxOfFdHead)];
            uint32_t s2SplitNumOfFdHead[ARRAY_SIZE(tilingData->fdParams.s2SplitNumOfFdHead)];
            copy_data_align64((uint8_t *)bN2IdxOfFdHead, (uint8_t *)(tilingData->fdParams.bN2IdxOfFdHead),
                            sizeof(bN2IdxOfFdHead));
            copy_data_align64((uint8_t *)gS1IdxOfFdHead, (uint8_t *)(tilingData->fdParams.gS1IdxOfFdHead),
                            sizeof(gS1IdxOfFdHead));
            copy_data_align64((uint8_t *)s2SplitNumOfFdHead, (uint8_t *)(tilingData->fdParams.s2SplitNumOfFdHead),
                            sizeof(s2SplitNumOfFdHead));
        #endif
        uint32_t usedCoreNum = tilingData->singleCoreParams.usedCoreNum;
        while (taskId < usedCoreNum && (bN2IdxOfFdHead[taskId] != bN2Idx || gS1IdxOfFdHead[taskId] * constInfo.mBaseSize != gS1Idx)) {  // 考虑在tiling阶段直接算出accumOut的偏置，则可以省略CalcAccumOffset()
            accumTmpOutNum += s2SplitNumOfFdHead[taskId]; // 计算前面的workspace数
            taskId++;
        }
    }
    return accumTmpOutNum;
}

// template <typename SFAAT>
// __aicore__ inline uint64_t SFAAVectorServiceGqa<SFAAT>::CalcAccumOffset(uint32_t bN2Idx, uint32_t gS1Idx)
// {

//     #ifdef ASCENDC_CPU_DEBUG
//         const uint32_t *bN2IdxOfFdHead = tilingData->fdParams.bN2IdxOfFdHead;
//         const uint32_t *gS1IdxOfFdHead = tilingData->fdParams.gS1IdxOfFdHead;
//         const uint32_t *s2SplitNumOfFdHead = tilingData->fdParams.s2SplitNumOfFdHead;
//     #else
//         uint32_t bN2IdxOfFdHead[ARRAY_SIZE(tilingData->fdParams.bN2IdxOfFdHead)];
//         uint32_t gS1IdxOfFdHead[ARRAY_SIZE(tilingData->fdParams.gS1IdxOfFdHead)];
//         uint32_t s2SplitNumOfFdHead[ARRAY_SIZE(tilingData->fdParams.s2SplitNumOfFdHead)];
//         copy_data_align64((uint8_t *)bN2IdxOfFdHead, (uint8_t *)(tilingData->fdParams.bN2IdxOfFdHead),
//                         sizeof(bN2IdxOfFdHead));
//         copy_data_align64((uint8_t *)gS1IdxOfFdHead, (uint8_t *)(tilingData->fdParams.gS1IdxOfFdHead),
//                         sizeof(gS1IdxOfFdHead));
//         copy_data_align64((uint8_t *)s2SplitNumOfFdHead, (uint8_t *)(tilingData->fdParams.s2SplitNumOfFdHead),
//                         sizeof(s2SplitNumOfFdHead));
//     #endif
//     uint32_t usedCoreNum = tilingData->singleCoreParams.usedCoreNum;

//     uint64_t accumTmpOutNum = 0;
//     uint32_t taskId = 0;
//     while (taskId < usedCoreNum && (bN2IdxOfFdHead[taskId] != bN2Idx || gS1IdxOfFdHead[taskId] * constInfo.mBaseSize != gS1Idx)) {  // 考虑在tiling阶段直接算出accumOut的偏置，则可以省略CalcAccumOffset()
//         accumTmpOutNum += s2SplitNumOfFdHead[taskId]; // 计算前面的workspace数
//         taskId++;
//     }
//     return accumTmpOutNum;
// }

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqa<SFAAT>::ProcessVec2SingleBuf(const RunInfo &info,
                                                                      const MSplitInfo &mSplitInfo)
{
    if (mSplitInfo.vecDealM == 0) {
        return;
    }

    uint32_t gPreSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / constInfo.headDim;
    if (gPreSplitSize > mSplitInfo.vecDealM) {
        gPreSplitSize = mSplitInfo.vecDealM;
    }
    uint32_t loopCount = (mSplitInfo.vecDealM + gPreSplitSize - 1) / gPreSplitSize;
    uint32_t tailSplitSize = mSplitInfo.vecDealM - (loopCount - 1) * gPreSplitSize;
    for (uint32_t i = 0, dealSize = gPreSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm2ResBaseBlock(info, mSplitInfo, i * gPreSplitSize, dealSize, constInfo.headDim, constInfo.headDim);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqa<SFAAT>::DealBmm2ResBaseBlock(const RunInfo &info, const MSplitInfo &mSplitInfo,
                                                                      uint32_t startRow, uint32_t dealRowCount,
                                                                      uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t vec2ComputeSize = dealRowCount * columnCount;
    uint32_t baseOffset = startRow;
    LocalTensor<T> bmm2ResUb = tmpBuff1.Get<T>();
    bmm2ResUb.SetSize(vec2ComputeSize);

    size_t batchBase = 0;
    uint64_t inOutBaseOffset = (mSplitInfo.vecStartM + startRow) * columnCount;
    uint64_t srcGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;
    
    LocalTensor<MM2_OUT_T> tmpBmm2ResUb = inputBuff1.Get<MM2_OUT_T>();
    tmpBmm2ResUb = tmpBmm2ResUb[pingpongFlag * INPUT1_BUFFER_OFFSET / sizeof(MM2_OUT_T)];
    WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    DataCopy(tmpBmm2ResUb, mm2ResGm[srcGmOffset + batchBase], vec2ComputeSize);
    SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF1_FLAG);
    DataCopy(bmm2ResUb, tmpBmm2ResUb, vec2ComputeSize);
    SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF1_FLAG + pingpongFlag);

    // 除第一个循环外，均需要更新中间计算结果
    if (info.s2Idx > 0) {
        event_t eventIdMte2WaitMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte2WaitMte3);
        LocalTensor<T> bmm2ResPreUb = inputBuff2.Get<T>();
        WaitFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
        uint64_t vecPre2ResGmOffset = ((info.loop - 1) % constInfo.preLoadNum) * constInfo.bmm2ResUbSize +
            inOutBaseOffset;
        DataCopy(bmm2ResPreUb, vec2ResGm[vecPre2ResGmOffset + batchBase], vec2ComputeSize);
        SetFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF2_FLAG);
        WaitFlag<AscendC::HardEvent::MTE2_V>(SYNC_INPUT_BUF2_FLAG);
        LocalTensor<T> softmaxExpBrcb =  tmpBuff2.Get<T>();
        Brcb(softmaxExpBrcb, softmaxExpUb[(info.loop % constInfo.preLoadNum) * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset],
            (mSplitInfo.vecDealM + 7) / 8, {1, 8});
        pipe_barrier(PIPE_V);
        RowMuls(bmm2ResPreUb, bmm2ResPreUb, softmaxExpBrcb, dealRowCount, columnCount, actualColumnCount);
        pipe_barrier(PIPE_V);
        Add(bmm2ResUb, bmm2ResUb, bmm2ResPreUb, vec2ComputeSize);
        SetFlag<AscendC::HardEvent::V_MTE2>(SYNC_INPUT_BUF2_FLAG);
    }
    // 最后一次输出计算结果，否则将中间结果暂存至workspace
    if (info.s2Idx + 1 == info.curSInnerLoopTimes) {
        LocalTensor<T> softmaxSumBrcb =  tmpBuff2.Get<T>();
        Brcb(softmaxSumBrcb, softmaxSumUb[(info.loop % constInfo.preLoadNum) * SOFTMAX_TMP_BUFFER_OFFSET + baseOffset],
            (mSplitInfo.vecDealM + 7) / 8, {1, 8});
        pipe_barrier(PIPE_V);
        RowDivs(bmm2ResUb, bmm2ResUb, softmaxSumBrcb, dealRowCount, columnCount, actualColumnCount);

        pipe_barrier(PIPE_V);
        Bmm2ResCopyOut(info, bmm2ResUb, mSplitInfo.vecStartM + startRow, dealRowCount, columnCount, actualColumnCount);
    } else {
        pipe_barrier(PIPE_V);
        LocalTensor<T> tmpBmm2Res = outputBuff1.Get<T>();
        WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
        DataCopy(tmpBmm2Res, bmm2ResUb, dealRowCount * columnCount);
        SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
        WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);

        uint64_t vecPre2ResGmOffset = (info.loop % constInfo.preLoadNum) * constInfo.bmm2ResUbSize + inOutBaseOffset;
        DataCopy(vec2ResGm[vecPre2ResGmOffset + batchBase], tmpBmm2Res, vec2ComputeSize);
        SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    }
}

template <typename SFAAT> __aicore__ inline void SFAAVectorServiceGqa<SFAAT>::ProcessVec2L(const RunInfo &info)
{
    uint32_t nBufferLoopTimes = (info.actMBaseSize + constInfo.nBufferMBaseSize - 1) / constInfo.nBufferMBaseSize;
    uint32_t nBufferTail = info.actMBaseSize - (nBufferLoopTimes - 1) * constInfo.nBufferMBaseSize;
    for (uint32_t i = 0; i < nBufferLoopTimes; i++) {
        MSplitInfo mSplitInfo;
        mSplitInfo.nBufferIdx = i;
        mSplitInfo.nBufferStartM = i * constInfo.nBufferMBaseSize;
        mSplitInfo.nBufferDealM = (i + 1 != nBufferLoopTimes) ? constInfo.nBufferMBaseSize : nBufferTail;

        mSplitInfo.vecDealM = (mSplitInfo.nBufferDealM <= 16) ? mSplitInfo.nBufferDealM :
            (((mSplitInfo.nBufferDealM + 15) / 16 + 1) / 2 * 16);
        mSplitInfo.vecStartM = 0;
        if (GetBlockIdx() % 2 == 1) {
            mSplitInfo.vecStartM = mSplitInfo.vecDealM;
            mSplitInfo.vecDealM = mSplitInfo.nBufferDealM - mSplitInfo.vecDealM;
        }
        CrossCoreWaitFlag(constInfo.syncC2V2);
        ProcessVec2SingleBuf(info, mSplitInfo);
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqa<SFAAT>::ProcessVec2Inner(const RunInfo &info,
                                                                  const MSplitInfo &mSplitInfo,
                                                                  uint32_t mStartRow, uint32_t mDealSize)
{
    uint32_t mSplitSize = BASE_BLOCK_MAX_ELEMENT_NUM / constInfo.headDim;
    if (mSplitSize > mDealSize) {
        mSplitSize = mDealSize;
    }

    uint32_t loopCount = (mDealSize + mSplitSize - 1) / mSplitSize;
    uint32_t tailSplitSize = mDealSize - (loopCount - 1) * mSplitSize;
    for (uint32_t i = 0, dealSize = mSplitSize; i < loopCount; i++) {
        if (i == (loopCount - 1)) {
            dealSize = tailSplitSize;
        }
        DealBmm2ResBaseBlock(info, mSplitInfo, i * mSplitSize + mStartRow, dealSize,
                             constInfo.headDim, constInfo.headDim);
        pingpongFlag ^= 1; // pingpong 0 1切换
    }
}


template <typename SFAAT>
__aicore__ inline void SFAAVectorServiceGqa<SFAAT>::GetConfusionTransposeTiling(
    int64_t numR, int64_t numC, const uint32_t stackBufferSize, const uint32_t typeSize,
    ConfusionTransposeTiling &tiling)
{
    (void)stackBufferSize;
    uint32_t blockSize = ONE_BLK_SIZE / typeSize;
    uint32_t height = numC;
    uint32_t width = numR;
    uint32_t highBlock = height / BLOCK_CUBE;
    uint32_t stride = height * blockSize * typeSize / ONE_BLK_SIZE;
    uint32_t repeat = width / blockSize;

    tiling.param0 = blockSize;
    tiling.param1 = height;
    tiling.param2 = width;
    tiling.param3 = highBlock;
    tiling.param4 = stride;
    tiling.param5 = repeat;
}

template <typename SFAAT>
__aicore__ inline void
SFAAVectorServiceGqa<SFAAT>::Bmm2FDDataCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb,
                                            uint32_t wsMStart, uint32_t dealRowCount, uint32_t columnCount,
                                            uint32_t actualColumnCount)
{
    LocalTensor<T> tmp = outputBuff1.Get<T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    DataCopy(tmp, bmm2ResUb, columnCount * dealRowCount);
    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    uint64_t accumTmpOutNum = CalcAccumOffset(info.bN2Idx, info.gS1Idx);
    uint64_t offset = accumTmpOutNum * constInfo.mBaseSize * constInfo.headDim + // taskoffset
                      info.tndCoreStartKVSplitPos * constInfo.mBaseSize * constInfo.headDim + // 份数offset
                      wsMStart * actualColumnCount; // m轴offset
    GlobalTensor<T> dst = accumOutGm[offset];
    if (info.actualSingleProcessSInnerSize != 0) {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = dealRowCount;
        dataCopyParams.blockLen = actualColumnCount * sizeof(T);
        dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(T));
        dataCopyParams.dstStride = 0;
        DataCopyPad(dst, tmp, dataCopyParams);
    } else {
        matmul::InitOutput<T>(dst, dealRowCount * actualColumnCount, ConstInfo::FLOAT_ZERO);
    }
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename SFAAT>
__aicore__ inline void
SFAAVectorServiceGqa<SFAAT>::Bmm2DataCopyOutTrans(const RunInfo &info, LocalTensor<OUT_T> &attenOutUb,
                                               uint32_t wsMStart, uint32_t dealRowCount,
                                               uint32_t columnCount, uint32_t actualColumnCount)
{
    uint32_t s1StartIdx = wsMStart / constInfo.gSize;
    uint32_t startGOffset = wsMStart % constInfo.gSize;
    uint32_t s1EndIdx = SfaaCeilDiv(wsMStart + dealRowCount, static_cast<uint32_t>(constInfo.gSize)) - 1;
    uint32_t curStartRow = wsMStart;
    uint32_t curDealRowCount = 0;
    uint32_t ubOffset = 0;
    for (uint32_t curS1idx = s1StartIdx; curS1idx <= s1EndIdx; curS1idx++) {
        uint32_t outOffset = info.attenOutOffset + curS1idx * constInfo.qHeadNum * constInfo.headDim + startGOffset * constInfo.headDim;
        if (curS1idx != s1EndIdx) {
            curDealRowCount = (curS1idx + 1) * constInfo.gSize - curStartRow;
        } else {
            curDealRowCount = wsMStart + dealRowCount - curStartRow;
        }
        ubOffset = (curStartRow - wsMStart) * columnCount;
        LocalTensor<OUT_T> curAttenOutUb = attenOutUb[ubOffset];
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = curDealRowCount;
        dataCopyParams.blockLen = actualColumnCount * sizeof(OUT_T);
        dataCopyParams.srcStride = (columnCount - actualColumnCount) / (BYTE_BLOCK / sizeof(OUT_T));
        dataCopyParams.dstStride = 0;
        DataCopyPad(attentionOutGm[outOffset], curAttenOutUb, dataCopyParams);
        curStartRow += curDealRowCount;
        startGOffset = 0;
    }
}

template <typename SFAAT>
__aicore__ inline void
SFAAVectorServiceGqa<SFAAT>::Bmm2CastAndCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb,
                                             uint32_t wsMStart, uint32_t dealRowCount, uint32_t columnCount,
                                             uint32_t actualColumnCount)
{
    LocalTensor<OUT_T> tmpBmm2ResCastTensor = outputBuff1.Get<OUT_T>();
    WaitFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
    if constexpr (IsSameType<OUT_T, bfloat16_t>::value) { // bf16 采取四舍六入五成双模式
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_RINT, dealRowCount * columnCount);
    } else {
        Cast(tmpBmm2ResCastTensor, bmm2ResUb, AscendC::RoundMode::CAST_ROUND, dealRowCount * columnCount);
    }

    SetFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    WaitFlag<AscendC::HardEvent::V_MTE3>(SYNC_OUTPUT_BUF1_FLAG);
    Bmm2DataCopyOutTrans(info, tmpBmm2ResCastTensor, wsMStart, dealRowCount, columnCount, actualColumnCount);
    SetFlag<AscendC::HardEvent::MTE3_V>(SYNC_OUTPUT_BUF1_FLAG);
}

template <typename SFAAT>
__aicore__ inline void
SFAAVectorServiceGqa<SFAAT>::Bmm2ResCopyOut(const RunInfo &info, LocalTensor<T> &bmm2ResUb, uint32_t wsMStart,
                                         uint32_t dealRowCount, uint32_t columnCount,
                                         uint32_t actualColumnCount)
{
    if (FLASH_DECODE) {
        if (info.tndIsS2SplitCore) {
            Bmm2FDDataCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
        } else {
            Bmm2CastAndCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
        }
    } else {
        Bmm2CastAndCopyOut(info, bmm2ResUb, wsMStart, dealRowCount, columnCount, actualColumnCount);
    }
}

template <typename SFAAT>
__aicore__ inline void
SFAAVectorServiceGqa<SFAAT>::RowDivs(LocalTensor<float> dstUb, LocalTensor<float> src0Ub, LocalTensor<float> src1Ub,
                                  uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // divs by row, 每行的元素除以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] / src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount], src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    uint32_t dtypeMask = FP32_REPEAT_ELEMENT_NUM;
    uint32_t dLoop = actualColumnCount / dtypeMask;
    uint32_t dRemain = actualColumnCount % dtypeMask;

    BinaryRepeatParams repeatParamsDiv;
    repeatParamsDiv.src0BlkStride = 1;
    repeatParamsDiv.src1BlkStride = 0;
    repeatParamsDiv.dstBlkStride = 1;
    repeatParamsDiv.src0RepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    repeatParamsDiv.src1RepStride = 1;
    repeatParamsDiv.dstRepStride = columnCount / FP32_BLOCK_ELEMENT_NUM;
    uint32_t columnRepeatCount = dLoop;
    if (columnRepeatCount <= dealRowCount) {
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dLoop; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub, dtypeMask, dealRowCount, repeatParamsDiv);
            offset += dtypeMask;
        }
    } else {
        BinaryRepeatParams columnRepeatParams;
        columnRepeatParams.src0BlkStride = 1;
        columnRepeatParams.src1BlkStride = 0;
        columnRepeatParams.dstBlkStride = 1;
        columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        columnRepeatParams.src1RepStride = 0;
        columnRepeatParams.dstRepStride = 8;  // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
        uint32_t offset = 0;
        for (uint32_t i = 0; i < dealRowCount; i++) {
            Div(dstUb[offset], src0Ub[offset], src1Ub[i * FP32_BLOCK_ELEMENT_NUM], dtypeMask, columnRepeatCount,
                columnRepeatParams);
            offset += columnCount;
        }
    }
    if (dRemain > 0) {
        Div(dstUb[dLoop * dtypeMask], src0Ub[dLoop * dtypeMask], src1Ub, dRemain, dealRowCount, repeatParamsDiv);
    }
}

template <typename SFAAT>
__aicore__ inline void
SFAAVectorServiceGqa<SFAAT>::RowMuls(LocalTensor<T> dstUb, LocalTensor<T> src0Ub, LocalTensor<T> src1Ub,
                                  uint32_t dealRowCount, uint32_t columnCount, uint32_t actualColumnCount)
{
    // muls by row, 每行的元素乘以相同的元素
    // dstUb[i, (j * 8) : (j * 8 + 7)] = src0Ub[i, (j * 8) : (j * 8 + 7)] * src1Ub[i, 0 : 7]
    // src0Ub:[dealRowCount, columnCount] src1Ub:[dealRowCount, FP32_BLOCK_ELEMENT_NUM] dstUb:[dealRowCount,
    // columnCount]
    // dealRowCount is repeat times, must be less 256
    uint32_t repeatElementNum = FP32_REPEAT_ELEMENT_NUM;
    uint32_t blockElementNum = FP32_BLOCK_ELEMENT_NUM;

    if constexpr (std::is_same<T, half>::value) {
        // 此限制由于每个repeat至多连续读取256B数据
        repeatElementNum = FP32_REPEAT_ELEMENT_NUM * 2; // 256/4 * 2=128
        blockElementNum = FP32_BLOCK_ELEMENT_NUM * 2;   // 32/4 * 2 = 16
    }

    // 每次只能连续读取256B的数据进行计算，故每次只能处理256B/sizeof(dType)=
    // 列方向分dLoop次，每次处理8列数据
    uint32_t dLoop = actualColumnCount / repeatElementNum;
    uint32_t dRemain = actualColumnCount % repeatElementNum;
    // REPEATE_STRIDE_UP_BOUND=256， 此限制由于src0RepStride数据类型为uint8之多256个datablock间距
    if (columnCount < REPEATE_STRIDE_UP_BOUND * blockElementNum) {
        BinaryRepeatParams repeatParams;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstBlkStride = 1;
        repeatParams.src0RepStride = columnCount / blockElementNum;
        repeatParams.src1RepStride = 1;
        repeatParams.dstRepStride = columnCount / blockElementNum;

        // 如果以列为repeat所处理的次数小于行处理次数，则以列方式处理。反之则以行进行repeat处理
        if (dLoop <= dealRowCount) {
            uint32_t offset = 0;
            for (uint32_t i = 0; i < dLoop; i++) {
                Mul(dstUb[offset], src0Ub[offset], src1Ub, repeatElementNum, dealRowCount, repeatParams);
                offset += repeatElementNum;
            }
        } else {
            BinaryRepeatParams columnRepeatParams;
            columnRepeatParams.src0BlkStride = 1;
            columnRepeatParams.src1BlkStride = 0;
            columnRepeatParams.dstBlkStride = 1;
            columnRepeatParams.src0RepStride = 8; // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            columnRepeatParams.src1RepStride = 0;
            columnRepeatParams.dstRepStride = 8;  // 列方向上两次repeat起始地址间隔dtypeMask=64个元素，即8个block
            for (uint32_t i = 0; i < dealRowCount; i++) {
                Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum], repeatElementNum,
                    dLoop, columnRepeatParams);
            }
        }

        // 最后一次完成[dealRowCount, dRemain] * [dealRowCount, blockElementNum] 只计算有效部分
        if (dRemain > 0) {
            Mul(dstUb[dLoop * repeatElementNum], src0Ub[dLoop * repeatElementNum], src1Ub, dRemain, dealRowCount,
                repeatParams);
        }
    } else {
        BinaryRepeatParams repeatParams;
        repeatParams.src0RepStride = 8; // 每个repeat为256B数据，正好8个datablock
        repeatParams.src0BlkStride = 1;
        repeatParams.src1RepStride = 0;
        repeatParams.src1BlkStride = 0;
        repeatParams.dstRepStride = 8;
        repeatParams.dstBlkStride = 1;
        // 每次计算一行，共计算dealRowCount行
        for (uint32_t i = 0; i < dealRowCount; i++) {
            // 计算一行中的dLoop个repeat, 每个repeat计算256/block_size 个data_block
            Mul(dstUb[i * columnCount], src0Ub[i * columnCount], src1Ub[i * blockElementNum], repeatElementNum, dLoop,
                repeatParams);
            //  计算一行中的尾块
            if (dRemain > 0) {
                Mul(dstUb[i * columnCount + dLoop * repeatElementNum],
                    src0Ub[i * columnCount + dLoop * repeatElementNum], src1Ub[i * blockElementNum], dRemain, 1,
                    repeatParams);
            }
        }
    }
}

#endif // SPARSE_FLASH_ATTENTION_ANTIQUANT_SERVICE_VECTOR_GQA_H
