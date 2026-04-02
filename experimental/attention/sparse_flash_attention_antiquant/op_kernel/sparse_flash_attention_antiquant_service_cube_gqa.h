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
 * \file sparse_flash_attention_antiquant_service_cube_gqa.h
 * \brief use 7 buffer for matmul l1, better pipeline
 */
#ifndef SPARSE_FLASH_ATTENTION_ANTIQUANT_SERVICE_CUBE_GQA_H
#define SPARSE_FLASH_ATTENTION_ANTIQUANT_SERVICE_CUBE_GQA_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "sparse_flash_attention_antiquant_common.h"

template <typename SFAAT> class SFAAMatmulServiceGqa {
public:
    // 中间计算数据类型为float, 高精度模式
    using T = float;
    using Q_T = typename SFAAT::queryType;
    using KV_T = typename SFAAT::kvType;
    using KV_ORIGIN_T = typename SFAAT::queryType;
    using OUT_T = typename SFAAT::outputType;
    using MM_OUT_T = T;

    __aicore__ inline SFAAMatmulServiceGqa(){};
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void InitMm1GlobalTensor(GlobalTensor<Q_T> queryGm, GlobalTensor<KV_T> keyGm,
                                               GlobalTensor<MM_OUT_T> mm1ResGm);
    __aicore__ inline void InitMm2GlobalTensor(GlobalTensor<KV_ORIGIN_T> vec1ResGm, GlobalTensor<KV_T> valueGm,
                                               GlobalTensor<MM_OUT_T> mm2ResGm, GlobalTensor<OUT_T> attentionOutGm);
    __aicore__ inline void InitPageAttentionInfo(const GlobalTensor<KV_ORIGIN_T>& keyMergeGm, const GlobalTensor<KV_ORIGIN_T>& valueMergeGm,
                                                 GlobalTensor<int32_t> blockTableGm, GlobalTensor<int32_t> topKGm,
                                                 uint32_t blockSize, uint32_t maxBlockNumPerBatch);
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void UpdateKey(GlobalTensor<KV_T> keyGm);
    __aicore__ inline void UpdateValue(GlobalTensor<KV_T> valueGm);

    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void CalcTopKBlockInfo(const RunInfo &info, uint32_t &curTopKIdx,
                                             uint64_t &curOffsetInSparseBlock, uint32_t curSeqIdx,
                                             uint32_t &copyRowCnt, uint64_t &idInTopK);
    __aicore__ inline void ComputeMm1(const RunInfo &info, const MSplitInfo mSplitInfo);
    __aicore__ inline void ComputeMm2(const RunInfo &info, const MSplitInfo mSplitInfo);

private:
    static constexpr bool PAGE_ATTENTION = SFAAT::pageAttention;
    static constexpr int TEMPLATE_MODE = SFAAT::templateMode;
    // static constexpr bool FLASH_DECODE = SFAAT::flashDecode;
    static constexpr SFAA_LAYOUT LAYOUT_T = SFAAT::layout;
    static constexpr SFAA_LAYOUT KV_LAYOUT_T = SFAAT::kvLayout;

    // L1轴切分大小
    static constexpr uint32_t S2_SPLIT_SIZE = 256;     // S2方向切分，对于mm1是N轴，对于mm2是K轴

    // L0轴切分大小
    static constexpr uint32_t M_BASE_SIZE = 128;     // m方向基本块大小
    static constexpr uint32_t K_BASE_SIZE = 128;     // k方向基本块大小
    static constexpr uint32_t N_BASE_SIZE = 128;     // n方向基本块大小

    static constexpr uint32_t L1Q_BLOCK_SIZE= 32 * 1024; // 32K
    static constexpr uint32_t L1_BLOCK_SIZE = 64 * 1024; // 64K
    static constexpr uint32_t L1_BLOCK_OFFSET = L1_BLOCK_SIZE / sizeof(KV_ORIGIN_T);

    static constexpr uint32_t L0A_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0B_PP_SIZE = (32 * 1024);
    static constexpr uint32_t L0C_PP_SIZE = (64 * 1024);
    static constexpr uint32_t L0AB_BLOCK_OFFSET = L0A_PP_SIZE / sizeof(KV_ORIGIN_T);
    static constexpr uint32_t L0C_BLOCK_OFFSET = L0C_PP_SIZE / sizeof(MM_OUT_T);

    // mte2 <> mte1 EventID
    // L1 3buf, 使用3个eventId
    static constexpr uint32_t L1Q_EVENT0 = EVENT_ID0;
    static constexpr uint32_t L1KP_EVENT0 = EVENT_ID1;
    static constexpr uint32_t L1KP_EVENT1 = EVENT_ID2;
    static constexpr uint32_t L1KP_EVENT2 = EVENT_ID3;
    static constexpr uint32_t L1V_EVENT0 = EVENT_ID4;
    static constexpr uint32_t L1V_EVENT1 = EVENT_ID5;
    static constexpr uint32_t L1V_EVENT2 = EVENT_ID6;
    static constexpr uint32_t L1V_BUFFER_NUM = 3;
    static constexpr uint32_t L1KP_BUFFER_NUM = 3;

    // m <> mte1 EventID
    static constexpr uint32_t L0AB_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0AB_EVENT1 = EVENT_ID4;

    // fix <> m
    static constexpr uint32_t L0C_EVENT0 = EVENT_ID3;
    static constexpr uint32_t L0C_EVENT1 = EVENT_ID4;

    static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true}; // isSetFMatrix isSetPadding;

    static constexpr uint32_t BLOCK_ELEMENT_NUM = ConstInfo::BUFFER_SIZE_BYTE_32B / sizeof(KV_ORIGIN_T);

    uint32_t kvCacheBlockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    ConstInfo constInfo{};

    // L1分成3块buf, 用于记录
    uint32_t kpL1BufIter = -1;
    uint32_t vL1BufIter = -1;
    uint32_t abL0BufIter = 0;
    uint32_t cL0BufIter = 0;

    // mm1
    GlobalTensor<Q_T> queryGm;
    GlobalTensor<KV_T> keyGm;
    GlobalTensor<MM_OUT_T> mm1ResGm;
    GlobalTensor<KV_ORIGIN_T> keyMergeGm_;

    // mm2
    GlobalTensor<KV_ORIGIN_T> vec1ResGm;
    GlobalTensor<KV_T> valueGm;
    GlobalTensor<MM_OUT_T> mm2ResGm;
    GlobalTensor<OUT_T> attentionOutGm;
    GlobalTensor<KV_ORIGIN_T> valueMergeGm_;

    // block_table
    GlobalTensor<int32_t> blockTableGm;
    GlobalTensor<int32_t> topKGm;

    TBuf<TPosition::A1> tmpBufL1Q;
    LocalTensor<KV_ORIGIN_T> qL1Buffers;
    TBuf<TPosition::A1> tmpBufL1KP;
    LocalTensor<KV_ORIGIN_T> kpL1Buffers;
    TBuf<TPosition::A1> tmpBufL1V;
    LocalTensor<KV_ORIGIN_T> vL1Buffers;

    TBuf<TPosition::A2> tmpBufL0A;
    LocalTensor<KV_ORIGIN_T> aL0TensorPingPong;
    // L0B
    TBuf<TPosition::B2> tmpBufL0B;
    LocalTensor<KV_ORIGIN_T> bL0TensorPingPong;
    // L0C
    TBuf<TPosition::CO1> tmpBufL0C;
    LocalTensor<MM_OUT_T> cL0TensorPingPong;

    __aicore__ inline void CopyGmToL1(LocalTensor<KV_ORIGIN_T> &l1Tensor, GlobalTensor<KV_ORIGIN_T> &gmSrcTensor,
                                      uint32_t srcN, uint32_t srcD, uint32_t srcDstride);
    __aicore__ inline void CopyInMm1AToL1(LocalTensor<KV_ORIGIN_T> &l1Tensor, const RunInfo &info,
                                          uint32_t mSizeAct, uint32_t headSize);
    __aicore__ inline void CopyInMm2AToL1(LocalTensor<KV_ORIGIN_T> &aL1Tensor, const RunInfo &info, uint32_t mSeqIdx,
                                          uint32_t subMSizeAct, uint32_t nSize, uint32_t nOffset);
    __aicore__ inline void LoadDataMm1A(LocalTensor<KV_ORIGIN_T> &aL0Tensor, LocalTensor<KV_ORIGIN_T> &aL1Tensor,
                                        uint32_t idx, uint32_t kSplitSize, uint32_t mSize, uint32_t kSize);
    __aicore__ inline void LoadDataMm1B(LocalTensor<KV_ORIGIN_T> &bL0Tensor, LocalTensor<KV_ORIGIN_T> &bL1Tensor,
                                        uint32_t idx, uint32_t kSplitSize, uint32_t kSize, uint32_t nSize);
    __aicore__ inline void CopyInMmBToL1(LocalTensor<KV_ORIGIN_T> &bl1Tensor, GlobalTensor<KV_ORIGIN_T> &gmSrcTensor,
                                          uint32_t subNSizeAct, uint32_t nOffset, const RunInfo &info);
};

template <typename SFAAT> __aicore__ inline void SFAAMatmulServiceGqa<SFAAT>::InitParams(const ConstInfo &constInfo)
{
    this->constInfo = constInfo;
}

template <typename SFAAT>
__aicore__ inline void
SFAAMatmulServiceGqa<SFAAT>::InitMm1GlobalTensor(GlobalTensor<Q_T> queryGm, GlobalTensor<KV_T> keyGm,
                                              GlobalTensor<MM_OUT_T> mm1ResGm)
{
    // mm1
    this->queryGm = queryGm;
    this->keyGm = keyGm;
    this->mm1ResGm = mm1ResGm;
}

template <typename SFAAT>
__aicore__ inline void
SFAAMatmulServiceGqa<SFAAT>::InitMm2GlobalTensor(GlobalTensor<KV_ORIGIN_T> vec1ResGm, GlobalTensor<KV_T> valueGm,
                                              GlobalTensor<MM_OUT_T> mm2ResGm, GlobalTensor<OUT_T> attentionOutGm)
{
    // mm2
    this->vec1ResGm = vec1ResGm;
    this->valueGm = valueGm;
    this->mm2ResGm = mm2ResGm;
    this->attentionOutGm = attentionOutGm;
}

template <typename SFAAT>
__aicore__ inline void
SFAAMatmulServiceGqa<SFAAT>::InitPageAttentionInfo(const GlobalTensor<KV_ORIGIN_T>& keyMergeGm, const GlobalTensor<KV_ORIGIN_T>& valueMergeGm,
                                                GlobalTensor<int32_t> blockTableGm, GlobalTensor<int32_t> topKGm,
                                                uint32_t blockSize, uint32_t maxBlockNumPerBatch)
{
    this->blockTableGm = blockTableGm;
    this->topKGm = topKGm;
    this->kvCacheBlockSize = blockSize;
    this->maxBlockNumPerBatch = maxBlockNumPerBatch;
    this->keyMergeGm_ = keyMergeGm;
    this->valueMergeGm_ = valueMergeGm;
}

template <typename SFAAT> __aicore__ inline void SFAAMatmulServiceGqa<SFAAT>::InitBuffers(TPipe *pipe)
{
    pipe->InitBuffer(tmpBufL1Q, L1Q_BLOCK_SIZE); // (32K) * 1
    qL1Buffers = tmpBufL1Q.Get<Q_T>();
    pipe->InitBuffer(tmpBufL1KP, L1_BLOCK_SIZE * L1KP_BUFFER_NUM); // 64K * 3
    kpL1Buffers = tmpBufL1KP.Get<KV_ORIGIN_T>();
    pipe->InitBuffer(tmpBufL1V, L1_BLOCK_SIZE * L1V_BUFFER_NUM); // 64K * 3
    vL1Buffers = tmpBufL1V.Get<KV_ORIGIN_T>();

    // L0A
    pipe->InitBuffer(tmpBufL0A, L0A_PP_SIZE * 2); // 64K
    aL0TensorPingPong = tmpBufL0A.Get<KV_ORIGIN_T>();
    // L0B
    pipe->InitBuffer(tmpBufL0B, L0B_PP_SIZE * 2); // 64K
    bL0TensorPingPong = tmpBufL0B.Get<KV_ORIGIN_T>();
    // L0C
    pipe->InitBuffer(tmpBufL0C, L0C_PP_SIZE * 2); // 128K
    cL0TensorPingPong = tmpBufL0C.Get<MM_OUT_T>();
}

template <typename SFAAT> __aicore__ inline void SFAAMatmulServiceGqa<SFAAT>::UpdateKey(GlobalTensor<KV_T> keyGm)
{
    this->keyGm = keyGm;
}

template <typename SFAAT> __aicore__ inline void SFAAMatmulServiceGqa<SFAAT>::UpdateValue(GlobalTensor<KV_T> valueGm)
{
    this->valueGm = valueGm;
}

template <typename SFAAT> __aicore__ inline void SFAAMatmulServiceGqa<SFAAT>::AllocEventID()
{
    SetFlag<HardEvent::MTE1_MTE2>(L1Q_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(L1KP_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(L1KP_EVENT1);
    SetFlag<HardEvent::MTE1_MTE2>(L1KP_EVENT2);
    SetFlag<HardEvent::MTE1_MTE2>(L1V_EVENT0);
    SetFlag<HardEvent::MTE1_MTE2>(L1V_EVENT1);
    SetFlag<HardEvent::MTE1_MTE2>(L1V_EVENT2);
    SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0);
    SetFlag<HardEvent::M_MTE1>(L0AB_EVENT1);
    SetFlag<HardEvent::FIX_M>(L0C_EVENT0);
    SetFlag<HardEvent::FIX_M>(L0C_EVENT1);
}

template <typename SFAAT> __aicore__ inline void SFAAMatmulServiceGqa<SFAAT>::FreeEventID()
{
    WaitFlag<HardEvent::MTE1_MTE2>(L1Q_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(L1KP_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(L1KP_EVENT1);
    WaitFlag<HardEvent::MTE1_MTE2>(L1KP_EVENT2);
    WaitFlag<HardEvent::MTE1_MTE2>(L1V_EVENT0);
    WaitFlag<HardEvent::MTE1_MTE2>(L1V_EVENT1);
    WaitFlag<HardEvent::MTE1_MTE2>(L1V_EVENT2);
    WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0);
    WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT1);
    WaitFlag<HardEvent::FIX_M>(L0C_EVENT0);
    WaitFlag<HardEvent::FIX_M>(L0C_EVENT1);
}

template <typename SFAAT>
__aicore__ inline void SFAAMatmulServiceGqa<SFAAT>::CopyGmToL1(LocalTensor<KV_ORIGIN_T> &l1Tensor,
                                                            GlobalTensor<KV_ORIGIN_T> &gmSrcTensor, uint32_t srcN,
                                                            uint32_t srcD, uint32_t srcDstride)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = srcN; // 行数
    nd2nzPara.dValue = srcD;
    nd2nzPara.srcDValue = srcDstride;
    nd2nzPara.dstNzC0Stride = SFAAAlign(srcN, 16U); // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(l1Tensor, gmSrcTensor, nd2nzPara);
}

template <typename SFAAT>
__aicore__ inline void SFAAMatmulServiceGqa<SFAAT>::CopyInMm1AToL1(LocalTensor<KV_ORIGIN_T> &l1Tensor, const RunInfo &info,
                                                                uint32_t mSizeAct, uint32_t headSize)
{
    auto srcGm = queryGm[info.tensorAOffset];
    // 不切G, mSizeAct必然是constInfo.gSize的整数倍
    uint32_t s1Size = mSizeAct / constInfo.gSize;
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = s1Size;
    nd2nzPara.nValue = constInfo.gSize; // 行数
    nd2nzPara.dValue = headSize;
    nd2nzPara.srcDValue = headSize;
    nd2nzPara.dstNzC0Stride = (mSizeAct + 15) / 16 * 16; // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = constInfo.qHeadNum * constInfo.headDim; // 这里小于65536
    nd2nzPara.dstNzMatrixStride = constInfo.gSize * BLOCK_ELEMENT_NUM;
    DataCopy(l1Tensor, srcGm, nd2nzPara);
}

template <typename SFAAT>
__aicore__ inline void SFAAMatmulServiceGqa<SFAAT>::LoadDataMm1A(LocalTensor<KV_ORIGIN_T> &aL0Tensor,
                                                              LocalTensor<KV_ORIGIN_T> &aL1Tensor, uint32_t idx,
                                                              uint32_t kSplitSize, uint32_t mSize, uint32_t kSize)
{
    LocalTensor<KV_ORIGIN_T> srcTensor = aL1Tensor[mSize * kSplitSize * idx];
    LoadData3DParamsV2<KV_ORIGIN_T> loadData3DParams;
    // SetFmatrixParams
    loadData3DParams.l1H = mSize / 16; // Hin=M1=8
    loadData3DParams.l1W = 16;         // Win=M0
    loadData3DParams.padList[0] = 0;
    loadData3DParams.padList[1] = 0;
    loadData3DParams.padList[2] = 0;
    loadData3DParams.padList[3] = 255; // 尾部数据不影响滑窗的结果

    // SetLoadToA0Params
    loadData3DParams.mExtension = mSize; // M
    loadData3DParams.kExtension = kSize; // K
    loadData3DParams.mStartPt = 0;
    loadData3DParams.kStartPt = 0;
    loadData3DParams.strideW = 1;
    loadData3DParams.strideH = 1;
    loadData3DParams.filterW = 1;
    loadData3DParams.filterSizeW = (1 >> 8) & 255;
    loadData3DParams.filterH = 1;
    loadData3DParams.filterSizeH = (1 >> 8) & 255;
    loadData3DParams.dilationFilterW = 1;
    loadData3DParams.dilationFilterH = 1;
    loadData3DParams.enTranspose = 0;
    loadData3DParams.fMatrixCtrl = 0;
    loadData3DParams.channelSize = kSize; // Cin=K
    LoadData<KV_ORIGIN_T, LOAD3DV2_CONFIG>(aL0Tensor, srcTensor, loadData3DParams);
}

template <typename SFAAT>
__aicore__ inline void SFAAMatmulServiceGqa<SFAAT>::LoadDataMm1B(LocalTensor<KV_ORIGIN_T> &l0Tensor,
                                                              LocalTensor<KV_ORIGIN_T> &l1Tensor, uint32_t idx,
                                                              uint32_t kSplitSize, uint32_t kSize, uint32_t nSize)
{
    // N 方向全载
    LocalTensor<KV_ORIGIN_T> srcTensor = l1Tensor[nSize * kSplitSize * idx];

    LoadData2DParams loadData2DParams;
    loadData2DParams.startIndex = 0;
    loadData2DParams.repeatTimes = (nSize + 15) / 16 * kSize / (32 / sizeof(KV_ORIGIN_T));
    loadData2DParams.srcStride = 1;
    loadData2DParams.dstGap = 0;
    loadData2DParams.ifTranspose = false;
    LoadData(l0Tensor, srcTensor, loadData2DParams);
}

template <typename SFAAT>
__aicore__ inline void SFAAMatmulServiceGqa<SFAAT>::CopyInMm2AToL1(LocalTensor<KV_ORIGIN_T> &aL1Tensor, const RunInfo &info,
                                                                uint32_t mSeqIdx, uint32_t subMSizeAct,
                                                                uint32_t nSize, uint32_t nOffset)
{
    auto srcGm = vec1ResGm[(info.loop % constInfo.preLoadNum) * constInfo.mmResUbSize +
                           mSeqIdx * info.actualSingleProcessSInnerSizeAlign + nOffset];
    CopyGmToL1(aL1Tensor, srcGm, subMSizeAct, nSize, info.actualSingleProcessSInnerSizeAlign);
}

template <typename SFAAT>
__aicore__ inline void SFAAMatmulServiceGqa<SFAAT>::CopyInMmBToL1(LocalTensor<KV_ORIGIN_T> &bl1Tensor,
                                                                   GlobalTensor<KV_ORIGIN_T> &gmSrcTensor,
                                                                   uint32_t subNSizeAct, uint32_t nOffset,
                                                                   const RunInfo &info)
{
    auto srcGm = gmSrcTensor[(info.loop % 4) * constInfo.s2BaseSize * constInfo.headDim + nOffset];

    CopyGmToL1(bl1Tensor, srcGm, subNSizeAct, constInfo.headDim, constInfo.headDim);
}

template <typename SFAAT>
__aicore__ inline void SFAAMatmulServiceGqa<SFAAT>::ComputeMm1(const RunInfo &info, const MSplitInfo mSplitInfo)
{
    LocalTensor<KV_ORIGIN_T> aL1Tensor = qL1Buffers;
    uint32_t mSize = mSplitInfo.nBufferDealM;
    uint32_t mSizeAlign = SFAAAlign(mSize, 16U);
    uint32_t kSize = constInfo.headDim;
    if (info.isFirstSInnerLoop) {
        WaitFlag<HardEvent::MTE1_MTE2>(L1Q_EVENT0);
        CopyInMm1AToL1(aL1Tensor, info, mSize, kSize);
        SetFlag<HardEvent::MTE2_MTE1>(L1Q_EVENT0);
        WaitFlag<HardEvent::MTE2_MTE1>(L1Q_EVENT0);
    }
    uint32_t nL1Loops = SfaaCeilDiv(info.actualSingleProcessSInnerSize, S2_SPLIT_SIZE);
    uint32_t nTailCopyRowCount = info.actualSingleProcessSInnerSize - (nL1Loops - 1) * S2_SPLIT_SIZE;
    for (uint32_t nL1 = 0, nL1Size = S2_SPLIT_SIZE, nL1SizeAlign = S2_SPLIT_SIZE; nL1 < nL1Loops; nL1++) {
        if (nL1 + 1 == nL1Loops) {
            nL1Size = nTailCopyRowCount;
            nL1SizeAlign = SFAAAlign(nL1Size, 16U);
        }
        kpL1BufIter++;
        LocalTensor<KV_ORIGIN_T> bL1Tensor = kpL1Buffers[(kpL1BufIter % L1KP_BUFFER_NUM) * L1_BLOCK_OFFSET]; // 3 buffer
        WaitFlag<HardEvent::MTE1_MTE2>(L1KP_EVENT0 + (kpL1BufIter % L1KP_BUFFER_NUM));

        uint32_t nL0Loops = SfaaCeilDiv(nL1Size, N_BASE_SIZE);
        uint32_t nTail = nL1Size - (nL0Loops - 1) * N_BASE_SIZE;
        for (uint32_t nL0 = 0, nL0Size = N_BASE_SIZE, nL0SizeAlign = N_BASE_SIZE; nL0 < nL0Loops; nL0++) {
            if (nL0 + 1 == nL0Loops) {
                nL0Size = nTail;
                nL0SizeAlign = SFAAAlign(nL0Size, 16U);
            }
            LocalTensor<KV_ORIGIN_T> L1bTensor = bL1Tensor[nL0 * N_BASE_SIZE * kSize];
            CopyInMmBToL1(L1bTensor, keyMergeGm_, nL0Size,
                (nL1 * S2_SPLIT_SIZE + nL0 * N_BASE_SIZE) * constInfo.headDim, info);
        }

        SetFlag<HardEvent::MTE2_MTE1>(L1KP_EVENT0 + (kpL1BufIter % L1KP_BUFFER_NUM));
        WaitFlag<HardEvent::MTE2_MTE1>(L1KP_EVENT0 + (kpL1BufIter % L1KP_BUFFER_NUM));

        for (uint32_t nL0 = 0, nL0Size = N_BASE_SIZE, nL0SizeAlign = N_BASE_SIZE; nL0 < nL0Loops; nL0++) {
            if (nL0 + 1 == nL0Loops) {
                nL0Size = nTail;
                nL0SizeAlign = SFAAAlign(nL0Size, 16U);
            }

            uint32_t kL0Loops = SfaaCeilDiv(kSize, K_BASE_SIZE);
            uint32_t kTail = kSize - (kL0Loops - 1) * K_BASE_SIZE;
            LocalTensor<MM_OUT_T> cL0Tensor = cL0TensorPingPong[(cL0BufIter % 2) * L0C_BLOCK_OFFSET];
            WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + (cL0BufIter % 2));
            LocalTensor<KV_ORIGIN_T> bTensor = bL1Tensor[nL0 * N_BASE_SIZE * kSize];
            for (uint32_t kL0 = 0, kL0Size = K_BASE_SIZE; kL0 < kL0Loops; kL0++) {
                if (kL0 + 1 == nL0Loops) {
                    kL0Size = kTail;
                }
                WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0 +  abL0BufIter % 2);
                LocalTensor<Q_T> aL0Tensor = aL0TensorPingPong[(abL0BufIter % 2) * L0AB_BLOCK_OFFSET];
                LoadDataMm1A(aL0Tensor, aL1Tensor, kL0, kL0Size, mSizeAlign, kL0Size);
                LocalTensor<KV_ORIGIN_T> bL0Tensor = bL0TensorPingPong[(abL0BufIter % 2) * L0AB_BLOCK_OFFSET];
                LoadDataMm1B(bL0Tensor, bTensor, kL0, kL0Size, kL0Size, nL0SizeAlign);
                SetFlag<HardEvent::MTE1_M>(L0AB_EVENT0 +  abL0BufIter % 2);
                WaitFlag<HardEvent::MTE1_M>(L0AB_EVENT0 +  abL0BufIter % 2);

                // m == 1的时候需要特殊处理
                MmadParams mmadParams;
                mmadParams.m = mSizeAlign;
                mmadParams.n = nL0SizeAlign;
                mmadParams.k = kL0Size;
                mmadParams.cmatrixInitVal = (kL0 == 0);
                mmadParams.cmatrixSource = false;
                Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);

                if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
                    PipeBarrier<PIPE_M>();
                }
                SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0 +  abL0BufIter % 2);
                abL0BufIter++;
            }
            FixpipeParamsV220 fixParams;
            fixParams.nSize = nL0SizeAlign;
            fixParams.mSize = mSize;
            fixParams.srcStride = mSizeAlign;
            // 改成nSizeAlign
            fixParams.dstStride = info.actualSingleProcessSInnerSizeAlign; // mm1ResGm两行之间的间隔
            fixParams.ndNum = 1; // 输出ND

            SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + (cL0BufIter % 2));
            WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + (cL0BufIter % 2));
            Fixpipe(mm1ResGm[(info.loop % (constInfo.preLoadNum)) * constInfo.mmResUbSize + nL1 * S2_SPLIT_SIZE +
                                nL0 * N_BASE_SIZE + mSplitInfo.nBufferStartM * info.actualSingleProcessSInnerSizeAlign],
                    cL0Tensor, fixParams);
            SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + (cL0BufIter % 2));
            cL0BufIter++;
        }
        SetFlag<HardEvent::MTE1_MTE2>(L1KP_EVENT0 + (kpL1BufIter % L1KP_BUFFER_NUM));
    }
    // will change batch
    if ((info.s2Idx + 1) == info.curSInnerLoopTimes) {
        SetFlag<HardEvent::MTE1_MTE2>(L1Q_EVENT0);
    }
}

template <typename SFAAT>
__aicore__ inline void SFAAMatmulServiceGqa<SFAAT>::ComputeMm2(const RunInfo &info, const MSplitInfo mSplitInfo)
{
    // [m, S2Inner] * [S2Inner, D]
    uint32_t mSize = mSplitInfo.nBufferDealM;
    uint32_t mSizeAlign = SFAAAlign(mSize, 16U);

    uint32_t nSize = constInfo.headDim;
    uint32_t nL1Loops = SfaaCeilDiv(nSize, N_BASE_SIZE);

    uint32_t kSize = info.actualSingleProcessSInnerSize;
    uint32_t kL1Loops = (kSize + S2_SPLIT_SIZE - 1) / S2_SPLIT_SIZE;
    for (uint32_t nL1 = 0, nL1Size = N_BASE_SIZE, nL1SizeAlign = N_BASE_SIZE; nL1 < nL1Loops; nL1++) { // n切L1
        if (nL1 == (nL1Loops - 1)) {
            // 尾块
            nL1Size = nSize - (nL1Loops - 1) * N_BASE_SIZE;
            nL1SizeAlign = SFAAAlign(nL1Size, 16U);
        }

        // k l1写成一个循环, 和mm1保持一致
        uint32_t kL1Size = S2_SPLIT_SIZE;
        uint32_t kL1SizeAlign = SFAAAlign(kL1Size, 16U);
        
        LocalTensor<MM_OUT_T> cL0Tensor = cL0TensorPingPong[(cL0BufIter % 2) * L0C_BLOCK_OFFSET]; // 需要保证cL0BufIter和m步调一致
        WaitFlag<HardEvent::FIX_M>(L0C_EVENT0 + (cL0BufIter % 2));
        for (uint32_t kL1 = 0; kL1 < kL1Loops; kL1++) { // k切L1, 这里套了一层l0来操作
            if (kL1 == (kL1Loops - 1)) {
                // 尾块
                kL1Size = kSize - (kL1Loops - 1) * S2_SPLIT_SIZE;
                kL1SizeAlign = SFAAAlign(kL1Size, 16U);
            }
            kpL1BufIter++;
            LocalTensor<KV_ORIGIN_T> aL1Tensor = kpL1Buffers[(kpL1BufIter % L1KP_BUFFER_NUM) * L1_BLOCK_OFFSET];
            WaitFlag<HardEvent::MTE1_MTE2>(L1KP_EVENT0 + (kpL1BufIter % L1KP_BUFFER_NUM));
            CopyInMm2AToL1(aL1Tensor, info, mSplitInfo.nBufferStartM, mSize, kL1Size, S2_SPLIT_SIZE * kL1);
            
            vL1BufIter++;
            LocalTensor<KV_ORIGIN_T> bL1Tensor = vL1Buffers[(vL1BufIter % L1V_BUFFER_NUM) * L1_BLOCK_OFFSET];
            WaitFlag<HardEvent::MTE1_MTE2>(L1V_EVENT0 + (vL1BufIter % L1V_BUFFER_NUM));
            uint32_t kL0Loops = SfaaCeilDiv(kL1Size , K_BASE_SIZE);
            uint32_t kTail = kL1Size - (kL0Loops - 1) * K_BASE_SIZE;
            for (uint32_t kL0 = 0, kL0Size = K_BASE_SIZE,  kL0SizeAlign = K_BASE_SIZE; kL0 < kL0Loops; kL0++) {
                if (kL0 == kL0Loops - 1) {
                    kL0Size = kTail;
                    kL0SizeAlign = SFAAAlign(kL0Size, 16U);
                }
                LocalTensor<KV_ORIGIN_T> L1bTensor = bL1Tensor[kL0 * K_BASE_SIZE * N_BASE_SIZE];
                CopyInMmBToL1(L1bTensor, valueMergeGm_, kL0Size,
                    (kL1 * S2_SPLIT_SIZE + kL0 * K_BASE_SIZE) * constInfo.headDim +
                     nL1 * N_BASE_SIZE * constInfo.s2BaseSize, info);
            }
            SetFlag<HardEvent::MTE2_MTE1>(L1KP_EVENT0);
            WaitFlag<HardEvent::MTE2_MTE1>(L1KP_EVENT0);        

            for (uint32_t kL0 = 0, kL0Size = K_BASE_SIZE,  kL0SizeAlign = K_BASE_SIZE; kL0 < kL0Loops; kL0++) {
                if (kL0 + 1 == kL0Loops) {
                    kL0Size = kTail;
                    kL0SizeAlign = SFAAAlign(kL0Size, 16U);
                }
                WaitFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + abL0BufIter % 2);
                LocalTensor<KV_ORIGIN_T> bL0Tensor = bL0TensorPingPong[(abL0BufIter % 2) * L0AB_BLOCK_OFFSET];
                LoadData3DParamsV2<KV_ORIGIN_T> loadData3DParamsForB;
                loadData3DParamsForB.l1H = kL0SizeAlign / 16; // 源操作数height
                loadData3DParamsForB.l1W = 16;                // 源操作数weight=16，目的height=l1H*L1W
                loadData3DParamsForB.padList[0] = 0;
                loadData3DParamsForB.padList[1] = 0;
                loadData3DParamsForB.padList[2] = 0;
                loadData3DParamsForB.padList[3] = 255;        // 尾部数据不影响滑窗的结果

                loadData3DParamsForB.mExtension = kL0SizeAlign; // 在目的操作数height维度的传输长度
                loadData3DParamsForB.kExtension = nL1SizeAlign; // 在目的操作数width维度的传输长度
                loadData3DParamsForB.mStartPt = 0;              // 卷积核在目的操作数width维度的起点
                loadData3DParamsForB.kStartPt = 0;              // 卷积核在目的操作数height维度的起点
                loadData3DParamsForB.strideW = 1;
                loadData3DParamsForB.strideH = 1;
                loadData3DParamsForB.filterW = 1;
                loadData3DParamsForB.filterSizeW = false; // 是否在filterW的基础上将卷积核width增加256个元素
                loadData3DParamsForB.filterH = 1;
                loadData3DParamsForB.filterSizeH = false; // 是否在filterH的基础上将卷积核height增加256个元素
                loadData3DParamsForB.dilationFilterW = 1; // 卷积核width膨胀系数
                loadData3DParamsForB.dilationFilterH = 1; // 卷积核height膨胀系数
                loadData3DParamsForB.enTranspose = 1;     // 是否启用转置功能
                // 使用FMATRIX_LEFT还是使用FMATRIX_RIGHT，=0使用FMATRIX_LEFT，=1使用FMATRIX_RIGHT 1
                loadData3DParamsForB.fMatrixCtrl = 0;
                // 源操作数的通道数。膨胀系数为1时，目的weight为filterW*filterH*channelSize
                loadData3DParamsForB.channelSize = nL1SizeAlign;
                LoadData<KV_ORIGIN_T, LOAD3DV2_CONFIG>(bL0Tensor, bL1Tensor[kL0 * K_BASE_SIZE * N_BASE_SIZE],
                    loadData3DParamsForB);
                LocalTensor<KV_ORIGIN_T> aL0Tensor = aL0TensorPingPong[(abL0BufIter % 2) * L0AB_BLOCK_OFFSET];
                LoadData3DParamsV2<KV_ORIGIN_T> loadData3DParamsForA;
                loadData3DParamsForA.l1H = mSizeAlign / 16;    // 源操作数height
                loadData3DParamsForA.l1W = 16;                   // 源操作数weight
                loadData3DParamsForA.padList[0] = 0;
                loadData3DParamsForA.padList[1] = 0;
                loadData3DParamsForA.padList[2] = 0;
                loadData3DParamsForA.padList[3] = 255;           // 尾部数据不影响滑窗的结果

                loadData3DParamsForA.mExtension = mSizeAlign;  // 在目的操作数height维度的传输长度
                loadData3DParamsForA.kExtension = kL0SizeAlign;  // 在目的操作数width维度的传输长度
                loadData3DParamsForA.mStartPt = 0;               // 卷积核在目的操作数width维度的起点
                loadData3DParamsForA.kStartPt = 0;               // 卷积核在目的操作数height维度的起点
                loadData3DParamsForA.strideW = 1;                // 卷积核在源操作数width维度滑动的步长
                loadData3DParamsForA.strideH = 1;                // 卷积核在源操作数height维度滑动的步长
                loadData3DParamsForA.filterW = 1;                // 卷积核width
                loadData3DParamsForA.filterSizeW = false;        // 是否在filterW的基础上将卷积核width增加256个元素
                loadData3DParamsForA.filterH = 1;                // 卷积核height
                loadData3DParamsForA.filterSizeH = false;        // 是否在filterH的基础上将卷积核height增加256个元素
                loadData3DParamsForA.dilationFilterW = 1;        // 卷积核width膨胀系数
                loadData3DParamsForA.dilationFilterH = 1;        // 卷积核height膨胀系数
                loadData3DParamsForA.enTranspose = 0;            // 是否启用转置功能，对整个目标矩阵进行转置
                loadData3DParamsForA.fMatrixCtrl = 0;
                // 源操作数的通道数。膨胀系数为1时，目的weight为filterW*filterH*channelSize
                loadData3DParamsForA.channelSize = kL0SizeAlign;
                LoadData<KV_ORIGIN_T, LOAD3DV2_CONFIG>(aL0Tensor, aL1Tensor[kL0 * K_BASE_SIZE * mSizeAlign],
                                                loadData3DParamsForA);
                SetFlag<HardEvent::MTE1_M>(L0AB_EVENT0);
                WaitFlag<HardEvent::MTE1_M>(L0AB_EVENT0);

                MmadParams mmadParams;
                mmadParams.m = mSizeAlign;
                mmadParams.n = nL1SizeAlign;
                mmadParams.k = kL0SizeAlign;
                mmadParams.cmatrixInitVal = (kL0 == 0 && kL1 == 0);
                mmadParams.cmatrixSource = false;

                Mmad(cL0Tensor, aL0Tensor, bL0Tensor, mmadParams);
                if ((mmadParams.m / 16) * (mmadParams.n / 16) < 10) {
                    PipeBarrier<PIPE_M>();
                }
                SetFlag<HardEvent::M_MTE1>(L0AB_EVENT0 + abL0BufIter % 2);
                abL0BufIter++;
            }

            SetFlag<HardEvent::MTE1_MTE2>(L1KP_EVENT0 + (kpL1BufIter % L1KP_BUFFER_NUM)); // 反向同步, 表示L1中的A已经被mte1消费完
            SetFlag<HardEvent::MTE1_MTE2>(L1V_EVENT0 + (vL1BufIter % L1V_BUFFER_NUM)); // 反向同步, 表示L1中的A已经被mte1消费完
        }
        SetFlag<HardEvent::M_FIX>(L0C_EVENT0 + (cL0BufIter % 2));
        WaitFlag<HardEvent::M_FIX>(L0C_EVENT0 + (cL0BufIter % 2));
        FixpipeParamsV220 fixParams;
        fixParams.nSize = nL1SizeAlign;
        fixParams.mSize = mSize;
        fixParams.srcStride = mSizeAlign;
        fixParams.dstStride = nSize; // mm2ResGm两行之间的间隔
        fixParams.ndNum = 1;         // 输出ND
        uint64_t mm2Offset = (mSplitInfo.nBufferStartM) * nSize + nL1 * N_BASE_SIZE;
        Fixpipe(mm2ResGm[(info.loop % (constInfo.preLoadNum)) *
                constInfo.bmm2ResUbSize + mm2Offset], cL0Tensor, fixParams);
        SetFlag<HardEvent::FIX_M>(L0C_EVENT0 + (cL0BufIter % 2));
        cL0BufIter++;
    }
}

#endif // SPARSE_FLASH_ATTENTION_ANTIQUANT_SERVICE_CUBE_GQA_H
