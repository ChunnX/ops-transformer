/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or
 * modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 *
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS
 * SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT
 * NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file stem_indexer_service_cube.h
 * \brief matmul 使用多 buffer 流水，提升流水效率
 */
#ifndef STEM_INDEXER_SERVICE_CUBE_H
#define STEM_INDEXER_SERVICE_CUBE_H

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "stem_indexer_common.h"

namespace SIKernel {
using namespace SICommon;
template <typename SIT>
class SIMatmul {
public:
    using Q_T = typename SIT::queryType;
    using K_T = typename SIT::keyType;
    using QK_T = float32_t;

    __aicore__ inline SIMatmul(){};
    __aicore__ inline void InitBuffers(TPipe *pipe);
    __aicore__ inline void InitMm1GlobalTensor(const GlobalTensor<Q_T> &queryGm, const GlobalTensor<K_T> &keyGm);
    __aicore__ inline void InitParams(const ConstInfo &constInfo);
    __aicore__ inline void AllocEventID();
    __aicore__ inline void FreeEventID();
    __aicore__ inline void ComputeMm1(const SICommon::RunInfo &runInfo);

    static constexpr IsResetLoad3dConfig LOAD3DV2_CONFIG = {true, true}; // isSetFMatrix isSetPadding;
    static constexpr uint64_t KEY_BUF_NUM = 4;
    static constexpr uint64_t QUERY_BUF_NUM = 4;
    static constexpr uint64_t L0A_BUF_NUM = 2;
    static constexpr uint64_t L0B_BUF_NUM = 2;
    static constexpr uint64_t L0C_BUF_NUM = 2;

    event_t keyMte1Mte2Event_[KEY_BUF_NUM];
    event_t queryMte1Mte2Event_[QUERY_BUF_NUM];
    event_t mMte1Event_[L0A_BUF_NUM];
    event_t mte2Mte1Event_;
    event_t mte1MEvent_;
    event_t fixMEvent_[L0C_BUF_NUM];
    event_t mFixEvent_[L0C_BUF_NUM];

    static constexpr uint64_t M_BASIC_BLOCK = 64;
    static constexpr uint64_t S2_BASIC_BLOCK = 512;
    static constexpr uint64_t RES_MM1_SLOT_BYTES = M_BASIC_BLOCK * S2_BASIC_BLOCK * sizeof(QK_T);
    static constexpr uint64_t RES_MM1_DOUBLE_BUFFER_BYTES = 2U * RES_MM1_SLOT_BYTES;

    static constexpr uint64_t M_BASIC_BLOCK_L1 = 64;
    static constexpr uint64_t D_BASIC_BLOCK_L1 = 512;
    static constexpr uint64_t S2_BASIC_BLOCK_L1 = 64;

    static constexpr uint64_t M_BASIC_BLOCK_L0 = 64;
    static constexpr uint64_t D_BASIC_BLOCK_L0 = 256;
    static constexpr uint64_t S2_BASIC_BLOCK_L0 = 64;

    static constexpr uint64_t BF16_BLOCK_CUBE = 16;
    static constexpr uint32_t PIPE_M_BARRIER_THRESHOLD = 10U;
    // ROW_MAJOR使能NZ2ND并输出ND格式；true表示目的地址位于UB。
    static constexpr FixpipeConfig SI_CFG_ROW_MAJOR_UB = {CO2Layout::ROW_MAJOR, true};

    static constexpr uint64_t QUERY_BUFFER_OFFSET = M_BASIC_BLOCK_L1 * D_BASIC_BLOCK_L1;
    static constexpr uint64_t KEY_BUFFER_OFFSET = S2_BASIC_BLOCK_L1 * D_BASIC_BLOCK_L1;
    static constexpr uint64_t L0A_BUFFER_OFFSET = M_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0;
    static constexpr uint64_t L0B_BUFFER_OFFSET = S2_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0;
    static constexpr uint64_t L0C_BUFFER_OFFSET = M_BASIC_BLOCK * S2_BASIC_BLOCK;
    static_assert(M_BASIC_BLOCK == M_BASIC_BLOCK_L1 && M_BASIC_BLOCK_L1 == M_BASIC_BLOCK_L0,
                  "M basic blocks must stay equal after removing single-iteration M loops");
    static_assert(S2_BASIC_BLOCK_L1 == S2_BASIC_BLOCK_L0,
                  "S2 L1/L0 basic blocks must stay equal after removing the single-iteration S2 loop");

protected:
    __aicore__ inline void Fixp(uint64_t s1gRealSize, uint64_t s2RealSize, uint32_t mm1BufferIdx,
                                uint32_t l0cSlot);
    __aicore__ inline void ComuteL0c(uint64_t l0cOffset, uint32_t l0Slot, MmadParams &mmadParams,
                                     bool needMBarrier);
    __aicore__ inline void LoadKeyToL0b(const LoadData2DParamsV2 &loadDataParams, uint32_t l0Slot,
                                        uint32_t keyL1Slot);
    __aicore__ inline void LoadQueryToL0a(const LoadData2DParamsV2 &loadDataParams, uint32_t l0Slot,
                                          uint32_t queryL1Slot);
    __aicore__ inline void CopyQuerySegmentNd2Nz(LocalTensor<Q_T> queryL1Base, uint64_t l1RowOffset,
                                                 uint64_t l1RowAlign, uint64_t gmOffset, uint64_t nValue);
    __aicore__ inline void QueryNd2Nz(uint64_t s1gL1RealSize, uint64_t s1gL1Offset, uint64_t kGmOffset,
                                      uint32_t queryL1Slot, const SICommon::RunInfo &runInfo);
    __aicore__ inline void KeyNd2Nz(uint64_t s2L1RealSize, uint64_t s2GmOffset, uint64_t kGmOffset,
                                    uint32_t keyL1Slot, const SICommon::RunInfo &runInfo);
    GlobalTensor<int32_t> blkTableGm_;
    GlobalTensor<K_T> keyGm_;
    GlobalTensor<Q_T> queryGm_;

    TBuf<TPosition::A1> bufQL1_;
    LocalTensor<Q_T> queryL1_;
    TBuf<TPosition::B1> bufKeyL1_;
    LocalTensor<K_T> keyL1_;

    TBuf<TPosition::A2> bufQL0_;
    LocalTensor<Q_T> queryL0_;
    TBuf<TPosition::B2> bufKeyL0_;
    LocalTensor<K_T> keyL0_;

    TBuf<TPosition::CO1> bufL0C_;
    LocalTensor<float> cL0_;

    TBuf<TPosition::VECCALC> bufUB_;
    LocalTensor<QK_T> mm1ResUB_;

    uint64_t keyL1BufIdx_ = 0;
    uint64_t l0BufIdx_ = 0;
    uint64_t l0cBufIdx_ = 0;

    ConstInfo constInfo_;

private:
    static constexpr bool PAGE_ATTENTION = SIT::pageAttention;
};

template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::InitParams(const ConstInfo &constInfo)
{
    constInfo_ = constInfo;
}

template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::InitBuffers(TPipe *pipe)
{
    // V121 1C:1V: 2 * 64 * 512 * 4B = 256 KiB.
    pipe->InitBuffer(bufUB_, RES_MM1_DOUBLE_BUFFER_BYTES);
    mm1ResUB_ = bufUB_.Get<QK_T>();
    pipe->InitBuffer(bufQL1_, QUERY_BUF_NUM * M_BASIC_BLOCK_L1 * D_BASIC_BLOCK_L1 * sizeof(Q_T));
    queryL1_ = bufQL1_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL1_, KEY_BUF_NUM * S2_BASIC_BLOCK_L1 * D_BASIC_BLOCK_L1 * sizeof(K_T));
    keyL1_ = bufKeyL1_.Get<K_T>();

    pipe->InitBuffer(bufQL0_, L0A_BUF_NUM * M_BASIC_BLOCK_L0 * D_BASIC_BLOCK_L0 * sizeof(Q_T));
    queryL0_ = bufQL0_.Get<Q_T>();
    pipe->InitBuffer(bufKeyL0_, L0B_BUF_NUM * D_BASIC_BLOCK_L0 * S2_BASIC_BLOCK_L0 * sizeof(K_T));
    keyL0_ = bufKeyL0_.Get<K_T>();

    pipe->InitBuffer(bufL0C_, L0C_BUF_NUM * M_BASIC_BLOCK * S2_BASIC_BLOCK * sizeof(float));
    cL0_ = bufL0C_.Get<float>();
}

template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::InitMm1GlobalTensor(const GlobalTensor<Q_T> &queryGm,
                                                          const GlobalTensor<K_T> &keyGm)
{
    queryGm_ = queryGm;
    keyGm_ = keyGm;
}

template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::ComputeMm1(const SICommon::RunInfo &runInfo)
{
    uint32_t mm1BufferIdx = runInfo.loop & 1U;
    CrossCoreWaitFlag<SICommon::SI_SYNC_MODE4, PIPE_FIX>(SICommon::CROSS_VC_EVENT + mm1BufferIdx);
    uint64_t s1gProcessSize = runInfo.actMBaseSize;
    uint64_t s2ProcessSize = runInfo.actualSingleProcessSInnerSize;
    uint64_t kProcessSize = constInfo_.headDim;
    uint32_t l0cSlot = l0cBufIdx_ & (L0C_BUF_NUM - 1U);
    uint64_t mAlignSize = CeilAlign(s1gProcessSize, static_cast<uint64_t>(BLOCK_CUBE));
    uint32_t mBlockNum = static_cast<uint32_t>(mAlignSize / BLOCK_CUBE);

    LoadData2DParamsV2 queryLoadParams;
    queryLoadParams.mStartPosition = 0U;
    queryLoadParams.kStartPosition = 0U;
    queryLoadParams.mStep = mBlockNum;
    queryLoadParams.kStep = D_BASIC_BLOCK_L0 / BF16_BLOCK_CUBE;
    queryLoadParams.srcStride = mBlockNum;
    queryLoadParams.dstStride = mBlockNum;
    queryLoadParams.ifTranspose = false;

    // s2轴循环
    WaitFlag<HardEvent::FIX_M>(fixMEvent_[l0cSlot]);
    for (uint64_t s2GmOffset = 0; s2GmOffset < s2ProcessSize; s2GmOffset += S2_BASIC_BLOCK_L1) {
        uint64_t s2L1RealSize =
            s2GmOffset + S2_BASIC_BLOCK_L1 > s2ProcessSize ? s2ProcessSize - s2GmOffset : S2_BASIC_BLOCK_L1;
        uint32_t nBlockNum = static_cast<uint32_t>(CeilDiv(s2L1RealSize, static_cast<uint64_t>(BLOCK_CUBE)));
        LoadData2DParamsV2 keyLoadParams;
        keyLoadParams.mStartPosition = 0U;
        keyLoadParams.kStartPosition = 0U;
        keyLoadParams.mStep = nBlockNum;
        keyLoadParams.kStep = D_BASIC_BLOCK_L0 / BF16_BLOCK_CUBE;
        keyLoadParams.srcStride = nBlockNum;
        keyLoadParams.dstStride = nBlockNum;
        keyLoadParams.ifTranspose = false;

        MmadParams mm1Params;
        mm1Params.m = mAlignSize;
        mm1Params.n = s2L1RealSize;
        mm1Params.k = D_BASIC_BLOCK_L0;
        mm1Params.cmatrixInitVal = false;
        mm1Params.cmatrixSource = false;
        bool needMBarrier =
            (mBlockNum * (static_cast<uint32_t>(s2L1RealSize) / BLOCK_CUBE)) < PIPE_M_BARRIER_THRESHOLD;
        uint64_t l0cOffset = l0cSlot * L0C_BUFFER_OFFSET + s2GmOffset * mAlignSize;

        uint32_t dL1Idx = 0U;
        for (uint64_t kGmOffset = 0; kGmOffset < kProcessSize;
             kGmOffset += D_BASIC_BLOCK_L1, ++dL1Idx) {
            uint32_t keyL1Slot = keyL1BufIdx_ & (KEY_BUF_NUM - 1U);
            uint32_t queryL1Slot = dL1Idx & (QUERY_BUF_NUM - 1U);
            WaitFlag<HardEvent::MTE1_MTE2>(keyMte1Mte2Event_[keyL1Slot]);
            KeyNd2Nz(s2L1RealSize, s2GmOffset, kGmOffset, keyL1Slot, runInfo);

            SetFlag<HardEvent::MTE2_MTE1>(mte2Mte1Event_);
            WaitFlag<HardEvent::MTE2_MTE1>(mte2Mte1Event_);
            if (runInfo.isFirstS2InnerLoop && s2GmOffset == 0U) {
                WaitFlag<HardEvent::MTE1_MTE2>(queryMte1Mte2Event_[queryL1Slot]);
                QueryNd2Nz(s1gProcessSize, 0U, kGmOffset, queryL1Slot, runInfo);
                SetFlag<HardEvent::MTE2_MTE1>(mte2Mte1Event_);
                WaitFlag<HardEvent::MTE2_MTE1>(mte2Mte1Event_);
            }

            for (uint64_t kL1Offset = 0; kL1Offset < D_BASIC_BLOCK_L1; kL1Offset += D_BASIC_BLOCK_L0) {
                uint32_t l0Slot = l0BufIdx_ & (L0A_BUF_NUM - 1U);
                uint32_t kStartPosition = static_cast<uint32_t>(kL1Offset / BLOCK_CUBE);
                queryLoadParams.kStartPosition = kStartPosition;
                keyLoadParams.kStartPosition = kStartPosition;

                WaitFlag<HardEvent::M_MTE1>(mMte1Event_[l0Slot]);
                LoadQueryToL0a(queryLoadParams, l0Slot, queryL1Slot);
                LoadKeyToL0b(keyLoadParams, l0Slot, keyL1Slot);

                SetFlag<HardEvent::MTE1_M>(mte1MEvent_);
                WaitFlag<HardEvent::MTE1_M>(mte1MEvent_);

                mm1Params.cmatrixInitVal = (kGmOffset + kL1Offset) == 0U;
                ComuteL0c(l0cOffset, l0Slot, mm1Params, needMBarrier);
                SetFlag<HardEvent::M_MTE1>(mMte1Event_[l0Slot]);
                l0BufIdx_++;
            }
            if (s2GmOffset + S2_BASIC_BLOCK_L1 >= s2ProcessSize && runInfo.isLastS2InnerLoop) {
                SetFlag<HardEvent::MTE1_MTE2>(queryMte1Mte2Event_[queryL1Slot]);
            }
            SetFlag<HardEvent::MTE1_MTE2>(keyMte1Mte2Event_[keyL1Slot]);
            keyL1BufIdx_++;
        }
    }
    Fixp(s1gProcessSize, s2ProcessSize, mm1BufferIdx, l0cSlot);
    SetFlag<HardEvent::FIX_M>(fixMEvent_[l0cSlot]);
    l0cBufIdx_++;
    CrossCoreSetFlag<SICommon::SI_SYNC_MODE4, PIPE_FIX>(SICommon::CROSS_CV_EVENT + mm1BufferIdx);
}

template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::KeyNd2Nz(uint64_t s2L1RealSize, uint64_t s2GmOffset, uint64_t kGmOffset,
                                               uint32_t keyL1Slot, const SICommon::RunInfo &runInfo)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = s2L1RealSize; // 行数
    nd2nzPara.dValue = D_BASIC_BLOCK_L1;
    nd2nzPara.srcDValue = constInfo_.headDim;
    nd2nzPara.dstNzC0Stride = CeilAlign(s2L1RealSize, (uint64_t)BLOCK_CUBE); // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    // 默认一块buf最多放两份
    DataCopy(keyL1_[keyL1Slot * KEY_BUFFER_OFFSET],
             keyGm_[runInfo.tensorKeyOffset + s2GmOffset * constInfo_.headDim + kGmOffset], nd2nzPara);
}

// batch, n2, g, s1, d
template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::CopyQuerySegmentNd2Nz(LocalTensor<Q_T> queryL1Base, uint64_t l1RowOffset,
                                                            uint64_t l1RowAlign, uint64_t gmOffset, uint64_t nValue)
{
    Nd2NzParams nd2nzPara;
    nd2nzPara.ndNum = 1;
    nd2nzPara.nValue = nValue; // 行数
    nd2nzPara.dValue = D_BASIC_BLOCK_L1;
    nd2nzPara.srcDValue = constInfo_.headDim;
    nd2nzPara.dstNzC0Stride = l1RowAlign; // 对齐到16 单位block
    nd2nzPara.dstNzNStride = 1;
    nd2nzPara.srcNdMatrixStride = 0;
    nd2nzPara.dstNzMatrixStride = 0;
    DataCopy(queryL1Base[l1RowOffset * BLOCK_CUBE], queryGm_[gmOffset], nd2nzPara);
}

template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::QueryNd2Nz(uint64_t s1gL1RealSize, uint64_t s1gGmOffset, uint64_t kGmOffset,
                                                 uint32_t queryL1Slot, const SICommon::RunInfo &runInfo)
{
    LocalTensor<Q_T> queryL1Base = queryL1_[queryL1Slot * QUERY_BUFFER_OFFSET];
    uint64_t l1RowAlign = CeilAlign(s1gL1RealSize, (uint64_t)BLOCK_CUBE);
    uint64_t logicalMStart = static_cast<uint64_t>(runInfo.gS1Idx) * constInfo_.mBaseSize + s1gGmOffset;
    uint64_t queryBaseOffset = runInfo.tensorQueryOffset;
    if (runInfo.actS1Size == constInfo_.qSeqSize) {
        CopyQuerySegmentNd2Nz(queryL1Base, 0, l1RowAlign,
                              queryBaseOffset + logicalMStart * constInfo_.headDim + kGmOffset, s1gL1RealSize);
        return;
    }

    uint64_t copiedRows = 0;
    uint64_t logicalMEnd = logicalMStart + s1gL1RealSize;
    while (logicalMStart < logicalMEnd) {
        uint64_t globalGIdx = logicalMStart / runInfo.actS1Size;
        uint64_t globalS1Idx = logicalMStart % runInfo.actS1Size;
        uint64_t copyRows = Min(logicalMEnd - logicalMStart, static_cast<uint64_t>(runInfo.actS1Size) - globalS1Idx);
        uint64_t gmOffset =
            queryBaseOffset + (globalGIdx * constInfo_.qSeqSize + globalS1Idx) * constInfo_.headDim + kGmOffset;
        CopyQuerySegmentNd2Nz(queryL1Base, copiedRows, l1RowAlign, gmOffset, copyRows);
        logicalMStart += copyRows;
        copiedRows += copyRows;
    }
}

template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::LoadQueryToL0a(const LoadData2DParamsV2 &loadDataParams, uint32_t l0Slot,
                                                     uint32_t queryL1Slot)
{
    LoadData(queryL0_[l0Slot * L0A_BUFFER_OFFSET], queryL1_[queryL1Slot * QUERY_BUFFER_OFFSET], loadDataParams);
}

template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::LoadKeyToL0b(const LoadData2DParamsV2 &loadDataParams, uint32_t l0Slot,
                                                   uint32_t keyL1Slot)
{
    LoadData(keyL0_[l0Slot * L0B_BUFFER_OFFSET], keyL1_[keyL1Slot * KEY_BUFFER_OFFSET], loadDataParams);
}

template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::ComuteL0c(uint64_t l0cOffset, uint32_t l0Slot, MmadParams &mmadParams,
                                                bool needMBarrier)
{
    Mmad(cL0_[l0cOffset], queryL0_[l0Slot * L0A_BUFFER_OFFSET], keyL0_[l0Slot * L0B_BUFFER_OFFSET], mmadParams);
    if (needMBarrier) {
        PipeBarrier<PIPE_M>();
    }
}

template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::Fixp(uint64_t s1gRealSize, uint64_t s2RealSize, uint32_t mm1BufferIdx,
                                           uint32_t l0cSlot)
{
    SetFlag<HardEvent::M_FIX>(mFixEvent_[l0cSlot]);
    WaitFlag<HardEvent::M_FIX>(mFixEvent_[l0cSlot]);

    FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixpipeParams;

    uint32_t mSize = (s1gRealSize + 1U) >> 1U << 1U;
    uint32_t nSize = (s2RealSize + 7U) >> 3U << 3U; // 32B对齐
    fixpipeParams.nSize = nSize;
    fixpipeParams.mSize = mSize;
    fixpipeParams.srcStride = ((fixpipeParams.mSize + BLOCK_CUBE - 1U) / BLOCK_CUBE) * BLOCK_CUBE;
    fixpipeParams.dstStride = constInfo_.s2BaseSize;
    fixpipeParams.dualDstCtl = 0;
    fixpipeParams.params.ndNum = 1;
    fixpipeParams.params.srcNdStride = 0;
    fixpipeParams.params.dstNdStride = 0;
    // 将matmul结果从L0C搬运到UB。
    Fixpipe<QK_T, float, SI_CFG_ROW_MAJOR_UB>(
        mm1ResUB_[mm1BufferIdx * (RES_MM1_SLOT_BYTES / sizeof(QK_T))],
        cL0_[l0cSlot * L0C_BUFFER_OFFSET], fixpipeParams);
}

template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::AllocEventID()
{
    SetMMLayoutTransform(true);
    for (uint32_t slot = 0U; slot < KEY_BUF_NUM; slot++) {
        keyMte1Mte2Event_[slot] =
            static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>());
        SetFlag<HardEvent::MTE1_MTE2>(keyMte1Mte2Event_[slot]);
    }
    for (uint32_t slot = 0U; slot < QUERY_BUF_NUM; slot++) {
        queryMte1Mte2Event_[slot] =
            static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_MTE2>());
        SetFlag<HardEvent::MTE1_MTE2>(queryMte1Mte2Event_[slot]);
    }
    for (uint32_t slot = 0U; slot < L0A_BUF_NUM; slot++) {
        mMte1Event_[slot] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_MTE1>());
        SetFlag<HardEvent::M_MTE1>(mMte1Event_[slot]);
    }
    for (uint32_t slot = 0U; slot < L0C_BUF_NUM; slot++) {
        fixMEvent_[slot] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::FIX_M>());
        SetFlag<HardEvent::FIX_M>(fixMEvent_[slot]);
        mFixEvent_[slot] = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::M_FIX>());
    }
    mte2Mte1Event_ = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE2_MTE1>());
    mte1MEvent_ = static_cast<event_t>(GetTPipePtr()->AllocEventID<HardEvent::MTE1_M>());
}

template <typename SIT>
__aicore__ inline void SIMatmul<SIT>::FreeEventID()
{
    SetMMLayoutTransform(false);
    for (uint32_t slot = 0U; slot < KEY_BUF_NUM; slot++) {
        WaitFlag<HardEvent::MTE1_MTE2>(keyMte1Mte2Event_[slot]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(keyMte1Mte2Event_[slot]);
    }
    for (uint32_t slot = 0U; slot < QUERY_BUF_NUM; slot++) {
        WaitFlag<HardEvent::MTE1_MTE2>(queryMte1Mte2Event_[slot]);
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_MTE2>(queryMte1Mte2Event_[slot]);
    }
    for (uint32_t slot = 0U; slot < L0A_BUF_NUM; slot++) {
        WaitFlag<HardEvent::M_MTE1>(mMte1Event_[slot]);
        GetTPipePtr()->ReleaseEventID<HardEvent::M_MTE1>(mMte1Event_[slot]);
    }
    for (uint32_t slot = 0U; slot < L0C_BUF_NUM; slot++) {
        WaitFlag<HardEvent::FIX_M>(fixMEvent_[slot]);
        GetTPipePtr()->ReleaseEventID<HardEvent::FIX_M>(fixMEvent_[slot]);
        GetTPipePtr()->ReleaseEventID<HardEvent::M_FIX>(mFixEvent_[slot]);
    }
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_MTE1>(mte2Mte1Event_);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE1_M>(mte1MEvent_);
}
} // namespace SIKernel
#endif
