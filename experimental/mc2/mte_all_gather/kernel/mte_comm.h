/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mte_comm.h
 * \brief quant_all_gather MTE通信相关 kernel代码公共部分
 */

#ifndef MTE_COMMON_H
#define MTE_COMMON_H

#include "adv_api/hccl/hccl.h"
#include "adv_api/reduce/sum.h"

namespace MTECommImpl {

using namespace AscendC;
constexpr static uint32_t UB_ALIGN_BYTES = 32U;
constexpr uint32_t BUFFER_NUM = 2U;
constexpr static uint32_t X_BLOCK_BYTES = 1024U;
constexpr static uint64_t STATE_WIN_SIZE = 1024UL * 1024UL;
constexpr static uint64_t SINGLE_STATE_REGION_SIZE = 462UL * 1024UL;
constexpr static uint32_t ZERONE_STATE_POS = 0U;
constexpr static uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint32_t FLOAT_UB_ALIGN_NUM = 8U;

template<typename XType>
class MTECommunication {
public:
    __aicore__ inline MTECommunication() {};
    __aicore__ inline void InitHcclContext();
    __aicore__ inline void InitHcclContextByAddr(GM_ADDR mc2Context);
    __aicore__ inline void InitParams();
    __aicore__ inline void InitGMTensor(GM_ADDR x, GM_ADDR output, uint64_t xNums, uint64_t alignedXSize);
    __aicore__ inline void InitBuffer(TPipe *tPipe);
    __aicore__ inline void SetBlockSize(uint32_t elementsPerBlock, uint64_t aivNum, uint64_t lastBlockNum);
    __aicore__ inline void CopyDataToWin();
    __aicore__ inline void WriteStatusToWin();
    __aicore__ inline void ReadStatus();
    __aicore__ inline void CopyXBlockToOutput(uint64_t outOffsetGM, uint64_t winXOffset, uint32_t count);
    __aicore__ inline void ComputeTailAivId(uint64_t totalAivCount);
    __aicore__ inline GM_ADDR GetWinAddrGm(uint32_t rankId, uint64_t offset = 0);
    __aicore__ inline GM_ADDR GetWinDataAddrGm(uint32_t rankId, uint32_t winFlag);
    __aicore__ inline GM_ADDR GetWinStatusAddrGm(uint32_t rankId, uint32_t winFlag);

    __gm__ HcclA5OpResParam *hcclContext_;
    uint32_t aivId_{0};
    uint64_t aivNum_{0};
    uint32_t round_{0};
    uint32_t tailBlockNums_{0};
    uint32_t assignedBlockNums_{0};
    uint64_t xOffset_{0};
    uint64_t lastAivId_{0};
    uint32_t winBufferFlags_{0};
    uint64_t xNums_{0};
    GlobalTensor<XType> outputTensor_;
private:

    uint32_t xNumPerBlock_{0};
    uint64_t tailXNums_{0};

    __aicore__ inline void CopyDataBlock(uint64_t curXOffset, uint32_t count);

    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<XType> localWinXGMTensor_;
    GlobalTensor<uint32_t> selfWinFlagGMTensor_;

    LocalTensor<XType> xTmpTensor_;
    LocalTensor<float> stateResetTensor_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_;
    TBuf<> winFlagsBuf_;
    TBuf<> writeStateBuf_;
    TBuf<> readStateBuf_;
    TBuf<> stateResetBuf_;
};

template <typename XType>
__aicore__ inline void MTECommunication<XType>::InitHcclContext()
{
    hcclContext_ = (__gm__ HcclA5OpResParam*)GetHcclContext<HCCL_GROUP_ID_0>();
}

template <typename XType>
__aicore__ inline void MTECommunication<XType>::InitHcclContextByAddr(GM_ADDR mc2Context)
{
    hcclContext_ = (__gm__ HcclA5OpResParam*)(mc2Context);
}

template <typename XType>
__aicore__ inline void MTECommunication<XType>::InitParams()
{
    aivId_ = GetBlockIdx();
    assignedBlockNums_ = aivId_ < tailBlockNums_ ? round_ + 1 : round_;
    uint64_t blockIdx = aivId_ * round_ + (aivId_ < tailBlockNums_ ? aivId_ : tailBlockNums_);
    xOffset_ = blockIdx * xNumPerBlock_;
}

template <typename XType>
__aicore__ inline void MTECommunication<XType>::InitGMTensor(GM_ADDR x, GM_ADDR output, uint64_t xNums, uint64_t xSize)
{
    xNums_ = xNums;
    xGMTensor_.SetGlobalBuffer((__gm__ XType*)x);
    outputTensor_.SetGlobalBuffer((__gm__ XType*)output);

    uint64_t currCoreFlagOffset = 2UL * SINGLE_STATE_REGION_SIZE + aivId_ * WIN_ADDR_ALIGN;
    selfWinFlagGMTensor_.SetGlobalBuffer((__gm__ uint32_t*)GetWinAddrGm(hcclContext_->rankId, currCoreFlagOffset));
    LocalTensor<uint32_t> winFlagLocalTensor = winFlagsBuf_.Get<uint32_t>();
    DataCopy(winFlagLocalTensor, selfWinFlagGMTensor_, UB_ALIGN_BYTES / sizeof(uint32_t));
    SyncFunc<AscendC::HardEvent::MTE2_S>();
    winBufferFlags_ = winFlagLocalTensor.GetValue(ZERONE_STATE_POS);
    winFlagLocalTensor.SetValue(ZERONE_STATE_POS, 1 - winBufferFlags_);
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy(selfWinFlagGMTensor_, winFlagLocalTensor, UB_ALIGN_BYTES / sizeof(uint32_t));

    GM_ADDR localDataSpaceGm = GetWinDataAddrGm(hcclContext_->rankId, winBufferFlags_);
    localWinXGMTensor_.SetGlobalBuffer((__gm__ XType*)localDataSpaceGm);
}

template <typename XType>
__aicore__ inline void MTECommunication<XType>::InitBuffer(TPipe *tPipe)
{
    tPipe->InitBuffer(xQueue_, BUFFER_NUM, X_BLOCK_BYTES);
    tPipe->InitBuffer(winFlagsBuf_, UB_ALIGN_BYTES);
    tPipe->InitBuffer(writeStateBuf_, UB_ALIGN_BYTES);
    tPipe->InitBuffer(readStateBuf_, hcclContext_->rankDim * UB_ALIGN_BYTES);
    tPipe->InitBuffer(stateResetBuf_, hcclContext_->rankDim * UB_ALIGN_BYTES);

    stateResetTensor_ = stateResetBuf_.Get<float>();
    Duplicate<float>(stateResetTensor_, (float)0.0, static_cast<uint32_t>(hcclContext_->rankDim * FLOAT_UB_ALIGN_NUM));
}

template <typename XType>
__aicore__ inline void MTECommunication<XType>::SetBlockSize(uint32_t elementsPerBlock, uint64_t aivNum, uint64_t lastBlockNum)
{
    xNumPerBlock_ = elementsPerBlock;
    aivNum_ = aivNum;
    tailXNums_ = lastBlockNum;
}

template <typename XType>
__aicore__ inline void MTECommunication<XType>::ComputeTailAivId(uint64_t totalAivCount)
{
    if (round_ == 0) {
        lastAivId_ = tailBlockNums_ - 1;
    } else {
        lastAivId_ = totalAivCount - 1;
    }
}

template <typename XType>
__aicore__ inline void MTECommunication<XType>::CopyDataBlock(uint64_t curXOffset, uint32_t count)
{
    xTmpTensor_ = xQueue_.AllocTensor<XType>();
    DataCopy(xTmpTensor_, xGMTensor_[curXOffset], count);
    xQueue_.EnQue(xTmpTensor_);
    xTmpTensor_ = xQueue_.DeQue<XType>();
    DataCopy(localWinXGMTensor_[curXOffset], xTmpTensor_, count);
    xQueue_.FreeTensor<XType>(xTmpTensor_);
}

template <typename XType>
__aicore__ inline void MTECommunication<XType>::CopyDataToWin()
{
    for (uint64_t curBlock = 0; curBlock < assignedBlockNums_; ++curBlock) {
        uint64_t curXOffset = xOffset_ + curBlock * xNumPerBlock_;
        uint32_t copyBlockNum = xNumPerBlock_;
        if ((aivId_ ==  lastAivId_) && (curBlock == assignedBlockNums_ - 1)) {
            copyBlockNum = tailXNums_;
        }
        CopyDataBlock(curXOffset, copyBlockNum);
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename XType>
__aicore__ inline void MTECommunication<XType>::WriteStatusToWin()
{
    uint32_t coreOffset = aivId_ * hcclContext_->rankDim;
    for (uint32_t curRank = 0; curRank < hcclContext_->rankDim; ++curRank) {
        LocalTensor<float> statusTensor = writeStateBuf_.Get<float>();
        DataCopy<float>(statusTensor, stateResetTensor_, FLOAT_UB_ALIGN_NUM);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        statusTensor(0) = (float)1.0;
        GM_ADDR remoteWinStateGM = GetWinStatusAddrGm(curRank, winBufferFlags_);
        GlobalTensor<float> stateGMTensor;
        stateGMTensor.SetGlobalBuffer((__gm__ float*)remoteWinStateGM);
        uint64_t curOffset = (coreOffset + hcclContext_->rankId) * FLOAT_UB_ALIGN_NUM;
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        DataCopy(stateGMTensor[curOffset], statusTensor, FLOAT_UB_ALIGN_NUM);
        SyncFunc<AscendC::HardEvent::MTE3_S>();
    }
}

template <typename XType>
__aicore__ inline void MTECommunication<XType>::ReadStatus()
{
    GM_ADDR stateGM = GetWinStatusAddrGm(hcclContext_->rankId, winBufferFlags_);
    GlobalTensor<float> selfStatusWinTensor;
    uint32_t offset = aivId_ * hcclContext_->rankDim * FLOAT_UB_ALIGN_NUM;
    selfStatusWinTensor.SetGlobalBuffer((__gm__ float*)(stateGM));
    LocalTensor<float> statusTensor = readStateBuf_.Get<float>();
    float flag = 0;
    uint32_t statusCnt = hcclContext_->rankDim * FLOAT_UB_ALIGN_NUM;
    SumParams sumParams{1, statusCnt, statusCnt};
    float minTarget = hcclContext_->rankDim - (float)0.5;
    float maxTarget = hcclContext_->rankDim + (float)0.5;
    while ((flag < minTarget) || (flag > maxTarget)) {
        SyncFunc<AscendC::HardEvent::S_MTE2>();
        DataCopy<float>(statusTensor, selfStatusWinTensor[offset], statusCnt);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        Sum(statusTensor, statusTensor, sumParams);
        SyncFunc<AscendC::HardEvent::V_S>();
        flag = statusTensor(0);
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopy<float>(selfStatusWinTensor[offset], stateResetTensor_, statusCnt);
}

template <typename XType>
__aicore__ inline void MTECommunication<XType>::CopyXBlockToOutput(uint64_t outOffsetGM, uint64_t winXOffset, uint32_t count)
{
    DataCopy(outputTensor_[outOffsetGM], localWinXGMTensor_[winXOffset], count);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <typename XType>
__aicore__ inline GM_ADDR MTECommunication<XType>::GetWinAddrGm(uint32_t rankId, uint64_t offset)
{
    return (GM_ADDR)(hcclContext_->windowsIn[rankId] + offset);
}

template <typename XType>
__aicore__ inline GM_ADDR MTECommunication<XType>::GetWinDataAddrGm(uint32_t rankId, uint32_t winFlag)
{
    if (winFlag == 0U) {
        return GetWinAddrGm(rankId, STATE_WIN_SIZE);
    }
    else {
        return (GM_ADDR)(hcclContext_->windowsOut[rankId]);
    }
}

template <typename XType>
__aicore__ inline GM_ADDR MTECommunication<XType>::GetWinStatusAddrGm(uint32_t rankId, uint32_t winFlag)
{
    if (winFlag == 0U) {
        return GetWinAddrGm(rankId);
    }
    else {
        return GetWinAddrGm(rankId, SINGLE_STATE_REGION_SIZE);
    }
}
}
#endif  // MTE_COMMON_H
