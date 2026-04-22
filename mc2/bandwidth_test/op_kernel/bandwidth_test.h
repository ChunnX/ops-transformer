/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file bandwidth_test.h
 * \brief Bandwidth test kernel - simplified version of moe_distribute_dispatch_v2 fullmesh_v1
 */

#ifndef BANDWIDTH_TEST_H
#define BANDWIDTH_TEST_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "bandwidth_test_tiling.h"
#include "adv_api/hccl/hccl.h"
#include "../../common/op_kernel/moe_distribute_base.h"
#include "../../common/op_kernel/mc2_kernel_utils.h"
#if __has_include("../moe_distribute_dispatch_v2/moe_distribute_v2_base.h")
#include "../moe_distribute_dispatch_v2/moe_distribute_v2_base.h"
#else
#include "../../moe_distribute_dispatch_v2/op_kernel/moe_distribute_v2_base.h"
#endif

namespace BandwidthTestImpl {

using namespace AscendC;
using namespace Mc2Kernel;

#define TemplateBandwidthTestTypeClass \
    typename XType, \
    typename YType
#define TemplateBandwidthTestTypeFunc XType, YType

template <TemplateBandwidthTestTypeClass>
class BandwidthTest {
public:
    __aicore__ inline BandwidthTest() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertId, GM_ADDR y, GM_ADDR receiveCnt,
        GM_ADDR workspaceGM, TPipe *pipe, const BandwidthTestTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void AlltoAllDispatch();
    __aicore__ inline void SendToMoeExpert();
    __aicore__ inline void SetStatus();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void BufferInit();
    __aicore__ inline void ProcessToken(GlobalTensor<XType>& outTokenGT, uint32_t tokenIndex);
    __aicore__ inline GM_ADDR GetWindAddrByRankId(const int32_t rankId)
    {
        return Mc2Kernel::GetBaseWindAddrByRankId(winContext_, rankId, epRankId_) + winDataSizeOffset_;
    }

    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(const int32_t rankId)
    {
        return Mc2Kernel::GetBaseWindStateAddrByRankId(winContext_, rankId, epRankId_) + dataState_ * WIN_STATE_OFFSET;
    }

    TPipe *tpipe_{nullptr};
    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<int32_t> expertIdGMTensor_;
    GlobalTensor<YType> yGMTensor_;
    GlobalTensor<int32_t> receiveCntGMTensor_;
    GlobalTensor<float> windowInStatusFp32Tensor_;

    LocalTensor<XType> xTmpTensor_;
    LocalTensor<YType> yTmpTensor_;
    LocalTensor<int32_t> expertIdsTensor_;
    LocalTensor<int32_t> statusTensor_;
    LocalTensor<float> statusFp32Tensor_;

    TBuf<> expertIdsBuf_;
    TBuf<> statusBuf_;
    TBuf<> waitStatusBuf_;
    TBuf<> gatherMaskOutBuf_;
    TBuf<> scalarBuf_;
    TBuf<> tokenNumBuf_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> xQueue_;

    GM_ADDR yGM_;
    GM_ADDR receiveCntGM_;
    GM_ADDR statusSpaceGm_;
    GM_ADDR windowGM_;

    __gm__ HcclOpParam *winContext_{nullptr};

    uint32_t axisBS_{0};
    uint32_t axisH_{0};
    uint32_t aivNum_{0};
    uint32_t epWorldSize_{0};
    int32_t epRankId_{0};
    int32_t mode_{0};
    uint32_t aivId_{0};
    uint32_t globalBS_{0};
    uint32_t hOutSize_{0};
    uint32_t hAlignWinSize_{0};
    uint32_t hAlignWinCnt_{0};
    uint32_t hOutAlignUbSize_{0};
    uint32_t hAlignSize_{0};
    uint32_t startExpertId_{0};
    uint32_t endExpertId_{0};
    uint32_t sendExpertNum_{0};
    uint32_t totalCnt_{0};
    uint32_t dataState_{0};
    uint32_t totalUsedUB_{0};
    uint64_t winDataSizeOffset_{0};
    uint64_t expertPerSizeOnWin_{0};
    uint32_t stateOffset_{0};
    uint32_t recStatusNumPerCore_{0};
    uint32_t rscvStatusNum_{0};
    uint32_t remainderRankNum_{0};
    uint32_t startStatusIndex_{0};
    uint32_t bufferNum_{0};
    uint32_t totalUbSize_{0};
    uint64_t totalWinSizeEp_{0};
    float sumTarget_;
};

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::Init(
    GM_ADDR x, GM_ADDR expertId, GM_ADDR y, GM_ADDR receiveCnt,
    GM_ADDR workspaceGM, TPipe *pipe, const BandwidthTestTilingData *tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    epRankId_ = tilingData->rank_id;

    totalWinSizeEp_ = tilingData->total_win_size;
    winContext_ = (__gm__ HcclOpParam *)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();

    totalUbSize_ = static_cast<uint32_t>(tilingData->total_ub_size);
    axisBS_ = tilingData->bs;
    axisH_ = tilingData->data_size;
    epWorldSize_ = tilingData->world_size;
    globalBS_ = tilingData->max_bs;
    aivNum_ = tilingData->aiv_num;
    mode_ = tilingData->mode;

    TBuf<> dataStateBuf;
    tpipe_->InitBuffer(dataStateBuf, UB_ALIGN);
    statusSpaceGm_ = Mc2Kernel::GetStatusDataSpaceGm(winContext_);
    GlobalTensor<uint32_t> selfDataStatusGMTensor;
    selfDataStatusGMTensor.SetGlobalBuffer((__gm__ uint32_t*)(statusSpaceGm_ + DISPATCH_STATE_WIN_OFFSET + aivId_ * WIN_ADDR_ALIGN));
    uint32_t epRankIdHccl = Mc2Kernel::GetRankId(winContext_);
    uint32_t epWorldSizeHccl = Mc2Kernel::GetRankDim(winContext_);
    dataState_ = MoeDistributeV2Base::InitWinState(selfDataStatusGMTensor, epRankIdHccl, epWorldSizeHccl,
                                                    epRankId_, epWorldSize_, epWorldSize_, globalBS_, dataStateBuf);

    xGMTensor_.SetGlobalBuffer((__gm__ XType*)x);
    expertIdGMTensor_.SetGlobalBuffer((__gm__ int32_t*)expertId);
    yGMTensor_.SetGlobalBuffer((__gm__ YType*)y);
    receiveCntGMTensor_.SetGlobalBuffer((__gm__ int32_t*)receiveCnt);

    yGM_ = y;
    receiveCntGM_ = receiveCnt;

    hOutSize_ = axisH_ * sizeof(XType);
    hAlignSize_ = Ceil(axisH_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;
    hAlignWinSize_ = Ceil(hAlignSize_, WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    hAlignWinCnt_ = hAlignWinSize_ / sizeof(XType);
    hOutAlignUbSize_ = hAlignSize_;

    uint32_t axisMaxBS = globalBS_ / epWorldSize_;
    expertPerSizeOnWin_ = axisMaxBS * hAlignWinSize_;

    stateOffset_ = STATE_OFFSET;
    rscvStatusNum_ = epWorldSize_;
    recStatusNumPerCore_ = rscvStatusNum_ / aivNum_;
    remainderRankNum_ = rscvStatusNum_ % aivNum_;
    startStatusIndex_ = recStatusNumPerCore_ * aivId_;
    if (aivId_ < remainderRankNum_) {
        recStatusNumPerCore_ += 1;
        startStatusIndex_ += aivId_;
    } else {
        startStatusIndex_ += remainderRankNum_;
    }

    uint32_t statusBufCntAlign = Ceil(Ceil(epWorldSize_, aivNum_), 8) * 8;
    uint32_t statusBufSize = statusBufCntAlign * UB_ALIGN;
    tpipe_->InitBuffer(statusBuf_, statusBufSize);
    totalUsedUB_ += statusBufSize;
    statusTensor_ = statusBuf_.Get<int32_t>();
    Duplicate<int32_t>(statusTensor_, 0, statusBufCntAlign * 8);
    statusSpaceGm_ = GetWindStateAddrByRankId(epRankId_);

    sumTarget_ = static_cast<float>(1.0);
    uint64_t mask[2] = { 0x101010101010101, 0 };
    PipeBarrier<PIPE_V>();
    Duplicate<int32_t>(statusTensor_, 0x3F800000, mask, statusBufCntAlign / 8, 1, 8);

    uint64_t hSizeAlignCombine = Ceil(axisH_ * sizeof(XType), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    winDataSizeOffset_ = dataState_ * (totalWinSizeEp_ / 2) + axisMaxBS * hSizeAlignCombine;
    windowGM_ = GetWindAddrByRankId(epRankId_);
    windowInStatusFp32Tensor_.SetGlobalBuffer((__gm__ float*)(statusSpaceGm_));

    uint32_t expertIdsSize = Ceil(axisBS_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(expertIdsBuf_, expertIdsSize);
    totalUsedUB_ += expertIdsSize;
    expertIdsTensor_ = expertIdsBuf_.Get<int32_t>();

    uint32_t maxSize = expertIdsSize;
    tpipe_->InitBuffer(gatherMaskOutBuf_, maxSize);
    totalUsedUB_ += maxSize;

    uint32_t tmpTotalUB = totalUsedUB_ + hOutAlignUbSize_ * BUFFER_NUM;
    bufferNum_ = tmpTotalUB > MAX_UB_SIZE ? BUFFER_SINGLE : BUFFER_NUM;
    tpipe_->InitBuffer(xQueue_, bufferNum_, hOutAlignUbSize_);
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::ProcessToken(
    GlobalTensor<XType>& outTokenGT, uint32_t tokenIndex)
{
    DataCopyPadParams padParams = {true, 0, 0, 0};
    DataCopyParams xCopyParams = {1U, static_cast<uint16_t>(axisH_ * sizeof(XType)), 0U, 0U};
    DataCopyParams hCommuCopyOutParams = {1U, static_cast<uint16_t>(hAlignSize_), 0U, 0U};

    xTmpTensor_ = xQueue_.AllocTensor<XType>();
    DataCopyPad(xTmpTensor_, xGMTensor_[tokenIndex * axisH_], xCopyParams, padParams);
    xQueue_.EnQue(xTmpTensor_);
    xTmpTensor_ = xQueue_.DeQue<XType>();

    SyncFunc<AscendC::HardEvent::MTE3_S>();
    LocalTensor<int32_t> xOutTint32 = xTmpTensor_.template ReinterpretCast<int32_t>();
    xOutTint32(axisH_) = epRankId_;
    xOutTint32(axisH_ + 1) = tokenIndex;
    SyncFunc<AscendC::HardEvent::S_MTE3>();

    DataCopyPad(outTokenGT, xTmpTensor_, hCommuCopyOutParams);
    xQueue_.FreeTensor<XType>(xTmpTensor_);
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::SendToMoeExpert()
{
    uint32_t sendTokenNumPerCore = axisBS_ / aivNum_;
    uint32_t remainder = axisBS_ % aivNum_;
    uint32_t startTokenId = sendTokenNumPerCore * aivId_;
    if (aivId_ < remainder) {
        sendTokenNumPerCore += 1;
        startTokenId += aivId_;
    } else {
        startTokenId += remainder;
    }
    uint32_t endTokenId = startTokenId + sendTokenNumPerCore;

    if (startTokenId >= axisBS_) { return; }

    GlobalTensor<XType> dstWinGMTensor;
    DataCopyExtParams expertIdsParams = {1U, static_cast<uint32_t>(axisBS_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> expertIdsCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(expertIdsTensor_, expertIdGMTensor_, expertIdsParams, expertIdsCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    for (uint32_t tokenIndex = startTokenId; tokenIndex < endTokenId; ++tokenIndex) {
        uint32_t dstExpertId = expertIdsTensor_(tokenIndex);
        uint32_t dstRankId = dstExpertId;

        GM_ADDR rankGM = (__gm__ uint8_t*)(GetWindAddrByRankId(dstRankId) +
                                           epRankId_ * expertPerSizeOnWin_);
        dstWinGMTensor.SetGlobalBuffer((__gm__ XType*)rankGM);
        GlobalTensor<XType> tempTensor = dstWinGMTensor[tokenIndex * hAlignWinCnt_];
        ProcessToken(tempTensor, tokenIndex);
    }
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::AlltoAllDispatch()
{
    SendToMoeExpert();
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::SetStatus()
{
    uint32_t sendExpertNum = epWorldSize_ / aivNum_;
    uint32_t remainder = epWorldSize_ % aivNum_;
    uint32_t startExpertId = sendExpertNum * aivId_;
    if (aivId_ < remainder) {
        sendExpertNum += 1;
        startExpertId += aivId_;
    } else {
        startExpertId += remainder;
    }
    uint32_t endExpertId = startExpertId + sendExpertNum;

    TBuf<> dstExpBuf;
    tpipe_->InitBuffer(dstExpBuf, Ceil(axisBS_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);
    LocalTensor<int32_t> dstExpIdTensor = dstExpBuf.Get<int32_t>();
    TBuf<> subExpBuf;
    tpipe_->InitBuffer(subExpBuf, Ceil(axisBS_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN);
    LocalTensor<int32_t> subExpIdTensor = subExpBuf.Get<int32_t>();
    LocalTensor<float> workLocalTensor = gatherMaskOutBuf_.Get<float>();

    DataCopyExtParams expertIdsParams = {1U, static_cast<uint32_t>(axisBS_ * sizeof(int32_t)), 0U, 0U, 0U};
    DataCopyPadExtParams<int32_t> expertIdsCopyPadParams{false, 0U, 0U, 0U};
    DataCopyPad(expertIdsTensor_, expertIdGMTensor_, expertIdsParams, expertIdsCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();

    for (uint32_t curExpertId = startExpertId; curExpertId < endExpertId; ++curExpertId) {
        int32_t curExpertCnt = 0;
        SyncFunc<AscendC::HardEvent::S_V>();
        Duplicate<int32_t>(dstExpIdTensor, static_cast<int32_t>(curExpertId), axisBS_);
        PipeBarrier<PIPE_V>();
        Sub(subExpIdTensor, expertIdsTensor_, dstExpIdTensor, axisBS_);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> tmpFp32 = subExpIdTensor.ReinterpretCast<float>();
        LocalTensor<float> tmpoutFp32 = dstExpIdTensor.ReinterpretCast<float>();
        Abs(tmpoutFp32, tmpFp32, axisBS_);
        PipeBarrier<PIPE_V>();
        Mins(subExpIdTensor, dstExpIdTensor, 1, axisBS_);
        PipeBarrier<PIPE_V>();
        ReduceSum<float>(tmpoutFp32, tmpoutFp32, workLocalTensor, axisBS_);
        SyncFunc<AscendC::HardEvent::V_S>();
        curExpertCnt = dstExpIdTensor(0);
        statusTensor_((curExpertId - startExpertId) * 8 + 1) = curExpertCnt;
    }

    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
    if (startExpertId >= epWorldSize_) { return; }

    GlobalTensor<int32_t> rankGMTensor;
    for (uint32_t expertIndex = startExpertId; expertIndex < endExpertId; ++expertIndex) {
        uint32_t dstRankId = expertIndex;
        uint32_t offset = stateOffset_ * epRankId_;
        GM_ADDR rankGM = (__gm__ uint8_t*)(GetWindStateAddrByRankId(dstRankId) + offset);
        rankGMTensor.SetGlobalBuffer((__gm__ int32_t*)rankGM);
        DataCopy<int32_t>(rankGMTensor, statusTensor_[(expertIndex - startExpertId) * 8], 8UL);
    }
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::BufferInit()
{
    tpipe_->Reset();
    totalUsedUB_ = 0U;
    uint32_t waitStatusBufSize = (((recStatusNumPerCore_ * UB_ALIGN) > 256) ? (recStatusNumPerCore_ * UB_ALIGN) : 256);
    tpipe_->InitBuffer(waitStatusBuf_, waitStatusBufSize);
    totalUsedUB_ += waitStatusBufSize;

    uint64_t recStatusNumPerCoreSpace = Ceil(recStatusNumPerCore_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    tpipe_->InitBuffer(gatherMaskOutBuf_, recStatusNumPerCoreSpace);
    totalUsedUB_ += recStatusNumPerCoreSpace;

    tpipe_->InitBuffer(scalarBuf_, UB_ALIGN * 2);
    totalUsedUB_ += UB_ALIGN * 2;

    tpipe_->InitBuffer(xQueue_, BUFFER_NUM, hOutAlignUbSize_);
    totalUsedUB_ += BUFFER_NUM * hOutAlignUbSize_;

    tpipe_->InitBuffer(tokenNumBuf_, UB_ALIGN);
    totalUsedUB_ += UB_ALIGN;

    uint32_t statusBufSize = epWorldSize_ * UB_ALIGN;
    tpipe_->InitBuffer(statusBuf_, statusBufSize);
    totalUsedUB_ += statusBufSize;
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::WaitDispatch()
{
    BufferInit();
    startExpertId_ = startStatusIndex_;
    endExpertId_ = startExpertId_ + recStatusNumPerCore_;
    sendExpertNum_ = recStatusNumPerCore_;

    if (unlikely(startStatusIndex_ >= rscvStatusNum_)) {
        SyncAll<true>();
        return;
    }

    LocalTensor<float> statusSumOutTensor = scalarBuf_.GetWithOffset<float>(UB_ALIGN / sizeof(float), UB_ALIGN);
    statusFp32Tensor_ = waitStatusBuf_.Get<float>();
    float compareTarget = sumTarget_ * recStatusNumPerCore_;
    float sumOfFlag = static_cast<float>(-1.0);
    DataCopyParams intriParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, 0};

    SyncFunc<AscendC::HardEvent::S_V>();
    while (sumOfFlag != compareTarget) {
        DataCopy(statusFp32Tensor_, windowInStatusFp32Tensor_[startStatusIndex_ * stateOffset_ / sizeof(float)], intriParams);
        SyncFunc<AscendC::HardEvent::MTE2_S>();
        SyncFunc<AscendC::HardEvent::S_V>();
        ReduceSum(statusSumOutTensor, statusFp32Tensor_, gatherMaskOutBuf_.Get<float>(), 1, recStatusNumPerCore_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = statusSumOutTensor.GetValue(0);
    }

    DataCopyParams intriOutParams{static_cast<uint16_t>(recStatusNumPerCore_), 1, 0, 0};
    uint64_t duplicateMask[2] = { 0x101010101010101, 0 };
    LocalTensor<int32_t> cleanStateTensor = waitStatusBuf_.Get<int32_t>();
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(cleanStateTensor, 0, duplicateMask, Ceil(recStatusNumPerCore_, 8), 1, 8);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(windowInStatusFp32Tensor_[startStatusIndex_ * stateOffset_ / sizeof(float)],
             cleanStateTensor.ReinterpretCast<float>(), intriOutParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();

    SyncAll<true>();
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::LocalWindowCopy()
{
    if (startExpertId_ >= rscvStatusNum_) { return; }

    statusTensor_ = waitStatusBuf_.Get<int32_t>();
    uint32_t preCnt = 0;
    for (uint32_t i = 0; i < aivId_; i++) {
        uint32_t expertId = startStatusIndex_ + i;
        if (expertId < epWorldSize_) {
            preCnt += statusTensor_((expertId - startExpertId_) * 8 + 1);
        }
    }

    uint32_t hOutElemCount = hOutSize_ / sizeof(XType);
    DataCopyPadParams padParams = {false, 0U, 0U, 0U};
    DataCopyExtParams dataCopyOutParams{1U, static_cast<uint32_t>(sendExpertNum_ * sizeof(int32_t)), 0U, 0U, 0U};

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> tripleQueue;
    uint32_t leftUBSize = ((totalUbSize_ - totalUsedUB_) / UB_ALIGN) * UB_ALIGN;
    uint32_t dataSize = leftUBSize / BUFFER_NUM;
    tpipe_->InitBuffer(tripleQueue, BUFFER_NUM, dataSize);

    uint32_t countPerDataSize = dataSize / hOutAlignUbSize_;
    uint32_t beginIdx = preCnt;
    constexpr uint32_t EXPAND_IDX_INFO_SIMPLIFIED = 2U;

    for (uint32_t index = startExpertId_; index < endExpertId_; index++) {
        uint32_t i = index - startExpertId_;
        uint32_t count = statusTensor_.GetValue(i * 8 + 1);
        GM_ADDR wAddr = (__gm__ uint8_t*)(windowGM_) + index * expertPerSizeOnWin_;
        GlobalTensor<YType> tokGlobal, yOutGlobal;
        tokGlobal.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

        uint32_t loopTimes = Ceil(count, countPerDataSize);
        uint32_t lastTimeCount = count - countPerDataSize * (loopTimes - 1);
        for (uint32_t loopIdx = 0; loopIdx < loopTimes; loopIdx++) {
            uint32_t tokenCopiedNumThisExpert = countPerDataSize * loopIdx;
            tokGlobal.SetGlobalBuffer((__gm__ YType *)(wAddr + tokenCopiedNumThisExpert * hAlignWinSize_));

            uint16_t blockCount = static_cast<uint16_t>(((loopIdx < loopTimes - 1) ? countPerDataSize : lastTimeCount));
            uint16_t copyBlockLen = static_cast<uint16_t>(hOutAlignUbSize_ / UB_ALIGN);
            uint16_t srcStrideTokGlobal = static_cast<uint16_t>((hAlignWinSize_ - hOutAlignUbSize_) / UB_ALIGN);
            DataCopyParams dataCopyParamsTokGlobal{blockCount, copyBlockLen, srcStrideTokGlobal, 0U};
            LocalTensor<YType> tripleLocal = tripleQueue.AllocTensor<YType>();
            DataCopy(tripleLocal, tokGlobal, dataCopyParamsTokGlobal);
            tripleQueue.EnQue(tripleLocal);
            tripleLocal = tripleQueue.DeQue<YType>();

            yOutGlobal.SetGlobalBuffer((__gm__ YType *)(yGM_ + (beginIdx + tokenCopiedNumThisExpert) * hOutSize_));
            uint32_t srcStrideY = (hOutAlignUbSize_ - hOutSize_) / UB_ALIGN;
            DataCopyExtParams dataCopyParamsY{blockCount, hOutSize_, srcStrideY, 0U, 0U};
            DataCopyPad(yOutGlobal, tripleLocal, dataCopyParamsY);

            tripleQueue.FreeTensor(tripleLocal);
        }
        beginIdx += count;
    }
    totalCnt_ = beginIdx;
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    DataCopyPad(receiveCntGMTensor_[startExpertId_], statusTensor_, dataCopyOutParams);
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        if (mode_ == 0) {
            AlltoAllDispatch();
            SetStatus();
            WaitDispatch();
            LocalWindowCopy();
        } else if (mode_ == 1) {
            AlltoAllDispatch();
        }
    }
}

} // namespace BandwidthTestImpl
#endif // BANDWIDTH_TEST_H
