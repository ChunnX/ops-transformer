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

namespace BandwidthTestImpl {
constexpr uint8_t BUFFER_NUM = 1;
constexpr uint32_t UB_ALIGN = 32;
constexpr uint64_t WIN_ADDR_ALIGN = 512UL;
constexpr uint32_t STATE_OFFSET = 32U;
constexpr uint32_t WIN_STATE_OFFSET = 384UL * 1024UL;
constexpr uint32_t FLAG_FIELD_OFFSET = 768UL * 1024UL;
constexpr uint32_t TIME_CYCLE = 50;
constexpr uint32_t MAX_TILE_SIZE = 8192;
constexpr uint32_t UB_ALIGN_DATA_COUNT = 8U;
constexpr uint32_t CUMSUM_MAX_CORE_NUM = 16U;
constexpr uint32_t SIZE_ALIGN_256 = 256U;
constexpr uint32_t BITS_PER_BYTE = 8U;

#define TemplateBandwidthTestTypeClass \
    typename XType, \
    typename YType
#define TemplateBandwidthTestTypeFunc XType, YType

using namespace AscendC;
using namespace Mc2Kernel;

template <TemplateBandwidthTestTypeClass>
class BandwidthTest {
public:
    __aicore__ inline BandwidthTest() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertId, GM_ADDR y, GM_ADDR receiveCnt,
        GM_ADDR workspaceGM, TPipe *pipe, const BandwidthTestTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void AllToAllDispatch();
    __aicore__ inline void TokenToExpert(GlobalTensor<YType> &dstWinGMTensor, uint32_t tokenIndex, uint32_t dstRank);
    __aicore__ inline void CalExpertSendNum();
    __aicore__ inline void SendToExpert();
    __aicore__ inline void CalAndSendCnt();
    __aicore__ inline void WaitDispatch();
    __aicore__ inline void LocalWindowCopy();
    __aicore__ inline void SplitToCore(uint32_t totalCnt, uint32_t useAivNum, 
        uint32_t &startIdx, uint32_t &endIdx, uint32_t &procNum, bool isFront = true);
    __aicore__ inline GM_ADDR GetWindAddrByRankId(const int32_t rankId);
    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(const int32_t rankId);

    TPipe *tpipe_{nullptr};
    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<int32_t> expertIdGMTensor_;
    GlobalTensor<YType> yGMTensor_;
    GlobalTensor<int32_t> receiveCntGMTensor_;
    GlobalTensor<float> selfRankWinInGMTensor_;
    GlobalTensor<float> windowInStatusFp32Tensor_;

    LocalTensor<YType> yLocalTensor_;
    LocalTensor<int32_t> statusTensor_;
    LocalTensor<float> statusFp32Tensor_;
    LocalTensor<float> statusSumOutTensor_;
    LocalTensor<int32_t> cleanStatusTensor_;
    LocalTensor<int32_t> sendCntTensor_;
    LocalTensor<int32_t> recvCntTensor_;
    LocalTensor<int32_t> expertIdLocalTensor_;
    LocalTensor<int32_t> tokenNumToExpertTensor_;
    LocalTensor<uint8_t> expertMaskTensor_;
    LocalTensor<int32_t> gatherIndexTensor_;
    LocalTensor<XType> xLocalTensor_;
    LocalTensor<float> workLocalTensor_;
    
    TBuf<> statusBuf_;
    TBuf<> statusFp32Buf_;
    TBuf<> statusSumBuf_;
    TBuf<> cleanStatusBuf_;
    TBuf<> sendCntBuf_;
    TBuf<> recvCntBuf_;
    TBuf<> expertIdBuf_;
    TBuf<> workLocalBuf_;
    TBuf<> expertMaskBuf_;
    TBuf<> tokenNumToExpertBuf_;
    TBuf<> gatherIndexBuf_;
    TBuf<> xBuf_;
    TBuf<> yBuf_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue_;

    GM_ADDR yGM_;
    GM_ADDR receiveCntGM_;
    GM_ADDR windowGM_;
    GM_ADDR statusSpaceGM_;
    GM_ADDR recvCntWorkspaceGM_;

    __gm__ HcclOpParam *winContext_{nullptr};

    uint32_t aivId_{0};
    uint32_t aivNum_{0};
    uint32_t aivUsedAllToAll_{0};
    uint32_t aivUsedCumSum_{0};
    uint32_t axisBS_{0};
    uint32_t dataSize_{0};
    uint32_t epWorldSize_{0};
    int32_t epRankId_{0};
    uint32_t axisMaxBS_{0};
    uint32_t mode_{0};
    uint64_t expertPerSizeOnWin_{0};
    uint64_t winDataSizeOffset_{0};
    uint32_t tileSize_{0};
    uint32_t statusCntAlign_{0};
    int64_t commStartTime_{0};
    int64_t commEndTime_{0};
    
    uint32_t startId_{0};
    uint32_t endId_{0};
    uint32_t sendNum_{0};
    uint32_t startStatusIndex_{0};
    uint32_t endStatusIndex_{0};
    uint32_t recStatusNumPerCore_{0};
    uint32_t expertIdsSize_{0};
    uint32_t maskSizePerExpert_{0};
    uint32_t hCommuSize_{0};
};

template <TemplateBandwidthTestTypeClass>
__aicore__ inline GM_ADDR BandwidthTest<TemplateBandwidthTestTypeFunc>::GetWindAddrByRankId(const int32_t rankId)
{
    return GetBaseWindAddrByRankId(winContext_, rankId, epRankId_) + winDataSizeOffset_;
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline GM_ADDR BandwidthTest<TemplateBandwidthTestTypeFunc>::GetWindStateAddrByRankId(const int32_t rankId)
{
    return GetBaseWindStateAddrByRankId(winContext_, rankId, epRankId_) + WIN_STATE_OFFSET;
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::SplitToCore(
    uint32_t totalCnt, uint32_t useAivNum, 
    uint32_t &startIdx, uint32_t &endIdx, uint32_t &procNum, bool isFront)
{
    procNum = totalCnt / useAivNum;
    uint32_t remainderNum = totalCnt % useAivNum;
    uint32_t newAivId;
    
    if (isFront) {
        newAivId = aivId_;
    } else {
        newAivId = aivId_ - aivUsedAllToAll_;
    }
    
    startIdx = procNum * newAivId;
    if (newAivId < remainderNum) {
        procNum += 1;
        startIdx += newAivId;
    } else {
        startIdx += remainderNum;
    }
    endIdx = startIdx + procNum;
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::Init(
    GM_ADDR x, GM_ADDR expertId, GM_ADDR y, GM_ADDR receiveCnt,
    GM_ADDR workspaceGM, TPipe *pipe, const BandwidthTestTilingData *tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    
    axisBS_ = tilingData->bs;
    dataSize_ = tilingData->data_size;
    epWorldSize_ = tilingData->world_size;
    epRankId_ = tilingData->rank_id;
    axisMaxBS_ = tilingData->max_bs;
    mode_ = tilingData->mode;
    aivNum_ = tilingData->aiv_num;

    aivUsedCumSum_ = epWorldSize_ / 32;
    aivUsedCumSum_ = (aivUsedCumSum_ == 0) ? 1 : aivUsedCumSum_;
    aivUsedCumSum_ = (aivUsedCumSum_ >= (aivNum_ / 2)) ? (aivNum_ / 2) : aivUsedCumSum_;
    aivUsedCumSum_ = (aivUsedCumSum_ >= CUMSUM_MAX_CORE_NUM) ? CUMSUM_MAX_CORE_NUM : aivUsedCumSum_;
    aivUsedCumSum_ = (aivUsedCumSum_ >= epWorldSize_) ? epWorldSize_ : aivUsedCumSum_;
    aivUsedAllToAll_ = aivNum_ - aivUsedCumSum_;

    winContext_ = (__gm__ HcclOpParam *)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    
    winDataSizeOffset_ = 0;
    windowGM_ = GetWindAddrByRankId(epRankId_);
    statusSpaceGM_ = GetWindStateAddrByRankId(epRankId_);
    
    selfRankWinInGMTensor_.SetGlobalBuffer((__gm__ float*)(statusSpaceGM_));
    windowInStatusFp32Tensor_.SetGlobalBuffer((__gm__ float*)(statusSpaceGM_));

    xGMTensor_.SetGlobalBuffer((__gm__ XType*)x);
    expertIdGMTensor_.SetGlobalBuffer((__gm__ int32_t*)expertId);
    yGMTensor_.SetGlobalBuffer((__gm__ YType*)y);
    receiveCntGMTensor_.SetGlobalBuffer((__gm__ int32_t*)receiveCnt);

    yGM_ = y;
    receiveCntGM_ = receiveCnt;
    recvCntWorkspaceGM_ = workspaceGM;

    uint64_t hCommuSizeAlign = Ceil(dataSize_ * sizeof(YType), WIN_ADDR_ALIGN) * WIN_ADDR_ALIGN;
    expertPerSizeOnWin_ = static_cast<uint64_t>(axisMaxBS_) * hCommuSizeAlign;
    hCommuSize_ = hCommuSizeAlign;

    expertIdsSize_ = Ceil(axisBS_ * sizeof(int32_t), SIZE_ALIGN_256) * SIZE_ALIGN_256;
    uint32_t statusBufSize = Ceil(epWorldSize_ * UB_ALIGN, UB_ALIGN);
    uint32_t workLocalBufSize = Ceil(epWorldSize_ * sizeof(float), UB_ALIGN) * UB_ALIGN;
    
    tileSize_ = (MAX_TILE_SIZE < axisMaxBS_) ? MAX_TILE_SIZE : axisMaxBS_;
    uint32_t ySize = tileSize_ * dataSize_ * sizeof(YType);
    statusCntAlign_ = Ceil(epWorldSize_, UB_ALIGN_DATA_COUNT) * UB_ALIGN_DATA_COUNT;

    maskSizePerExpert_ = Ceil((expertIdsSize_ / sizeof(int32_t)) / BITS_PER_BYTE, UB_ALIGN) * UB_ALIGN;
    uint32_t expertMaskBufSize = maskSizePerExpert_ * Ceil(epWorldSize_, aivUsedAllToAll_);
    uint32_t tokenNumToExpertBufSize = Ceil(epWorldSize_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t gatherIndexBufSize = Ceil(axisBS_ * sizeof(int32_t), UB_ALIGN) * UB_ALIGN;
    uint32_t xBufSize = Ceil(tileSize_ * dataSize_ * sizeof(XType), UB_ALIGN) * UB_ALIGN;

    tpipe_->InitBuffer(yQueue_, BUFFER_NUM, ySize);
    tpipe_->InitBuffer(statusBuf_, statusBufSize);
    tpipe_->InitBuffer(statusFp32Buf_, statusBufSize);
    tpipe_->InitBuffer(statusSumBuf_, UB_ALIGN);
    tpipe_->InitBuffer(cleanStatusBuf_, UB_ALIGN);
    tpipe_->InitBuffer(sendCntBuf_, statusBufSize);
    tpipe_->InitBuffer(recvCntBuf_, statusBufSize);
    tpipe_->InitBuffer(expertIdBuf_, expertIdsSize_);
    tpipe_->InitBuffer(workLocalBuf_, workLocalBufSize);
    tpipe_->InitBuffer(expertMaskBuf_, expertMaskBufSize);
    tpipe_->InitBuffer(tokenNumToExpertBuf_, tokenNumToExpertBufSize);
    tpipe_->InitBuffer(gatherIndexBuf_, gatherIndexBufSize);
    tpipe_->InitBuffer(xBuf_, xBufSize);
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::TokenToExpert(
    GlobalTensor<YType> &dstWinGMTensor, uint32_t tokenIndex, uint32_t dstRank)
{
    xLocalTensor_ = xBuf_.Get<XType>();
    yLocalTensor_ = yQueue_.AllocTensor<YType>();
    
    DataCopyExtParams xCopyParams{1U, static_cast<uint32_t>(dataSize_ * sizeof(XType)), 0U, 0U, 0U};
    DataCopyPadExtParams<XType> xPadParams{false, 0U, 0U, 0U};
    DataCopyPad(xLocalTensor_, xGMTensor_[tokenIndex * dataSize_], xCopyParams, xPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();
    
    Cast(yLocalTensor_, xLocalTensor_, RoundMode::CAST_NONE, dataSize_);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    
    uint32_t offset = sendCntTensor_(dstRank);
    DataCopy(dstWinGMTensor[offset * dataSize_], yLocalTensor_, dataSize_);
    DataCacheCleanAndInvalid<YType, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(dstWinGMTensor[offset * dataSize_]);
    
    sendCntTensor_(dstRank) = offset + 1;
    
    yQueue_.FreeTensor(yLocalTensor_);
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::CalExpertSendNum()
{
    expertIdLocalTensor_ = expertIdBuf_.Get<int32_t>();
    expertMaskTensor_ = expertMaskBuf_.Get<uint8_t>();
    gatherIndexTensor_ = gatherIndexBuf_.Get<int32_t>();
    tokenNumToExpertTensor_ = tokenNumToExpertBuf_.Get<int32_t>();
    sendCntTensor_ = sendCntBuf_.Get<int32_t>();
    
    Duplicate<int32_t>(sendCntTensor_, 0, epWorldSize_);
    
    uint32_t compareCount = Ceil(axisBS_ * sizeof(int32_t), SIZE_ALIGN_256) * SIZE_ALIGN_256 / sizeof(int32_t);
    
    for (uint32_t rankIndex = startId_; rankIndex < endId_; rankIndex++) {
        uint32_t expertIndex = rankIndex - startId_;
        
        CompareScalar(expertMaskTensor_[maskSizePerExpert_ * expertIndex], 
                      expertIdLocalTensor_, 
                      static_cast<int32_t>(rankIndex), 
                      CMPMODE::EQ, compareCount);
        PipeBarrier<PIPE_V>();
        
        uint64_t maskCnt = 0;
        GatherMask(gatherIndexTensor_, expertIdLocalTensor_, 
                   expertMaskTensor_[maskSizePerExpert_ * expertIndex].ReinterpretCast<uint32_t>(),
                   true, axisBS_, {1, 1, 0, 0}, maskCnt);
        SyncFunc<AscendC::HardEvent::V_S>();
        
        tokenNumToExpertTensor_(expertIndex) = static_cast<uint32_t>(maskCnt);
    }
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::SendToExpert()
{
    GlobalTensor<YType> dstWinGMTensor;
    gatherIndexTensor_ = gatherIndexBuf_.Get<int32_t>();
    tokenNumToExpertTensor_ = tokenNumToExpertBuf_.Get<int32_t>();
    sendCntTensor_ = sendCntBuf_.Get<int32_t>();
    
    uint32_t maskN64Num = Ceil(axisBS_, 64);
    LocalTensor<uint64_t> expertMaskTensorU64 = expertMaskBuf_.Get<uint64_t>();
    
    for (uint32_t expertIndex = 0; expertIndex < sendNum_; expertIndex++) {
        uint32_t dstRank = startId_ + expertIndex;
        
        if (tokenNumToExpertTensor_(expertIndex) == 0) {
            continue;
        }
        
        uint32_t maskExpertU64Cnt = maskSizePerExpert_ * expertIndex / sizeof(uint64_t);
        uint64_t dstWinOffset = epRankId_ * expertPerSizeOnWin_;
        GM_ADDR dstWinAddr = GetWindAddrByRankId(dstRank) + dstWinOffset;
        dstWinGMTensor.SetGlobalBuffer((__gm__ YType*)dstWinAddr);
        
        for (uint32_t maskIndex = 0; maskIndex < maskN64Num; maskIndex++) {
            uint64_t dstExpInfoMask = expertMaskTensorU64(maskIndex + maskExpertU64Cnt);
            int64_t curValidIdx = ScalarGetSFFValue<1>(dstExpInfoMask);
            
            while (curValidIdx >= 0) {
                uint32_t tokenIdx = curValidIdx + 64 * maskIndex;
                if (tokenIdx >= axisBS_) {
                    break;
                }
                TokenToExpert(dstWinGMTensor, tokenIdx, dstRank);
                curValidIdx = ScalarGetSFFValue<1>(dstExpInfoMask);
            }
        }
    }
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::AllToAllDispatch()
{
    expertIdLocalTensor_ = expertIdBuf_.Get<int32_t>();
    sendCntTensor_ = sendCntBuf_.Get<int32_t>();

    uint32_t expertIdsAlignCnt = Ceil(axisBS_, BITS_PER_BYTE) * BITS_PER_BYTE;
    uint32_t rightPadding = expertIdsAlignCnt - axisBS_;
    DataCopyPadExtParams<int32_t> expertIdCopyPadParams{true, 0U, uint8_t(rightPadding), -1};
    DataCopyExtParams expertIdParams{1U, static_cast<uint32_t>(axisBS_ * sizeof(int32_t)), 0U, 0U, 0U};
    SyncFunc<AscendC::HardEvent::V_MTE2>();
    DataCopyPad(expertIdLocalTensor_, expertIdGMTensor_, expertIdParams, expertIdCopyPadParams);
    SyncFunc<AscendC::HardEvent::MTE2_S>();

    SplitToCore(epWorldSize_, aivUsedAllToAll_, startId_, endId_, sendNum_, true);
    
    if (startId_ >= epWorldSize_) {
        return;
    }

    expertMaskTensor_ = expertMaskBuf_.Get<uint8_t>();
    LocalTensor<int32_t> expertMaskTensorInt32 = expertMaskBuf_.Get<int32_t>();
    Duplicate<int32_t>(expertMaskTensorInt32, 0, Ceil(epWorldSize_, aivUsedAllToAll_) * maskSizePerExpert_ / sizeof(int32_t));
    SyncFunc<AscendC::HardEvent::V_S>();

    CalExpertSendNum();

    commStartTime_ = AscendC::GetSystemCycle() / TIME_CYCLE;

    SendToExpert();
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::CalAndSendCnt()
{
    SplitToCore(epWorldSize_, aivUsedCumSum_, startStatusIndex_, endStatusIndex_, recStatusNumPerCore_, false);
    
    if (startStatusIndex_ >= epWorldSize_) {
        return;
    }

    statusTensor_ = statusBuf_.Get<int32_t>();
    sendCntTensor_ = sendCntBuf_.Get<int32_t>();
    
    Duplicate<int32_t>(statusTensor_, 0, statusCntAlign_ * UB_ALIGN_DATA_COUNT);
    PipeBarrier<PIPE_V>();
    
    uint64_t mask[2] = {0x101010101010101, 0};
    Duplicate<int32_t>(statusTensor_, 0x3F800000, mask, statusCntAlign_ / 8, 1, 8);
    PipeBarrier<PIPE_ALL>();

    GlobalTensor<int32_t> rankGMTensor;
    for (uint32_t curRank = startStatusIndex_; curRank < endStatusIndex_; ++curRank) {
        int32_t curExpertCnt = sendCntTensor_(curRank);
        int32_t cntPosIndex = (curRank - startStatusIndex_) * 8 + 1;
        statusTensor_(cntPosIndex) = curExpertCnt;
        SyncFunc<AscendC::HardEvent::S_MTE3>();
        
        uint32_t dstRankId = curRank;
        uint32_t offset = STATE_OFFSET * epRankId_;
        
        GM_ADDR rankGM = (__gm__ uint8_t*)(GetWindStateAddrByRankId(dstRankId) + offset);
        rankGMTensor.SetGlobalBuffer((__gm__ int32_t*)rankGM);
        DataCopy<int32_t>(rankGMTensor, statusTensor_[(curRank - startStatusIndex_) * 8], 8);
    }
    PipeBarrier<PIPE_ALL>();
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::WaitDispatch()
{
    statusFp32Tensor_ = statusFp32Buf_.Get<float>();
    statusSumOutTensor_ = statusSumBuf_.Get<float>();
    
    float compareTarget = static_cast<float>(1.0) * epWorldSize_;
    float sumOfFlag = static_cast<float>(-1.0);
    DataCopyParams intriParams{static_cast<uint16_t>(epWorldSize_), 1, 0, 0};
    
    while (sumOfFlag != compareTarget) {
        DataCopy(statusFp32Tensor_, windowInStatusFp32Tensor_, intriParams);
        SyncFunc<AscendC::HardEvent::MTE2_V>();
        ReduceSum(statusSumOutTensor_, statusFp32Tensor_, cleanStatusTensor_.ReinterpretCast<float>(), 1, epWorldSize_, 1);
        SyncFunc<AscendC::HardEvent::V_S>();
        sumOfFlag = statusSumOutTensor_(0);
    }

    SyncFunc<AscendC::HardEvent::MTE3_S>();
    cleanStatusTensor_ = cleanStatusBuf_.Get<int32_t>();
    DataCopyParams intriOutParams{static_cast<uint16_t>(epWorldSize_), 1, 0, 0};
    uint64_t duplicateMask[2] = {0x101010101010101, 0};
    SyncFunc<AscendC::HardEvent::S_V>();
    Duplicate<int32_t>(cleanStatusTensor_, 0, duplicateMask, Ceil(epWorldSize_, 8), 1, 8);
    SyncFunc<AscendC::HardEvent::V_MTE3>();
    DataCopy(windowInStatusFp32Tensor_, cleanStatusTensor_.ReinterpretCast<float>(), intriOutParams);
    SyncFunc<AscendC::HardEvent::MTE3_S>();
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::LocalWindowCopy()
{
    SplitToCore(epWorldSize_, aivNum_, startId_, endId_, sendNum_, true);
    
    if (sendNum_ == 0) {
        return;
    }

    recvCntTensor_ = recvCntBuf_.Get<int32_t>();
    LocalTensor<float> statusFp32Local = statusFp32Buf_.Get<float>();
    
    DataCopyParams intriParams{static_cast<uint16_t>(epWorldSize_), 1, 0, 0};
    DataCopy(statusFp32Local, windowInStatusFp32Tensor_, intriParams);
    SyncFunc<AscendC::HardEvent::MTE2_V>();

    LocalTensor<int32_t> statusInt32Local = statusFp32Local.ReinterpretCast<int32_t>();
    for (uint32_t srcRank = 0; srcRank < epWorldSize_; srcRank++) {
        recvCntTensor_(srcRank) = statusInt32Local(srcRank * 8 + 1);
    }
    SyncFunc<AscendC::HardEvent::V_S>();

    yLocalTensor_ = yQueue_.AllocTensor<YType>();
    
    for (uint32_t srcRank = startId_; srcRank < endId_; srcRank++) {
        if (srcRank == static_cast<uint32_t>(epRankId_)) {
            continue;
        }
        
        int32_t recvCount = recvCntTensor_(srcRank);
        if (recvCount > 0) {
            uint64_t srcOffset = srcRank * expertPerSizeOnWin_;
            GM_ADDR srcWinAddr = windowGM_ + srcOffset;
            GlobalTensor<YType> srcWinTensor;
            srcWinTensor.SetGlobalBuffer((__gm__ YType*)srcWinAddr);
            
            DataCacheCleanAndInvalid<YType, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(srcWinTensor[0]);
            
            uint32_t remainCount = static_cast<uint32_t>(recvCount);
            uint32_t curOffset = 0;
            
            while (remainCount > 0) {
                uint32_t curCount = (remainCount > tileSize_) ? tileSize_ : remainCount;
                
                DataCopy(yLocalTensor_, srcWinTensor[curOffset * dataSize_], curCount * dataSize_);
                SyncFunc<AscendC::HardEvent::MTE2_MTE3>();
                
                uint64_t dstOffset = static_cast<uint64_t>(srcRank) * axisMaxBS_ * dataSize_ + curOffset * dataSize_;
                DataCopyExtParams copyParams = {1, static_cast<uint32_t>(curCount * dataSize_ * sizeof(YType)), 0, 0, 0};
                DataCopyPad(yGMTensor_[dstOffset], yLocalTensor_, copyParams);
                
                curOffset += curCount;
                remainCount -= curCount;
            }
        }
    }
    
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    
    for (uint32_t rankIndex = 1; rankIndex < epWorldSize_; rankIndex++) {
        recvCntTensor_(rankIndex) += recvCntTensor_(rankIndex - 1);
    }
    SyncFunc<AscendC::HardEvent::S_MTE3>();
    
    DataCopy(receiveCntGMTensor_, recvCntTensor_, epWorldSize_);
    yQueue_.FreeTensor(yLocalTensor_);
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        if (aivId_ < aivUsedAllToAll_) {
            AllToAllDispatch();
        } else if (mode_ == 0) {
            sendCntTensor_ = sendCntBuf_.Get<int32_t>();
            Duplicate<int32_t>(sendCntTensor_, 0, epWorldSize_);
            PipeBarrier<PIPE_V>();
            CalAndSendCnt();
            WaitDispatch();
        }
        
        if (mode_ == 0) {
            PipeBarrier<PIPE_ALL>();
            LocalWindowCopy();
        }
        
        commEndTime_ = AscendC::GetSystemCycle() / TIME_CYCLE;
    }
}

} // namespace BandwidthTestImpl
#endif // BANDWIDTH_TEST_H
