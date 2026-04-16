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
 * \brief Bandwidth test kernel - measures real communication bandwidth
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
constexpr uint32_t STATE_ALIGN = 64;
constexpr uint32_t STATE_OFFSET = 512;
constexpr uint32_t TIME_CYCLE = 50;
constexpr uint32_t MAX_TILE_ROWS = 4095;

#define TemplateBandwidthTestTypeClass \
    typename XType, \
    typename YType
#define TemplateBandwidthTestTypeFunc XType, YType

using namespace AscendC;

template <TemplateBandwidthTestTypeClass>
class BandwidthTest {
public:
    __aicore__ inline BandwidthTest() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR expertId, GM_ADDR y, GM_ADDR receiveCnt,
        GM_ADDR workspaceGM, TPipe *pipe, const BandwidthTestTilingData *tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void ComputeAndSend();
    __aicore__ inline void ReceiveAndWriteBack();
    __aicore__ inline GM_ADDR GetWindAddrByRankId(const int32_t rankId);
    __aicore__ inline GM_ADDR GetWindStateAddrByRankId(const int32_t rankId);
    __aicore__ inline void SendDataToRank(uint32_t dstRank, uint32_t offset, uint32_t count);
    __aicore__ inline void SetStatus(uint32_t dstRank, uint32_t count);
    __aicore__ inline uint32_t WaitStatus(uint32_t srcRank);
    __aicore__ inline int64_t GetCurrentTimestampUs();

    TPipe *tpipe_{nullptr};
    GlobalTensor<XType> xGMTensor_;
    GlobalTensor<int32_t> expertIdGMTensor_;
    GlobalTensor<YType> yGMTensor_;
    GlobalTensor<int32_t> receiveCntGMTensor_;

    LocalTensor<XType> xLocalTensor_;
    LocalTensor<int32_t> expertIdLocalTensor_;
    LocalTensor<YType> yLocalTensor_;
    LocalTensor<int32_t> receiveCntLocalTensor_;
    LocalTensor<int32_t> sendCntLocalTensor_;
    
    TBuf<> sendCntBuf_;
    TQue<QuePosition::VECIN, BUFFER_NUM> xQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> expertIdQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> receiveCntQueue_;

    GM_ADDR yGM_;
    GM_ADDR receiveCntGM_;
    GM_ADDR windowGM_;
    GM_ADDR statusSpaceGM_;

    __gm__ Mc2Kernel::HcclOpParam *winContext_{nullptr};

    uint32_t aivId_{0};
    uint32_t bs_{0};
    uint32_t dataSize_{0};
    uint32_t worldSize_{0};
    uint32_t rankId_{0};
    uint32_t maxBs_{0};
    uint32_t moeNum_{0};
    uint64_t expertPerSizeOnWin_{0};
    int64_t commStartTime_{0};
    int64_t commEndTime_{0};
    uint32_t tileSize_{0};
};

template <TemplateBandwidthTestTypeClass>
__aicore__ inline int64_t BandwidthTest<TemplateBandwidthTestTypeFunc>::GetCurrentTimestampUs()
{
    return AscendC::GetSystemCycle() / TIME_CYCLE;
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline GM_ADDR BandwidthTest<TemplateBandwidthTestTypeFunc>::GetWindAddrByRankId(const int32_t rankId)
{
    return Mc2Kernel::GetBaseWindAddrByRankId(winContext_, rankId, rankId_);
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline GM_ADDR BandwidthTest<TemplateBandwidthTestTypeFunc>::GetWindStateAddrByRankId(const int32_t rankId)
{
    return Mc2Kernel::GetBaseWindStateAddrByRankId(winContext_, rankId, rankId_);
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::Init(
    GM_ADDR x, GM_ADDR expertId, GM_ADDR y, GM_ADDR receiveCnt,
    GM_ADDR workspaceGM, TPipe *pipe, const BandwidthTestTilingData *tilingData)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();
    
    bs_ = tilingData->bs;
    dataSize_ = tilingData->data_size;
    worldSize_ = tilingData->world_size;
    rankId_ = tilingData->rank_id;
    maxBs_ = tilingData->max_bs;
    moeNum_ = tilingData->moe_num;

    winContext_ = (__gm__ Mc2Kernel::HcclOpParam *)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    windowGM_ = GetWindAddrByRankId(rankId_);
    statusSpaceGM_ = GetWindStateAddrByRankId(rankId_);

    xGMTensor_.SetGlobalBuffer((__gm__ XType*)x);
    expertIdGMTensor_.SetGlobalBuffer((__gm__ int32_t*)expertId);
    yGMTensor_.SetGlobalBuffer((__gm__ YType*)y);
    receiveCntGMTensor_.SetGlobalBuffer((__gm__ int32_t*)receiveCnt);

    yGM_ = y;
    receiveCntGM_ = receiveCnt;

    expertPerSizeOnWin_ = static_cast<uint64_t>(maxBs_) * dataSize_ * sizeof(YType);

    uint32_t xSize = bs_ * dataSize_ * sizeof(XType);
    uint32_t expertIdSize = bs_ * sizeof(int32_t);
    uint32_t receiveCntSize = worldSize_ * sizeof(int32_t);
    
    tileSize_ = (MAX_TILE_ROWS < maxBs_) ? MAX_TILE_ROWS : maxBs_;
    uint32_t ySize = tileSize_ * dataSize_ * sizeof(YType);

    tpipe_->InitBuffer(xQueue_, BUFFER_NUM, xSize);
    tpipe_->InitBuffer(expertIdQueue_, BUFFER_NUM, expertIdSize);
    tpipe_->InitBuffer(yQueue_, BUFFER_NUM, ySize);
    tpipe_->InitBuffer(receiveCntQueue_, BUFFER_NUM, receiveCntSize);
    tpipe_->InitBuffer(sendCntBuf_, worldSize_ * sizeof(int32_t));
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::CopyIn()
{
    xLocalTensor_ = xQueue_.AllocTensor<XType>();
    expertIdLocalTensor_ = expertIdQueue_.AllocTensor<int32_t>();

    DataCopyExtParams xCopyParams = {1, static_cast<uint32_t>(bs_ * dataSize_ * sizeof(XType)), 0, 0, 0};
    DataCopyPadExtParams<XType> xPadParams{false, 0, 0, 0};
    DataCopyPad(xLocalTensor_, xGMTensor_[0], xCopyParams, xPadParams);

    DataCopyExtParams expertIdCopyParams = {1, static_cast<uint32_t>(bs_ * sizeof(int32_t)), 0, 0, 0};
    DataCopyPadExtParams<int32_t> expertIdPadParams{false, 0, 0, 0};
    DataCopyPad(expertIdLocalTensor_, expertIdGMTensor_[0], expertIdCopyParams, expertIdPadParams);

    xQueue_.EnQue(xLocalTensor_);
    expertIdQueue_.EnQue(expertIdLocalTensor_);
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::SendDataToRank(
    uint32_t dstRank, uint32_t offset, uint32_t count)
{
    if (count == 0) {
        return;
    }

    uint64_t dstOffset = rankId_ * expertPerSizeOnWin_ + offset * dataSize_ * sizeof(YType);
    GM_ADDR dstWinAddr = GetWindAddrByRankId(dstRank) + dstOffset;
    GlobalTensor<YType> dstWinTensor;
    dstWinTensor.SetGlobalBuffer((__gm__ YType*)dstWinAddr);

    uint32_t remainCount = count;
    uint32_t curOffset = 0;
    
    while (remainCount > 0) {
        uint32_t curCount = (remainCount > tileSize_) ? tileSize_ : remainCount;
        uint32_t srcOffset = offset + curOffset;
        
        yLocalTensor_ = yQueue_.AllocTensor<YType>();
        DataCopyExtParams copyParams = {1, static_cast<uint32_t>(curCount * dataSize_ * sizeof(YType)), 0, 0, 0};
        DataCopyPadExtParams<YType> padParams{false, 0, 0, 0};
        
        uint64_t gmOffset = static_cast<uint64_t>(dstRank) * maxBs_ * dataSize_ + srcOffset * dataSize_;
        DataCopyPad(yLocalTensor_, yGMTensor_[gmOffset], copyParams, padParams);
        
        DataCopy(dstWinTensor[curOffset * dataSize_], yLocalTensor_, curCount * dataSize_);
        DataCacheCleanAndInvalid<YType, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(dstWinTensor[curOffset * dataSize_]);
        
        yQueue_.FreeTensor(yLocalTensor_);
        
        curOffset += curCount;
        remainCount -= curCount;
    }
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::SetStatus(
    uint32_t dstRank, uint32_t count)
{
    GM_ADDR dstStatusAddr = GetWindStateAddrByRankId(dstRank) + rankId_ * STATE_OFFSET;
    GlobalTensor<int32_t> statusTensor;
    statusTensor.SetGlobalBuffer((__gm__ int32_t*)dstStatusAddr);
    
    statusTensor.SetValue(0, 1);
    statusTensor.SetValue(1, static_cast<int32_t>(count));
    
    DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(statusTensor[0]);
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline uint32_t BandwidthTest<TemplateBandwidthTestTypeFunc>::WaitStatus(uint32_t srcRank)
{
    GM_ADDR srcStatusAddr = GetWindStateAddrByRankId(srcRank) + srcRank * STATE_OFFSET;
    GlobalTensor<int32_t> statusTensor;
    statusTensor.SetGlobalBuffer((__gm__ int32_t*)srcStatusAddr);

    int32_t flag = 0;
    int32_t count = 0;
    uint32_t timeout = 1000000;
    uint32_t waitCnt = 0;

    while (flag != 1 && waitCnt < timeout) {
        DataCacheCleanAndInvalid<int32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(statusTensor[0]);
        flag = statusTensor.GetValue(0);
        waitCnt++;
    }

    if (flag == 1) {
        count = statusTensor.GetValue(1);
    }

    return static_cast<uint32_t>(count);
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::ComputeAndSend()
{
    xLocalTensor_ = xQueue_.DeQue<XType>();
    expertIdLocalTensor_ = expertIdQueue_.DeQue<int32_t>();
    yLocalTensor_ = yQueue_.AllocTensor<YType>();
    receiveCntLocalTensor_ = receiveCntQueue_.AllocTensor<int32_t>();
    sendCntLocalTensor_ = sendCntBuf_.Get<int32_t>();

    Duplicate<int32_t>(receiveCntLocalTensor_, 0, worldSize_);
    Duplicate<int32_t>(sendCntLocalTensor_, 0, worldSize_);
    Duplicate<YType>(yLocalTensor_, 0, tileSize_ * dataSize_);

    PipeBarrier<PIPE_V>();

    for (uint32_t i = 0; i < bs_; i++) {
        int32_t dstRank = expertIdLocalTensor_.GetValue(i) % static_cast<int32_t>(worldSize_);
        int32_t offset = sendCntLocalTensor_.GetValue(dstRank);
        
        for (uint32_t j = 0; j < dataSize_; j++) {
            YType val = static_cast<YType>(xLocalTensor_.GetValue(i * dataSize_ + j));
            yLocalTensor_.SetValue(offset * dataSize_ + j, val);
        }
        sendCntLocalTensor_.SetValue(dstRank, offset + 1);
    }

    uint64_t yGMBase = 0;
    for (uint32_t rankIdx = 0; rankIdx < worldSize_; rankIdx++) {
        int32_t count = sendCntLocalTensor_.GetValue(rankIdx);
        if (count > 0) {
            uint32_t remainCount = static_cast<uint32_t>(count);
            uint32_t curOffset = 0;
            
            while (remainCount > 0) {
                uint32_t curCount = (remainCount > tileSize_) ? tileSize_ : remainCount;
                uint64_t dstOffset = static_cast<uint64_t>(rankIdx) * maxBs_ * dataSize_ + curOffset * dataSize_;
                
                DataCopyExtParams copyParams = {1, static_cast<uint32_t>(curCount * dataSize_ * sizeof(YType)), 0, 0, 0};
                DataCopyPad(yGMTensor_[dstOffset], yLocalTensor_, copyParams);
                
                curOffset += curCount;
                remainCount -= curCount;
            }
        }
        receiveCntLocalTensor_.SetValue(rankIdx, count);
    }

    commStartTime_ = GetCurrentTimestampUs();

    for (uint32_t rankIndex = 0; rankIndex < worldSize_; rankIndex++) {
        int32_t sendCount = sendCntLocalTensor_.GetValue(rankIndex);
        if (sendCount > 0) {
            SendDataToRank(rankIndex, 0, static_cast<uint32_t>(sendCount));
            SetStatus(rankIndex, static_cast<uint32_t>(sendCount));
        }
    }

    xQueue_.FreeTensor(xLocalTensor_);
    expertIdQueue_.FreeTensor(expertIdLocalTensor_);
    yQueue_.FreeTensor(yLocalTensor_);
    receiveCntQueue_.EnQue<int32_t>(receiveCntLocalTensor_);
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::ReceiveAndWriteBack()
{
    receiveCntLocalTensor_ = receiveCntQueue_.DeQue<int32_t>();
    yLocalTensor_ = yQueue_.AllocTensor<YType>();

    for (uint32_t rankIndex = 0; rankIndex < worldSize_; rankIndex++) {
        if (rankIndex != rankId_) {
            uint32_t recvCount = WaitStatus(rankIndex);
            receiveCntLocalTensor_.SetValue(rankIndex, static_cast<int32_t>(recvCount));
            
            if (recvCount > 0) {
                uint64_t srcOffset = rankIndex * expertPerSizeOnWin_;
                GM_ADDR srcWinAddr = windowGM_ + srcOffset;
                GlobalTensor<YType> srcWinTensor;
                srcWinTensor.SetGlobalBuffer((__gm__ YType*)srcWinAddr);
                
                DataCacheCleanAndInvalid<YType, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(srcWinTensor[0]);
                
                uint32_t remainCount = recvCount;
                uint32_t curOffset = 0;
                
                while (remainCount > 0) {
                    uint32_t curCount = (remainCount > tileSize_) ? tileSize_ : remainCount;
                    
                    DataCopy(yLocalTensor_, srcWinTensor[curOffset * dataSize_], curCount * dataSize_);
                    
                    uint64_t dstOffset = static_cast<uint64_t>(rankIndex) * maxBs_ * dataSize_ + curOffset * dataSize_;
                    DataCopyExtParams copyParams = {1, static_cast<uint32_t>(curCount * dataSize_ * sizeof(YType)), 0, 0, 0};
                    DataCopyPad(yGMTensor_[dstOffset], yLocalTensor_, copyParams);
                    
                    curOffset += curCount;
                    remainCount -= curCount;
                }
            }
        }
    }

    commEndTime_ = GetCurrentTimestampUs();

    DataCopy(receiveCntGMTensor_, receiveCntLocalTensor_, worldSize_);

    yQueue_.FreeTensor(yLocalTensor_);
    receiveCntQueue_.FreeTensor(receiveCntLocalTensor_);
}

template <TemplateBandwidthTestTypeClass>
__aicore__ inline void BandwidthTest<TemplateBandwidthTestTypeFunc>::Process()
{
    if ASCEND_IS_AIV {
        if (aivId_ != 0) {
            return;
        }
        
        CopyIn();
        ComputeAndSend();
        ReceiveAndWriteBack();
    }
}

} // namespace BandwidthTestImpl
#endif // BANDWIDTH_TEST_H
