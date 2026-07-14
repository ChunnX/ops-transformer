/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file collective_comm_base.h
 * \brief 基于 hcomm 的集合通信基类 - CRTP 模式 + 统一状态管理
 */

#pragma once

#include "kernel_operator.h"
#include "collective_comm_context.h"
#include "../utils/comm_tiling_data.h"
#include "tensor_api/tensor/pointer.h"

namespace Atcoss {
namespace AivComm {

using namespace AscendC;
using Utils::CommTilingData;

template<typename Impl, typename Dtype>
class CollectiveCommBase {
    friend Impl;
protected:
    __gm__ CommUdmaContext* udmaCtx_;
    __gm__ CommUbmemContext* ubmemCtx_;
    const CommTilingData* tilingData_;

    Hcomm<AscendC::COMM_PROTOCOL_UBC_CTP> comm_;
    GM_ADDR localAddr_;
    __ubuf__ uint8_t* commBuf_;
    __ubuf__ uint8_t* syncBuf_;

    ChannelHandle channels_[COMM_MAX_RANK_NUM];
    GM_ADDR commBufferAddrs_[COMM_MAX_RANK_NUM];

    uint64_t chunkSize_;
    uint64_t nonSplitAxisBytes_;
    uint64_t chunkBytes_;

    uint64_t totalTiles_;
    uint64_t slotStride_;
    uint64_t myRankOffset_;
    uint64_t currentTileIdx_;
    uint64_t currentTileOffset_;
    uint64_t remainingChunkSize_;

    uint32_t targetRankStart_;
    uint32_t targetRankCnt_;

    uint32_t useCoreNum_;
    uint32_t curCoreId_;

public:
    __aicore__ inline CollectiveCommBase() {}

    __aicore__ inline uint64_t Init(
        __gm__ CommUdmaContext* udmaCtx,
        __gm__ CommUbmemContext* ubmemCtx,
        const CommTilingData& tilingData,
        GM_ADDR localAddr,
        __ubuf__ uint8_t* commbuf,
        uint32_t useCoreNum,
        uint32_t curCoreId)
    {
        udmaCtx_ = udmaCtx;
        ubmemCtx_ = ubmemCtx;
        localAddr_ = localAddr;
        tilingData_ = &tilingData;
        useCoreNum_ = useCoreNum;
        curCoreId_ = curCoreId;
        commBuf_ = commbuf;

        for (uint32_t i = 0; i < udmaCtx_->rankSize; i++) {
            channels_[i] = static_cast<ChannelHandle>(udmaCtx_->channelHandles[i]);
            commBufferAddrs_[i] = reinterpret_cast<GM_ADDR>(udmaCtx_->commBufferAddrs[i]);
        }

        comm_.Init(commBuf_, COMM_WORKSPACE_SIZE);
        syncBuf_ = commBuf_ + COMM_WORKSPACE_SIZE;

        uint32_t rankSize = udmaCtx_->rankSize;
        uint32_t targetRankPerCore = (rankSize + useCoreNum - 1) / useCoreNum;
        uint32_t targetRankStart = curCoreId * targetRankPerCore;

        uint32_t targetRankCnt;
        if (targetRankStart + targetRankPerCore <= rankSize) {
            targetRankCnt = targetRankPerCore;
        } else if (targetRankStart < rankSize) {
            targetRankCnt = rankSize - targetRankStart;
        } else {
            targetRankCnt = 0;
        }

        targetRankStart_ = targetRankStart;
        targetRankCnt_ = targetRankCnt;

        chunkSize_ = tilingData.splitAxisTileSize * tilingData.splitAxisTileCnt +
                     tilingData.splitAxisTailSize * tilingData.splitAxisTailCnt;
        nonSplitAxisBytes_ = tilingData.nonSplitAxisSize * sizeof(Dtype);
        chunkBytes_ = chunkSize_ * nonSplitAxisBytes_;
        totalTiles_ = tilingData_->splitAxisTileCnt + tilingData_->splitAxisTailCnt;
        uint32_t tileMaxByteSize = (tilingData_->splitAxisTileSize > tilingData_->splitAxisTailSize ?
            tilingData_->splitAxisTileSize : tilingData_->splitAxisTailSize) * nonSplitAxisBytes_;
        slotStride_ = rankSize * tileMaxByteSize;
        myRankOffset_ = udmaCtx_->rankId * tileMaxByteSize;
        currentTileIdx_ = 0;
        currentTileOffset_ = 0;
        remainingChunkSize_ = chunkSize_;

        static_cast<Impl*>(this)->PostInit();
        return totalTiles_;
    }

    __aicore__ inline void Commit()
    {
        if (remainingChunkSize_ <= 0) {
            return;
        }

        uint64_t currentTileSize;
        if (currentTileIdx_ < tilingData_->splitAxisTileCnt) {
            currentTileSize = tilingData_->splitAxisTileSize;
        } else {
            currentTileSize = tilingData_->splitAxisTailSize;
        }
        if (currentTileSize > remainingChunkSize_) {
            currentTileSize = remainingChunkSize_;
        }

        if (targetRankCnt_ > 0) {
            const uint32_t rankId = udmaCtx_->rankId;
            const uint32_t targetRankStart = targetRankStart_;
            const uint32_t targetRankCnt = targetRankCnt_;

            for (uint32_t i = 0; i < targetRankCnt; i++) {
                uint32_t targetRankId = targetRankStart + i;
                static_cast<Impl*>(this)->DoCommit(
                    targetRankId, currentTileOffset_, currentTileSize);
            }
        }

        currentTileIdx_++;
        currentTileOffset_ += currentTileSize;
        remainingChunkSize_ -= currentTileSize;
    }

    __aicore__ inline void Wait(bool waitLast=false) {
        if (waitLast) {
            return;
        }
        const uint32_t rankId = udmaCtx_->rankId;
        for (uint32_t i = 0; i < targetRankCnt_; i++) {
            uint32_t targetRankId = targetRankStart_ + i;
            static_cast<Impl*>(this)->DoWait(targetRankId);
        }
    }

    __aicore__ inline void Finalize() {
        static_cast<Impl*>(this)->DoFinalize();
    }
};

} // namespace AivComm
} // namespace Atcoss
