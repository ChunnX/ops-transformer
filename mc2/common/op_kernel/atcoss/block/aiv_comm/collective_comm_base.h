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
 *        不依赖 shmem，使用 Hcomm<COMM_PROTOCOL_UBC_CTP> 实现点对点通信
 */

#pragma once

#include "kernel_operator.h"
#include "adv_api/hcomm/hcomm.h"
#include "collective_comm_context.h"
#include "dista/utils/comm_tiling_data.h"
#include "tensor_api/tensor/pointer.h"

namespace Dista {
namespace AivComm {

using namespace AscendC;
using Utils::CommTilingData;

template<typename Impl, typename Dtype>
class CollectiveCommBase {
    friend Impl;
protected:
    const CommContext* ctx_;
    GM_ADDR localAddr_;
    const CommTilingData* tilingData_;
    const SyncBufferContext* sbc_;

    Hcomm<AscendC::COMM_PROTOCOL_UBC_CTP> comm_;
    __ubuf__ uint8_t* syncBuf_;

    ChannelHandle channels_[HCOMM_MAX_RANK_NUM];
    GM_ADDR remoteAddrs_[HCOMM_MAX_RANK_NUM];
    GM_ADDR localBufferBase_;

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
        const CommContext* ctx,
        const SyncBufferContext* sbc,
        GM_ADDR localAddr,
        const CommTilingData& tilingData,
        uint32_t useCoreNum,
        uint32_t curCoreId)
    {
        ctx_ = ctx;
        sbc_ = sbc;
        localAddr_ = localAddr;
        tilingData_ = &tilingData;
        useCoreNum_ = useCoreNum;
        curCoreId_ = curCoreId;

        __gm__ CommChannelContext* hctx = ctx->commCtx;
        localBufferBase_ = reinterpret_cast<GM_ADDR>(hctx->localBufferAddr);
        for (uint32_t i = 0; i < ctx->rankSize; i++) {
            channels_[i] = static_cast<ChannelHandle>(hctx->channelHandles[i]);
            remoteAddrs_[i] = reinterpret_cast<GM_ADDR>(hctx->remoteBufferAddrs[i]);
        }

        uint64_t ubSize = GetUBSizeInBytes();
        uint64_t hcommOffset = (ubSize - HCOMM_WORKSPACE_SIZE) & ~31UL;
        auto memPtr = AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(hcommOffset);
        comm_.Init(memPtr.Get(), HCOMM_WORKSPACE_SIZE);

        constexpr uint64_t EXTRA_BUF_SIZE = 32;
        uint64_t syncOffset = (hcommOffset - EXTRA_BUF_SIZE) & ~31UL;
        auto syncPtr_ = AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(syncOffset);
        syncBuf_ = syncPtr_.Get();

        uint32_t rankSize = ctx->rankSize;
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
        myRankOffset_ = ctx->myRankId * tileMaxByteSize;
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
            const uint32_t myRankId = ctx_->myRankId;
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

    __aicore__ inline void Wait() {
        static_cast<Impl*>(this)->DoWait();
    }

    __aicore__ inline void Finalize() {
        static_cast<Impl*>(this)->DoFinalize();
    }
};

} // namespace AivComm
} // namespace Dista
