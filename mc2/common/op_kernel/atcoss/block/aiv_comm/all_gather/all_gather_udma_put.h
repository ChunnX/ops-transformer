/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_gather_udma_put.h
 * \brief AllGather PUT 模式实现 - 基于 hcomm WriteNbi
 */

#pragma once

#include "adv_api/hcomm/hcomm.h"
#include "../collective_comm_base.h"

namespace Atcoss {
namespace AivComm {

using namespace AscendC;

template<typename Dtype, typename Barrier = TeamBarrier, CommAddrLayout Layout = CommAddrLayout::DEFAULT>
class AllGatherCommPutImpl : public CollectiveCommBase<AllGatherCommPutImpl<Dtype, Barrier, Layout>, Dtype, Barrier, Layout> {
    friend class CollectiveCommBase<AllGatherCommPutImpl<Dtype, Barrier, Layout>, Dtype, Barrier, Layout>;

private:
    __aicore__ inline void PostInit() 
    {
        this->barrier_.Execute();
    }

    __aicore__ inline void DoCommit(uint32_t targetRankId, uint64_t tileOffset, uint64_t tileSize)
    {
        if ((this->curCoreId_ >= this->useCoreNum_) || (targetRankId == this->udmaCtx_->rankId)) {
            return;
        }
        GM_ADDR srcAddr = this->localAddr_ + tileOffset * this->nonSplitAxisBytes_;

        GM_ADDR dstAddr;
        if constexpr (Layout == CommAddrLayout::CONTIGUOUS) {
            dstAddr = this->udmaCtx_->commBufferAddrs[targetRankId] + this->winOffsset_ + this->currentTileIdx_ * this->udmaCtx_->rankSize * this->tileMaxByteSize_ +
                      this->udmaCtx_->rankId * tileSize * this->nonSplitAxisBytes_;
        } else {
            dstAddr = this->udmaCtx_->commBufferAddrs[targetRankId] + this->winOffsset_ + this->currentTileIdx_ * this->chunkBytes_ +
                      tileOffset * this->nonSplitAxisBytes_;
        }

        this->comm_.WriteNbi(
            static_cast<ChannelHandle>(this->udmaCtx_->channelHandles[targetRankId]), dstAddr, srcAddr,
            static_cast<uint64_t>(tileSize * this->nonSplitAxisBytes_));
    }

    __aicore__ inline void DoWait(uint32_t targetRankId)
    {
        this->barrier_.Execute();
        if ((this->curCoreId_ >= this->useCoreNum_) || (targetRankId == this->udmaCtx_->rankId)) {
            return;
        }
        this->comm_.Drain(this->udmaCtx_->channelHandles[targetRankId]);
    }

    __aicore__ inline void DoFinalize() {}
};

} // namespace AivComm
} // namespace Atcoss
