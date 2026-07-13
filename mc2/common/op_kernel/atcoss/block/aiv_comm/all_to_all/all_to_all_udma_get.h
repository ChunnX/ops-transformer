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
 * \file all_to_all_udma_get.h
 * \brief AllToAll GET 模式实现 - 基于 hcomm ReadNbi
 */

#pragma once

#include "kernel_operator.h"
#include "adv_api/hcomm/hcomm.h"
#include "../collective_comm_base.h"
#include "../collective_comm_context.h"
#include "../barrier/barrier_ubmem.h"

namespace Dista {
namespace AivComm {

using namespace AscendC;

template<typename Dtype>
class AllToAllCommGetImpl : public CollectiveCommBase<AllToAllCommGetImpl<Dtype>, Dtype> {
    friend class CollectiveCommBase<AllToAllCommGetImpl<Dtype>, Dtype>;

private:
    __aicore__ inline void PostInit() {
    }

    __aicore__ inline void DoCommit(uint32_t targetRankId, uint64_t tileOffset, uint64_t tileSize) {
        Barrier barrier;
        barrier.Execute(this->syncBuf_, this->sbc_, this->useCoreNum_, this->curCoreId_);
        if ((this->curCoreId_ >= this->useCoreNum_) || (this->curCoreId_ == this->ctx_->myRankId)) {
            return;
        }
        GM_ADDR srcAddr = this->remoteAddrs_[targetRankId]
            + this->currentTileIdx_ * this->slotStride_
            + this->myRankOffset_;
        GM_ADDR dstAddr = this->localAddr_ + targetRankId * this->chunkBytes_ +
            tileOffset * this->nonSplitAxisBytes_;

        this->comm_.ReadNbi(
            this->channels_[targetRankId], dstAddr, srcAddr,
            static_cast<uint64_t>(tileSize * this->nonSplitAxisBytes_));
    }

    __aicore__ inline void DoWait() {
        const uint32_t myRankId = this->ctx_->myRankId;
        for (uint32_t i = 0; i < this->targetRankCnt_; i++) {
            uint32_t targetRankId = this->targetRankStart_ + i;
            if (targetRankId != myRankId) {
                this->comm_.Drain(this->channels_[targetRankId]);
            }
        }
    }

    __aicore__ inline void DoFinalize() {
        Barrier barrier;
        barrier.Execute(this->syncBuf_, this->sbc_, this->useCoreNum_, this->curCoreId_);
    }
};

} // namespace AivComm
} // namespace Dista
