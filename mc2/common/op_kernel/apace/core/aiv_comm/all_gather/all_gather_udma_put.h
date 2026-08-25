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

namespace Apace {
namespace AivComm {

using namespace AscendC;

template <typename Dtype, typename Barrier = TeamBarrier>
class AllGatherCommPutImpl : public CollectiveCommBase<AllGatherCommPutImpl<Dtype, Barrier>, Dtype, Barrier> {
    friend class CollectiveCommBase<AllGatherCommPutImpl<Dtype, Barrier>, Dtype, Barrier>;

private:
    template <uint8_t BarrierMode>
    __aicore__ inline void PostInit()
    {
        if constexpr (BarrierMode & BARRIER_DEVICE) {
            this->barrier_.CrossDevice();
        }
        if constexpr (BarrierMode & BARRIER_CORE) {
            this->barrier_.CrossCore();
        }
    }

    template <uint8_t BarrierMode>
    __aicore__ inline void DoCommit(uint32_t targetRankId, uint64_t tileByteSize)
    {
        if (targetRankId == this->udmaCtx_->rankId) {
            return;
        }
        GM_ADDR srcAddr = this->localAddr_ + this->tileByteOffset_;

        GM_ADDR dstAddr = reinterpret_cast<GM_ADDR>(this->udmaCtx_->commBufferAddrs[targetRankId] + this->winOffset_) +
                          this->udmaCtx_->rankId * this->chunkBytes_ + this->tileByteOffset_;

        int32_t ret = this->comm_.WriteNbi(
            static_cast<ChannelHandle>(this->udmaCtx_->channelHandles[targetRankId]), dstAddr, srcAddr, tileByteSize);
        ascendc_assert(ret == 0, "Urma writeNbi failed, ret=%d, targetRankId=%u", ret, targetRankId);
    }

    template <uint8_t BarrierMode>
    __aicore__ inline void DoWait(uint32_t targetRankId)
    {
        if (targetRankId != this->udmaCtx_->rankId) {
            int32_t ret = this->comm_.Drain(static_cast<ChannelHandle>(this->udmaCtx_->channelHandles[targetRankId]));
            ascendc_assert(ret == 0, "Urma drain failed, ret=%d, targetRankId=%u", ret, targetRankId);
        }
        if constexpr (BarrierMode & BARRIER_DEVICE) {
            this->barrier_.CrossDevice();
        }
        if constexpr (BarrierMode & BARRIER_CORE) {
            this->barrier_.CrossCore();
        }
    }

    template <uint8_t BarrierMode>
    __aicore__ inline void DoFinalize()
    {
        // [dcci 替代方案-研究中] hcomm Lock/Unlock 方案（依赖 CANN hcomm Lock/Unlock 接口，CANN 819 已打 patch，
        // 见疑问整理.md 问题18/19/20）。效果与 dcci ENTIRE 持平，连跑精度全 PASS。启用时需同步恢复
        // collective_comm_base.h 中的 jobIndex/totalJobs Finalize 保护。
        // for (uint32_t i = 0; i < this->targetRankCnt_; i++) {
        //     uint32_t targetRankId = this->targetRankStart_ + i;
        //     this->comm_.Lock(this->udmaCtx_->channelHandles[targetRankId]);
        //     this->comm_.Unlock(this->udmaCtx_->channelHandles[targetRankId]);
        // }
    }
};

} // namespace AivComm
} // namespace Apace
