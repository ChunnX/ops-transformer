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
 * \file all_gather_mx_matmul_urma_impl.h
 * \brief AllGatherQuantMatmul fusion — URMA variant (refactored).
 *
 * Win buffer reuse: remote compact layout + circular slot allocation.
 * When commTurn <= effBufferCount, reuse naturally degenerates to no-reuse.
 * BufferChannel serves as the bridge between AIV (producer) and AIC (consumer).
 */

#pragma once

#include "apace/basic/buffer_channel.h"
#include "all_gather_qbmm_mx_kernel.h"
#include "apace/core/aiv_comm/collective_comm_api.h"
#include "apace/core/aiv_comm/collective_comm_context.h"
#include "apace/core/aiv_comm/barrier/barrier_ubmem.h"
#include "apace/tiling/comm_tiling_data.h"
#include "apace/tiling/quant_matmul_tiling_data.h"
#include "blaze/gemm/utils/common_utils.h"
#include "include/tensor_api/tensor.h"
#include "../all_gather_mx_matmul_urma_tiling_data.h"

using namespace AscendC;
using namespace Apace::AivComm;

constexpr uint32_t SYNCALL_FLAG_ID_MAX = 14;

namespace Apace {

template <typename AType, typename BType, typename CType>
class AllGatherMxMatmulUrmaImpl {
private:
    using Barrier = TeamBarrier;
    using CommData = CollectiveComm<CommCollectiveOp::AllGather, CommMode::PUT, AType, Barrier>;
    using CommScale = CollectiveComm<CommCollectiveOp::AllGather, CommMode::PUT, AscendC::fp8_e8m0_t, Barrier>;
    using KernelImpl = AllGatherQbmmMxKernel<AType, BType, CType, AllGatherMxMatmulUrmaTilingData>;
    using TensorAddrs = typename KernelImpl::TensorAddrs;
    using TilingDerivedParams = typename KernelImpl::TilingDerivedParams;

    struct CommUbBuffers {
        __ubuf__ uint8_t *commBuf;
        __ubuf__ uint8_t *commScaleBuf;
        __ubuf__ uint8_t *barrierBuf;
    };

    static __aicore__ inline CommUbBuffers AllocCommUbBuffers();

public:
    __aicore__ inline AllGatherMxMatmulUrmaImpl() {}

    __aicore__ inline void Init(__gm__ CommContext *commCtx, GM_ADDR aGM, GM_ADDR aScaleGM, GM_ADDR bGM,
                                GM_ADDR bScaleGM, GM_ADDR cGM, GM_ADDR biasGM,
                                const AllGatherMxMatmulUrmaTilingData *tilingData);

    __aicore__ inline void Process();

    KernelImpl quantMatmulKernelImpl_;

private:
    __aicore__ inline void AllGatherProcess();

    CommData allGatherData_;
    CommScale allGatherScale_;
    Barrier teamBarrier_;
    BufferChannel dataChannel_;
    BufferChannel scaleChannel_;

    __gm__ CommContext *commCtx_{};
    TensorAddrs addrs_{};
    TilingDerivedParams tp_{};
    const AllGatherMxMatmulUrmaTilingData *tilingData_{};
};

template <typename AType, typename BType, typename CType>
__aicore__ inline typename AllGatherMxMatmulUrmaImpl<AType, BType, CType>::CommUbBuffers
AllGatherMxMatmulUrmaImpl<AType, BType, CType>::AllocCommUbBuffers()
{
    uint32_t offset = 0;
    auto commBuf = AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(offset);
    offset += COMM_WORKSPACE_SIZE;
    auto commScaleBuf = AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(offset);
    offset += COMM_WORKSPACE_SIZE;
    auto barrierBuf = AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, uint8_t>(offset);
    return {commBuf.Get(), commScaleBuf.Get(), barrierBuf.Get()};
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void
AllGatherMxMatmulUrmaImpl<AType, BType, CType>::Init(__gm__ CommContext *commCtx, GM_ADDR aGM, GM_ADDR aScaleGM,
                                                     GM_ADDR bGM, GM_ADDR bScaleGM, GM_ADDR cGM, GM_ADDR biasGM,
                                                     const AllGatherMxMatmulUrmaTilingData *tilingData)
{
    tilingData_ = tilingData;
    commCtx_ = commCtx;

    addrs_.aGM = aGM;
    addrs_.aScaleGM = aScaleGM;
    addrs_.bGM = bGM;
    addrs_.bScaleGM = bScaleGM;
    addrs_.cGM = cGM;
    addrs_.biasGM = biasGM;

    tp_.rankId = commCtx_->udmaCtx.rankId;
    tp_.rankSize = commCtx_->udmaCtx.rankSize;

    auto ub = AllocCommUbBuffers();

    teamBarrier_.Init(ub.barrierBuf, &(commCtx_->ubmemCtx), tp_.rankSize,
                      static_cast<uint32_t>(GetBlockIdx()));

    allGatherData_.template Init<BARRIER_NONE>(&(commCtx_->udmaCtx), teamBarrier_, tilingData_->commTile, addrs_.aGM,
                                               ub.commBuf, tp_.rankSize, static_cast<uint32_t>(GetBlockIdx()));

    tp_.commTurn = static_cast<uint32_t>(allGatherData_.GetCommTurn());

    tp_.effBufferCount = (tilingData_->commTile.slotNum < tp_.commTurn) ?
                          tilingData_->commTile.slotNum : tp_.commTurn;

    GM_ADDR winDataBase = reinterpret_cast<GM_ADDR>(commCtx_->udmaCtx.commBufferAddrs[tp_.rankId]);
    uint64_t scaleWinOffset = tp_.effBufferCount * tilingData_->commTile.dataSlotSize;
    dataChannel_.Init(winDataBase, tilingData_->commTile.dataSlotSize, static_cast<uint32_t>(tp_.effBufferCount));
    scaleChannel_.Init(winDataBase + scaleWinOffset, tilingData_->commTile.scaleSlotSize,
                       static_cast<uint32_t>(tp_.commTurn));

    CommTilingData scaleCommTiling = tilingData_->commTile;
    scaleCommTiling.nonSplitAxisSize = tilingData_->commTile.scaleNonSplitAxisSize;
    scaleCommTiling.slotNum = 0;
    allGatherScale_.Init(&(commCtx_->udmaCtx), teamBarrier_, scaleCommTiling, addrs_.aScaleGM, ub.commScaleBuf,
                         tp_.rankSize, static_cast<uint32_t>(GetBlockIdx()), scaleWinOffset);
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherMxMatmulUrmaImpl<AType, BType, CType>::Process()
{
    if ASCEND_IS_AIV {
        AllGatherProcess();
    }
    if ASCEND_IS_AIC {
        quantMatmulKernelImpl_.Run(tilingData_, addrs_, tp_, dataChannel_, scaleChannel_);
    }
}

template <typename AType, typename BType, typename CType>
__aicore__ inline void AllGatherMxMatmulUrmaImpl<AType, BType, CType>::AllGatherProcess()
{
    CrossCoreSetFlag<0x2, PIPE_MTE3>(0);

    for (uint32_t round = 0; round < tp_.commTurn; ++round) {
        if (round >= tp_.effBufferCount) {
            AscendC::CrossCoreWaitFlag<0x2, PIPE_MTE3>((round - tp_.effBufferCount) % SYNCALL_FLAG_ID_MAX);
            AscendC::SyncAll<true>();
            if (static_cast<uint32_t>(GetBlockIdx()) < tp_.rankSize) {
                teamBarrier_.CrossDevice();
            }
        }
        if (static_cast<uint32_t>(GetBlockIdx()) < tp_.rankSize) {
            auto scaleSlot = scaleChannel_.GetNextSlot();
            auto dataSlot = dataChannel_.GetNextSlot();
            allGatherScale_.Commit(scaleSlot.offset);
            allGatherData_.Commit(dataSlot.offset);
            allGatherData_.template Wait<BARRIER_DEVICE>();
        }
        AscendC::SyncAll<true>();
        CrossCoreSetFlag<0x2, PIPE_MTE3>(round + 1);
    }
    allGatherScale_.Finalize();
    allGatherData_.Finalize();

    dcci(reinterpret_cast<__gm__ void *>(commCtx_->udmaCtx.commBufferAddrs),
         static_cast<uint64_t>(CacheLine::ENTIRE_DATA_CACHE), static_cast<uint64_t>(DcciDst::CACHELINE_OUT));
}

} // namespace Apace
