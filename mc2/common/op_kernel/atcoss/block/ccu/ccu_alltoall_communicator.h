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
 * \file ccu_alltoall_communicator.h
 * \brief
 */

#ifndef ATCOSS_CCU_ALLTOALL_COMMUNICATOR_H
#define ATCOSS_CCU_ALLTOALL_COMMUNICATOR_H

#include "lib/hccl/hccl.h"

namespace Dista {

__aicore__ inline uint64_t CeilDivU64(uint64_t a, uint64_t b) { return (a + b - 1) / b; }

namespace AivComm {
namespace Ccu {

struct MC2AlltoAllContext {
    uint32_t taskCnt;
    GM_ADDR sendBuffer;
    GM_ADDR recvBuffer;
    uint64_t sendOffset;
    uint64_t recvOffset;
    uint64_t sendCount;
    uint64_t strideCount;
    uint64_t hcclDataType;
    uint8_t repeat = 1;
};

template <typename HcclType>
class MC2AlltoAllPrimitives {
public:
    __aicore__ inline AscendC::HcclHandle Prepare(HcclType *hccl, MC2AlltoAllContext *context, uint32_t taskIndex)
    {
        return hccl->template AlltoAll<false>(context->sendBuffer + taskIndex * context->sendOffset,
                                              context->recvBuffer + taskIndex * context->recvOffset, context->sendCount,
                                              (AscendC::HcclDataType)(static_cast<uint8_t>(context->hcclDataType)),
                                              context->strideCount, context->repeat);
    }

};

template <bool AICSync, AscendC::HcclServerType ServerType, typename ContextType, typename TilingDataType,
          template <typename> class Primitive, uint32_t SendCnt, uint32_t RecvCnt>
class HcclCommunicator {
public:
    __aicore__ inline HcclCommunicator(TilingDataType *tiling) : tiling_(tiling){};
    __aicore__ inline void Init();
    __aicore__ inline ContextType *GetContextPtr();
    __aicore__ inline void PrepareAll(uint32_t taskCnt);
    __aicore__ inline void Process(uint32_t taskIndex);
    __aicore__ inline void AddIndex(uint32_t taskCnt);
    __aicore__ inline void End();

private:
    enum Communicationtype {
        COMMUNICATION_WAIT_ONE,
        COMMUNICATION_SEND_ONE
    };
    TilingDataType *tiling_;
    AscendC::Hccl<ServerType> hccl_;
    ContextType context_;
    Primitive<AscendC::Hccl<ServerType>> primitive_;
    uint32_t startIndex_ = 0;
    uint32_t endIndex_ = 0;
    bool notifyFlag_ = false;
    Communicationtype communicationType_ = COMMUNICATION_WAIT_ONE;
    static constexpr uint8_t MAX_HCCL_HANDLE_ = 63;
    AscendC::HcclHandle hTasks_[MAX_HCCL_HANDLE_];
    bool taskSuccess_[MAX_HCCL_HANDLE_];
};

template <bool AICSync, AscendC::HcclServerType ServerType, typename ContextType, typename TilingDataType,
          template <typename> class Primitive, uint32_t SendCnt, uint32_t RecvCnt>
__aicore__ inline void
HcclCommunicator<AICSync, ServerType, ContextType, TilingDataType, Primitive, SendCnt, RecvCnt>::Init()
{
    notifyFlag_ = false;
    if ASCEND_IS_AIV {
        if (!AICSync && AscendC::GetBlockIdx() == 0) {
            notifyFlag_ = true;
        }
    } else if ASCEND_IS_AIC {
        if (AICSync && AscendC::GetBlockIdx() == 0) {
            notifyFlag_ = true;
        }
    }

    hccl_.InitV2(AscendC::GetHcclContext<0>(), &(tiling_->mc2InitTiling));
    hccl_.SetCcTilingV2(offsetof(TilingDataType, mc2CcTiling));
    if constexpr (SendCnt == 1U && RecvCnt == 0U) {
        communicationType_ = Communicationtype::COMMUNICATION_SEND_ONE;
    } else if constexpr (SendCnt == 0U && RecvCnt == 1U) {
        communicationType_ = Communicationtype::COMMUNICATION_WAIT_ONE;
    }

    for (uint8_t i = 0; i < MAX_HCCL_HANDLE_; ++i) {
        taskSuccess_[i] = true;
    }
}

template <bool AICSync, AscendC::HcclServerType ServerType, typename ContextType, typename TilingDataType,
          template <typename> class Primitive, uint32_t SendCnt, uint32_t RecvCnt>
__aicore__ inline ContextType *
HcclCommunicator<AICSync, ServerType, ContextType, TilingDataType, Primitive, SendCnt, RecvCnt>::GetContextPtr()
{
    return &context_;
}

template <bool AICSync, AscendC::HcclServerType ServerType, typename ContextType, typename TilingDataType,
          template <typename> class Primitive, uint32_t SendCnt, uint32_t RecvCnt>
__aicore__ inline void
HcclCommunicator<AICSync, ServerType, ContextType, TilingDataType, Primitive, SendCnt, RecvCnt>::PrepareAll(
    uint32_t taskCnt)
{
    if (!notifyFlag_) {
        return;
    }
    for (uint32_t i = 0; i < taskCnt; i++) {
        hTasks_[endIndex_ + i] = primitive_.Prepare(&hccl_, &context_, i);
    }
    AddIndex(taskCnt);
    if (communicationType_ == Communicationtype::COMMUNICATION_WAIT_ONE) {
        for (uint32_t i = 0; i < taskCnt; i++) {
            hccl_.Commit(hTasks_[startIndex_ + i]);
            taskSuccess_[startIndex_ + i] = false;
        }
    }
}

template <bool AICSync, AscendC::HcclServerType ServerType, typename ContextType, typename TilingDataType,
          template <typename> class Primitive, uint32_t SendCnt, uint32_t RecvCnt>
__aicore__ inline void
HcclCommunicator<AICSync, ServerType, ContextType, TilingDataType, Primitive, SendCnt, RecvCnt>::Process(
    uint32_t taskIndex)
{
    if (!notifyFlag_) {
        return;
    }
    if (communicationType_ == Communicationtype::COMMUNICATION_WAIT_ONE) {
        hccl_.Wait(hTasks_[startIndex_ + taskIndex]);
        taskSuccess_[startIndex_ + taskIndex] = true;
    } else if (communicationType_ == Communicationtype::COMMUNICATION_SEND_ONE) {
        hccl_.Commit(hTasks_[startIndex_ + taskIndex]);
        taskSuccess_[startIndex_ + taskIndex] = false;
    }
}

template <bool AICSync, AscendC::HcclServerType ServerType, typename ContextType, typename TilingDataType,
          template <typename> class Primitive, uint32_t SendCnt, uint32_t RecvCnt>
__aicore__ inline void
HcclCommunicator<AICSync, ServerType, ContextType, TilingDataType, Primitive, SendCnt, RecvCnt>::AddIndex(
    uint32_t taskCnt)
{
    if (!notifyFlag_) {
        return;
    }
    startIndex_ = endIndex_;
    endIndex_ += taskCnt;
}

template <bool AICSync, AscendC::HcclServerType ServerType, typename ContextType, typename TilingDataType,
          template <typename> class Primitive, uint32_t SendCnt, uint32_t RecvCnt>
__aicore__ inline void
HcclCommunicator<AICSync, ServerType, ContextType, TilingDataType, Primitive, SendCnt, RecvCnt>::End()
{
    if (notifyFlag_) {
        for (uint32_t i = 0; i < endIndex_; ++i) {
            if (!taskSuccess_[i]) {
                hccl_.Wait(hTasks_[i]);
                taskSuccess_[i] = true;
            }
        }
    }

    if (notifyFlag_) {
        hccl_.Finalize();
    }
}

template <typename TilingDataType, bool IsMxFp4 = false>
class MxQuantAlltoAllComm {
public:
    __aicore__ inline MxQuantAlltoAllComm(TilingDataType *tiling);
    __aicore__ inline void Init(GM_ADDR sendScale, GM_ADDR recvScale,
                                 GM_ADDR sendData, GM_ADDR recvData,
                                 uint64_t scaleHcclDtype, uint64_t dataHcclDtype);
    __aicore__ inline void ProcessScale();
    template <typename DataType>
    __aicore__ inline void ProcessData();
    __aicore__ inline void End();
    __aicore__ inline uint64_t GetRankForComm() const { return rankForComm_; }
    __aicore__ inline uint32_t GetScaleKPerRank() const { return scaleKPerRank_; }

private:
    static constexpr AscendC::HcclServerType kServerType = AscendC::HcclServerType::HCCL_SERVER_TYPE_CCU;
    using CommEngine = HcclCommunicator<false, kServerType, MC2AlltoAllContext,
                                         TilingDataType, MC2AlltoAllPrimitives, 0, 1>;
    CommEngine engine_;
    TilingDataType *tiling_;
    GM_ADDR sendScale_;
    GM_ADDR recvScale_;
    GM_ADDR sendData_;
    GM_ADDR recvData_;
    uint64_t scaleHcclDtype_, dataHcclDtype_;
    uint64_t rankForComm_;
    uint32_t scaleKPerRank_;
    static constexpr uint64_t MXFP_GROUP_SIZE = 64UL;
    static constexpr uint64_t NUM_TWO = 2UL;
};

template <typename TilingDataType, bool IsMxFp4>
__aicore__ inline MxQuantAlltoAllComm<TilingDataType, IsMxFp4>::MxQuantAlltoAllComm(TilingDataType *tiling)
    : engine_(tiling), tiling_(tiling) {}

template <typename TilingDataType, bool IsMxFp4>
__aicore__ inline void MxQuantAlltoAllComm<TilingDataType, IsMxFp4>::Init(
    GM_ADDR sendScale, GM_ADDR recvScale,
    GM_ADDR sendData, GM_ADDR recvData,
    uint64_t scaleHcclDtype, uint64_t dataHcclDtype)
{
    sendScale_ = sendScale;  recvScale_ = recvScale;
    sendData_ = sendData;    recvData_ = recvData;
    scaleHcclDtype_ = scaleHcclDtype;
    dataHcclDtype_ = dataHcclDtype;

    auto &&info = tiling_->alltoAllQuantMatmulTilingInfo;
    if constexpr (IsMxFp4) {
        rankForComm_ = CeilDivU64(info.rankK, 2) / info.rankDim;
    } else {
        rankForComm_ = info.rankK / info.rankDim;
    }
    scaleKPerRank_ = static_cast<uint32_t>(
        CeilDivU64(info.rankK / info.rankDim, MXFP_GROUP_SIZE) * NUM_TWO);

    engine_.Init();
}

template <typename TilingDataType, bool IsMxFp4>
__aicore__ inline void MxQuantAlltoAllComm<TilingDataType, IsMxFp4>::ProcessScale()
{
    auto &&info = tiling_->alltoAllQuantMatmulTilingInfo;
    uint64_t rankKdivByGroupSize = CeilDivU64(info.rankK / info.rankDim, MXFP_GROUP_SIZE) * NUM_TWO;

    auto *ctx = engine_.GetContextPtr();
    ctx->taskCnt = 1;
    ctx->sendBuffer = sendScale_;
    ctx->recvBuffer = recvScale_;
    ctx->sendCount = static_cast<uint64_t>(info.rankM) * rankKdivByGroupSize;
    ctx->strideCount = ctx->sendCount;
    ctx->hcclDataType = scaleHcclDtype_;

    engine_.PrepareAll(1);
    engine_.Process(0);
}

template <typename TilingDataType, bool IsMxFp4>
template <typename DataType>
__aicore__ inline void MxQuantAlltoAllComm<TilingDataType, IsMxFp4>::ProcessData()
{
    auto &&info = tiling_->alltoAllQuantMatmulTilingInfo;
    uint64_t strideVal = static_cast<uint64_t>(info.rankM) * rankForComm_;

    auto *ctx = engine_.GetContextPtr();
    ctx->sendBuffer = sendData_;
    ctx->recvBuffer = recvData_;
    ctx->strideCount = strideVal;
    ctx->hcclDataType = dataHcclDtype_;

    if (info.tileCnt > 0) {
        uint64_t tileMMultiRankK = static_cast<uint64_t>(info.tileM) * rankForComm_;
        ctx->taskCnt = info.tileCnt;
        ctx->sendOffset = tileMMultiRankK * sizeof(DataType);
        ctx->recvOffset = ctx->sendOffset;
        ctx->sendCount = tileMMultiRankK;

        engine_.PrepareAll(info.tileCnt);
        for (uint32_t i = 0; i < info.tileCnt; ++i) {
            engine_.Process(i);
            AscendC::SyncAll<true>();
            AscendC::CrossCoreSetFlag<0, PIPE_MTE3>(8);
            AscendC::CrossCoreWaitFlag(8);
            AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(9);
        }
    }

    if (info.tailCnt > 0) {
        uint64_t tilePhaseSize = static_cast<uint64_t>(info.tileCnt) * info.tileM
                                 * rankForComm_ * sizeof(DataType);
        uint64_t tailMMultiRankK = static_cast<uint64_t>(info.tailM) * rankForComm_;
        ctx->taskCnt = info.tailCnt;
        ctx->sendBuffer = sendData_ + tilePhaseSize;
        ctx->recvBuffer = recvData_ + tilePhaseSize;
        ctx->sendOffset = tailMMultiRankK * sizeof(DataType);
        ctx->recvOffset = ctx->sendOffset;
        ctx->sendCount = tailMMultiRankK;

        engine_.PrepareAll(info.tailCnt);
        for (uint32_t i = 0; i < info.tailCnt; ++i) {
            engine_.Process(i);
            AscendC::SyncAll<true>();
            AscendC::CrossCoreSetFlag<0, PIPE_MTE3>(8);
            AscendC::CrossCoreWaitFlag(8);
            AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(9);
        }
    }
}

template <typename TilingDataType, bool IsMxFp4>
__aicore__ inline void MxQuantAlltoAllComm<TilingDataType, IsMxFp4>::End()
{
    engine_.End();
}

} // namespace Ccu
} // namespace AivComm
} // namespace Dista

#endif
