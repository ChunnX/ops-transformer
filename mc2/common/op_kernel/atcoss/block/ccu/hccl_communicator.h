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
 * \file hccl_communicator.h
 * \brief
 */

#ifndef DISTA_CCU_HCCL_COMMUNICATOR_H
#define DISTA_CCU_HCCL_COMMUNICATOR_H

#include "lib/hccl/hccl.h"

namespace Dista {
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

    __aicore__ inline AscendC::HcclHandle SyncSend(HcclType *hccl, MC2AlltoAllContext *context, uint32_t taskIndex)
    {
        return hccl->template AlltoAll<true>(context->sendBuffer + taskIndex * context->sendOffset,
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
    __aicore__ inline void ProcessWithPrepare(uint32_t taskIndex);
    __aicore__ inline void AddIndex(uint32_t taskCnt);
    __aicore__ inline void PrepareAll();
    __aicore__ inline void Process();
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
HcclCommunicator<AICSync, ServerType, ContextType, TilingDataType, Primitive, SendCnt, RecvCnt>::ProcessWithPrepare(
    uint32_t taskIndex)
{
    if (!notifyFlag_) {
        return;
    }
    if (communicationType_ == Communicationtype::COMMUNICATION_WAIT_ONE) {
        AscendC::HcclHandle handleId = primitive_.SyncSend(&hccl_, &context_, taskIndex);
        hccl_.Wait(handleId);
    } else if (communicationType_ == Communicationtype::COMMUNICATION_SEND_ONE) {
        hTasks_[endIndex_ + taskIndex] = primitive_.SyncSend(&hccl_, &context_, taskIndex);
        taskSuccess_[endIndex_ + taskIndex] = false;
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
HcclCommunicator<AICSync, ServerType, ContextType, TilingDataType, Primitive, SendCnt, RecvCnt>::PrepareAll()
{
    if (!notifyFlag_) {
        return;
    }
    hTasks_[endIndex_] = primitive_.Prepare(&hccl_, &context_, 0);
    for (uint32_t i = 1; i < context_.repeat; ++i) {
        hTasks_[endIndex_ + i] = hTasks_[endIndex_];
    }
    AddIndex(context_.repeat);
    if (communicationType_ == Communicationtype::COMMUNICATION_WAIT_ONE) {
        for (uint32_t i = 0; i < context_.repeat; ++i) {
            hccl_.Commit(hTasks_[startIndex_ + i]);
            taskSuccess_[startIndex_ + i] = false;
        }
    }
}

template <bool AICSync, AscendC::HcclServerType ServerType, typename ContextType, typename TilingDataType,
          template <typename> class Primitive, uint32_t SendCnt, uint32_t RecvCnt>
__aicore__ inline void
HcclCommunicator<AICSync, ServerType, ContextType, TilingDataType, Primitive, SendCnt, RecvCnt>::Process()
{
    Process(0);
    startIndex_ += 1;
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

} // namespace Ccu
} // namespace AivComm
} // namespace Dista

#ifndef DEFINE_DISTA_CCU_HCCL_FOR_COMMUNICATION
#define DEFINE_DISTA_CCU_HCCL_FOR_COMMUNICATION(AICSync, ServerType, HcclContextType, TilingDataType, Primitive,       \
                                                sendCntPerTask, recvCntPerTask, CommunicationType)                     \
    using CommunicationType =                                                                                          \
        Dista::AivComm::Ccu::HcclCommunicator<AICSync, ServerType, HcclContextType, TilingDataType, Primitive,         \
                                              sendCntPerTask, recvCntPerTask>
#endif

#endif
