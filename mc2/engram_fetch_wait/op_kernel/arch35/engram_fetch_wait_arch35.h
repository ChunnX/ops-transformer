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
 * \file engram_fetch_wait_arch35.h
 * \brief engram_fetch_wait算子arch35 kernel实现
 */


#ifndef ENGRAM_FETCH_WAIT_ARCH35_H
#define ENGRAM_FETCH_WAIT_ARCH35_H

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "kernel_tiling/kernel_tiling.h"
#include "../engram_fetch_wait_tiling_data.h"
#if __has_include("../../engram_fetch/engram_fetch.h")
#include "../../engram_fetch/engram_fetch.h"
#else
#include "../../../engram_fetch/op_kernel/engram_fetch.h"
#endif
#include "adv_api/hccl/hccl.h"
#include "adv_api/hcomm/hcomm.h"

namespace Mc2Kernel {

using namespace AscendC;

class EngramFetchWaitArch35 {
public:
    __aicore__ inline EngramFetchWaitArch35() = default;

    __aicore__ inline void Init(GM_ADDR commContext, GM_ADDR workspaceGM, TPipe *pipe);

    __aicore__ inline void Process();

private:
    TPipe *tpipe_{nullptr};
    __gm__ EngramCommContext *ctxPtr_{nullptr};

    uint32_t aivId_{0};
    uint32_t rankId_{0};
    uint32_t numRanks_{0};

    TBuf<> hcommBuf_;

    AscendC::Hcomm<COMM_PROTOCOL_UBC_CTP> hcomm_;
};

__aicore__ inline void EngramFetchWaitArch35::Init(GM_ADDR commContext, GM_ADDR workspaceGM, TPipe *pipe)
{
    tpipe_ = pipe;
    aivId_ = GetBlockIdx();

    ctxPtr_ = (__gm__ EngramCommContext *)commContext;
    rankId_ = ctxPtr_->rankId;
    numRanks_ = ctxPtr_->rankSize;

    tpipe_->InitBuffer(hcommBuf_, Mc2Kernel::HCOMM_INIT_SIZE);
    LocalTensor<uint8_t> hcommTensor = hcommBuf_.Get<uint8_t>();
    hcomm_.Init(hcommTensor, Mc2Kernel::HCOMM_INIT_SIZE);
}

__aicore__ inline void EngramFetchWaitArch35::Process()
{
    if ASCEND_IS_AIV {
        uint32_t totalBlocks = GetBlockNum();
        for (uint32_t channelIdx = aivId_; channelIdx < numRanks_; channelIdx += totalBlocks) {
            if (channelIdx == rankId_) {
                continue;
            }
            hcomm_.Drain(ctxPtr_->hcommHandle_[channelIdx]);
        }
    }
}

} // namespace Mc2Kernel

#endif
