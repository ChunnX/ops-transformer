/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEGA_MOE_SHARED_EXPERT_INPUT_H
#define MEGA_MOE_SHARED_EXPERT_INPUT_H

#include "../common/mega_moe_utils.h"
#include "mega_moe_token_quant.h"

namespace MegaMoeImpl {

using namespace AscendC;

template <typename ActivationType>
struct SharedExpertPrepareScratch {
    LocalTensor<ActivationType> copyBuffer0;
    LocalTensor<ActivationType> copyBuffer1;
};

// 拆分一个逻辑 AIV 任务上的共享专家 token data 和 scale。
// 通信记录布局直接复用 token quant 的 QuantProcessConfig，不再维护单独的拷贝。
template <typename ActivationType, typename QuantScaleType, uint32_t ActivationElementsPerByte>
__aicore__ inline void PrepareSharedExpertInput(const AivJobContext &job, const MoeStageCommonConfig &common,
                                                const Params &params, const QuantProcessConfig &config,
                                                SharedExpertPrepareScratch<ActivationType> &scratch)
{
    if constexpr (g_coreType == AIC) {
        return;
    }
    if (job.totalJobs == 0U || job.jobIndex >= job.totalJobs) {
        return;
    }

    WorkRange tokenRange = TilingByJobContext(common.tokenNum, job.jobIndex, job.totalJobs, 1U);

    int64_t tokenDataElementCount = common.tokenHiddenDim / ActivationElementsPerByte;
    int64_t tokenScaleElementCount =
        Ops::Base::CeilDiv(static_cast<int64_t>(common.tokenHiddenDim), static_cast<int64_t>(MXFP_DIVISOR_SIZE)) *
        MXFP_MULTI_BASE_SIZE;
    uint32_t copyTokenScaleBytes = config.quantTokenAlignBytes + config.quantScaleAlignBytes;

    GlobalTensor<ActivationType> srcGlobalTensor;
    GlobalTensor<ActivationType> dataDstGlobalTensor;
    GlobalTensor<QuantScaleType> scaleDstGlobalTensor;
    srcGlobalTensor.SetGlobalBuffer(reinterpret_cast<__gm__ ActivationType *>(params.peermemInfo.quantTokenScalePtr));
    dataDstGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ ActivationType *>(params.workspaceInfo.sharedExpertInputDataPtr));
    scaleDstGlobalTensor.SetGlobalBuffer(
        reinterpret_cast<__gm__ QuantScaleType *>(params.workspaceInfo.sharedExpertInputScalePtr));

    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    for (uint32_t index = 0; index < tokenRange.count; ++index) {
        uint32_t tokenIdx = tokenRange.start + index;
        uint64_t srcOffset = static_cast<uint64_t>(tokenIdx) * static_cast<uint64_t>(config.quantTokenScaleAlignBytes);
        bool useFirstBuffer = index % DOUBLE_BUFFER == 0;
        auto event = useFirstBuffer ? EVENT_ID0 : EVENT_ID1;
        auto copyBuffer = useFirstBuffer ? scratch.copyBuffer0 : scratch.copyBuffer1;

        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(event);
        DataCopy(copyBuffer, srcGlobalTensor[srcOffset], copyTokenScaleBytes);
        SetFlag<AscendC::HardEvent::MTE2_MTE3>(event);
        WaitFlag<AscendC::HardEvent::MTE2_MTE3>(event);

        LocalTensor<QuantScaleType> scaleBuffer =
            copyBuffer[config.quantTokenAlignBytes].template ReinterpretCast<QuantScaleType>();
        DataCopyPad(dataDstGlobalTensor[static_cast<int64_t>(tokenIdx) * tokenDataElementCount], copyBuffer,
                    {1, static_cast<uint16_t>(tokenDataElementCount * sizeof(ActivationType)), 0U, 0U, 0U});
        DataCopyPad(scaleDstGlobalTensor[static_cast<int64_t>(tokenIdx) * tokenScaleElementCount], scaleBuffer,
                    {1, static_cast<uint16_t>(tokenScaleElementCount * sizeof(QuantScaleType)), 0U, 0U, 0U});
        SetFlag<AscendC::HardEvent::MTE3_MTE2>(event);
    }
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    // 本屏障同时兜住 a8w8 路径窗口 0 权重预取在飞块的排空（shared>0 分支在 SyncAll 前无其他
    // 排空点，调用方另有防御性 Drain 兜底，见 mega_moe_wave_a8w8.h Process）。本函数不得新增
    // 早于本屏障的提前返回，否则在飞预取写 [192K,208K) 会与 GMM epilogue 的 C tile 跨队列竞争。
    PipeBarrier<PIPE_ALL>();
}

} // namespace MegaMoeImpl

#endif // MEGA_MOE_SHARED_EXPERT_INPUT_H
