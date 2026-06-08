/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/* !
 * \file pipeline_template_comm_compute.h
 * \brief
 */

#ifndef MC2_PIPELINE_TEMPLATE_COMM_COMPUTE_H
#define MC2_PIPELINE_TEMPLATE_COMM_COMPUTE_H

#include "../a2av_gmm_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif

using namespace AscendC;

namespace MC2KernelTemplate {
template <typename CommOpType, typename ComputeOpType, typename LocalComputeOpType>
class A2avGmmScheduler {
public:
    __aicore__ inline A2avGmmScheduler(CommOpType &commOp, ComputeOpType &computeOp, LocalComputeOpType &localComputeOp,
                                        const TaskTilingInfo *taskTilingInfo, bool isNeedMM)
        : commOp_(commOp), computeOp_(computeOp), localComputeOp_(localComputeOp),
          taskTilingInfo_(taskTilingInfo), isNeedMM_(isNeedMM){};
    __aicore__ inline void Init();

    __aicore__ inline void Process()
    {
        // shared expert
        if (isNeedMM_) {
            localComputeOp_.Process(0, 1);
        }
        // routing expert
        uint32_t expertNum = taskTilingInfo_->expertNum;
        uint32_t e = taskTilingInfo_->e;
        bool isNeedPermute = expertNum > 1;
        for (uint32_t startExpertIdx = 0; startExpertIdx < e; startExpertIdx += expertNum) {
            uint32_t actualExpertNum = (startExpertIdx + expertNum > e) ? (e - startExpertIdx) : expertNum;
            if constexpr (MX_QUANT_MODE) {
                commOp_.LaunchCommScale(startExpertIdx, actualExpertNum);
            }
            commOp_.LaunchCommData(startExpertIdx, actualExpertNum);
        }
        for (uint32_t startExpertIdx = 0; startExpertIdx < e; startExpertIdx += expertNum) {
            uint32_t actualExpertNum = (startExpertIdx + expertNum > e) ? (e - startExpertIdx) : expertNum;
            if ASCEND_IS_AIV {
                if constexpr (MX_QUANT_MODE) {
                    commOp_.WaitScale(startExpertIdx);
                    if (isNeedPermute) {
                        SyncAll<true>();
                        commOp_.PermuteScale(startExpertIdx, actualExpertNum);
                    }
                }
                commOp_.Wait(startExpertIdx);
                if (isNeedPermute) {
                    SyncAll<true>();
                    commOp_.PermuteData(startExpertIdx, actualExpertNum);
                }
                SyncAll<true>();
                AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(9);
            }
            if ASCEND_IS_AIC {
                AscendC::CrossCoreWaitFlag(9);
                computeOp_.Process(startExpertIdx, actualExpertNum);
            }
        }
        this->End();
    }

protected:
    __aicore__ inline void End()
    {
        commOp_.End();
        computeOp_.End();
        localComputeOp_.End();
    }

private:
    __aicore__ inline void ProcessCommRoundMode();

private:
    CommOpType &commOp_;
    ComputeOpType &computeOp_;
    LocalComputeOpType &localComputeOp_;
    const TaskTilingInfo* taskTilingInfo_;
    bool isNeedMM_{false};
};
}; // namespace MC2KernelTemplate
#endif