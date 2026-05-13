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
 * \file allto_all_mx_tensor_matmul_pipeline.h
 * \brief Pipeline for MX-FP8 tensor_api branch (transpose removal optimization)
 *        Uses GM auto-accumulation for rank-level K-axis summation
 */

#ifndef ALLTO_ALL_MX_TENSOR_MATMUL_PIPELINE_H
#define ALLTO_ALL_MX_TENSOR_MATMUL_PIPELINE_H

#include "../../common/op_kernel/mc2_templates/scheduler/pipeline_builder.h"

namespace Mc2Kernel {
template <typename ComputationContextType>
struct AlltoAllMXTensorQmmPipelineContext {
    ComputationContextType *computationContext;
    MC2KernelTemplate::MC2AlltoAllContext *communicationContext;
};

template <typename CommunicationType, typename ComputationType, typename ContextType>
class AlltoAllMXTensorMatmulPipeLine {
public:
    __aicore__ inline AlltoAllMXTensorMatmulPipeLine(CommunicationType *commStage, ComputationType *computeStage)
        : commStage_(commStage), computeStage_(computeStage){}

    __aicore__ inline void Init();
    __aicore__ inline void GetContext(ContextType *context);
    __aicore__ inline void Process(uint32_t taskCnt);
    __aicore__ inline void ProcessScale();
    __aicore__ inline void End();

private:
    CommunicationType *commStage_;
    ComputationType *computeStage_;
};

template <typename CommunicationType, typename ComputationType, typename ContextType>
__aicore__ inline void AlltoAllMXTensorMatmulPipeLine<CommunicationType, ComputationType, ContextType>::Init()
{
    commStage_->Init();
    computeStage_->Init();
}

template <typename CommunicationType, typename ComputationType, typename ContextType>
__aicore__ inline void AlltoAllMXTensorMatmulPipeLine<CommunicationType, ComputationType, ContextType>::GetContext(ContextType *context)
{
    context->communicationContext = commStage_->GetContextPtr();
    context->computationContext = computeStage_->GetContextPtr();
}

template <typename CommunicationType, typename ComputationType, typename ContextType>
__aicore__ inline void AlltoAllMXTensorMatmulPipeLine<CommunicationType, ComputationType, ContextType>::Process(uint32_t taskCnt)
{
    commStage_->PrepareAll(taskCnt);
    for (uint32_t index = 0; index < taskCnt; index++) {
        if ASCEND_IS_AIC {
            commStage_->Process(index);
            computeStage_->Process(index);
        }
    }
}

template <typename CommunicationType, typename ComputationType, typename ContextType>
__aicore__ inline void AlltoAllMXTensorMatmulPipeLine<CommunicationType, ComputationType, ContextType>::ProcessScale()
{
    if ASCEND_IS_AIC {
        commStage_->PrepareAll(1);
        commStage_->Process(0);
    }
}

template <typename CommunicationType, typename ComputationType, typename ContextType>
__aicore__ inline void AlltoAllMXTensorMatmulPipeLine<CommunicationType, ComputationType, ContextType>::End()
{
    commStage_->End();
    computeStage_->End();
}
}; // namespace Mc2Kernel

#endif