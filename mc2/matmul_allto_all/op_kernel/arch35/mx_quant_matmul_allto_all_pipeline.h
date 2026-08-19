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
 * \file mx_quant_matmul_allto_all_pipeline.h
 * \brief MX量化专用流水线，支持scale通信+转置、数据通信+转置+计算
 */

#ifndef MX_QUANT_MATMUL_ALLTO_ALL_PIPELINE_H
#define MX_QUANT_MATMUL_ALLTO_ALL_PIPELINE_H

#include "../../common/op_kernel/mc2_templates/scheduler/pipeline_builder.h"

namespace Mc2Kernel {
template <typename ComputationContextType>
struct MxQuantMatmulAlltoAllPipelineContext {
    // computation
    ComputationContextType *computationContext;
    // transpose
    MC2KernelTemplate::MC2TransposeContext *transposeContext;
    // scaleTranspose
    MC2KernelTemplate::MC2TransposeContext *scaleTransposeContext;
    // communication
    MC2KernelTemplate::MC2AlltoAllContext *communicationContext;
};

// 通信转置计算模板（MX量化专用）
template <typename CommunicationType, typename TransposeType, typename ScaleTransposeType, typename ComputationType,
          typename ContextType>
class MxQuantMatmulAlltoAllPipeLine {
public:
    __aicore__ inline MxQuantMatmulAlltoAllPipeLine(CommunicationType *commStage, TransposeType *transStage,
                                                     ScaleTransposeType *scaleTransStage, ComputationType *computeStage)
        : commStage_(commStage), transStage_(transStage), scaleTransStage_(scaleTransStage),
          computeStage_(computeStage){};

    __aicore__ inline void Init();

    __aicore__ inline void GetContext(ContextType *context);

    __aicore__ inline void Process(uint32_t taskCnt);

    __aicore__ inline void ProcessScale();

    __aicore__ inline void End();

private:
    CommunicationType *commStage_;        // 通信节点
    TransposeType *transStage_;           // 转置计算的计算节点
    ScaleTransposeType *scaleTransStage_; // 缩放转置的计算节点
    ComputationType *computeStage_;       // 矩阵乘的计算节点
};

template <typename CommunicationType, typename TransposeType, typename ScaleTransposeType, typename ComputationType,
          typename ContextType>
__aicore__ inline void
MxQuantMatmulAlltoAllPipeLine<CommunicationType, TransposeType, ScaleTransposeType, ComputationType,
                              ContextType>::Init()
{
    commStage_->Init();
    computeStage_->Init();
}

template <typename CommunicationType, typename TransposeType, typename ScaleTransposeType, typename ComputationType,
          typename ContextType>
__aicore__ inline void
MxQuantMatmulAlltoAllPipeLine<CommunicationType, TransposeType, ScaleTransposeType, ComputationType,
                              ContextType>::GetContext(ContextType *context)
{
    context->communicationContext = commStage_->GetContextPtr();
    context->transposeContext = transStage_->GetContextPtr();
    context->scaleTransposeContext = scaleTransStage_->GetContextPtr();
    context->computationContext = computeStage_->GetContextPtr();
}

template <typename CommunicationType, typename TransposeType, typename ScaleTransposeType, typename ComputationType,
          typename ContextType>
__aicore__ inline void
MxQuantMatmulAlltoAllPipeLine<CommunicationType, TransposeType, ScaleTransposeType, ComputationType,
                              ContextType>::Process(uint32_t taskCnt)
{
    commStage_->PrepareAll(taskCnt);
    for (uint32_t index = 0; index < taskCnt; index++) {
        if ASCEND_IS_AIC {
            // matmul_allto_all: 计算在前，通信在后
            // 1. AIC 先执行 matmul
            computeStage_->Process(index);
            AscendC::CrossCoreSetFlag<0, PIPE_FIX>(8);
            AscendC::CrossCoreWaitFlag(8);
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(9);
        }
        if ASCEND_IS_AIV {
            // 2. AIV 等待 AIC 计算完成
            AscendC::CrossCoreWaitFlag(9);
            // 3. 转置 matmul 结果
            transStage_->Process(index);
            AscendC::SyncAll<true>();
            // 4. AllToAll 通信发送转置后的结果
            commStage_->Process(index);
        }
    }
}

template <typename CommunicationType, typename TransposeType, typename ScaleTransposeType, typename ComputationType,
          typename ContextType>
__aicore__ inline void
MxQuantMatmulAlltoAllPipeLine<CommunicationType, TransposeType, ScaleTransposeType, ComputationType,
                              ContextType>::ProcessScale()
{
    // matmul_allto_all: scale 在 matmul 阶段本地消费，无需独立通信+转置
    // 此处保留空实现以匹配 pipeline 接口
}

template <typename CommunicationType, typename TransposeType, typename ScaleTransposeType, typename ComputationType,
          typename ContextType>
__aicore__ inline void
MxQuantMatmulAlltoAllPipeLine<CommunicationType, TransposeType, ScaleTransposeType, ComputationType,
                              ContextType>::End()
{
    commStage_->End();
    computeStage_->End();
};
} // namespace Mc2Kernel

#endif
