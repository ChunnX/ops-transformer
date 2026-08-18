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
 * \file allto_all_matmul_pipeline.h
 * \brief
 */

#ifndef ALLTO_ALL_MATMUL_PIPELINE_H
#define ALLTO_ALL_MATMUL_PIPELINE_H

#include "../../common/op_kernel/mc2_templates/communication/comm_stage.h"
#include "../../common/op_kernel/mc2_templates/computation/math/mc2_vec_transpose.h"
#include "../../common/op_kernel/mc2_templates/computation/math/mc2_vec_add_bias.h"
#include "../../common/op_kernel/mc2_templates/computation/matmul/mc2_fp_matmul.h"

// 流水线模板
namespace Mc2Kernel {
template <typename ComputationContextType>
struct AlltoAllMmPipelineContext {
    // computation
    ComputationContextType *computationContext;
    // transpose
    MC2KernelTemplate::MC2TransposeContext *transposeContext;
    // add bias (独立加bias节点, arch35 FP场景启用; 其他场景该指针不读取)
    MC2KernelTemplate::MC2AddBiasContext *addBiasContext;
    // communication
    MC2KernelTemplate::MC2AlltoAllContext *communicationContext;
};

// 通信转置计算模板
template <typename CommunicationType, typename TransposeType, typename ComputationType, typename ContextType>
class AlltoAllMatmulPipeLine {
public:
    __aicore__ inline AlltoAllMatmulPipeLine(CommunicationType *commStage, TransposeType *transStage,
                                             ComputationType *computeStage)
        : commStage_(commStage), transStage_(transStage), computeStage_(computeStage){};

    __aicore__ inline void Init();

    __aicore__ inline void GetContext(ContextType *context);

    __aicore__ inline void Process(uint32_t taskCnt);

    __aicore__ inline void End();

private:
    CommunicationType *commStage_;  // 通信节点
    TransposeType *transStage_;     // 转置计算的计算节点
    ComputationType *computeStage_; // 矩阵乘的计算节点
};

template <typename CommunicationType, typename TransposeType, typename ComputationType, typename ContextType>
__aicore__ inline void AlltoAllMatmulPipeLine<CommunicationType, TransposeType, ComputationType, ContextType>::Init()
{
    commStage_->Init();
    computeStage_->Init();
}

template <typename CommunicationType, typename TransposeType, typename ComputationType, typename ContextType>
__aicore__ inline void
AlltoAllMatmulPipeLine<CommunicationType, TransposeType, ComputationType, ContextType>::GetContext(ContextType *context)
{
    context->communicationContext = commStage_->GetContextPtr();
    context->transposeContext = transStage_->GetContextPtr();
    context->computationContext = computeStage_->GetContextPtr();
}

template <typename CommunicationType, typename TransposeType, typename ComputationType, typename ContextType>
__aicore__ inline void
AlltoAllMatmulPipeLine<CommunicationType, TransposeType, ComputationType, ContextType>::Process(uint32_t taskCnt)
{
    commStage_->PrepareAll(taskCnt);
    uint32_t index;
    for (index = 0; index < taskCnt; index++) {
        if ASCEND_IS_AIV {
            commStage_->Process(index);
            AscendC::SyncAll<true>();
            transStage_->Process(index);
            // 核间使用软同步，当前搭配的matmul没有使用软同步标识索引8和9，后续如果matmul有变化需要同步调整
            AscendC::CrossCoreSetFlag<0, PIPE_MTE3>(8);
            AscendC::CrossCoreWaitFlag(8);
            AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(9);
        }
        if ASCEND_IS_AIC {
            AscendC::CrossCoreWaitFlag(9);
            computeStage_->Process(index);
        }
    }
}

template <typename CommunicationType, typename TransposeType, typename ComputationType, typename ContextType>
__aicore__ inline void AlltoAllMatmulPipeLine<CommunicationType, TransposeType, ComputationType, ContextType>::End()
{
    commStage_->End();
    computeStage_->End();
}

// 通信转置计算 + 独立加 bias 模板 (arch35 FP 非量化场景使用)
// 每轮 task 流水: comm(AIV) → transpose(AIV) → matmul(AIC) → add_bias(AIV)
// 核间同步采用 SyncAll 硬件 barrier (参考 arch35 KC 量化流水线):
//   barrier1(转置后): 转置结果对 AIC 可见
//   barrier2(matmul后): matmul 输出对 AIV 可见, 随后 AIV 对当前 tile 加 bias
template <typename CommunicationType, typename TransposeType, typename ComputationType, typename AddBiasType,
          typename ContextType>
class AlltoAllMatmulAddBiasPipeLine {
public:
    __aicore__ inline AlltoAllMatmulAddBiasPipeLine(CommunicationType *commStage, TransposeType *transStage,
                                                    ComputationType *computeStage, AddBiasType *addBiasStage)
        : commStage_(commStage), transStage_(transStage), computeStage_(computeStage), addBiasStage_(addBiasStage){};

    __aicore__ inline void Init();

    __aicore__ inline void GetContext(ContextType *context);

    __aicore__ inline void Process(uint32_t taskCnt);

    __aicore__ inline void End();

private:
    CommunicationType *commStage_;  // 通信节点
    TransposeType *transStage_;     // 转置计算的计算节点
    ComputationType *computeStage_; // 矩阵乘的计算节点
    AddBiasType *addBiasStage_;     // 加 bias 的计算节点
};

template <typename CommunicationType, typename TransposeType, typename ComputationType, typename AddBiasType,
          typename ContextType>
__aicore__ inline void
AlltoAllMatmulAddBiasPipeLine<CommunicationType, TransposeType, ComputationType, AddBiasType, ContextType>::Init()
{
    commStage_->Init();
    computeStage_->Init();
}

template <typename CommunicationType, typename TransposeType, typename ComputationType, typename AddBiasType,
          typename ContextType>
__aicore__ inline void
AlltoAllMatmulAddBiasPipeLine<CommunicationType, TransposeType, ComputationType, AddBiasType, ContextType>::GetContext(
    ContextType *context)
{
    context->communicationContext = commStage_->GetContextPtr();
    context->transposeContext = transStage_->GetContextPtr();
    context->computationContext = computeStage_->GetContextPtr();
    context->addBiasContext = addBiasStage_->GetContextPtr();
}

template <typename CommunicationType, typename TransposeType, typename ComputationType, typename AddBiasType,
          typename ContextType>
__aicore__ inline void
AlltoAllMatmulAddBiasPipeLine<CommunicationType, TransposeType, ComputationType, AddBiasType, ContextType>::Process(
    uint32_t taskCnt)
{
    commStage_->PrepareAll(taskCnt);
    // 每轮 task 完整执行 通信→转置→matmul→加bias 四个阶段
    for (uint32_t index = 0; index < taskCnt; index++) {
        // 阶段1/2: AIV 通信 + 转置
        if ASCEND_IS_AIV {
            commStage_->Process(index);
            AscendC::SyncAll<true>();
            transStage_->Process(index);
        }
        // barrier1: 转置结果对 AIC 可见
        AscendC::SyncAll<false>();
        // 阶段3: AIC matmul (AIV 侧 return)
        computeStage_->Process(index);
        // barrier2: matmul 输出 y 写回对 AIV 可见
        AscendC::SyncAll<false>();
        // 阶段4: AIV 对当前 tile 的 matmul 输出加 bias
        if ASCEND_IS_AIV {
            addBiasStage_->Process(index);
        }
    }
}

template <typename CommunicationType, typename TransposeType, typename ComputationType, typename AddBiasType,
          typename ContextType>
__aicore__ inline void
AlltoAllMatmulAddBiasPipeLine<CommunicationType, TransposeType, ComputationType, AddBiasType, ContextType>::End()
{
    commStage_->End();
    computeStage_->End();
}
}; // namespace Mc2Kernel

#endif
