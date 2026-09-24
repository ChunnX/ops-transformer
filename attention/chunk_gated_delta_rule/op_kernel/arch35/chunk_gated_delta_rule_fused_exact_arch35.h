/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHUNK_GATED_DELTA_RULE_FUSED_EXACT_ARCH35_H
#define CHUNK_GATED_DELTA_RULE_FUSED_EXACT_ARCH35_H

#include "chunk_gated_delta_rule_basic_block.h"
#include "chunk_gated_delta_rule_scheduler.h"

namespace ChunkGatedDeltaRule {

using namespace AscendC;

/*
 * arch35 三阶段融合实现：一个 AIC 及其两个 AIV 持续拥有一个 (batch, value-head)，
 * 沿该序列顺序递推 chunk。AIC 负责矩阵乘，AIV 负责门控、求逆和状态更新；中间量驻留 UB/L1，
 * 避免旧实现各阶段之间的 GM 往返及 SyncAll。计算量很小时可改走纯 AIV 逐 token 递推。
 *
 * 单个 chunk 的数学约定（省略 BF16 转换）：
 * Q/K 为 [C,Dk]，V 为 [C,Dv]，S 为 [Dv,Dk]。
 * G_i = sum(g_0..g_i)
 * L_ij = beta_i * (K_i K_j^T) * exp(G_i-G_j)，仅 j<i 非零，A=(I+L)^(-1)。
 * rawKcd_i=-beta_i*exp(G_i)*K_i
 * vBeta_i=beta_i*V_i
 * qPrime_i=scale*exp(G_i)*Q_i
 * kg_i=exp(G_last-G_i)*K_i；令 kcd=A*rawKcd，则：
 *   U = A*vBeta:[C,Dv]；R = U + kcd*S^T:[C,Dv]；S_next = exp(G_last)*S + R^T*kg:[Dv,Dk]；
 *   O = qPrime*S^T + maskedQkt*R:[C,Dv]，maskedQkt_ij=scale*(Q_i K_j^T)*exp(G_i-G_j)，j<=i。
 * rawKcd 已带负号，因此残差可直接累加进 L0C。Exact 表示实现递推语义，不表示无 BF16 舍入。
 *
 * Cube/Vector 分工:
 *   -> PrepareVectorPrefix 准备对角逆块/rawKcd/Q'/Kg/M
 *   -> ComputeCubeChunk 计算 R/O/delta
 *   -> ComputeVectorChunk 更新 FP32 S
 * shape 默认是单 head，物理 DIM 固定为 128
 */
// 外层驱动 Scheduler::Next(task) -> BasicBlock::Run(task)；调度与计算仅以完整head任务衔接。
// BasicBlock 选择路径，再由 Cube/Vector/VectorStreamCompute 递推。
// 所有层均为编译期类型和内联调用，无虚函数、动态分配或额外的设备侧缓冲区。
template <typename stateType, bool hasGamma>
class FusedStage123Exact {
public:
    // 参数仍由CGDR持有；Scheduler只读参数，BasicBlock管理Compute当前任务及计算侧初始化。
    __aicore__ static inline void Process(KernelContext<stateType> &context)
    {
        BasicBlock<stateType, hasGamma> block(context);
        Scheduler<stateType> scheduler(context.params);
        WorkItem task;
        while (scheduler.Next(task)) {
            block.Run(task);
        }
        // 各 head 的计数握手均已收尾；最后汇合所有核，确保流水排空。
        SyncAll<false>();
    }
};

} // namespace ChunkGatedDeltaRule

#endif // CHUNK_GATED_DELTA_RULE_FUSED_EXACT_ARCH35_H
