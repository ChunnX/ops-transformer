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
 * \file chunk_gated_delta_rule_arch35.h
 * \brief
 */
#pragma once

#include "basic_api/kernel_basic_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../chunk_gated_delta_rule_tiling_data.h"
#include "chunk_gated_delta_rule_basic_block.h"
#include "chunk_gated_delta_rule_scheduler.h"

namespace ChunkGatedDeltaRule {

using namespace AscendC;

/*
 * arch35 三阶段融合实现：一个 AIC 及其两个 AIV 持续拥有一个 (batch, value-head)，
 * 沿该序列顺序递推 chunk。AIC 负责矩阵乘，AIV 负责门控、求逆和状态更新；中间量驻留 UB/L1，
 * 避免旧实现各阶段之间的 GM 往返及 SyncAll。计算量很小时可改走纯 AIV 逐 token 递推。
 *
 * 数学步骤索引（与 docs/math_to_kernel.md、pipeline_sample 的 ①~⑨ 对齐）：
 * 单 head：Q/K:[C,Dk]，V:[C,Dv]，beta/g:[C]，S:[Dv,Dk]；S 是进入当前 chunk 的状态。
 * [1, Cube]   KKT=K@K.T，QKT=Q@K.T，均为 [C,C]；一次打包 MMAD 同时计算两者。
 * [2a, Vector] G=cumsum(g)，expG=exp(G)，均为 [C]。
 * [2b, Vector] rawKcd=-beta*expG*K，vBeta=beta*V，Q'=scale*expG*Q，Kg=exp(G_last-G)*K。
 * [3a, Vector] L=tril(beta*KKT*exp(G_i-G_j),-1)，M=tril(scale*QKT*exp(G_i-G_j),0)，均为 [C,C]。
 * [3b, Vector] 两个 AIV 各求一个 [32,32] 对角逆块：A00=(I+L00)^-1，A11=(I+L11)^-1。
 * [4a/b, Cube] tmp=(-L10)@A00；A10=A11@tmp；与 [3b] 合成 A=(I+L)^-1:[C,C]。
 * [5a, Cube]  Kcd=A@rawKcd:[C,Dk]；[5b, Cube] U=A@vBeta:[C,Dv]，两者不依赖 S。
 * [6a, Cube]  R=U+Kcd@S.T:[C,Dv]；[6b, Cube] O_inter=Q'@S.T:[C,Dv]，共享一次 S 的 L0B 装入。
 * [6c, Vector] R32 -> R16，并搬 UB -> L1，供 [7]/[8] 的 Cube 使用。
 * [7, Cube]   delta=R16.T@Kg:[Dv,Dk]，FIX 搬到 Vector UB。
 * [8, Cube]   O=O_inter+M@R16:[C,Dv]，FIX 写回 GM。
 * [9a, Vector] S_next=exp(G_last)*S+delta:[Dv,Dk]；[9b] 按需将 BF16 副本送到 L1 供下一块读取。
 * R 已包含 beta；rawKcd 已包含负号；M 的对角线保留而 L 的对角线清零。
 * 编号是数学依赖顺序，不是跨核串行执行顺序：[1]/[2] 并行，下块 [1~3] 与当前块 [6~9] 重叠。
 * 公式省略中间 BF16 舍入；Exact 表示实现递推语义，不表示无舍入。不得据实数结合律重排乘法。
 *
 * 符号和存储命名约定（Python golden 使用相同词根的 snake_case）：
 *   G -> gCum，expG -> gExp，Q' -> qPrime，Kg -> kg，M -> maskedQkt，L -> lower，A -> inverse。
 *   rawKcd / vBeta 保留公式名；golden 的 Kcd/U 字段为 kcd/u；tmp -> lowerTimesInverse。
 *   delta=R.T@Kg -> stateDelta；Streaming 的 beta*(v-alpha*S@k) 则叫 residual。
 *   kcdQPrime 是 [Kcd;Q'] 的连续存储，不是额外计算结果。
 *   residual/output 缓冲分别跨阶段保存 U->R / O_inter->O，保留生命周期说明。
 *
 * Cube/Vector 分工:
 *   -> PrepareStateIndependentChunkInputs 准备对角逆块/rawKcd/Q'/Kg/M
 *   -> ComputeCubeChunk 计算 R/O/delta
 *   -> ComputeVectorChunk 更新 FP32 S
 * shape 默认是单 head，物理 DIM 固定为 128
 *
 * 外层驱动 Scheduler::Next(task) -> BasicBlock::Run(task)；调度与计算仅以完整head任务衔接。
 * BasicBlock 选择路径，再由 Cube/Vector/VectorStreamCompute 递推。
 */
template <typename stateType, bool hasGamma>
class CGDR {
public:
    __aicore__ inline CGDR(const CGDRInitParams& addresses, const ChunkGatedDeltaRuleTilingData& tiling)
        : context_(addresses, tiling, hasGamma)
    {}

    __aicore__ inline void Process()
    {
        SyncAll<false>();
        BasicBlock<stateType, hasGamma> block(context_);
        Scheduler<stateType> scheduler(context_.params);
        WorkItem task;
        while (scheduler.Next(task)) {
            block.Run(task);
        }
        // 各 head 的计数握手均已收尾；最后汇合所有核，确保流水排空。
        SyncAll<false>();
    }

private:
    const KernelContext<stateType> context_;
};

} // namespace ChunkGatedDeltaRule
