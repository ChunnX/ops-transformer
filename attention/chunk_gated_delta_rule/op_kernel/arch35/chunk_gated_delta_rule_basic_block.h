/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHUNK_GATED_DELTA_RULE_BASIC_BLOCK_H
#define CHUNK_GATED_DELTA_RULE_BASIC_BLOCK_H

#include "chunk_gated_delta_rule_cube_compute.h"
#include "chunk_gated_delta_rule_vector_compute.h"
#include "chunk_gated_delta_rule_vector_stream_compute.h"

namespace ChunkGatedDeltaRule {

using namespace AscendC;

// BasicBlock 为一个 (batch,value-head) 选择实现路径；AIC/AIV 共享同一判据。
// 各 Compute 内部负责本路径的 chunk/token 循环和缓冲生命周期，调度层不接触私有布局。
template <typename stateType, bool hasGamma>
class BasicBlock : private FusedLayout {
public:
    __aicore__ inline explicit BasicBlock(KernelContext<stateType> &context)
        : context_(context), cube_(context), vector_(context), stream_(context)
    {
        // 计算侧初始化每个核只做一次，不属于任务分配策略。
        if ASCEND_IS_AIC {
            SetLoadDataPaddingValue<bfloat16_t>(static_cast<bfloat16_t>(0));
        } else {
            subBlock_ = static_cast<uint32_t>(GetSubBlockIdx());
        }
    }

    __aicore__ inline void Run(const WorkItem &task)
    {
        // 借用外层产出的任务，更新Compute共同读取的当前任务；不保存对外层局部变量的引用。
        context_.task = task;
        if ASCEND_IS_AIC {
            if (!UseStreamingExact()) {
                cube_.Run(task.valueHead);
            }
        } else {
            if (UseStreamingExact()) {
                stream_.Run(task.valueHead, subBlock_);
            } else {
                vector_.Run(task.valueHead, subBlock_);
            }
        }
    }

private:
    __aicore__ inline bool UseStreamingExact() const
    {
        // AIC/AIV 使用相同元数据选择路径；空序列只复制状态，不进入 chunk 握手。
        if (context_.task.seqLength == 0) {
            return true;
        }
        // Dk=Dv=128 的专用 chunk VF 已融合四个变换。实测除单 token 接近持平外，
        // chunk 明显更快；保留这条常见网络路径，避免用通用维度的拟合外推。
        if (context_.params.dk == DIM && context_.params.dv == DIM) {
            return false;
        }

        // 优化方案：packed VF 最多8个行组，L<=18时新旧判据都选择stream。
        // 旧式最坏 (33*8-49)*18=3870<4000；新式 (33*8-36)*18=4104<4700。
        // 因此也覆盖超出已校准head范围时的旧判据回退，是精确短路。
        if (context_.params.dk <= 8 && context_.task.seqLength <= 18) {
            return true;
        }

        // 优化方案：按实际串行的“状态行组 × token”估计 stream 成本，而非 Dk*Dv FLOP。
        // AIV 按 Dv 各持有最多64行，以较慢的 AIV0 为准；Dk<=8 时一次打包8行，
        // 其余 Dk 每次处理1行。Dk>64 需要两段规约，实测成本略高，不是简单翻倍。
        const uint64_t stateRows = context_.params.dv < HALF_DIM ? context_.params.dv : HALF_DIM;
        const uint64_t rowGroups = context_.params.dk <= 8 ? (stateRows + 7) / 8 : stateRows;
        // 单行组保留实测稳定的 stream 快路径，避免极小用例支付完整判定开销。
        // 这是独立规则；多 head 的边界拟合不能外推成单行组的长期耗时模型。
        if (rowGroups == 1) {
            return true;
        }
        const uint64_t tokenCount = context_.task.seqLength;
        const uint64_t chunkCount = (tokenCount + CHUNK - 1) / CHUNK;

        // 优化方案：直接最小化选错路径的相对耗时损失，不再拟合绝对耗时交点。
        // 相对成本分数：Δ = (a*rowGroups - b)*L - (3600 + 1100*ceil(L/64))。
        // Δ<0 选 stream；相等选 chunk。a=33/37 对应一段/两段规约。
        // b=43（Nv=1）或36（Nv>1）：交叉实验确认多 head 的边界不同；
        // beta/V/O 的 GM 行步长也随 Nv 改变。该修正是经验校准，不是独立搬运计时。
        // 校准同时约束相对原判据的误选回退；418点含首轮失败的验证集，最终另用新数据验收。
        // 数据、被拒候选和独立验证见 docs/stream_chunk_dispatch。
        constexpr uint64_t STREAM_ROW_TOKEN_COST = 33;
        constexpr uint64_t STREAM_WIDE_ROW_TOKEN_COST = 37;
        constexpr uint64_t CHUNK_SETUP_BUDGET = 3600;
        constexpr uint64_t CHUNK_BATCH_BUDGET = 1100;
        constexpr uint64_t SINGLE_HEAD_TOKEN_CREDIT = 43;
        constexpr uint64_t MULTI_HEAD_TOKEN_CREDIT = 36;
        const uint64_t rowTokenCost = context_.params.dk > HALF_DIM ?
            STREAM_WIDE_ROW_TOKEN_COST : STREAM_ROW_TOKEN_COST;
        // 校准覆盖Nv<=32。64head补测显示直接外推会改变多波任务的最优路径，
        // 因此超过实测校准范围时保留原判据，待调度/多波成本另行校准。
        constexpr uint32_t MAX_CALIBRATED_VALUE_HEADS = 32;
        if (context_.params.nv > MAX_CALIBRATED_VALUE_HEADS) {
            return (rowTokenCost * rowGroups - 49) * tokenCount < 3200 + 800 * chunkCount;
        }

        // 先比较单 head 成本，多 head 再补逐 token 修正；与 Δ 判据完全等价。
        // 避免把 head 选择混入行组乘法，使公共的单 head 路径保持简单。
        const uint64_t singleHeadCost =
            (rowTokenCost * rowGroups - SINGLE_HEAD_TOKEN_CREDIT) * tokenCount;
        const uint64_t chunkBudget = CHUNK_SETUP_BUDGET + CHUNK_BATCH_BUDGET * chunkCount;
        if (context_.params.nv == 1) {
            return singleHeadCost < chunkBudget;
        }
        const uint64_t multiHeadCorrection =
            (SINGLE_HEAD_TOKEN_CREDIT - MULTI_HEAD_TOKEN_CREDIT) * tokenCount;
        return singleHeadCost + multiHeadCorrection < chunkBudget;
    }

    KernelContext<stateType> &context_;
    CubeCompute<stateType, hasGamma> cube_;
    VectorCompute<stateType, hasGamma> vector_;
    VectorStreamCompute<stateType, hasGamma> stream_;
    uint32_t subBlock_ = 0;
};

} // namespace ChunkGatedDeltaRule

#endif // CHUNK_GATED_DELTA_RULE_BASIC_BLOCK_H
