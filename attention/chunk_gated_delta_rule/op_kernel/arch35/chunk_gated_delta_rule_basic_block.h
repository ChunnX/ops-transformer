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
        // 优化方案：小工作量跳过固定 [64,128] Cube tile，用 AIV 按有效 Dk/Dv 直接递推。
        // AIC/AIV 必须用相同元数据作出相同选择；Streaming 下 AIC 不进入任何 chunk 握手。
        // 该算术量模型未显式计入批搬运/同步与活跃核数，长序列下 L 近乎相消；空序列仅复制状态。
        // stream≈7*L*Dk*Dv，再乘 Vector 相对代价256；Cube每chunk固定5832704个MAC。
        // 乘2统一为乘/加操作数，此模型只用于两种核角色一致选择路径，不是设备耗时预测。
        constexpr uint64_t STREAM_ARITHMETIC_FACTOR = 7;
        constexpr uint64_t STREAM_VECTOR_COST = 256;
        constexpr uint64_t CUBE_MAC_PER_CHUNK = 5832704;
        if (context_.task.seqLength == 0) {
            return true;
        }
        const uint64_t streamScore = static_cast<uint64_t>(context_.task.seqLength) * context_.params.dk * context_.params.dv *
                                     STREAM_ARITHMETIC_FACTOR * STREAM_VECTOR_COST;
        const uint64_t chunkCount = (static_cast<uint64_t>(context_.task.seqLength) + CHUNK - 1) / CHUNK;
        const uint64_t cubeScore = chunkCount * 2 * CUBE_MAC_PER_CHUNK;
        return streamScore < cubeScore;
    }

    KernelContext<stateType> &context_;
    CubeCompute<stateType, hasGamma> cube_;
    VectorCompute<stateType, hasGamma> vector_;
    VectorStreamCompute<stateType, hasGamma> stream_;
    uint32_t subBlock_ = 0;
};

} // namespace ChunkGatedDeltaRule

#endif // CHUNK_GATED_DELTA_RULE_BASIC_BLOCK_H
