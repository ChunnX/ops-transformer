/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

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
    __aicore__ inline explicit BasicBlock(const KernelContext<stateType>& context)
        : context_(context),
          cube_(context),
          vector_(context),
          stream_(context)
    {
        // 计算侧初始化每个核只做一次，不属于任务分配策略。
        if ASCEND_IS_AIC {
            SetLoadDataPaddingValue<bfloat16_t>(static_cast<bfloat16_t>(0));
        } else {
            subBlockIdx_ = static_cast<uint32_t>(GetSubBlockIdx());
        }
    }

    __aicore__ inline void Run(const WorkItem& task)
    {
        // task仅在本次 Run 调用链中借用；Compute 显式接收它，Context 始终只读。
        if ASCEND_IS_AIC {
            if (!UseStreamingExact(task.seqLength)) {
                cube_.Run(task);
            }
        } else {
            if (UseStreamingExact(task.seqLength)) {
                stream_.Run(task, subBlockIdx_);
            } else {
                vector_.Run(task, subBlockIdx_);
            }
        }
    }

private:
    __aicore__ inline bool UseStreamingExact(uint64_t seqLength) const
    {
        // 优化方案：小工作量跳过固定 [64,128] Cube tile，用 AIV 按有效 Dk/Dv 直接递推。
        // AIC/AIV 必须用相同元数据作出相同选择；Streaming 下 AIC 不进入任何 chunk 握手。
        // Streaming 每 token 的主项：S:[Dv,Dk]，k/q:[Dk]，v/p/r:[Dv]，alpha=exp(g)。
        // 一次乘、一次加各计 1；点积的 Dk 次乘和 Dk-1 次加近似计为 2*Dk。
        // r=beta*(v-alpha*p) 后更新 S；以下系数乘 Dk*Dv，合计 7*Dk*Dv。
        constexpr uint64_t STREAM_ARITHMETIC_FACTOR = 2 + // p=S@k：矩阵向量乘，约 2*Dk*Dv。
                                                      1 + // S_decay=alpha*S：逐元素乘，Dk*Dv。
                                                      2 + // S_next=S_decay+r*k.T：外积及累加，2*Dk*Dv。
                                                      2;  // o=S_next@(scale*q)：矩阵向量乘，约 2*Dk*Dv。
        // 仅包含核心逻辑运算项，非 VF 指令计数：省略 residual/exp、scale*q（VF 中可能重复）、
        // 类型转换、规约/广播及 padding 的开销；256 仅为选路的固定相对权重。
        constexpr uint64_t STREAM_VECTOR_COST = 256;

        // Cube 按实际 MMAD 的物理 shape 计数：[M,K]@[K,N] 为 M*N*K 个 MAC（乘加对）。
        // C=CHUNK=64，D=DIM=128，H=HALF_CHUNK=32；尾块/窄维也按补齐后的 C/D 计算。
        // 编号对应 arch35.h 的公式及 CubeCompute；三角矩阵仍执行完整的矩阵乘。
        constexpr uint64_t CUBE_MAC_PER_CHUNK =
            2ULL * CHUNK * CHUNK * DIM +                  // [1] KKT/QKT：[2C,D]@[D,C]。
            2ULL * HALF_CHUNK * HALF_CHUNK * HALF_CHUNK + // [4a/b] tmp=(-L10)@A00，A10=A11@tmp；两次 [H,H] 乘。
            2ULL * CHUNK * CHUNK * DIM +                  // [5a/b] Kcd=A@rawKcd，U=A@vBeta；两次 [C,C]@[C,D]。
            2ULL * CHUNK * DIM * DIM +                    // [6a/b] R=U+Kcd@S.T，O_inter=Q'@S.T；两次 [C,D]@[D,D]。
            1ULL * CHUNK * DIM * DIM +                    // [7] delta=R16.T@Kg：[D,C]@[C,D]。
            1ULL * CHUNK * CHUNK * DIM;                   // [8] O=O_inter+M@R16：[C,C]@[C,D]。
        // =5*C*C*D + 3*C*D*D + 2*H*H*H =2621440+3145728+65536 =5832704 MAC/chunk。
        // cubeScore 乘 2，与 streamScore 统一为乘/加操作数；此处未计 chunk 路径的 Vector 计算。
        // 两侧均未显式计入搬运/同步与活跃核数，长序列下 L 近乎相消；分数不是设备耗时预测。
        // 输入长度仍受 int32 接口约束，Dk/Dv<=128；以下乘积在 uint64 范围内。
        if (seqLength == 0) {
            return true;
        }
        const uint64_t streamScore =
            seqLength * context_.params.dk * context_.params.dv * STREAM_ARITHMETIC_FACTOR * STREAM_VECTOR_COST;
        const uint64_t chunkCount = ChunkNum(seqLength);
        const uint64_t cubeScore = chunkCount * 2 * CUBE_MAC_PER_CHUNK;
        return streamScore < cubeScore;
    }

    const KernelContext<stateType>& context_;
    CubeCompute<stateType, hasGamma> cube_;
    VectorCompute<stateType, hasGamma> vector_;
    VectorStreamCompute<stateType, hasGamma> stream_;
    uint32_t subBlockIdx_ = 0;
};

} // namespace ChunkGatedDeltaRule
