/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHUNK_GATED_DELTA_RULE_CONTEXT_H
#define CHUNK_GATED_DELTA_RULE_CONTEXT_H

#include "../../chunk_gated_delta_rule_tiling_data.h"
#include "chunk_gated_delta_rule_layout.h"

namespace ChunkGatedDeltaRule {

using namespace AscendC;

// 入口只打包 GM 地址；张量描述符直接绑定到 Context 内的 KernelParams。
struct CGDRInitParams {
    GM_ADDR query;
    GM_ADDR key;
    GM_ADDR value;
    GM_ADDR beta;
    GM_ADDR initState;
    GM_ADDR seqlens;
    GM_ADDR gOptional;
    GM_ADDR attnOut;
    GM_ADDR finalState;
};

template <typename stateType>
struct KernelParams {
    // GM 逻辑 shape：Q/K:[T,Nk,Dk]，V/O:[T,Nv,Dv]，beta/g:[T,Nv]。
    GlobalTensor<bfloat16_t> query;
    GlobalTensor<bfloat16_t> key;
    GlobalTensor<bfloat16_t> value;
    GlobalTensor<bfloat16_t> beta;
    GlobalTensor<float> g;
    GlobalTensor<stateType> initialState; // [B,Nv,Dv,Dk]，batch/head 按 stateStride0/1 寻址
    GlobalTensor<bfloat16_t> out;
    GlobalTensor<stateType> finalState; // [B,Nv,Dv,Dk]，连续布局
    GlobalTensor<int32_t> actualSeqLengths; // [B]，各 batch 长度，总和为 T
    float scale;
    uint32_t nk;
    uint32_t nv;
    uint32_t dk;
    uint32_t dv;
    uint32_t aiCoreNum;
    uint64_t batchCount;
    uint64_t stateStride0; // 初始 state 的 batch 步长，单位为元素；0 表示广播
    uint64_t stateStride1; // 初始 state 的 value-head 步长，单位为元素；0 表示广播

    // 优化方案：在最终参数对象中就地绑定描述符，省去逐层构造、赋值和复制参数对象。
    // SetGlobalBuffer 只设置地址/长度等元数据，不搬运 GM 数据。
    // hasGamma 来自 CGDR 的模板常量；内联后可消除无 g 分支，无需为此扩展 Context 的模板参数。
    __aicore__ inline KernelParams(const CGDRInitParams &addresses,
                                  const ChunkGatedDeltaRuleTilingData &tiling, bool hasGamma)
    {
        // Q, K: [T,Nk,Dk]；value 头 h 共享 QK 头 floor(h/(Nv/Nk))。
        const uint64_t queryKeyElements = tiling.t * tiling.nk * tiling.dk;
        query.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(addresses.query), queryKeyElements);
        key.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(addresses.key), queryKeyElements);

        // V, O: [T,Nv,Dv]。
        const uint64_t valueElements = tiling.t * tiling.nv * tiling.dv;
        value.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(addresses.value), valueElements);
        out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(addresses.attnOut), valueElements);

        // beta, g: [T,Nv]；g 为 log decay，alpha=exp(g)，缺省 g 时 alpha=1。
        const uint64_t gateElements = tiling.t * tiling.nv;
        beta.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(addresses.beta), gateElements);
        if (hasGamma) {
            g.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(addresses.gOptional), gateElements);
        }

        // S_in, S_out: [B,Nv,Dv,Dk]。初态按 stateStride0/1 寻址（可广播），终态连续写回。
        const uint64_t stateElements = tiling.b * tiling.nv * tiling.dv * tiling.dk;
        initialState.SetGlobalBuffer(reinterpret_cast<__gm__ stateType *>(addresses.initState), stateElements);
        finalState.SetGlobalBuffer(reinterpret_cast<__gm__ stateType *>(addresses.finalState), stateElements);

        // seq_lens: [B]，存各 batch 长度，非累积终点；空序列的终态等于初态。
        actualSeqLengths.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(addresses.seqlens), tiling.b);

        scale = tiling.scale;
        nk = static_cast<uint32_t>(tiling.nk);
        nv = static_cast<uint32_t>(tiling.nv);
        dk = static_cast<uint32_t>(tiling.dk);
        dv = static_cast<uint32_t>(tiling.dv);
        aiCoreNum = static_cast<uint32_t>(tiling.aiCoreNum);
        batchCount = static_cast<uint64_t>(tiling.b);
        stateStride0 = static_cast<uint64_t>(tiling.stateStride0);
        stateStride1 = static_cast<uint64_t>(tiling.stateStride1);
    }
};

// 一个完整的 (batch, value-head) 任务；Scheduler 产出，BasicBlock 执行。
// tokenStart=sum(seq_lens[0:batchId]) 是原始TND位置，不取决于后续调度顺序。
// 整条序列的 chunk 递推期间，这四个字段保持不变。
struct WorkItem {
    uint64_t batchId;
    uint64_t tokenStart;
    uint32_t seqLength;
    uint32_t valueHead;
};

// Context 是参数的唯一持有者；BasicBlock/Compute 借用它，Scheduler 只借用只读 params。
// params 直接从入口地址和 tiling 构造，之后只读；不保存对入口临时参数的引用。
template <typename stateType>
struct KernelContext : FusedLayout {
    const KernelParams<stateType> params;
    WorkItem task;

    __aicore__ inline KernelContext(const CGDRInitParams &addresses,
                                   const ChunkGatedDeltaRuleTilingData &tiling, bool hasGamma)
        : params(addresses, tiling, hasGamma)
    {}

    // Q/K:[T,Nk,Dk]，V:[T,Nv,Dv]；每 Nv/Nk 个相邻 value 头共享一个 QK 头。
    __aicore__ inline uint32_t QueryKeyHead(uint32_t valueHead) const
    {
        return valueHead / (params.nv / params.nk);
    }

    // S:[B,Nv,Dv,Dk]，偏移单位为元素。初态 batch/head stride 可为 0，以支持广播。
    __aicore__ inline uint64_t InitialStateOffset(uint32_t valueHead, uint32_t valueRowBegin) const
    {
        return task.batchId * params.stateStride0 + static_cast<uint64_t>(valueHead) * params.stateStride1 +
               static_cast<uint64_t>(valueRowBegin) * params.dk;
    }

    // 终态始终连续写回；每个 AIV 独占 [valueRowBegin:valueRowBegin+validRows, :Dk]。
    __aicore__ inline uint64_t FinalStateOffset(uint32_t valueHead, uint32_t valueRowBegin) const
    {
        return (task.batchId * params.nv + valueHead) * params.dv * params.dk +
               static_cast<uint64_t>(valueRowBegin) * params.dk;
    }

    __aicore__ inline uint32_t ChunkNum() const
    {
        return (task.seqLength + CHUNK - 1) / CHUNK;
    }

    __aicore__ inline uint32_t ChunkValidRows(uint32_t chunk) const
    {
        const uint32_t chunkBegin = chunk * CHUNK;
        const uint32_t remaining = task.seqLength - chunkBegin;
        return remaining < CHUNK ? remaining : CHUNK;
    }

    __aicore__ inline uint32_t SubBlockValidRows(uint32_t chunk, uint32_t subBlockIdx) const
    {
        const uint32_t validRows = ChunkValidRows(chunk);
        const uint32_t rowBegin = subBlockIdx * HALF_CHUNK;
        if (validRows <= rowBegin) {
            return 0;
        }
        const uint32_t remaining = validRows - rowBegin;
        return remaining < HALF_CHUNK ? remaining : HALF_CHUNK;
    }

    __aicore__ inline uint32_t StateValidRows(uint32_t subBlockIdx) const
    {
        const uint32_t rowBegin = subBlockIdx * HALF_DIM;
        if (params.dv <= rowBegin) {
            return 0;
        }
        const uint32_t remaining = params.dv - rowBegin;
        return remaining < HALF_DIM ? remaining : HALF_DIM;
    }

};

} // namespace ChunkGatedDeltaRule

#endif // CHUNK_GATED_DELTA_RULE_CONTEXT_H
