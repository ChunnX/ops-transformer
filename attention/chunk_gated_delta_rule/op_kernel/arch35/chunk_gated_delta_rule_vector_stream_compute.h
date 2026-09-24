/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHUNK_GATED_DELTA_RULE_VECTOR_STREAM_COMPUTE_H
#define CHUNK_GATED_DELTA_RULE_VECTOR_STREAM_COMPUTE_H

#include "chunk_gated_delta_rule_context.h"
#include "chunk_gated_delta_rule_vector_copy.h"
#include "vf/chunk_gated_delta_rule_stage2_vf.h"

namespace ChunkGatedDeltaRule {

using namespace AscendC;

// 小工作量的 AIV 递推：按有效 Dk/Dv 计算，主状态保持 FP32，不与 Cube 握手。
// 外层只传 value head 和 AIV 子块；批次划分、UB 布局与流水同步均在本类内部完成。
template <typename stateType, bool hasGamma>
class VectorStreamCompute : private FusedLayout {
private:
    static constexpr bool STATE_FP32 = std::is_same_v<stateType, float>;
    static constexpr uint32_t STREAM_BATCH_TOKENS = CHUNK;
    // Streaming 与 chunk 分支互斥；这里独立管理自己的两项同步资源，可复用相同 ID。
    static constexpr uint32_t STATE_MUTEX = 2;
    static constexpr uint32_t INPUT_MUTEX = 0;
    static_assert(STATE_MUTEX < MUTEX_COUNT && INPUT_MUTEX < MUTEX_COUNT,
                  "streaming mutex index exceeds hardware capacity");

    struct Buffers {
        LocalTensor<float> stateFp32;       // [HALF_DIM,DIM]，跨 token 的主状态
        LocalTensor<bfloat16_t> stateBf16;   // [HALF_DIM,DIM]，接口状态转换区
        LocalTensor<bfloat16_t> query;      // [STREAM_BATCH_TOKENS,DIM]
        LocalTensor<bfloat16_t> key;        // [STREAM_BATCH_TOKENS,DIM]
        LocalTensor<bfloat16_t> valueBf16;  // [STREAM_BATCH_TOKENS,DIM]，有效列最多 HALF_DIM
        LocalTensor<float> valueFp32;      // [STREAM_BATCH_TOKENS,DIM]
        LocalTensor<bfloat16_t> betaBf16;   // 每 token 一个 32B 块
        LocalTensor<float> betaFp32;       // 每 token 两个 32B 块，直接 Cast 得到
        LocalTensor<float> g;              // 每 token 一个 32B 块
        LocalTensor<float> outFp32;        // [STREAM_BATCH_TOKENS,HALF_DIM]
        LocalTensor<bfloat16_t> outBf16;    // [STREAM_BATCH_TOKENS,HALF_DIM]
    };

    struct Batch {
        uint32_t valueHead;
        uint32_t queryKeyHead;
        uint32_t stateRowBegin;
        uint32_t stateRows;
        uint32_t tokenCount;
        uint64_t globalToken;
    };

    struct Head {
        uint32_t valueHead;
        uint32_t queryKeyHead;
        uint32_t subBlock; // 0/1：主状态沿 Dv 划分为两个最多 64 行的切片
    };

public:
    __aicore__ inline explicit VectorStreamCompute(KernelContext<stateType> &context) : context_(context) {}

    __aicore__ inline void Run(uint32_t valueHead, uint32_t subBlock)
    {
        const uint32_t stateRows = context_.StateValidRows(subBlock);
        if (stateRows == 0) {
            return;
        }
        if (context_.task.seqLength == 0) {
            CopyEmptyStateGmToUbToGm(valueHead, subBlock);
            return;
        }
        const Head head{valueHead, context_.QueryKeyHead(valueHead), subBlock};
        const auto buffers = BindStreamingBuffers();
        CopyInitialStreamingStateGmToUb(head, buffers);
        // STATE:[stateRows,DIM] 的 V 使用期覆盖初态转换、全部批次及最终量化。
        // 在本层成对管理，两个搬运函数分别负责自身的 MTE2/MTE3 同步。
        const MutexID stateMutex = MutexId(STATE_MUTEX);
        Mutex::Lock<PIPE_V>(stateMutex);
        if constexpr (!STATE_FP32) {
            Cast(buffers.stateFp32, buffers.stateBf16, RoundMode::CAST_NONE, stateRows * DIM);
            PipeBarrier<PIPE_V>();
        }
        for (uint32_t token = 0; token < context_.task.seqLength; token += STREAM_BATCH_TOKENS) {
            const uint32_t remaining = context_.task.seqLength - token;
            const Batch batch{valueHead, head.queryKeyHead, subBlock * HALF_DIM, stateRows,
                remaining < STREAM_BATCH_TOKENS ? remaining : STREAM_BATCH_TOKENS,
                context_.task.tokenStart + token};
            // 单批缓冲按 MTE2 -> V -> MTE3 复用；批间无预取，状态始终驻留 UB。
            CopyStreamingInputsGmToUb(batch, buffers);
            EvaluateStreamingBatch(batch, buffers);
            CopyStreamingOutputUbToGm(batch, buffers);
        }
        // 与 chunk 路径一致，BF16 接口仅在整段序列结束时量化主状态。
        if constexpr (!STATE_FP32) {
            Cast(buffers.stateBf16, buffers.stateFp32, RoundMode::CAST_RINT, stateRows * DIM);
            PipeBarrier<PIPE_V>();
        }
        Mutex::Unlock<PIPE_V>(stateMutex);
        CopyFinalStreamingStateUbToGm(head, buffers);
    }

private:
    __aicore__ inline Buffers BindStreamingBuffers()
    {
        // Streaming 与chunk互斥，独占当前AIV的UB，不使用L1；不能套用chunk路径的留白区。
        // UB布局：[begin,end) 单位KiB，T=STREAM_BATCH_TOKENS=64、D=128、H=HALF_DIM=64。
        // [  0, 32) stateFp32:[H,D] FP32     跨token常驻，MTE2/Cast -> V递推
        // [ 32, 48) stateBf16:[H,D] BF16     初/末态转换区，MTE2 -> V 或 V -> MTE3
        // [ 48, 64) query:[T,D] BF16         MTE2 -> V
        // [ 64, 80) key:[T,D] BF16           MTE2 -> V
        // [ 80, 96) valueBf16:[T,D] BF16     MTE2 -> V；最多H列有效，DMA行跨度仍为D
        // [ 96,128) valueFp32:[T,D] FP32     V Cast -> VF
        // [128,130) betaBf16:[T,16] BF16     MTE2，每token一个32B块，仅第0元素有效
        // [130,134) betaFp32:[T,16] FP32     V Cast -> VF，每token由32B扩大为64B
        // [134,136) g:[T,8] FP32             MTE2 -> VF，每token一个32B块，仅第0元素有效
        // [136,152) outFp32:[T,H] FP32       VF -> V Cast
        // [152,160) outBf16:[T,H] BF16       V -> MTE3
        // [160,248) 当前路径未使用
        // 偏移单位为字节，逐区使用 next=begin+shape*sizeof(T)，不需要额外槽间留白。
        // 批内各区同时存在；整批按MTE2 -> V -> MTE3交接，完成后才覆盖下一批输入/输出。
        // 行步长按32B数据块和固定物理D/H推导，不是按逻辑dk/dv紧凑排列。
        constexpr uint32_t stateFp32Offset = 0;
        constexpr uint32_t stateBf16Offset = stateFp32Offset + STATE_SLICE_ELEMS * sizeof(float);
        constexpr uint32_t queryOffset = stateBf16Offset + STATE_SLICE_ELEMS * sizeof(bfloat16_t);
        constexpr uint32_t keyOffset = queryOffset + STREAM_BATCH_TOKENS * DIM * sizeof(bfloat16_t);
        constexpr uint32_t valueBf16Offset = keyOffset + STREAM_BATCH_TOKENS * DIM * sizeof(bfloat16_t);
        constexpr uint32_t valueFp32Offset = valueBf16Offset + STREAM_BATCH_TOKENS * DIM * sizeof(bfloat16_t);
        constexpr uint32_t betaBf16Offset = valueFp32Offset + STREAM_BATCH_TOKENS * DIM * sizeof(float);
        // beta 按每 token 一个 32B BF16 块搬入，Cast 后每 token 占两个 32B FP32 块。
        constexpr uint32_t betaFp32Offset = betaBf16Offset + STREAM_BATCH_TOKENS * BLOCK_SIZE;
        constexpr uint32_t gateOffset = betaFp32Offset + STREAM_BATCH_TOKENS * 2 * BLOCK_SIZE;
        constexpr uint32_t outFp32Offset = gateOffset + STREAM_BATCH_TOKENS * BLOCK_SIZE;
        constexpr uint32_t outBf16Offset = outFp32Offset + STREAM_BATCH_TOKENS * HALF_DIM * sizeof(float);
        constexpr uint32_t endOffset = outBf16Offset + STREAM_BATCH_TOKENS * HALF_DIM * sizeof(bfloat16_t);
        static_assert(endOffset <= UB_WORKSPACE_BYTES, "arch35 streaming UB layout exceeds workspace capacity");

        return {BufferViews::Ub<float>(stateFp32Offset),
                BufferViews::Ub<bfloat16_t>(stateBf16Offset),
                BufferViews::Ub<bfloat16_t>(queryOffset),
                BufferViews::Ub<bfloat16_t>(keyOffset),
                BufferViews::Ub<bfloat16_t>(valueBf16Offset),
                BufferViews::Ub<float>(valueFp32Offset),
                BufferViews::Ub<bfloat16_t>(betaBf16Offset),
                BufferViews::Ub<float>(betaFp32Offset),
                BufferViews::Ub<float>(gateOffset),
                BufferViews::Ub<float>(outFp32Offset),
                BufferViews::Ub<bfloat16_t>(outBf16Offset)};
    }

    // 每个 token 取一个 beta/g 标量，放入独立 32B 数据块，供 VF 广播读取。
    template <typename T>
    __aicore__ inline void CopyTokenScalarsGmToUb(LocalTensor<T> dst, GlobalTensor<T> src, uint32_t tokenCount)
    {
        constexpr uint32_t elementsPerBlock = BLOCK_SIZE / sizeof(T);
        DataCopyExtParams copy{static_cast<uint16_t>(tokenCount), sizeof(T),
                               static_cast<int64_t>((context_.params.nv - 1) * sizeof(T)), 0, 0};
        DataCopyPadExtParams<T> pad{true, 0, elementsPerBlock - 1, static_cast<T>(0)};
        DataCopyPad(dst, src, copy, pad);
    }

    __aicore__ inline void CopyOutputRowsUbToGm(GlobalTensor<bfloat16_t> dst, LocalTensor<bfloat16_t> src,
                                               uint32_t rows, uint32_t count)
    {
        // UB rows have HALF_DIM elements; GM rows have nv * dv elements.
        DataCopyExtParams params{static_cast<uint16_t>(rows), static_cast<uint32_t>(count * sizeof(bfloat16_t)),
                                 (HALF_DIM - MmAlign(count, MM_NZ_FRACTAL)) / MM_NZ_FRACTAL,
                                 static_cast<int64_t>((context_.params.nv * context_.params.dv - count) * sizeof(bfloat16_t)), 0};
        DataCopyPad(dst, src, params);
    }

    // Streaming 主状态为 FP32 [HALF_DIM,DIM]；Run 已排除空序列和无有效 Dv 行的切片。
    // 清零物理状态区并加载初态；后续类型转换及递推的 V 同步由 Run 管理。
    __aicore__ inline void CopyInitialStreamingStateGmToUb(const Head &head, const Buffers &buffers)
    {
        const uint32_t stateRowBegin = head.subBlock * HALF_DIM;
        const uint32_t validRows = context_.StateValidRows(head.subBlock);
        const MutexID stateMutex = MutexId(STATE_MUTEX);
        Mutex::Lock<PIPE_V>(stateMutex);
        Duplicate(buffers.stateFp32, 0.0f, STATE_SLICE_ELEMS);
        PipeBarrier<PIPE_V>();
        Mutex::Unlock<PIPE_V>(stateMutex);

        const uint64_t initialStateOffset = context_.InitialStateOffset(head.valueHead, stateRowBegin);
        Mutex::Lock<PIPE_MTE2>(stateMutex);
        if constexpr (STATE_FP32) {
            CopyRowsGmToUb(buffers.stateFp32, context_.params.initialState[initialStateOffset], validRows,
                          context_.params.dk, context_.params.dk);
        } else {
            CopyRowsGmToUb(buffers.stateBf16, context_.params.initialState[initialStateOffset], validRows,
                          context_.params.dk, context_.params.dk);
        }
        Mutex::Unlock<PIPE_MTE2>(stateMutex);
    }

    __aicore__ inline void CopyFinalStreamingStateUbToGm(const Head &head, const Buffers &buffers)
    {
        const uint32_t stateRowBegin = head.subBlock * HALF_DIM;
        const uint32_t validRows = context_.StateValidRows(head.subBlock);
        const MutexID stateMutex = MutexId(STATE_MUTEX);
        // Run 已完成最终量化并释放 V；此处只管理 MTE3 写回。
        Mutex::Lock<PIPE_MTE3>(stateMutex);
        const uint64_t finalStateOffset = context_.FinalStateOffset(head.valueHead, stateRowBegin);
        if constexpr (STATE_FP32) {
            CopyStateRowsUbToGm(context_.params.finalState[finalStateOffset], buffers.stateFp32, validRows, context_.params.dk);
        } else {
            CopyStateRowsUbToGm(context_.params.finalState[finalStateOffset], buffers.stateBf16, validRows, context_.params.dk);
        }
        Mutex::Unlock<PIPE_MTE3>(stateMutex);
    }

    __aicore__ inline void CopyStreamingInputsGmToUb(const Batch &batch, const Buffers &buffers)
    {
        const uint64_t qkOffset = batch.globalToken * context_.params.nk * context_.params.dk +
                                  static_cast<uint64_t>(batch.queryKeyHead) * context_.params.dk;
        const uint64_t valueOffset = batch.globalToken * context_.params.nv * context_.params.dv +
                                     static_cast<uint64_t>(batch.valueHead) * context_.params.dv + batch.stateRowBegin;
        const uint64_t scalarOffset = batch.globalToken * context_.params.nv + batch.valueHead;
        const MutexID tokenMutex = MutexId(INPUT_MUTEX);
        Mutex::Lock<PIPE_MTE2>(tokenMutex);
        CopyRowsGmToUb(buffers.query, context_.params.query[qkOffset], batch.tokenCount, context_.params.dk, context_.params.nk * context_.params.dk);
        CopyRowsGmToUb(buffers.key, context_.params.key[qkOffset], batch.tokenCount, context_.params.dk, context_.params.nk * context_.params.dk);
        CopyRowsGmToUb(buffers.valueBf16, context_.params.value[valueOffset], batch.tokenCount, batch.stateRows, context_.params.nv * context_.params.dv);
        CopyTokenScalarsGmToUb(buffers.betaBf16, context_.params.beta[scalarOffset], batch.tokenCount);
        if constexpr (hasGamma) {
            CopyTokenScalarsGmToUb(buffers.g, context_.params.g[scalarOffset], batch.tokenCount);
        }
        Mutex::Unlock<PIPE_MTE2>(tokenMutex);
    }

    __aicore__ inline void EvaluateStreamingBatch(const Batch &batch, const Buffers &buffers)
    {
        // 每行计算 r=beta*(v-exp(g)*dot(S,k))；S=exp(g)*S+r*k；o=dot(S,scale*q)。
        // 优化方案：Dk<=8 将 8 个独立 Dv 行打包进寄存器；其余维度按行处理，状态跨 token 驻留。
        const MutexID tokenMutex = MutexId(INPUT_MUTEX);
        Mutex::Lock<PIPE_V>(tokenMutex);
        Cast(buffers.valueFp32, buffers.valueBf16, RoundMode::CAST_NONE, batch.tokenCount * DIM);
        Cast(buffers.betaFp32, buffers.betaBf16, RoundMode::CAST_NONE, batch.tokenCount * BLOCK_SIZE / sizeof(bfloat16_t));
        Duplicate(buffers.outFp32, 0.0f, batch.tokenCount * HALF_DIM);
        PipeBarrier<PIPE_V>();
        auto stateAddr = UbAddress(buffers.stateFp32);
        auto queryAddr = UbAddress(buffers.query);
        auto keyAddr = UbAddress(buffers.key);
        auto valueAddr = UbAddress(buffers.valueFp32);
        auto betaAddr = UbAddress(buffers.betaFp32);
        auto gAddr = UbAddress(buffers.g);
        auto outAddr = UbAddress(buffers.outFp32);
        if (context_.params.dk <= BLOCK_SIZE / sizeof(float)) {
            AscendC::VF_CALL<StreamingExactPackedRowsVF<hasGamma>>(
                stateAddr, queryAddr, keyAddr, valueAddr, betaAddr, gAddr, outAddr, context_.params.scale,
                static_cast<uint16_t>(batch.stateRows), static_cast<uint16_t>(context_.params.dk),
                static_cast<uint16_t>(DIM), static_cast<uint16_t>(batch.tokenCount));
        } else {
            AscendC::VF_CALL<StreamingExactBatchVF<hasGamma>>(
                stateAddr, queryAddr, keyAddr, valueAddr, betaAddr, gAddr, outAddr, context_.params.scale,
                static_cast<uint16_t>(batch.stateRows), static_cast<uint16_t>(context_.params.dk),
                static_cast<uint16_t>(DIM), static_cast<uint16_t>(batch.tokenCount));
        }
        PipeBarrier<PIPE_V>();
        Cast(buffers.outBf16, buffers.outFp32, RoundMode::CAST_RINT, batch.tokenCount * HALF_DIM);
        PipeBarrier<PIPE_V>();
        Mutex::Unlock<PIPE_V>(tokenMutex);
    }

    __aicore__ inline void CopyStreamingOutputUbToGm(const Batch &batch, const Buffers &buffers)
    {
        const MutexID tokenMutex = MutexId(INPUT_MUTEX);
        Mutex::Lock<PIPE_MTE3>(tokenMutex);
        const uint64_t outOffset =
            batch.globalToken * context_.params.nv * context_.params.dv + static_cast<uint64_t>(batch.valueHead) * context_.params.dv + batch.stateRowBegin;
        CopyOutputRowsUbToGm(context_.params.out[outOffset], buffers.outBf16, batch.tokenCount, batch.stateRows);
        Mutex::Unlock<PIPE_MTE3>(tokenMutex);
    }

    __aicore__ inline void CopyEmptyStateGmToUbToGm(uint32_t nvId, uint32_t subBlockIdx)
    {
        // L=0 时 finalState=initialState。初态支持 batch/head stride（包括 0），终态按连续布局写回。
        const uint32_t stateRowBegin = subBlockIdx * HALF_DIM;
        const uint32_t stateValidRows = context_.StateValidRows(subBlockIdx);
        if (stateValidRows == 0) {
            return;
        }
        const uint64_t initialStateOffset = context_.InitialStateOffset(nvId, stateRowBegin);
        const uint64_t finalStateOffset = context_.FinalStateOffset(nvId, stateRowBegin);
        const MutexID stateMutex = MutexId(STATE_MUTEX);
        constexpr uint32_t stateOffset = 0;
        auto state = BufferViews::Ub<stateType>(stateOffset);

        Mutex::Lock<PIPE_MTE2>(stateMutex);
        CopyRowsGmToUb(state, context_.params.initialState[initialStateOffset], stateValidRows,
                      context_.params.dk, context_.params.dk);
        Mutex::Unlock<PIPE_MTE2>(stateMutex);
        Mutex::Lock<PIPE_MTE3>(stateMutex);
        CopyStateRowsUbToGm(context_.params.finalState[finalStateOffset], state, stateValidRows, context_.params.dk);
        Mutex::Unlock<PIPE_MTE3>(stateMutex);
    }

    KernelContext<stateType> &context_;
};

} // namespace ChunkGatedDeltaRule

#endif // CHUNK_GATED_DELTA_RULE_VECTOR_STREAM_COMPUTE_H
