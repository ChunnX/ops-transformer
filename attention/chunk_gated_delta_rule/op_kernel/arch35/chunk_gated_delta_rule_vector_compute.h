/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHUNK_GATED_DELTA_RULE_VECTOR_COMPUTE_H
#define CHUNK_GATED_DELTA_RULE_VECTOR_COMPUTE_H

#include "chunk_gated_delta_rule_context.h"
#include "chunk_gated_delta_rule_sync.h"
#include "chunk_gated_delta_rule_vector_copy.h"
#include "vf/chunk_gated_delta_rule_stage1_vf.h"
#include "vf/chunk_gated_delta_rule_stage2_vf.h"

namespace ChunkGatedDeltaRule {

using namespace AscendC;

// AIV chunk 路径：本类组织前缀/求逆/状态递推及其搬运；工作集不跨计算类传递。
// Streaming 的 UB 布局和逐 token 递推由独立的 VectorStreamCompute 管理。
template <typename stateType, bool hasGamma>
class VectorCompute : private FusedLayout {
private:
    struct VectorHead {
        uint32_t valueHead;
        uint32_t queryKeyHead;
        uint32_t subBlock; // 0/1：前缀沿 token 分片，主状态沿 Dv 分片
    };

    struct VectorBuffers {
        LocalTensor<float> state;          // [HALF_DIM,DIM]，FP32 主状态
        LocalTensor<float> lower;          // [HALF_CHUNK,CHUNK]，严格下三角 L
        LocalTensor<float> inverse;        // [HALF_CHUNK,CHUNK]，对角逆块及 -L10
        LocalTensor<float> identity;       // [HALF_CHUNK,CHUNK]，对角求逆单位行
        LocalTensor<bfloat16_t> query;     // [HALF_CHUNK,DIM]
        LocalTensor<bfloat16_t> key;       // [HALF_CHUNK,DIM]
        LocalTensor<bfloat16_t> value;     // [HALF_CHUNK,DIM]
        LocalTensor<float> expGate;        // [CHUNK]，exp(G)
        LocalTensor<float> beta;           // [HALF_CHUNK]
    };

    struct PrefixBuffers {
        uint32_t slot;
        LocalTensor<float> cumulativeGate;           // [CHUNK]，本 chunk 的 G，不可被下块覆盖
        LocalTensor<bfloat16_t> transforms;          // 四个 [HALF_CHUNK,DIM] NZ 矩阵连续存放
        LocalTensor<bfloat16_t> rawStateProjection;  // -beta*exp(G)*K
        LocalTensor<bfloat16_t> weightedValue;       // beta*V
        LocalTensor<bfloat16_t> gatedQuery;          // scale*exp(G)*Q
        LocalTensor<bfloat16_t> decayedKey;          // exp(G_last-G)*K
        LocalTensor<bfloat16_t> inverse;             // [HALF_CHUNK,CHUNK] NZ
        LocalTensor<bfloat16_t> causalScores;        // [HALF_CHUNK,CHUNK] NZ
    };

public:
    __aicore__ inline explicit VectorCompute(KernelContext<stateType> &context) : context_(context) {}

    __aicore__ inline void Run(uint32_t valueHead, uint32_t subBlock)
    {
        const VectorHead head{valueHead, context_.QueryKeyHead(valueHead), subBlock};
        const auto buffers = BindVectorBuffers();
        InitializeVectorConstants(buffers);
        // 等待 S 写完 identity/gather；整个 head 的 V 使用期由本层成对管理。
        const MutexID scalarMutex = MutexId(AIV_SCALAR_MUTEX);
        Mutex::Lock<PIPE_V>(scalarMutex);
        CopyInitialStateGmToUb(head, buffers);
        PrepareVectorPrefix<false>(head, 0, buffers);
        Mutex::Lock<PIPE_V>(MutexId(AIV_STATE_MUTEX));
        CopyStateUbToL1(subBlock, buffers.state);
        for (uint32_t chunk = 0; chunk < context_.ChunkNum(); ++chunk) {
            ComputeVectorChunk(head, chunk, buffers);
        }
        CopyFinalStateUbToGm(head, buffers);
        Mutex::Unlock<PIPE_V>(scalarMutex);
    }

private:
    // Chunk 路径每个 AIV 的 UB 布局：[begin,end) 单位 KiB，C=64、D=128。
    // 区间          内容 / 物理shape                         流水或生命周期
    // [  0, 32)     FP32 S:[D/2,D]                           MTE2 -> V -> MTE3，跨chunk常驻
    // [ 32, 40)     当前路径未使用，沿用8KiB留白
    // [ 40, 48)     FP32 lower:[C/2,C]                       V，构造L/对角求逆
    // [ 48, 56)     FP32 inverse:[C/2,C]                     V，求逆结果及-L10
    // [ 56, 64)     FP32 identity:[C/2,C]                    S初始化 -> V，只读单位行
    // [ 64, 68)     BF16 M槽0:[C/2,C]                       V -> MTE3
    // [ 68, 72)     BF16 M槽1:[C/2,C]                       V -> MTE3
    // [ 72, 88)     BF16 delta:[D/2,D]                       AIC FIX -> V，共享地址
    // [ 88, 92)     当前路径未使用，沿用4KiB留白
    // [ 92,100)     BF16 Q:[C/2,D]                           MTE2 -> V，按前/后16行交接
    // [100,108)     BF16 K:[C/2,D]                           MTE2 -> V
    // [108,116)     BF16 V:[C/2,D]                           MTE2 -> V
    // [116,132)     BF16 state副本:[D/2,D]                   初/末态转换或 V -> MTE3 -> L1
    // [132,134)     rawG:[C,8] FP32，每token独占32B块        MTE2 -> V Gather
    // [134,135)     rawBeta:[C/2,16] BF16，每token独占32B块  MTE2 -> V Gather
    // [135,136)     betaBF16槽，实用[C/2]*2B=64B             V Gather -> Cast
    // [136,137)     cumulativeGate槽0，实用[C]*4B=256B       V，G不可被下一chunk覆盖
    // [137,138)     cumulativeGate槽1，实用[C]*4B=256B       V
    // [138,139)     expGate槽，实用[C]*4B=256B               V
    // [139,140)     betaFP32槽，实用[C/2]*4B=128B            V
    // [140,160)     gather:[C] uint32只占256B，余19.75KiB留白 S初始化 -> V
    // [160,168)     KKT/QKT槽0；[168,172)/[172,176) inverse发布槽0/1
    // [176,208)     Stage1槽0；[208,240) Stage1槽1；[240,248) KKT/QKT槽1
    // 顶部共享区各shape、流水及Stage1阶段复用见 FusedLayout。
    // 1KiB标量槽和8/4KiB留白属于现有分区约定，未声称硬件需要这些对齐或有bank优化。
    // rawG/rawBeta的32B行跨度则来自当前DataCopyPad参数；不能仅按标量有效字节计算占用。
    // 以上“留白”仅针对chunk路径；Streaming有独立布局，会使用其中部分地址。
    // 单处使用的起点仍在相应Bind函数内，跨方法使用的边界留在本类。
    static constexpr uint32_t UB_STAGE1_MATRIX_BYTES = HALF_CHUNK * DIM * sizeof(bfloat16_t);
    static constexpr uint32_t UB_INPUT_GAP_BYTES = 4 * 1024; // delta之后、Q之前的既有留白
    static constexpr uint32_t UB_SCALAR_SLOT_BYTES = 1024; // 标量向量的既有槽容量，非有效数据大小
    static constexpr uint32_t UB_MASKED_QKT_PUB_OFFSET =
        UB_STATE_DELTA_BF16_OFFSET - 2 * SCORE_SLICE_ELEMS * sizeof(bfloat16_t);
    static constexpr uint32_t UB_STATE_PUB_OFFSET = UB_STATE_DELTA_BF16_OFFSET +
        STATE_SLICE_ELEMS * sizeof(bfloat16_t) + UB_INPUT_GAP_BYTES + 3 * UB_STAGE1_MATRIX_BYTES;
    static constexpr uint32_t UB_RAW_G_OFFSET = UB_STATE_PUB_OFFSET + STATE_SLICE_ELEMS * sizeof(bfloat16_t);
    static constexpr uint32_t UB_RAW_BETA_OFFSET = UB_RAW_G_OFFSET + CHUNK * BLOCK_SIZE;
    static constexpr uint32_t UB_EXP_GATE_OFFSET = UB_RAW_BETA_OFFSET + HALF_CHUNK * BLOCK_SIZE +
        3 * UB_SCALAR_SLOT_BYTES; // 跳过紧凑betaBF16、G0、G1三个槽
    static constexpr uint32_t UB_GATHER_OFFSET = UB_EXP_GATE_OFFSET + 2 * UB_SCALAR_SLOT_BYTES; // expGate、betaFP32
    static constexpr uint32_t INPUT_BATCH_ROWS = HALF_CHUNK / 2;

    // 每条 AIV 路径独立使用 0..7；chunk 与 streaming 互斥，可复用相同编号。
    static constexpr uint32_t AIV_INPUT_HEAD_MUTEX = 0;
    static constexpr uint32_t AIV_INPUT_TAIL_MUTEX = 1;
    static constexpr uint32_t AIV_STATE_MUTEX = 2;
    static constexpr uint32_t AIV_STAGE1_PUB_MUTEX_BASE = 3;
    static constexpr uint32_t AIV_PREFIX_PUB_MUTEX_BASE = 5;
    static constexpr uint32_t AIV_SCALAR_MUTEX = 7;

    static_assert(AIV_SCALAR_MUTEX < MUTEX_COUNT, "Vector mutex ID exceeds hardware capacity");
    static_assert(UB_MASKED_QKT_PUB_OFFSET + 2 * SCORE_SLICE_ELEMS * sizeof(bfloat16_t) <= UB_STATE_DELTA_BF16_OFFSET,
                  "maskedQKT staging overlaps shared BF16 delta");
    static_assert(UB_STATE_PUB_OFFSET + STATE_SLICE_ELEMS * sizeof(bfloat16_t) <= UB_RAW_G_OFFSET,
                  "state staging overlaps gate input");
    static_assert(UB_GATHER_OFFSET + CHUNK * sizeof(uint32_t) <= UB_PREFIX_DIRECT0_OFFSET,
                  "gather indices overlap shared prefix scores");
    static_assert(4 * UB_STAGE1_MATRIX_BYTES <= UB_STAGE1_SLOT_BYTES,
                  "four prefix transforms exceed their shared Stage1 slot");
    static_assert(CHUNK * sizeof(float) <= UB_SCALAR_SLOT_BYTES &&
                  HALF_CHUNK * sizeof(float) <= UB_SCALAR_SLOT_BYTES,
                  "gate/beta vectors exceed their reserved scalar slots");

    // 每个 AIV 的固定工作集；偏移只在此处使用，shape 和边界与绑定放在一起。
    __aicore__ inline VectorBuffers BindVectorBuffers()
    {
        constexpr uint32_t stateOffset = 0;
        constexpr uint32_t lowerOffset = stateOffset + STATE_SLICE_ELEMS * sizeof(float) + 8 * 1024; // 保留8KiB
        constexpr uint32_t inverseOffset = lowerOffset + SCORE_SLICE_ELEMS * sizeof(float);
        constexpr uint32_t identityOffset = inverseOffset + SCORE_SLICE_ELEMS * sizeof(float);
        constexpr uint32_t queryOffset = UB_STATE_DELTA_BF16_OFFSET +
            STATE_SLICE_ELEMS * sizeof(bfloat16_t) + UB_INPUT_GAP_BYTES;
        constexpr uint32_t keyOffset = queryOffset + HALF_CHUNK * DIM * sizeof(bfloat16_t);
        constexpr uint32_t valueOffset = keyOffset + HALF_CHUNK * DIM * sizeof(bfloat16_t);
        constexpr uint32_t betaOffset = UB_EXP_GATE_OFFSET + UB_SCALAR_SLOT_BYTES;
        static_assert(stateOffset + STATE_SLICE_ELEMS * sizeof(float) <= lowerOffset,
                      "FP32 state overlaps attention scratch");
        static_assert(lowerOffset + SCORE_SLICE_ELEMS * sizeof(float) <= inverseOffset &&
                      inverseOffset + SCORE_SLICE_ELEMS * sizeof(float) <= identityOffset &&
                      identityOffset + SCORE_SLICE_ELEMS * sizeof(float) <= UB_MASKED_QKT_PUB_OFFSET,
                      "attention/inverse/identity scratch overlaps its next region");
        static_assert(UB_STATE_DELTA_BF16_OFFSET + STATE_SLICE_ELEMS * sizeof(bfloat16_t) <= queryOffset,
                      "shared BF16 delta overlaps Q input");
        static_assert(queryOffset + HALF_CHUNK * DIM * sizeof(bfloat16_t) <= keyOffset &&
                      keyOffset + HALF_CHUNK * DIM * sizeof(bfloat16_t) <= valueOffset &&
                      valueOffset + HALF_CHUNK * DIM * sizeof(bfloat16_t) <= UB_STATE_PUB_OFFSET,
                      "Q/K/V inputs overlap their next region");
        static_assert(UB_EXP_GATE_OFFSET + CHUNK * sizeof(float) <= betaOffset &&
                      betaOffset + HALF_CHUNK * sizeof(float) <= UB_GATHER_OFFSET,
                      "gate/beta scratch overlaps its next region");
        return {BufferViews::Ub<float>(stateOffset), BufferViews::Ub<float>(lowerOffset),
                BufferViews::Ub<float>(inverseOffset), BufferViews::Ub<float>(identityOffset),
                BufferViews::Ub<bfloat16_t>(queryOffset), BufferViews::Ub<bfloat16_t>(keyOffset),
                BufferViews::Ub<bfloat16_t>(valueOffset), BufferViews::Ub<float>(UB_EXP_GATE_OFFSET),
                BufferViews::Ub<float>(betaOffset)};
    }

    // G 与 inverse/score 发布区各有两个槽，避免预取下块时覆盖当前 chunk 的递推输入。
    __aicore__ inline PrefixBuffers BindPrefixBuffers(uint32_t chunk)
    {
        constexpr uint32_t cumulativeGate0Offset =
            UB_RAW_BETA_OFFSET + HALF_CHUNK * BLOCK_SIZE + UB_SCALAR_SLOT_BYTES; // rawBeta、紧凑betaBF16之后
        constexpr uint32_t cumulativeGate1Offset = cumulativeGate0Offset + UB_SCALAR_SLOT_BYTES;
        constexpr uint32_t inverse0Offset = UB_PREFIX_DIRECT0_OFFSET + UB_PREFIX_DIRECT_BYTES;
        constexpr uint32_t inverse1Offset = inverse0Offset + SCORE_SLICE_ELEMS * sizeof(bfloat16_t);
        static_assert(cumulativeGate0Offset + CHUNK * sizeof(float) <= cumulativeGate1Offset &&
                      cumulativeGate1Offset + CHUNK * sizeof(float) <= UB_EXP_GATE_OFFSET,
                      "cumulative gates overlap each other or exp(g)");
        static_assert(UB_PREFIX_DIRECT0_OFFSET + UB_PREFIX_DIRECT_BYTES <= inverse0Offset,
                      "shared prefix scores overlap inverse staging");
        static_assert(inverse0Offset + SCORE_SLICE_ELEMS * sizeof(bfloat16_t) <= inverse1Offset &&
                      inverse1Offset + SCORE_SLICE_ELEMS * sizeof(bfloat16_t) <= UB_STAGE1_PUB0_OFFSET,
                      "inverse staging overlaps its next slot or shared Stage1");
        const uint32_t slot = chunk & 1;
        auto transforms = BufferViews::Ub<bfloat16_t>(PrefixTransformsUbOffset(slot));
        constexpr uint32_t matrixElements = UB_STAGE1_MATRIX_BYTES / sizeof(bfloat16_t);
        return {slot,
                BufferViews::Ub<float>(slot == 0 ? cumulativeGate0Offset : cumulativeGate1Offset),
                transforms, transforms, transforms[matrixElements], transforms[2 * matrixElements],
                transforms[3 * matrixElements],
                BufferViews::Ub<bfloat16_t>(slot == 0 ? inverse0Offset : inverse1Offset),
                BufferViews::Ub<bfloat16_t>(UB_MASKED_QKT_PUB_OFFSET + slot * SCORE_SLICE_ELEMS * sizeof(bfloat16_t))};
    }

    template <bool headPrefetched>
    __aicore__ inline void PrepareVectorPrefix(const VectorHead &head, uint32_t chunk, const VectorBuffers &buffers)
    {
        const auto prefix = BindPrefixBuffers(chunk);
        WaitForCube<CubeToVectorEvent::ChunkSlotFree>(prefix.slot);
        PrepareChunkInputs<headPrefetched>(head, chunk, buffers, prefix); // G=cumsum(g)，beta 整理为连续向量
        TransformPrefix(head, chunk, buffers, prefix);                  // rawKcd/vBeta/Q'/Kg
        CopyPrefixTransformsUbToL1(head, prefix);                        // 不依赖分数，提前送往 L1
        BuildCausalInverse(head, buffers, prefix);                      // L/M 与两个 32x32 对角逆块
        CopyInverseAndScoresUbToL1(head, prefix);
    }

    __aicore__ inline void ComputeVectorChunk(const VectorHead &head, uint32_t chunk, const VectorBuffers &buffers)
    {
        CopyResidualUbToL1(chunk & 1, head.subBlock);          // R32 -> BF16 Cube 副本
        if (chunk + 1 < context_.ChunkNum()) {
            PrepareVectorPrefix<true>(head, chunk + 1, buffers); // 提前准备下块，与 Cube 递推重叠
        }
        UpdateChunkState(head, chunk, buffers);              // S=exp(G_last)*S+delta，并发布下一块的 S
    }

    template <uint32_t rows, uint32_t cols, uint32_t fullRows, uint32_t matrixCount = 1>
    __aicore__ inline void CopyCompactNzUbToL1(LocalTensor<bfloat16_t> src, LocalTensor<bfloat16_t> dst,
                                             uint32_t rowBegin)
    {
        // MTE3：UB [matrixCount,rows,cols] -> L1[:,rowBegin:rowBegin+rows,:]。
        // NZ 布局为 [matrixCount,cols/16,rows,16]，每行恰好一个 32B 块；L1 行跨度为 fullRows。
        // 优化方案：相邻矩阵合并为一次 DMA；静态模板参数集中推导块数/长度/间隔。
        static_assert(rows % MM_BLOCK_CUBE == 0 && fullRows % MM_BLOCK_CUBE == 0,
                      "NZ rows must be fractal aligned");
        static_assert(cols % MM_NZ_FRACTAL == 0 && rows <= fullRows,
                      "the aligned UB tile must fit into the L1 matrix");
        DataCopyParams params{static_cast<uint16_t>(matrixCount * cols / MM_NZ_FRACTAL),
                              static_cast<uint16_t>(rows), 0, static_cast<uint16_t>(fullRows - rows)};
        DataCopy(dst[GetZnOffset(rowBegin, 0, fullRows)], src, params);
    }

    // Q/K:[T,Nk,Dk]、V:[T,Nv,Dv] -> 当前 AIV 的 [HALF_CHUNK,DIM] UB。
    // localRow 指明前/后 16 行；调用方持有对应输入 Mutex，搬运后由消费者处理无效尾部。
    __aicore__ inline void CopyQkvRowsGmToUb(const VectorHead &head, uint32_t chunk, const VectorBuffers &buffers,
                                       uint32_t localRow, uint32_t validRows)
    {
        const uint64_t token = context_.task.tokenStart + static_cast<uint64_t>(chunk) * CHUNK +
                               head.subBlock * HALF_CHUNK + localRow;
        const uint64_t qkOffset = token * context_.params.nk * context_.params.dk + head.queryKeyHead * context_.params.dk;
        const uint64_t valueOffset = token * context_.params.nv * context_.params.dv +
                                     static_cast<uint64_t>(head.valueHead) * context_.params.dv;
        CopyRowsGmToUb(buffers.query[localRow * DIM], context_.params.query[qkOffset], validRows,
                        context_.params.dk, context_.params.nk * context_.params.dk);
        CopyRowsGmToUb(buffers.key[localRow * DIM], context_.params.key[qkOffset], validRows,
                        context_.params.dk, context_.params.nk * context_.params.dk);
        CopyRowsGmToUb(buffers.value[localRow * DIM], context_.params.value[valueOffset], validRows,
                        context_.params.dv, context_.params.nv * context_.params.dv);
    }

    __aicore__ inline void CopyInputHeadGmToUb(const VectorHead &head, uint32_t chunk, const VectorBuffers &buffers)
    {
        const uint32_t tokenBegin = head.subBlock * HALF_CHUNK;
        const uint32_t validRows = context_.ChunkValidRows(chunk);
        const uint32_t localValidRows = context_.SubBlockValidRows(chunk, head.subBlock);
        const uint32_t headValidRows = localValidRows < INPUT_BATCH_ROWS ? localValidRows : INPUT_BATCH_ROWS;
        const uint64_t tokenStart = context_.task.tokenStart + static_cast<uint64_t>(chunk) * CHUNK;
        auto rawG = BufferViews::Ub<float>(UB_RAW_G_OFFSET);
        auto rawBeta = BufferViews::Ub<bfloat16_t>(UB_RAW_BETA_OFFSET);
        const MutexID inputHeadMutex = MutexId(AIV_INPUT_HEAD_MUTEX);
        Mutex::Lock<PIPE_MTE2>(inputHeadMutex);
        if (headValidRows > 0) {
            CopyQkvRowsGmToUb(head, chunk, buffers, 0, headValidRows);
        }

        if constexpr (hasGamma) {
            DataCopyExtParams gParams{static_cast<uint16_t>(validRows), static_cast<uint32_t>(sizeof(float)),
                                      static_cast<uint32_t>((context_.params.nv - 1) * sizeof(float)), 0, 0};
            DataCopyPadExtParams<float> gPad{true, 0, 7, 0.0f};
            DataCopyPad(rawG, context_.params.g[tokenStart * context_.params.nv + head.valueHead], gParams, gPad);
        }
        if (localValidRows > 0) {
            DataCopyExtParams betaParams{static_cast<uint16_t>(localValidRows),
                                         static_cast<uint32_t>(sizeof(bfloat16_t)),
                                         static_cast<uint32_t>((context_.params.nv - 1) * sizeof(bfloat16_t)), 0, 0};
            DataCopyPadExtParams<bfloat16_t> betaPad{true, 0, 15, static_cast<bfloat16_t>(0)};
            DataCopyPad(rawBeta, context_.params.beta[(tokenStart + tokenBegin) * context_.params.nv + head.valueHead], betaParams, betaPad);
        }
        Mutex::Unlock<PIPE_MTE2>(inputHeadMutex);
    }

    template <bool headPrefetched>
    __aicore__ inline void PrepareChunkInputs(const VectorHead &head, uint32_t chunk, const VectorBuffers &buffers,
                                              const PrefixBuffers &prefix)
    {
        auto gCum = prefix.cumulativeGate;
        auto gExp = buffers.expGate;
        auto beta = buffers.beta;
        const uint32_t validRows = context_.ChunkValidRows(chunk);
        const uint32_t localValidRows = context_.SubBlockValidRows(chunk, head.subBlock);
        const uint32_t tailValidRows = localValidRows > INPUT_BATCH_ROWS ? localValidRows - INPUT_BATCH_ROWS : 0;
        auto rawG = BufferViews::Ub<float>(UB_RAW_G_OFFSET);
        auto rawBeta = BufferViews::Ub<bfloat16_t>(UB_RAW_BETA_OFFSET);
        constexpr uint32_t betaBf16Offset = UB_RAW_BETA_OFFSET + HALF_CHUNK * BLOCK_SIZE;
        auto betaBf16 = BufferViews::Ub<bfloat16_t>(betaBf16Offset);
        auto gather = BufferViews::Ub<uint32_t>(UB_GATHER_OFFSET);

        if constexpr (!headPrefetched) {
            CopyInputHeadGmToUb(head, chunk, buffers);
        }

        const MutexID inputHeadMutex = MutexId(AIV_INPUT_HEAD_MUTEX);
        const MutexID inputTailMutex = MutexId(AIV_INPUT_TAIL_MUTEX);
        Mutex::Lock<PIPE_MTE2>(inputTailMutex);
        if (tailValidRows > 0) {
            CopyQkvRowsGmToUb(head, chunk, buffers, INPUT_BATCH_ROWS, tailValidRows);
        }
        Mutex::Unlock<PIPE_MTE2>(inputTailMutex);

        // g/beta 整理后继续持有前 16 行输入的 V 锁，TransformPrefix 消费后释放，才能预取下一块。
        // 尾 chunk 的 g 补 0，使 G[63] 等于真实末 token 的累积门控；beta 补 0 使补齐行没有更新量。
        Mutex::Lock<PIPE_V>(inputHeadMutex);
        if constexpr (hasGamma) {
            if (validRows < CHUNK) {
                Duplicate(gCum, 0.0f, CHUNK);
            }
        } else {
            Duplicate(gCum, 0.0f, CHUNK);
            Duplicate(gExp, 1.0f, CHUNK);
        }
        if (localValidRows < HALF_CHUNK) {
            Duplicate(betaBf16, static_cast<bfloat16_t>(0), HALF_CHUNK);
        }
        if (validRows < CHUNK) {
            PipeBarrier<PIPE_V>();
        }
        if constexpr (hasGamma) {
            Gather(gCum, rawG, gather, static_cast<uint32_t>(0), validRows);
        }
        if (localValidRows > 0) {
            Gather(betaBf16, rawBeta, gather, static_cast<uint32_t>(0), localValidRows);
        }
        PipeBarrier<PIPE_V>();
        Cast(beta, betaBf16, RoundMode::CAST_NONE, HALF_CHUNK);
        PipeBarrier<PIPE_V>();
        if constexpr (hasGamma) {
            auto gAddr = UbAddress(gCum);
            auto gExpAddr = UbAddress(gExp);
            AscendC::VF_CALL<CumSumExpExactVF>(gAddr, gAddr, gExpAddr, static_cast<uint16_t>(CHUNK));
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void BuildInverseSlice(uint32_t subBlockIdx, LocalTensor<float> attn, LocalTensor<float> inverse,
                                             LocalTensor<float> identity, LocalTensor<bfloat16_t> derived)
    {
        // UB attn/inverse:[C/2,C] FP32，derived 为同 shape 的 BF16 NZ 发布副本。
        // 每个 AIV 仅求自己的 32x32 对角块；inverse 其余位置保留初始化的 -L，供 AIC 拼出左下逆块。
        InverseAIVPanel8VF(attn, inverse, identity, subBlockIdx * HALF_CHUNK, CHUNK);
        PipeBarrier<PIPE_V>();
        auto inverseAddr = UbAddress(inverse);
        auto derivedAddr = UbAddress(derived);
        AscendC::VF_CALL<CastFp32ToCompactZnBf16VF>(inverseAddr, derivedAddr, static_cast<uint16_t>(HALF_CHUNK),
                                                    static_cast<uint16_t>(CHUNK));
        PipeBarrier<PIPE_V>();
    }

    template <uint32_t rowOffset>
    __aicore__ inline void TransformPrefixRows(const VectorHead &head, const VectorBuffers &buffers,
                                               const PrefixBuffers &prefix, uint32_t validRows)
    {
        // rawKcd=-beta*exp(G)*K，vBeta=beta*V，Q'=scale*exp(G)*Q，Kg=exp(G_last-G)*K。
        // 有效 shape 为 [validRows,Dk/Dv]，发布布局为 [HALF_CHUNK,DIM] compact NZ。
        auto query = UbAddress(buffers.query) + rowOffset * DIM;
        auto key = UbAddress(buffers.key) + rowOffset * DIM;
        auto value = UbAddress(buffers.value) + rowOffset * DIM;
        auto beta = UbAddress(buffers.beta) + rowOffset;
        const uint16_t tokenBegin = static_cast<uint16_t>(head.subBlock * HALF_CHUNK + rowOffset);
        if (context_.params.dk == DIM && context_.params.dv == DIM) {
            AscendC::VF_CALL<PrepareStage1FullDimSliceVF>(
                query, key, value, UbAddress(prefix.rawStateProjection), UbAddress(prefix.weightedValue),
                UbAddress(prefix.gatedQuery), UbAddress(prefix.decayedKey), beta,
                UbAddress(prefix.cumulativeGate), UbAddress(buffers.expGate), context_.params.scale, tokenBegin,
                static_cast<uint16_t>(validRows), static_cast<uint16_t>(DIM),
                static_cast<uint16_t>(rowOffset), static_cast<uint16_t>(HALF_CHUNK));
        } else {
            AscendC::VF_CALL<PrepareStage1SliceVF>(
                query, key, value, UbAddress(prefix.rawStateProjection), UbAddress(prefix.weightedValue),
                UbAddress(prefix.gatedQuery), UbAddress(prefix.decayedKey), beta,
                UbAddress(prefix.cumulativeGate), UbAddress(buffers.expGate), context_.params.scale, tokenBegin,
                static_cast<uint16_t>(validRows), static_cast<uint16_t>(context_.params.dk),
                static_cast<uint16_t>(context_.params.dv), static_cast<uint16_t>(DIM), static_cast<uint16_t>(DIM),
                static_cast<uint16_t>(rowOffset), static_cast<uint16_t>(HALF_CHUNK));
        }
    }

    __aicore__ inline void TransformPrefix(const VectorHead &head, uint32_t chunk, const VectorBuffers &buffers,
                                           const PrefixBuffers &prefix)
    {
        const uint32_t validRows = context_.SubBlockValidRows(chunk, head.subBlock);
        const uint32_t firstRows = validRows < INPUT_BATCH_ROWS ? validRows : INPUT_BATCH_ROWS;
        const uint32_t lastRows = validRows > INPUT_BATCH_ROWS ? validRows - INPUT_BATCH_ROWS : 0;
        const MutexID prefixMutex = MutexId(AIV_STAGE1_PUB_MUTEX_BASE + prefix.slot);
        Mutex::Lock<PIPE_V>(prefixMutex);
        if (validRows < HALF_CHUNK || context_.params.dk < DIM || context_.params.dv < DIM) {
            Duplicate(prefix.transforms, static_cast<bfloat16_t>(0), 4 * UB_STAGE1_MATRIX_BYTES / sizeof(bfloat16_t));
            PipeBarrier<PIPE_V>();
        }
        // PrepareChunkInputs 已持有前半输入的 V 锁。消费前 16 行后，马上复用其 UB 预取下一块。
        TransformPrefixRows<0>(head, buffers, prefix, firstRows);
        Mutex::Unlock<PIPE_V>(MutexId(AIV_INPUT_HEAD_MUTEX));
        if (chunk + 1 < context_.ChunkNum()) {
            CopyInputHeadGmToUb(head, chunk + 1, buffers);
        }
        Mutex::Lock<PIPE_V>(MutexId(AIV_INPUT_TAIL_MUTEX));
        TransformPrefixRows<INPUT_BATCH_ROWS>(head, buffers, prefix, lastRows);
        PipeBarrier<PIPE_V>();
        // 后 16 行输入消费完毕，再释放整个前缀输出区供 MTE3 搬运。
        Mutex::Unlock<PIPE_V>(MutexId(AIV_INPUT_TAIL_MUTEX));
        Mutex::Unlock<PIPE_V>(prefixMutex);
    }

    __aicore__ inline void CopyPrefixTransformsUbToL1(const VectorHead &head, const PrefixBuffers &prefix)
    {
        const uint32_t base = prefix.slot * L1_SLOT_BYTES;
        const uint32_t tokenBegin = head.subBlock * HALF_CHUNK;
        const MutexID prefixMutex = MutexId(AIV_STAGE1_PUB_MUTEX_BASE + prefix.slot);
        Mutex::Lock<PIPE_MTE3>(prefixMutex);
        // 优化方案：rawKcd/vBeta 在 UB 和 L1 均相邻，合并一次 DMA；不必等待 KKT/QKT。
        CopyCompactNzUbToL1<HALF_CHUNK, DIM, CHUNK, 2>(
            prefix.rawStateProjection, BufferViews::L1<bfloat16_t>(base + L1_RAW_KCD_OFFSET), tokenBegin);
        CopyCompactNzUbToL1<HALF_CHUNK, DIM, DIM>(
            prefix.gatedQuery, BufferViews::L1<bfloat16_t>(base + L1_STATE_PROJECTIONS_OFFSET), CHUNK + tokenBegin);
        CopyCompactNzUbToL1<HALF_CHUNK, DIM, CHUNK>(
            prefix.decayedKey, BufferViews::L1<bfloat16_t>(base + L1_KG_OFFSET), tokenBegin);
        Mutex::Unlock<PIPE_MTE3>(prefixMutex);
    }

    __aicore__ inline void BuildCausalInverse(const VectorHead &head, const VectorBuffers &buffers,
                                              const PrefixBuffers &prefix)
    {
        WaitForCube<CubeToVectorEvent::PrefixScoresReady>(prefix.slot);
        auto scores = BufferViews::Ub<bfloat16_t>(PrefixDirectOffset(prefix.slot));
        const MutexID prefixMutex = MutexId(AIV_PREFIX_PUB_MUTEX_BASE + prefix.slot);
        Mutex::Lock<PIPE_V>(prefixMutex);
        // L=tril(beta*KKT*exp(G_i-G_j),-1)，M=tril(scale*QKT*exp(G_i-G_j))；mask 的对角线不同。
        AscendC::VF_CALL<BuildAttnMaskedQktExactSliceVF>(
            UbAddress(buffers.lower), UbAddress(buffers.inverse), UbAddress(prefix.causalScores),
            UbAddress(scores), UbAddress(scores) + SCORE_SLICE_ELEMS, UbAddress(prefix.cumulativeGate),
            UbAddress(buffers.beta), context_.params.scale, static_cast<uint16_t>(head.subBlock * HALF_CHUNK),
            static_cast<uint16_t>(HALF_CHUNK), static_cast<uint16_t>(CHUNK));
        PipeBarrier<PIPE_V>();
        BuildInverseSlice(head.subBlock, buffers.lower, buffers.inverse, buffers.identity, prefix.inverse);
        NotifyCube<PIPE_V, VectorToCubeEvent::PrefixScoresBufferFree>(prefix.slot);
        Mutex::Unlock<PIPE_V>(prefixMutex);
    }

    __aicore__ inline void CopyInverseAndScoresUbToL1(const VectorHead &head, const PrefixBuffers &prefix)
    {
        const uint32_t base = prefix.slot * L1_SLOT_BYTES;
        const uint32_t tokenBegin = head.subBlock * HALF_CHUNK;
        const MutexID prefixMutex = MutexId(AIV_PREFIX_PUB_MUTEX_BASE + prefix.slot);
        Mutex::Lock<PIPE_MTE3>(prefixMutex);
        CopyCompactNzUbToL1<HALF_CHUNK, CHUNK, CHUNK>(
            prefix.inverse, BufferViews::L1<bfloat16_t>(base + L1_INVERSE_OFFSET), tokenBegin);
        // 优化方案：先发布递推关键输入，Cube 可在 M 搬运期间拼接 A 的左下逆块。
        NotifyCube<PIPE_MTE3, VectorToCubeEvent::InverseInputsReady>(prefix.slot);
        CopyCompactNzUbToL1<HALF_CHUNK, CHUNK, CHUNK>(
            prefix.causalScores, BufferViews::L1<bfloat16_t>(base + L1_MASKED_QKT_OFFSET), tokenBegin);
        NotifyCube<PIPE_MTE3, VectorToCubeEvent::CausalScoresReady>(prefix.slot);
        Mutex::Unlock<PIPE_MTE3>(prefixMutex);
    }

    __aicore__ inline void CopyStateUbToL1(uint32_t subBlockIdx, LocalTensor<float> state)
    {
        // 主状态为 row-major FP32 [HALF_DIM,DIM]；只将副本转 BF16 NZ 并合并进 L1 S:[DIM,DIM]。
        // 调用方已锁定 STATE 的 PIPE_V：首次由 Run 申请，后续由 UpdateChunkState 申请。
        // 转换完成后释放 V，再由 MTE3 读取同一缓冲区。
        const uint32_t stateRowBegin = subBlockIdx * HALF_DIM;
        const uint32_t validRows = context_.StateValidRows(subBlockIdx);
        auto statePub = BufferViews::Ub<bfloat16_t>(UB_STATE_PUB_OFFSET);
        auto l1State = BufferViews::L1<bfloat16_t>(L1_STATE_OFFSET);
        const MutexID stateMutex = MutexId(AIV_STATE_MUTEX);

        if (validRows < HALF_DIM || context_.params.dk < DIM) {
            Duplicate(statePub, static_cast<bfloat16_t>(0), STATE_SLICE_ELEMS);
            PipeBarrier<PIPE_V>();
        }
        AscendC::VF_CALL<RowMajorFp32ToCompactZnBf16VF>(
            UbAddress(state), UbAddress(statePub), static_cast<uint16_t>(validRows),
            static_cast<uint16_t>(context_.params.dk), static_cast<uint16_t>(DIM),
            static_cast<uint16_t>(HALF_DIM));
        PipeBarrier<PIPE_V>();
        Mutex::Unlock<PIPE_V>(stateMutex);
        Mutex::Lock<PIPE_MTE3>(stateMutex);
        CopyCompactNzUbToL1<HALF_DIM, DIM, DIM>(statePub, l1State, stateRowBegin);
        NotifyCube<PIPE_MTE3, VectorToCubeEvent::StateReady>();
        Mutex::Unlock<PIPE_MTE3>(stateMutex);
    }

    // R=U+Kcd*S^T:[C,Dv]；每 AIV 收到物理 [DIM/16,HALF_CHUNK,16] NZ 切片。
    // 优化方案：复用已消费的 Stage1 发布区，FP32 R 落 UB 后仅生成一个 BF16 Cube 输入副本。
    __aicore__ inline void CopyResidualUbToL1(uint32_t slot, uint32_t subBlockIdx)
    {
        // 当前 Stage1 槽已由 Cube 消费：复用前16 KiB接收 R32，后8 KiB发布 R16。
        // 保留 R32 落地，但没有低位补偿；后续 Cube 的输入精度仍为 BF16。
        WaitForCube<CubeToVectorEvent::ResidualReady>();
        constexpr uint32_t residualFp32Bytes = HALF_CHUNK * DIM * sizeof(float);
        static_assert(residualFp32Bytes + HALF_CHUNK * DIM * sizeof(bfloat16_t) <= UB_STAGE1_SLOT_BYTES,
                      "residual receive and BF16 copy must fit their reused Stage1 slot");
        const uint32_t residualOffset = PrefixTransformsUbOffset(slot);
        auto residual = BufferViews::Ub<float>(residualOffset);
        auto residualBf16 =
            BufferViews::Ub<bfloat16_t>(residualOffset + residualFp32Bytes);
        const MutexID residualMutex = MutexId(AIV_STAGE1_PUB_MUTEX_BASE + slot);
        Mutex::Lock<PIPE_V>(residualMutex);
        Cast(residualBf16, residual, RoundMode::CAST_RINT, HALF_CHUNK * DIM);
        PipeBarrier<PIPE_V>();
        Mutex::Unlock<PIPE_V>(residualMutex);
        Mutex::Lock<PIPE_MTE3>(residualMutex);
        auto l1Residual = BufferViews::L1<bfloat16_t>(slot * L1_SLOT_BYTES + L1_RESIDUAL_OFFSET);
        CopyCompactNzUbToL1<HALF_CHUNK, DIM, CHUNK>(residualBf16, l1Residual, subBlockIdx * HALF_CHUNK);
        NotifyCube<PIPE_MTE3, VectorToCubeEvent::ResidualReady>();
        Mutex::Unlock<PIPE_MTE3>(residualMutex);
    }

    __aicore__ inline void UpdateChunkState(const VectorHead &head, uint32_t chunk, const VectorBuffers &buffers)
    {
        const uint32_t slot = chunk & 1;
        auto gCum = BindPrefixBuffers(chunk).cumulativeGate;
        WaitForCube<CubeToVectorEvent::StateDeltaReady>();
        WaitForCube<CubeToVectorEvent::StateBufferFree>();
        // S_next=exp(G_last)*S+delta，S/delta 均为当前 AIV 的 [HALF_DIM,DIM] FP32 row-major 切片。
        // 等待当前增量到达且旧 L1 状态已读走，再更新并发布下个 chunk 的状态副本。
        // gCum 必须按 chunk 双缓冲，避免下一 chunk 的前缀和覆盖本次递推仍需使用的 G_last。
        // 后续 chunk 交给 CopyStateUbToL1；最后一次交给 CopyFinalStateUbToGm，由对应函数在转换完成后释放 PIPE_V 锁。
        Mutex::Lock<PIPE_V>(MutexId(AIV_STATE_MUTEX));
        auto gCumAddr = UbAddress(gCum);
        auto direct = BufferViews::Ub<float>(PrefixTransformsUbOffset(slot));
        if constexpr (!STATE_FP32) {
            // 升回 FP32 仅便于累加，不能恢复 delta 已丢弃的位；未引入 FP32 delta 优化。
            auto deltaBf16 = BufferViews::Ub<bfloat16_t>(UB_STATE_DELTA_BF16_OFFSET);
            Cast(direct, deltaBf16, RoundMode::CAST_NONE, STATE_SLICE_ELEMS);
            PipeBarrier<PIPE_V>();
        }
        auto stateAddr = UbAddress(buffers.state);
        auto directAddr = UbAddress(direct);
        AscendC::VF_CALL<UpdateFp32StateByExpVF>(stateAddr, directAddr, gCumAddr + CHUNK - 1, STATE_SLICE_ELEMS);
        PipeBarrier<PIPE_V>();
        NotifyCube<PIPE_V, VectorToCubeEvent::StateDeltaBufferFree>();
        if (chunk + 1 < context_.ChunkNum()) {
            CopyStateUbToL1(head.subBlock, buffers.state);
        }
    }

    // 每个 head 初始化一次 I/gather；本函数只管理初始化所需的 V/S 同步。
    __aicore__ inline void InitializeVectorConstants(const VectorBuffers &buffers)
    {
        auto identity = buffers.identity;
        auto gather = BufferViews::Ub<uint32_t>(UB_GATHER_OFFSET);
        const MutexID scalarMutex = MutexId(AIV_SCALAR_MUTEX);
        Mutex::Lock<PIPE_V>(scalarMutex);
        constexpr uint32_t inverseSliceElements = HALF_CHUNK * CHUNK;
        Duplicate(identity, 0.0f, inverseSliceElements);
        Mutex::Unlock<PIPE_V>(scalarMutex);
        Mutex::Lock<PIPE_S>(scalarMutex);
        for (uint32_t i = 0; i < HALF_CHUNK; ++i) {
            identity.SetValue(i * CHUNK + i, 1.0f);
        }
        for (uint32_t i = 0; i < CHUNK; ++i) {
            gather.SetValue(i, i * BLOCK_SIZE);
        }
        Mutex::Unlock<PIPE_S>(scalarMutex);
    }

    __aicore__ inline void CopyInitialStateGmToUb(const VectorHead &head, const VectorBuffers &buffers)
    {
        const uint32_t stateRowBegin = head.subBlock * HALF_DIM;
        const uint32_t stateValidRows = context_.StateValidRows(head.subBlock);
        const MutexID stateMutex = MutexId(AIV_STATE_MUTEX);
        auto state = buffers.state;
        const uint64_t initialStateOffset = context_.InitialStateOffset(head.valueHead, stateRowBegin);
        // 主状态始终为 FP32；BF16 初态经独立的发布暂存区转换，避免输入与目标重叠。
        Mutex::Lock<PIPE_V>(stateMutex);
        Duplicate(state, 0.0f, STATE_SLICE_ELEMS);
        Mutex::Unlock<PIPE_V>(stateMutex);
        auto stateStaging = BufferViews::Ub<bfloat16_t>(UB_STATE_PUB_OFFSET);
        Mutex::Lock<PIPE_MTE2>(stateMutex);
        if (stateValidRows > 0) {
            if constexpr (STATE_FP32) {
                CopyRowsGmToUb(state, context_.params.initialState[initialStateOffset], stateValidRows,
                              context_.params.dk, context_.params.dk);
            } else {
                CopyRowsGmToUb(stateStaging, context_.params.initialState[initialStateOffset], stateValidRows,
                              context_.params.dk, context_.params.dk);
            }
        }
        Mutex::Unlock<PIPE_MTE2>(stateMutex);
        if constexpr (!STATE_FP32) {
            Mutex::Lock<PIPE_V>(stateMutex);
            if (stateValidRows > 0) {
                Cast(state, stateStaging, RoundMode::CAST_NONE, stateValidRows * DIM);
                PipeBarrier<PIPE_V>();
            }
            Mutex::Unlock<PIPE_V>(stateMutex);
        }
    }

    // UpdateChunkState 的最后一次更新保留 V 锁，本函数完成 dtype 转换、写回并释放。
    __aicore__ inline void CopyFinalStateUbToGm(const VectorHead &head, const VectorBuffers &buffers)
    {
        const uint32_t stateRowBegin = head.subBlock * HALF_DIM;
        const uint32_t stateValidRows = context_.StateValidRows(head.subBlock);
        const MutexID stateMutex = MutexId(AIV_STATE_MUTEX);
        auto state = buffers.state;
        auto stateStaging = BufferViews::Ub<bfloat16_t>(UB_STATE_PUB_OFFSET);
        // 最终接口输出仍保持原 dtype；仅到整个 batch 递推结束才量化 BF16 主状态。
        if constexpr (!STATE_FP32) {
            if (stateValidRows > 0) {
                Cast(stateStaging, state, RoundMode::CAST_RINT, stateValidRows * DIM);
                PipeBarrier<PIPE_V>();
            }
        }
        Mutex::Unlock<PIPE_V>(stateMutex);
        Mutex::Lock<PIPE_MTE3>(stateMutex);
        if (stateValidRows > 0) {
            const uint64_t finalStateOffset = context_.FinalStateOffset(head.valueHead, stateRowBegin);
            if constexpr (STATE_FP32) {
                CopyStateRowsUbToGm(context_.params.finalState[finalStateOffset], state, stateValidRows, context_.params.dk);
            } else {
                CopyStateRowsUbToGm(context_.params.finalState[finalStateOffset], stateStaging, stateValidRows, context_.params.dk);
            }
        }
        Mutex::Unlock<PIPE_MTE3>(stateMutex);
    }

    static constexpr bool STATE_FP32 = std::is_same_v<stateType, float>;
    KernelContext<stateType> &context_;
};

} // namespace ChunkGatedDeltaRule

#endif // CHUNK_GATED_DELTA_RULE_VECTOR_COMPUTE_H
