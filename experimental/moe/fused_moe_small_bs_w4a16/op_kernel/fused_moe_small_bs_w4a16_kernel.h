/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fused_moe_small_bs_w4a16_kernel.h
 * \brief fused moe w4a16 (small batch size) kernel implementation
 *
 * 计算流程（workspace 分为三段）：
 *   Phase 0: 清零 workspace
 *   Phase 1: W13 GroupedGemv (W4A16, 广播X, FP32原子加输出)  -> [TotalTokens, 2 * InterDim] FP32
 *   Phase 2: SwiGLU (FP32 输入, T 输出)                       -> [TotalTokens, InterDim] T
 *   Phase 3: W2 GroupedGemv (W4A16, 加权求和, FP32原子加输出) -> [BatchSize, HiddenSize] FP32
 *   Phase 4: Cast FP32 -> T                                   -> y [BatchSize, HiddenSize]
 */

#ifndef FUSED_MOE_SMALL_BS_W4A16_KERNEL_H_
#define FUSED_MOE_SMALL_BS_W4A16_KERNEL_H_

#include "kernel_operator.h"
#include <type_traits>

namespace NsFusedMoeSmallBsW4a16 {

using namespace AscendC;

// -----------------------------------------------------------------------------
// Type Traits for BFloat16 detection
// -----------------------------------------------------------------------------
template <typename T>
struct IsBFloat16 : std::false_type {};

template <>
struct IsBFloat16<bfloat16_t> : std::true_type {};

// -----------------------------------------------------------------------------
// Constants
// -----------------------------------------------------------------------------
constexpr int32_t BLOCK_SIZE = 128;                         // 一次处理的计算块大小
constexpr int32_t GROUP_SIZE = 32;                          // 量化 group 大小
constexpr int32_t GROUPS_PER_BLOCK = BLOCK_SIZE / GROUP_SIZE; // 4个
constexpr int32_t COMPUTE_ROWS = 32;                        // 每次内循环拷贝计算的行数
constexpr int32_t PACK_RATIO = 8;                           // 一个 int32 打包 8 个 int4
constexpr int32_t GROUP_TILE = 8;

// -----------------------------------------------------------------------------
// HardEvent IDs (only 0/1/2/3)
// Same HardEvent must use the same event_id everywhere.
// -----------------------------------------------------------------------------
static constexpr int EID_MTE2_V = 0;  // MTE2 -> V
static constexpr int EID_V_MTE2 = 1;  // V -> MTE2
static constexpr int EID_V_MTE3 = 2;  // V -> MTE3
static constexpr int EID_MTE3_V = 3;  // MTE3 -> V

// -----------------------------------------------------------------------------
// Simple Low-level ZeroOut (no pipe)
// -----------------------------------------------------------------------------
template <typename T>
class KernelZeroOutLL {
public:
    __aicore__ inline void Init(GM_ADDR dst, uint64_t elemCnt)
    {
        dstGm.SetGlobalBuffer((__gm__ T *)dst);
        total = elemCnt;
    }

    __aicore__ inline void Process()
    {
        uint32_t addr = 0;
        LocalTensor<T> tile = LocalTensor<T>(TPosition::VECOUT, addr, TILE_LEN);

        int32_t coreIdx = GetBlockIdx();
        int32_t coreNum = GetBlockNum();

        Duplicate(tile, (T)0, TILE_LEN);
        // V -> MTE3
        SetFlag<HardEvent::V_MTE3>(EID_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EID_V_MTE3);

        for (uint64_t base = (uint64_t)coreIdx * TILE_LEN; base < total; base += (uint64_t)coreNum * TILE_LEN) {
            uint32_t len = (uint32_t)((total - base < (uint64_t)TILE_LEN) ? (total - base) : TILE_LEN);
            DataCopy(dstGm[base], tile, len);
        }
        DataSyncBarrier<MemDsbT::DDR>();
    }

private:
    GlobalTensor<T> dstGm;
    uint64_t total = 0;
    static constexpr int32_t TILE_LEN = 2048;
};

// -----------------------------------------------------------------------------
// Kernel SwiGLU (Low-level, NO double-buffer; simple)
// Layout: [Gate | Value] split at inter_dim
// Input: FP32, Output: T
// -----------------------------------------------------------------------------
template <typename T>
class KernelSwiGLU {
public:
    __aicore__ inline KernelSwiGLU() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, int32_t totalRows, int32_t interDim)
    {
        inputGm.SetGlobalBuffer((__gm__ float *)input);
        outputGm.SetGlobalBuffer((__gm__ T *)output);
        this->totalRows = totalRows;
        this->interDim = interDim;

        uint32_t addr = 0;
        fIn = LocalTensor<float>(TPosition::VECCALC, addr, 2 * TILE_LEN);
        addr += (uint32_t)(2 * TILE_LEN * sizeof(float));
        fOut = LocalTensor<float>(TPosition::VECCALC, addr, TILE_LEN);
        addr += (uint32_t)(TILE_LEN * sizeof(float));

        tOut = LocalTensor<T>(TPosition::VECOUT, addr, TILE_LEN);
        addr += (uint32_t)(TILE_LEN * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        int32_t coreIdx = GetBlockIdx();
        int32_t coreNum = GetBlockNum();

        for (int32_t r = coreIdx; r < totalRows; r += coreNum) {
            ComputeRow(r);
        }
    }

private:
    __aicore__ inline void ComputeRow(int32_t rowIdx)
    {
        uint64_t inBase = (uint64_t)rowIdx * 2u * (uint64_t)interDim;
        uint64_t outBase = (uint64_t)rowIdx * (uint64_t)interDim;

        bool firstTile = true;
        for (int32_t i = 0; i < interDim; i += (int32_t)TILE_LEN) {
            int32_t len = (interDim - i < (int32_t)TILE_LEN) ? (interDim - i) : (int32_t)TILE_LEN;

            // reverse deps to avoid overwrite
            if (!firstTile) {
                WaitFlag<HardEvent::V_MTE2>(EID_V_MTE2);     // previous compute done before overwriting t_in
                WaitFlag<HardEvent::MTE3_V>(EID_MTE3_V);     // previous store done before overwriting t_out
            }

            // Copy FP32 input directly
            DataCopyParams params;
            params.blockCount = 2;
            params.blockLen = (uint16_t)(len * (int32_t)sizeof(float) / 32);
            params.srcStride = (uint16_t)((interDim - len) * (int32_t)sizeof(float) / 32);
            params.dstStride = 0;

            DataCopy(fIn, inputGm[inBase + (uint64_t)i], params);
            SetFlag<HardEvent::MTE2_V>(EID_MTE2_V);
            WaitFlag<HardEvent::MTE2_V>(EID_MTE2_V);

            Silu(fOut, fIn, len);
            Mul(fOut, fOut, fIn[len], len);
            Cast(tOut, fOut, RoundMode::CAST_ROUND, len);

            // mark V done for next tile's CopyIn overwrite protection
            SetFlag<HardEvent::V_MTE2>(EID_V_MTE2);

            // store
            SetFlag<HardEvent::V_MTE3>(EID_V_MTE3);
            WaitFlag<HardEvent::V_MTE3>(EID_V_MTE3);

            DataCopy(outputGm[outBase + (uint64_t)i], tOut, len);

            SetFlag<HardEvent::MTE3_V>(EID_MTE3_V);

            firstTile = false;
        }

        // close last outstanding pairs (important across rows, avoid "Set twice" later)
        if (!firstTile) {
            WaitFlag<HardEvent::V_MTE2>(EID_V_MTE2);
            WaitFlag<HardEvent::MTE3_V>(EID_MTE3_V);
        }
    }

private:
    GlobalTensor<float> inputGm;
    GlobalTensor<T> outputGm;

    LocalTensor<float> fIn, fOut;
    LocalTensor<T> tOut;

    int32_t totalRows = 0;
    int32_t interDim = 0;
    static constexpr int TILE_LEN = 1024;
};

// -----------------------------------------------------------------------------
// Kernel Grouped GEMV W4A16 MoE
// X: [TotalTokens or BatchSize(broadcast), InDim] T
// Weight: [NumExperts, InDim, OutDim / 8] int32 (packed int4)
// Scales: [NumExperts, InDim / 32, OutDim] T (对称量化, 无 zero-point)
// Output: OutputT (fused 流程中固定为 float, 使用原子加累加)
// -----------------------------------------------------------------------------
template <typename T, typename OutputT = T>
class KernelGroupedGemvW4A16Moe {
public:
    __aicore__ inline KernelGroupedGemvW4A16Moe() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR scales,
                                GM_ADDR expertIds, GM_ADDR y, GM_ADDR topkWeights,
                                int32_t totalTokens, int32_t inDim, int32_t outDim,
                                int32_t numExperts, int32_t topK,
                                bool isBroadcastX, bool isWeightedSum)
    {
        this->totalTokens = totalTokens;
        this->topK = topK;
        this->inDim = inDim;
        this->outDim = outDim;
        this->outDimPacked = outDim / PACK_RATIO;
        this->numExperts = numExperts;
        // Block 是数据加载单元，Group 是 Scale 单元
        this->numBlocks = inDim / BLOCK_SIZE;
        this->totalGroups = inDim / GROUP_SIZE;

        this->isBroadcastX = isBroadcastX;
        this->isWeightedSum = isWeightedSum;

        xGm.SetGlobalBuffer((__gm__ T *)x);
        weightGm.SetGlobalBuffer((__gm__ int32_t *)weight);
        scalesGm.SetGlobalBuffer((__gm__ T *)scales);
        expertIdsGm.SetGlobalBuffer((__gm__ int32_t *)expertIds);
        yGm.SetGlobalBuffer((__gm__ OutputT *)y);

        if (isWeightedSum) {
            topkWeightsGm.SetGlobalBuffer((__gm__ float *)topkWeights);
        }

        BuildLocalTensors();
    }

    __aicore__ inline void Process()
    {
        int32_t totalTasks = totalTokens * numBlocks;
        int32_t coreIdx = GetBlockIdx();
        int32_t coreNum = GetBlockNum();

        int32_t base = totalTasks / coreNum;
        int32_t rem = totalTasks % coreNum;

        int32_t startTask, taskCnt;
        if (coreIdx < rem) {
            taskCnt = base + 1;
            startTask = coreIdx * (base + 1);
        } else {
            taskCnt = base;
            startTask = rem * (base + 1) + (coreIdx - rem) * base;
        }
        int32_t endTask = startTask + taskCnt;
        if (taskCnt <= 0) {
            return;
        }

        for (int32_t nStart = 0; nStart < outDim; nStart += TILE_N) {
            int32_t curNLen = (outDim - nStart < TILE_N) ? (outDim - nStart) : TILE_N;
            Duplicate(yFp32, 0.0f, curNLen);

            int32_t currentRowIdx = -1;
            if (startTask < endTask) {
                currentRowIdx = startTask / numBlocks;
            }

            for (int32_t taskId = startTask; taskId < endTask; ++taskId) {
                int32_t rowIdx = taskId / numBlocks;
                int32_t bIdx = taskId % numBlocks;

                if (rowIdx != currentRowIdx) {
                    CopyOut(currentRowIdx, nStart, curNLen);
                    Duplicate(yFp32, 0.0f, curNLen);
                    currentRowIdx = rowIdx;
                }

                int32_t expertId = expertIdsGm.GetValue(rowIdx);
                ProcessBlock(bIdx, expertId, rowIdx, nStart, curNLen, yFp32);
            }

            if (currentRowIdx != -1) {
                CopyOut(currentRowIdx, nStart, curNLen);
            }
        }
    }

private:
    __aicore__ inline void BuildLocalTensors()
    {
        constexpr int32_t tileNPacked = TILE_N / PACK_RATIO;
        uint32_t addr = 0;

        xLocal = LocalTensor<T>(TPosition::VECIN, addr, BLOCK_SIZE);
        addr += (uint32_t)(BLOCK_SIZE * sizeof(T));

        wLocal[0] = LocalTensor<int32_t>(TPosition::VECIN, addr, COMPUTE_ROWS * tileNPacked);
        addr += (uint32_t)(COMPUTE_ROWS * tileNPacked * sizeof(int32_t));
        wLocal[1] = LocalTensor<int32_t>(TPosition::VECIN, addr, COMPUTE_ROWS * tileNPacked);
        addr += (uint32_t)(COMPUTE_ROWS * tileNPacked * sizeof(int32_t));

        // 保留 4 个 Scale 行缓存
        sLocal = LocalTensor<T>(TPosition::VECIN, addr, GROUPS_PER_BLOCK * TILE_N);
        addr += (uint32_t)(GROUPS_PER_BLOCK * TILE_N * sizeof(T));

        yFp32 = LocalTensor<float>(TPosition::VECOUT, addr, TILE_N);
        addr += (uint32_t)(TILE_N * sizeof(float));

        // 开辟 4 组计算和的缓存区，用于延迟乘 Scale
        groupAcc = LocalTensor<half>(TPosition::VECCALC, addr, GROUPS_PER_BLOCK * TILE_N);
        addr += (uint32_t)(GROUPS_PER_BLOCK * TILE_N * sizeof(half) + 256);

        wHalf = LocalTensor<half>(TPosition::VECCALC, addr, GROUP_TILE * TILE_N);
        addr += (uint32_t)(GROUP_TILE * TILE_N * sizeof(half));
    }

    __aicore__ inline void PrefetchW(int32_t expertId, int32_t bIdx, int32_t kInnerStart,
                                     int32_t nOffset, int32_t curNLen, int bufId)
    {
        uint64_t wStrideK = (uint64_t)outDimPacked;
        int32_t curNPacked = curNLen / PACK_RATIO;
        int32_t nOffsetPacked = nOffset / PACK_RATIO;

        uint64_t wOffset = (uint64_t)expertId * (uint64_t)inDim * wStrideK +
                           (uint64_t)(bIdx * BLOCK_SIZE + kInnerStart) * wStrideK +
                           (uint64_t)nOffsetPacked;

        DataCopyExtParams p{
            (uint16_t)COMPUTE_ROWS,
            (uint32_t)(curNPacked * sizeof(int32_t)),
            (uint32_t)((wStrideK - (uint64_t)curNPacked) * sizeof(int32_t)),
            0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(wLocal[bufId], weightGm[wOffset], p, pad);
    }

    __aicore__ inline void PrefetchS(int32_t expertId, int32_t bIdx,
                                     int32_t nOffset, int32_t curNLen)
    {
        // 当前Block的基础 group 索引
        uint64_t baseGroup = (uint64_t)bIdx * GROUPS_PER_BLOCK;

        DataCopyExtParams p{
            (uint16_t)GROUPS_PER_BLOCK,
            (uint32_t)(curNLen * sizeof(T)),
            (uint32_t)((outDim - (uint64_t)curNLen) * sizeof(T)),
            (uint32_t)((TILE_N - curNLen) * sizeof(T) / 32), 0};

        DataCopyPadExtParams<T> pad{false, 0, 0, 0};

        // 将属于这个Block里的 4 行 Scale 读出来
        uint64_t szOffset = (uint64_t)expertId * (uint64_t)totalGroups * (uint64_t)outDim +
                            baseGroup * (uint64_t)outDim +
                            (uint64_t)nOffset;
        DataCopyPad(sLocal, scalesGm[szOffset], p, pad);
    }

    __aicore__ inline void ComputeChunk(LocalTensor<int32_t> &wI32,
                                        int32_t xOffsetIdx,
                                        int32_t tileN,
                                        int32_t tileNPacked,
                                        LocalTensor<half> &xFullHalf,
                                        LocalTensor<half> &currentGroupAcc)
    {
        LocalTensor<uint64_t> xU64 = xFullHalf.template ReinterpretCast<uint64_t>();

        constexpr int STEP = 4;
        half xValBuf[COMPUTE_ROWS];
        uint64_t *xValBufU64 = (uint64_t *)xValBuf;
        for (int i = 0; i < COMPUTE_ROWS / STEP; ++i) {
            xValBufU64[i] = xU64.GetValue(xOffsetIdx / STEP + i);
        }

        for (int i = 0; i < COMPUTE_ROWS; i += GROUP_TILE) {
            LocalTensor<int4b_t> wInt4 = wI32[i * tileNPacked].template ReinterpretCast<int4b_t>();
            Cast(wHalf, wInt4, RoundMode::CAST_NONE, GROUP_TILE * tileN);
            for (int k = 0; k < GROUP_TILE; ++k) {
                Axpy(currentGroupAcc, wHalf[k * tileN], xValBuf[i + k], tileN);
            }
        }
    }

    __aicore__ inline void ProcessBlock(int32_t bIdx, int32_t expertId, int32_t rowIdx,
                                        int32_t nOffset, int32_t curNLen,
                                        LocalTensor<float> &globalAcc)
    {
        // 非法 expert id 直接跳过：行累加器已在 Process() 中清零，
        // 跳过所有 K block 后该行贡献即为 0。
        if (expertId < 0 || expertId >= numExperts) {
            return;
        }

        uint64_t xOffset;
        if (isBroadcastX) {
            int32_t batchIdx = rowIdx / topK;
            xOffset = (uint64_t)batchIdx * (uint64_t)inDim + (uint64_t)bIdx * BLOCK_SIZE;
        } else {
            xOffset = (uint64_t)rowIdx * (uint64_t)inDim + (uint64_t)bIdx * BLOCK_SIZE;
        }

        DataCopy(xLocal, xGm[xOffset], BLOCK_SIZE);
        SetFlag<HardEvent::MTE2_V>(EID_MTE2_V);
        WaitFlag<HardEvent::MTE2_V>(EID_MTE2_V);

        // 如果是 BF16，需要转换为 half 供后续计算使用
        // 转换流程：bf16 -> fp32 -> fp16
        if constexpr (IsBFloat16<T>::value) {
            LocalTensor<float> fTmp = wHalf.template ReinterpretCast<float>();
            LocalTensor<half> xHalf = xLocal.template ReinterpretCast<half>();
            // bf16 -> fp32
            Cast(fTmp, xLocal, RoundMode::CAST_NONE, BLOCK_SIZE);
            // fp32 -> fp16 (原地覆盖 xLocal)
            Cast(xHalf, fTmp, RoundMode::CAST_ROUND, BLOCK_SIZE);
        }

        PrefetchW(expertId, bIdx, 0, nOffset, curNLen, 0);
        SetFlag<HardEvent::MTE2_V>(EID_MTE2_V);

        // 初始化 4 个 group 的累加器
        Duplicate(groupAcc, (half)0.0f, GROUPS_PER_BLOCK * TILE_N);

        int32_t curNPacked = curNLen / PACK_RATIO;
        int ping = 0;

        for (int32_t kInner = 0; kInner < BLOCK_SIZE; kInner += COMPUTE_ROWS) {
            WaitFlag<HardEvent::MTE2_V>(EID_MTE2_V);
            if (kInner != 0) {
                WaitFlag<HardEvent::V_MTE2>(EID_V_MTE2);
            }

            if (kInner + COMPUTE_ROWS < BLOCK_SIZE) {
                int nextBuf = ping ^ 1;
                PrefetchW(expertId, bIdx, kInner + COMPUTE_ROWS, nOffset, curNLen, nextBuf);
                SetFlag<HardEvent::MTE2_V>(EID_MTE2_V);
            } else {
                // 最后一次拷贝 4 个对应的 Scale
                PrefetchS(expertId, bIdx, nOffset, curNLen);
                SetFlag<HardEvent::MTE2_V>(EID_MTE2_V);
            }

            // 计算该 32 行该放入哪个 groupAcc
            int step = kInner / COMPUTE_ROWS;
            LocalTensor<half> currentAcc = groupAcc[step * TILE_N];

            // xLocal 现在已经是 half 类型（如果是 bf16 已经转换过了）
            LocalTensor<half> xInput = xLocal.template ReinterpretCast<half>();
            ComputeChunk(wLocal[ping], kInner, curNLen, curNPacked, xInput, currentAcc);

            if (kInner + COMPUTE_ROWS < BLOCK_SIZE) {
                SetFlag<HardEvent::V_MTE2>(EID_V_MTE2);
            }
            ping ^= 1;
        }

        WaitFlag<HardEvent::MTE2_V>(EID_MTE2_V);

        // 对称量化，省去了 zero-point offset 补偿
        // 遍历 4 个分组，将 4 个 FP16 Group累加器 分别乘上对应的 Scale 后加入全局 Float 累加器中
        if constexpr (IsBFloat16<T>::value) {
            // 对于 BF16，需要复用 wHalf 作为 fp32 workspace
            // wHalf 大小为 GROUP_TILE * TILE_N = 8 * 2048 个 half
            // 作为 fp32 可以容纳 8192 个 float，足够覆盖 curNLen (<= 2048)
            LocalTensor<float> fAcc = wHalf.template ReinterpretCast<float>();
            LocalTensor<float> fScale = fAcc[TILE_N];

            for (int i = 0; i < GROUPS_PER_BLOCK; ++i) {
                LocalTensor<half> currentAcc = groupAcc[i * TILE_N];
                LocalTensor<T> currentScale = sLocal[i * TILE_N];

                // currentAcc (half) -> fAcc (float)
                Cast(fAcc, currentAcc, RoundMode::CAST_NONE, curNLen);
                // currentScale (bfloat16) -> fScale (float)
                Cast(fScale, currentScale, RoundMode::CAST_NONE, curNLen);
                // fAcc * fScale -> fAcc
                MulAddDst(globalAcc, fAcc, fScale, curNLen);
            }
        } else {
            // FP16 使用原来的 MulAddDst
            for (int i = 0; i < GROUPS_PER_BLOCK; ++i) {
                LocalTensor<half> currentAcc = groupAcc[i * TILE_N];
                LocalTensor<T> currentScale = sLocal[i * TILE_N];
                MulAddDst(globalAcc, currentAcc, currentScale, curNLen);
            }
        }
    }

    __aicore__ inline void CopyOut(int32_t rowIdx, int32_t nOffset, int32_t curNLen)
    {
        if (rowIdx < 0) {
            return;
        }

        if (isWeightedSum) {
            float wVal = topkWeightsGm.GetValue(rowIdx);
            Muls(yFp32, yFp32, wVal, curNLen);
        }

        SetFlag<HardEvent::V_MTE3>(EID_V_MTE3);
        WaitFlag<HardEvent::V_MTE3>(EID_V_MTE3);

        if constexpr (std::is_same_v<OutputT, float>) {
            // FP32 output: copy directly without cast
            SetAtomicAdd<float>();

            if (isWeightedSum) {
                int32_t batchIdx = rowIdx / topK;
                uint64_t yOffset = (uint64_t)batchIdx * (uint64_t)outDim + (uint64_t)nOffset;
                DataCopy(yGm[yOffset], yFp32, curNLen);
            } else {
                uint64_t yOffset = (uint64_t)rowIdx * (uint64_t)outDim + (uint64_t)nOffset;
                DataCopy(yGm[yOffset], yFp32, curNLen);
            }

            SetAtomicNone();
        } else {
            // T output: cast from FP32 to T
            LocalTensor<T> yHalf = yFp32.template ReinterpretCast<T>();
            Cast(yHalf, yFp32, RoundMode::CAST_ROUND, curNLen);

            SetAtomicAdd<T>();

            if (isWeightedSum) {
                int32_t batchIdx = rowIdx / topK;
                uint64_t yOffset = (uint64_t)batchIdx * (uint64_t)outDim + (uint64_t)nOffset;
                DataCopy(yGm[yOffset], yHalf, curNLen);
            } else {
                uint64_t yOffset = (uint64_t)rowIdx * (uint64_t)outDim + (uint64_t)nOffset;
                DataCopy(yGm[yOffset], yHalf, curNLen);
            }

            SetAtomicNone();
        }

        SetFlag<HardEvent::MTE3_V>(EID_MTE3_V);
        WaitFlag<HardEvent::MTE3_V>(EID_MTE3_V);
    }

private:
    GlobalTensor<float> topkWeightsGm;

    GlobalTensor<T> xGm;
    GlobalTensor<int32_t> weightGm;
    GlobalTensor<T> scalesGm;
    GlobalTensor<int32_t> expertIdsGm;
    GlobalTensor<OutputT> yGm;

    // UB
    LocalTensor<T> xLocal;
    LocalTensor<int32_t> wLocal[2];
    LocalTensor<T> sLocal;

    LocalTensor<float> yFp32;
    LocalTensor<half> groupAcc;
    LocalTensor<half> wHalf;

    int32_t topK = 0;
    int32_t inDim = 0;
    int32_t outDim = 0;
    int32_t outDimPacked = 0;
    int32_t numExperts = 0;
    int32_t numBlocks = 0;
    int32_t totalGroups = 0;
    int32_t totalTokens = 0;

    // Runtime flags
    bool isBroadcastX = false;
    bool isWeightedSum = false;

    static constexpr int32_t TILE_N = 2048;
};

// -----------------------------------------------------------------------------
// Kernel Cast FP32 to T (for final output conversion)
// -----------------------------------------------------------------------------
template <typename T>
class KernelCastFP32ToT {
public:
    __aicore__ inline KernelCastFP32ToT() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, int32_t totalElems)
    {
        inputGm.SetGlobalBuffer((__gm__ float *)input);
        outputGm.SetGlobalBuffer((__gm__ T *)output);
        this->totalElems = totalElems;

        uint32_t addr = 0;
        fLocal = LocalTensor<float>(TPosition::VECIN, addr, TILE_LEN);
        addr += (uint32_t)(TILE_LEN * sizeof(float));
        tLocal = LocalTensor<T>(TPosition::VECOUT, addr, TILE_LEN);
        addr += (uint32_t)(TILE_LEN * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        int32_t coreIdx = GetBlockIdx();
        int32_t coreNum = GetBlockNum();

        for (int64_t base = (int64_t)coreIdx * TILE_LEN; base < totalElems; base += (int64_t)coreNum * TILE_LEN) {
            int32_t len = (int32_t)((totalElems - base < (int64_t)TILE_LEN) ? (totalElems - base) : TILE_LEN);

            DataCopy(fLocal, inputGm[base], len);
            SetFlag<HardEvent::MTE2_V>(EID_MTE2_V);
            WaitFlag<HardEvent::MTE2_V>(EID_MTE2_V);

            Cast(tLocal, fLocal, RoundMode::CAST_ROUND, len);

            SetFlag<HardEvent::V_MTE3>(EID_V_MTE3);
            WaitFlag<HardEvent::V_MTE3>(EID_V_MTE3);

            DataCopy(outputGm[base], tLocal, len);

            SetFlag<HardEvent::MTE3_V>(EID_MTE3_V);
            WaitFlag<HardEvent::MTE3_V>(EID_MTE3_V);
        }
    }

private:
    GlobalTensor<float> inputGm;
    GlobalTensor<T> outputGm;

    LocalTensor<float> fLocal;
    LocalTensor<T> tLocal;

    int64_t totalElems = 0;
    static constexpr uint32_t TILE_LEN = 2048;
};

// -----------------------------------------------------------------------------
// Fused Kernel Implementation
// -----------------------------------------------------------------------------
template <typename T>
__aicore__ inline void FusedMoeSmallBsImpl(
    GM_ADDR x,
    GM_ADDR w13Weight, GM_ADDR w13Scales,
    GM_ADDR w2Weight, GM_ADDR w2Scales,
    GM_ADDR expertIds, GM_ADDR topkWeights,
    GM_ADDR y,
    GM_ADDR workspace,
    int32_t batchSize, int32_t hiddenSize, int32_t interSize,
    int32_t numExperts, int32_t topK)
{
    int32_t totalTokens = batchSize * topK;

    // workspace layout:
    // [0 : w13OutSize) : W13 output (FP32) [TotalTokens, 2 * InterDim]
    // [w13OutSize : w13OutSize + w2InSize) : SwiGLU output (T) [TotalTokens, InterDim]
    // [w13OutSize + w2InSize : end) : W2 output (FP32) [BatchSize, HiddenSize]
    GM_ADDR w13OutPtr = workspace;
    uint64_t w13OutSize = (uint64_t)totalTokens * (uint64_t)(interSize * 2) * sizeof(float);

    GM_ADDR w2InPtr = (GM_ADDR)((__gm__ uint8_t *)w13OutPtr + w13OutSize);
    uint64_t w2InSize = (uint64_t)totalTokens * (uint64_t)interSize * sizeof(T);

    GM_ADDR w2OutPtr = (GM_ADDR)((__gm__ uint8_t *)w2InPtr + w2InSize);

    // Phase 0: Zero out entire workspace at once (W13 FP32 + W2_in T + W2 FP32)
    {
        uint64_t w2OutSize = (uint64_t)batchSize * (uint64_t)hiddenSize * sizeof(float);
        uint64_t totalWorkspaceSize = w13OutSize + w2InSize + w2OutSize;
        uint64_t totalElems = totalWorkspaceSize / sizeof(int32_t);

        KernelZeroOutLL<int32_t> zeroOp;
        zeroOp.Init(workspace, totalElems);
        zeroOp.Process();
    }

    AscendC::SyncAll();

    // Phase 1: W13 Gemv (Broadcast X = true, Weighted = false, FP32 output)
    {
        KernelGroupedGemvW4A16Moe<T, float> opW13;
        opW13.Init(x, w13Weight, w13Scales,
                   expertIds, w13OutPtr, nullptr,
                   totalTokens, hiddenSize, interSize * 2, numExperts, topK,
                   true, false);
        opW13.Process();
    }

    AscendC::SyncAll();

    // Phase 2: SwiGLU (FP32 input, T output)
    // SwiGLU reads from w13Out and writes to w2In (separate memory areas)
    {
        KernelSwiGLU<T> opAct;
        opAct.Init(w13OutPtr, w2InPtr, totalTokens, interSize);
        opAct.Process();
    }

    AscendC::SyncAll();

    // Phase 3: W2 Gemv (Broadcast X = false, Weighted = true, FP32 output)
    {
        KernelGroupedGemvW4A16Moe<T, float> opW2;
        opW2.Init(w2InPtr, w2Weight, w2Scales,
                  expertIds, w2OutPtr, topkWeights,
                  totalTokens, interSize, hiddenSize, numExperts, topK,
                  false, true);
        opW2.Process();
    }

    AscendC::SyncAll();

    // Phase 4: Cast FP32 to T (final output to y)
    {
        KernelCastFP32ToT<T> opCast;
        opCast.Init(w2OutPtr, y, (int32_t)((uint64_t)batchSize * (uint64_t)hiddenSize));
        opCast.Process();
    }
}

} // namespace NsFusedMoeSmallBsW4a16

#endif // FUSED_MOE_SMALL_BS_W4A16_KERNEL_H_
