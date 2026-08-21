/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file reduce_sum_ref.h
 * \brief 使用 AIV 进行 reduce_sum — Te::MakeTensor + pingpong + float32 中间转换累加
 *
 * 参考 ops-transformer ReadRemoteDataAdd 实现:
 *   - Te::MakeTensor + Te::Copy + Te::Transform (不使用 LocalTensor/VECIN)
 *   - pingpong: input buffer 双份交替 (pingPongCount_ & 1)
 *   - float32 中间转换: BF16→Cast float32→Add→Cast 回 BF16 (精度优先)
 *   - self/received 分离读取
 *
 * 方案3优化 (ExecuteReduceSumBatch):
 *   - 批量处理 BATCH_ROWS 行, 减少每行 flag 操作次数 (从 ~10/行 降到 ~10/批)
 *   - 每批内: 先批量清零 sum, 再逐 rank 批量 GM→UB + Add, 最后批量 UB→GM
 *   - flag 操作: 每批每 rank 仅 1 次 SetFlag/WaitFlag (MTE2_V, V_MTE2), 每批仅 1 次 V_MTE3/MTE3_V
 *   - 仅用于 tileCnt=1 + SCALAR BOUND (小 shape) 路径
 */

#ifndef REDUCE_SUM_REF_H
#define REDUCE_SUM_REF_H

#include "reduce_sum_utils.h"
#include "mc2_kernel_utils.h"
#include "kernel_operator.h"

namespace AiVReduceSumImpl {

using namespace AscendC;

constexpr static uint64_t UB_ALIGN_BYTES = 32UL;
constexpr int32_t FLAG_ID_NUM_ONE = 1;
constexpr uint32_t PINGPONG_NUM = 2U;

template <typename DataType>
class ReduceSumForAlltoAll {
public:
    using MakeLayoutUB =
        Te::FrameLayoutFormat<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<DataType>>;
    using CopyMakerGM2UB = decltype(Te::MakeCopy(Te::CopyGM2UB{}));
    using CopyMakerUB2GM = decltype(Te::MakeCopy(Te::CopyUB2GM{}));

    __aicore__ inline ReduceSumForAlltoAll()
        : copyGM2UB_(Te::MakeCopy(Te::CopyGM2UB{})),
          copyUB2GM_(Te::MakeCopy(Te::CopyUB2GM{})) {};

    __aicore__ inline void Init(uint64_t outputSize, uint64_t stride, uint64_t rankDim, uint64_t rankId,
                                uint64_t aivNum, GM_ADDR srcAddr, GM_ADDR selfAddr, GM_ADDR dstAddr)
    {
        rankDim_ = rankDim;
        rankId_ = rankId;
        aivNum_ = aivNum;
        strideSize_ = stride;
        srcAddr_ = srcAddr;
        selfAddr_ = selfAddr;
        dstAddr_ = dstAddr;
    }

    __aicore__ inline void ExecuteReduceSum(uint64_t tileM, uint64_t nSize)
    {
        nSize_ = nSize;

        uint64_t rowPaddedBytes =
            CeilDiv(static_cast<uint64_t>(nSize_ * sizeof(DataType)), UB_ALIGN_BYTES) * UB_ALIGN_BYTES;
        uint64_t paddingN = rowPaddedBytes / sizeof(DataType);

        // UB 布局: [sumF32 | inputBF16_A | inputF32_A | inputBF16_B | inputF32_B | outBF16]
        // 所有 slot 按 float 大小对齐 (BF16 buffer 多分配, 地址对齐简单)
        uint64_t slotBytes = paddingN * sizeof(float);
        uint64_t sumOffset = 0;
        uint64_t bf16Off[PINGPONG_NUM], f32Off[PINGPONG_NUM];
        bf16Off[0] = sumOffset + slotBytes;
        f32Off[0] = bf16Off[0] + slotBytes;
        bf16Off[1] = f32Off[0] + slotBytes;
        f32Off[1] = bf16Off[1] + slotBytes;
        uint64_t outOff = f32Off[1] + slotBytes;

        uint32_t startRowId = 0, endRowId = 0, rowNum = 0;
        SplitToCore(static_cast<uint32_t>(tileM), static_cast<uint32_t>(aivNum_),
                    static_cast<uint32_t>(GetBlockIdx()), startRowId, endRowId, rowNum);

        // LocalTensor 覆盖整个 UB 空间, 用偏移寻址
        constexpr uint64_t TOTAL_UB = 192UL * 1024UL;
        LocalTensor<float> ubF32{TPosition::VECIN, 0, static_cast<uint32_t>(TOTAL_UB / sizeof(float))};
        LocalTensor<DataType> ubBF16{TPosition::VECIN, 0, static_cast<uint32_t>(TOTAL_UB / sizeof(DataType))};

        for (uint32_t tileIdx = 0; tileIdx < rowNum; ++tileIdx) {
            uint32_t tokenIndex = startRowId + tileIdx;

            // 1. 清零 sum (float32 累加器)
            Duplicate<float>(ubF32[sumOffset / sizeof(float)], static_cast<float>(0.0), paddingN);
            PipeBarrier<PIPE_V>();

            // 2. 逐 rank 累加 (pingpong, float32 中间转换)
            for (uint64_t curRank = 0; curRank < rankDim_; ++curRank) {
                uint32_t ppId = pingPongCount_ & 1;
                uint64_t rankOffset = curRank * strideSize_;
                uint64_t curOffset = rankOffset + static_cast<uint64_t>(tokenIndex) * nSize_;

                GM_ADDR dataBaseAddr = (curRank == rankId_) ? selfAddr_ : srcAddr_;
                __gm__ DataType* gmPtr = reinterpret_cast<__gm__ DataType*>(
                    dataBaseAddr + curOffset * sizeof(DataType));

                // GM → UB (BF16)
                GlobalTensor<DataType> gmData;
                gmData.SetGlobalBuffer(gmPtr);
                DataCopy(ubBF16[bf16Off[ppId] / sizeof(DataType)], gmData, paddingN);
                SetFlag<HardEvent::MTE2_V>(FLAG_ID_NUM_ONE);
                WaitFlag<HardEvent::MTE2_V>(FLAG_ID_NUM_ONE);

                // Cast BF16 → float32
                Cast(ubF32[f32Off[ppId] / sizeof(float)],
                     ubBF16[bf16Off[ppId] / sizeof(DataType)],
                     RoundMode::CAST_NONE, paddingN);
                PipeBarrier<PIPE_V>();

                // Add sumF32 += inputF32
                Add(ubF32[sumOffset / sizeof(float)],
                    ubF32[sumOffset / sizeof(float)],
                    ubF32[f32Off[ppId] / sizeof(float)], paddingN);
                SetFlag<HardEvent::V_MTE2>(FLAG_ID_NUM_ONE);
                WaitFlag<HardEvent::V_MTE2>(FLAG_ID_NUM_ONE);

                pingPongCount_++;
            }

            // 3. Cast float32 → BF16 (输出)
            PipeBarrier<PIPE_V>();
            Cast(ubBF16[outOff / sizeof(DataType)],
                 ubF32[sumOffset / sizeof(float)],
                 RoundMode::CAST_RINT, paddingN);
            SetFlag<HardEvent::V_MTE3>(FLAG_ID_NUM_ONE);
            WaitFlag<HardEvent::V_MTE3>(FLAG_ID_NUM_ONE);

            // 4. UB → GM
            __gm__ DataType* outPtr = reinterpret_cast<__gm__ DataType*>(
                dstAddr_ + static_cast<uint64_t>(tokenIndex) * nSize_ * sizeof(DataType));
            GlobalTensor<DataType> gmOut;
            gmOut.SetGlobalBuffer(outPtr);
            DataCopy(gmOut, ubBF16[outOff / sizeof(DataType)], paddingN);
            SetFlag<HardEvent::MTE3_V>(FLAG_ID_NUM_ONE);
            WaitFlag<HardEvent::MTE3_V>(FLAG_ID_NUM_ONE);
        }
    }

    /*!
     * \brief 批量 reduceSum — 方案3优化: 减少 flag 操作次数
     *
     * 每批处理 batchRows 行, 将每行 ~10 次 flag 操作降低到每批 ~10 次。
     * 批内处理顺序: 批量清零 → 逐 rank 批量 GM→UB + Cast + Add → 批量 Cast + UB→GM
     * flag 操作利用 MTE2/V/MTE3 流水线的 FIFO 特性: 多个操作后 1 次 SetFlag 即可同步。
     */
    __aicore__ inline void ExecuteReduceSumBatch(uint64_t tileM, uint64_t nSize)
    {
        nSize_ = nSize;

        uint64_t rowPaddedBytes =
            CeilDiv(static_cast<uint64_t>(nSize_ * sizeof(DataType)), UB_ALIGN_BYTES) * UB_ALIGN_BYTES;
        uint64_t paddingN = rowPaddedBytes / sizeof(DataType);

        // 自适应批量大小: 根据 UB 容量和 nSize 选择 batchRows
        // UB 布局 (float32 中间转换): [sumF32 | inputBF16_A | inputF32_A | inputBF16_B | inputF32_B | outBF16]
        // 所有 slot 按 float 大小对齐, 6 slots × float
        constexpr uint64_t MAX_BATCH_UB_BYTES = 180UL * 1024UL;
        uint64_t perRowUbBytes = paddingN * 6U * sizeof(float);
        uint32_t maxBatchByUb = static_cast<uint32_t>(MAX_BATCH_UB_BYTES / perRowUbBytes);
        uint32_t batchRows = (maxBatchByUb >= 4U) ? 4U : maxBatchByUb;
        if (batchRows == 0U) { batchRows = 1U; }

        uint64_t slotBytes = static_cast<uint64_t>(batchRows) * paddingN * sizeof(float);
        uint64_t sumOffset = 0;
        uint64_t bf16Off[PINGPONG_NUM], f32Off[PINGPONG_NUM];
        bf16Off[0] = sumOffset + slotBytes;
        f32Off[0] = bf16Off[0] + slotBytes;
        bf16Off[1] = f32Off[0] + slotBytes;
        f32Off[1] = bf16Off[1] + slotBytes;
        uint64_t outOff = f32Off[1] + slotBytes;

        uint32_t startRowId = 0, endRowId = 0, rowNum = 0;
        SplitToCore(static_cast<uint32_t>(tileM), static_cast<uint32_t>(aivNum_),
                    static_cast<uint32_t>(GetBlockIdx()), startRowId, endRowId, rowNum);

        constexpr uint64_t TOTAL_UB = 192UL * 1024UL;
        LocalTensor<float> ubF32{TPosition::VECIN, 0, static_cast<uint32_t>(TOTAL_UB / sizeof(float))};
        LocalTensor<DataType> ubBF16{TPosition::VECIN, 0, static_cast<uint32_t>(TOTAL_UB / sizeof(DataType))};

        for (uint32_t batchStart = 0; batchStart < rowNum; batchStart += batchRows) {
            uint32_t curBatch = (batchStart + batchRows <= rowNum) ? batchRows : (rowNum - batchStart);

            // 1. 批量清零 sum (curBatch 行)
            for (uint32_t r = 0U; r < curBatch; ++r) {
                uint64_t rowSumOff = sumOffset + static_cast<uint64_t>(r) * paddingN * sizeof(float);
                Duplicate<float>(ubF32[rowSumOff / sizeof(float)], static_cast<float>(0.0), paddingN);
            }
            PipeBarrier<PIPE_V>();

            // 2. 逐 rank 批量累加 (pingpong, float32 中间转换)
            for (uint64_t curRank = 0; curRank < rankDim_; ++curRank) {
                uint32_t ppId = pingPongCount_ & 1;
                uint64_t curBf16Off = bf16Off[ppId];
                uint64_t curF32Off = f32Off[ppId];
                uint64_t rankOffset = curRank * strideSize_;

                GM_ADDR dataBaseAddr = (curRank == rankId_) ? selfAddr_ : srcAddr_;

                // 批量 GM → UB (curBatch 行)
                for (uint32_t r = 0U; r < curBatch; ++r) {
                    uint32_t tokenIndex = startRowId + batchStart + r;
                    uint64_t curOffset = rankOffset + static_cast<uint64_t>(tokenIndex) * nSize_;
                    __gm__ DataType* gmPtr = reinterpret_cast<__gm__ DataType*>(
                        dataBaseAddr + curOffset * sizeof(DataType));

                    uint64_t rowBf16Off = curBf16Off + static_cast<uint64_t>(r) * paddingN * sizeof(float);
                    GlobalTensor<DataType> gmData;
                    gmData.SetGlobalBuffer(gmPtr);
                    DataCopy(ubBF16[rowBf16Off / sizeof(DataType)], gmData, paddingN);
                }
                SetFlag<HardEvent::MTE2_V>(FLAG_ID_NUM_ONE);
                WaitFlag<HardEvent::MTE2_V>(FLAG_ID_NUM_ONE);

                // 批量 Cast BF16 → F32 + Add sumF32 += inputF32
                for (uint32_t r = 0U; r < curBatch; ++r) {
                    uint64_t rowBf16Off = curBf16Off + static_cast<uint64_t>(r) * paddingN * sizeof(float);
                    uint64_t rowF32Off = curF32Off + static_cast<uint64_t>(r) * paddingN * sizeof(float);
                    uint64_t rowSumOff = sumOffset + static_cast<uint64_t>(r) * paddingN * sizeof(float);

                    Cast(ubF32[rowF32Off / sizeof(float)],
                         ubBF16[rowBf16Off / sizeof(DataType)],
                         RoundMode::CAST_NONE, paddingN);
                    PipeBarrier<PIPE_V>();
                    Add(ubF32[rowSumOff / sizeof(float)],
                        ubF32[rowSumOff / sizeof(float)],
                        ubF32[rowF32Off / sizeof(float)], paddingN);
                }
                SetFlag<HardEvent::V_MTE2>(FLAG_ID_NUM_ONE);
                WaitFlag<HardEvent::V_MTE2>(FLAG_ID_NUM_ONE);

                pingPongCount_++;
            }

            // 3. 批量 Cast F32 → BF16 + UB → GM (curBatch 行)
            PipeBarrier<PIPE_V>();
            for (uint32_t r = 0U; r < curBatch; ++r) {
                uint64_t rowSumOff = sumOffset + static_cast<uint64_t>(r) * paddingN * sizeof(float);
                uint64_t rowOutOff = outOff + static_cast<uint64_t>(r) * paddingN * sizeof(float);
                Cast(ubBF16[rowOutOff / sizeof(DataType)],
                     ubF32[rowSumOff / sizeof(float)],
                     RoundMode::CAST_RINT, paddingN);
            }
            SetFlag<HardEvent::V_MTE3>(FLAG_ID_NUM_ONE);
            WaitFlag<HardEvent::V_MTE3>(FLAG_ID_NUM_ONE);

            for (uint32_t r = 0U; r < curBatch; ++r) {
                uint32_t tokenIndex = startRowId + batchStart + r;
                __gm__ DataType* outPtr = reinterpret_cast<__gm__ DataType*>(
                    dstAddr_ + static_cast<uint64_t>(tokenIndex) * nSize_ * sizeof(DataType));
                uint64_t rowOutOff = outOff + static_cast<uint64_t>(r) * paddingN * sizeof(float);
                GlobalTensor<DataType> gmOut;
                gmOut.SetGlobalBuffer(outPtr);
                DataCopy(gmOut, ubBF16[rowOutOff / sizeof(DataType)], paddingN);
            }
            SetFlag<HardEvent::MTE3_V>(FLAG_ID_NUM_ONE);
            WaitFlag<HardEvent::MTE3_V>(FLAG_ID_NUM_ONE);
        }
    }

private:
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t coreId,
                                       uint32_t &startId, uint32_t &endId, uint32_t &rowNum)
    {
        rowNum = curSendCnt / curUseAivNum;
        uint32_t remainderNum = curSendCnt % curUseAivNum;
        startId = rowNum * coreId;
        if (coreId < remainderNum) {
            rowNum += 1;
            startId += coreId;
        } else {
            startId += remainderNum;
        }
        endId = startId + rowNum;
    }

    CopyMakerGM2UB copyGM2UB_;
    CopyMakerUB2GM copyUB2GM_;

    GM_ADDR srcAddr_{0};
    GM_ADDR selfAddr_{0};
    GM_ADDR dstAddr_{0};

    uint64_t rankDim_{0};
    uint64_t rankId_{0};
    uint64_t aivNum_{0};
    uint64_t strideSize_{0};
    uint64_t nSize_{0};
    uint32_t pingPongCount_{0};
};

} // namespace AiVReduceSumImpl
#endif // REDUCE_SUM_REF_H
