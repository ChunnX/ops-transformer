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
 * \file reduce_sum_ref.h
 * \brief 使用 AIV 进行 reduce_sum — pingpong + float32 中间转换累加
 *
 *   - LocalTensor (VECIN) + GlobalTensor + DataCopy/Cast/Add 实现
 *   - pingpong: input buffer 双份交替 (pingPongCount_ & 1)
 *   - float32 中间转换: BF16→Cast float32→Add→Cast 回 BF16 (精度优先)
 *   - self/received 分离读取
 *
 * ExecuteReduceSumBatch 已停用 (多行批量 + 单次 flag 模式存在时序竞态),
 * 保留实现供后续修复参考; 现统一走逐行同步的 ExecuteReduceSum。
 */

#ifndef REDUCE_SUM_REF_H
#define REDUCE_SUM_REF_H

#include "kernel_operator.h"

namespace AiVReduceSumImpl {

using namespace AscendC;

constexpr static uint64_t UB_ALIGN_BYTES = 32UL;
constexpr int32_t FLAG_ID_NUM_ONE = 1;
constexpr uint32_t PINGPONG_NUM = 2U;

// 向上取整除法
__aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b)
{
    return (b == 0) ? 0 : (a + b - 1) / b;
}

template <typename DataType>
class ReduceSumForAlltoAll {
public:
    __aicore__ inline ReduceSumForAlltoAll(){};

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

        // N 分段: 6 slots × float, UB 上限 180KB
        constexpr uint64_t MAX_UB_BYTES = 180UL * 1024UL;
        uint64_t maxNPerSeg = MAX_UB_BYTES / (6U * sizeof(float));
        if (maxNPerSeg > paddingN) {
            maxNPerSeg = paddingN;
        }
        // 对齐到 32B (UB_ALIGN_BYTES / sizeof(float) = 8)
        maxNPerSeg = (maxNPerSeg / 8U) * 8U;
        if (maxNPerSeg == 0U) {
            maxNPerSeg = 8U;
        }

        uint32_t startRowId = 0, endRowId = 0, rowNum = 0;
        SplitToCore(static_cast<uint32_t>(tileM), static_cast<uint32_t>(aivNum_), static_cast<uint32_t>(GetBlockIdx()),
                    startRowId, endRowId, rowNum);

        constexpr uint64_t TOTAL_UB = 192UL * 1024UL;
        LocalTensor<float> ubF32{TPosition::VECIN, 0, static_cast<uint32_t>(TOTAL_UB / sizeof(float))};
        LocalTensor<DataType> ubBF16{TPosition::VECIN, 0, static_cast<uint32_t>(TOTAL_UB / sizeof(DataType))};

        for (uint32_t tileIdx = 0; tileIdx < rowNum; ++tileIdx) {
            uint32_t tokenIndex = startRowId + tileIdx;

            // N 分段循环
            for (uint64_t nStart = 0; nStart < paddingN; nStart += maxNPerSeg) {
                uint64_t nSegLen = (nStart + maxNPerSeg <= paddingN) ? maxNPerSeg : (paddingN - nStart);
                uint64_t segSlotBytes = nSegLen * sizeof(float);
                uint64_t sumOffset = 0;
                uint64_t bf16Off[PINGPONG_NUM], f32Off[PINGPONG_NUM];
                bf16Off[0] = sumOffset + segSlotBytes;
                f32Off[0] = bf16Off[0] + segSlotBytes;
                bf16Off[1] = f32Off[0] + segSlotBytes;
                f32Off[1] = bf16Off[1] + segSlotBytes;
                uint64_t outOff = f32Off[1] + segSlotBytes;

                // 1. 清零 sum (float32 累加器)
                Duplicate<float>(ubF32[sumOffset / sizeof(float)], static_cast<float>(0.0), nSegLen);
                PipeBarrier<PIPE_V>();

                // 2. 逐 rank 累加 (pingpong, float32 中间转换)
                for (uint64_t curRank = 0; curRank < rankDim_; ++curRank) {
                    uint32_t ppId = pingPongCount_ & 1;
                    uint64_t rankOffset = curRank * strideSize_;
                    uint64_t curOffset = rankOffset + static_cast<uint64_t>(tokenIndex) * nSize_ + nStart;

                    GM_ADDR dataBaseAddr = (curRank == rankId_) ? selfAddr_ : srcAddr_;
                    __gm__ DataType *gmPtr =
                        reinterpret_cast<__gm__ DataType *>(dataBaseAddr + curOffset * sizeof(DataType));

                    // GM → UB (BF16)
                    GlobalTensor<DataType> gmData;
                    gmData.SetGlobalBuffer(gmPtr);
                    DataCopy(ubBF16[bf16Off[ppId] / sizeof(DataType)], gmData, nSegLen);
                    SetFlag<HardEvent::MTE2_V>(FLAG_ID_NUM_ONE);
                    WaitFlag<HardEvent::MTE2_V>(FLAG_ID_NUM_ONE);

                    // Cast BF16 → float32
                    Cast(ubF32[f32Off[ppId] / sizeof(float)], ubBF16[bf16Off[ppId] / sizeof(DataType)],
                         RoundMode::CAST_NONE, nSegLen);
                    PipeBarrier<PIPE_V>();

                    // Add sumF32 += inputF32
                    Add(ubF32[sumOffset / sizeof(float)], ubF32[sumOffset / sizeof(float)],
                        ubF32[f32Off[ppId] / sizeof(float)], nSegLen);
                    SetFlag<HardEvent::V_MTE2>(FLAG_ID_NUM_ONE);
                    WaitFlag<HardEvent::V_MTE2>(FLAG_ID_NUM_ONE);

                    pingPongCount_++;
                }

                // 3. Cast float32 → BF16 (输出)
                PipeBarrier<PIPE_V>();
                Cast(ubBF16[outOff / sizeof(DataType)], ubF32[sumOffset / sizeof(float)], RoundMode::CAST_RINT,
                     nSegLen);
                SetFlag<HardEvent::V_MTE3>(FLAG_ID_NUM_ONE);
                WaitFlag<HardEvent::V_MTE3>(FLAG_ID_NUM_ONE);

                // 4. UB → GM
                __gm__ DataType *outPtr = reinterpret_cast<__gm__ DataType *>(
                    dstAddr_ + (static_cast<uint64_t>(tokenIndex) * nSize_ + nStart) * sizeof(DataType));
                GlobalTensor<DataType> gmOut;
                gmOut.SetGlobalBuffer(outPtr);
                DataCopy(gmOut, ubBF16[outOff / sizeof(DataType)], nSegLen);
                SetFlag<HardEvent::MTE3_V>(FLAG_ID_NUM_ONE);
                WaitFlag<HardEvent::MTE3_V>(FLAG_ID_NUM_ONE);
            }
        }
    }

    /*!
     * \brief 批量 reduceSum — 已停用, 保留供后续修复参考
     *
     * 多行 DataCopy 连发 + 单次 SetFlag/WaitFlag 的模式存在时序竞态,
     * 尾段(非整 maxNPerSeg 段)数据错乱(M=1024,K=128,N=8192,R=2 稳定复现)。
     * 单 tile 路径已改用逐行同步的 ExecuteReduceSum。
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

        // N 分段: 6 slots × float × batchRows, UB 上限 180KB
        constexpr uint64_t MAX_UB_BYTES = 180UL * 1024UL;
        uint64_t maxNPerSeg = MAX_UB_BYTES / (6U * sizeof(float) * 4U); // 预留 batchRows=4
        if (maxNPerSeg > paddingN) {
            maxNPerSeg = paddingN;
        }
        maxNPerSeg = (maxNPerSeg / 8U) * 8U;
        if (maxNPerSeg == 0U) {
            maxNPerSeg = 8U;
        }

        // batchRows 自适应 (基于 N 分段后的 per-seg UB)
        uint64_t perRowUbBytes = maxNPerSeg * 6U * sizeof(float);
        uint32_t maxBatchByUb = static_cast<uint32_t>(MAX_UB_BYTES / perRowUbBytes);
        uint32_t batchRows = (maxBatchByUb >= 4U) ? 4U : maxBatchByUb;
        if (batchRows == 0U) {
            batchRows = 1U;
        }

        uint32_t startRowId = 0, endRowId = 0, rowNum = 0;
        SplitToCore(static_cast<uint32_t>(tileM), static_cast<uint32_t>(aivNum_), static_cast<uint32_t>(GetBlockIdx()),
                    startRowId, endRowId, rowNum);

        constexpr uint64_t TOTAL_UB = 192UL * 1024UL;
        LocalTensor<float> ubF32{TPosition::VECIN, 0, static_cast<uint32_t>(TOTAL_UB / sizeof(float))};
        LocalTensor<DataType> ubBF16{TPosition::VECIN, 0, static_cast<uint32_t>(TOTAL_UB / sizeof(DataType))};

        for (uint32_t batchStart = 0; batchStart < rowNum; batchStart += batchRows) {
            uint32_t curBatch = (batchStart + batchRows <= rowNum) ? batchRows : (rowNum - batchStart);

            // N 分段循环
            for (uint64_t nStart = 0; nStart < paddingN; nStart += maxNPerSeg) {
                uint64_t nSegLen = (nStart + maxNPerSeg <= paddingN) ? maxNPerSeg : (paddingN - nStart);
                uint64_t segSlotBytes = static_cast<uint64_t>(batchRows) * nSegLen * sizeof(float);
                uint64_t sumOffset = 0;
                uint64_t bf16Off[PINGPONG_NUM], f32Off[PINGPONG_NUM];
                bf16Off[0] = sumOffset + segSlotBytes;
                f32Off[0] = bf16Off[0] + segSlotBytes;
                bf16Off[1] = f32Off[0] + segSlotBytes;
                f32Off[1] = bf16Off[1] + segSlotBytes;
                uint64_t outOff = f32Off[1] + segSlotBytes;

                // 1. 批量清零 sum (curBatch 行)
                for (uint32_t r = 0U; r < curBatch; ++r) {
                    uint64_t rowSumOff = sumOffset + static_cast<uint64_t>(r) * nSegLen * sizeof(float);
                    Duplicate<float>(ubF32[rowSumOff / sizeof(float)], static_cast<float>(0.0), nSegLen);
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
                        uint64_t curOffset = rankOffset + static_cast<uint64_t>(tokenIndex) * nSize_ + nStart;
                        __gm__ DataType *gmPtr =
                            reinterpret_cast<__gm__ DataType *>(dataBaseAddr + curOffset * sizeof(DataType));

                        uint64_t rowBf16Off = curBf16Off + static_cast<uint64_t>(r) * nSegLen * sizeof(float);
                        GlobalTensor<DataType> gmData;
                        gmData.SetGlobalBuffer(gmPtr);
                        DataCopy(ubBF16[rowBf16Off / sizeof(DataType)], gmData, nSegLen);
                    }
                    SetFlag<HardEvent::MTE2_V>(FLAG_ID_NUM_ONE);
                    WaitFlag<HardEvent::MTE2_V>(FLAG_ID_NUM_ONE);

                    // 批量 Cast BF16 → F32 + Add sumF32 += inputF32
                    for (uint32_t r = 0U; r < curBatch; ++r) {
                        uint64_t rowBf16Off = curBf16Off + static_cast<uint64_t>(r) * nSegLen * sizeof(float);
                        uint64_t rowF32Off = curF32Off + static_cast<uint64_t>(r) * nSegLen * sizeof(float);
                        uint64_t rowSumOff = sumOffset + static_cast<uint64_t>(r) * nSegLen * sizeof(float);

                        Cast(ubF32[rowF32Off / sizeof(float)], ubBF16[rowBf16Off / sizeof(DataType)],
                             RoundMode::CAST_NONE, nSegLen);
                        PipeBarrier<PIPE_V>();
                        Add(ubF32[rowSumOff / sizeof(float)], ubF32[rowSumOff / sizeof(float)],
                            ubF32[rowF32Off / sizeof(float)], nSegLen);
                    }
                    SetFlag<HardEvent::V_MTE2>(FLAG_ID_NUM_ONE);
                    WaitFlag<HardEvent::V_MTE2>(FLAG_ID_NUM_ONE);

                    pingPongCount_++;
                }

                // 3. 批量 Cast F32 → BF16 + UB → GM (curBatch 行)
                PipeBarrier<PIPE_V>();
                for (uint32_t r = 0U; r < curBatch; ++r) {
                    uint64_t rowSumOff = sumOffset + static_cast<uint64_t>(r) * nSegLen * sizeof(float);
                    uint64_t rowOutOff = outOff + static_cast<uint64_t>(r) * nSegLen * sizeof(float);
                    Cast(ubBF16[rowOutOff / sizeof(DataType)], ubF32[rowSumOff / sizeof(float)], RoundMode::CAST_RINT,
                         nSegLen);
                }
                SetFlag<HardEvent::V_MTE3>(FLAG_ID_NUM_ONE);
                WaitFlag<HardEvent::V_MTE3>(FLAG_ID_NUM_ONE);

                for (uint32_t r = 0U; r < curBatch; ++r) {
                    uint32_t tokenIndex = startRowId + batchStart + r;
                    __gm__ DataType *outPtr = reinterpret_cast<__gm__ DataType *>(
                        dstAddr_ + (static_cast<uint64_t>(tokenIndex) * nSize_ + nStart) * sizeof(DataType));
                    uint64_t rowOutOff = outOff + static_cast<uint64_t>(r) * nSegLen * sizeof(float);
                    GlobalTensor<DataType> gmOut;
                    gmOut.SetGlobalBuffer(outPtr);
                    DataCopy(gmOut, ubBF16[rowOutOff / sizeof(DataType)], nSegLen);
                }
                SetFlag<HardEvent::MTE3_V>(FLAG_ID_NUM_ONE);
                WaitFlag<HardEvent::MTE3_V>(FLAG_ID_NUM_ONE);
            }
        }
    }

private:
    __aicore__ inline void SplitToCore(uint32_t curSendCnt, uint32_t curUseAivNum, uint32_t coreId, uint32_t &startId,
                                       uint32_t &endId, uint32_t &rowNum)
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
