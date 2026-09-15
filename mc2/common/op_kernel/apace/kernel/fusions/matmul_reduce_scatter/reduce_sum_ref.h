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
 * 注: 曾有 ExecuteReduceSumBatch 批量接口, 因多行批量 + 单次 flag 模式存在
 * 时序竞态(尾段数据错乱)已删除; 统一走逐行同步的 ExecuteReduceSum。
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
