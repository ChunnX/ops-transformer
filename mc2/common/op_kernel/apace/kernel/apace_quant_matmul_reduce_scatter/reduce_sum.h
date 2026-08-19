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
 * \brief 使用 AIV 进行 reduce_sum — Te::MakeTensor + pingpong + BF16 直接累加
 *
 * 参考 ops-transformer ReadRemoteDataAdd 实现:
 *   - Te::MakeTensor + Te::Copy + Te::Transform (不使用 LocalTensor/VECIN)
 *   - pingpong: input buffer 双份交替 (pingPongCount_ & 1)
 *   - BF16 直接 Add (950 硬件级精度, 不需要 Cast float32)
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

        // pingpong UB 布局: [sumBF16 | inputBF16_A | inputBF16_B]
        // sum: 1 份 (BF16 累加器, 950 硬件级精度)
        // inputBF16: 2 份 (pingpong)
        // 注意: sum buffer 按 float 大小分配 (用于 float MulScalar 清零), 但实际用 BF16 操作
        uint64_t sumSlotBytes = paddingN * sizeof(float);  // 按 float 大小分配
        uint64_t inputSlotBytes = paddingN * sizeof(DataType);
        uint64_t sumOffset = 0;
        uint64_t inputOff[PINGPONG_NUM];
        inputOff[0] = sumOffset + sumSlotBytes;
        inputOff[1] = inputOff[0] + inputSlotBytes;

        uint32_t startRowId = 0, endRowId = 0, rowNum = 0;
        SplitToCore(static_cast<uint32_t>(tileM), static_cast<uint32_t>(aivNum_),
                    static_cast<uint32_t>(GetBlockIdx()), startRowId, endRowId, rowNum);

        auto layoutPaddingUB = MakeLayoutUB{}(1, paddingN);
        auto layoutPaddingUBF32 = Te::FrameLayoutFormat<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<float>>{}(1, paddingN);

        // sum tensor: float 视图用于清零, BF16 视图用于 Add
        auto sumF32Tensor = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::UB, float>(sumOffset), layoutPaddingUBF32);
        auto sumUbTensor = Te::MakeTensor(
            Te::MakeMemPtr<Te::Location::UB, DataType>(sumOffset), layoutPaddingUB);

        for (uint32_t tileIdx = 0; tileIdx < rowNum; ++tileIdx) {
            uint32_t tokenIndex = startRowId + tileIdx;

            // 1. 清零 sum (用 float MulScalar, 然后 BF16 Add 复用同一内存)
            Te::Transform<Te::Inst::MulScalar>(sumF32Tensor, sumF32Tensor, 0.0f);
            PipeBarrier<PIPE_V>();

            // 2. 逐 rank 累加 (pingpong, BF16 直接 Add)
            for (uint64_t curRank = 0; curRank < rankDim_; ++curRank) {
                uint32_t ppId = pingPongCount_ & 1;
                uint64_t rankOffset = curRank * strideSize_;
                uint64_t curOffset = rankOffset + static_cast<uint64_t>(tokenIndex) * nSize_;

                GM_ADDR dataBaseAddr = (curRank == rankId_) ? selfAddr_ : srcAddr_;
                __gm__ DataType* gmPtr = reinterpret_cast<__gm__ DataType*>(
                    dataBaseAddr + curOffset * sizeof(DataType));

                // GM → UB
                auto ubTensor = Te::MakeTensor(
                    Te::MakeMemPtr<Te::Location::UB, DataType>(inputOff[ppId]), layoutPaddingUB);
                auto gmTensor = Te::MakeTensor(
                    Te::MakeMemPtr<Te::Location::GM>(gmPtr), layoutPaddingUB);
                Te::Copy(copyGM2UB_, ubTensor, gmTensor);
                SetFlag<HardEvent::MTE2_V>(FLAG_ID_NUM_ONE);
                WaitFlag<HardEvent::MTE2_V>(FLAG_ID_NUM_ONE);

                // Add sumBF16 += inputBF16 (BF16 直接累加, 无 Cast)
                Te::Transform<Te::Inst::Add>(sumUbTensor, sumUbTensor, ubTensor);
                SetFlag<HardEvent::V_MTE2>(FLAG_ID_NUM_ONE);
                WaitFlag<HardEvent::V_MTE2>(FLAG_ID_NUM_ONE);

                pingPongCount_++;
            }

            // 3. UB → GM (直接输出 BF16, 无 Cast)
            __gm__ DataType* outPtr = reinterpret_cast<__gm__ DataType*>(
                dstAddr_ + static_cast<uint64_t>(tokenIndex) * nSize_ * sizeof(DataType));
            auto gmOutTensor = Te::MakeTensor(
                Te::MakeMemPtr<Te::Location::GM>(outPtr), layoutPaddingUB);
            SetFlag<HardEvent::V_MTE3>(FLAG_ID_NUM_ONE);
            WaitFlag<HardEvent::V_MTE3>(FLAG_ID_NUM_ONE);
            Te::Copy(copyUB2GM_, gmOutTensor, sumUbTensor);
            SetFlag<HardEvent::MTE3_V>(FLAG_ID_NUM_ONE);
            WaitFlag<HardEvent::MTE3_V>(FLAG_ID_NUM_ONE);
        }
    }

    /*!
     * \brief 批量 reduceSum — 方案3优化: 减少 flag 操作次数
     *
     * 每批处理 batchRows 行, 将每行 ~10 次 flag 操作降低到每批 ~10 次。
     * 批内处理顺序: 批量清零 → 逐 rank 批量 GM→UB + Add → 批量 UB→GM
     * flag 操作利用 MTE2/V/MTE3 流水线的 FIFO 特性: 多个操作后 1 次 SetFlag 即可同步。
     */
    __aicore__ inline void ExecuteReduceSumBatch(uint64_t tileM, uint64_t nSize)
    {
        nSize_ = nSize;

        uint64_t rowPaddedBytes =
            CeilDiv(static_cast<uint64_t>(nSize_ * sizeof(DataType)), UB_ALIGN_BYTES) * UB_ALIGN_BYTES;
        uint64_t paddingN = rowPaddedBytes / sizeof(DataType);

        // 自适应批量大小: 根据 UB 容量和 nSize 选择 batchRows
        // UB 布局: [sum (batchRows * paddingN * float)] [inputA (batchRows * paddingN * DataType)] [inputB (batchRows * paddingN * DataType)]
        // 总 UB = batchRows * paddingN * (sizeof(float) + 2 * sizeof(DataType)) = batchRows * paddingN * 8 (BF16)
        // 可用 UB ~ 180KB (扣除 commBuf 512B + barrierBuf 64B)
        constexpr uint64_t MAX_BATCH_UB_BYTES = 180UL * 1024UL;
        uint64_t perRowUbBytes = paddingN * (sizeof(float) + PINGPONG_NUM * sizeof(DataType));
        uint32_t maxBatchByUb = static_cast<uint32_t>(MAX_BATCH_UB_BYTES / perRowUbBytes);
        // 批量大小: 取 min(maxBatchByUb, 8), 至少 1
        uint32_t batchRows = (maxBatchByUb >= 8U) ? 8U : maxBatchByUb;
        if (batchRows == 0U) { batchRows = 1U; }

        uint64_t sumSlotBytes = static_cast<uint64_t>(batchRows) * paddingN * sizeof(float);
        uint64_t inputSlotBytes = static_cast<uint64_t>(batchRows) * paddingN * sizeof(DataType);
        uint64_t sumOffset = 0;
        uint64_t inputOff[PINGPONG_NUM];
        inputOff[0] = sumOffset + sumSlotBytes;
        inputOff[1] = inputOff[0] + inputSlotBytes;

        uint32_t startRowId = 0, endRowId = 0, rowNum = 0;
        SplitToCore(static_cast<uint32_t>(tileM), static_cast<uint32_t>(aivNum_),
                    static_cast<uint32_t>(GetBlockIdx()), startRowId, endRowId, rowNum);

        auto layoutRowUB = MakeLayoutUB{}(1, paddingN);
        auto layoutRowF32 = Te::FrameLayoutFormat<Te::NDExtLayoutPtn, Te::LayoutTraitDefault<float>>{}(1, paddingN);

        for (uint32_t batchStart = 0; batchStart < rowNum; batchStart += batchRows) {
            uint32_t curBatch = (batchStart + batchRows <= rowNum) ? batchRows : (rowNum - batchStart);

            // 1. 批量清零 sum (curBatch 行, 每行单独 MulScalar)
            for (uint32_t r = 0U; r < curBatch; ++r) {
                uint64_t rowSumOffset = sumOffset + static_cast<uint64_t>(r) * paddingN * sizeof(float);
                auto sumF32Row = Te::MakeTensor(
                    Te::MakeMemPtr<Te::Location::UB, float>(rowSumOffset), layoutRowF32);
                Te::Transform<Te::Inst::MulScalar>(sumF32Row, sumF32Row, 0.0f);
            }
            PipeBarrier<PIPE_V>();

            // 2. 逐 rank 批量累加 (pingpong)
            for (uint64_t curRank = 0; curRank < rankDim_; ++curRank) {
                uint32_t ppId = pingPongCount_ & 1;
                uint64_t curInputOff = inputOff[ppId];
                uint64_t rankOffset = curRank * strideSize_;

                GM_ADDR dataBaseAddr = (curRank == rankId_) ? selfAddr_ : srcAddr_;

                // 批量 GM → UB (curBatch 行, 不插 flag)
                for (uint32_t r = 0U; r < curBatch; ++r) {
                    uint32_t tokenIndex = startRowId + batchStart + r;
                    uint64_t curOffset = rankOffset + static_cast<uint64_t>(tokenIndex) * nSize_;
                    __gm__ DataType* gmPtr = reinterpret_cast<__gm__ DataType*>(
                        dataBaseAddr + curOffset * sizeof(DataType));

                    uint64_t rowInputOff = curInputOff + static_cast<uint64_t>(r) * paddingN * sizeof(DataType);
                    auto ubRowTensor = Te::MakeTensor(
                        Te::MakeMemPtr<Te::Location::UB, DataType>(rowInputOff), layoutRowUB);
                    auto gmRowTensor = Te::MakeTensor(
                        Te::MakeMemPtr<Te::Location::GM>(gmPtr), layoutRowUB);
                    Te::Copy(copyGM2UB_, ubRowTensor, gmRowTensor);
                }
                // 1 次 SetFlag/WaitFlag 同步所有 curBatch 行的 GM→UB
                SetFlag<HardEvent::MTE2_V>(FLAG_ID_NUM_ONE);
                WaitFlag<HardEvent::MTE2_V>(FLAG_ID_NUM_ONE);

                // 批量 Add (curBatch 行, 不插 flag)
                // 注意: sum 行偏移用 sizeof(float) (sum 按 float 大小分配), 与清零一致
                for (uint32_t r = 0U; r < curBatch; ++r) {
                    uint64_t rowSumOff = sumOffset + static_cast<uint64_t>(r) * paddingN * sizeof(float);
                    uint64_t rowInputOff = curInputOff + static_cast<uint64_t>(r) * paddingN * sizeof(DataType);
                    auto sumRowTensor = Te::MakeTensor(
                        Te::MakeMemPtr<Te::Location::UB, DataType>(rowSumOff), layoutRowUB);
                    auto inputRowTensor = Te::MakeTensor(
                        Te::MakeMemPtr<Te::Location::UB, DataType>(rowInputOff), layoutRowUB);
                    Te::Transform<Te::Inst::Add>(sumRowTensor, sumRowTensor, inputRowTensor);
                }
                // 1 次 SetFlag/WaitFlag 同步所有 curBatch 行的 Add
                SetFlag<HardEvent::V_MTE2>(FLAG_ID_NUM_ONE);
                WaitFlag<HardEvent::V_MTE2>(FLAG_ID_NUM_ONE);

                pingPongCount_++;
            }

            // 3. 批量 UB → GM (curBatch 行)
            // 注意: sum 行偏移用 sizeof(float) (sum 按 float 大小分配), 与清零/Add 一致
            SetFlag<HardEvent::V_MTE3>(FLAG_ID_NUM_ONE);
            WaitFlag<HardEvent::V_MTE3>(FLAG_ID_NUM_ONE);
            for (uint32_t r = 0U; r < curBatch; ++r) {
                uint32_t tokenIndex = startRowId + batchStart + r;
                __gm__ DataType* outPtr = reinterpret_cast<__gm__ DataType*>(
                    dstAddr_ + static_cast<uint64_t>(tokenIndex) * nSize_ * sizeof(DataType));
                uint64_t rowSumOff = sumOffset + static_cast<uint64_t>(r) * paddingN * sizeof(float);
                auto sumRowTensor = Te::MakeTensor(
                    Te::MakeMemPtr<Te::Location::UB, DataType>(rowSumOff), layoutRowUB);
                auto gmOutTensor = Te::MakeTensor(
                    Te::MakeMemPtr<Te::Location::GM>(outPtr), layoutRowUB);
                Te::Copy(copyUB2GM_, gmOutTensor, sumRowTensor);
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
