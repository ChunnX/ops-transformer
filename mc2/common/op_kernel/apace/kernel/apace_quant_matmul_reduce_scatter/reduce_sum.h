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
