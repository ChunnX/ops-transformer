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
 * \brief 使用 AIV 进行 reduce_sum — 手动 UB 管理 (无 TPipe)
 *
 * 参考 ops-transformer ReadRemoteDataAdd 实现:
 *   - 直接 UB 偏移管理 buffer (不使用 TPipe/TQue/TBuf)
 *   - 逐行处理: GM→UB→Cast→Add→Cast→UB→GM
 *   - float32 累加保证精度
 *   - self/received 分离读取 (curRank==rankId 从 workspace, 其他从 win 区)
 */

#ifndef REDUCE_SUM_REF_H
#define REDUCE_SUM_REF_H

#include "reduce_sum_utils.h"
#include "mc2_kernel_utils.h"
#include "kernel_operator.h"

namespace AiVReduceSumImpl {

using namespace AscendC;

constexpr static uint32_t UB_ALIGN_BYTES = 32U;
constexpr static uint64_t MAX_PER_ROW_NUM = 1024UL * 15UL;
constexpr uint32_t FLAG_ID = 1;

template <typename DataType>
class ReduceSumForAlltoAll {
public:
    __aicore__ inline ReduceSumForAlltoAll() {};

    __aicore__ inline void Init(uint64_t outputSize, uint64_t stride, uint64_t rankDim, uint64_t rankId,
                                uint64_t aivNum, GM_ADDR srcAddr, GM_ADDR selfAddr, GM_ADDR dstAddr)
    {
        rankDim_ = rankDim;
        rankId_ = rankId;
        aivNum_ = aivNum;
        strideSize_ = stride;
        srcAddr_ = srcAddr;
        selfAddr_ = selfAddr;

        srcGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(srcAddr));
        selfGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(selfAddr));
        dstGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DataType *>(dstAddr));
    }

    __aicore__ inline void ExecuteReduceSum(uint64_t tileM, uint64_t nSize)
    {
        nSize_ = nSize;

        // 行对齐 padding: CeilDiv(n * sizeof(DataType), 32B) * 32B / sizeof(DataType)
        uint64_t rowPaddedBytes = CeilDiv(nSize * sizeof(DataType), UB_ALIGN_BYTES) * UB_ALIGN_BYTES;
        uint64_t paddingN = rowPaddedBytes / sizeof(DataType);

        // 多行 block: 一次处理 rowsPerBlock 行, 减少 GM↔UB 同步次数
        // UB 容量: 4 个 buffer (sumF32 + bf16 + f32 + outBF16), 每个 rowsPerBlock × paddingN
        // 总 UB = 4 × rowsPerBlock × paddingN × 4 (float) ≤ TOTAL_UB_SIZE
        uint64_t ubPerRow = paddingN * sizeof(float);  // float 最大
        uint64_t maxRowsByUb = TOTAL_UB_SIZE / (4 * ubPerRow);
        uint64_t rowsPerBlock = MIN(maxRowsByUb, MAX_PER_ROW_NUM / paddingN);
        if (rowsPerBlock == 0) { rowsPerBlock = 1; }
        uint64_t blockElems = rowsPerBlock * paddingN;  // 每 block 的元素数 (paddingN 对齐)

        // UB 布局: [sumfp32(0) | localbf16(off1) | localfp32(off2) | sumbf16(off3)]
        uint64_t sumfp32Off = 0;
        uint64_t localbf16Off = blockElems * sizeof(float);
        uint64_t localfp32Off = localbf16Off + blockElems * sizeof(DataType);
        uint64_t sumbf16Off = localfp32Off + blockElems * sizeof(float);
        uint64_t totalUbBytes = sumbf16Off + blockElems * sizeof(DataType);

        // 核分配 (按行分配)
        uint32_t startRow = 0, endRow = 0, rowNum = 0;
        SplitToCore(static_cast<uint32_t>(tileM), static_cast<uint32_t>(aivNum_),
                    static_cast<uint32_t>(GetBlockIdx()), startRow, endRow, rowNum);

        // UB tensors (手动偏移)
        uint32_t totalUbElemsF32 = static_cast<uint32_t>(totalUbBytes / sizeof(float));
        uint32_t totalUbElemsDT = static_cast<uint32_t>(totalUbBytes / sizeof(DataType));
        LocalTensor<float> sumF32{TPosition::VECIN, 0, totalUbElemsF32};
        LocalTensor<DataType> ubBF16{TPosition::VECIN, 0, totalUbElemsDT};

        // 逐 block 处理 (每 block rowsPerBlock 行)
        uint32_t processedRows = 0;
        while (processedRows < rowNum) {
            uint32_t curRows = (processedRows + rowsPerBlock <= rowNum)
                             ? static_cast<uint32_t>(rowsPerBlock)
                             : (rowNum - processedRows);
            uint32_t curElems = curRows * static_cast<uint32_t>(nSize_);
            uint32_t tokenIndex = startRow + processedRows;

            // 1. 清零 float32 累加器
            Duplicate<float>(sumF32[sumfp32Off / sizeof(float)], static_cast<float>(0.0), curRows * paddingN);
            PipeBarrier<PIPE_V>();

            // 2. 逐 rank 累加
            for (uint64_t curRank = 0; curRank < rankDim_; ++curRank) {
                uint64_t rankOffset = curRank * strideSize_;
                uint64_t curOffset = rankOffset + static_cast<uint64_t>(tokenIndex) * nSize_;

                // 2a. GM → UB (BF16, 多行)
                GM_ADDR dataBaseAddr = (curRank == rankId_) ? selfAddr_ : srcAddr_;
                __gm__ DataType* gmPtr = reinterpret_cast<__gm__ DataType*>(
                    dataBaseAddr + curOffset * sizeof(DataType));
                GlobalTensor<DataType> gmData;
                gmData.SetGlobalBuffer(gmPtr);
                DataCopy(ubBF16[localbf16Off / sizeof(DataType)], gmData, curElems);
                SetFlag<HardEvent::MTE2_V>(FLAG_ID);
                WaitFlag<HardEvent::MTE2_V>(FLAG_ID);

                // 2b. Cast BF16 → float32
                Cast(sumF32[localfp32Off / sizeof(float)], ubBF16[localbf16Off / sizeof(DataType)],
                     RoundMode::CAST_NONE, curElems);
                PipeBarrier<PIPE_V>();

                // 2c. Add sumF32 += localF32
                Add(sumF32[sumfp32Off / sizeof(float)], sumF32[sumfp32Off / sizeof(float)],
                    sumF32[localfp32Off / sizeof(float)], curElems);
                SetFlag<HardEvent::V_MTE2>(FLAG_ID);
                WaitFlag<HardEvent::V_MTE2>(FLAG_ID);
            }

            // 3. Cast float32 → BF16
            PipeBarrier<PIPE_V>();
            Cast(ubBF16[sumbf16Off / sizeof(DataType)], sumF32[sumfp32Off / sizeof(float)],
                 RoundMode::CAST_RINT, curElems);
            SetFlag<HardEvent::V_MTE3>(FLAG_ID);
            WaitFlag<HardEvent::V_MTE3>(FLAG_ID);

            // 4. UB → GM (多行)
            DataCopy(dstGm_[static_cast<uint64_t>(tokenIndex) * nSize_],
                     ubBF16[sumbf16Off / sizeof(DataType)], curElems);
            SetFlag<HardEvent::MTE3_V>(FLAG_ID);
            WaitFlag<HardEvent::MTE3_V>(FLAG_ID);

            processedRows += curRows;
        }
    }

private:
    __aicore__ inline void SplitToCore(uint32_t totalRows, uint32_t coreNum, uint32_t coreId,
                                       uint32_t &startId, uint32_t &endId, uint32_t &rowNum)
    {
        rowNum = totalRows / coreNum;
        uint32_t remainder = totalRows % coreNum;
        startId = rowNum * coreId;
        if (coreId < remainder) {
            rowNum += 1;
            startId += coreId;
        } else {
            startId += remainder;
        }
        endId = startId + rowNum;
    }

    GlobalTensor<DataType> srcGm_;
    GlobalTensor<DataType> selfGm_;
    GlobalTensor<DataType> dstGm_;

    GM_ADDR srcAddr_{0};
    GM_ADDR selfAddr_{0};

    uint64_t rankDim_{0};
    uint64_t rankId_{0};
    uint64_t aivNum_{0};
    uint64_t strideSize_{0};
    uint64_t nSize_{0};
};

} // namespace AiVReduceSumImpl
#endif // REDUCE_SUM_REF_H
