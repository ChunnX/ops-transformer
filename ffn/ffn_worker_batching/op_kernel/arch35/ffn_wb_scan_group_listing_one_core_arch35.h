/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file ffn_wb_scan_group_listing_one_core_arch35.h
 * \brief
 */

#ifndef OP_KERNEL_ARCH35_FFN_WB_SCAN_GROUP_LISTING_ONE_CORE_ARCH35_H
#define OP_KERNEL_ARCH35_FFN_WB_SCAN_GROUP_LISTING_ONE_CORE_ARCH35_H

#include "ffn_wb_arch35_reuse.h"
#include "kernel_operator.h"

namespace FfnWbBatchingArch35 {
using namespace AscendC;
constexpr int64_t GROUP_LIST_UB_CAPACITY = 1024; // arch35 RECV group_list 分批 flush 阈值(A2 common 无此常量,本地定义)

class KernelFfnWBScanGroupListingOneCoreArch35 {
public:
    __aicore__ inline KernelFfnWBScanGroupListingOneCoreArch35(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR groupList, ScheduleContextInfo *scheduleContext, TPipe *tPipe)
    {
        int64_t outQueSize = 1024 * NUM_TWO * sizeof(int64_t) * NUM_TWO; // 1024个数, key & value, doblebuffer
        int64_t ubMaxRows = Align((scheduleContext->ubSize - outQueSize) / sizeof(int32_t) / NUM_TWO, sizeof(int32_t));
        int64_t validGatherIdxLength = scheduleContext->validGatherIdxLength;

        perLoopRows = Min(8192L, ubMaxRows); // max 8192
        perLoopRows = Max(1L, perLoopRows);

        loops = validGatherIdxLength / perLoopRows;
        lastLoopRows = validGatherIdxLength - loops * perLoopRows;

        pipe = tPipe;
        expertNum = scheduleContext->expertNum;
        blockIdx = GetBlockIdx() - (scheduleContext->coreNum - GROUP_LISTING_SCAN_AIV_NUM);
        expandedExpertIdsGm.SetGlobalBuffer((__gm__ int32_t *)x);
        groupListGm.SetGlobalBuffer((__gm__ int64_t *)groupList, expertNum * NUM_TWO);

        pipe->InitBuffer(copyInQueue, 1, perLoopRows * sizeof(int32_t));
        pipe->InitBuffer(copyOutQueue, 1,
                         GROUP_LIST_UB_CAPACITY * NUM_TWO * sizeof(int64_t) +
                             32); // GROUP_LIST_UB_CAPACITY个数, 多留32Bytes给最后的输出加上[0, 0]
        pipe->InitBuffer(maskBuf, perLoopRows * sizeof(int32_t));
        pipe->InitBuffer(expertIdsBuf, perLoopRows * sizeof(int32_t));
    }

    __aicore__ inline void Process()
    {
        if (blockIdx == 0) {
            groupListOutLocal = copyOutQueue.AllocTensor<int64_t>();
            curExpertIdOffset = 0;
            totalOutExpertIdNum = 0;
            for (int32_t i = 0; i < loops; i++) {
                CopyIn(i, perLoopRows);
                Compute(perLoopRows, i);
            }
            if (lastLoopRows > 0) {
                CopyIn(loops, lastLoopRows);
                Compute(lastLoopRows, -1);
            }
            // 更新最后一个expertId和tokenCnt，并且需要在groupList后面添加一个[0， 0]
            if (lastExpertId != -1) {
                groupListOutLocal.SetValue(curExpertIdOffset * NUM_TWO, lastExpertId);
                groupListOutLocal.SetValue(curExpertIdOffset * NUM_TWO + 1, tokenCount);
                curExpertIdOffset += 1;
            }
            // group list未满的情况下，最后补充[0， 0]
            if (totalOutExpertIdNum + curExpertIdOffset < expertNum) {
                groupListOutLocal.SetValue(curExpertIdOffset * NUM_TWO, 0);
                groupListOutLocal.SetValue(curExpertIdOffset * NUM_TWO + 1, 0);
                curExpertIdOffset += 1;
            }

            DataCopyExtParams copyParams{static_cast<uint16_t>(1),
                                         static_cast<uint32_t>(curExpertIdOffset * NUM_TWO * sizeof(int64_t)), 0, 0, 0};
            SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
            DataCopyPad(groupListGm[totalOutExpertIdNum * NUM_TWO], groupListOutLocal, copyParams);
            totalOutExpertIdNum += curExpertIdOffset;

            copyOutQueue.FreeTensor(groupListOutLocal);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t inputNum)
    {
        LocalTensor<int32_t> inLocal = copyInQueue.AllocTensor<int32_t>();
        DataCopyExtParams dataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(inputNum * sizeof(int32_t)), 0,
                                         0, 0};
        DataCopyPadExtParams<int32_t> dataCopyPadParams{false, 0, 0, 0};
        DataCopyPad(inLocal, expandedExpertIdsGm[progress * perLoopRows], dataCopyParams, dataCopyPadParams);
        copyInQueue.EnQue<int32_t>(inLocal);
    }

    // 中途 flush：groupListOutLocal 仅容纳 GROUP_LIST_UB_CAPACITY 条专家，当 expertNum 超过该容量时，
    // 必须分批把已填满的条目搬到 GM 并复用 buffer，否则 SetValue 会越界（表现为 expert>1024 后
    // group_list 读回 0）。totalOutExpertIdNum 记录已写入 GM 的条目偏移，本就为分批设计。
    __aicore__ inline void FlushOut()
    {
        if (curExpertIdOffset <= 0) {
            return;
        }
        DataCopyExtParams copyParams{static_cast<uint16_t>(1),
                                     static_cast<uint32_t>(curExpertIdOffset * NUM_TWO * sizeof(int64_t)), 0, 0, 0};
        SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        DataCopyPad(groupListGm[totalOutExpertIdNum * NUM_TWO], groupListOutLocal, copyParams);
        // 等本次搬出完成后才能复用同一 UB buffer 再 SetValue，避免数据竞争
        SetWaitFlag<HardEvent::MTE3_S>(HardEvent::MTE3_S);
        totalOutExpertIdNum += curExpertIdOffset;
        curExpertIdOffset = 0;
    }

    __aicore__ inline void Compute(int64_t inputNum, int64_t curLoop)
    {
        uint64_t rsvdCnt = 0;
        uint32_t inLocalIndex = 0;
        int32_t curExpertId = -1;
        int32_t processExpertCount = 0;
        LocalTensor<int32_t> inLocal = copyInQueue.DeQue<int32_t>();
        LocalTensor<int32_t> expertIdsBufFp32 = expertIdsBuf.Get<int32_t>();

        SetWaitFlag<HardEvent::MTE2_S>(HardEvent::MTE2_S);
        int32_t firstExpertId = inLocal.GetValue(inLocalIndex);
        // 专家在上一loop结尾处截止的场景
        if (firstExpertId != lastExpertId && lastExpertId != -1 && curExpertIdOffset < expertNum) {
            groupListOutLocal.SetValue(curExpertIdOffset * NUM_TWO, lastExpertId);
            groupListOutLocal.SetValue(curExpertIdOffset * NUM_TWO + 1, tokenCount);
            curExpertIdOffset += 1;
            tokenCount = 0;
            if (curExpertIdOffset >= GROUP_LIST_UB_CAPACITY) {
                FlushOut();
            }
        }

        while (inLocalIndex < inputNum) {
            curExpertId = inLocal.GetValue(inLocalIndex);
            SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
            LocalTensor<uint32_t> maskLocalUInt32 = maskBuf.Get<uint32_t>();
            LocalTensor<uint8_t> maskLocalTensorUInt8 = maskLocalUInt32.ReinterpretCast<uint8_t>();
            AscendC::CompareScalar(maskLocalTensorUInt8, inLocal, curExpertId, AscendC::CMPMODE::EQ,
                                   (inputNum + ONE_REPEAT_COMPARE_NUM - 1) / ONE_REPEAT_COMPARE_NUM *
                                       ONE_REPEAT_COMPARE_NUM);
            PipeBarrier<PIPE_V>();

            GatherMaskParams gatherMaskParams;
            gatherMaskParams.repeatTimes = 1;
            gatherMaskParams.src0BlockStride = 1;
            gatherMaskParams.src0RepeatStride = 8; // 8 blocks
            gatherMaskParams.src1RepeatStride = 0;
            GatherMask(expertIdsBufFp32, inLocal, maskLocalUInt32, true, inputNum, gatherMaskParams, rsvdCnt);
            tokenCount += rsvdCnt;
            if (inLocalIndex + rsvdCnt >= inputNum) {
                lastExpertId = curExpertId;
                break;
            } else if (inLocalIndex + rsvdCnt < inputNum && curExpertIdOffset < expertNum) {
                groupListOutLocal.SetValue(curExpertIdOffset * NUM_TWO, curExpertId);
                groupListOutLocal.SetValue(curExpertIdOffset * NUM_TWO + 1, tokenCount);
                tokenCount = 0;
                curExpertIdOffset += 1;
                inLocalIndex += rsvdCnt;
                if (curExpertIdOffset >= GROUP_LIST_UB_CAPACITY) {
                    FlushOut();
                }
            }
            processExpertCount++;
            if (processExpertCount > inputNum) {
                break;
            }
        } // end while
        copyInQueue.FreeTensor(inLocal);
    }

private:
    TPipe *pipe;
    TQue<QuePosition::VECIN, 1> copyInQueue;
    TQue<QuePosition::VECOUT, 1> copyOutQueue;
    TBuf<TPosition::VECCALC> maskBuf;
    TBuf<TPosition::VECCALC> expertIdsBuf;

    GlobalTensor<int64_t> groupListGm;
    GlobalTensor<int32_t> expandedExpertIdsGm;
    LocalTensor<int64_t> groupListOutLocal;

    FfnWBGroupListingTileInfo groupListingTileInfo;

    int64_t blockIdx;
    int64_t perLoopRows;
    int64_t loops;
    int64_t lastLoopRows;
    int64_t expertNum;

    int64_t tokenCount = 0;
    int64_t curExpertIdOffset = 0;
    int64_t totalOutExpertIdNum = 0;
    int32_t lastExpertId = -1;
    int32_t firstExpertId = -1;
};

} // namespace FfnWbBatchingArch35
#endif // OP_KERNEL_ARCH35_FFN_WB_SCAN_GROUP_LISTING_ONE_CORE_ARCH35_H
