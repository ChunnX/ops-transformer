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
 * \file attention_worker_combine_split_k.h
 * \brief
 */

#ifndef ATTENTION_WORKER_COMBINE_SPLIT_K_REGBASE_H_
#define ATTENTION_WORKER_COMBINE_SPLIT_K_REGBASE_H_

#include "attention_worker_combine_common.h"
#include "../attention_worker_combine_common_utils.h"

namespace AttentionWorkerCombineRegbase {

template <typename T>
class KernelAttentionWorkerCombineSplitK {
public:
    __aicore__ inline KernelAttentionWorkerCombineSplitK() {}

    __aicore__ inline void Init(
        GM_ADDR schedule_context,
        GM_ADDR expert_scales,
        GM_ADDR layer_id,
        GM_ADDR y,
        GM_ADDR next_layer_id,
        GM_ADDR workspace,
        const AttentionWorkerCombineRegbaseTilingData *tilingDataPtr,
        TPipe *pipePtr)
    {
        pipe = pipePtr;
        tilingData = tilingDataPtr;

        contextGm0.SetGlobalBuffer((__gm__ uint32_t *)schedule_context);
        contextGm1.SetGlobalBuffer((__gm__ uint64_t *)schedule_context);

        if (tilingData->needSchedule == 1) {
            uint32_t micro_batch_num = contextGm0(GET_OFFSET_B32(ScheduleContext, common.micro_batch_num));
            micro_batch_id = (contextGm0(GET_OFFSET_B32(ScheduleContext, attention.micro_batch_id)) + 1) % micro_batch_num;
        } else {
            micro_batch_id = contextGm0(GET_OFFSET_B32(ScheduleContext, attention.micro_batch_id));
        }

        uint64_t token_data_addr = contextGm1(GET_OFFSET_B64(ScheduleContext, attention.token_data_buf));
        uint64_t token_info_addr = contextGm1(GET_OFFSET_B64(ScheduleContext, attention.token_info_buf));

        srcTokenGm.SetGlobalBuffer(
            (__gm__ T *)(token_data_addr + micro_batch_id * tilingData->BS * (tilingData->K + 1) * tilingData->H * sizeof(T)));
        srcTokenInfoGm.SetGlobalBuffer(
            (__gm__ int32_t *)(token_info_addr + micro_batch_id * tilingData->BS * (tilingData->K + 1) * sizeof(int32_t)));
        srcScalesGm.SetGlobalBuffer((__gm__ float *)expert_scales);
        srcLayerIdGm.SetGlobalBuffer((__gm__ int32_t *)layer_id);
        dstGm.SetGlobalBuffer((__gm__ T *)y);
        dstNextLayerIdGm.SetGlobalBuffer((__gm__ int32_t *)next_layer_id);

        pipe->InitBuffer(tokenDataQue, DOUBLE_BUFFER, tilingData->KSplitFactor * tilingData->HSplitFactor * sizeof(T));
        pipe->InitBuffer(yQue, DOUBLE_BUFFER, tilingData->HSplitFactor * sizeof(float));
        pipe->InitBuffer(tmpBuf, tilingData->HSplitFactor * sizeof(float) * 2);
    }

    __aicore__ inline void Process()
    {
        int64_t blockId = GetBlockIdx();

        if (blockId == tilingData->usedCoreNum - 1) {
            bsLoopNum = tilingData->tailCoreBsLoopNum;
        } else {
            bsLoopNum = tilingData->mainCoreBsLoopNum;
        }

        if (blockId == 0) {
            dstNextLayerIdGm(0) = srcLayerIdGm(0) + 1;
        }

        if (tilingData->needSchedule == 1) {
            WaitForTokensReady(bsLoopNum);
        }

        int64_t tokenSrcKInOffset = tilingData->KSplitFactor * tilingData->H;
        int64_t tokenSrcBsOffset = (tilingData->K + 1) * tilingData->H;
        int64_t tokenSrcStart = blockId * tilingData->mainCoreBsLoopNum * tokenSrcBsOffset;
        int64_t tokenDstStart = blockId * tilingData->mainCoreBsLoopNum * tilingData->H;
        int64_t scaleStart = blockId * tilingData->mainCoreBsLoopNum * tilingData->K;

        for (int64_t bsLoopId = 0; bsLoopId < bsLoopNum; bsLoopId++) {
            LocalTensor<float> tmpLocal = tmpBuf.Get<float>()[tilingData->HSplitFactor];
            Duplicate(tmpLocal, static_cast<float>(0), tilingData->HSplitFactor);

            for (int64_t kSplitLoopId = 0; kSplitLoopId < tilingData->KSplitLoopNum; kSplitLoopId++) {
                CopyIn(tokenSrcStart + bsLoopId * tokenSrcBsOffset + kSplitLoopId * tokenSrcKInOffset,
                       tilingData->KSplitFactor);
                Compute(scaleStart + bsLoopId * tilingData->K + kSplitLoopId * tilingData->KSplitFactor,
                        tilingData->KSplitFactor);
            }

            CopyIn(tokenSrcStart + bsLoopId * tokenSrcBsOffset + tilingData->KSplitLoopNum * tokenSrcKInOffset,
                   tilingData->KSplitTailFactor + 1);
            ComputeTail(scaleStart + bsLoopId * tilingData->K + tilingData->KSplitLoopNum * tilingData->KSplitFactor,
                        tilingData->KSplitTailFactor);

            CopyOut(tokenDstStart + bsLoopId * tilingData->H);
        }

        if (tilingData->needSchedule == 1) {
            ClearTokenInfo(blockId * tilingData->mainCoreBsLoopNum * (tilingData->K + 1),
                           bsLoopNum * (tilingData->K + 1));
            SyncAll();
            if (blockId == 0) {
                contextGm0(GET_OFFSET_B32(ScheduleContext, attention.micro_batch_id)) = micro_batch_id;
                DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_ALL>(
                    contextGm0[GET_OFFSET_B32(ScheduleContext, attention.micro_batch_id)]);
            }
        }
    }

private:
    __aicore__ inline void WaitForTokensReady(int64_t bsLoopNum)
    {
        int32_t sumRes = 0;
        int32_t expectedSum = bsLoopNum * (tilingData->K + 1);

        while (sumRes != expectedSum) {
            sumRes = ScanTokenInfo();
        }
    }

    __aicore__ inline int32_t ScanTokenInfo()
    {
        int64_t tokenInfoStart = GetBlockIdx() * tilingData->mainCoreBsLoopNum * (tilingData->K + 1);
        int64_t kSize = tilingData->mainCoreBsLoopNum * (tilingData->K + 1);

        LocalTensor<int32_t> srcLocal = tokenDataQue.AllocTensor<int32_t>();
        DataCopyExtParams copyParams(1, kSize * sizeof(int32_t), 0, 0, 0);
        DataCopyPadExtParams<int32_t> padParams(false, 0, 0, 0);
        DataCopyPad(srcLocal, srcTokenInfoGm[tokenInfoStart], copyParams, padParams);
        tokenDataQue.EnQue(srcLocal);
        srcLocal = tokenDataQue.DeQue<int32_t>();

        int32_t sum = 0;
        for (int64_t i = 0; i < kSize; i++) {
            sum += srcLocal.GetValue(i);
        }

        tokenDataQue.FreeTensor(srcLocal);
        return sum;
    }

    __aicore__ inline void ClearTokenInfo(int64_t offset, int64_t kSize)
    {
        LocalTensor<int32_t> dstLocal = yQue.AllocTensor<int32_t>();
        Duplicate(dstLocal, static_cast<int32_t>(0), kSize);
        yQue.EnQue(dstLocal);
        dstLocal = yQue.DeQue<int32_t>();
        DataCopyExtParams copyParams(1, kSize * sizeof(int32_t), 0, 0, 0);
        DataCopyPad(srcTokenInfoGm[offset], dstLocal, copyParams);
        yQue.FreeTensor(dstLocal);
    }

    __aicore__ inline void CopyIn(int64_t offset, int64_t kSize)
    {
        LocalTensor<T> srcLocal = tokenDataQue.AllocTensor<T>();
        int64_t hAlign32B = AlignUp(tilingData->H * sizeof(T), BLOCK_SIZE);
        DataCopyExtParams copyParams(kSize, tilingData->H * sizeof(T), 0,
                                     (tilingData->HSplitFactor * sizeof(T) - hAlign32B) / BLOCK_SIZE, 0);
        DataCopyPadExtParams<T> padParams(false, 0, 0, 0);
        DataCopyPad(srcLocal, srcTokenGm[offset], copyParams, padParams);
        tokenDataQue.EnQue(srcLocal);
    }

    __aicore__ inline void Compute(int64_t offset, int64_t kSize)
    {
        LocalTensor<T> srcLocal = tokenDataQue.DeQue<T>();
        LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
        LocalTensor<T> tmpCastLocal = tmpLocal.template ReinterpretCast<T>();

        for (int64_t i = 0; i < kSize; i++) {
            Cast(tmpLocal, srcLocal[i * tilingData->HSplitFactor], AscendC::RoundMode::CAST_NONE,
                 tilingData->HSplitFactor);
            PipeBarrier<PIPE_V>();

            Muls(tmpLocal, tmpLocal, srcScalesGm.GetValue(offset + i), tilingData->HSplitFactor);
            PipeBarrier<PIPE_V>();

            Add(tmpLocal[tilingData->HSplitFactor], tmpLocal[tilingData->HSplitFactor], tmpLocal,
                tilingData->HSplitFactor);
            PipeBarrier<PIPE_V>();
        }

        tokenDataQue.FreeTensor(srcLocal);
    }

    __aicore__ inline void ComputeTail(int64_t offset, int64_t kSize)
    {
        LocalTensor<T> srcLocal = tokenDataQue.DeQue<T>();
        LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
        LocalTensor<T> tmpCastLocal = tmpLocal.template ReinterpretCast<T>();
        LocalTensor<T> dstLocal = yQue.AllocTensor<T>();

        for (int64_t i = 0; i < kSize; i++) {
            Cast(tmpLocal, srcLocal[i * tilingData->HSplitFactor], AscendC::RoundMode::CAST_NONE,
                 tilingData->HSplitFactor);
            PipeBarrier<PIPE_V>();

            Muls(tmpLocal, tmpLocal, srcScalesGm.GetValue(offset + i), tilingData->HSplitFactor);
            PipeBarrier<PIPE_V>();

            Add(tmpLocal[tilingData->HSplitFactor], tmpLocal[tilingData->HSplitFactor], tmpLocal,
                tilingData->HSplitFactor);
            PipeBarrier<PIPE_V>();
        }

        Cast(tmpLocal, srcLocal[kSize * tilingData->HSplitFactor], AscendC::RoundMode::CAST_NONE,
             tilingData->HSplitFactor);
        PipeBarrier<PIPE_V>();
        Add(tmpLocal, tmpLocal, tmpLocal[tilingData->HSplitFactor], tilingData->HSplitFactor);
        PipeBarrier<PIPE_V>();

        if constexpr (std::is_same<T, bfloat16_t>::value) {
            Cast(dstLocal, tmpLocal, AscendC::RoundMode::CAST_RINT, tilingData->HSplitFactor);
        } else if constexpr (std::is_same<T, half>::value) {
            Cast(dstLocal, tmpLocal, AscendC::RoundMode::CAST_NONE, tilingData->HSplitFactor);
        }

        yQue.EnQue(dstLocal);
        tokenDataQue.FreeTensor(srcLocal);
    }

    __aicore__ inline void CopyOut(int64_t offset)
    {
        LocalTensor<T> dstLocal = yQue.DeQue<T>();
        int64_t hAlign32B = AlignUp(tilingData->H * sizeof(T), BLOCK_SIZE);
        DataCopyExtParams copyParams(1, tilingData->H * sizeof(T),
                                     (tilingData->HSplitFactor * sizeof(T) - hAlign32B) / BLOCK_SIZE, 0, 0);
        DataCopyPad(dstGm[offset], dstLocal, copyParams);
        yQue.FreeTensor(dstLocal);
    }

    TPipe *pipe = nullptr;
    const AttentionWorkerCombineRegbaseTilingData *tilingData = nullptr;

    GlobalTensor<uint32_t> contextGm0;
    GlobalTensor<uint64_t> contextGm1;
    GlobalTensor<T> srcTokenGm;
    GlobalTensor<int32_t> srcTokenInfoGm;
    GlobalTensor<float> srcScalesGm;
    GlobalTensor<int32_t> srcLayerIdGm;
    GlobalTensor<T> dstGm;
    GlobalTensor<int32_t> dstNextLayerIdGm;

    TQue<QuePosition::VECIN, 1> tokenDataQue;
    TQue<QuePosition::VECOUT, 1> yQue;
    TBuf<TPosition::VECCALC> tmpBuf;

    LocalTensor<T> tokenDataLocal;
    LocalTensor<float> yLocal;

    int64_t bsLoopNum = 0;
    uint32_t micro_batch_id = 0;
};

} // namespace AttentionWorkerCombineRegbase

#endif // ATTENTION_WORKER_COMBINE_SPLIT_K_REGBASE_H_
