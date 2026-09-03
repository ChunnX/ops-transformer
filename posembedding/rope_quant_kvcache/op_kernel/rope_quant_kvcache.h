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
 * \file rope_quant_kvcache.h
 * \brief
 */
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace RopeQuantKvcacheND {
using namespace AscendC;
constexpr static int64_t TASK_NUM = 2;
constexpr static int64_t ROPE_TENSOR_NUM = 2;
constexpr static int64_t QUANTIZE_TENSOR_NUM = 2;
constexpr static int64_t ROPE_LAST_DIM_SPLIT = 2;
constexpr static int64_t FP16_ONE_BLOCK_NUM = 16;
constexpr static int64_t FP16_ONE_REPEAT_NUM = 128;
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102))
constexpr static DivConfig DIV_PRECISION_0ULP = {DivAlgo::PRECISION_0ULP_FTZ_FALSE};
#endif
class RopeQuantKvcache {
public:
    __aicore__ inline RopeQuantKvcache(const RopeQuantKvcacheTilingData *tilingData)
    {
        this->cacheSeqlen = tilingData->cacheSeqlen;
        this->qHeadNum = tilingData->qHeadNum;
        this->kvHeadNum = tilingData->kvHeadNum;
        this->hiddenSize = tilingData->hiddenSize;
        this->qHiddenSize = tilingData->qHiddenSize;
        this->kHiddenSize = tilingData->kHiddenSize;
        this->vHiddenSize = tilingData->vHiddenSize;
    }

    __aicore__ inline void Init(GM_ADDR qkv, GM_ADDR cos, GM_ADDR sin, GM_ADDR quant_scale, GM_ADDR quant_offset,
                                GM_ADDR k_cache, GM_ADDR v_cache, GM_ADDR indice, GM_ADDR q_out, GM_ADDR k_cache_out,
                                GM_ADDR v_cache_out)
    {
        auto blockIdx = GetBlockIdx();
        auto batchId = (blockIdx / TASK_NUM);
        uint64_t kvDataNum = this->kvHeadNum * this->hiddenSize;
        uint64_t qkvBlockOffset = batchId * (this->qHiddenSize + this->kHiddenSize + this->vHiddenSize);
        uint64_t cossinBlockOffset = batchId * this->hiddenSize;
        uint64_t kvCacheBlockOffset = batchId * this->cacheSeqlen * kvDataNum;

        taskId = blockIdx % TASK_NUM;
        if (taskId == 1) {
            uint64_t qDataNum = this->qHeadNum * this->hiddenSize;
            uint64_t qBlockOffset = batchId * qDataNum;
            inputGm.SetGlobalBuffer((__gm__ half *)qkv + qkvBlockOffset);

            cosGm.SetGlobalBuffer((__gm__ half *)cos + cossinBlockOffset);
            cosGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

            sinGm.SetGlobalBuffer((__gm__ half *)sin + cossinBlockOffset);
            sinGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

            outputGm.SetGlobalBuffer((__gm__ half *)q_out + qBlockOffset);

            quantScaleGm.SetGlobalBuffer((__gm__ float *)quant_scale);
            quantScaleGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

            quantOffsetGm.SetGlobalBuffer((__gm__ int32_t *)quant_offset);
            quantOffsetGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

            indiceGm.SetGlobalBuffer((__gm__ int32_t *)indice + batchId);
            indiceGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            int32_t idx = indiceGm.GetValue(0);
            vCacheGm.SetGlobalBuffer((__gm__ int8_t *)v_cache_out + kvCacheBlockOffset + idx * kvDataNum);

            pipe.InitBuffer(inQueue, 1,
                            qDataNum * sizeof(half) + kvDataNum * sizeof(half) +
                                this->hiddenSize * sizeof(half) * ROPE_TENSOR_NUM +
                                this->hiddenSize * sizeof(float) * QUANTIZE_TENSOR_NUM);
            pipe.InitBuffer(outQueue, 1,
                            qDataNum * sizeof(half) + kvDataNum * sizeof(float) + this->hiddenSize * sizeof(float) * 4);
        } else if (taskId == 0) {
            inputGm.SetGlobalBuffer((__gm__ half *)qkv + qkvBlockOffset + this->qHiddenSize);

            cosGm.SetGlobalBuffer((__gm__ half *)cos + cossinBlockOffset);
            cosGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

            sinGm.SetGlobalBuffer((__gm__ half *)sin + cossinBlockOffset);
            sinGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

            quantScaleGm.SetGlobalBuffer((__gm__ float *)quant_scale);
            quantScaleGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

            quantOffsetGm.SetGlobalBuffer((__gm__ int32_t *)quant_offset);
            quantOffsetGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);

            indiceGm.SetGlobalBuffer((__gm__ int32_t *)indice + batchId);
            indiceGm.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
            int32_t idx = indiceGm.GetValue(0);
            kCacheGm.SetGlobalBuffer((__gm__ int8_t *)k_cache_out + kvCacheBlockOffset + idx * kvDataNum);

            pipe.InitBuffer(inQueue, 1,
                            kvDataNum * sizeof(half) + this->hiddenSize * sizeof(half) * ROPE_TENSOR_NUM +
                                this->hiddenSize * sizeof(float) * QUANTIZE_TENSOR_NUM);
            pipe.InitBuffer(outQueue, 1, kvDataNum * sizeof(float) + this->hiddenSize * sizeof(float) * 3);
        }
    }

    __aicore__ inline void Process()
    {
        if (this->taskId == 0) {
            ProcessK();
        } else if (this->taskId == 1) {
            ProcessQV();
        }
    }

private:
    __aicore__ inline void ProcessQV()
    {
        // copy in cos/sin
        LocalTensor<half> qInUb = inQueue.AllocTensor<half>();
        LocalTensor<half> vInUb = qInUb[this->qHeadNum * this->hiddenSize];
        LocalTensor<half> cosUb = qInUb[this->qHeadNum * this->hiddenSize + this->kvHeadNum * this->hiddenSize];
        LocalTensor<half> sinUb = cosUb[this->hiddenSize];
        LocalTensor<float> quantScaleUb = cosUb[this->hiddenSize * ROPE_TENSOR_NUM].ReinterpretCast<float>();
        LocalTensor<float> quantOffsetUb = quantScaleUb[this->hiddenSize];
        LocalTensor<int32_t> quantOffsetUbOri = quantOffsetUb.ReinterpretCast<int32_t>();

        DataCopy(quantScaleUb, quantScaleGm, this->hiddenSize);
        DataCopy(quantOffsetUbOri, quantOffsetGm, this->hiddenSize);
        DataCopy(cosUb, cosGm, this->hiddenSize);
        DataCopy(sinUb, sinGm, this->hiddenSize);
        DataCopy(qInUb, inputGm, this->qHeadNum * this->hiddenSize);
        DataCopy(vInUb, inputGm[this->qHiddenSize + this->kHiddenSize], this->kvHeadNum * this->hiddenSize);
        inQueue.EnQue(qInUb);
        qInUb = inQueue.DeQue<half>();
        Cast(quantOffsetUb, quantOffsetUbOri, RoundMode::CAST_NONE, this->hiddenSize);
        PipeBarrier<PIPE_V>();

        // caculate q
        LocalTensor<half> qOutUb = outQueue.AllocTensor<half>();
        uint64_t qDataNum = this->qHeadNum * this->hiddenSize;
        uint64_t kvDataNum = this->kvHeadNum * this->hiddenSize;
        LocalTensor<float> outUbF32 = qOutUb.ReinterpretCast<float>();
        LocalTensor<float> vOutUbF32 = outUbF32[qDataNum * sizeof(half) / sizeof(float)];
        LocalTensor<float> qInUbF32 = vOutUbF32[kvDataNum];
        LocalTensor<float> cosUbF32 = qInUbF32[this->hiddenSize];
        LocalTensor<float> sinUbF32 = cosUbF32[this->hiddenSize];
        LocalTensor<float> qRopeUbF32 = sinUbF32[this->hiddenSize];
        Cast(vOutUbF32, vInUb, RoundMode::CAST_NONE, this->kvHeadNum * this->hiddenSize);
        Cast(cosUbF32, cosUb, RoundMode::CAST_NONE, this->hiddenSize);
        Cast(sinUbF32, sinUb, RoundMode::CAST_NONE, this->hiddenSize);
        PipeBarrier<PIPE_V>();
        uint64_t halfSize = this->hiddenSize / ROPE_LAST_DIM_SPLIT;
        Muls(sinUbF32, sinUbF32, -1.0f, halfSize);
        PipeBarrier<PIPE_V>();
        for (uint64_t r = 0; r < this->qHeadNum; r++) {
            uint64_t rowOffset = r * this->hiddenSize;
            Cast(qInUbF32, qInUb[rowOffset], RoundMode::CAST_NONE, this->hiddenSize);
            PipeBarrier<PIPE_V>();
            Mul(qRopeUbF32, qInUbF32[halfSize], sinUbF32, halfSize);
            Mul(qRopeUbF32[halfSize], qInUbF32, sinUbF32[halfSize], halfSize);
            Mul(qInUbF32, qInUbF32, cosUbF32, this->hiddenSize);
            PipeBarrier<PIPE_V>();
            Add(qRopeUbF32, qRopeUbF32, qInUbF32, this->hiddenSize);
            PipeBarrier<PIPE_V>();
            Cast(qOutUb[rowOffset], qRopeUbF32, RoundMode::CAST_NONE, this->hiddenSize);
            PipeBarrier<PIPE_V>();
        }
        for (uint64_t r = 0; r < this->kvHeadNum; r++) {
            uint64_t rowOffset = r * this->hiddenSize;
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102))
            Div<float, DIV_PRECISION_0ULP>(vOutUbF32[rowOffset], vOutUbF32[rowOffset], quantScaleUb, this->hiddenSize);
#else
            Div(vOutUbF32[rowOffset], vOutUbF32[rowOffset], quantScaleUb, this->hiddenSize);
#endif
        }
        PipeBarrier<PIPE_V>();
        for (uint64_t r = 0; r < this->kvHeadNum; r++) {
            uint64_t rowOffset = r * this->hiddenSize;
            Add(vOutUbF32[rowOffset], vOutUbF32[rowOffset], quantOffsetUb, this->hiddenSize);
        }

        event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        DataCopy(outputGm, qOutUb, this->qHeadNum * this->hiddenSize);

        // caculate v
        PipeBarrier<PIPE_V>();
        LocalTensor<int16_t> vOutUbS16 = vOutUbF32.ReinterpretCast<int16_t>();
        Cast(vOutUbS16, vOutUbF32, RoundMode::CAST_RINT, this->kvHeadNum * this->hiddenSize);
        PipeBarrier<PIPE_V>();
        LocalTensor<half> vOutUbF16 = vOutUbF32.ReinterpretCast<half>();
        Cast(vOutUbF16, vOutUbS16, RoundMode::CAST_NONE, this->kvHeadNum * this->hiddenSize);
        PipeBarrier<PIPE_V>();
        LocalTensor<int8_t> vOutUbS8 = vOutUbF32.ReinterpretCast<int8_t>();
        Cast(vOutUbS8, vOutUbF16, RoundMode::CAST_NONE, this->kvHeadNum * this->hiddenSize);
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        DataCopy(vCacheGm, vOutUbS8, this->kvHeadNum * this->hiddenSize);
    }

    __aicore__ inline void ProcessK()
    {
        // copy in cos/sin/scale/offset
        LocalTensor<half> kInUb = inQueue.AllocTensor<half>();
        LocalTensor<half> cosUb = kInUb[this->kvHeadNum * this->hiddenSize];
        LocalTensor<half> sinUb = cosUb[this->hiddenSize];
        LocalTensor<float> quantScaleUb = cosUb[this->hiddenSize * ROPE_TENSOR_NUM].ReinterpretCast<float>();
        LocalTensor<float> quantOffsetUb = quantScaleUb[this->hiddenSize];
        LocalTensor<int32_t> quantOffsetUbOri = quantOffsetUb.ReinterpretCast<int32_t>();

        DataCopy(quantScaleUb, quantScaleGm, this->hiddenSize);
        DataCopy(quantOffsetUbOri, quantOffsetGm, this->hiddenSize);
        DataCopy(cosUb, cosGm, this->hiddenSize);
        DataCopy(sinUb, sinGm, this->hiddenSize);
        DataCopy(kInUb, inputGm, this->kvHeadNum * this->hiddenSize);

        inQueue.EnQue(kInUb);
        kInUb = inQueue.DeQue<half>();

        Cast(quantOffsetUb, quantOffsetUbOri, RoundMode::CAST_NONE, this->hiddenSize);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> kOutUbF32 = outQueue.AllocTensor<float>();
        uint64_t kvDataNum = this->kvHeadNum * this->hiddenSize;
        LocalTensor<float> kInUbF32 = kOutUbF32[kvDataNum];
        LocalTensor<float> cosUbF32 = kInUbF32[this->hiddenSize];
        LocalTensor<float> sinUbF32 = cosUbF32[this->hiddenSize];
        Cast(cosUbF32, cosUb, RoundMode::CAST_NONE, this->hiddenSize);
        Cast(sinUbF32, sinUb, RoundMode::CAST_NONE, this->hiddenSize);
        PipeBarrier<PIPE_V>();
        uint64_t halfSize = this->hiddenSize / ROPE_LAST_DIM_SPLIT;
        Muls(sinUbF32, sinUbF32, -1.0f, halfSize);
        PipeBarrier<PIPE_V>();
        for (uint64_t r = 0; r < this->kvHeadNum; r++) {
            uint64_t rowOffset = r * this->hiddenSize;
            Cast(kInUbF32, kInUb[rowOffset], RoundMode::CAST_NONE, this->hiddenSize);
            PipeBarrier<PIPE_V>();
            Mul(kOutUbF32[rowOffset], kInUbF32[halfSize], sinUbF32, halfSize);
            Mul(kOutUbF32[rowOffset + halfSize], kInUbF32, sinUbF32[halfSize], halfSize);
            Mul(kInUbF32, kInUbF32, cosUbF32, this->hiddenSize);
            PipeBarrier<PIPE_V>();
            Add(kOutUbF32[rowOffset], kOutUbF32[rowOffset], kInUbF32, this->hiddenSize);
            PipeBarrier<PIPE_V>();
        }

        for (uint64_t r = 0; r < this->kvHeadNum; r++) {
            uint64_t rowOffset = r * this->hiddenSize;
#if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 3510) || (__NPU_ARCH__ == 5102))
            Div<float, DIV_PRECISION_0ULP>(kOutUbF32[rowOffset], kOutUbF32[rowOffset], quantScaleUb, this->hiddenSize);
#else
            Div(kOutUbF32[rowOffset], kOutUbF32[rowOffset], quantScaleUb, this->hiddenSize);
#endif
        }
        PipeBarrier<PIPE_V>();

        for (uint64_t r = 0; r < this->kvHeadNum; r++) {
            uint64_t rowOffset = r * this->hiddenSize;
            Add(kOutUbF32[rowOffset], kOutUbF32[rowOffset], quantOffsetUb, this->hiddenSize);
        }
        PipeBarrier<PIPE_V>();

        LocalTensor<int16_t> kOutUbS16 = kOutUbF32.ReinterpretCast<int16_t>();
        Cast(kOutUbS16, kOutUbF32, RoundMode::CAST_RINT, this->kvHeadNum * this->hiddenSize);
        PipeBarrier<PIPE_V>();
        LocalTensor<half> kOutUbF16 = kOutUbF32.ReinterpretCast<half>();
        Cast(kOutUbF16, kOutUbS16, RoundMode::CAST_NONE, this->kvHeadNum * this->hiddenSize);
        PipeBarrier<PIPE_V>();
        LocalTensor<int8_t> kOutUbS8 = kOutUbF32.ReinterpretCast<int8_t>();
        Cast(kOutUbS8, kOutUbF16, RoundMode::CAST_NONE, this->kvHeadNum * this->hiddenSize);

        event_t eventIdVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);

        DataCopy(kCacheGm, kOutUbS8, this->kvHeadNum * this->hiddenSize);
    }

    /* global memory address */
    GlobalTensor<half> inputGm;

    GlobalTensor<half> cosGm;
    GlobalTensor<half> sinGm;
    GlobalTensor<float> quantScaleGm;
    GlobalTensor<int32_t> quantOffsetGm;

    GlobalTensor<half> outputGm;
    GlobalTensor<int8_t> kCacheGm;
    GlobalTensor<int8_t> vCacheGm;

    GlobalTensor<int32_t> indiceGm;

    /* variable */
    uint64_t cacheSeqlen;
    uint64_t qHeadNum;
    uint64_t kvHeadNum;
    uint64_t hiddenSize;
    uint64_t qHiddenSize;
    uint64_t kHiddenSize;
    uint64_t vHiddenSize;
    uint64_t taskId;

    /* ascendc variable */
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> othersQueue;

    TQue<QuePosition::VECIN, 1> inQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
};
} // namespace RopeQuantKvcacheND
