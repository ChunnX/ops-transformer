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
 * \file upsample_nearest_exact2d_grad_transpose.h
 * \brief
 */

#ifndef UPSAMPLE_NEAREST_EXACT2D_GRAD_TRANSPOSE
#define UPSAMPLE_NEAREST_EXACT2D_GRAD_TRANSPOSE

#include <type_traits>
#include "kernel_operator.h"

namespace UpSampleNearestExact2dGrad {
using namespace AscendC;

struct IdxInfo {
    int64_t hIdx;
    int64_t hDown;
    int64_t hUp;
    int64_t wIdx;
    int64_t srcStartW;
    int64_t srcEndW;
    int64_t batch;
    uint16_t batchNum;
    uint16_t srcGap;
    uint16_t dstGap;
};

constexpr int32_t BUFFERS_NUM = 1;
constexpr float ONE_FLOAT = 1;
constexpr uint16_t MAX_UINT16 = 65535; // 2**16-1
constexpr int64_t C0 = 16;
constexpr uint16_t BLOCK_C0_FP32 = 2;
constexpr uint16_t BLOCK_C0_FP16 = 1;
constexpr int32_t INDEX_LEN = 128;

template <typename T, bool BigToSmall, bool WAlign>
class UpSampleNearestExact2dGradTranspose {
public:
    __aicore__ inline UpSampleNearestExact2dGradTranspose(){};
    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR output, GM_ADDR workspace, TPipe* pipeIn,
        UpsampleNearestExact2dGradTransposeTilingData* tilingData);
    __aicore__ inline void Process();

private:
    template <typename T1>
    __aicore__ inline T1 Ceil(T1 x)
    {
        int32_t floor_x = int32_t(x);
        if (x == floor_x) {
            return floor_x;
        }
        return floor_x + 1;
    };

    template <typename T1>
    __aicore__ inline T1 Min(T1 a, T1 b)
    {
        return a < b ? a : b;
    };

    template <typename T1, typename T2>
    __aicore__ inline T1 CeilA2B(T1 a, T2 b)
    {
        if (b == 0) {
            return a;
        }
        return (a + b - 1) / b;
    };

    __aicore__ inline void ParseTilingData(UpsampleNearestExact2dGradTransposeTilingData* tilingData);
    __aicore__ inline void ComputeBigToSmall(int64_t hIdx, int64_t hDown, int64_t hUp);
    __aicore__ inline void ComputeNotBigToSmall(int64_t hIdx, int64_t hDown, int64_t hUp);
    __aicore__ inline void CopyInAndDoComputeBigToSmall(
        struct IdxInfo idxInfo, LocalTensor<T> srcLocal, LocalTensor<T> dstLocal, LocalTensor<T> inLocal,
        LocalTensor<float> castOutLocal);
    __aicore__ inline void CopyInAndDoComputeNotBigToSmall(
        struct IdxInfo idxInfo, LocalTensor<T> srcLocal, LocalTensor<T> dstLocal, LocalTensor<T> inLocal,
        LocalTensor<float> castOutLocal);
    __aicore__ inline void CopyOutNotBigToSmall(
        struct IdxInfo idxInfo, int64_t dstWLen, int64_t resizeW, LocalTensor<int32_t> srcIndexLocal);
    __aicore__ inline void calIndex(int64_t startIdx, int64_t len, LocalTensor<int32_t> srcIndexLocal);

private:
    TBuf<QuePosition::VECCALC> dstIndexBuf;
    TBuf<QuePosition::VECCALC> srcIndexBuf;

    TQue<QuePosition::VECIN, BUFFERS_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFERS_NUM> outQueue;

    GlobalTensor<T> inTensorsGM;
    GlobalTensor<T> outTensorsGM;
    TPipe* pipe;

    int64_t blockIdx = 0;
    int64_t slide_size = 0;

    int64_t input_shapes[4] = {0, 0, 0, 0};
    int64_t output_shapes[4] = {0, 0, 0, 0};

    float scaleW = 0.0f;
    float scaleH = 0.0f;
    int64_t oh = 0;
    int64_t ow = 0;
    int64_t ih = 0;
    int64_t iw = 0;
    int64_t needCoreNum = 0;
    int64_t startW = 0;
    int64_t endW = 0;
    int64_t startH = 0;
    int64_t endH = 0;
    int64_t startBatches = 0;
    int64_t endBatches = 0;
    int64_t srcStride = 0;
    int64_t dstStride = 0;

    bool isWResizeSmall = false;
    bool isHResizeSmall = false;
    bool isWAlign = false;
    bool isHAlign = false;

    int64_t slideSize = 0;
    int64_t slideSizeH = 0;
    int64_t slideSizeW = 0;
    uint16_t blockLen = 0;
};

template <typename T, bool BigToSmall, bool WAlign>
__aicore__ inline void UpSampleNearestExact2dGradTranspose<T, BigToSmall, WAlign>::Init(
    GM_ADDR input, GM_ADDR output, GM_ADDR workspace, TPipe* pipeIn,
    UpsampleNearestExact2dGradTransposeTilingData* tilingData)
{
    pipe = pipeIn;
    blockIdx = GetBlockIdx();
    ParseTilingData(tilingData);
    pipe->InitBuffer(dstIndexBuf, INDEX_LEN * sizeof(float));
    pipe->InitBuffer(srcIndexBuf, INDEX_LEN * sizeof(int32_t));
    pipe->InitBuffer(inQueue, BUFFERS_NUM, slideSize * sizeof(float));
    pipe->InitBuffer(outQueue, BUFFERS_NUM, slideSize * sizeof(float));

    inTensorsGM.SetGlobalBuffer((__gm__ T*)input);
    outTensorsGM.SetGlobalBuffer((__gm__ T*)output);
};

template <typename T, bool BigToSmall, bool WAlign>
__aicore__ inline void UpSampleNearestExact2dGradTranspose<T, BigToSmall, WAlign>::Process()
{
    if (blockIdx >= needCoreNum) {
        return;
    }
    for (int64_t i = startH; i < endH; i += slideSizeH) {
        // 获取当前行的映射上界和下界
        int64_t hDown = Ceil(i * scaleH);
        int64_t hUp = Ceil((i + 1) * scaleH);
        hUp = Min(hUp, ih);
        if (hDown - hUp == 0) {
            continue;
        }
        if constexpr (BigToSmall) {
            // w缩小
            ComputeBigToSmall(i, hDown, hUp);
        } else {
            ComputeNotBigToSmall(i, hDown, hUp);
        }
    }
}
template <typename T, bool BigToSmall, bool WAlign>
__aicore__ inline void UpSampleNearestExact2dGradTranspose<T, BigToSmall, WAlign>::ComputeBigToSmall(
    int64_t hIdx, int64_t hDown, int64_t hUp)
{
    for (int64_t wIdx = startW; wIdx < endW; wIdx += slideSizeW) {
        int64_t srcStartW = Ceil(wIdx * scaleW);
        int64_t srcEndW = Ceil((wIdx + 1) * scaleW);
        srcEndW = Min(srcEndW, iw);
        int64_t batch = startBatches;
        uint16_t srcGap = (ih * iw - 1) * blockLen;
        uint16_t dstGap = (oh * ow - 1) * blockLen;
        while (batch < endBatches) {
            uint16_t batchNum = slideSize / C0;
            batchNum = Min(static_cast<int64_t>(batchNum), endBatches - batch);
            LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
            LocalTensor<T> srcLocal;
            LocalTensor<T> inLocal;
            LocalTensor<float> castOutLocal;
            if constexpr (std::is_same<T, float>::value) {
                srcLocal = inQueue.AllocTensor<float>();
                Duplicate(dstLocal, static_cast<float>(0), C0 * batchNum);
            } else {
                inLocal = inQueue.AllocTensor<T>();
                srcLocal = inLocal[slideSize];
                castOutLocal = dstLocal.template ReinterpretCast<float>();
                Duplicate(castOutLocal, static_cast<float>(0), C0 * batchNum);
            }
            struct IdxInfo idxInfo = {hIdx,    hDown, hUp,      wIdx,   srcStartW,
                                      srcEndW, batch, batchNum, srcGap, dstGap};
            CopyInAndDoComputeBigToSmall(idxInfo, srcLocal, dstLocal, inLocal, castOutLocal);

            dstLocal = outQueue.DeQue<T>();
            if (dstStride == 0) {
                for (int64_t n = batch; n < batch + batchNum; n++) {
                    int64_t outOffset = (n * oh * ow + hIdx * ow + wIdx) * C0;
                    DataCopy(outTensorsGM[outOffset], dstLocal[(n - batch) * C0], {1, blockLen, 0, 0});
                }
            } else {
                int64_t outOffset = (batch * oh * ow + hIdx * ow + wIdx) * C0;
                DataCopy(outTensorsGM[outOffset], dstLocal, {batchNum, blockLen, 0, dstGap});
            }
            event_t eventID2 = static_cast<event_t>(pipe->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventID2);
            WaitFlag<HardEvent::MTE3_V>(eventID2);
            if constexpr (std::is_same<T, float>::value) {
                inQueue.FreeTensor(srcLocal);
            } else {
                inQueue.FreeTensor(inLocal);
            }
            outQueue.FreeTensor(dstLocal);
            batch += batchNum;
        }
    }
}

template <typename T, bool BigToSmall, bool WAlign>
__aicore__ inline void UpSampleNearestExact2dGradTranspose<T, BigToSmall, WAlign>::ComputeNotBigToSmall(
    int64_t hIdx, int64_t hDown, int64_t hUp)
{
    int64_t resizeW = 1;
    if constexpr (WAlign) {
        resizeW = ow / iw;
    }
    for (int64_t wIdx = startW; wIdx < endW; wIdx += slideSizeW) {
        int64_t dstWLen = Min(slideSizeW, endW - wIdx);
        LocalTensor<int32_t> srcIndexLocal = srcIndexBuf.Get<int32_t>();
        calIndex(wIdx, dstWLen + 1, srcIndexLocal);

        int64_t srcStartW = srcIndexLocal.GetValue(0);
        int64_t srcEndW = srcIndexLocal.GetValue(dstWLen);
        int64_t srcWLen = srcEndW - srcStartW;
        if (srcWLen < 1) {
            continue;
        }
        int64_t batch = startBatches;
        uint16_t srcGap = (ih * iw - srcWLen) * blockLen;
        uint16_t dstGap = (oh * ow - 1) * blockLen;
        while (batch < endBatches) {
            uint16_t batchNum = slideSize / C0 / srcWLen;
            batchNum = Min(static_cast<int64_t>(batchNum), endBatches - batch);
            LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
            LocalTensor<T> srcLocal;
            LocalTensor<T> inLocal;
            LocalTensor<float> castInLocal;
            LocalTensor<float> castOutLocal;
            if constexpr (std::is_same<T, float>::value) {
                srcLocal = inQueue.AllocTensor<float>();
                Duplicate(dstLocal, static_cast<float>(0), batchNum * srcWLen * C0);
            } else {
                inLocal = inQueue.AllocTensor<T>();
                srcLocal = inLocal[slideSize];
                castInLocal = inLocal.template ReinterpretCast<float>();
                castOutLocal = dstLocal.template ReinterpretCast<float>();
                Duplicate(castOutLocal, static_cast<float>(0), batchNum * srcWLen * C0);
            }
            struct IdxInfo idxInfo = {hIdx,    hDown, hUp,      wIdx,   srcStartW,
                                      srcEndW, batch, batchNum, srcGap, dstGap};
            CopyInAndDoComputeNotBigToSmall(idxInfo, srcLocal, dstLocal, inLocal, castOutLocal);
            if constexpr (std::is_same<T, float>::value) {
                inQueue.FreeTensor(srcLocal);
            } else {
                inQueue.FreeTensor(inLocal);
            }
            CopyOutNotBigToSmall(idxInfo, dstWLen, resizeW, srcIndexLocal);
            batch += batchNum;
        }
    }
}

template <typename T, bool BigToSmall, bool WAlign>
__aicore__ inline void UpSampleNearestExact2dGradTranspose<T, BigToSmall, WAlign>::CopyInAndDoComputeBigToSmall(
    struct IdxInfo idxInfo, LocalTensor<T> srcLocal, LocalTensor<T> dstLocal, LocalTensor<T> inLocal,
    LocalTensor<float> castOutLocal)
{
    LocalTensor<float> castInLocal;
    if constexpr (!std::is_same<T, float>::value) {
        castInLocal = inLocal.template ReinterpretCast<float>();
    }
    for (int64_t h = idxInfo.hDown; h < idxInfo.hUp; h++) {
        for (int64_t w = idxInfo.srcStartW; w < idxInfo.srcEndW; w++) {
            if (srcStride == 0) {
                for (int64_t n = idxInfo.batch; n < idxInfo.batch + idxInfo.batchNum; n++) {
                    int64_t inOffset = (n * ih * iw + h * iw + w) * C0;
                    DataCopy(srcLocal[(n - idxInfo.batch) * C0], inTensorsGM[inOffset], {1, blockLen, 0, 0});
                }
            } else {
                int64_t inOffset = (idxInfo.batch * ih * iw + h * iw + w) * C0;
                DataCopy(srcLocal, inTensorsGM[inOffset], {idxInfo.batchNum, blockLen, idxInfo.srcGap, 0});
            }
            PipeBarrier<PIPE_MTE2>();
            if constexpr (std::is_same<T, float>::value) {
                inQueue.EnQue(srcLocal);
                srcLocal = inQueue.DeQue<float>();
                Add(dstLocal, dstLocal, srcLocal, C0 * idxInfo.batchNum);
            } else {
                inQueue.EnQue(inLocal);
                srcLocal = inQueue.DeQue<T>()[slideSize];
                Cast(castInLocal, srcLocal, RoundMode::CAST_NONE, C0 * idxInfo.batchNum);
                PipeBarrier<PIPE_V>();
                Add(castOutLocal, castOutLocal, castInLocal, C0 * idxInfo.batchNum);
            }
            event_t eventID1 = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_MTE2));
            SetFlag<HardEvent::V_MTE2>(eventID1);
            WaitFlag<HardEvent::V_MTE2>(eventID1);
        }
    }
    if constexpr (!std::is_same<T, float>::value) {
        Cast(dstLocal, castOutLocal, RoundMode::CAST_RINT, C0 * idxInfo.batchNum);
    }
    outQueue.EnQue(dstLocal);
}

template <typename T, bool BigToSmall, bool WAlign>
__aicore__ inline void UpSampleNearestExact2dGradTranspose<T, BigToSmall, WAlign>::CopyInAndDoComputeNotBigToSmall(
    struct IdxInfo idxInfo, LocalTensor<T> srcLocal, LocalTensor<T> dstLocal, LocalTensor<T> inLocal,
    LocalTensor<float> castOutLocal)
{
    int64_t srcWLen = idxInfo.srcEndW - idxInfo.srcStartW;
    LocalTensor<float> castInLocal;
    if constexpr (!std::is_same<T, float>::value) {
        castInLocal = inLocal.template ReinterpretCast<float>();
    }
    for (int64_t h = idxInfo.hDown; h < idxInfo.hUp; h++) {
        if (srcStride == 0) {
            for (int64_t n = idxInfo.batch; n < idxInfo.batch + idxInfo.batchNum; n++) {
                int64_t inOffset = (n * ih * iw + h * iw + idxInfo.srcStartW) * C0;
                DataCopy(
                    srcLocal[(n - idxInfo.batch) * srcWLen * C0], inTensorsGM[inOffset],
                    {1, static_cast<uint16_t>(blockLen * srcWLen), 0, 0});
            }
        } else {
            int64_t inOffset = (idxInfo.batch * ih * iw + h * iw + idxInfo.srcStartW) * C0;
            DataCopy(
                srcLocal, inTensorsGM[inOffset],
                {idxInfo.batchNum, static_cast<uint16_t>(blockLen * srcWLen), idxInfo.srcGap, 0});
        }
        PipeBarrier<PIPE_MTE2>();
        if constexpr (std::is_same<T, float>::value) {
            inQueue.EnQue(srcLocal);
            srcLocal = inQueue.DeQue<float>();
            Add(dstLocal, dstLocal, srcLocal, idxInfo.batchNum * srcWLen * C0);
        } else {
            inQueue.EnQue(inLocal);
            srcLocal = inQueue.DeQue<T>()[slideSize];
            Cast(castInLocal, srcLocal, RoundMode::CAST_NONE, idxInfo.batchNum * srcWLen * C0);
            PipeBarrier<PIPE_V>();
            Add(castOutLocal, castOutLocal, castInLocal, idxInfo.batchNum * srcWLen * C0);
        }
        event_t eventID1 = static_cast<event_t>(pipe->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventID1);
        WaitFlag<HardEvent::V_MTE2>(eventID1);
    }
    if constexpr (!std::is_same<T, float>::value) {
        Cast(dstLocal, castOutLocal, RoundMode::CAST_RINT, idxInfo.batchNum * srcWLen * C0);
    }
    outQueue.EnQue(dstLocal);
}

template <typename T, bool BigToSmall, bool WAlign>
__aicore__ inline void UpSampleNearestExact2dGradTranspose<T, BigToSmall, WAlign>::CopyOutNotBigToSmall(
    struct IdxInfo idxInfo, int64_t dstWLen, int64_t resizeW, LocalTensor<int32_t> srcIndexLocal)
{
    int64_t srcWLen = idxInfo.srcEndW - idxInfo.srcStartW;
    LocalTensor<T> dstLocal = outQueue.DeQue<T>();
    if constexpr (WAlign) {
        if (dstStride == 0) {
            for (int64_t n = idxInfo.batch; n < idxInfo.batch + idxInfo.batchNum; n++) {
                int64_t outOffset = (n * oh * ow + idxInfo.hIdx * ow + idxInfo.wIdx) * C0;
                DataCopy(
                    outTensorsGM[outOffset], dstLocal[(idxInfo.batch - n) * srcWLen * C0],
                    {static_cast<uint16_t>(srcWLen), blockLen, 0, static_cast<uint16_t>((resizeW - 1) * blockLen)});
            }
        } else {
            int64_t baseOutOffset = (idxInfo.batch * oh * ow + idxInfo.hIdx * ow + idxInfo.wIdx) * C0;
            for (int64_t m = 0; m < srcWLen; m++) {
                int64_t outOffset = baseOutOffset + m * resizeW * C0;
                DataCopy(
                    outTensorsGM[outOffset], dstLocal[m * C0],
                    {idxInfo.batchNum, blockLen, static_cast<uint16_t>((srcWLen - 1) * blockLen), idxInfo.dstGap});
            }
        }
    } else {
        for (int64_t m = 0; m < dstWLen; m++) {
            if (srcIndexLocal.GetValue(m) == srcIndexLocal.GetValue(m + 1)) {
                continue;
            }
            if (dstStride == 0) {
                for (int64_t n = idxInfo.batch; n < idxInfo.batch + idxInfo.batchNum; n++) {
                    int64_t outOffset = (n * oh * ow + idxInfo.hIdx * ow + idxInfo.wIdx + m) * C0;
                    DataCopy(
                        outTensorsGM[outOffset],
                        dstLocal[(n - idxInfo.batch) * srcWLen * C0 + (srcIndexLocal.GetValue(m) - idxInfo.srcStartW) * C0],
                        {1, blockLen, 0, 0});
                }
            } else {
                int64_t outOffset = (idxInfo.batch * oh * ow + idxInfo.hIdx * ow + idxInfo.wIdx + m) * C0;
                DataCopy(
                    outTensorsGM[outOffset], dstLocal[(srcIndexLocal.GetValue(m) - idxInfo.srcStartW) * C0],
                    {idxInfo.batchNum, blockLen, static_cast<uint16_t>((srcWLen - 1) * blockLen), idxInfo.dstGap});
            }
        }
    }

    event_t eventID2 = static_cast<event_t>(pipe->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventID2);
    WaitFlag<HardEvent::MTE3_V>(eventID2);
    outQueue.FreeTensor(dstLocal);
}

template <typename T, bool BigToSmall, bool WAlign>
__aicore__ inline void UpSampleNearestExact2dGradTranspose<T, BigToSmall, WAlign>::calIndex(
    int64_t startIdx, int64_t len, LocalTensor<int32_t> srcIndexLocal)
{
    LocalTensor<float> dstIndexLocal = dstIndexBuf.Get<float>();
    Arange(dstIndexLocal, static_cast<float>(startIdx), ONE_FLOAT, len);
    PipeBarrier<PIPE_V>();
    Muls(dstIndexLocal, dstIndexLocal, scaleW, len);
    PipeBarrier<PIPE_V>();
    Cast(srcIndexLocal, dstIndexLocal, RoundMode::CAST_CEIL, len);
    PipeBarrier<PIPE_V>();
    Mins(srcIndexLocal, srcIndexLocal, static_cast<int32_t>(iw), len);
}

template <typename T, bool BigToSmall, bool WAlign>
__aicore__ inline void UpSampleNearestExact2dGradTranspose<T, BigToSmall, WAlign>::ParseTilingData(
    UpsampleNearestExact2dGradTransposeTilingData* tilingData)
{
    if constexpr (std::is_same<T, float>::value) {
        blockLen = BLOCK_C0_FP32;
    } else {
        blockLen = BLOCK_C0_FP16;
    }
    scaleH = tilingData->scaleH;
    scaleW = tilingData->scaleW;
    for (int8_t i = 0; i < 4; i++) {
        input_shapes[i] = tilingData->input_shapes[i];
        output_shapes[i] = tilingData->output_shapes[i];
    }
    oh = output_shapes[2];
    ow = output_shapes[3];
    ih = input_shapes[2];
    iw = input_shapes[3];
    if (ow * oh <= MAX_UINT16 / 2) {
        dstStride = 1;
    }
    if (iw * ih <= MAX_UINT16 / 2) {
        srcStride = 1;
    }

    needCoreNum = tilingData->needCoreNum;
    startW = tilingData->startW[blockIdx];
    endW = tilingData->endW[blockIdx];
    startH = tilingData->startH[blockIdx];
    endH = tilingData->endH[blockIdx];
    startBatches = tilingData->startBatches[blockIdx];
    endBatches = tilingData->endBatches[blockIdx];

    isWResizeSmall = tilingData->isWResizeSmall;
    isHResizeSmall = tilingData->isHResizeSmall;
    isWAlign = tilingData->isWAlign;
    isHAlign = tilingData->isHAlign;

    slideSize = tilingData->slideSize;
    slideSizeH = tilingData->slideSizeH;
    slideSizeW = tilingData->slideSizeW;
}

} // namespace UpSampleNearestExact2dGrad
#endif
