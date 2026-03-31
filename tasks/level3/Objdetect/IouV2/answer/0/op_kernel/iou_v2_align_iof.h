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
 * \file iou_v2_align_iof.h
 * \brief
 */
#ifndef _IOU_V2_ALIGN_IOF_H
#define _IOU_V2_ALIGN_IOF_H

#include "kernel_operator.h"

namespace IouV2
{
    using namespace AscendC;
    constexpr uint8_t POS_NUM = 4;

    template <typename inType>
    class KernelIofV2Align
    {
    public:
        __aicore__ inline KernelIofV2Align() {}

        __aicore__ inline void Init(GM_ADDR bboxes, GM_ADDR gtboxes, GM_ADDR overlap,
                                    const IouV2TilingData *tilingData)
        {
            ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");
            this->eps = tilingData->eps;
            this->totalLength = tilingData->gtBoxLength;
            this->tileLen = tilingData->tileLength;
            this->doubleTileLen = 2 * tileLen;         // 2倍tileLen
            this->tripTileLen = 3 * tileLen;           // 3倍tileLen
            this->quadTileLen = 4 * tileLen;           // 4倍tileLen
            this->octaTileLen = 8 * tileLen;           // 8倍tileLen
            this->doubleTotalLength = 2 * totalLength; // 2倍totalLength
            this->tripTotalLength = 3 * totalLength;   // 3倍totalLength

            uint32_t blockId = GetBlockIdx();
            // blockOffsetLen是核间偏移量
            uint32_t blockOffsetLen = blockId > tilingData->frontCoreNum ? tileLen * (tilingData->frontCoreNum + tilingData->loopNum * blockId) : tileLen * (tilingData->loopNum + 1) * blockId;
            this->loopNum = blockId < tilingData->frontCoreNum ? (tilingData->loopNum + 1) : tilingData->loopNum;

            uint64_t bufferSize = tileLen * 4; // 输入shape为(4, n)
            box1Gm.SetGlobalBuffer((__gm__ inType *)bboxes + blockOffsetLen, bufferSize);
            box2Gm.SetGlobalBuffer((__gm__ inType *)gtboxes + blockOffsetLen, bufferSize);
            outGm.SetGlobalBuffer((__gm__ inType *)overlap + blockOffsetLen, tileLen);

            // 核内每次循环的大小，数据类型经过cast之后都是float
            pipe.InitBuffer(box1Que, 1, quadTileLen * 4); // float大小为4
            pipe.InitBuffer(box2Que, 1, quadTileLen * 4); // float大小为4
            pipe.InitBuffer(outQue, 1, tileLen * 4);      // float大小为4
            pipe.InitBuffer(tmpTensor, quadTileLen * 4);  // float大小为4

            if constexpr (!std::is_same<inType, float>::value)
            {
                pipe.InitBuffer(fp16Tensor, octaTileLen * 2); // sizeof(inType) = 2
            }
        }

        __aicore__ inline void Process()
        {
            for (int32_t bBoxLoop = 0; bBoxLoop < this->loopNum; ++bBoxLoop)
            {
                CopyIn(bBoxLoop);
                Compute();
                CopyOut(bBoxLoop);
            }
        }

    private:
        __aicore__ inline void CopyIn(uint32_t bBoxLoop)
        {
            LocalTensor<float> box1Local = box1Que.AllocTensor<float>();
            LocalTensor<float> box2Local = box2Que.AllocTensor<float>();
            if constexpr (!std::is_same<inType, float>::value)
            {
                LocalTensor<inType> fp16Buf = fp16Tensor.Get<inType>();
                event_t eventVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
                SetFlag<HardEvent::V_MTE2>(eventVToMTE2);
                WaitFlag<HardEvent::V_MTE2>(eventVToMTE2);
                event_t eventMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                SetFlag<HardEvent::MTE3_MTE2>(eventMTE3ToMTE2);
                WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3ToMTE2);
                for (uint8_t posId = 0; posId < POS_NUM; ++posId)
                { // 第一维输入为4
                    DataCopy(fp16Buf[tileLen * posId], box1Gm[bBoxLoop * tileLen + posId * totalLength], tileLen);
                    DataCopy(fp16Buf[tileLen * (posId + 4)], box2Gm[bBoxLoop * tileLen + posId * totalLength], tileLen); // 第一维输入为4
                }
                event_t eventMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(eventMTE2ToV);
                WaitFlag<HardEvent::MTE2_V>(eventMTE2ToV);
                Cast(box1Local, fp16Buf, RoundMode::CAST_NONE, quadTileLen);
                Cast(box2Local, fp16Buf[quadTileLen], RoundMode::CAST_NONE, quadTileLen);
            }
            else
            {
                for (uint8_t posId = 0; posId < POS_NUM; ++posId)
                { // 第一维输入为4
                    DataCopy(box1Local[tileLen * posId].ReinterpretCast<inType>(), box1Gm[bBoxLoop * tileLen + posId * totalLength], tileLen);
                    DataCopy(box2Local[tileLen * posId].ReinterpretCast<inType>(), box2Gm[bBoxLoop * tileLen + posId * totalLength], tileLen); // 第一维输入为4
                }
            }
            box1Que.EnQue(box1Local);
            box2Que.EnQue(box2Local);
        }

        __aicore__ inline void Compute()
        {
            LocalTensor<float> box1Local = box1Que.DeQue<float>();
            LocalTensor<float> box2Local = box2Que.DeQue<float>();
            LocalTensor<float> tmpBuffer = tmpTensor.Get<float>();
            Sub(tmpBuffer, box2Local[doubleTileLen], box2Local, doubleTileLen);
            Adds(tmpBuffer, tmpBuffer, eps, doubleTileLen);

            LocalTensor<float> outLocal = outQue.AllocTensor<float>();
            Mul(outLocal, tmpBuffer, tmpBuffer[tileLen], tileLen);

            Max(tmpBuffer, box1Local, box2Local, doubleTileLen);
            Min(tmpBuffer[doubleTileLen], box1Local[doubleTileLen], box2Local[doubleTileLen], doubleTileLen);
            Adds(tmpBuffer[doubleTileLen], tmpBuffer[doubleTileLen], eps, doubleTileLen);
            SubRelu(tmpBuffer, tmpBuffer[doubleTileLen], tmpBuffer, doubleTileLen);

            Mul(tmpBuffer, tmpBuffer, tmpBuffer[tileLen], tileLen);
            Div(outLocal, tmpBuffer, outLocal, tileLen);

            outQue.EnQue<float>(outLocal);
            box1Que.FreeTensor(box1Local);
            box2Que.FreeTensor(box2Local);
        }

        __aicore__ inline void CopyOut(uint32_t bBoxLoop)
        {
            LocalTensor<float> outLocal = outQue.DeQue<float>();
            if constexpr (!std::is_same<inType, float>::value)
            {
                LocalTensor<inType> fp16Buf = fp16Tensor.Get<inType>();
#if __CCE_AICORE__ == 200 // 310p
                Cast(fp16Buf, outLocal, RoundMode::CAST_NONE, tileLen);
#else
                Cast(fp16Buf, outLocal, RoundMode::CAST_RINT, tileLen);
#endif
                event_t eventVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(eventVToMTE3);
                WaitFlag<HardEvent::V_MTE3>(eventVToMTE3);
                DataCopy(outGm[bBoxLoop * tileLen], fp16Buf, tileLen);
            }
            else
            {
                DataCopy(outGm[bBoxLoop * tileLen], outLocal.ReinterpretCast<inType>(), tileLen);
            }
            outQue.FreeTensor(outLocal);
        }

    private:
        TPipe pipe;
        TQue<QuePosition::VECIN, 1> box1Que;
        TQue<QuePosition::VECIN, 1> box2Que;
        TQue<QuePosition::VECOUT, 1> outQue;
        TBuf<TPosition::VECCALC> tmpTensor;
        TBuf<TPosition::VECCALC> fp16Tensor;
        GlobalTensor<inType> box1Gm;
        GlobalTensor<inType> box2Gm;
        GlobalTensor<inType> outGm;
        uint64_t loopNum;
        uint64_t tileLen;
        uint64_t doubleTileLen;
        uint64_t tripTileLen;
        uint64_t quadTileLen;
        uint64_t octaTileLen;
        uint64_t totalLength;
        uint64_t doubleTotalLength;
        uint64_t tripTotalLength;
        float eps;
    };

} // namespace IouV2
#endif // _IOU_V2_ALIGN_IOF_H