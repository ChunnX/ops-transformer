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
 * \file iou_v2_not_align_iof.h
 * \brief
 */
#ifndef _IOU_V2_NOT_ALIGN_IOF_H
#define _IOU_V2_NOT_ALIGN_IOF_H

#include "kernel_operator.h"

namespace IouV2
{
    using namespace AscendC;

    template <typename inType>
    class KernelIofV2NotAlign
    {
    public:
        __aicore__ inline KernelIofV2NotAlign() {}

        __aicore__ inline void Init(GM_ADDR bboxes, GM_ADDR gtboxes, GM_ADDR overlap,
                                    const IouV2TilingData *tilingData)
        {
            ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");
            this->eps = tilingData->eps;
            this->bBoxLength = tilingData->bBoxLength;
            this->gtBoxLength = tilingData->gtBoxLength;
            this->tileLen = tilingData->tileLength;
            this->quadTileLen = 4 * tileLen; // 4倍tileLen
            this->loopTileLen = tilingData->subTileLen;
            this->doubleLoopTileLen = 2 * loopTileLen; // 2倍loopTileLen
            this->tripLoopTileLen = 3 * loopTileLen;   // 3倍loopTileLen
            this->quadLoopTileLen = 4 * loopTileLen;   // 4倍loopTileLen
            this->mulLen = tileLen * loopTileLen;
            this->doubleMulLen = 2 * mulLen; // 2倍mulLen
            this->tripMulLen = 3 * mulLen;   // 3倍mulLen
            this->quadMulLen = 4 * mulLen;   // 4倍mulLen
            this->pentaMulLen = 5 * mulLen;  // 5倍mulLen
            this->hexaMulLen = 6 * mulLen;   // 6倍mulLen
            this->septaMulLen = 7 * mulLen;  // 7倍mulLen
            this->octaMulLen = 8 * mulLen;   // 8倍mulLen
            this->totalLen = tileLen + loopTileLen;
            this->doubleTotalLen = 2 * totalLen; // 2倍totalLen
            this->tripTotalLen = 3 * totalLen;   // 3倍totalLen
            this->quadTotalLen = 4 * totalLen;   // 4倍totalLen
            this->loopTileNum = (bBoxLength + tileLen - 1) / tileLen;

            uint32_t blockId = GetBlockIdx();
            // blockOffsetLen是核间偏移量
            this->blockOffsetLen = blockId > tilingData->frontCoreNum ? tileLen * (tilingData->frontCoreNum + tilingData->loopNum * blockId) : tileLen * (tilingData->loopNum + 1) * blockId;
            this->loopNum = blockId < tilingData->frontCoreNum ? tilingData->loopNum + 1 : tilingData->loopNum;

            box1Gm.SetGlobalBuffer((__gm__ inType *)bboxes, loopTileNum * loopTileLen * 4);
            box2Gm.SetGlobalBuffer((__gm__ inType *)gtboxes + blockOffsetLen * 4, tileLen * 4);
            outGm.SetGlobalBuffer((__gm__ inType *)overlap + blockOffsetLen * bBoxLength, bBoxLength * tileLen);

            // 核内每次循环的大小，数据类型经过cast之后都是float
            pipe.InitBuffer(inQue, 1, octaMulLen * 4);     // float大小为4
            pipe.InitBuffer(outQue, 1, mulLen * 4);        // float大小为4
            pipe.InitBuffer(tmpTensor1, quadTotalLen * 4); // float大小为4
            pipe.InitBuffer(tmpTensor2, quadTotalLen * 4); // float大小为4

            if constexpr (!std::is_same<inType, float>::value)
            {
                pipe.InitBuffer(fp16Tensor, mulLen * 2); // sizeof(inType) = 2
            }
        }

        __aicore__ inline void Process()
        {
            for (int32_t gtBoxLoop = 0; gtBoxLoop < this->loopNum; ++gtBoxLoop)
            {
                gmOffset = gtBoxLoop * quadTileLen;
                for (int32_t bBoxLoop = loopTileNum - 1; bBoxLoop >= 0; --bBoxLoop)
                {
                    CopyIn(bBoxLoop);
                    Compute(bBoxLoop);
                    CopyOut(gtBoxLoop, bBoxLoop);
                }
            }
        }

    private:
        __aicore__ inline void CopyIn(uint32_t bBoxLoop)
        {
            LocalTensor<float> boxLocal = inQue.AllocTensor<float>();
            if constexpr (!std::is_same<inType, float>::value)
            {
                LocalTensor<inType> fp16Buf = fp16Tensor.Get<inType>();
                event_t eventMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                SetFlag<HardEvent::MTE3_MTE2>(eventMTE3ToMTE2);
                WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3ToMTE2);
                DataCopy(fp16Buf, box1Gm[bBoxLoop * quadLoopTileLen], quadLoopTileLen);
                DataCopy(fp16Buf[quadLoopTileLen], box2Gm[gmOffset], quadTileLen);
                event_t eventMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(eventMTE2ToV);
                WaitFlag<HardEvent::MTE2_V>(eventMTE2ToV);
                Cast(boxLocal, fp16Buf, RoundMode::CAST_NONE, quadTotalLen);
            }
            else
            {
                DataCopy(boxLocal.ReinterpretCast<inType>(), box1Gm[bBoxLoop * quadLoopTileLen], quadLoopTileLen);
                DataCopy(boxLocal[quadLoopTileLen].ReinterpretCast<inType>(), box2Gm[gmOffset], quadTileLen);
            }
            event_t eventMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2ToV);
            uint64_t rsvdCnt = 0;
            LocalTensor<float> tmpBuffer1 = tmpTensor1.Get<float>();
            GatherMask(tmpBuffer1, boxLocal, 3, true, quadTotalLen, {1, 1, 8, 8}, rsvdCnt);                 // src1Pattern = 3，每四个元素取第一个
            GatherMask(tmpBuffer1[totalLen], boxLocal, 4, true, quadTotalLen, {1, 1, 8, 8}, rsvdCnt);       // src1Pattern = 4，每四个元素取第二个
            GatherMask(tmpBuffer1[doubleTotalLen], boxLocal, 5, true, quadTotalLen, {1, 1, 8, 8}, rsvdCnt); // src1Pattern = 5，每四个元素取第三个
            GatherMask(tmpBuffer1[tripTotalLen], boxLocal, 6, true, quadTotalLen, {1, 1, 8, 8}, rsvdCnt);   // src1Pattern = 6，每四个元素取第四个
            inQue.FreeTensor(boxLocal);
            LocalTensor<float> tmpBuffer2 = tmpTensor2.Get<float>();
            uint16_t blockLen = loopTileLen / 8; // 8个数组成一个block
            uint16_t srcStride = tileLen / 8;    // 8个数组成一个block
            DataCopy(tmpBuffer2, tmpBuffer1, {4, blockLen, srcStride, 0});
            blockLen = tileLen / 8;      // 8个数组成一个block
            srcStride = loopTileLen / 8; // 8个数组成一个block
            DataCopy(tmpBuffer2[quadLoopTileLen], tmpBuffer1[loopTileLen], {4, blockLen, srcStride, 0});
        }

        __aicore__ inline void Compute(uint32_t bBoxLoop)
        {
            LocalTensor<float> boxLocal = inQue.AllocTensor<float>();
            LocalTensor<float> tmpBuffer2 = tmpTensor2.Get<float>();
            event_t eventMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2ToV);

            Duplicate(boxLocal, (float)0.0, octaMulLen);
            uint8_t bBoxRepStride = loopTileLen / 8; // 8个数组成一个block
            BinaryRepeatParams bBoxAddRepParams{1, 1, 1, bBoxRepStride, bBoxRepStride, 0};
            Add(boxLocal, boxLocal, tmpBuffer2, loopTileLen, tileLen, bBoxAddRepParams);
            Add(boxLocal[mulLen], boxLocal[mulLen], tmpBuffer2[loopTileLen], loopTileLen, tileLen, bBoxAddRepParams);
            Add(boxLocal[doubleMulLen], boxLocal[doubleMulLen], tmpBuffer2[doubleLoopTileLen], loopTileLen, tileLen, bBoxAddRepParams);
            Add(boxLocal[tripMulLen], boxLocal[tripMulLen], tmpBuffer2[tripLoopTileLen], loopTileLen, tileLen, bBoxAddRepParams);

            uint16_t gtBoxRepTimes = loopTileLen / 8; // 8个数组成一个block
            uint8_t brcbRepTimes = tileLen / 2;
            BrcbRepeatParams gtBoxBrcbRepParams{gtBoxRepTimes, static_cast<uint16_t>(loopTileLen)};
            for (uint16_t repTime = 0; repTime < gtBoxRepTimes; ++repTime)
            {
                Brcb(boxLocal[quadMulLen + repTime * 8], tmpBuffer2[quadLoopTileLen], brcbRepTimes, gtBoxBrcbRepParams); // 8个数组成一个block
            }
            PipeBarrier<PIPE_V>();
            Max(boxLocal, boxLocal, boxLocal[quadMulLen], mulLen);
            Max(boxLocal[mulLen], boxLocal[mulLen], boxLocal[pentaMulLen], mulLen);
            Min(boxLocal[doubleMulLen], boxLocal[doubleMulLen], boxLocal[hexaMulLen], mulLen);
            Min(boxLocal[tripMulLen], boxLocal[tripMulLen], boxLocal[septaMulLen], mulLen);
            Adds(boxLocal[doubleMulLen], boxLocal[doubleMulLen], eps, doubleMulLen);
            PipeBarrier<PIPE_V>();
            SubRelu(boxLocal, boxLocal[doubleMulLen], boxLocal, doubleMulLen);

            Sub(boxLocal[quadMulLen], boxLocal[hexaMulLen], boxLocal[quadMulLen], doubleMulLen);
            PipeBarrier<PIPE_V>();
            Adds(boxLocal[quadMulLen], boxLocal[quadMulLen], eps, doubleMulLen);
            PipeBarrier<PIPE_V>();
            Mul(boxLocal[quadMulLen], boxLocal[quadMulLen], boxLocal[pentaMulLen], mulLen);
            Mul(boxLocal[septaMulLen], boxLocal, boxLocal[mulLen], mulLen);

            LocalTensor<float> outLocal = outQue.AllocTensor<float>();
            PipeBarrier<PIPE_V>();
            Div(outLocal, boxLocal[septaMulLen], boxLocal[quadMulLen], mulLen);

            outQue.EnQue<float>(outLocal);
            inQue.FreeTensor(boxLocal);
        }

        template <typename T>
        __aicore__ inline void CommonCopyOut(LocalTensor<T> &tmpLocal, uint32_t gtBoxLoop, uint32_t bBoxLoop)
        {
            uint64_t num1 = tileLen;
            if (gtBoxLoop == (loopNum - 1) && (gtBoxLength - blockOffsetLen - gtBoxLoop * tileLen) < tileLen)
            {
                num1 = gtBoxLength - blockOffsetLen - gtBoxLoop * tileLen;
            }
            uint64_t offset;
            uint64_t gmLenEachLoop = bBoxLength * tileLen;
            uint64_t bBoxNotAlign = bBoxLength - bBoxLoop * loopTileLen;
            uint64_t tmpGmOffset{0};
            uint64_t tmpUbOffset{0};
            uint64_t addNum{0};
            uint64_t headNum{0};
            for (uint64_t i = 0; i < num1; ++i)
            {
                offset = gtBoxLoop * gmLenEachLoop + bBoxLoop * loopTileLen + i * bBoxLength;
                PipeBarrier<PIPE_MTE3>();
                if (bBoxLoop == (loopTileNum - 1) && bBoxNotAlign > 0 && (offset + loopTileLen) > gmLenEachLoop * (gtBoxLoop + 1))
                {
                    event_t eventMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
                    SetFlag<HardEvent::MTE3_S>(eventMTE3ToS);
                    WaitFlag<HardEvent::MTE3_S>(eventMTE3ToS);
                    if (addNum == 0)
                    {
                        headNum = loopTileLen - (tileLen - i) * bBoxNotAlign;
                        if (headNum != bBoxNotAlign)
                        {
                            for (uint64_t index = 0; index < headNum; ++index)
                            {
                                T tensorValue = tmpLocal.GetValue(tmpUbOffset + bBoxNotAlign - headNum + index);
                                tmpLocal.SetValue(tmpUbOffset + index, tensorValue);
                            }
                        }
                        addNum += headNum;
                    }
                    for (uint64_t index = 0; index < bBoxNotAlign; ++index)
                    {
                        T tensorValue = tmpLocal.GetValue(loopTileLen * i + index);
                        tmpLocal.SetValue(tmpUbOffset + addNum + index, tensorValue);
                    }
                    addNum += bBoxNotAlign;
                    if (i == num1 - 1)
                    {
                        DataCopy(outGm[tmpGmOffset + bBoxLength - headNum], tmpLocal[tmpUbOffset], loopTileLen);
                    }
                }
                else
                {
                    DataCopy(outGm[offset], tmpLocal[loopTileLen * i], loopTileLen);
                    tmpGmOffset = offset;
                    tmpUbOffset = loopTileLen * i;
                }
            }
        }

        __aicore__ inline void CopyOut(uint32_t gtBoxLoop, uint32_t bBoxLoop)
        {
            LocalTensor<float> outLocal = outQue.DeQue<float>();

            if constexpr (!std::is_same<inType, float>::value)
            {
                LocalTensor<inType> fp16Buf = fp16Tensor.Get<inType>();
#if __CCE_AICORE__ == 200 // 310p
                Cast(fp16Buf, outLocal, RoundMode::CAST_NONE, mulLen);
#else
                Cast(fp16Buf, outLocal, RoundMode::CAST_RINT, mulLen);
#endif
                event_t eventVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(eventVToMTE3);
                WaitFlag<HardEvent::V_MTE3>(eventVToMTE3);
                CommonCopyOut<inType>(fp16Buf, gtBoxLoop, bBoxLoop);
            }
            else
            {
                CommonCopyOut<float>(outLocal, gtBoxLoop, bBoxLoop);
            }
            outQue.FreeTensor(outLocal);
        }

    private:
        TPipe pipe;
        TQue<QuePosition::VECIN, 1> inQue;
        TQue<QuePosition::VECOUT, 1> outQue;
        TBuf<TPosition::VECCALC> tmpTensor1;
        TBuf<TPosition::VECCALC> tmpTensor2;
        TBuf<TPosition::VECCALC> fp16Tensor;
        GlobalTensor<inType> box1Gm;
        GlobalTensor<inType> box2Gm;
        GlobalTensor<inType> outGm;
        uint64_t loopNum;
        uint64_t tileLen;
        uint64_t quadTileLen;
        uint64_t mulLen;
        uint64_t doubleMulLen;
        uint64_t tripMulLen;
        uint64_t quadMulLen;
        uint64_t pentaMulLen;
        uint64_t hexaMulLen;
        uint64_t septaMulLen;
        uint64_t octaMulLen;
        uint64_t gmOffset;
        uint64_t loopTileNum;
        uint64_t bBoxLength;
        uint64_t gtBoxLength;
        uint64_t loopTileLen;
        uint64_t doubleLoopTileLen;
        uint64_t tripLoopTileLen;
        uint64_t quadLoopTileLen;
        uint64_t totalLen;
        uint64_t doubleTotalLen;
        uint64_t tripTotalLen;
        uint64_t quadTotalLen;
        uint32_t blockOffsetLen;
        float eps;
    };

} // namespace IouV2
#endif // _IOU_V2_NOT_ALIGN_IOF_H