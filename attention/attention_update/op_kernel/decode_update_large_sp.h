/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file decode_update_large_sp.h
 * \brief Template class for sp > 16 (large sp) scenario, isolated from original sp <= 16 path.
 *        This allows independent optimization without affecting the original performance.
 */

#ifndef ASCENDC_ATTENTION_UPDATE_DECODE_UPDATE_LARGE_SP_H_
#define ASCENDC_ATTENTION_UPDATE_DECODE_UPDATE_LARGE_SP_H_

#include <limits>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "decode_update_tiling.h"

using namespace AscendC;

namespace AttentionUpdate {
static constexpr uint32_t LARGE_SP_BUFFER_NUM = 2;
static constexpr uint32_t LARGE_SP_NUM0 = 0;
static constexpr uint32_t LARGE_SP_NUM1 = 1;
static constexpr uint32_t LARGE_SP_NUM2 = 2;
static constexpr uint32_t LARGE_SP_NUM7 = 7;
static constexpr uint32_t LARGE_SP_NUM8 = 8;
static constexpr uint32_t LARGE_SP_NUM16 = 16;
static constexpr uint32_t LARGE_SP_NUM32 = 32;
static constexpr uint32_t LARGE_SP_NUM64 = 64;
static constexpr uint32_t LARGE_SP_NUM255 = 255;
static constexpr uint32_t LARGE_SP_NUM256 = 256;
static constexpr uint32_t LARGE_SP_MAX_EFFECTIVE_SP = 16;
static constexpr float LARGE_SP_POS_INF = std::numeric_limits<float>::infinity();
static constexpr float LARGE_SP_NEG_INF = -std::numeric_limits<float>::infinity();
static constexpr float LARGE_SP_DUMMY_LSE = -1e38f;
static constexpr uint32_t LARGE_SP_MAX_UB_SIZE = 188 * 1024;
static const uint16_t LARGE_SP_ALIGNED_TO_8 = 8;
static const int32_t LARGE_SP_ALIGNED_TO_2 = 2;
static const uint32_t LARGE_SP_SPLIT_TO_2 = 2;
static const uint32_t LARGE_SP_ELEM_PER_256B = LARGE_SP_NUM256 / sizeof(float);

template <typename lseType, typename outType>
class DecodeUpdateLargeSp {
public:
    __aicore__ inline DecodeUpdateLargeSp() {}
    __aicore__ inline void Init(GM_ADDR lse, GM_ADDR in, GM_ADDR out, GM_ADDR lesout, const DecodeUpdateTilingData *tdata)
    {
        this->lsePtr = GetTensorPtr(lse);
        this->inPtr = GetTensorPtr(in);
        ParseTilingData(tdata);
        CalcTileLength();
        outGm.SetGlobalBuffer((__gm__ outType *)out, totalLength * hDim);
        lseoutGm.SetGlobalBuffer((__gm__ lseType *)lesout, totalLength);
        InitBuffers();
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->loopCount; i++) {
            if (lastLength && i == this->loopCount - 1) {
                curLength = lastLength;
            }
            LocalTensor<float> outAccLocal = lseexpBroadcastBuffer.Get<float>();
            for (uint32_t spLoop = 0; spLoop < spLoopCount; spLoop++) {
                uint32_t currentSp;
                uint32_t effectiveSp;
                if (spLoop == 0) {
                    currentSp = spFirstLoop;
                    effectiveSp = 1 + currentSp;
                } else {
                    uint32_t remainingSp = sp - spFirstLoop;
                    uint32_t processedNewSp = (spLoop - 1) * spNextLoop;
                    uint32_t remainingNewSp = remainingSp > processedNewSp ? remainingSp - processedNewSp : 0;
                    currentSp = remainingNewSp < spNextLoop ? remainingNewSp : spNextLoop;
                    effectiveSp = 1 + currentSp;
                }
                CopyInSp(i, spLoop, currentSp, effectiveSp);
                Compute(effectiveSp);
                SaveAcc(outAccLocal);
            }
            WriteAcc(outAccLocal);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline __gm__ uint64_t* GetTensorPtr(GM_ADDR gmAddr) {
        __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(gmAddr);
        uint64_t tensorPtrOffset = *dataAddr;
        __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
        return tensorPtr;
    }

    __aicore__ inline void ParseTilingData(const DecodeUpdateTilingData *tdata)
    {
        this->hDim = tdata->hDim;
        this->sp = tdata->sp;
        this->totalLength = tdata->totalLength;
        this->updateType = tdata->updateType;
        if (GetBlockIdx() < tdata->formerNum) {
            blockLength = tdata->formerLength;
            this->gmStartOffset = GetBlockIdx() * tdata->formerLength;
        } else {
            blockLength = tdata->tailLength;
            this->gmStartOffset = tdata->formerNum * tdata->formerLength +
                                  (GetBlockIdx() - tdata->formerNum) * tdata->tailLength;
        }

        this->spFirstLoop = LARGE_SP_MAX_EFFECTIVE_SP - 1;
        this->spNextLoop = LARGE_SP_MAX_EFFECTIVE_SP - 1;
        this->spLoopCount = 1 + (sp - spFirstLoop + spNextLoop - 1) / spNextLoop;
        this->spAlloc = LARGE_SP_MAX_EFFECTIVE_SP;
    }

    __aicore__ inline void CalcTileLength()
    {
        uint32_t spAligned = (spAlloc + LARGE_SP_NUM7) / LARGE_SP_NUM8 * LARGE_SP_NUM8;
        uint32_t maxTileLength =
            (LARGE_SP_MAX_UB_SIZE - LARGE_SP_NUM8 * sizeof(uint32_t) - (LARGE_SP_ELEM_PER_256B - 1) * (sizeof(float) + sizeof(uint8_t))) /
            (sizeof(float) * spAligned * LARGE_SP_BUFFER_NUM * (LARGE_SP_NUM2 * (LARGE_SP_NUM1 + hDim) + hDim / spAligned) +
             hDim * spAligned * sizeof(float) * LARGE_SP_NUM2 +
             sizeof(float) * LARGE_SP_BUFFER_NUM +
             sizeof(float) * LARGE_SP_NUM2 +
             sizeof(uint8_t) * spAligned);
        if constexpr (!std::is_same<outType, float>::value) {
            maxTileLength =
                (LARGE_SP_MAX_UB_SIZE - LARGE_SP_NUM8 * sizeof(uint32_t) - (LARGE_SP_ELEM_PER_256B - 1) * (sizeof(float) + sizeof(uint8_t))) /
                (sizeof(float) * spAligned * LARGE_SP_BUFFER_NUM * (LARGE_SP_NUM2 * (LARGE_SP_NUM1 + hDim) + hDim / spAligned) +
                 hDim * spAligned * sizeof(float) * LARGE_SP_NUM2 +
                 sizeof(float) * LARGE_SP_BUFFER_NUM +
                 sizeof(float) * LARGE_SP_NUM2 + (spAlloc + LARGE_SP_NUM1) * LARGE_SP_NUM16 * LARGE_SP_BUFFER_NUM +
                 sizeof(uint8_t) * spAligned);
        }
        if (sp >= LARGE_SP_NUM8) {
            maxTileLength = maxTileLength / LARGE_SP_SPLIT_TO_2;
        }
        maxTileLength = maxTileLength < LARGE_SP_NUM1 ? LARGE_SP_NUM1 : maxTileLength;
        this->tileLength = maxTileLength < blockLength ? maxTileLength : blockLength;
        this->curLength = this->tileLength;
        this->lastLength = blockLength % tileLength;
        this->loopCount = blockLength / tileLength + (lastLength == LARGE_SP_NUM0 ? LARGE_SP_NUM0 : LARGE_SP_NUM1);
        this->tileLengthAlig = ((tileLength + LARGE_SP_NUM7) / LARGE_SP_NUM8) * LARGE_SP_NUM8;
        this->lastLengthAlig = ((lastLength + LARGE_SP_NUM7) / LARGE_SP_NUM8) * LARGE_SP_NUM8;
    }

    __aicore__ inline void InitBuffers()
    {
        uint32_t inQueueLseLengthAlign = (tileLengthAlig * spAlloc * sizeof(float) + LARGE_SP_NUM255) / LARGE_SP_NUM256 * LARGE_SP_NUM256;
        pipe.InitBuffer(inQueueLse, LARGE_SP_BUFFER_NUM, inQueueLseLengthAlign);
        if constexpr (std::is_same<outType, float>::value) {
            pipe.InitBuffer(inQueueIn, LARGE_SP_BUFFER_NUM, tileLength * hDim * spAlloc * sizeof(float));
        } else {
            pipe.InitBuffer(inQueueIn, LARGE_SP_BUFFER_NUM,
                            tileLength * hDim * spAlloc * sizeof(float) + (spAlloc + LARGE_SP_NUM1) * LARGE_SP_NUM16);
        }

        pipe.InitBuffer(outQueueOut, LARGE_SP_BUFFER_NUM, tileLength * hDim * sizeof(float));
        pipe.InitBuffer(outQueueLse, LARGE_SP_BUFFER_NUM, tileLengthAlig * sizeof(float));

        pipe.InitBuffer(lsemaxBuffer, tileLengthAlig * sizeof(float));
        pipe.InitBuffer(lseexpsumBuffer, tileLengthAlig * sizeof(float));
        pipe.InitBuffer(lseexpBuffer, tileLengthAlig * spAlloc * sizeof(float));
        if (hDim > LARGE_SP_NUM256 && spAlloc >= LARGE_SP_NUM8) {
            pipe.InitBuffer(lseexpBroadcastBuffer, tileLengthAlig * spAlloc * LARGE_SP_NUM64 * sizeof(float));
        } else {
            pipe.InitBuffer(lseexpBroadcastBuffer, tileLengthAlig * spAlloc * hDim * sizeof(float));
        }
        pipe.InitBuffer(selMaskBuffer, inQueueLseLengthAlign * sizeof(uint8_t));
    }

    __aicore__ inline void CopyInSpAccData(LocalTensor<float>& lseLocal, LocalTensor<float>& inLocal,
                                           int32_t spLoopIdx, uint32_t curLengthPad)
    {
        LocalTensor<float> lseAccLocal = lseexpsumBuffer.Get<float>();
        LocalTensor<float> outAccLocal = lseexpBroadcastBuffer.Get<float>();
        DataCopy(lseLocal[0], lseAccLocal, curLengthPad);
        PipeBarrier<PIPE_V>();
        DataCopy(inLocal[0], outAccLocal, curLength * hDim);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyInLsePerSp(LocalTensor<float>& lseLocal, uint32_t posIdx,
                                          uint32_t gmSpIdx, int32_t progress, uint32_t curLengthPad)
    {
        lseGm.SetGlobalBuffer(reinterpret_cast<__gm__ lseType*>(*(lsePtr + gmSpIdx)), totalLength);
        if (curLength % LARGE_SP_ALIGNED_TO_8 == 0) {
            DataCopy(lseLocal[posIdx * curLengthPad], lseGm[progress * tileLength + gmStartOffset], curLength);
        } else {
            DataCopyPad(lseLocal[posIdx * curLengthPad], lseGm[progress * tileLength + gmStartOffset],
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(curLength * sizeof(lseType)), 0, 0, 0},
                        {true, 0, static_cast<uint8_t>(LARGE_SP_NUM8 - curLength % LARGE_SP_NUM8), 0});
        }
    }

    __aicore__ inline void CopyInOPerSp(LocalTensor<float>& inLocal, LocalTensor<outType>& inLocalFp16,
                                         uint32_t posIdx, uint32_t gmSpIdx, int32_t progress)
    {
        inGm.SetGlobalBuffer(reinterpret_cast<__gm__ outType*>(*(inPtr + gmSpIdx)), totalLength * hDim);
        uint32_t inOffset = posIdx * curLength * hDim;
        uint32_t gmOffset = progress * tileLength * hDim + gmStartOffset * hDim;
        if constexpr (std::is_same<outType, float>::value) {
            DataCopy(inLocal[inOffset], inGm[gmOffset], curLength * hDim);
        } else {
            if (curLength % LARGE_SP_ALIGNED_TO_8 == 0) {
                DataCopy(inLocalFp16[inOffset], inGm[gmOffset], curLength * hDim);
            } else {
                uint64_t fp16Offset = posIdx * curLength * hDim * LARGE_SP_NUM2;
                if (fp16Offset % LARGE_SP_NUM32 == LARGE_SP_NUM16) {
                    fp16Offset += LARGE_SP_NUM16;
                }
                DataCopyPad(inLocalFp16[fp16Offset / LARGE_SP_NUM2], inGm[gmOffset],
                            {static_cast<uint16_t>(1),
                             static_cast<uint32_t>(curLength * hDim * sizeof(outType)), 0, 0, 0},
                            {true, 0,
                             static_cast<uint8_t>((LARGE_SP_NUM32 - curLength * hDim * sizeof(outType) % LARGE_SP_NUM32) / LARGE_SP_NUM2), 0});
                PipeBarrier<PIPE_ALL>();
                Cast(inLocal[inOffset], inLocalFp16[fp16Offset / LARGE_SP_NUM2], RoundMode::CAST_NONE, curLength * hDim);
            }
        }
    }

    __aicore__ inline void CopyInSp(int32_t progress, int32_t spLoopIdx, uint32_t currentSp, uint32_t effectiveSp)
    {
        LocalTensor<float> lseLocal = inQueueLse.AllocTensor<float>();
        LocalTensor<float> inLocal = inQueueIn.AllocTensor<float>();

        uint32_t curLengthPad = ((curLength + LARGE_SP_NUM7) / LARGE_SP_NUM8) * LARGE_SP_NUM8;
        uint32_t posOffset = (spLoopIdx == 0) ? 1 : (spLoopIdx > 0 ? 1 : 0);
        uint32_t gmSpBase = (spLoopIdx == 0) ? 0 : (spFirstLoop + (spLoopIdx - 1) * spNextLoop);

        if (spLoopIdx == 0) {
            Duplicate<float>(lseLocal, LARGE_SP_DUMMY_LSE, static_cast<int32_t>(curLengthPad));
            PipeBarrier<PIPE_V>();
            Duplicate<float>(inLocal, 0.0f, static_cast<int32_t>(curLength * hDim));
            PipeBarrier<PIPE_V>();
        } else {
            CopyInSpAccData(lseLocal, inLocal, spLoopIdx, curLengthPad);
        }

        uint64_t inLocalFp16Offset = tileLength * hDim * spAlloc;
        if ((inLocalFp16Offset * LARGE_SP_NUM2) % LARGE_SP_NUM32 == LARGE_SP_NUM16) {
            inLocalFp16Offset += LARGE_SP_NUM8;
        }
        LocalTensor<outType> inLocalFp16 = inLocal.template ReinterpretCast<outType>()[inLocalFp16Offset];

        for (uint32_t i = 0; i < currentSp; i++) {
            uint32_t posIdx = posOffset + i;
            uint32_t gmSpIdx = gmSpBase + i;
            CopyInLsePerSp(lseLocal, posIdx, gmSpIdx, progress, curLengthPad);
            CopyInOPerSp(inLocal, inLocalFp16, posIdx, gmSpIdx, progress);
        }

        if constexpr (!std::is_same<outType, float>::value) {
            if (curLength % LARGE_SP_ALIGNED_TO_8 == 0) {
                inQueueIn.EnQue(inLocal);
                inLocal = inQueueIn.DeQue<float>();
                inLocalFp16 = inLocal.template ReinterpretCast<outType>()[inLocalFp16Offset];
                Cast(inLocal[posOffset * curLength * hDim], inLocalFp16[posOffset * curLength * hDim],
                     RoundMode::CAST_NONE, curLength * hDim * currentSp);
            }
        }

        inQueueLse.EnQue(lseLocal);
        inQueueIn.EnQue(inLocal);
    }

    __aicore__ inline void SaveAcc(LocalTensor<float>& outAccLocal)
    {
        LocalTensor<float> lseoutLocal = outQueueLse.DeQue<float>();
        LocalTensor<float> outLocal = outQueueOut.DeQue<float>();

        LocalTensor<float> lseAccLocal = lseexpsumBuffer.Get<float>();
        DataCopy(lseAccLocal, lseoutLocal, ((curLength + LARGE_SP_NUM7) / LARGE_SP_NUM8) * LARGE_SP_NUM8);
        PipeBarrier<PIPE_V>();
        DataCopy(outAccLocal, outLocal, curLength * hDim);
        PipeBarrier<PIPE_V>();

        outQueueLse.FreeTensor(lseoutLocal);
        outQueueOut.FreeTensor(outLocal);
    }

    __aicore__ inline void WriteAcc(LocalTensor<float>& outAccLocal)
    {
        LocalTensor<float> lseAccLocal = lseexpsumBuffer.Get<float>();
        LocalTensor<float> outLocal = outQueueOut.AllocTensor<float>();
        LocalTensor<float> lseoutLocal = outQueueLse.AllocTensor<float>();

        uint32_t curLengthPad = ((curLength + LARGE_SP_NUM7) / LARGE_SP_NUM8) * LARGE_SP_NUM8;
        DataCopy(outLocal, outAccLocal, curLength * hDim);
        PipeBarrier<PIPE_V>();
        DataCopy(lseoutLocal, lseAccLocal, curLengthPad);
        PipeBarrier<PIPE_V>();

        outQueueOut.EnQue(outLocal);
        outQueueLse.EnQue(lseoutLocal);
    }

    __aicore__ inline void ProcessLseInfReplacement(LocalTensor<float>& lseLocal, uint32_t effectiveSp)
    {
        LocalTensor<uint8_t> selMask = selMaskBuffer.Get<uint8_t>();
        uint32_t curLengthPad = ((curLength + LARGE_SP_NUM7) / LARGE_SP_NUM8) * LARGE_SP_NUM8;
        uint32_t alignedCount = ((curLengthPad * effectiveSp + LARGE_SP_ELEM_PER_256B - 1) / LARGE_SP_ELEM_PER_256B) * LARGE_SP_ELEM_PER_256B;

        CompareScalar(selMask, lseLocal, LARGE_SP_POS_INF, CMPMODE::EQ, alignedCount);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> negInfTensor = lseexpBuffer.Get<float>();
        Duplicate<float>(negInfTensor, LARGE_SP_NEG_INF, static_cast<int32_t>(curLengthPad * effectiveSp));
        PipeBarrier<PIPE_V>();

        Select<float, uint8_t>(lseLocal, selMask, negInfTensor, lseLocal,
                               SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<uint32_t>(curLengthPad * effectiveSp));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Compute(uint32_t effectiveSp)
    {
        LocalTensor<float> lseLocal = inQueueLse.DeQue<float>();
        LocalTensor<float> inLocal = inQueueIn.DeQue<float>();
        LocalTensor<float> lseexpLocal = lseexpBuffer.Get<float>();
        LocalTensor<float> outLocal = outQueueOut.AllocTensor<float>();
        LocalTensor<float> lsemaxLocal = lsemaxBuffer.Get<float>();
        LocalTensor<float> lseexpsumLocal = lseexpsumBuffer.Get<float>();
        LocalTensor<float> lseexpBroadcastLocal = lseexpBroadcastBuffer.Get<float>();
        LocalTensor<float> lseoutLocal = outQueueLse.AllocTensor<float>();

        ProcessLseInfReplacement(lseLocal, effectiveSp);

        uint32_t curLengthPad = ((curLength + LARGE_SP_NUM7) / LARGE_SP_NUM8) * LARGE_SP_NUM8;
        const uint32_t srcShape[2] = {effectiveSp * curLengthPad, 1};
        uint32_t dstShape[2] = {effectiveSp * curLengthPad, hDim};
        if (hDim > LARGE_SP_NUM256 && spAlloc >= LARGE_SP_NUM8) {
            dstShape[1] = LARGE_SP_NUM64;
        }

        DataCopy(lsemaxLocal, lseLocal, curLengthPad);
        PipeBarrier<PIPE_V>();

        for (uint32_t i = 1; i < effectiveSp; i++) {
            Max(lsemaxLocal, lsemaxLocal, lseLocal[i * curLengthPad], curLengthPad);
            PipeBarrier<PIPE_V>();
        }

        PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < effectiveSp; i++) {
            Sub(lseexpLocal[i * curLengthPad], lseLocal[i * curLengthPad], lsemaxLocal, curLengthPad);
            PipeBarrier<PIPE_V>();
        }

        Exp(lseexpLocal, lseexpLocal, curLengthPad * effectiveSp);
        PipeBarrier<PIPE_V>();

        DataCopy(lseexpsumLocal, lseexpLocal, curLengthPad);
        PipeBarrier<PIPE_V>();
        for (uint32_t i = 1; i < effectiveSp; i++) {
            Add(lseexpsumLocal, lseexpsumLocal, lseexpLocal[i * curLengthPad], curLengthPad);
            PipeBarrier<PIPE_V>();
        }

        Log(lseexpsumLocal, lseexpsumLocal, curLengthPad);
        PipeBarrier<PIPE_V>();

        Add(lseoutLocal, lsemaxLocal, lseexpsumLocal, curLengthPad);
        PipeBarrier<PIPE_V>();

        for (uint32_t i = 0; i < effectiveSp; i++) {
            Sub(lseexpLocal[i * curLengthPad], lseLocal[i * curLengthPad], lseoutLocal, curLengthPad);
            PipeBarrier<PIPE_V>();
        }

        Exp(lseexpLocal, lseexpLocal, curLengthPad * effectiveSp);
        PipeBarrier<PIPE_V>();
        BroadCast<float, LARGE_SP_ALIGNED_TO_2, 1>(lseexpBroadcastLocal, lseexpLocal, dstShape, srcShape);
        PipeBarrier<PIPE_V>();
        if (hDim > LARGE_SP_NUM256 && spAlloc >= LARGE_SP_NUM8) {
            int64_t tmpTailLength = hDim % LARGE_SP_NUM64;
            int64_t tmpTailStart = hDim - tmpTailLength;
            for (uint32_t i = 0; i < effectiveSp; i++) {
                for (uint32_t k = 0; k < curLength; k++) {
                    Mul(inLocal[i * curLength * hDim + k * hDim],
                                inLocal[i * curLength * hDim + k * hDim],
                                lseexpBroadcastLocal[i * curLengthPad * LARGE_SP_NUM64 + k * LARGE_SP_NUM64],
                                LARGE_SP_NUM64, hDim / LARGE_SP_NUM64, {1, 1, 1, 8, 8, 0});
                    if (tmpTailLength > 0) {
                        Mul(inLocal[i * curLength * hDim + k * hDim + tmpTailStart],
                                inLocal[i * curLength * hDim + k * hDim + tmpTailStart],
                                lseexpBroadcastLocal[i * curLengthPad * LARGE_SP_NUM64 + k * LARGE_SP_NUM64],
                                tmpTailLength, 1, {1, 1, 1, 8, 8, 0});
                    }
                }
            }
        } else {
            for (uint32_t i = 0; i < effectiveSp; i++) {
                Mul(inLocal[i * curLength * hDim], inLocal[i * curLength * hDim], lseexpBroadcastLocal[i * curLengthPad * hDim],
                    curLength * hDim);
            }
        }
        PipeBarrier<PIPE_V>();

        DataCopy(outLocal, inLocal, curLength * hDim);
        PipeBarrier<PIPE_V>();
        for (uint32_t i = 1; i < effectiveSp; i++) {
            Add(outLocal, outLocal, inLocal[i * curLength * hDim], curLength * hDim);
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_V>();

        outQueueLse.EnQue<float>(lseoutLocal);
        outQueueOut.EnQue<float>(outLocal);
        inQueueLse.FreeTensor(lseLocal);
        inQueueIn.FreeTensor(inLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<float> outLocal = outQueueOut.DeQue<float>();
        if constexpr (std::is_same<outType, float>::value) {
            DataCopy(outGm[gmStartOffset * hDim + progress * tileLength * hDim], outLocal, curLength * hDim);
        } else if constexpr (std::is_same<outType, bfloat16_t>::value) {
            LocalTensor<outType> outLocal16 = outLocal.template ReinterpretCast<outType>();
            Cast(outLocal16, outLocal, RoundMode::CAST_RINT, curLength * hDim);
            PipeBarrier<PIPE_V>();
            DataCopyPad(outGm[gmStartOffset * hDim + progress * tileLength * hDim], outLocal16,
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(curLength * hDim * sizeof(outType)), 0, 0, 0});
        } else {
            LocalTensor<outType> outLocal16 = outLocal.template ReinterpretCast<outType>();
            Cast(outLocal16, outLocal, RoundMode::CAST_NONE, curLength * hDim);
            PipeBarrier<PIPE_V>();
            DataCopyPad(outGm[gmStartOffset * hDim + progress * tileLength * hDim], outLocal16,
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(curLength * hDim * sizeof(outType)), 0, 0, 0});
        }
        outQueueOut.FreeTensor(outLocal);
        LocalTensor<float> lseoutLocal = outQueueLse.DeQue<float>();
        if (updateType == 1) {
            uint32_t lseoutGmStart = gmStartOffset + progress * tileLength;
            uint32_t validBytes = static_cast<uint32_t>(curLength * sizeof(float));
            DataCopyExtParams copyParams{1, validBytes, 0, 0, 0};
            DataCopyPad(lseoutGm[lseoutGmStart], lseoutLocal, copyParams);
        }
        outQueueLse.FreeTensor(lseoutLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, LARGE_SP_BUFFER_NUM> inQueueLse;
    TQue<QuePosition::VECIN, LARGE_SP_BUFFER_NUM> inQueueIn;
    TQue<QuePosition::VECOUT, LARGE_SP_BUFFER_NUM> outQueueOut;
    TQue<QuePosition::VECOUT, LARGE_SP_BUFFER_NUM> outQueueLse;
    TBuf<QuePosition::VECCALC> lseexpBuffer;
    TBuf<QuePosition::VECCALC> lsemaxBuffer;
    TBuf<QuePosition::VECCALC> lseexpsumBuffer;
    TBuf<QuePosition::VECCALC> lseexpBroadcastBuffer;
    TBuf<QuePosition::VECCALC> selMaskBuffer;

    GlobalTensor<lseType> lseGm;
    GlobalTensor<outType> inGm;
    GlobalTensor<outType> outGm;
    GlobalTensor<lseType> lseoutGm;

    __gm__ uint64_t* lsePtr;
    __gm__ uint64_t* inPtr;

    uint32_t blockLength;
    uint16_t tileLength;
    uint16_t curLength;
    uint16_t lastLength;
    uint32_t loopCount;
    uint32_t hDim;
    uint32_t sp;
    uint32_t tileLengthAlig;
    uint32_t lastLengthAlig;
    uint32_t totalLength;
    uint32_t gmStartOffset;
    uint32_t updateType;
    uint32_t spLoopCount;
    uint32_t spFirstLoop;
    uint32_t spNextLoop;
    uint32_t spAlloc;
};
} // namespace AttentionUpdate

#endif // ASCENDC_ATTENTION_UPDATE_DECODE_UPDATE_LARGE_SP_H_