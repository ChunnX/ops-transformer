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
 * \file decode_update.h
 * \brief
 */

#ifndef ASCENDC_ATTENTION_UPDATE_DECODE_UPDATE_H_
#define ASCENDC_ATTENTION_UPDATE_DECODE_UPDATE_H_

#include <limits>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

namespace AttentionUpdate {
static constexpr uint32_t BUFFER_NUM = 2;
static constexpr uint32_t NUM0 = 0;
static constexpr uint32_t NUM1 = 1;
static constexpr uint32_t NUM2 = 2;
static constexpr uint32_t NUM7 = 7;
static constexpr uint32_t NUM8 = 8;
static constexpr uint32_t NUM15 = 15;
static constexpr uint32_t NUM16 = 16;
static constexpr uint32_t NUM32 = 32;
static constexpr uint32_t NUM64 = 64;
static constexpr uint32_t NUM255 = 255;
static constexpr uint32_t NUM256 = 256;
static constexpr uint32_t MAX_EFFECTIVE_SP = 16;  // 每轮最多处理的路数（含acc路）
static constexpr float POS_INF = std::numeric_limits<float>::infinity();
static constexpr float NEG_INF = -std::numeric_limits<float>::infinity();
static constexpr uint32_t MAX_UB_SIZE = 188 * 1024; //  double buffer, 每块94KB共188KB
static const uint16_t ALIGNED_TO_8 = 8;
static const int32_t ALIGNED_TO_2 = 2;
static const uint32_t SPLIT_TO_2 = 2;
static const uint32_t ELEM_PER_256B = NUM256 / sizeof(float);

template <typename lseType, typename outType>
class DecodeUpdate {
public:
    __aicore__ inline DecodeUpdate() {}
    __aicore__ inline void Init(GM_ADDR lse, GM_ADDR in, GM_ADDR out, GM_ADDR lesout, const DecodeUpdateTilingData *tdata)
    {
        this->lsePtr = GetTensorPtr(lse);
        this->inPtr = GetTensorPtr(in);

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
                                  (GetBlockIdx() - tdata->formerNum) * tdata->tailLength; // tail block
        }
        
        // sp方向循环参数计算
        if (sp <= MAX_EFFECTIVE_SP) {
            this->spLoopCount = NUM1;
            this->spFirstLoop = sp;
            this->spNextLoop = NUM0;
        } else {
            this->spFirstLoop = MAX_EFFECTIVE_SP;
            this->spNextLoop = MAX_EFFECTIVE_SP - NUM1;  // 后续轮：1路acc + 15路新数据 = 16路effective
            this->spLoopCount = NUM1 + (sp - MAX_EFFECTIVE_SP + spNextLoop - NUM1) / spNextLoop;  // 1 + ceil((sp-16)/15)
        }
        
        // UB分配固定按MAX_EFFECTIVE_SP路计算（最大effectiveSp）
        uint32_t spAligned = (MAX_EFFECTIVE_SP + NUM7) / NUM8 * NUM8;
        //  用94K的UB大小推算出 tileLength 最大能设置到多少
        uint32_t maxTileLength = (MAX_UB_SIZE - NUM8 * sizeof(uint32_t) - (ELEM_PER_256B - 1) * (sizeof(float) + sizeof(uint8_t))) /
                                (sizeof(float) * spAligned * BUFFER_NUM * (NUM2 * (NUM1 + hDim) + hDim / spAligned) +
                                hDim * spAligned * sizeof(float) * NUM2 +
                                sizeof(float) * BUFFER_NUM +
                                sizeof(float) * NUM2 +
                                sizeof(uint8_t) * spAligned);
        if constexpr (!std::is_same<outType, float>::value) {
            maxTileLength = (MAX_UB_SIZE - NUM8 * sizeof(uint32_t) - (ELEM_PER_256B - 1) * (sizeof(float) + sizeof(uint8_t))) /
                                    (sizeof(float) * spAligned * BUFFER_NUM * (NUM2 * (NUM1 + hDim) + hDim / spAligned) +
                                    hDim * spAligned * sizeof(float) * NUM2 +
                                    sizeof(float) * BUFFER_NUM +
                                    sizeof(float) * NUM2 + (sp + NUM1) * NUM16 * BUFFER_NUM +
                                    sizeof(uint8_t) * spAligned);
        }
        if (sp >= NUM8) {
            maxTileLength = maxTileLength / SPLIT_TO_2;
        }
        maxTileLength = maxTileLength < NUM1 ? NUM1 : maxTileLength;
        this->tileLength = maxTileLength < blockLength ? maxTileLength : blockLength;
        this->curLength = this->tileLength;
        this->lastLength = blockLength % tileLength;
        this->loopCount = blockLength / tileLength + (lastLength == NUM0 ? NUM0 : NUM1);
        this->tileLengthAlig = ((tileLength + NUM7) / NUM8) * NUM8;
        this->lastLengthAlig = ((lastLength + NUM7) / NUM8) * NUM8;
        // 设置全局变量的起始地址与总长度BLOCK_LENGTH; sp, B*s*hc in1 sp; B*s*hc, hd in2
        outGm.SetGlobalBuffer((__gm__ outType *)out, totalLength * hDim);
        lseoutGm.SetGlobalBuffer((__gm__ lseType *)lesout, totalLength);

        // Buffer分配固定按MAX_EFFECTIVE_SP路（最大effectiveSp）
        uint32_t inQueueLseLengthAlign = (tileLengthAlig * MAX_EFFECTIVE_SP * sizeof(float) + NUM255) / NUM256 * NUM256;
        pipe.InitBuffer(inQueueLse, BUFFER_NUM, inQueueLseLengthAlign);
        if constexpr (std::is_same<outType, float>::value) {
            pipe.InitBuffer(inQueueIn, BUFFER_NUM, tileLength * hDim * MAX_EFFECTIVE_SP * sizeof(float));
        } else {
            pipe.InitBuffer(inQueueIn, BUFFER_NUM, tileLength * hDim * MAX_EFFECTIVE_SP * sizeof(float) + (MAX_EFFECTIVE_SP + NUM1) * NUM16);
        }
        
        pipe.InitBuffer(outQueueOut, BUFFER_NUM, tileLength * hDim * sizeof(float));
        pipe.InitBuffer(outQueueLse, BUFFER_NUM, tileLengthAlig * sizeof(float));

        pipe.InitBuffer(lsemaxBuffer, tileLengthAlig * sizeof(float));
        pipe.InitBuffer(lseexpsumBuffer, tileLengthAlig * sizeof(float));
        pipe.InitBuffer(lseexpBuffer, tileLengthAlig * MAX_EFFECTIVE_SP * sizeof(float));
        if (hDim > NUM256 && sp >= NUM8) {
            pipe.InitBuffer(lseexpBroadcastBuffer, tileLengthAlig * MAX_EFFECTIVE_SP * NUM64 * sizeof(float));
        } else {
            pipe.InitBuffer(lseexpBroadcastBuffer, tileLengthAlig * MAX_EFFECTIVE_SP * hDim * sizeof(float));
        }
        pipe.InitBuffer(selMaskBuffer, inQueueLseLengthAlign * sizeof(uint8_t));
    }
    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < this->loopCount; i++) {
            if (lastLength && i == this->loopCount - 1) {
                curLength = lastLength;
            }
            LocalTensor<float> outAccLocal = lseexpBroadcastBuffer.Get<float>();

            for (int32_t spLoop = 0; spLoop < spLoopCount; spLoop++) {
                uint32_t currentSp, effectiveSp;
                if (spLoop == 0) {
                    currentSp = spFirstLoop;
                    effectiveSp = currentSp;
                } else {
                    // 后续轮：从第MAX_EFFECTIVE_SP路开始，每轮处理spNextLoop路新数据
                    uint32_t remainingSp = sp - MAX_EFFECTIVE_SP;
                    uint32_t processedNewSp = (spLoop - NUM1) * spNextLoop;
                    uint32_t remainingNewSp = remainingSp > processedNewSp ? remainingSp - processedNewSp : 0;
                    currentSp = remainingNewSp < spNextLoop ? remainingNewSp : spNextLoop;
                    effectiveSp = NUM1 + currentSp;  // 1路acc + currentSp路新数据
                }
                CopyInSp(i, spLoop, currentSp, effectiveSp);
                Compute(effectiveSp);
                SaveAcc(outAccLocal);
            }
            ComputeFinalize(outAccLocal);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline __gm__ uint64_t* GetTensorPtr(GM_ADDR gmAddr) {
        __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(gmAddr);
        uint64_t tensorPtrOffset = *dataAddr;  // The offset of the data address from the first address.
        // Moving 3 bits to the right means dividing by sizeof(uint64 t).
        __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
        return tensorPtr;
    }

    __aicore__ inline void CopyInSp(int32_t progress, int32_t spLoopIdx, uint32_t currentSp, uint32_t effectiveSp)
    {
        LocalTensor<float> lseLocal = inQueueLse.AllocTensor<float>();
        LocalTensor<float> inLocal = inQueueIn.AllocTensor<float>();
        
        uint32_t curLengthPad = ((curLength + NUM7) / NUM8) * NUM8;
        // posOffset: 首轮(spLoopIdx==0)数据从position 0开始存放；后续轮position 0留给acc累积结果，新数据从position 1开始
        uint32_t posOffset = (spLoopIdx == 0) ? 0 : 1;
        // gmSpBase: 首轮从GM的sp索引0开始读；后续轮从MAX_EFFECTIVE_SP起，按spNextLoop步进
        uint32_t gmSpBase = (spLoopIdx == 0) ? 0 : (MAX_EFFECTIVE_SP + (spLoopIdx - NUM1) * spNextLoop);

        // 后续轮：将上一轮Compute的acc结果填入position 0，作为本轮effectiveSp中的第0路参与计算
        // lse acc复用lseexpsumBuffer，out acc复用lseexpBroadcastBuffer（与Compute中使用时机互斥）
        if (spLoopIdx > 0) {
            LocalTensor<float> lseAccLocal = lseexpsumBuffer.Get<float>();
            LocalTensor<float> outAccLocal = lseexpBroadcastBuffer.Get<float>();
            DataCopy(lseLocal[0], lseAccLocal, curLengthPad);
            PipeBarrier<PIPE_V>();
            DataCopy(inLocal[0], outAccLocal, curLength * hDim);
            PipeBarrier<PIPE_V>();
        }

        // fp16/bf16模式下，inLocal的前半段存fp32，后半段(inLocalFp16Offset起)暂存原始fp16/bf16待Cast
        uint64_t inLocalFp16Offset = tileLength * hDim * MAX_EFFECTIVE_SP;
        if ((inLocalFp16Offset * NUM2) % NUM32 == NUM16) {
            inLocalFp16Offset += NUM8;
        }
        LocalTensor<outType> inLocalFp16 = inLocal.template ReinterpretCast<outType>()[inLocalFp16Offset];

        if (curLength % ALIGNED_TO_8 == 0) {
            // 对齐路径：curLength是8的倍数，可用DataCopy直接搬运
            for (uint32_t i = 0; i < currentSp; i++) {
                uint32_t gmSpIdx = gmSpBase + i;
                lseGm.SetGlobalBuffer(reinterpret_cast<__gm__ lseType*>(*(lsePtr + gmSpIdx)), totalLength);
                inGm.SetGlobalBuffer(reinterpret_cast<__gm__ outType*>(*(inPtr + gmSpIdx)), totalLength * hDim);
                // lse: [effectiveSp, curLengthPad]，第(posOffset+i)路存入lseLocal对应位置
                DataCopy(lseLocal[(posOffset + i) * curLengthPad], lseGm[progress * tileLength + gmStartOffset], curLength);
                if constexpr (std::is_same<outType, float>::value) {
                    // fp32: 直接搬入inLocal
                    DataCopy(inLocal[(posOffset + i) * curLength * hDim],
                         inGm[progress * tileLength * hDim + gmStartOffset * hDim],
                         curLength * hDim);
                } else {
                    // fp16/bf16: 先搬入inLocalFp16暂存区，循环结束后统一Cast
                    DataCopy(inLocalFp16[(posOffset + i) * curLength * hDim],
                         inGm[progress * tileLength * hDim + gmStartOffset * hDim],
                         curLength * hDim);
                }
            }
            // fp16/bf16: 将暂存区的currentSp路数据批量Cast为fp32，写入inLocal对应位置
            // 起始偏移posOffset*curLength*hDim，跳过position 0的acc数据
            if constexpr (!std::is_same<outType, float>::value) {
                inQueueIn.EnQue(inLocal);
                inLocal = inQueueIn.DeQue<float>();
                inLocalFp16 = inLocal.template ReinterpretCast<outType>()[inLocalFp16Offset];
                Cast(inLocal[posOffset * curLength * hDim], inLocalFp16[posOffset * curLength * hDim],
                     RoundMode::CAST_NONE, curLength * hDim * currentSp);
            }
        } else {
            // 非对齐路径：curLength不是8的倍数，lse需DataCopyPad补齐，fp16/bf16需逐路Cast
            for (uint32_t i = 0; i < currentSp; i++) {
                uint32_t gmSpIdx = gmSpBase + i;
                lseGm.SetGlobalBuffer(reinterpret_cast<__gm__ lseType*>(*(lsePtr + gmSpIdx)), totalLength);
                inGm.SetGlobalBuffer(reinterpret_cast<__gm__ outType*>(*(inPtr + gmSpIdx)), totalLength * hDim);
                // lse: DataCopyPad自动尾部补0对齐到8字节
                DataCopyPad(lseLocal[(posOffset + i) * curLengthPad], lseGm[progress * tileLength + gmStartOffset],
                            {static_cast<uint16_t>(1), static_cast<uint32_t>(curLength * sizeof(lseType)), 0, 0, 0},
                            {true, 0, static_cast<uint8_t>(NUM8 - curLength % NUM8), 0});
                if constexpr (std::is_same<outType, float>::value) {
                    // fp32: in的hDim维度对齐，直接DataCopy
                    DataCopy(inLocal[(posOffset + i) * curLength * hDim],
                         inGm[progress * tileLength * hDim + gmStartOffset * hDim],
                         curLength * hDim);
                } else {
                    // fp16/bf16: DataCopyPad搬入暂存区，立即Cast为fp32（非对齐无法批量Cast）
                    uint64_t fp16Offset = (posOffset + i) * curLength * hDim * NUM2;
                    if (fp16Offset % NUM32 == NUM16) {
                        fp16Offset += NUM16;
                    }
                    DataCopyPad(inLocalFp16[fp16Offset / NUM2],
                         inGm[progress * tileLength * hDim + gmStartOffset * hDim],
                         {static_cast<uint16_t>(1), static_cast<uint32_t>(curLength * hDim * sizeof(outType)), 0, 0, 0},
                            {true, 0, static_cast<uint8_t>((NUM32 - curLength * hDim * sizeof(outType) % NUM32) / NUM2), 0});
                    PipeBarrier<PIPE_ALL>();
                    Cast(inLocal[(posOffset + i) * curLength * hDim], inLocalFp16[fp16Offset / NUM2],
                         RoundMode::CAST_NONE, curLength * hDim);
                }
            }
        }

        inQueueLse.EnQue(lseLocal);
        inQueueIn.EnQue(inLocal);
    }

    __aicore__ inline void SaveAcc(LocalTensor<float>& outAccLocal)
    {
        LocalTensor<float> lseoutLocal = outQueueLse.DeQue<float>();
        LocalTensor<float> outLocal = outQueueOut.DeQue<float>();
        
        // 将Compute结果保存到acc buffer
        // lse acc buffer 复用 lseexpsumBuffer（使用时机互斥）
        LocalTensor<float> lseAccLocal = lseexpsumBuffer.Get<float>();
        DataCopy(lseAccLocal, lseoutLocal, ((curLength + NUM7) / NUM8) * NUM8);
        PipeBarrier<PIPE_V>();
        DataCopy(outAccLocal, outLocal, curLength * hDim);
        PipeBarrier<PIPE_V>();
        
        outQueueLse.FreeTensor(lseoutLocal);
        outQueueOut.FreeTensor(outLocal);
    }

    __aicore__ inline void ComputeFinalize(LocalTensor<float>& outAccLocal)
    {
        // sp循环结束后，将acc结果写入outQueue供CopyOut使用
        // lse acc buffer 复用 lseexpsumBuffer（使用时机互斥）
        LocalTensor<float> lseAccLocal = lseexpsumBuffer.Get<float>();
        LocalTensor<float> outLocal = outQueueOut.AllocTensor<float>();
        LocalTensor<float> lseoutLocal = outQueueLse.AllocTensor<float>();
        
        uint32_t curLengthPad = ((curLength + NUM7) / NUM8) * NUM8;
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
        uint32_t curLengthPad = ((curLength + NUM7) / NUM8) * NUM8;
        uint32_t alignedCount = ((curLengthPad * effectiveSp + ELEM_PER_256B - NUM1) / ELEM_PER_256B) * ELEM_PER_256B;

        CompareScalar(selMask, lseLocal, POS_INF, CMPMODE::EQ, alignedCount);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> negInfTensor = lseexpBuffer.Get<float>();
        Duplicate<float>(negInfTensor, NEG_INF, static_cast<int32_t>(curLengthPad * effectiveSp));
        PipeBarrier<PIPE_V>();

        Select<float, uint8_t>(lseLocal, selMask, negInfTensor, lseLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE, static_cast<uint32_t>(curLengthPad * effectiveSp));
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

        // broad to 8
        uint32_t curLengthPad = ((curLength + NUM7) / NUM8) * NUM8;
        const uint32_t srcShape[2] = {effectiveSp * curLengthPad, NUM1};
        uint32_t dstShape[2] = {effectiveSp * curLengthPad, hDim};
        if (hDim > NUM256 && effectiveSp >= NUM8) {
            dstShape[1] = NUM64;
        }

        DataCopy(lsemaxLocal, lseLocal, curLengthPad);
        PipeBarrier<PIPE_V>();

        for (uint32_t i = NUM1; i < effectiveSp; i++) {
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
        for (uint32_t i = NUM1; i < effectiveSp; i++) {
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
        BroadCast<float, ALIGNED_TO_2, NUM1>(lseexpBroadcastLocal, lseexpLocal, dstShape, srcShape);
        PipeBarrier<PIPE_V>();
        if (hDim > NUM256 && effectiveSp >= NUM8) {
            int64_t tmpTailLength = hDim % NUM64;
            int64_t tmpTailStart = hDim - tmpTailLength;
            for (uint32_t i = 0; i < effectiveSp; i++) {
                for (int32_t k = 0; k < curLength; k++) {
                    Mul(inLocal[i * curLength * hDim + k * hDim], 
                                inLocal[i * curLength * hDim + k * hDim], 
                                lseexpBroadcastLocal[i * curLengthPad * NUM64 + k * NUM64],
                                NUM64, hDim / NUM64, {1, 1, 1, 8, 8, 0});
                    if (tmpTailLength > 0) {
                        Mul(inLocal[i * curLength * hDim + k * hDim + tmpTailStart],
                                inLocal[i * curLength * hDim + k * hDim + tmpTailStart], 
                                lseexpBroadcastLocal[i * curLengthPad * NUM64 + k * NUM64],
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
        for (uint32_t i = NUM1; i < effectiveSp; i++) {
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
        if constexpr (std::is_same<outType, float>::value) { // fp32直接搬运
            DataCopy(outGm[gmStartOffset * hDim + progress * tileLength * hDim], outLocal, curLength * hDim);
        } else if constexpr (std::is_same<outType, bfloat16_t>::value){ // 先转fp32，再搬运
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
            uint32_t lseoutGmStart = gmStartOffset + progress * tileLength; // lseout无hDim维度，直接按长度偏移
            uint32_t validBytes = static_cast<uint32_t>(curLength * sizeof(float));
            DataCopyExtParams copyParams{1, validBytes, 0, 0, 0};
            DataCopyPad(lseoutGm[lseoutGmStart], lseoutLocal, copyParams);
        }
        outQueueLse.FreeTensor(lseoutLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueLse;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueIn;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueOut;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueLse;
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

    uint32_t blockLength; //  单核上数据总长度
    uint16_t tileLength;  //  单核循环的非最后一轮数据长度
    uint16_t curLength; //  Tiling 循环数据长度为 curLength，最后一轮为 lastLength，内存申请使用 tileLength，数据量用
                        //  curLength
    uint16_t lastLength; //  单核循环的最后一轮数据长度
    uint32_t loopCount;  //  单核循环次数
    uint32_t hDim;       //  head dimension
    uint32_t sp;
    uint32_t tileLengthAlig;
    uint32_t lastLengthAlig;
    uint32_t totalLength;
    uint32_t gmStartOffset;
    uint32_t updateType;
    uint32_t spLoopCount;   // sp方向循环次数
    uint32_t spFirstLoop;   // 第一轮sp数（≤16）
    uint32_t spNextLoop;    // 后续轮新数据sp数（≤15）
};
} // namespace AttentionUpdate

#endif // ASCENDC_ATTENTION_UPDATE_DECODE_UPDATE_H_