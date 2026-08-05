/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "op_kernel/utils.h"

#include <cmath>
#include <limits>

namespace LightningAttn::Kernel
{

template <typename T, typename TOut = bfloat16_t> class CmmDecayGenerator
{
public:
    __aicore__ inline CmmDecayGenerator() {}

    __aicore__ inline void Init(int64_t numHeads, int64_t headSize, int64_t blockSize, AscendC::TPipe* pipe)
    {
        this->pipe = pipe;
        this->numHeads = numHeads;
        this->headSize = headSize;
        this->blockSize = blockSize;

        const int64_t alignSize = 256l;

        numHeadsAlignedBytes = numHeads * sizeof(T);
        numHeadsAlignedBytes = CeilDiv(numHeadsAlignedBytes, alignSize) * alignSize;
        numHeadsAligned = numHeadsAlignedBytes / sizeof(T);

        blockSizeAlignedBytes = blockSize * sizeof(T);
        blockSizeAlignedBytes = CeilDiv(blockSizeAlignedBytes, alignSize) * alignSize;
        blockSizeAligned = blockSizeAlignedBytes / sizeof(T);

        idtQkSize = blockSize * blockSize;
        idtQkSizeAlignedBytes = idtQkSize * sizeof(T);
        idtQkSizeAlignedBytes = CeilDiv(idtQkSizeAlignedBytes, alignSize) * alignSize;
        idtQkSizeAligned = idtQkSizeAlignedBytes / sizeof(T);

        idtKvSize = headSize * headSize;
        idtKvSizeAlignedBytes = idtKvSize * sizeof(T);
        idtKvSizeAlignedBytes = CeilDiv(idtKvSizeAlignedBytes, alignSize) * alignSize;
        idtKvSizeAligned = idtKvSizeAlignedBytes / sizeof(T);

        idtQkOutSizeAlignedBytes = idtQkSize * sizeof(TOut);
        idtQkOutSizeAlignedBytes = CeilDiv(idtQkOutSizeAlignedBytes, alignSize) * alignSize;

        idtKvOutSizeAlignedBytes = idtKvSize * sizeof(TOut);
        idtKvOutSizeAlignedBytes = CeilDiv(idtKvOutSizeAlignedBytes, alignSize) * alignSize;

        coreIdx = AscendC::GetBlockIdx();
        headsPerCore = 1;
    }

    __aicore__ inline void GenerateQK(AscendC::GlobalTensor<TOut>& out,
                                      AscendC::GlobalTensor<T>& slopeGm,
                                      AscendC::LocalTensor<T>& indexes,
                                      AscendC::LocalTensor<T>& tmp)
    {
        for (int64_t i = 0; i < headsPerCore; i++)
        {
            int64_t head = headsPerCore * coreIdx + i;
            int64_t outOffset = head * idtQkSize;

            T slope = -1 * slopeGm.GetValue(head);
            pipe_barrier(PIPE_V);

            AscendC::Muls(tmp, indexes, slope, blockSizeAligned);
            pipe_barrier(PIPE_V);

            AscendC::Exp(tmp, tmp, blockSizeAligned);
            pipe_barrier(PIPE_V);

            auto qkCalc = qkCalcQueue.AllocTensor<T>();
            AscendC::Duplicate(qkCalc, T(0), idtQkSizeAligned);
            qkCalcQueue.EnQue(qkCalc);
            qkCalcQueue.DeQue<T>();

            for (size_t j = 0; j < blockSize; j++)
            {
                T val = tmp.GetValue(j);
                qkCalc.SetValue(j * blockSize + j, val);
            }

            auto qkOut = qkOutQueue.AllocTensor<TOut>();
            Cast(qkOut, qkCalc, AscendC::RoundMode::CAST_ROUND, idtQkSizeAligned);
            qkOutQueue.EnQue(qkOut);
            qkCalcQueue.FreeTensor(qkCalc);

            qkOutQueue.DeQue<T>();
            AscendC::DataCopy(out[outOffset], qkOut, idtQkSize);
            qkOutQueue.FreeTensor(qkOut);
        }
    }

    __aicore__ inline void GenerateKVDecay(AscendC::GlobalTensor<T>& slopeGm)
    {
        AscendC::DataCopyParams dataCopyParams;
        AscendC::DataCopyPadParams dataCopyPadParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.dstStride = 0;
        dataCopyParams.srcStride = 0;
        dataCopyParams.blockLen = numHeads * sizeof(T);
        dataCopyPadParams.rightPadding = 0;
        dataCopyPadParams.paddingValue = 0;

        auto slopeLocal = slopeInQueue.AllocTensor<T>();
        AscendC::DataCopyPad(slopeLocal, slopeGm, dataCopyParams, dataCopyPadParams);
        slopeInQueue.EnQue(slopeLocal);

        slopeInQueue.DeQue<T>();
        AscendC::Muls(slopeLocal, slopeLocal, T(-1) * static_cast<T>(blockSize), numHeadsAligned);
        slopeInQueue.EnQue(slopeLocal);

        slopeInQueue.DeQue<T>();
        AscendC::Exp(slopeLocal, slopeLocal, numHeadsAligned);
        slopeInQueue.EnQue(slopeLocal);
    }

    __aicore__ inline void Generate(AscendC::GlobalTensor<TOut>& outQGm,
                                    AscendC::GlobalTensor<TOut>& outKGm,
                                    AscendC::GlobalTensor<TOut>& outKVGm,
                                    AscendC::GlobalTensor<T>& slopeGm)
    {
        if (coreIdx < numHeads / headsPerCore)
        {
            // Q Decay

            pipe->InitBuffer(calcBufPing, blockSizeAlignedBytes);
            pipe->InitBuffer(calcBufPong, blockSizeAlignedBytes);
            pipe->InitBuffer(qkCalcQueue, 1, idtQkSizeAlignedBytes);
            pipe->InitBuffer(qkOutQueue, 1, idtQkOutSizeAlignedBytes);

            auto idx = calcBufPing.Get<T>();
            auto tmp = calcBufPong.Get<T>();

            AscendC::CreateVecIndex(idx, (T)1, blockSizeAligned);
            pipe_barrier(PIPE_V);

            GenerateQK(outQGm, slopeGm, idx, tmp);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
        pipe->Reset();

        if (coreIdx < numHeads / headsPerCore)
        {
            // K Decay

            pipe->InitBuffer(calcBufPing, blockSizeAlignedBytes);
            pipe->InitBuffer(calcBufPong, blockSizeAlignedBytes);
            pipe->InitBuffer(qkCalcQueue, 1, idtQkSizeAlignedBytes);
            pipe->InitBuffer(qkOutQueue, 1, idtQkOutSizeAlignedBytes);

            auto idx = calcBufPing.Get<T>();
            auto tmp = calcBufPong.Get<T>();

            AscendC::CreateVecIndex(idx, (T)1, blockSizeAligned);
            pipe_barrier(PIPE_V);

            AscendC::Duplicate(tmp, static_cast<T>(blockSize), blockSizeAligned);
            pipe_barrier(PIPE_V);

            AscendC::Sub(idx, tmp, idx, blockSizeAligned);
            pipe_barrier(PIPE_V);

            GenerateQK(outKGm, slopeGm, idx, tmp);
        }

        AscendC::PipeBarrier<PIPE_ALL>();
        pipe->Reset();

        if (coreIdx < numHeads / headsPerCore)
        {
            // KV Decay

            pipe->InitBuffer(kvCalcQueue, 1, idtKvSizeAlignedBytes);
            pipe->InitBuffer(kvOutQueue, 1, idtKvOutSizeAlignedBytes);
            pipe->InitBuffer(slopeInQueue, 1, numHeadsAlignedBytes);

            GenerateKVDecay(slopeGm);
            auto slopeLocal = slopeInQueue.DeQue<T>();

            for (int64_t i = 0; i < headsPerCore; i++)
            {
                int64_t head = headsPerCore * coreIdx + i;
                int64_t outOffset = head * idtKvSize;

                T val = slopeLocal.GetValue(head);
                pipe_barrier(PIPE_V);

                auto kvCalc = kvCalcQueue.AllocTensor<T>();
                AscendC::Duplicate(kvCalc, T(0), idtKvSizeAligned);
                kvCalcQueue.EnQue(kvCalc);

                kvCalcQueue.DeQue<T>();
                for (size_t j = 0; j < headSize; j++)
                {
                    kvCalc.SetValue(j * headSize + j, val);
                }
                kvCalcQueue.EnQue(kvCalc);

                kvCalcQueue.DeQue<T>();
                auto kvOut = kvOutQueue.AllocTensor<TOut>();
                Cast(kvOut, kvCalc, AscendC::RoundMode::CAST_ROUND, idtKvSizeAligned);
                kvOutQueue.EnQue(kvOut);
                kvCalcQueue.FreeTensor(kvCalc);

                kvOutQueue.DeQue<T>();
                AscendC::DataCopy(outKVGm[outOffset], kvOut, idtKvSize);
                kvOutQueue.FreeTensor(kvOut);
            }

            slopeInQueue.FreeTensor(slopeLocal);
        }
    }

    __aicore__ inline void
    Generate(__gm__ uint8_t* outQ, __gm__ uint8_t* outK, __gm__ uint8_t* outKV, __gm__ uint8_t* slope)
    {
        AscendC::GlobalTensor<TOut> outQGm;
        AscendC::GlobalTensor<TOut> outKGm;
        AscendC::GlobalTensor<TOut> outKVGm;
        AscendC::GlobalTensor<T> slopeGm;

        outQGm.SetGlobalBuffer((__gm__ TOut*)outQ, numHeads * idtQkSize);
        outKGm.SetGlobalBuffer((__gm__ TOut*)outK, numHeads * idtQkSize);
        outKVGm.SetGlobalBuffer((__gm__ TOut*)outKV, numHeads * idtKvSize);
        slopeGm.SetGlobalBuffer((__gm__ T*)slope, numHeads);

        Generate(outQGm, outKGm, outKVGm, slopeGm);
    }

private:
    AscendC::TPipe* pipe;

    int64_t headSize;

    int64_t numHeads;
    int64_t numHeadsAligned;
    int64_t numHeadsAlignedBytes;

    int64_t blockSize;
    int64_t blockSizeAligned;
    int64_t blockSizeAlignedBytes;

    int64_t idtQkSize;
    int64_t idtQkSizeAlignedBytes;
    int64_t idtQkSizeAligned;
    int64_t idtQkOutSizeAlignedBytes;

    int64_t idtKvSize;
    int64_t idtKvSizeAlignedBytes;
    int64_t idtKvSizeAligned;
    int64_t idtKvOutSizeAlignedBytes;

    int64_t headsPerCore;
    int64_t coreIdx;

    AscendC::TQue<AscendC::QuePosition::VECIN, 1> slopeInQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> qkOutQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> kvOutQueue;

    AscendC::TQue<AscendC::QuePosition::VECCALC, 1> qkCalcQueue;
    AscendC::TQue<AscendC::QuePosition::VECCALC, 1> kvCalcQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBufPing;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBufPong;
};

} // namespace LightningAttn::Kernel
