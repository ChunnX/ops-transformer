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

#include <limits>

namespace LightningAttn::Kernel
{

template <typename T> class QkDecayGenerator
{
public:
    __aicore__ inline QkDecayGenerator() {}

    __aicore__ inline void Init(AscendC::TPipe* pipe, int64_t numHeads, int64_t blockSize)
    {
        this->pipe = pipe;
        this->numHeads = numHeads;
        this->blockSize = blockSize;

        const int64_t alignSize = 256l;

        blockSizeAlignedBytes = blockSize * sizeof(T);
        blockSizeAlignedBytes = CeilDiv(blockSizeAlignedBytes, alignSize) * alignSize;
        blockSizeAligned = blockSizeAlignedBytes / sizeof(T);

        coreIdx = AscendC::GetBlockIdx();
        headsPerCore = 1;

        pipe->InitBuffer(outQueue, 1, blockSizeAlignedBytes);
        pipe->InitBuffer(calcBufPing, blockSizeAlignedBytes);
        pipe->InitBuffer(calcBufPong, blockSizeAlignedBytes);
    }

    __aicore__ inline void GenerateQKDecay(AscendC::GlobalTensor<T>& out,
                                           AscendC::GlobalTensor<T>& slopes,
                                           AscendC::LocalTensor<T>& indexes,
                                           AscendC::LocalTensor<T>& tmp)
    {
        for (int64_t i = 0; i < headsPerCore; i++)
        {
            int64_t head = headsPerCore * coreIdx + i;
            int64_t outOffset = head * blockSize;

            T slope = -1 * slopes.GetValue(head);
            pipe_barrier(PIPE_V);

            AscendC::Muls(tmp, indexes, slope, blockSizeAligned);
            pipe_barrier(PIPE_V);

            auto outLocal = outQueue.AllocTensor<T>();
            AscendC::Exp(outLocal, tmp, blockSizeAligned);

            outQueue.EnQue(outLocal);
            outQueue.DeQue<T>();

            AscendC::DataCopyParams dataCopyParams;
            dataCopyParams.blockCount = 1;
            dataCopyParams.dstStride = 0;
            dataCopyParams.srcStride = 0;
            dataCopyParams.blockLen = blockSize * sizeof(T);
            AscendC::DataCopyPad(out[outOffset], outLocal, dataCopyParams);
            outQueue.FreeTensor(outLocal);
        }
    }

    __aicore__ inline void GenerateQDecay(AscendC::GlobalTensor<T>& out, AscendC::GlobalTensor<T>& slopes)
    {
        if (coreIdx >= numHeads / headsPerCore)
        {
            return;
        }

        pipe->InitBuffer(calcBufPing, blockSizeAlignedBytes);
        pipe->InitBuffer(calcBufPong, blockSizeAlignedBytes);

        auto idx = calcBufPing.Get<T>();
        auto tmp = calcBufPong.Get<T>();

        AscendC::CreateVecIndex(idx, (T)1, blockSizeAligned);
        pipe_barrier(PIPE_V);

        GenerateQKDecay(out, slopes, idx, tmp);
    }

    __aicore__ inline void GenerateKDecay(AscendC::GlobalTensor<T>& out, AscendC::GlobalTensor<T>& slopes)
    {
        if (coreIdx >= numHeads / headsPerCore)
        {
            return;
        }

        auto idx = calcBufPing.Get<T>();
        auto tmp = calcBufPong.Get<T>();

        AscendC::CreateVecIndex(idx, (T)1, blockSizeAligned);
        pipe_barrier(PIPE_V);

        AscendC::Duplicate(tmp, static_cast<T>(blockSize), blockSizeAligned);
        pipe_barrier(PIPE_V);

        AscendC::Sub(idx, tmp, idx, blockSizeAligned);
        pipe_barrier(PIPE_V);

        GenerateQKDecay(out, slopes, idx, tmp);
    }

public:
    __aicore__ inline void GenerateQDecay(__gm__ uint8_t* out, __gm__ uint8_t* slopes)
    {
        AscendC::GlobalTensor<T> outGm;
        AscendC::GlobalTensor<T> slopesGm;
        outGm.SetGlobalBuffer((__gm__ T*)out, numHeads * blockSize);
        slopesGm.SetGlobalBuffer((__gm__ T*)slopes, numHeads);
        GenerateQDecay(outGm, slopesGm);
    }

    __aicore__ inline void GenerateKDecay(__gm__ uint8_t* out, __gm__ uint8_t* slopes)
    {
        AscendC::GlobalTensor<T> outGm;
        AscendC::GlobalTensor<T> slopesGm;
        outGm.SetGlobalBuffer((__gm__ T*)out, numHeads * blockSize);
        slopesGm.SetGlobalBuffer((__gm__ T*)slopes, numHeads);
        GenerateKDecay(outGm, slopesGm);
    }

private:
    AscendC::TPipe* pipe;

    int64_t numHeads;

    int64_t blockSize;
    int64_t blockSizeAligned;
    int64_t blockSizeAlignedBytes;

    int64_t headsPerCore;
    int64_t coreIdx;

    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBufPing;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBufPong;
};

template <typename T> class KvDecayGenerator
{
public:
    __aicore__ inline KvDecayGenerator() {}

    __aicore__ inline void Init(AscendC::TPipe* pipe, int64_t numHeads, int64_t blockSize)
    {
        this->pipe = pipe;
        this->numHeads = numHeads;
        this->blockSize = blockSize;

        const int64_t alignSize = 256l;

        numHeadsAlignedBytes = numHeads * sizeof(T);
        numHeadsAlignedBytes = CeilDiv(numHeadsAlignedBytes, alignSize) * alignSize;
        numHeadsAligned = numHeadsAlignedBytes / sizeof(T);

        coreIdx = AscendC::GetBlockIdx();
        headsPerCore = 1;

        pipe->InitBuffer(inQueue, 1, numHeadsAlignedBytes);
        pipe->InitBuffer(outQueue, 1, numHeadsAlignedBytes);
    }

    __aicore__ inline void GenerateKVDecay(AscendC::GlobalTensor<T>& out, AscendC::GlobalTensor<T>& slopes)
    {
        if (coreIdx > 0)
        {
            return;
        }

        AscendC::DataCopyParams dataCopyParams;
        AscendC::DataCopyPadParams dataCopyPadParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.dstStride = 0;
        dataCopyParams.srcStride = 0;
        dataCopyParams.blockLen = numHeads * sizeof(T);
        dataCopyPadParams.rightPadding = 0;
        dataCopyPadParams.paddingValue = 0;

        auto slopesLocal = inQueue.AllocTensor<T>();
        AscendC::DataCopyPad(slopesLocal, slopes, dataCopyParams, dataCopyPadParams);

        inQueue.EnQue(slopesLocal);
        inQueue.DeQue<T>();

        AscendC::Muls(slopesLocal, slopesLocal, T(-1) * static_cast<T>(blockSize), numHeads);
        inQueue.EnQue(slopesLocal);
        inQueue.DeQue<T>();

        auto outLocal = outQueue.AllocTensor<T>();
        AscendC::Exp(outLocal, slopesLocal, numHeads);
        inQueue.FreeTensor(slopesLocal);

        outQueue.EnQue(outLocal);
        outQueue.DeQue<T>();

        AscendC::DataCopyPad(out, outLocal, dataCopyParams);
        outQueue.FreeTensor(outLocal);
    }

public:
    __aicore__ inline void GenerateKVDecay(__gm__ uint8_t* out, __gm__ uint8_t* slopes)
    {
        AscendC::GlobalTensor<T> outGm;
        AscendC::GlobalTensor<T> slopesGm;
        outGm.SetGlobalBuffer((__gm__ T*)out, numHeads);
        slopesGm.SetGlobalBuffer((__gm__ T*)slopes, numHeads);
        GenerateKVDecay(outGm, slopesGm);
    }

private:
    AscendC::TPipe* pipe;

    int64_t numHeads;
    int64_t numHeadsAligned;
    int64_t numHeadsAlignedBytes;

    int64_t blockSize;
    int64_t headsPerCore;
    int64_t coreIdx;

    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueue;
};

template <typename T> class DiagDecayGenerator
{
public:
    __aicore__ inline DiagDecayGenerator() {}

    __aicore__ inline void Init(AscendC::TPipe* pipe, int64_t numHeads, int64_t blockSize)
    {
        this->pipe = pipe;
        this->numHeads = numHeads;
        this->blockSize = blockSize;

        subBlockSize = 64;
        if (blockSize < subBlockSize)
        {
            subBlockSize = blockSize;
        }

        diagBlockSize = blockSize * blockSize;
        diagSubBlockSize = subBlockSize * subBlockSize;

        coreIdx = AscendC::GetBlockIdx();
        headsPerCore = 1;
        elInBlock = 32 / sizeof(T);

        pipe->InitBuffer(outQueue, 1, diagSubBlockSize * sizeof(T));
        pipe->InitBuffer(calcBuf, diagSubBlockSize * sizeof(T));
        pipe->InitBuffer(idxBuf, subBlockSize * sizeof(T));
    }

    __aicore__ inline void
    FillSubBlock(const AscendC::LocalTensor<T>& out, T slope, int64_t idxOffset, bool needMask = true)
    {
        uint64_t maskShift = subBlockSize - 1;
        uint64_t maxMask = (1ul << maskShift) | ((1ul << maskShift) - 1ul);
        uint64_t mask[] = {maxMask};

        auto idx = idxBuf.Get<T>();
        AscendC::CreateVecIndex(idx, T(idxOffset), mask, 1, 1, 1);
        pipe_barrier(PIPE_V);

        for (int64_t i = 0; i < subBlockSize; i++)
        {
            AscendC::LocalTensor idxL = out[i * subBlockSize];

            AscendC::Copy(idxL, idx, mask, 1, {1, 1, 1, 1});
            pipe_barrier(PIPE_V);

            AscendC::Adds(idxL, idxL, T(-1) * T(i), mask, 1, {1, 1, 1, 1});
            pipe_barrier(PIPE_V);

            AscendC::Muls(idxL, idxL, slope, mask, 1, {1, 1, 1, 1});
            pipe_barrier(PIPE_V);

            if (needMask)
            {
                mask[0] = mask[0] ^ (1ul << i);
            }
        }
    }

    __aicore__ inline void TransposeSubBlock(const AscendC::LocalTensor<T>& out, const AscendC::LocalTensor<T>& in)
    {
        const int64_t width = 8;
        const int64_t height = 16;

        AscendC::TransDataTo5HDParams transDataParams;
        transDataParams.dstHighHalf = false;
        transDataParams.srcHighHalf = false;
        transDataParams.repeatTimes = subBlockSize / width;
        transDataParams.dstRepStride = subBlockSize;
        transDataParams.srcRepStride = 1;

        uint64_t dstLocalList[height];
        uint64_t srcLocalList[height];

        for (uint64_t k = 0; k < subBlockSize / height; k++)
        {
            uint64_t srcBaseOffset = k * subBlockSize * height;
            uint64_t dstBaseOffset = k * height;

            for (uint64_t i = 0; i < height; i++)
            {
                uint64_t srcOffset = srcBaseOffset + subBlockSize * i;
                uint64_t dstOffset = dstBaseOffset + (i / 2) * subBlockSize + (i % 2) * width;

                srcLocalList[i] = static_cast<uint64_t>(in[srcOffset].GetPhyAddr());
                dstLocalList[i] = static_cast<uint64_t>(out[dstOffset].GetPhyAddr());
            }

            AscendC::TransDataTo5HD<T>(dstLocalList, srcLocalList, transDataParams);
        }
    }

    __aicore__ inline void GenerateDiagDecay(AscendC::GlobalTensor<T>& out, AscendC::GlobalTensor<T>& slopes)
    {
        if (coreIdx >= numHeads / headsPerCore)
        {
            return;
        }

        int64_t subRows = blockSize / subBlockSize;
        int64_t subCols = blockSize / subBlockSize;

        for (int64_t h = 0; h < headsPerCore; h++)
        {
            int64_t head = headsPerCore * coreIdx + h;
            int64_t outOffset = head * diagBlockSize;
            T slope = -1 * slopes.GetValue(head);
            pipe_barrier(PIPE_V);

            for (int64_t row = 0; row < subRows; row++)
            {
                for (int64_t col = 0; col < subCols; col++)
                {
                    if (col >= row)
                    {
                        auto subBlock = calcBuf.Get<T>();
                        AscendC::Duplicate(subBlock, -std::numeric_limits<T>::infinity(), diagSubBlockSize);
                        pipe_barrier(PIPE_V);

                        FillSubBlock(subBlock, slope, (col - row) * subBlockSize, col == row);

                        AscendC::Exp(subBlock, subBlock, diagSubBlockSize);
                        pipe_barrier(PIPE_V);

                        auto outLocal = outQueue.AllocTensor<T>();
                        TransposeSubBlock(outLocal, subBlock);
                        outQueue.EnQue(outLocal);
                    }
                    else
                    {
                        auto outLocal = outQueue.AllocTensor<T>();
                        AscendC::Duplicate(outLocal, T(0), diagSubBlockSize);
                        outQueue.EnQue(outLocal);
                    }

                    auto outLocal = outQueue.DeQue<T>();

                    AscendC::DataCopyParams dataCopyParams;
                    dataCopyParams.blockCount = subBlockSize;
                    dataCopyParams.blockLen = subBlockSize / elInBlock;
                    dataCopyParams.srcStride = 0;
                    dataCopyParams.dstStride = (blockSize - subBlockSize) / elInBlock;

                    // swapping col and row due to transpose
                    const auto& outBlock = out[outOffset][col * blockSize * subBlockSize + row * subBlockSize];

                    DataCopy(outBlock, outLocal, dataCopyParams);
                    outQueue.FreeTensor(outLocal);
                }
            }
        }
    }

public:
    __aicore__ inline void GenerateDiagDecay(__gm__ uint8_t* out, __gm__ uint8_t* slopes)
    {
        AscendC::GlobalTensor<T> outGm;
        AscendC::GlobalTensor<T> slopesGm;
        outGm.SetGlobalBuffer((__gm__ T*)out, numHeads * blockSize * blockSize);
        slopesGm.SetGlobalBuffer((__gm__ T*)slopes, numHeads);
        GenerateDiagDecay(outGm, slopesGm);
    }

private:
    AscendC::TPipe* pipe;

    int64_t numHeads;
    int64_t blockSize;
    int64_t subBlockSize;

    int64_t diagBlockSize;
    int64_t diagSubBlockSize;

    int64_t headsPerCore;
    int64_t coreIdx;

    int64_t elInBlock;

    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> idxBuf;
};

} // namespace LightningAttn::Kernel
