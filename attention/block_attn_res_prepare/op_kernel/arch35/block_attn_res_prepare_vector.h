/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_ATTN_RES_PREPARE_VECTOR_H
#define BLOCK_ATTN_RES_PREPARE_VECTOR_H

#include "block_attn_res_prepare_tiling_data.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "vf/block_attn_res_prepare_vf.h"

namespace BlockAttnResPrepare {

using namespace AscendC;

class BlockAttnResPrepareVector {
public:
    __aicore__ inline explicit BlockAttnResPrepareVector(
        const optiling::BlockAttnResPrepareTilingData *__restrict tilingData)
        : tilingData_(tilingData)
    {
        for (uint32_t eventIndex = 0; eventIndex < MAX_PIPE_ID; ++eventIndex) {
            set_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(eventIndex));
        }
        for (uint32_t eventIndex = 0; eventIndex < MTE3_MAX_PIPE_ID; ++eventIndex) {
            set_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(eventIndex));
        }
    }

    __aicore__ inline ~BlockAttnResPrepareVector()
    {
        for (uint32_t eventIndex = 0; eventIndex < MAX_PIPE_ID; ++eventIndex) {
            wait_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(eventIndex));
        }
        for (uint32_t eventIndex = 0; eventIndex < MTE3_MAX_PIPE_ID; ++eventIndex) {
            wait_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(eventIndex));
        }
    }

    __aicore__ inline bool Init(GM_ADDR blockResidual, GM_ADDR effectiveQuery, GM_ADDR validBlocks,
                                GM_ADDR softmaxMax, GM_ADDR weightedOutput, GM_ADDR softmaxSum)
    {
        initialized_ = false;
        const uint64_t dAlign = (tilingData_->totalD / FP32_32B_ALIGN_ELEMS) * FP32_32B_ALIGN_ELEMS +
                                ((tilingData_->totalD % FP32_32B_ALIGN_ELEMS) != 0U ? FP32_32B_ALIGN_ELEMS : 0U);
        const uint64_t dTileNum = tilingData_->totalD / tilingData_->baseD +
                                  ((tilingData_->totalD % tilingData_->baseD) != 0U ? 1U : 0U);
        if (dAlign > MAX_UINT32_VALUE || dTileNum == 0U || dTileNum > MAX_UINT32_VALUE) {
            return false;
        }
        dAlign_ = static_cast<uint32_t>(dAlign);
        dTileNum_ = static_cast<uint32_t>(dTileNum);

        // Q/V/O are UB rows; 32B (8 FP32 elements) is the required address
        // granularity.  VF keeps the 64-element register width internally and
        // masks tail elements during arithmetic.
        const uint64_t baseDAlign =
            ((static_cast<uint64_t>(tilingData_->baseD) + FP32_32B_ALIGN_ELEMS - 1U) /
             FP32_32B_ALIGN_ELEMS) * FP32_32B_ALIGN_ELEMS;
        if (baseDAlign > MAX_UINT32_VALUE) {
            return false;
        }
        baseDAlign_ = static_cast<uint32_t>(baseDAlign);
        qBufferNum_ = tilingData_->qBufferNum;
        vBufferNum_ = tilingData_->vBufferNum;

        blockResidualAddr_ = reinterpret_cast<__gm__ float *>(blockResidual);
        effectiveQueryAddr_ = reinterpret_cast<__gm__ float *>(effectiveQuery);
        validBlocksAddr_ = reinterpret_cast<__gm__ int32_t *>(validBlocks);
        softmaxMaxAddr_ = reinterpret_cast<__gm__ float *>(softmaxMax);
        weightedOutputAddr_ = reinterpret_cast<__gm__ float *>(weightedOutput);
        softmaxSumAddr_ = reinterpret_cast<__gm__ float *>(softmaxSum);

        BuildUbLayout();
        initialized_ = true;
        return true;
    }

    __aicore__ inline void Process()
    {
        if (!initialized_) {
            return;
        }
        const uint32_t blockIdx = GetBlockIdx();
        if (blockIdx >= tilingData_->usedCoreNum) {
            return;
        }
        int32_t validNValue = ReadValidBlocks();
        if (validNValue < 1) {
            // TODO 狗造相应场景的输出
            return;
        } else if (validNValue > tilingData_->totalN) {
            validNValue = static_cast<int32_t>(tilingData_->totalN);
        }
        const uint32_t validN = static_cast<uint32_t>(validNValue);
        // validN comes from valid_blocks at runtime. The host only provides
        // the capacity; choose cache/reload here for the concrete N.
        const bool cacheAllV = validNValue <= tilingData_->vCacheRows && baseDAlign_ >= dAlign_;

        const uint64_t workStart = static_cast<uint64_t>(blockIdx) * tilingData_->blockFactor +
                                   ((blockIdx < tilingData_->bigCoreNum) ? blockIdx : tilingData_->bigCoreNum);
        const uint32_t workCount = tilingData_->blockFactor + ((blockIdx < tilingData_->bigCoreNum) ? 1U : 0U);
        for (uint32_t workOffset = 0; workOffset < workCount; ++workOffset) {
            ProcessWork(workStart + workOffset, validN, cacheAllV);
        }
    }

private:
    static constexpr uint64_t FP32_BYTES = sizeof(float);
    static constexpr uint32_t FP32_32B_ALIGN_ELEMS = 8U;
    static constexpr uint32_t MAX_PIPE_ID = 8U;
    static constexpr uint32_t MTE3_MAX_PIPE_ID = 4U;
    static constexpr uint64_t MAX_UINT32_VALUE = 0xFFFFFFFFULL;

    __aicore__ inline static bool IsBufferNumValid(uint32_t bufferNum)
    {
        return bufferNum == 1U || bufferNum == 2U;
    }

    __aicore__ inline static event_t GetEvent(uint32_t bufferIndex)
    {
        return bufferIndex == 0U ? static_cast<event_t>(EVENT_ID0) : static_cast<event_t>(EVENT_ID1);
    }

    __aicore__ inline static void WaitMte2ToVector(event_t eventId)
    {
        set_flag(PIPE_MTE2, PIPE_V, eventId);
        wait_flag(PIPE_MTE2, PIPE_V, eventId);
    }

    __aicore__ inline static void WaitVectorToMte3(event_t eventId)
    {
        set_flag(PIPE_V, PIPE_MTE3, eventId);
        wait_flag(PIPE_V, PIPE_MTE3, eventId);
    }

    __aicore__ inline void BuildUbLayout()
    {
        qUbOffset_ = 0U;
        oUbOffset_ = qUbOffset_ +
                     static_cast<uint64_t>(tilingData_->vBufferNum) * baseDAlign_ * FP32_BYTES;
        statUbOffset_ = oUbOffset_ +
                        static_cast<uint64_t>(tilingData_->oBufferNum) * baseDAlign_ * FP32_BYTES;
        vCacheUbOffset_ = statUbOffset_ + static_cast<uint64_t>(tilingData_->statUbElems) * FP32_BYTES;
        vUbOffset_ = vCacheUbOffset_ +
                     static_cast<uint64_t>(tilingData_->qBufferNum) * baseDAlign_ * FP32_BYTES;
    }

    template <typename T>
    __aicore__ inline static __ubuf__ T *GetUbAddr(uint64_t byteOffset)
    {
        return AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, T>(byteOffset).Get();
    }

    template <typename T>
    __aicore__ inline static auto MakeNDExtLayout(int64_t rows, int64_t columns, int64_t rowPitch)
    {
        auto shape = AscendC::Te::MakeShape(
            AscendC::Te::MakeShape(AscendC::Std::Int<1>{}, rows),
            AscendC::Te::MakeShape(AscendC::Std::Int<1>{}, columns));
        auto stride = AscendC::Te::MakeStride(
            AscendC::Te::MakeStride(AscendC::Std::Int<0>{}, rowPitch),
            AscendC::Te::MakeStride(AscendC::Std::Int<0>{}, AscendC::Std::Int<1>{}));
        return AscendC::Te::MakePatternLayout<AscendC::Te::NDExtLayoutPtn,
                                              AscendC::Te::LayoutTraitDefault<T>>(shape, stride);
    }

    template <typename T>
    __aicore__ inline static void CopyGmToUb(uint64_t dstByteOffset, __gm__ T *src, int64_t validElements,
                                             int64_t ubPitch)
    {
        auto ubTensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, T>(dstByteOffset),
            MakeNDExtLayout<T>(1, validElements, ubPitch));
        auto gmTensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(src),
            MakeNDExtLayout<T>(1, validElements, validElements));
        auto copyGmToUb = AscendC::Te::MakeCopy(AscendC::Te::CopyGM2UB{});
        AscendC::Te::Copy(copyGmToUb, ubTensor, gmTensor);
    }

    template <typename T>
    __aicore__ inline static void CopyUbToGm(__gm__ T *dst, uint64_t srcByteOffset, int64_t validElements,
                                             int64_t ubPitch)
    {
        auto gmTensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(dst),
            MakeNDExtLayout<T>(1, validElements, validElements));
        auto ubTensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::UB, T>(srcByteOffset),
            MakeNDExtLayout<T>(1, validElements, ubPitch));
        auto copyUbToGm = AscendC::Te::MakeCopy(AscendC::Te::CopyUB2GM{});
        AscendC::Te::Copy(copyUbToGm, gmTensor, ubTensor);
    }

    __aicore__ inline int32_t ReadValidBlocks() const
    {
        auto validBlocksTensor = AscendC::Te::MakeTensor(
            AscendC::Te::MakeMemPtr<AscendC::Te::Location::GM>(validBlocksAddr_),
            MakeNDExtLayout<int32_t>(1, 1, 1));
        return validBlocksTensor[AscendC::Te::MakeCoord(static_cast<int64_t>(0), static_cast<int64_t>(0))];
    }

    __aicore__ inline __ubuf__ float *CopyInQ(uint64_t qOffset, uint32_t validD, uint32_t dLoop)
    {
        const uint32_t bufferIndex = dLoop % tilingData_->qBufferNum;
        const uint64_t byteOffset = qUbOffset_ +
                                    static_cast<uint64_t>(bufferIndex) * baseDAlign_ * FP32_BYTES;
        CopyGmToUb(byteOffset, effectiveQueryAddr_ + qOffset, validD, baseDAlign_);
        WaitMte2ToVector(GetEvent(bufferIndex));
        return GetUbAddr<float>(byteOffset);
    }

    __aicore__ inline __ubuf__ float *CopyInV(uint64_t vOffset, uint32_t validD, uint64_t vLoop_)
    {
        const uint32_t bufferIndex = static_cast<uint32_t>(vLoop_ % tilingData_->vBufferNum);
        const uint64_t byteOffset = vUbOffset_ +
                                    static_cast<uint64_t>(bufferIndex) * baseDAlign_ * FP32_BYTES;
        CopyGmToUb(byteOffset, blockResidualAddr_ + vOffset, validD, baseDAlign_);
        WaitMte2ToVector(GetEvent(bufferIndex));
        return GetUbAddr<float>(byteOffset);
    }

    __aicore__ inline __ubuf__ float *CopyVToCache(uint64_t vOffset, uint32_t cacheOffset, uint32_t validD,
                                                   uint32_t bufferIndex)
    {
        const uint64_t byteOffset = vCacheUbOffset_ + static_cast<uint64_t>(cacheOffset) * FP32_BYTES;
        CopyGmToUb(byteOffset, blockResidualAddr_ + vOffset, validD, dAlign_);
        WaitMte2ToVector(GetEvent(bufferIndex));
        return GetUbAddr<float>(byteOffset);
    }

    __aicore__ inline void CopyOutStats(uint64_t statOutputOffset)
    {
        const event_t eventId = static_cast<event_t>(EVENT_ID0);
        WaitVectorToMte3(eventId);
        CopyUbToGm(softmaxMaxAddr_ + statOutputOffset,
                   statUbOffset_ + static_cast<uint64_t>(BlockAttnResPrepareVF::MAX_OFFSET) * FP32_BYTES, 1, 8);
        CopyUbToGm(softmaxSumAddr_ + statOutputOffset,
                   statUbOffset_ + static_cast<uint64_t>(BlockAttnResPrepareVF::SUM_OFFSET) * FP32_BYTES, 1, 8);
    }

    __aicore__ inline void CopyOutO(uint64_t outputOffset, uint32_t validD, uint32_t dLoop)
    {
        const uint32_t bufferIndex = dLoop % tilingData_->oBufferNum;
        const uint64_t byteOffset = oUbOffset_ +
                                    static_cast<uint64_t>(bufferIndex) * baseDAlign_ * FP32_BYTES;
        const event_t eventId = GetEvent(bufferIndex);
        WaitVectorToMte3(eventId);
        CopyUbToGm(weightedOutputAddr_ + outputOffset, byteOffset, validD, baseDAlign_);
    }

    __aicore__ inline void AccumulateStats(__ubuf__ float *qAddr, __ubuf__ float *vAddr,
                                           __ubuf__ float *statAddr, uint32_t dLoop, uint32_t validD,
                                           uint32_t nIndex)
    {
        if (dLoop == 0U) {
            BlockAttnResPrepareVF::AccumulateSquareDot<true>(qAddr, vAddr, statAddr, validD, nIndex);
        } else {
            BlockAttnResPrepareVF::AccumulateSquareDot<false>(qAddr, vAddr, statAddr, validD, nIndex);
        }
    }

    __aicore__ inline void AccumulateOutput(__ubuf__ float *vAddr, __ubuf__ float *weightAddr,
                                            __ubuf__ float *outputAddr, uint32_t nIndex, uint32_t validN,
                                            uint32_t validD)
    {
        if (validN == 1U) {
            BlockAttnResPrepareVF::CopySingleBlock(vAddr, outputAddr, validD);
        } else if (nIndex == 0U) {
            BlockAttnResPrepareVF::WeightedAccumulate<true>(vAddr, weightAddr, outputAddr, validD);
        } else {
            BlockAttnResPrepareVF::WeightedAccumulate<false>(vAddr, weightAddr, outputAddr, validD);
        }
    }

    __aicore__ inline void FinalizeStats(__ubuf__ float *statAddr, uint32_t validN)
    {
        const float reciprocalD = 1.0F / static_cast<float>(tilingData_->totalD);
        if (validN == 1U) {
            BlockAttnResPrepareVF::FinalizeSingleBlock(statAddr, reciprocalD, tilingData_->epsilon);
        } else {
            BlockAttnResPrepareVF::FinalizeSoftmax(statAddr, validN, reciprocalD, tilingData_->epsilon);
        }
    }

    __aicore__ inline void ProcessWork(uint64_t workId, uint32_t validN, bool cacheAllV)
    {
        const uint64_t tIndex = workId / tilingData_->totalS;
        const uint64_t sIndex = workId - tIndex * tilingData_->totalS;
        __ubuf__ float *statAddr = GetUbAddr<float>(statUbOffset_);

        for (uint32_t dLoop = 0; dLoop < dTileNum_; ++dLoop) {
            const uint32_t dOffset = dLoop * tilingData_->baseD;
            const uint64_t remainingD = tilingData_->totalD - dOffset;
            const uint32_t validD = static_cast<uint32_t>(
                remainingD < tilingData_->baseD ? remainingD : tilingData_->baseD);
            const uint64_t qOffset = sIndex * tilingData_->totalD + dOffset;
            wait_flag(PIPE_V, PIPE_MTE2, qLoop_ & 1);
            __ubuf__ float *qAddr = CopyInQ(qOffset, validD, dLoop);

            for (uint32_t nIndex = 0; nIndex < validN; ++nIndex) {
                const uint64_t vOffset =
                    (tIndex * tilingData_->totalN + nIndex) * tilingData_->totalD + dOffset;
                __ubuf__ float *vAddr;
                if (cacheAllV) {
                    wait_flag(PIPE_V, PIPE_MTE2, qBufferNum_ + nIndex);
                    const uint32_t cacheOffset = nIndex * dAlign_ + dOffset;
                    vAddr = CopyVToCache(vOffset, cacheOffset, validD, nIndex & 1U);
                } else {
                    wait_flag(PIPE_V, PIPE_MTE2, qBufferNum_ + (vLoop_ & 1));
                    vAddr = CopyInV(vOffset, validD, vLoop_);
                }
                AccumulateStats(qAddr, vAddr, statAddr, dLoop, validD, nIndex);
                if (!cacheAllV) {
                    set_flag(PIPE_V, PIPE_MTE2, qBufferNum_ + (vLoop_++ & 1));
                }
            }
            set_flag(PIPE_V, PIPE_MTE2, qLoop_ & 1);
            qLoop_++;
        }

        FinalizeStats(statAddr, validN);
        CopyOutStats(sIndex * tilingData_->totalT + tIndex);

        __ubuf__ float *weightAddr = statAddr + BlockAttnResPrepareVF::DOT_OFFSET;
        for (uint32_t dLoop = 0; dLoop < dTileNum_; ++dLoop) {
            const uint32_t dOffset = dLoop * tilingData_->baseD;
            const uint64_t remainingD = tilingData_->totalD - dOffset;
            const uint32_t validD = static_cast<uint32_t>(
                remainingD < tilingData_->baseD ? remainingD : tilingData_->baseD);
            const uint32_t outputBufferIndex = oLoop_ % tilingData_->oBufferNum;
            const uint64_t outputByteOffset =
                oUbOffset_ + static_cast<uint64_t>(outputBufferIndex) * baseDAlign_ * FP32_BYTES;
            __ubuf__ float *outputAddr = GetUbAddr<float>(outputByteOffset);
            wait_flag(PIPE_MTE3, PIPE_V, oLoop_ & 1);
            for (uint32_t nIndex = 0; nIndex < validN; ++nIndex) {
                __ubuf__ float *vAddr;
                if (cacheAllV) {
                    const uint32_t cacheOffset = nIndex * dAlign_ + dOffset;
                    vAddr = GetUbAddr<float>(vCacheUbOffset_ + static_cast<uint64_t>(cacheOffset) * FP32_BYTES);
                } else {
                    wait_flag(PIPE_V, PIPE_MTE2, qBufferNum_ + (vLoop_ & 1));
                    const uint64_t vOffset =
                        (tIndex * tilingData_->totalN + nIndex) * tilingData_->totalD + dOffset;
                    vAddr = CopyInV(vOffset, validD, vLoop_);
                }
                AccumulateOutput(vAddr, weightAddr + nIndex, outputAddr, nIndex, validN, validD);
                if (!cacheAllV) {
                    set_flag(PIPE_V, PIPE_MTE2, qBufferNum_ + (vLoop_++ & 1));
                } else {
                    set_flag(PIPE_V, PIPE_MTE2, qBufferNum_ + nIndex);
                }
            }

            const uint64_t outputOffset =
                (sIndex * tilingData_->totalT + tIndex) * tilingData_->totalD + dOffset;
            CopyOutO(outputOffset, validD, oLoop_);
            set_flag(PIPE_MTE3, PIPE_V, oLoop_++ & 1);

        }
    }

    const optiling::BlockAttnResPrepareTilingData *__restrict tilingData_ = nullptr;
    bool initialized_ = false;

    uint64_t qUbOffset_ = 0U;
    uint64_t vUbOffset_ = 0U;
    uint64_t oUbOffset_ = 0U;
    uint64_t statUbOffset_ = 0U;
    uint64_t vCacheUbOffset_ = 0U;
    uint32_t baseDAlign_ = 0U;
    uint32_t dAlign_ = 0U;
    uint32_t dTileNum_ = 0U;
    uint32_t qLoop_ = 0;
    uint32_t vLoop_ = 0;
    uint32_t oLoop_ = 0;
    uint8_t qBufferNum_ = 1;
    uint8_t vBufferNum_ = 1;

    __gm__ float *blockResidualAddr_ = nullptr;
    __gm__ float *effectiveQueryAddr_ = nullptr;
    __gm__ int32_t *validBlocksAddr_ = nullptr;
    __gm__ float *softmaxMaxAddr_ = nullptr;
    __gm__ float *weightedOutputAddr_ = nullptr;
    __gm__ float *softmaxSumAddr_ = nullptr;
};

} // namespace BlockAttnResPrepare

#endif // BLOCK_ATTN_RES_PREPARE_VECTOR_H
