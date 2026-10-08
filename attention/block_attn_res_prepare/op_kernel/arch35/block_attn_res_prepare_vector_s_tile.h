/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BLOCK_ATTN_RES_PREPARE_VECTOR_S_TILE_H
#define BLOCK_ATTN_RES_PREPARE_VECTOR_S_TILE_H

#include "block_attn_res_prepare_tiling_data.h"
#include "kernel_operator.h"
#include "tensor_api/tensor.h"
#include "vf/block_attn_res_prepare_vf.h"

namespace BlockAttnResPrepare {

using namespace AscendC;

template <typename INPUT_T>
class BlockAttnResPrepareVectorSTile {
public:
    __aicore__ inline explicit BlockAttnResPrepareVectorSTile(
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

    __aicore__ inline ~BlockAttnResPrepareVectorSTile()
    {
        for (uint32_t eventIndex = 0; eventIndex < MAX_PIPE_ID; ++eventIndex) {
            wait_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(eventIndex));
        }
        for (uint32_t eventIndex = 0; eventIndex < MTE3_MAX_PIPE_ID; ++eventIndex) {
            wait_flag(PIPE_MTE3, PIPE_V, static_cast<event_t>(eventIndex));
        }
    }

    __aicore__ inline void Init(GM_ADDR blockRes, GM_ADDR validBlocks, GM_ADDR pseudoQuery, GM_ADDR numerator,
                                GM_ADDR logitMax, GM_ADDR expSum)
    {
        initialized_ = false;
        validBlocksAddr_ = reinterpret_cast<__gm__ uint64_t *>(validBlocks);
        validN_ = ReadValidBlocks();
        const uint64_t dAlign =
            ((static_cast<uint64_t>(tilingData_->totalD) + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS) * FP32_REG_ELEMS;
        const uint64_t dTileNum =
            (static_cast<uint64_t>(tilingData_->totalD) + tilingData_->baseD - 1U) / tilingData_->baseD;
        dAlign_ = static_cast<uint32_t>(dAlign);
        dTileNum_ = static_cast<uint32_t>(dTileNum);
        // LoadAlign reads a complete FP32 vector register. A 64-element pitch
        // keeps a masked tail load inside the current Q/V/O logical UB row.
        const uint64_t baseDAlign =
            ((static_cast<uint64_t>(tilingData_->baseD) + FP32_REG_ELEMS - 1U) / FP32_REG_ELEMS) * FP32_REG_ELEMS;
        baseDAlign_ = static_cast<uint32_t>(baseDAlign);
        qBufferNum_ = tilingData_->qBufferNum;
        vBufferNum_ = tilingData_->vBufferNum;

        blockResAddr_ = reinterpret_cast<__gm__ INPUT_T *>(blockRes);
        pseudoQueryAddr_ = reinterpret_cast<__gm__ INPUT_T *>(pseudoQuery);
        numeratorAddr_ = reinterpret_cast<__gm__ float *>(numerator);
        logitMaxAddr_ = reinterpret_cast<__gm__ float *>(logitMax);
        expSumAddr_ = reinterpret_cast<__gm__ float *>(expSum);

        InitUbOffset();
        initialized_ = true;
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
        const uint64_t workStart = static_cast<uint64_t>(blockIdx) * tilingData_->blockFactor +
                                   ((blockIdx < tilingData_->bigCoreNum) ? blockIdx : tilingData_->bigCoreNum);
        const uint32_t workCount = tilingData_->blockFactor + ((blockIdx < tilingData_->bigCoreNum) ? 1U : 0U);
        uint64_t validNValue = validN_;
        if (validNValue == 0U) {
            // Zero valid blocks produces the identity state of online softmax.
            for (uint32_t workOffset = 0; workOffset < workCount; ++workOffset) {
                const uint64_t workId = workStart + workOffset;
                const uint64_t tIndex = workId / tilingData_->sTileNum;
                const uint32_t sTileIndex = static_cast<uint32_t>(workId % tilingData_->sTileNum);
                const uint32_t sTileSize = (tilingData_->totalS + tilingData_->sTileNum - 1U) / tilingData_->sTileNum;
                const uint32_t sStart = sTileIndex * sTileSize;
                const uint32_t sEnd = sStart + sTileSize < tilingData_->totalS ? sStart + sTileSize : tilingData_->totalS;
                for (uint32_t sIndex = sStart; sIndex < sEnd; ++sIndex) {
                    ProcessEmptyWork(tIndex * tilingData_->totalS + sIndex);
                }
            }
            return;
        } else if (validNValue > tilingData_->totalN) {
            validNValue = static_cast<uint64_t>(tilingData_->totalN);
        }
        const uint32_t validN = static_cast<uint32_t>(validNValue);
        for (uint32_t workOffset = 0; workOffset < workCount; ++workOffset) {
            const uint64_t workId = workStart + workOffset;
            const uint64_t tIndex = workId / tilingData_->sTileNum;
            const uint32_t sTileIndex = static_cast<uint32_t>(workId % tilingData_->sTileNum);
            const uint32_t sTileSize = (tilingData_->totalS + tilingData_->sTileNum - 1U) / tilingData_->sTileNum;
            const uint32_t sStart = sTileIndex * sTileSize;
            const uint32_t sEnd = sStart + sTileSize < tilingData_->totalS ? sStart + sTileSize : tilingData_->totalS;
            if ((sEnd - sStart) % STAT_BUFFER_NUM == 0U) {
                ProcessTGrouped(tIndex, sStart, sEnd, validN);
            } else {
                ProcessT(tIndex, sStart, sEnd, validN);
            }
        }
    }

private:
    static constexpr uint64_t FP32_BYTES = sizeof(float);
    static constexpr uint64_t INPUT_BYTES = sizeof(INPUT_T);
    static constexpr uint32_t FP32_32B_ALIGN_ELEMS = 8U;
    static constexpr uint32_t FP32_REG_ELEMS = BlockAttnResPrepareVF::FP32_REG_ELEMS;
    static constexpr uint32_t MAX_PIPE_ID = 8U;
    static constexpr uint32_t MTE3_MAX_PIPE_ID = 4U;
    static constexpr uint32_t STAT_BUFFER_NUM = 4U;
    static constexpr uint32_t SINGLE_STAT_BUFFER_NUM = 2U;
    static constexpr uint32_t N_FOUR_GROUP = 4U;
    static constexpr uint32_t N_PAIR_GROUP = 2U;
    static constexpr uint32_t S_PAIR_ROWS = 2U;
    static constexpr uint32_t FOUR_N_MIN_T = 64U;
    static constexpr event_t GROUP_STAT_EVENT_ID = static_cast<event_t>(2);
    static constexpr event_t GROUP_Q_EVENT_ID = static_cast<event_t>(0);
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

    __aicore__ inline void InitUbOffset()
    {
        qUbOffset_ = 0U;
        oUbOffset_ = qUbOffset_ + static_cast<uint64_t>(tilingData_->qBufferNum) * baseDAlign_ * INPUT_BYTES;
        statUbOffset_ = oUbOffset_ + static_cast<uint64_t>(tilingData_->oBufferNum) * baseDAlign_ * FP32_BYTES;
        normUbOffset_ = statUbOffset_ + static_cast<uint64_t>(STAT_BUFFER_NUM) * tilingData_->statUbElems * FP32_BYTES;
        vCacheUbOffset_ = normUbOffset_ + FP32_REG_ELEMS * FP32_BYTES;
        vUbOffset_ = vCacheUbOffset_;
    }

    template <typename T>
    __aicore__ inline static __ubuf__ T *GetUbAddr(uint64_t byteOffset)
    {
        return asc::te::make_mem_ptr<asc::te::location::ub, T>(byteOffset).get();
    }

    template <typename T>
    __aicore__ inline static auto MakeGmTensor(__gm__ T *address, int64_t rows, int64_t columns)
    {
        return asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::gm>(address),
            asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<T>>(rows, columns));
    }

    template <typename T, typename GmTensor>
    __aicore__ inline static void CopyGmToUb(uint64_t dstByteOffset, const GmTensor &srcTensor, int64_t rowIndex,
                                             int64_t columnIndex, int64_t validElements, int64_t ubPitch)
    {
        auto ubStorageTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, T>(dstByteOffset),
            asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<T>>(1, ubPitch));
        auto ubTensor = ubStorageTensor.slice(asc::te::make_coord(static_cast<int64_t>(0), static_cast<int64_t>(0)),
                                              asc::te::make_shape(static_cast<int64_t>(1), validElements));
        auto gmTensor = srcTensor.slice(asc::te::make_coord(rowIndex, columnIndex),
                                        asc::te::make_shape(static_cast<int64_t>(1), validElements));
        auto copyGmToUb = asc::te::make_copy(asc::te::copy_gm_to_ub{});
        asc::te::copy(copyGmToUb, ubTensor, gmTensor);
    }

    template <typename T, typename GmTensor>
    __aicore__ inline static void CopyUbToGm(GmTensor &dstTensor, int64_t rowIndex, int64_t columnIndex,
                                             uint64_t srcByteOffset, int64_t validElements, int64_t ubPitch)
    {
        auto gmTensor = dstTensor.slice(asc::te::make_coord(rowIndex, columnIndex),
                                        asc::te::make_shape(static_cast<int64_t>(1), validElements));
        auto ubStorageTensor = asc::te::make_tensor(
            asc::te::make_mem_ptr<asc::te::location::ub, T>(srcByteOffset),
            asc::te::make_frame_layout<asc::te::nd_ext_layout_ptn, asc::te::layout_trait_default<T>>(1, ubPitch));
        auto ubTensor = ubStorageTensor.slice(asc::te::make_coord(static_cast<int64_t>(0), static_cast<int64_t>(0)),
                                              asc::te::make_shape(static_cast<int64_t>(1), validElements));
        auto copyUbToGm = asc::te::make_copy(asc::te::copy_ub_to_gm{});
        asc::te::copy(copyUbToGm, gmTensor, ubTensor);
    }

    __aicore__ inline uint64_t ReadValidBlocks() const
    {
        auto validBlocksTensor = MakeGmTensor<uint64_t>(validBlocksAddr_, 1, 1);
        return validBlocksTensor[asc::te::make_coord(static_cast<int64_t>(0), static_cast<int64_t>(0))];
    }

    __aicore__ inline void PrefetchQ(uint64_t sIndex, uint32_t qBufferIndex)
    {
        const uint64_t byteOffset = qUbOffset_ + static_cast<uint64_t>(qBufferIndex) * baseDAlign_ * INPUT_BYTES;
        auto pseudoQueryTensor = MakeGmTensor<INPUT_T>(pseudoQueryAddr_, tilingData_->totalS, tilingData_->totalD);
        CopyGmToUb<INPUT_T>(byteOffset, pseudoQueryTensor, static_cast<int64_t>(sIndex), 0, tilingData_->totalD,
                            baseDAlign_);
    }

    __aicore__ inline void LoadVTile(uint64_t tIndex, uint32_t validN)
    {
        wait_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(MAX_PIPE_ID - 1U));
        auto blockResTensor = MakeGmTensor<INPUT_T>(
            blockResAddr_, static_cast<int64_t>(tilingData_->totalT) * tilingData_->totalN, tilingData_->totalD);
        for (uint32_t nIndex = 0; nIndex < validN; ++nIndex) {
            const uint64_t byteOffset = vCacheUbOffset_ + static_cast<uint64_t>(nIndex) * dAlign_ * INPUT_BYTES;
            const int64_t rowIndex = static_cast<int64_t>(tIndex * tilingData_->totalN + nIndex);
            CopyGmToUb<INPUT_T>(byteOffset, blockResTensor, rowIndex, 0, tilingData_->totalD, dAlign_);
        }
        WaitMte2ToVector(static_cast<event_t>(MAX_PIPE_ID - 1U));
    }

    __aicore__ inline __ubuf__ INPUT_T *CopyInV(uint64_t tIndex, uint32_t nIndex, uint32_t dOffset, uint32_t validD,
                                                uint64_t vLoop)
    {
        const uint32_t bufferIndex = static_cast<uint32_t>(vLoop % vBufferNum_);
        const uint64_t byteOffset = vUbOffset_ + static_cast<uint64_t>(bufferIndex) * baseDAlign_ * INPUT_BYTES;
        const int64_t rowIndex = static_cast<int64_t>(tIndex * tilingData_->totalN + nIndex);
        auto blockResTensor = MakeGmTensor<INPUT_T>(
            blockResAddr_, static_cast<int64_t>(tilingData_->totalT) * tilingData_->totalN, tilingData_->totalD);
        CopyGmToUb<INPUT_T>(byteOffset, blockResTensor, rowIndex, dOffset, validD, baseDAlign_);
        WaitMte2ToVector(static_cast<event_t>(bufferIndex));
        return GetUbAddr<INPUT_T>(byteOffset);
    }

    __aicore__ inline __ubuf__ INPUT_T *CopyVToCache(uint64_t tIndex, uint32_t nIndex, uint32_t dOffset,
                                                     uint32_t cacheOffset, uint32_t validD)
    {
        const uint64_t byteOffset = vCacheUbOffset_ + static_cast<uint64_t>(cacheOffset) * INPUT_BYTES;
        const int64_t rowIndex = static_cast<int64_t>(tIndex * tilingData_->totalN + nIndex);
        auto blockResTensor = MakeGmTensor<INPUT_T>(
            blockResAddr_, static_cast<int64_t>(tilingData_->totalT) * tilingData_->totalN, tilingData_->totalD);
        CopyGmToUb<INPUT_T>(byteOffset, blockResTensor, rowIndex, dOffset, validD, dAlign_);
        WaitMte2ToVector(static_cast<event_t>(0));
        return GetUbAddr<INPUT_T>(byteOffset);
    }

    __aicore__ inline void CopyOutStats(uint64_t sIndex, uint64_t tIndex, uint64_t statByteOffset, event_t eventId)
    {
        WaitVectorToMte3(eventId);
        auto logitMaxTensor = MakeGmTensor<float>(logitMaxAddr_, tilingData_->totalS, tilingData_->totalT);
        auto expSumTensor = MakeGmTensor<float>(expSumAddr_, tilingData_->totalS, tilingData_->totalT);
        CopyUbToGm<float>(logitMaxTensor, static_cast<int64_t>(sIndex), static_cast<int64_t>(tIndex),
                          statByteOffset + static_cast<uint64_t>(BlockAttnResPrepareVF::MAX_OFFSET) * FP32_BYTES, 1,
                          FP32_32B_ALIGN_ELEMS);
        CopyUbToGm<float>(expSumTensor, static_cast<int64_t>(sIndex), static_cast<int64_t>(tIndex),
                          statByteOffset + static_cast<uint64_t>(BlockAttnResPrepareVF::SUM_OFFSET) * FP32_BYTES, 1,
                          FP32_32B_ALIGN_ELEMS);
    }

    __aicore__ inline void CopyOutO(uint64_t sIndex, uint64_t tIndex, uint32_t dOffset, uint32_t validD, uint32_t dLoop)
    {
        const uint32_t bufferIndex = dLoop % tilingData_->oBufferNum;
        const uint64_t byteOffset = oUbOffset_ + static_cast<uint64_t>(bufferIndex) * baseDAlign_ * FP32_BYTES;
        const event_t eventId = static_cast<event_t>(bufferIndex);
        WaitVectorToMte3(eventId);
        const int64_t rowIndex = static_cast<int64_t>(sIndex * tilingData_->totalT + tIndex);
        auto numeratorTensor = MakeGmTensor<float>(
            numeratorAddr_, static_cast<int64_t>(tilingData_->totalS) * tilingData_->totalT, tilingData_->totalD);
        CopyUbToGm<float>(numeratorTensor, rowIndex, dOffset, byteOffset, validD, baseDAlign_);
    }

    __aicore__ inline void AccumulateStats(__ubuf__ INPUT_T *qAddr, __ubuf__ INPUT_T *vAddr,
                                           __ubuf__ float *statAddr,
                                           uint32_t dLoop, uint32_t validD, uint32_t nIndex)
    {
        if (dLoop == 0U) {
            BlockAttnResPrepareVF::AccumulateSquareDot<INPUT_T, true>(qAddr, vAddr, statAddr, validD, nIndex);
        } else {
            BlockAttnResPrepareVF::AccumulateSquareDot<INPUT_T, false>(qAddr, vAddr, statAddr, validD, nIndex);
        }
    }

    __aicore__ inline void AccumulateOutput(__ubuf__ INPUT_T *vAddr, __ubuf__ float *weightAddr,
                                            __ubuf__ float *outputAddr, uint32_t nIndex, uint32_t validN,
                                            uint32_t validD)
    {
        if (validN == 1U) {
            BlockAttnResPrepareVF::CopySingleBlock<INPUT_T>(vAddr, outputAddr, validD);
        } else if (nIndex == 0U) {
            BlockAttnResPrepareVF::WeightedAccumulate<INPUT_T, true>(vAddr, weightAddr, outputAddr, validD);
        } else {
            BlockAttnResPrepareVF::WeightedAccumulate<INPUT_T, false>(vAddr, weightAddr, outputAddr, validD);
        }
    }

    __aicore__ inline void FinalizeStats(__ubuf__ float *statAddr, uint32_t validN)
    {
        const float reciprocalD = 1.0F / static_cast<float>(tilingData_->totalD);
        if (validN == 1U) {
            BlockAttnResPrepareVF::FinalizeSingleBlock(statAddr, reciprocalD, tilingData_->eps);
        } else {
            BlockAttnResPrepareVF::FinalizeSoftmax(statAddr, validN, reciprocalD, tilingData_->eps);
        }
    }

    __aicore__ inline void ProcessEmptyWork(uint64_t workId)
    {
        const uint64_t tIndex = workId / tilingData_->totalS;
        const uint64_t sIndex = workId - tIndex * tilingData_->totalS;
        const uint32_t statBufferIndex = 0U;
        const event_t statEvent = static_cast<event_t>(tilingData_->oBufferNum + statBufferIndex);
        const uint64_t statByteOffset =
            statUbOffset_ + static_cast<uint64_t>(statBufferIndex) * tilingData_->statUbElems * FP32_BYTES;
        wait_flag(PIPE_MTE3, PIPE_V, statEvent);
        __ubuf__ float *statAddr = GetUbAddr<float>(statByteOffset);
        BlockAttnResPrepareVF::InitializeEmptyOnlineSoftmax(statAddr);
        CopyOutStats(sIndex, tIndex, statByteOffset, statEvent);
        set_flag(PIPE_MTE3, PIPE_V, statEvent);

        for (uint32_t dLoop = 0; dLoop < dTileNum_; ++dLoop) {
            const uint32_t dOffset = dLoop * tilingData_->baseD;
            const uint64_t remainingD = tilingData_->totalD - dOffset;
            const uint32_t validD =
                static_cast<uint32_t>(remainingD < tilingData_->baseD ? remainingD : tilingData_->baseD);
            const uint32_t outputBufferIndex = oLoop_ % tilingData_->oBufferNum;
            const uint64_t outputByteOffset =
                oUbOffset_ + static_cast<uint64_t>(outputBufferIndex) * baseDAlign_ * FP32_BYTES;
            __ubuf__ float *outputAddr = GetUbAddr<float>(outputByteOffset);
            const event_t outputEvent = static_cast<event_t>(outputBufferIndex);
            wait_flag(PIPE_MTE3, PIPE_V, outputEvent);
            BlockAttnResPrepareVF::FillZero(outputAddr, validD);

            CopyOutO(sIndex, tIndex, dOffset, validD, oLoop_);
            set_flag(PIPE_MTE3, PIPE_V, outputEvent);
            ++oLoop_;
        }
        ++statLoop_;
    }

    __aicore__ inline void ProcessT(uint64_t tIndex, uint32_t sStart, uint32_t sEnd, uint32_t validN)
    {
        LoadVTile(tIndex, validN);
        wait_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(qLoop_ % qBufferNum_));
        PrefetchQ(sStart, qLoop_ % qBufferNum_);
        for (uint32_t sIndex = sStart; sIndex < sEnd; ++sIndex) {
            const uint32_t qBufferIndex = qLoop_ % qBufferNum_;
            const event_t qEvent = static_cast<event_t>(qBufferIndex);
            WaitMte2ToVector(qEvent);
            if (sIndex + 1U < sEnd) {
                const uint32_t nextQBufferIndex = (qLoop_ + 1U) % qBufferNum_;
                wait_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(nextQBufferIndex));
                PrefetchQ(sIndex + 1U, nextQBufferIndex);
            }
            __ubuf__ INPUT_T *qAddr = GetUbAddr<INPUT_T>(
                qUbOffset_ + static_cast<uint64_t>(qBufferIndex) * baseDAlign_ * INPUT_BYTES);
            ProcessWork(tIndex, sIndex, sIndex == sStart, validN, qAddr, qEvent);
            ++qLoop_;
        }
        PipeBarrier<PIPE_V>();
        set_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(MAX_PIPE_ID - 1U));
    }

    __aicore__ inline void ProcessWork(uint64_t tIndex, uint64_t sIndex, bool firstS, uint32_t validN,
                                       __ubuf__ INPUT_T *qAddr, event_t qEvent)
    {
        const uint32_t statBufferIndex = statLoop_ % SINGLE_STAT_BUFFER_NUM;
        const event_t statEvent = static_cast<event_t>(tilingData_->oBufferNum + statBufferIndex);
        const uint64_t statByteOffset =
            statUbOffset_ + static_cast<uint64_t>(statBufferIndex) * tilingData_->statUbElems * FP32_BYTES;
        __ubuf__ float *statAddr = GetUbAddr<float>(statByteOffset);

        wait_flag(PIPE_MTE3, PIPE_V, statEvent);
        for (uint32_t nIndex = 0; nIndex < validN; ++nIndex) {
            __ubuf__ INPUT_T *vAddr = GetUbAddr<INPUT_T>(
                vCacheUbOffset_ + static_cast<uint64_t>(nIndex) * dAlign_ * INPUT_BYTES);
            if (firstS) {
                AccumulateStats(qAddr, vAddr, statAddr, 0U, tilingData_->totalD, nIndex);
            } else {
                BlockAttnResPrepareVF::AccumulateDotWithCachedNorm<INPUT_T>(
                    qAddr, vAddr, statAddr, GetUbAddr<float>(normUbOffset_), tilingData_->totalD, nIndex);
            }
        }
        if (firstS) {
            BlockAttnResPrepareVF::CacheSquareNorms(statAddr, GetUbAddr<float>(normUbOffset_), validN);
        }
        set_flag(PIPE_V, PIPE_MTE2, qEvent);

        FinalizeStats(statAddr, validN);
        CopyOutStats(sIndex, tIndex, statByteOffset, statEvent);
        set_flag(PIPE_MTE3, PIPE_V, statEvent);

        __ubuf__ float *weightAddr = statAddr + BlockAttnResPrepareVF::DOT_OFFSET;
        for (uint32_t dLoop = 0; dLoop < dTileNum_; ++dLoop) {
            const uint32_t dOffset = dLoop * tilingData_->baseD;
            const uint64_t remainingD = tilingData_->totalD - dOffset;
            const uint32_t validD =
                static_cast<uint32_t>(remainingD < tilingData_->baseD ? remainingD : tilingData_->baseD);
            const uint32_t outputBufferIndex = oLoop_ % tilingData_->oBufferNum;
            const uint64_t outputByteOffset =
                oUbOffset_ + static_cast<uint64_t>(outputBufferIndex) * baseDAlign_ * FP32_BYTES;
            __ubuf__ float *outputAddr = GetUbAddr<float>(outputByteOffset);
            const event_t outputEvent = static_cast<event_t>(outputBufferIndex);
            wait_flag(PIPE_MTE3, PIPE_V, outputEvent);
            for (uint32_t nIndex = 0; nIndex < validN; ++nIndex) {
                __ubuf__ INPUT_T *vAddr = GetUbAddr<INPUT_T>(
                    vCacheUbOffset_ + static_cast<uint64_t>(nIndex) * dAlign_ * INPUT_BYTES);
                AccumulateOutput(vAddr, weightAddr + nIndex, outputAddr, nIndex, validN, validD);
            }

            CopyOutO(sIndex, tIndex, dOffset, validD, oLoop_);
            set_flag(PIPE_MTE3, PIPE_V, outputEvent);
            ++oLoop_;
        }
        PipeBarrier<PIPE_V>();
        ++statLoop_;
    }

    __aicore__ inline void CopyGroupedStats(uint64_t tIndex, uint32_t sBase)
    {
        WaitVectorToMte3(GROUP_STAT_EVENT_ID);
        auto logitMaxTensor = MakeGmTensor<float>(logitMaxAddr_, tilingData_->totalS, tilingData_->totalT);
        auto expSumTensor = MakeGmTensor<float>(expSumAddr_, tilingData_->totalS, tilingData_->totalT);
        for (uint32_t row = 0; row < STAT_BUFFER_NUM; ++row) {
            const uint64_t statOffset = statUbOffset_ + static_cast<uint64_t>(row) *
                                        tilingData_->statUbElems * FP32_BYTES;
            CopyUbToGm<float>(logitMaxTensor, sBase + row, tIndex,
                statOffset + static_cast<uint64_t>(BlockAttnResPrepareVF::MAX_OFFSET) * FP32_BYTES,
                1, FP32_32B_ALIGN_ELEMS);
            CopyUbToGm<float>(expSumTensor, sBase + row, tIndex,
                statOffset + static_cast<uint64_t>(BlockAttnResPrepareVF::SUM_OFFSET) * FP32_BYTES,
                1, FP32_32B_ALIGN_ELEMS);
        }
        set_flag(PIPE_MTE3, PIPE_V, GROUP_STAT_EVENT_ID);
    }

    __aicore__ inline void ProcessGroupedOutputPair(uint64_t tIndex, uint32_t sIndex,
                                                    uint32_t row, uint32_t validN)
    {
        __ubuf__ float *weight0Addr = GetUbAddr<float>(statUbOffset_ +
            static_cast<uint64_t>(row) * tilingData_->statUbElems * FP32_BYTES) +
            BlockAttnResPrepareVF::DOT_OFFSET;
        __ubuf__ float *weight1Addr = GetUbAddr<float>(statUbOffset_ +
            static_cast<uint64_t>(row + 1U) * tilingData_->statUbElems * FP32_BYTES) +
            BlockAttnResPrepareVF::DOT_OFFSET;
        for (uint32_t dLoop = 0; dLoop < dTileNum_; ++dLoop) {
            const uint32_t dOffset = dLoop * tilingData_->baseD;
            const uint32_t validD = tilingData_->totalD - dOffset < tilingData_->baseD ?
                                    tilingData_->totalD - dOffset : tilingData_->baseD;
            const uint32_t outputBuffer0 = oLoop_ % tilingData_->oBufferNum;
            const uint32_t outputBuffer1 = (oLoop_ + 1U) % tilingData_->oBufferNum;
            const event_t outputEvent0 = static_cast<event_t>(outputBuffer0);
            const event_t outputEvent1 = static_cast<event_t>(outputBuffer1);
            wait_flag(PIPE_MTE3, PIPE_V, outputEvent0);
            wait_flag(PIPE_MTE3, PIPE_V, outputEvent1);
            __ubuf__ float *output0Addr = GetUbAddr<float>(oUbOffset_ +
                static_cast<uint64_t>(outputBuffer0) * baseDAlign_ * FP32_BYTES);
            __ubuf__ float *output1Addr = GetUbAddr<float>(oUbOffset_ +
                static_cast<uint64_t>(outputBuffer1) * baseDAlign_ * FP32_BYTES);
            uint32_t nIndex = 0U;
            if (tilingData_->totalT >= FOUR_N_MIN_T) {
                for (; nIndex + N_FOUR_GROUP - 1U < validN; nIndex += N_FOUR_GROUP) {
                    __ubuf__ INPUT_T *vAddr = GetUbAddr<INPUT_T>(
                        vCacheUbOffset_ + static_cast<uint64_t>(nIndex) * dAlign_ * INPUT_BYTES);
                    if (nIndex == 0U) {
                        BlockAttnResPrepareVF::WeightedAccumulateTwoRowsFourN<INPUT_T, true>(
                            vAddr + dOffset, dAlign_, weight0Addr + nIndex, weight1Addr + nIndex,
                            output0Addr, output1Addr, validD);
                    } else {
                        BlockAttnResPrepareVF::WeightedAccumulateTwoRowsFourN<INPUT_T, false>(
                            vAddr + dOffset, dAlign_, weight0Addr + nIndex, weight1Addr + nIndex,
                            output0Addr, output1Addr, validD);
                    }
                }
            }
            for (; nIndex + N_PAIR_GROUP - 1U < validN; nIndex += N_PAIR_GROUP) {
                __ubuf__ INPUT_T *v0Addr = GetUbAddr<INPUT_T>(
                    vCacheUbOffset_ + static_cast<uint64_t>(nIndex) * dAlign_ * INPUT_BYTES);
                __ubuf__ INPUT_T *v1Addr = GetUbAddr<INPUT_T>(
                    vCacheUbOffset_ + static_cast<uint64_t>(nIndex + 1U) * dAlign_ * INPUT_BYTES);
                if (nIndex == 0U) {
                    BlockAttnResPrepareVF::WeightedAccumulateTwoRowsTwoN<INPUT_T, true>(
                        v0Addr + dOffset, v1Addr + dOffset, weight0Addr + nIndex, weight1Addr + nIndex,
                        output0Addr, output1Addr, validD);
                } else {
                    BlockAttnResPrepareVF::WeightedAccumulateTwoRowsTwoN<INPUT_T, false>(
                        v0Addr + dOffset, v1Addr + dOffset, weight0Addr + nIndex, weight1Addr + nIndex,
                        output0Addr, output1Addr, validD);
                }
            }
            if (nIndex < validN) {
                __ubuf__ INPUT_T *vAddr = GetUbAddr<INPUT_T>(
                    vCacheUbOffset_ + static_cast<uint64_t>(nIndex) * dAlign_ * INPUT_BYTES);
                if (validN == 1U) {
                    BlockAttnResPrepareVF::WeightedAccumulateTwoRows<INPUT_T, true, true>(
                        vAddr + dOffset, weight0Addr, weight1Addr, output0Addr, output1Addr, validD);
                } else {
                    BlockAttnResPrepareVF::WeightedAccumulateTwoRows<INPUT_T, false, false>(
                        vAddr + dOffset, weight0Addr + nIndex, weight1Addr + nIndex,
                        output0Addr, output1Addr, validD);
                }
            }
            CopyOutO(sIndex, tIndex, dOffset, validD, oLoop_);
            CopyOutO(sIndex + 1U, tIndex, dOffset, validD, oLoop_ + 1U);
            set_flag(PIPE_MTE3, PIPE_V, outputEvent0);
            set_flag(PIPE_MTE3, PIPE_V, outputEvent1);
            ++oLoop_;
            ++oLoop_;
        }
    }

    __aicore__ inline void ProcessTGrouped(uint64_t tIndex, uint32_t sStart, uint32_t sEnd, uint32_t validN)
    {
        LoadVTile(tIndex, validN);
        for (uint32_t sBase = sStart; sBase < sEnd; sBase += STAT_BUFFER_NUM) {
            wait_flag(PIPE_MTE3, PIPE_V, GROUP_STAT_EVENT_ID);
            wait_flag(PIPE_V, PIPE_MTE2, GROUP_Q_EVENT_ID);
            for (uint32_t row = 0; row < STAT_BUFFER_NUM; ++row) {
                PrefetchQ(sBase + row, row);
            }
            WaitMte2ToVector(GROUP_Q_EVENT_ID);
            __ubuf__ INPUT_T *qAddr = GetUbAddr<INPUT_T>(qUbOffset_);
            uint32_t nIndex = 0U;
            for (; nIndex + N_PAIR_GROUP - 1U < validN; nIndex += N_PAIR_GROUP) {
                __ubuf__ INPUT_T *v0Addr = GetUbAddr<INPUT_T>(
                    vCacheUbOffset_ + static_cast<uint64_t>(nIndex) * dAlign_ * INPUT_BYTES);
                __ubuf__ INPUT_T *v1Addr = GetUbAddr<INPUT_T>(
                    vCacheUbOffset_ + static_cast<uint64_t>(nIndex + 1U) * dAlign_ * INPUT_BYTES);
                if (sBase == sStart) {
                    BlockAttnResPrepareVF::AccumulateFourStatsTwoN<INPUT_T, true>(
                        qAddr, v0Addr, v1Addr, GetUbAddr<float>(statUbOffset_), GetUbAddr<float>(normUbOffset_),
                        tilingData_->totalD, nIndex, baseDAlign_, tilingData_->statUbElems);
                } else {
                    BlockAttnResPrepareVF::AccumulateFourStatsTwoN<INPUT_T, false>(
                        qAddr, v0Addr, v1Addr, GetUbAddr<float>(statUbOffset_), GetUbAddr<float>(normUbOffset_),
                        tilingData_->totalD, nIndex, baseDAlign_, tilingData_->statUbElems);
                }
            }
            if (nIndex < validN) {
                __ubuf__ INPUT_T *vAddr = GetUbAddr<INPUT_T>(
                    vCacheUbOffset_ + static_cast<uint64_t>(nIndex) * dAlign_ * INPUT_BYTES);
                if (sBase == sStart) {
                    BlockAttnResPrepareVF::AccumulateFourStats<INPUT_T, true>(
                        qAddr, vAddr, GetUbAddr<float>(statUbOffset_), GetUbAddr<float>(normUbOffset_),
                        tilingData_->totalD, nIndex, baseDAlign_, tilingData_->statUbElems);
                } else {
                    BlockAttnResPrepareVF::AccumulateFourStats<INPUT_T, false>(
                        qAddr, vAddr, GetUbAddr<float>(statUbOffset_), GetUbAddr<float>(normUbOffset_),
                        tilingData_->totalD, nIndex, baseDAlign_, tilingData_->statUbElems);
                }
            }
            set_flag(PIPE_V, PIPE_MTE2, GROUP_Q_EVENT_ID);
            for (uint32_t row = 0; row < STAT_BUFFER_NUM; ++row) {
                FinalizeStats(GetUbAddr<float>(statUbOffset_ +
                    static_cast<uint64_t>(row) * tilingData_->statUbElems * FP32_BYTES), validN);
            }
            CopyGroupedStats(tIndex, sBase);
            for (uint32_t row = 0; row < STAT_BUFFER_NUM; row += S_PAIR_ROWS) {
                ProcessGroupedOutputPair(tIndex, sBase + row, row, validN);
            }
        }
        PipeBarrier<PIPE_V>();
        set_flag(PIPE_V, PIPE_MTE2, static_cast<event_t>(MAX_PIPE_ID - 1U));
    }

    const optiling::BlockAttnResPrepareTilingData *__restrict tilingData_ = nullptr;
    bool initialized_ = false;

    uint64_t qUbOffset_ = 0U;
    uint64_t vUbOffset_ = 0U;
    uint64_t oUbOffset_ = 0U;
    uint64_t statUbOffset_ = 0U;
    uint64_t normUbOffset_ = 0U;
    uint64_t vCacheUbOffset_ = 0U;
    uint32_t baseDAlign_ = 0U;
    uint32_t dAlign_ = 0U;
    uint32_t dTileNum_ = 0U;
    uint32_t qLoop_ = 0;
    uint32_t vLoop_ = 0;
    uint32_t oLoop_ = 0;
    uint32_t statLoop_ = 0;
    uint64_t validN_ = 0U;
    uint8_t qBufferNum_ = 1;
    uint8_t vBufferNum_ = 1;

    __gm__ INPUT_T *blockResAddr_ = nullptr;
    __gm__ uint64_t *validBlocksAddr_ = nullptr;
    __gm__ INPUT_T *pseudoQueryAddr_ = nullptr;
    __gm__ float *numeratorAddr_ = nullptr;
    __gm__ float *logitMaxAddr_ = nullptr;
    __gm__ float *expSumAddr_ = nullptr;
};

} // namespace BlockAttnResPrepare

#endif // BLOCK_ATTN_RES_PREPARE_VECTOR_S_TILE_H
