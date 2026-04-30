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
 * \file ai_infra_kv_rms_norm_rope_cache_b16_pa_c8.h
 * \brief
 */
#ifndef _AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_PA_BNSD_C8_H_
#define _AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_PA_BNSD_C8_H_
#include "ai_infra_kv_rms_norm_rope_cache_comm.h"

namespace AiInfraKvRmsNormRopeCache {
using namespace AscendC;
constexpr int64_t PA_C8_NZ_NO_QUANT = 1;
constexpr int64_t PA_C8_BLK_BNSD_NO_QUANT = 2;
constexpr int64_t PA_C8_BNSD_NO_QUANT = 3;
constexpr int64_t D1_C8_SIZE4_PA = 4;
constexpr int64_t D1_C8_SIZE8_PA = 8;
constexpr int64_t D1_C8_SIZE12_PA = 12;
constexpr int64_t D1_C8_SIZE32_PA = 32;
constexpr int64_t D0_C8_SIZE16_PA = 16;
constexpr float DYNAMIC_QUANT_EPSINON = 1e-10;

template <bool isPagedAttention, typename KV_DTYPE, int64_t scatterType>
class KernelAiInfraKvRmsNormRopeCacheB16PAC8 : public KernelAiInfraKvRmsNormRopeCacheCutBC8<isPagedAttention, KV_DTYPE, KV_DTYPE, KV_DTYPE>
{
    // Simple scatter without quant: probable incomplete implemetation?
public:
    __aicore__ inline KernelAiInfraKvRmsNormRopeCacheB16PAC8(TPipe* pipe, const AiInfraKvRmsNormRopeCacheTilingData* tiling)
        : KernelAiInfraKvRmsNormRopeCacheCutBC8<isPagedAttention, KV_DTYPE, KV_DTYPE, KV_DTYPE>(pipe, tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR kv, GM_ADDR gamma, GM_ADDR cos, GM_ADDR sin, GM_ADDR index, GM_ADDR k_cache, GM_ADDR v_cache,
        GM_ADDR optional_k_rope, GM_ADDR optional_c_kv, GM_ADDR optional_v)
    {
        methodMode = optional_v == nullptr ? 0 : 1;
        this->InitSharedData(methodMode);                 
        int64_t currentBlockFactor = this->tilingData_->blockFactor;
        if (GetBlockIdx() == (this->tilingData_->numBlocks - 1)) {
            currentBlockFactor = this->tilingData_->batchSize * this->tilingData_->seqLength * this->tilingData_->numHead -
                                 (this->tilingData_->numBlocks - 1) * this->tilingData_->blockFactor;
        }

        this->ubFactor = this->tilingData_->ubFactor;
        this->ubLoop = currentBlockFactor / this->ubFactor;
        this->ubTail = currentBlockFactor - this->ubLoop * this->ubFactor;

        // init global memory
        if (methodMode == METHOD_V1) {
            this->kvGm.SetGlobalBuffer(
                (__gm__ KV_DTYPE*)kv + GetBlockIdx() * this->tilingData_->blockFactor * (this->RMS_NORM_LENGTH + this->ROPE_LENGTH));
        } else {
            this->kvGm.SetGlobalBuffer((__gm__ KV_DTYPE*)kv + GetBlockIdx() * this->tilingData_->blockFactor * this->RMS_NORM_LENGTH);
            this->vGm.SetGlobalBuffer((__gm__ KV_DTYPE*)optional_v + GetBlockIdx() * this->tilingData_->blockFactor * this->V_LENGTH);
        }
        this->gammaGm.SetGlobalBuffer((__gm__ KV_DTYPE*)gamma);
        this->cosGm.SetGlobalBuffer((__gm__ KV_DTYPE*)cos + GetBlockIdx() * this->tilingData_->blockFactor * this->ROPE_LENGTH);
        this->sinGm.SetGlobalBuffer((__gm__ KV_DTYPE*)sin + GetBlockIdx() * this->tilingData_->blockFactor * this->ROPE_LENGTH);
        this->indexGm.SetGlobalBuffer((__gm__ int64_t*)index);
        // this->kCacheGm.SetGlobalBuffer((__gm__ KV_DTYPE*)k_cache);
        this->vCacheGmInt8.SetGlobalBuffer((__gm__ int8_t*)v_cache); // int8
        if (this->tilingData_->isOutputKv) {
            this->kCacheGmNd.SetGlobalBuffer((__gm__ KV_DTYPE*)optional_k_rope);
            this->vCacheGmNd.SetGlobalBuffer((__gm__ KV_DTYPE*)optional_c_kv);
        }

        // init new tiling fields
        ropeMode = this->tilingData_->rotaryMode;
        kStride = this->tilingData_->kStride;
        vStride = this->tilingData_->vStride;
        cacheLayOut = this->tilingData_->cacheLayOut;

        // init pipe
        this->pipe_->InitBuffer(this->inQueueGamma, 2, this->RMS_NORM_LENGTH * sizeof(float));                 // 2*512*4/1024=4
        this->pipe_->InitBuffer(this->inQueueX, 2, this->ubFactor * this->RMS_NORM_LENGTH * sizeof(KV_DTYPE)); // 2*16*512*2/1024=32
        this->pipe_->InitBuffer(this->outQueue, 2, this->ubFactor * this->RMS_NORM_LENGTH * sizeof(KV_DTYPE)); // 2*16*512*2/1024=32
        this->pipe_->InitBuffer(this->wsBuffer, 3 * this->ubFactor * this->RMS_NORM_LENGTH * sizeof(float));   // 3*16*512*4/1024=96
        if (methodMode == METHOD_V2) {
            this->pipe_->InitBuffer(this->inQueueV, 2, this->ubFactor * this->V_LENGTH * sizeof(KV_DTYPE)); // 2*40*128*2/1024=20
        } else if (methodMode == METHOD_V1){
            this->pipe_->InitBuffer(this->outQueueV, 2, this->RMS_NORM_LENGTH * sizeof(KV_DTYPE));   // 2*512*2/1024= 2
        }
        this->pertileSize = this->tilingData_->pertileSize;
        this->pertileSizeAlign = Align(this->pertileSize, NUM_EIGHT);
        this->qTileNum = this->RMS_NORM_LENGTH / this->pertileSize;
        totalHeadSizeInt8 = this->RMS_NORM_LENGTH + this->qTileNum * 4 + this->ROPE_LENGTH * 2;
        this->qTileAlign = Align(this->qTileNum, NUM_EIGHT);
        this->pipe_->InitBuffer(this->tileQScale, this->ubFactor * this->qTileAlign * sizeof(float));   // 16*32/1024=0.5

    }

    __aicore__ inline void Process()
    {
        if (methodMode == METHOD_V1) {
            ProcessV1();
        }
    }  
    
    __aicore__ inline void ProcessV1()
    {
        // bool hasLargeStride = stride > this->tilingData_->blockSize * this->RMS_NORM_LENGTH;
        int64_t isSFACountiguous = vStride > this->tilingData_->blockSize * totalHeadSizeInt8 ? 0 : 1;
        if (ropeMode == INTERLEAVE_MODE && isSFACountiguous) {
            ProcessV1Impl<INTERLEAVE_MODE, SFA_COUNTIGUOUS>();
        } else if (ropeMode == INTERLEAVE_MODE && !isSFACountiguous){
            ProcessV1Impl<INTERLEAVE_MODE, SFA_NONCOUNTIGUOUS>();
        } else if (ropeMode == HALF_MODE && isSFACountiguous) {
            ProcessV1Impl<HALF_MODE, SFA_COUNTIGUOUS>();    
        } else if (ropeMode == HALF_MODE && !isSFACountiguous) {
            ProcessV1Impl<HALF_MODE, SFA_NONCOUNTIGUOUS>();    
        }
    }

    template <int64_t rpMode, int64_t isSFACountiguous>
    __aicore__ inline void ProcessV1Impl()
    {
        DataCopyPadExtParams<KV_DTYPE> padParams{false, 0, 0, 0};
        // DataCopyPadExtParams<int64_t> padParamsInt64{false, 0, 0, 0};
        DataCopyExtParams copyParamsContinguous;
        copyParamsContinguous.blockCount = 1;
        // CopyIn gamma: [this->RMS_NORM_LENGTH]
        LocalTensor<float> gammaLocalFp32 = this->inQueueGamma.template AllocTensor<float>();
        LocalTensor<KV_DTYPE> gammaLocal = gammaLocalFp32.template ReinterpretCast<KV_DTYPE>()[this->RMS_NORM_LENGTH];
        copyParamsContinguous.blockLen = this->RMS_NORM_LENGTH * sizeof(KV_DTYPE);
        DataCopyPad(gammaLocal, this->gammaGm, copyParamsContinguous, padParams);
        this->inQueueGamma.EnQue(gammaLocalFp32);
        gammaLocalFp32 = this->inQueueGamma.template DeQue<float>();
        gammaLocal = gammaLocalFp32.template ReinterpretCast<KV_DTYPE>()[this->RMS_NORM_LENGTH];
        Cast(gammaLocalFp32, gammaLocal, RoundMode::CAST_NONE, this->RMS_NORM_LENGTH);

        LocalTensor<float> workspaceBuffer = this->wsBuffer.template Get<float>();
        for (int64_t loopIdx = 0; loopIdx < this->ubLoop; ++loopIdx) {
            int64_t kvGlobalMemoryOffset = loopIdx * this->ubFactor * (this->RMS_NORM_LENGTH + this->ROPE_LENGTH);
            int64_t freqGlobalMemoryOffset = loopIdx * this->ubFactor * this->ROPE_LENGTH;
            // int64_t indexGlobalMemoryOffset = loopIdx * this->ubFactor;
            int64_t startIdx = GetBlockIdx() * this->tilingData_->blockFactor + loopIdx * this->ubFactor;

            LocalTensor<KV_DTYPE> ropeLocal = this->inQueueX.template AllocTensor<KV_DTYPE>();
            LocalTensor<KV_DTYPE> cosLocal = ropeLocal[this->ubFactor * this->ROPE_LENGTH];
            LocalTensor<KV_DTYPE> sinLocal = cosLocal[this->ubFactor * this->ROPE_LENGTH];
            // CopyIn x/cos/sin [this->ubFactor, this->ROPE_LENGTH]
            DataCopyExtParams copyParams{/* blockCount */ static_cast<uint16_t>(this->ubFactor),
                                         /* blockLen (Byte) */ static_cast<uint32_t>(this->ROPE_LENGTH * sizeof(KV_DTYPE)),
                                         /* srcStride */ static_cast<uint32_t>(this->RMS_NORM_LENGTH * sizeof(KV_DTYPE)),
                                         /* dstStride */ 0,
                                         /* rsv */ 0};
            DataCopyPad(ropeLocal, this->kvGm[kvGlobalMemoryOffset + this->RMS_NORM_LENGTH], copyParams, padParams);
            copyParamsContinguous.blockLen = this->ubFactor * this->ROPE_LENGTH * sizeof(KV_DTYPE);
            DataCopyPad(cosLocal, this->cosGm[freqGlobalMemoryOffset], copyParamsContinguous, padParams);
            DataCopyPad(sinLocal, this->sinGm[freqGlobalMemoryOffset], copyParamsContinguous, padParams);
            this->inQueueX.EnQue(ropeLocal);
            ropeLocal = this->inQueueX.template DeQue<KV_DTYPE>();
            cosLocal = ropeLocal[this->ubFactor * this->ROPE_LENGTH];
            sinLocal = cosLocal[this->ubFactor * this->ROPE_LENGTH];

            // Calc: RoPE
            LocalTensor<KV_DTYPE> outLocal = this->outQueue.template AllocTensor<KV_DTYPE>();
            RoPE<KV_DTYPE, true, rpMode>(outLocal, ropeLocal, cosLocal, sinLocal, workspaceBuffer, this->ubFactor, this->ROPE_LENGTH);
            this->inQueueX.FreeTensor(ropeLocal);
            this->outQueue.EnQue(outLocal);
            outLocal = this->outQueue.template DeQue<KV_DTYPE>();


            DoScatterRopeCache<KV_DTYPE, isSFACountiguous>(
                this->vCacheGmInt8, this->kCacheGmNd, outLocal, startIdx, this->ubFactor, this->ROPE_LENGTH, totalHeadSizeInt8);

            this->outQueue.FreeTensor(outLocal);

            // CopyIn x: [this->ubFactor, RmsLength]
            LocalTensor<KV_DTYPE> xLocal = this->inQueueX.template AllocTensor<KV_DTYPE>();
            copyParams.blockCount = static_cast<uint16_t>(this->ubFactor);
            copyParams.blockLen = this->RMS_NORM_LENGTH * sizeof(KV_DTYPE);
            copyParams.srcStride = this->ROPE_LENGTH * sizeof(KV_DTYPE);
            copyParams.dstStride = 0;
            DataCopyPad(xLocal, this->kvGm[kvGlobalMemoryOffset], copyParams, padParams);
            this->inQueueX.EnQue(xLocal);
            xLocal = this->inQueueX.template DeQue<KV_DTYPE>();

            // Calc: RmsNorm
            outLocal = this->outQueue.template AllocTensor<KV_DTYPE>();
            RmsNorm<KV_DTYPE>(outLocal, xLocal, gammaLocalFp32, workspaceBuffer, this->ubFactor, this->RMS_NORM_LENGTH);
            this->inQueueX.FreeTensor(xLocal);
            this->outQueue.EnQue(outLocal);
            outLocal = this->outQueue.template DeQue<KV_DTYPE>();
            //isOutputKv
            PipeBarrier<PIPE_ALL>();
            if (this->tilingData_->isOutputKv) {
                DataCopyExtParams copyParamsNd{1, static_cast<uint32_t>(this->ubFactor * this->RMS_NORM_LENGTH * sizeof(KV_DTYPE)), 0, 0, 0};
                DataCopyPad(this->vCacheGmNd[startIdx * this->RMS_NORM_LENGTH], outLocal, copyParamsNd);
            }

            c8QuantCompute<isSFACountiguous>(this->ubFactor, this->RMS_NORM_LENGTH, startIdx, totalHeadSizeInt8);

            this->outQueue.FreeTensor(outLocal);
        }
        if (this->ubTail > 0) {
            int64_t kvGlobalMemoryOffset = this->ubLoop * this->ubFactor * (this->RMS_NORM_LENGTH + this->ROPE_LENGTH);
            int64_t freqGlobalMemoryOffset = this->ubLoop * this->ubFactor * this->ROPE_LENGTH;
            // int64_t indexGlobalMemoryOffset = this->ubLoop * this->ubFactor;
            int64_t startIdx = GetBlockIdx() * this->tilingData_->blockFactor + this->ubLoop * this->ubFactor;

            LocalTensor<KV_DTYPE> ropeLocal = this->inQueueX.template AllocTensor<KV_DTYPE>();
            LocalTensor<KV_DTYPE> cosLocal = ropeLocal[this->ubTail * this->ROPE_LENGTH];
            LocalTensor<KV_DTYPE> sinLocal = cosLocal[this->ubTail * this->ROPE_LENGTH];
            // CopyIn x/cos/sin [this->ubTail, this->ROPE_LENGTH]
            DataCopyExtParams copyParams{/* blockCount */ static_cast<uint16_t>(this->ubTail),
                                         /* blockLen (Byte) */ static_cast<uint32_t>(this->ROPE_LENGTH * sizeof(KV_DTYPE)),
                                         /* srcStride */ static_cast<uint32_t>(this->RMS_NORM_LENGTH * sizeof(KV_DTYPE)),
                                         /* dstStride */ 0,
                                         /* rsv */ 0};
            DataCopyPad(ropeLocal, this->kvGm[kvGlobalMemoryOffset + this->RMS_NORM_LENGTH], copyParams, padParams);
            copyParamsContinguous.blockLen = this->ubTail * this->ROPE_LENGTH * sizeof(KV_DTYPE);
            DataCopyPad(cosLocal, this->cosGm[freqGlobalMemoryOffset], copyParamsContinguous, padParams);
            DataCopyPad(sinLocal, this->sinGm[freqGlobalMemoryOffset], copyParamsContinguous, padParams);
            this->inQueueX.EnQue(ropeLocal);
            ropeLocal = this->inQueueX.template DeQue<KV_DTYPE>();
            cosLocal = ropeLocal[this->ubTail * this->ROPE_LENGTH];
            sinLocal = cosLocal[this->ubTail * this->ROPE_LENGTH];

            // Calc: RoPE
            LocalTensor<KV_DTYPE> outLocal = this->outQueue.template AllocTensor<KV_DTYPE>();
            RoPE<KV_DTYPE, true, rpMode>(outLocal, ropeLocal, cosLocal, sinLocal, workspaceBuffer, this->ubTail, this->ROPE_LENGTH);
            this->inQueueX.FreeTensor(ropeLocal);
            this->outQueue.EnQue(outLocal);
            outLocal = this->outQueue.template DeQue<KV_DTYPE>();

            // Scatter Update kCache (or merged into vCache when cacheLayOut == SFA_NO_QUANT)
            DoScatterRopeCache<KV_DTYPE, isSFACountiguous>(
                this->vCacheGmInt8, this->kCacheGmNd, outLocal, startIdx, this->ubTail, this->ROPE_LENGTH, totalHeadSizeInt8);
            this->outQueue.FreeTensor(outLocal);

            // CopyIn x: [this->ubTail, RmsLength]
            LocalTensor<KV_DTYPE> xLocal = this->inQueueX.template AllocTensor<KV_DTYPE>();
            copyParams.blockCount = static_cast<uint16_t>(this->ubTail);
            copyParams.blockLen = this->RMS_NORM_LENGTH * sizeof(KV_DTYPE);
            copyParams.srcStride = this->ROPE_LENGTH * sizeof(KV_DTYPE);
            copyParams.dstStride = 0;
            DataCopyPad(xLocal, this->kvGm[kvGlobalMemoryOffset], copyParams, padParams);
            this->inQueueX.EnQue(xLocal);
            xLocal = this->inQueueX.template DeQue<KV_DTYPE>();

            // Calc: RmsNorm
            outLocal = this->outQueue.template AllocTensor<KV_DTYPE>();
            RmsNorm<KV_DTYPE>(outLocal, xLocal, gammaLocalFp32, workspaceBuffer, this->ubTail, this->RMS_NORM_LENGTH);
            this->inQueueX.FreeTensor(xLocal);
            this->outQueue.EnQue(outLocal);
            outLocal = this->outQueue.template DeQue<KV_DTYPE>();
            //isOutputKv
            PipeBarrier<PIPE_ALL>();
            if (this->tilingData_->isOutputKv) {
                DataCopyExtParams copyParamsNd{1, static_cast<uint32_t>(this->ubTail * this->RMS_NORM_LENGTH * sizeof(KV_DTYPE)), 0, 0, 0};
                DataCopyPad(this->vCacheGmNd[startIdx * this->RMS_NORM_LENGTH], outLocal, copyParamsNd);
            }
            // Scatter Update vCache (or merged into vCache first part when cacheLayOut == SFA_NO_QUANT)
            // DoScatterC8<KV_DTYPE, true>(
            //     this->kCacheGm, this->kCacheGmNd, outLocal, startIdx, this->ubTail, this->RMS_NORM_LENGTH, 
            //     totalHeadSizeInt8, 0);
            c8QuantCompute<isSFACountiguous>(this->ubTail, this->RMS_NORM_LENGTH, startIdx, totalHeadSizeInt8);

            this->outQueue.FreeTensor(outLocal);
        }
        this->inQueueGamma.FreeTensor(gammaLocalFp32);
    }

    template <typename T, bool isElementWise = true, int64_t rpMode = INTERLEAVE_MODE>
    __aicore__ inline void RoPE(
        const LocalTensor<T>& outLocal, const LocalTensor<T>& xLocal, const LocalTensor<T>& cosLocal,
        const LocalTensor<T>& sinLocal, const LocalTensor<float>& wsLocal, int64_t rows, int64_t headSize)
    {
        constexpr static int64_t NUM_ONE = 1;
        constexpr static int64_t NUM_TWO = 2;
        constexpr static int64_t NUM_FOUR = 4;
        constexpr static int64_t NUM_EIGHT = 8;
        if constexpr (isElementWise) {
            /**
             * cosLocalFp32 : [rows * headSize * 0, rows * headSize * 1]
             * sinLocalFp32 : [rows * headSize * 1, rows * headSize * 2]
             * y0           : [rows * headSize * 2, rows * headSize * 3]
             * y1           : [rows * headSize * 3, rows * headSize * 4]
             * realLocalFp32: [rows * headSize * 4, rows * headSize * 5]
             * imagLocalFp32: [rows * headSize * 5, rows * headSize * 6]
             * realLocal    : [rows * headSize * 6, rows * headSize * 7]
             * imagLocal    : [rows * headSize * 7, rows * headSize * 8]
             */
            int64_t cosLocalFp32Offset = rows * headSize * 0;
            int64_t sinLocalFp32Offset = rows * headSize * 1;
            int64_t y0Offset = rows * headSize * 2;
            int64_t y1Offset = rows * headSize * 3;
            int64_t realLocalFp32Offset = rows * headSize * 4;
            int64_t imagLocalFp32Offset = rows * headSize * 5;
            int64_t realLocalOffset = rows * headSize * 6;
            int64_t imagLocalOffset = rows * headSize * 7;

            LocalTensor<float> cosLocalFp32 = wsLocal[cosLocalFp32Offset];
            LocalTensor<float> sinLocalFp32 = wsLocal[sinLocalFp32Offset];
            LocalTensor<float> y0 = wsLocal[y0Offset];
            LocalTensor<float> y1 = wsLocal[y1Offset];
            LocalTensor<float> realLocalFp32 = wsLocal[realLocalFp32Offset];
            LocalTensor<float> imagLocalFp32 = wsLocal[imagLocalFp32Offset];
            LocalTensor<T> realLocal = wsLocal[realLocalOffset].template ReinterpretCast<T>();
            LocalTensor<T> imagLocal = wsLocal[imagLocalOffset].template ReinterpretCast<T>();

            // step #1: cast cosLocal and sinLocal to fp32
            Cast(cosLocalFp32, cosLocal, RoundMode::CAST_NONE, rows * headSize);
            Cast(sinLocalFp32, sinLocal, RoundMode::CAST_NONE, rows * headSize);
            PipeBarrier<PIPE_V>();

            // step #2: Extract real and imag
            if constexpr (rpMode == INTERLEAVE_MODE) {
                // Interleave-half: GatherMask to separate even/odd elements
                uint64_t rsvdCnt = 0;
                GatherMask(realLocal, xLocal, NUM_ONE, true, rows * headSize, {1, 1, NUM_EIGHT, 0}, rsvdCnt);
                GatherMask(imagLocal, xLocal, NUM_TWO, true, rows * headSize, {1, 1, NUM_EIGHT, 0}, rsvdCnt);
            } else {
                // Half-half: DataCopy to extract first/second half of each token
                int64_t halfSize = headSize >> 1;
                struct DataCopyParams copyParamsHalf;
                copyParamsHalf.blockCount = static_cast<uint16_t>(rows);
                copyParamsHalf.blockLen = static_cast<uint16_t>(halfSize * sizeof(T) / BLOCK_SIZE);
                copyParamsHalf.srcStride = static_cast<uint16_t>(halfSize * sizeof(T) / BLOCK_SIZE);
                copyParamsHalf.dstStride = 0;
                DataCopy(realLocal, xLocal, copyParamsHalf);
                DataCopy(imagLocal, xLocal[halfSize], copyParamsHalf);
            }
            PipeBarrier<PIPE_V>();

            // step #3: Cast realLocal and imagLocal to Fp32
            Cast(realLocalFp32, realLocal, RoundMode::CAST_NONE, rows * (headSize >> 1));
            Cast(imagLocalFp32, imagLocal, RoundMode::CAST_NONE, rows * (headSize >> 1));
            PipeBarrier<PIPE_V>();

            // step #4: y0 = (realLocalFp32, imagLocalFp32) * cosLocalFp32
            Mul(y0, realLocalFp32, cosLocalFp32, /*mask*/ (headSize >> 1), /*repeat*/ rows,
                {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
            Mul(y0[(headSize >> 1)], imagLocalFp32, cosLocalFp32[(headSize >> 1)], /*mask*/ (headSize >> 1),
                /*repeat*/ rows, {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
            PipeBarrier<PIPE_V>();

            // step #5: y1 = (-imagLocalFp32, realLocalFp32) * sinLocalFp32
            Muls<float>(imagLocalFp32, imagLocalFp32, -1.0f, rows * (headSize >> 1));
            PipeBarrier<PIPE_V>();
            Mul(y1, imagLocalFp32, sinLocalFp32, /*mask*/ (headSize >> 1), /*repeat*/ rows,
                {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
            Mul(y1[(headSize >> 1)], realLocalFp32, sinLocalFp32[(headSize >> 1)], /*mask*/ (headSize >> 1),
                /*repeat*/ rows, {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
            PipeBarrier<PIPE_V>();

            // step #6: y0 = y0 + y1
            Add(y0, y0, y1, rows * headSize);
            PipeBarrier<PIPE_V>();

            // step #7: outLocal = Cast(y0, T)
            if constexpr (std::is_same<T, bfloat16_t>::value) {
                Cast(outLocal, y0, RoundMode::CAST_RINT, rows * headSize);
            } else if constexpr (std::is_same<T, half>::value) {
                Cast(outLocal, y0, RoundMode::CAST_NONE, rows * headSize);
            }
        }
    }

    template <typename T, int64_t D1, int64_t D0>
    __aicore__ inline void ScatterUpdatePANZ(
        const GlobalTensor<T>& dst, const GlobalTensor<T>& dstNd, const LocalTensor<T>& outLocal, int64_t startIdx,
        int64_t rows, int64_t headSize)
    {
        DataCopyExtParams copyParamsNz{
            static_cast<uint16_t>(D1), static_cast<uint32_t>(D0 * sizeof(T)), 0,
            static_cast<uint32_t>(this->tilingData_->blockSize * D0 * sizeof(T)) - static_cast<uint32_t>(D0 * sizeof(T)), 0};
        DataCopyExtParams copyParamsNd{1, static_cast<uint32_t>(headSize * sizeof(T)), 0, 0, 0};
        int64_t pageIdOffset = D1 * this->tilingData_->blockSize * D0 * this->tilingData_->numHead;
        for (int64_t i = 0; i < rows; i++) {
            int64_t tokenId = startIdx + i;
            int64_t ubOffset = headSize * i;
            if (this->tilingData_->isOutputKv) {
                int64_t gmOffsetNd = tokenId * headSize;
                DataCopyPad(dstNd[gmOffsetNd], outLocal[ubOffset], copyParamsNd);
            }

            // Index: (B*S,)
            // Cache: (bn, bs, N, D) -> (bn, N, D1, bs, D0) N>1
            int64_t tokensPerBatch = this->tilingData_->seqLength * this->tilingData_->numHead;
            int64_t batchId = tokenId / tokensPerBatch;
            int64_t tokenInBatch = tokenId % tokensPerBatch;                                             //token in batch
            int64_t headIdx = tokenInBatch / this->tilingData_->seqLength;                               //numHead
            int64_t seqIdx = tokenInBatch % this->tilingData_->seqLength;                                //seq

            int64_t pageOffset = this->indexGm(batchId * this->tilingData_->seqLength + seqIdx);         //[batch, seq]
            int64_t pageId = pageOffset / this->tilingData_->blockSize;
            int64_t tokenOffsetInCurrentPage = pageOffset % this->tilingData_->blockSize;
            if (pageOffset >= 0) {
                int64_t gmOffsetNz = pageId * pageIdOffset +                               //begin
                                     headIdx * D1 * this->tilingData_->blockSize * D0 +    //head
                                     tokenOffsetInCurrentPage * D0;                        //token
                SToMTE3Sync();
                DataCopyPad(dst[gmOffsetNz], outLocal[ubOffset], copyParamsNz);
            }
        }
    }

    template <typename T>
    __aicore__ inline void ScatterUpdatePABlkBnsd(
        const GlobalTensor<T>& dst, const GlobalTensor<T>& dstNd, const LocalTensor<T>& outLocal, int64_t startIdx,
        int64_t rows, int64_t headSize)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(headSize * sizeof(T)), 0, 0, 0};
        int64_t indexPageLength = (this->tilingData_->seqLength + this->tilingData_->blockSize - 1) / this->tilingData_->blockSize;
        int64_t pageIdOffset = this->tilingData_->blockSize * headSize * this->tilingData_->numHead;
        for (int64_t i = 0; i < rows; i++) {
            int64_t tokenId = startIdx + i;
            int64_t ubOffset = headSize * i;
            if (this->tilingData_->isOutputKv) {
                int64_t gmOffsetNd = tokenId * headSize;
                DataCopyPad(dstNd[gmOffsetNd], outLocal[ubOffset], copyParams);
            }

            // Index: (B*ceil(S/bs),)
            // Cache[𝐼𝑛𝑑𝑒𝑥[𝑏∗(𝑠/𝑏𝑠)],𝑠%𝑏𝑠,𝑛,𝑑]=𝑉𝑎𝑙𝑢𝑒[𝑏,𝑛,𝑠,𝑑]
            int64_t tokensPerBatch = this->tilingData_->seqLength * this->tilingData_->numHead;
            int64_t batchId = tokenId / tokensPerBatch;
            int64_t tokenInBatch = tokenId % tokensPerBatch;                                             //token in batch
            int64_t headIdx = tokenInBatch / this->tilingData_->seqLength;                               //numHead
            int64_t seqIdx = tokenInBatch % this->tilingData_->seqLength;                                //seq

            int64_t indexPageId = seqIdx / this->tilingData_->blockSize;
            int64_t indexOffset = batchId * indexPageLength + indexPageId;
            int64_t pageOffset = this->indexGm(indexOffset);
            int64_t tokenOffsetInCurrentPage = seqIdx % this->tilingData_->blockSize;
            int64_t pageId = pageOffset / this->tilingData_->blockSize;
            if (pageOffset >= 0) {
                // [BlockNum, BlockSize, N, headSize]
                int64_t gmOffset = pageId * pageIdOffset +
                                  tokenOffsetInCurrentPage * this->tilingData_->numHead * headSize +
                                  headIdx * headSize;
                SToMTE3Sync();
                DataCopyPad(dst[gmOffset], outLocal[ubOffset], copyParams);
            }
        }
    }

    template <typename T>
    __aicore__ inline void ScatterUpdatePABnsd(
        const GlobalTensor<T>& dst, const GlobalTensor<T>& dstNd, const LocalTensor<T>& outLocal, int64_t startIdx,
        int64_t rows, int64_t headSize)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(headSize * sizeof(T)), 0, 0, 0};
        for (int64_t i = 0; i < rows; i++) {
            int64_t tokenId = startIdx + i;
            int64_t ubOffset = headSize * i;
            if (this->tilingData_->isOutputKv) {
                int64_t gmOffsetNd = tokenId * headSize;
                DataCopyPad(dstNd[gmOffsetNd], outLocal[ubOffset], copyParams);
            }

            int64_t tokensPerBatch = this->tilingData_->seqLength * this->tilingData_->numHead;
            int64_t batchId = tokenId / tokensPerBatch;
            int64_t tokenInBatch = tokenId % tokensPerBatch;                                             //token in batch
            int64_t headIdx = tokenInBatch / this->tilingData_->seqLength;                               //numHead
            int64_t seqIdx = tokenInBatch % this->tilingData_->seqLength;                                //seq
            int64_t offset = this->indexGm(batchId * this->tilingData_->seqLength + seqIdx);             //[batch, seq]
            if (offset >= 0) {
                int64_t gmOffset = offset * headSize * this->tilingData_->numHead + headIdx * headSize;
                SToMTE3Sync();
                DataCopyPad(dst[gmOffset], outLocal[ubOffset], copyParams);
            }
        }
    }

    template <typename T, int64_t D1, int64_t D0>
    __aicore__ inline void DoScatter(
        const GlobalTensor<T>& dst, const GlobalTensor<T>& dstNd, const LocalTensor<T>& outLocal, int64_t startIdx,
        int64_t rows, int64_t headSize)
    {
        PipeBarrier<PIPE_ALL>();
        if constexpr (scatterType == PA_C8_NZ_NO_QUANT) {
            ScatterUpdatePANZ<T, D1, D0>(dst, dstNd, outLocal, startIdx, rows, headSize);
        } else if constexpr (scatterType == PA_C8_BLK_BNSD_NO_QUANT) {
            ScatterUpdatePABlkBnsd<T>(dst, dstNd, outLocal, startIdx, rows, headSize);
        } else if constexpr (scatterType == PA_C8_BNSD_NO_QUANT) {
            ScatterUpdatePABnsd<T>(dst, dstNd, outLocal, startIdx, rows, headSize);
        }
        PipeBarrier<PIPE_ALL>();
    }

    // ScatterUpdate with stride for non-contiguous dim-0 (only for ProcessV1)
    template <typename T>
    __aicore__ inline void DoScatterWithStride(
        const GlobalTensor<T>& dst, const GlobalTensor<T>& dstNd, const LocalTensor<T>& outLocal, int64_t startIdx,
        int64_t rows, int64_t headSize, int64_t stride)
    {
        PipeBarrier<PIPE_ALL>();
        // Bnsd with stride: decompose offset into page/token for stride
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(headSize * sizeof(T)), 0, 0, 0};
        for (int64_t i = 0; i < rows; i++) {
            int64_t tokenId = startIdx + i;
            int64_t ubOffset = headSize * i;
            if (this->tilingData_->isOutputKv) {
                DataCopyPad(dstNd[tokenId * headSize], outLocal[ubOffset], copyParams);
            }
            int64_t tokensPerBatch = this->tilingData_->seqLength * this->tilingData_->numHead;
            int64_t batchId = tokenId / tokensPerBatch;
            int64_t tokenInBatch = tokenId % tokensPerBatch;
            int64_t headIdx = tokenInBatch / this->tilingData_->seqLength;
            int64_t seqIdx = tokenInBatch % this->tilingData_->seqLength;
            int64_t offset = this->indexGm(batchId * this->tilingData_->seqLength + seqIdx);
            if (offset >= 0) {
                int64_t pageIdx = offset / this->tilingData_->blockSize;
                int64_t tokenInPage = offset % this->tilingData_->blockSize;
                int64_t gmOffset = pageIdx * stride + tokenInPage * headSize * this->tilingData_->numHead +
                                    headIdx * headSize; // 多头下stride的定义？
                SToMTE3Sync();
                DataCopyPad(dst[gmOffset], outLocal[ubOffset], copyParams);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    // ScatterUpdate for merged k/v cache (cacheLayOut == 1, only for ProcessV1)
    // Writes headSize elements to dst at dimOffset within a combined totalHeadSize dimension
    // Internally handles stride > blockSize for non-contiguous dim-0
    template <typename T, int64_t isSFACountiguous>
    __aicore__ inline void DoScatterRopeCache(
        const GlobalTensor<int8_t>& dst, const GlobalTensor<T>& dstNd, const LocalTensor<T>& outLocal, int64_t startIdx,
        int64_t rows, int64_t headSize, int64_t totalHeadSize)
    {
        PipeBarrier<PIPE_ALL>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(headSize * sizeof(T)), 0, 0, 0};
        for (int64_t i = 0; i < rows; i++) {
            int64_t tokenId = startIdx + i;
            int64_t ubOffset = headSize * i;
            if (this->tilingData_->isOutputKv) {
                DataCopyPad(dstNd[tokenId * headSize], outLocal[ubOffset], copyParams);
            }

            int64_t tokensPerBatch = this->tilingData_->seqLength * this->tilingData_->numHead;
            int64_t batchId = tokenId / tokensPerBatch;
            int64_t tokenInBatch = tokenId % tokensPerBatch;
            int64_t headIdx = tokenInBatch / this->tilingData_->seqLength;
            int64_t seqIdx = tokenInBatch % this->tilingData_->seqLength;

            int64_t gmOffset = -1;
            int64_t offset = this->indexGm(batchId * this->tilingData_->seqLength + seqIdx);
            if (offset >= 0) {
                if constexpr (isSFACountiguous != SFA_COUNTIGUOUS) {
                    int64_t pageIdx = offset / this->tilingData_->blockSize;
                    int64_t tokenInPage = offset % this->tilingData_->blockSize;
                    gmOffset = pageIdx * vStride + tokenInPage * totalHeadSize * this->tilingData_->numHead +
                                headIdx * totalHeadSize;// 多头下stride的定义？
                } else {
                    gmOffset = offset * totalHeadSize * this->tilingData_->numHead +
                                headIdx * totalHeadSize;
                }
                SToMTE3Sync();
                DataCopyPad(dst[gmOffset + this->RMS_NORM_LENGTH],
                            outLocal[ubOffset].template ReinterpretCast<int8_t>(), copyParams);
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

// RoundFloat2Int8: convert fp32 -> int8 via int32 -> half -> int8
    __aicore__ inline void RoundFloat2Int8(
        const LocalTensor<int8_t>& dstTensor, const LocalTensor<float>& srcTensor, int32_t size)
    {
        Cast(srcTensor.ReinterpretCast<int32_t>(), srcTensor, RoundMode::CAST_RINT, size);
        PipeBarrier<PIPE_V>();
        SetDeqScale((half)1.000000e+00f);
        PipeBarrier<PIPE_V>();
        Cast(srcTensor.ReinterpretCast<half>(), srcTensor.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, size);
        PipeBarrier<PIPE_V>();
        Cast(dstTensor, srcTensor.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, size);
    }

    // c8QuantCompute: dynamic block INT8 symmetric quantization
    // Input: fp32 data from RmsNorm workspace (xLocalFp32), shape [rows, headSize]
    // Output: int8 data written to outLocal, per-tile scale written to tileQScale
    template <int64_t isSFACountiguous>
    __aicore__ inline void c8QuantCompute(int64_t rows, int64_t headSize, int64_t startIdx, int64_t totalHeadSize)
    {
        // Get fp32 data from RmsNorm workspace (still valid after RmsNorm)
        LocalTensor<float> xLocalFp32 = this->wsBuffer.template Get<float>(); // 测试时跨函数是否有同步问题？
        // Reuse xSquareLocal zone (zone 1) as temporary buffer
        LocalTensor<float> tmpBuf = xLocalFp32[rows * headSize];
        // Reuse xSumLocal zone (zone 2) for ReduceMax result
        LocalTensor<float> reduceResult = xLocalFp32[rows * headSize * 2];
        LocalTensor<float> scaleLocal = this->tileQScale.template Get<float>();
        // Expanded reciprocal scale buffer [headSize floats], each scale duplicated to pertileSize
        // LocalTensor<float> expandedRecipScale = this->quantExpandBuf.template Get<float>();

        Abs(tmpBuf, xLocalFp32, rows * headSize); // abs 放到循环外面
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(headSize * sizeof(int8_t)), 0, 0, 0};
        DataCopyExtParams copyScaleParams{1, static_cast<uint32_t>(this->qTileNum * sizeof(float)), 0, 0, 0};
        for (int64_t rowIdx = 0; rowIdx < rows; rowIdx++) {
            LocalTensor<int8_t> outputLocal = this->outQueueV.template AllocTensor<int8_t>();
            // Step 1: Compute per-tile scale and fill expanded reciprocal scale buffer
            for (int64_t gIdx = 0; gIdx < this->qTileNum; gIdx++) {
                int64_t srcOffset = rowIdx * headSize + gIdx * this->pertileSize;
                PipeBarrier<PIPE_V>();

                ReduceMax(reduceResult, tmpBuf[srcOffset], reduceResult, this->pertileSize); //reduceResult 同时作为输出和工作缓冲区，中间归约过程会覆盖 dst 区域
                SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

                float maxVal = reduceResult.GetValue(0);
                float scale = maxVal / 127.0f; // scale为0
                float scaleTmp = (scale < DYNAMIC_QUANT_EPSINON) ? DYNAMIC_QUANT_EPSINON : scale;                // if scale < min_value:
                //     scale = min_value;             
                scaleLocal.SetValue(rowIdx * this->qTileAlign + gIdx, scale); // todo,not qTileAlign

                // Fill reciprocal scale to pertileSize positions
                float recipScale = 1.0f / scaleTmp;
                // Duplicate(expandedRecipScale[gIdx * this->pertileSize], recipScale, this->pertileSize);
                Muls(tmpBuf[srcOffset], xLocalFp32[srcOffset], recipScale, this->pertileSize);
            }
            PipeBarrier<PIPE_V>();
            RoundFloat2Int8(outputLocal, tmpBuf[rowIdx * headSize], headSize);
            this->outQueueV.EnQue(outputLocal);
            outputLocal = this->outQueueV.template DeQue<int8_t>();
            SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);
            DoScatterScaleAndVCache<isSFACountiguous>(this->vCacheGmInt8, outputLocal, scaleLocal, startIdx, rowIdx, headSize, 
            totalHeadSize, copyParams, copyScaleParams);

            this->outQueueV.FreeTensor(outputLocal);
            // Step 2: Vector multiply entire row at once
            // Mul(tmpBuf[rowIdx * headSize], xLocalFp32[rowIdx * headSize], expandedRecipScale, headSize);
        }

        // PipeBarrier<PIPE_V>();

        // Step 3: Convert float to int8
        // LocalTensor<int8_t> int8Result = this->wsBuffer.template Get<int8_t>();
        // RoundFloat2Int8(outputLocal, tmpBuf, rows * headSize);
    }

    template <int64_t isSFACountiguous>
    __aicore__ inline void DoScatterScaleAndVCache(
        const GlobalTensor<int8_t>& dst, const LocalTensor<int8_t>& outLocal, 
        const LocalTensor<float>& outScaleLocal, int64_t startIdx, int64_t rowsId, int64_t headSize, 
        int64_t totalHeadSize, DataCopyExtParams copyParams, DataCopyExtParams copyScaleParams)
    {
        // PipeBarrier<PIPE_ALL>();
        int64_t tokenId = startIdx + rowsId;
        // int64_t ubOffset = 0;
        int64_t scaleOffset = this->qTileAlign * rowsId;

        int64_t tokensPerBatch = this->tilingData_->seqLength * this->tilingData_->numHead;
        int64_t batchId = tokenId / tokensPerBatch;
        int64_t tokenInBatch = tokenId % tokensPerBatch;
        int64_t headIdx = tokenInBatch / this->tilingData_->seqLength;
        int64_t seqIdx = tokenInBatch % this->tilingData_->seqLength;

        int64_t gmOffset = -1;
        int64_t offset = this->indexGm(batchId * this->tilingData_->seqLength + seqIdx);
        if (offset >= 0) {
            if constexpr (isSFACountiguous != SFA_COUNTIGUOUS) {
                int64_t pageIdx = offset / this->tilingData_->blockSize;
                int64_t tokenInPage = offset % this->tilingData_->blockSize;
                gmOffset = pageIdx * vStride + tokenInPage * totalHeadSize * this->tilingData_->numHead +
                            headIdx * totalHeadSize;// 多头下stride的定义？
            } else {
                gmOffset = offset * totalHeadSize * this->tilingData_->numHead +
                            headIdx * totalHeadSize;
            }
            SToMTE3Sync();
            // GlobalTensor<T> vRopeCache = dst[gmOffset + this->RMS_NORM_LENGTH].ReinterpretCast<T>();
            DataCopyPad(dst[gmOffset], outLocal, copyParams);
            DataCopyPad(dst[gmOffset + this->RMS_NORM_LENGTH + this->ROPE_LENGTH * 2],
                        outScaleLocal[scaleOffset].template ReinterpretCast<int8_t>(), copyScaleParams);
        }
        SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    }

    template <typename T>
    __aicore__ inline void RmsNorm(
        const LocalTensor<T>& outLocal, const LocalTensor<T>& xLocal, const LocalTensor<float>& gammaLocal,
        const LocalTensor<float>& wsLocal, int64_t rows, int64_t headSize)
    {
        constexpr static int64_t NUM_EIGHT = 8;
        constexpr static int64_t NUM_SIXTEEN = 16;
        constexpr static int64_t NUM_SIXTY_FOUR = 64;
        int64_t xLocalFp32Offset = 0;                  // [0, rows * headSize]
        int64_t xSquareLocalOffset = rows * headSize;  // [rows * headSize, rows * headSize * 2]
        int64_t xSumLocalOffset = rows * headSize * 2; // [rows * headSize * 2, rows * headSize * 3]
        LocalTensor<float> xLocalFp32 = wsLocal[xLocalFp32Offset];
        LocalTensor<float> xSquareLocal = wsLocal[xSquareLocalOffset];
        LocalTensor<float> xSumLocal = wsLocal[xSumLocalOffset];

        Cast(xLocalFp32, xLocal, RoundMode::CAST_NONE, rows * headSize);
        PipeBarrier<PIPE_V>();
        Mul(xSquareLocal, xLocalFp32, xLocalFp32, rows * headSize);
        PipeBarrier<PIPE_V>();
        int64_t repeatTimes = rows * (headSize >> 1) / NUM_SIXTY_FOUR;
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[NUM_SIXTY_FOUR], NUM_SIXTY_FOUR, repeatTimes,
            {1, 1, 1, NUM_EIGHT, NUM_SIXTEEN, NUM_SIXTEEN}); // 256
        PipeBarrier<PIPE_V>();
        repeatTimes = repeatTimes / 2;
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[NUM_SIXTY_FOUR], NUM_SIXTY_FOUR, repeatTimes,
            {1, 1, 1, NUM_EIGHT, NUM_SIXTEEN, NUM_SIXTEEN}); // 128
        PipeBarrier<PIPE_V>();
        repeatTimes = repeatTimes / 2;
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[NUM_SIXTY_FOUR], NUM_SIXTY_FOUR, repeatTimes,
            {1, 1, 1, NUM_EIGHT, NUM_SIXTEEN, NUM_SIXTEEN}); // 64
        PipeBarrier<PIPE_V>();

        WholeReduceSum(xSumLocal, xSquareLocal, NUM_SIXTY_FOUR, rows, 1, 1, NUM_EIGHT);
        PipeBarrier<PIPE_V>();
        // Calc: xSum = xSum * reciprocal
        Muls<float>(xSumLocal, xSumLocal, this->tilingData_->reciprocal, rows);
        PipeBarrier<PIPE_V>();
        // Calc: xSum = xSum + epsilon
        Adds<float>(xSumLocal, xSumLocal, this->tilingData_->epsilon, rows);
        PipeBarrier<PIPE_V>();
        // Calc: xSum = sqrt(xSum)
        Sqrt(xSumLocal, xSumLocal, rows);
        PipeBarrier<PIPE_V>();

        // Brcb
        int64_t block_count = (rows + NUM_EIGHT - 1) / NUM_EIGHT;
        Brcb(xSquareLocal, xSumLocal, block_count, {1, NUM_EIGHT});
        PipeBarrier<PIPE_V>();

        // Calc: xLocalFp32 = xLocalFp32 / xSquareLocal
        for (int64_t rowId = 0; rowId < rows; rowId++) {
            Div(xLocalFp32[rowId * headSize], xLocalFp32[rowId * headSize], xSquareLocal[rowId * NUM_EIGHT],
                NUM_SIXTY_FOUR, (headSize / NUM_SIXTY_FOUR), {1, 1, 0, NUM_EIGHT, NUM_EIGHT, 0});
        }
        PipeBarrier<PIPE_V>();

        // Calc: xLocalFp32 = xLocalFp32 * xSquareLocal [rows, this->RMS_NORM_LENGTH] * [this->RMS_NORM_LENGTH]
        for (int64_t rowId = 0; rowId < rows; rowId++) {
            Mul(xLocalFp32[rowId * headSize], xLocalFp32[rowId * headSize], gammaLocal, headSize);
        }
        PipeBarrier<PIPE_V>();

        // Cast xLocalFp32 to outLocal
        if constexpr (std::is_same<KV_DTYPE, bfloat16_t>::value) {
            Cast(outLocal, xLocalFp32, RoundMode::CAST_RINT, rows * headSize);
        } else if constexpr (std::is_same<KV_DTYPE, half>::value) {
            Cast(outLocal, xLocalFp32, RoundMode::CAST_NONE, rows * headSize);
        }
    }

    __aicore__ inline void SToMTE3Sync()
    {
        event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    }

private:
    int64_t methodMode = 0;
    int64_t ropeMode = INTERLEAVE_MODE; // INTERLEAVE_MODE or HALF_MODE
    int64_t kStride = 0;
    int64_t vStride = 0;
    int64_t cacheLayOut = CONTIGUOUS; // SWA_MODE or SFA_NO_QUANT
    int64_t pertileSize = 128;
    int64_t qTileNum = 0;
    int64_t qTileAlign = 0;
    int64_t totalHeadSizeInt8 = 0;
    int64_t pertileSizeAlign = 0;
};
} // namespace AiInfraKvRmsNormRopeCache

#endif // _AI_INFRA_KV_RMS_NORM_ROPE_CACHE_B16_PA_BNSD_C8_H_```