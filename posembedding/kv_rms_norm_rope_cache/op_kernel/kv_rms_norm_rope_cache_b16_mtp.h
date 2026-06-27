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
 * \file kv_rms_norm_rope_cache_b16_mtp.h
 * \brief
 */
#ifndef _KV_RMS_NORM_ROPE_CACHE_B16_MTP_H_
#define _KV_RMS_NORM_ROPE_CACHE_B16_MTP_H_
#include "kv_rms_norm_rope_cache_comm.h"

namespace KvRmsNormRopeCache {
using namespace AscendC;

template <bool isPagedAttention, typename KV_DTYPE>
class KernelKvRmsNormRopeCacheB16MTP :
public KernelKvRmsNormRopeCacheMTP<isPagedAttention, KV_DTYPE, KV_DTYPE, KV_DTYPE>
{
public:
    __aicore__ inline KernelKvRmsNormRopeCacheB16MTP(TPipe* pipe, const KvRmsNormRopeCacheTilingData* tiling)
        : KernelKvRmsNormRopeCacheMTP<isPagedAttention, KV_DTYPE, KV_DTYPE, KV_DTYPE> (pipe, tiling)
    {}

    __aicore__ inline void Init(
        GM_ADDR kv, GM_ADDR gamma, GM_ADDR cos, GM_ADDR sin, GM_ADDR index, GM_ADDR k_cache, GM_ADDR v_cache, GM_ADDR v)
    {
        methodMode = (v == nullptr) ? METHOD_V1 : METHOD_V2;
        this->InitSharedData(methodMode);

        // 获取常用tiling参数
        const auto& tiling = *this->tilingData_;
        this->rowsPerBlock = tiling.rowsPerBlock;
        const uint64_t seqLen = tiling.seqLength;

        // V1 does not support multi-head (N>1)
        const uint64_t numHead = (methodMode == METHOD_V2) ? tiling.numHead : NUM_ONE;

        const uint64_t cacheLen = tiling.cacheLength;
        const uint64_t rmsNormLen = this->RMS_NORM_LENGTH;
        const uint64_t ropeLen = this->ROPE_LENGTH;
        const uint64_t vLen = this->V_LENGTH;
        const uint64_t blockIdx = GetBlockIdx();

        const uint64_t kvRowOffset = blockIdx * this->rowsPerBlock;       // Local B
        const uint64_t kvRowHeadOffset = kvRowOffset * numHead;     // B * N
        const uint64_t kvTkOffset = kvRowOffset * seqLen;           // B * S
        const uint64_t bnsOffset = kvRowHeadOffset * seqLen;        // B * N * S

        const uint64_t cacheElemOffset = kvRowOffset * cacheLen;
        const uint64_t cacheElemHeadOffset = kvRowOffset * cacheLen * numHead;

        // Init shared Buffers
        this->gammaGm.SetGlobalBuffer((__gm__ KV_DTYPE*)gamma);

        const uint64_t ropeOffset = kvRowHeadOffset * ropeLen;
        this->cosGm.SetGlobalBuffer((__gm__ KV_DTYPE*)cos + ropeOffset);
        this->sinGm.SetGlobalBuffer((__gm__ KV_DTYPE*)sin + ropeOffset);

        // GM and Cache offsets
        uint64_t kvOffset = 0;
        uint64_t kCacheOffset = cacheElemHeadOffset * this->hDimK;
        uint64_t vCacheOffset = cacheElemHeadOffset * this->hDimV;

        // init global memory
        if (methodMode == METHOD_V1) {
            // KV concat
            kvOffset = bnsOffset * (this->hDimK + this->hDimV);

            this->kvGm.SetGlobalBuffer((__gm__ KV_DTYPE*)kv + kvOffset);
        } else if (methodMode == METHOD_V2) {
            // KV Split
            kvOffset = bnsOffset * this->hDimK;
            uint64_t vOffset = bnsOffset * this->hDimV;

            this->kvGm.SetGlobalBuffer((__gm__ KV_DTYPE*)kv + kvOffset);
            this->vGm.SetGlobalBuffer((__gm__ KV_DTYPE*)v + vOffset);
        }

        // Paged Attention
        if constexpr (isPagedAttention) {
            this->indexGm.SetGlobalBuffer((__gm__ int64_t*)index);
            this->kCacheGm.SetGlobalBuffer((__gm__ KV_DTYPE*)k_cache);
            this->vCacheGm.SetGlobalBuffer((__gm__ KV_DTYPE*)v_cache);
        } else {
            const uint64_t indexOffset = kvTkOffset;
            this->indexGm.SetGlobalBuffer((__gm__ int64_t*)index + indexOffset);
            this->kCacheGm.SetGlobalBuffer((__gm__ KV_DTYPE*)k_cache + kCacheOffset);
            this->vCacheGm.SetGlobalBuffer((__gm__ KV_DTYPE*)v_cache + vCacheOffset);
        }

        // Init pipes
        this->rmsNormElemPerSlice = static_cast<uint32_t>(this->rowsPerBlock * rmsNormLen);
        this->ropeElemPerSlice = static_cast<uint32_t>(this->rowsPerBlock * ropeLen);
        this->vElemPerSlice = static_cast<uint32_t>(this->rowsPerBlock * this->hDimV);
        this->kElemPerSlice = static_cast<uint32_t>(this->rowsPerBlock * this->hDimK);

        this->pipe_->InitBuffer(this->inQueueX, 1, this->rmsNormElemPerSlice
            * sizeof(float));  // MTE2 for x: rmsNormElemPerSlice * 4 / 1024 = rmsNormElemPerSlice / 256
        this->pipe_->InitBuffer(this->inQueueGamma, 1, rmsNormLen
            * sizeof(KV_DTYPE));  // rmsNormLen * 2 / 1024 = rmsNormLen / 512
        this->pipe_->InitBuffer(this->bufferXFp32, this->rmsNormElemPerSlice
            * sizeof(float));  // rmsNormElemPerSlice * 4 / 1024 = rmsNormLen / 256
        this->pipe_->InitBuffer(this->bufferXSquare, this->rmsNormElemPerSlice
            * sizeof(float));  // rmsNormElemPerSlice * 4 / 1024 = rmsNormLen / 256

        // SafeAlignUp(rowsPerBlock, 8) * 4 / 1024 = SafeAlignUp(rowsPerBlock, 8) / 256
        this->pipe_->InitBuffer(this->bufferSum, SafeAlignUp(this->rowsPerBlock, NUM_EIGHT) * sizeof(float));

        this->pipe_->InitBuffer(this->outQueueV, 1, this->vElemPerSlice
            * sizeof(KV_DTYPE));  // vElemPerSlice * 2 / 1024 = vElemPerSlice / 512
        this->pipe_->InitBuffer(this->outQueueK, 1, this->kElemPerSlice
            * sizeof(KV_DTYPE));  // kElemPerSlice * 2 / 1024 = kElemPerSlice / 512

        if (methodMode == METHOD_V2) {
            // init additional pipes for V2
            this->pipe_->InitBuffer(this->inQueueV, 1, this->vElemPerSlice
                * sizeof(float));  // vElemPerSlice * 4 / 1024 = vElemPerSlice / 256

            //  shared buffer for Rope MTE2 (in kv_type) and VEC computation in fp32
            this->pipe_->InitBuffer(this->CosSin, 1, NUM_TWO * this->ropeElemPerSlice
                * sizeof(float));   // 2 * (b * D_rope) * 4 / 1024 = (b * D_rope) / 128
        }
    }

    __aicore__ inline void Process()
    {
        if (methodMode == 0) {
            for (int64_t i = 0; i < this->tilingData_->seqLength; i++) {
                RopeK(i);
                PipeBarrier<PIPE_V>();
                ScatterUpdateK(i);
                PipeBarrier<PIPE_V>();
                RmsNorm(i);
                PipeBarrier<PIPE_V>();
                ScatterUpdateV(i);
            }
        } else if (methodMode == 1) {
            for (uint64_t n = 0; n < this->tilingData_->numHead; n++) {
                for (int64_t i = 0; i < this->tilingData_->seqLength; i++) {
                    // Process K
                    LocalTensor<KV_DTYPE> xB16Local = this->inQueueX.template AllocTensor<KV_DTYPE>();
                    DataCopyExtParams copyXParams{
                    static_cast<uint16_t>(this->rowsPerBlock),
                    static_cast<uint32_t>(this->RMS_NORM_LENGTH * sizeof(KV_DTYPE)),
                    static_cast<uint32_t>(
                        ((this->tilingData_->seqLength - 1) * this->RMS_NORM_LENGTH +
                        (this->tilingData_->numHead - 1) * this->tilingData_->seqLength * this->RMS_NORM_LENGTH) * sizeof(KV_DTYPE)),
                    0, 0};
                    DataCopyPadExtParams<KV_DTYPE> padXParams{false, 0, 0, 0};
                    int64_t nXOffset = n * this->tilingData_->seqLength * this->RMS_NORM_LENGTH + i * this->RMS_NORM_LENGTH;
                    DataCopyPad(xB16Local, this->kvGm[nXOffset], copyXParams, padXParams);
                    this->inQueueX.EnQue(xB16Local);
                    xB16Local = this->inQueueX.template DeQue<KV_DTYPE>();

                    LocalTensor<float> xFp32Local = this->bufferXFp32.template Get<float>();
                    Cast(xFp32Local, xB16Local, RoundMode::CAST_NONE, this->rmsNormElemPerSlice);
                    PipeBarrier<PIPE_V>();
                    this->inQueueX.FreeTensor(xB16Local);

                    LocalTensor<KV_DTYPE> gammaLocal = this->inQueueGamma.template AllocTensor<KV_DTYPE>();
                    DataCopy(gammaLocal, this->gammaGm, this->RMS_NORM_LENGTH);
                    this->inQueueGamma.EnQue(gammaLocal);
                    gammaLocal = this->inQueueGamma.template DeQue<KV_DTYPE>();

                    RmsNormV2<KV_DTYPE>(xFp32Local, gammaLocal);
                    this->inQueueGamma.FreeTensor(gammaLocal);
                    PipeBarrier<PIPE_V>();

                    LocalTensor<KV_DTYPE> cosLocal = this->CosSin.template AllocTensor<KV_DTYPE>();
                    LocalTensor<KV_DTYPE> sinLocal = cosLocal[this->rowsPerBlock * this->ROPE_LENGTH * NUM_TWO];
                    LocalTensor<KV_DTYPE> cosLocalB16 = cosLocal[this->rowsPerBlock * this->ROPE_LENGTH];
                    LocalTensor<KV_DTYPE> sinLocalB16 = sinLocal[this->rowsPerBlock * this->ROPE_LENGTH];

                    // load cosLocal, sinLocal [this->rowsPerBlock, this->ROPE_LENGTH]
                    LocalTensor<KV_DTYPE> xLocal = this->inQueueX.template DeQue<KV_DTYPE>();
                    DataCopyExtParams copyCosSinParams{
                        static_cast<uint16_t>(this->rowsPerBlock),
                        static_cast<uint32_t>(this->ROPE_LENGTH * sizeof(KV_DTYPE)),
                        static_cast<uint32_t>((this->tilingData_->numHead - 1) * this->ROPE_LENGTH * sizeof(KV_DTYPE)), 0, 0};
                    DataCopyPad(cosLocalB16, this->cosGm[n * this->ROPE_LENGTH], copyCosSinParams, {false, 0, 0, 0});
                    DataCopyPad(sinLocalB16, this->sinGm[n * this->ROPE_LENGTH], copyCosSinParams, {false, 0, 0, 0});
                    this->CosSin.EnQue(cosLocal);
                    cosLocal = this->CosSin.template DeQue<KV_DTYPE>();
                    sinLocal = cosLocal[this->rowsPerBlock * this->ROPE_LENGTH * NUM_TWO];
                    cosLocalB16 = cosLocal[this->rowsPerBlock * this->ROPE_LENGTH];
                    sinLocalB16 = sinLocal[this->rowsPerBlock * this->ROPE_LENGTH];

                    RopeKV2<KV_DTYPE>(xLocal, xFp32Local, cosLocal, sinLocal, cosLocalB16, sinLocalB16);
                    this->CosSin.FreeTensor(cosLocal);
                    PipeBarrier<PIPE_V>();

                    LocalTensor<KV_DTYPE> outLocalK = this->outQueueK.template AllocTensor<KV_DTYPE>();
                    DataCopy(outLocalK, xLocal, this->rmsNormElemPerSlice);
                    this->inQueueX.FreeTensor(xLocal);

                    ScatterUpdate<KV_DTYPE>(n, i, this->kCacheGm, outLocalK, this->RMS_NORM_LENGTH);
                    this->outQueueK.FreeTensor(outLocalK);
                    PipeBarrier<PIPE_V>();

                    // Process V
                    LocalTensor<KV_DTYPE> yLocal = this->inQueueV.template AllocTensor<KV_DTYPE>();
                    DataCopyExtParams copyYParams{
                    static_cast<uint16_t>(this->rowsPerBlock), static_cast<uint32_t>(this->V_LENGTH * sizeof(KV_DTYPE)),
                    static_cast<uint32_t>(
                        ((this->tilingData_->seqLength - 1) * this->V_LENGTH +
                        (this->tilingData_->numHead - 1) * this->tilingData_->seqLength * this->V_LENGTH) * sizeof(KV_DTYPE)),
                    0, 0};
                    DataCopyPadExtParams<KV_DTYPE> padYParams{false, 0, 0, 0};
                    int64_t nYOffset = n * this->tilingData_->seqLength * this->V_LENGTH + i * this->V_LENGTH;
                    DataCopyPad(yLocal, vGm[nYOffset], copyYParams, padYParams);
                    this->inQueueV.EnQue(yLocal);
                    yLocal = this->inQueueV.template DeQue<KV_DTYPE>();

                    LocalTensor<KV_DTYPE> outLocalV = this->outQueueV.template AllocTensor<KV_DTYPE>();
                    DataCopy(outLocalV, yLocal, this->vElemPerSlice);
                    this->inQueueV.FreeTensor(yLocal);

                    ScatterUpdate<KV_DTYPE>(n, i, this->vCacheGm, outLocalV, this->V_LENGTH);
                    this->outQueueV.FreeTensor(outLocalV);
                }
            }
        } else {
            //exception
        }
    }

    __aicore__ inline void RopeK(int64_t seqId)
    {
        LocalTensor<KV_DTYPE> kLocal = this->inQueueX.template AllocTensor<KV_DTYPE>();
        LocalTensor<KV_DTYPE> cosLocal = kLocal[this->rowsPerBlock * this->ROPE_LENGTH];
        LocalTensor<KV_DTYPE> sinLocal = cosLocal[this->rowsPerBlock * this->ROPE_LENGTH * NUM_TWO];
        LocalTensor<KV_DTYPE> cosLocalB16 = cosLocal[this->rowsPerBlock * this->ROPE_LENGTH];
        LocalTensor<KV_DTYPE> sinLocalB16 = sinLocal[this->rowsPerBlock * this->ROPE_LENGTH];

        // load kLocal [this->rowsPerBlock, this->ROPE_LENGTH]
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(this->rowsPerBlock), static_cast<uint32_t>(this->ROPE_LENGTH * sizeof(KV_DTYPE)),
            static_cast<uint32_t>(
                this->RMS_NORM_LENGTH * sizeof(KV_DTYPE) +
                (this->tilingData_->seqLength - 1) * (this->RMS_NORM_LENGTH + this->ROPE_LENGTH) * sizeof(KV_DTYPE)),
            0, 0};
        DataCopyPadExtParams<KV_DTYPE> padParams{false, 0, 0, 0};
        DataCopyPad(kLocal, this->kvGm[seqId * (this->RMS_NORM_LENGTH + this->ROPE_LENGTH) + this->RMS_NORM_LENGTH], copyParams, padParams);

        // load cosLocal, sinLocal [this->rowsPerBlock, this->ROPE_LENGTH]
        DataCopy(cosLocalB16, this->cosGm, this->rowsPerBlock * this->ROPE_LENGTH);
        DataCopy(sinLocalB16, this->sinGm, this->rowsPerBlock * this->ROPE_LENGTH);
        this->inQueueX.EnQue(kLocal);
        kLocal = this->inQueueX.template DeQue<KV_DTYPE>();
        cosLocal = kLocal[this->rowsPerBlock * this->ROPE_LENGTH];
        sinLocal = cosLocal[this->rowsPerBlock * this->ROPE_LENGTH * NUM_TWO];
        cosLocalB16 = cosLocal[this->rowsPerBlock * this->ROPE_LENGTH];
        sinLocalB16 = sinLocal[this->rowsPerBlock * this->ROPE_LENGTH];

        // realLocal, imagLocal comes from odd/even elements in kLocal
        LocalTensor<KV_DTYPE> k = this->bufferXFp32.template Get<KV_DTYPE>();
        LocalTensor<KV_DTYPE> realLocal = k[0];
        LocalTensor<KV_DTYPE> imagLocal = realLocal[this->rowsPerBlock * this->ROPE_LENGTH_HALF * NUM_TWO];
        LocalTensor<KV_DTYPE> buf_ = imagLocal[this->rowsPerBlock * this->ROPE_LENGTH_HALF * NUM_TWO];
        LocalTensor<KV_DTYPE> realLocalB16 = realLocal[this->rowsPerBlock * this->ROPE_LENGTH_HALF];
        LocalTensor<KV_DTYPE> imagLocalB16 = imagLocal[this->rowsPerBlock * this->ROPE_LENGTH_HALF];

        uint64_t rsvdCnt = 0;
        GatherMask(
            realLocalB16, kLocal, 1, true, this->rowsPerBlock * this->ROPE_LENGTH, {1, 1, NUM_EIGHT, 0}, rsvdCnt);
        GatherMask(
            imagLocalB16, kLocal, NUM_TWO, true, this->rowsPerBlock * this->ROPE_LENGTH, {1, 1, NUM_EIGHT, 0},
            rsvdCnt);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> realLocalFp32 = realLocal.template ReinterpretCast<float>();
        LocalTensor<float> imagLocalFp32 = imagLocal.template ReinterpretCast<float>();
        LocalTensor<float> cosLocalFp32 = cosLocal.template ReinterpretCast<float>();
        LocalTensor<float> sinLocalFp32 = sinLocal.template ReinterpretCast<float>();

        Cast(realLocalFp32, realLocalB16, RoundMode::CAST_NONE, this->rowsPerBlock * this->ROPE_LENGTH_HALF);
        Cast(imagLocalFp32, imagLocalB16, RoundMode::CAST_NONE, this->rowsPerBlock * this->ROPE_LENGTH_HALF);
        Cast(cosLocalFp32, cosLocalB16, RoundMode::CAST_NONE, this->rowsPerBlock * this->ROPE_LENGTH);
        Cast(sinLocalFp32, sinLocalB16, RoundMode::CAST_NONE, this->rowsPerBlock * this->ROPE_LENGTH);
        PipeBarrier<PIPE_V>();

        uint64_t mask[NUM_TWO] = {MASK_CONFIG, 0}; // mask this->ROPE_LENGTH_HALF Elements
        LocalTensor<KV_DTYPE> outLocal = this->outQueueK.template AllocTensor<KV_DTYPE>();
        LocalTensor<float> outLocalFp32 = this->bufferXSquare.template Get<float>();
        Mul(outLocalFp32, realLocalFp32, cosLocalFp32, mask, this->rowsPerBlock,
            {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
        Mul(outLocalFp32[this->ROPE_LENGTH_HALF], imagLocalFp32, cosLocalFp32[this->ROPE_LENGTH_HALF], mask,
            this->rowsPerBlock, {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
        PipeBarrier<PIPE_V>();

        Muls<float>(imagLocalFp32, imagLocalFp32, -1.0f, this->rowsPerBlock * this->ROPE_LENGTH_HALF);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> bufFp32 = buf_.template ReinterpretCast<float>();

        Mul(bufFp32, imagLocalFp32, sinLocalFp32, mask,
            this->rowsPerBlock, {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
        Mul(bufFp32[this->ROPE_LENGTH_HALF], realLocalFp32, sinLocalFp32[this->ROPE_LENGTH_HALF], mask,
            this->rowsPerBlock, {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
        PipeBarrier<PIPE_V>();
        this->inQueueX.FreeTensor(kLocal);

        Add(outLocalFp32, outLocalFp32, bufFp32, this->rowsPerBlock * this->ROPE_LENGTH);
        PipeBarrier<PIPE_V>();

        if constexpr (std::is_same<KV_DTYPE, bfloat16_t>::value) {
            Cast(outLocal, outLocalFp32, RoundMode::CAST_RINT, this->rowsPerBlock * this->ROPE_LENGTH);
        } else if constexpr (std::is_same<KV_DTYPE, half>::value) {
            Cast(outLocal, outLocalFp32, RoundMode::CAST_NONE, this->rowsPerBlock * this->ROPE_LENGTH);
        }

        this->outQueueK.EnQue(outLocal);
    }

    template<typename T>
    __aicore__ inline void RopeKV2(
        const LocalTensor<T>& xLocal, const LocalTensor<float>& xFp32Local, const LocalTensor<T>& cosLocal, const LocalTensor<T>& sinLocal,
        const LocalTensor<T>& cosLocalB16, const LocalTensor<T>& sinLocalB16)
    {
        // realLocal, imagLocal comes from odd/even elements in kLocal
        LocalTensor<KV_DTYPE> k = this->bufferXSquare.template Get<KV_DTYPE>();
        LocalTensor<KV_DTYPE> realLocal = k[0];
        LocalTensor<KV_DTYPE> imagLocal = realLocal[this->rowsPerBlock * this->ROPE_LENGTH_HALF * NUM_TWO];
        LocalTensor<KV_DTYPE> buf_ = imagLocal[this->rowsPerBlock * this->ROPE_LENGTH_HALF * NUM_TWO];
        // Split the tail block of bufferXSquare[b*192] as rope result buffer in [b*64]
        // this ONLY works under V2 scenario.
        LocalTensor<KV_DTYPE> outLocal = buf_[this->rowsPerBlock * this->ROPE_LENGTH * NUM_TWO];

        LocalTensor<float> realLocalFp32 = realLocal.template ReinterpretCast<float>();
        LocalTensor<float> imagLocalFp32 = imagLocal.template ReinterpretCast<float>();
        LocalTensor<float> cosLocalFp32 = cosLocal.template ReinterpretCast<float>();
        LocalTensor<float> sinLocalFp32 = sinLocal.template ReinterpretCast<float>();
        LocalTensor<float> outLocalFp32 = outLocal.template ReinterpretCast<float>();
        PipeBarrier<PIPE_V>();

        uint64_t rsvdCnt = 0;
        GatherMask(
            realLocalFp32, xFp32Local, 1, true, this->ROPE_LENGTH,
            {1, static_cast<uint16_t>(this->rowsPerBlock), 2 * NUM_TWELVE, 0},
            rsvdCnt); // Pick even (x1)
        GatherMask(
            imagLocalFp32, xFp32Local, NUM_TWO, true, this->ROPE_LENGTH,
            {1, static_cast<uint16_t>(this->rowsPerBlock), 2 * NUM_TWELVE, 0},
            rsvdCnt); // Pick Odd (x2)
        PipeBarrier<PIPE_V>();

        Cast(cosLocalFp32, cosLocalB16, RoundMode::CAST_NONE, this->rowsPerBlock * this->ROPE_LENGTH);
        Cast(sinLocalFp32, sinLocalB16, RoundMode::CAST_NONE, this->rowsPerBlock * this->ROPE_LENGTH);
        PipeBarrier<PIPE_V>();

        uint64_t mask[NUM_TWO] = {MASK_CONFIG, 0}; // mask this->ROPE_LENGTH_HALF Elements

        Mul(outLocalFp32, realLocalFp32, cosLocalFp32, mask,
            this->rowsPerBlock,
            {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
        Mul(outLocalFp32[this->ROPE_LENGTH_HALF], imagLocalFp32, cosLocalFp32[this->ROPE_LENGTH_HALF], mask,
            this->rowsPerBlock, {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
        PipeBarrier<PIPE_V>();

        Muls<float>(imagLocalFp32, imagLocalFp32, -1.0f, this->rowsPerBlock * this->ROPE_LENGTH_HALF);
        PipeBarrier<PIPE_V>();
        LocalTensor<float> bufFp32 = buf_.template ReinterpretCast<float>();

        Mul(bufFp32, imagLocalFp32, sinLocalFp32, mask,
            this->rowsPerBlock,
            {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
        Mul(bufFp32[this->ROPE_LENGTH_HALF], realLocalFp32, sinLocalFp32[this->ROPE_LENGTH_HALF], mask,
            this->rowsPerBlock,
            {1, 1, 1, NUM_EIGHT, NUM_FOUR, NUM_EIGHT});
        PipeBarrier<PIPE_V>();

        Add(outLocalFp32, outLocalFp32, bufFp32, this->rowsPerBlock * this->ROPE_LENGTH);
        PipeBarrier<PIPE_V>();

        // Overwrite the head 64 elements in each batch with rope result
        // xLocal in 2 * 192 / 32 = 12 blocks for each batch
        // outLocalFp32 from the EXACT tail of bufferXSquare in 4 * 64 / 32 = 8 blocks for each batch
        if constexpr (std::is_same<KV_DTYPE, bfloat16_t>::value) {
            Cast(xLocal, outLocalFp32, RoundMode::CAST_RINT, this->ROPE_LENGTH,
                this->rowsPerBlock, {1, 1, NUM_TWELVE, NUM_EIGHT});
        } else if constexpr (std::is_same<KV_DTYPE, half>::value) {
            Cast(xLocal, outLocalFp32, RoundMode::CAST_NONE, this->ROPE_LENGTH,
                this->rowsPerBlock, {1, 1, NUM_TWELVE, NUM_EIGHT});
        }
    }

    __aicore__ inline void RmsNorm(int64_t seqId)
    {
        // load xB16 [this->rowsPerBlock, this->RMS_NORM_LENGTH]
        LocalTensor<KV_DTYPE> xB16Local = this->inQueueX.template AllocTensor<KV_DTYPE>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(this->rowsPerBlock), static_cast<uint32_t>(this->RMS_NORM_LENGTH * sizeof(KV_DTYPE)),
            static_cast<uint32_t>(
                this->ROPE_LENGTH * sizeof(KV_DTYPE) +
                (this->tilingData_->seqLength - 1) * (this->RMS_NORM_LENGTH + this->ROPE_LENGTH) * sizeof(KV_DTYPE)),
            0, 0};
        DataCopyPadExtParams<KV_DTYPE> padParams{false, 0, 0, 0};
        DataCopyPad(xB16Local, this->kvGm[seqId * (this->RMS_NORM_LENGTH + this->ROPE_LENGTH)], copyParams, padParams);
        this->inQueueX.EnQue(xB16Local);
        xB16Local = this->inQueueX.template DeQue<KV_DTYPE>();

        // Cast xB16 to xFp32 [this->rowsPerBlock, this->RMS_NORM_LENGTH]
        LocalTensor<float> xFp32Local = this->bufferXFp32.template Get<float>();
        Cast(xFp32Local, xB16Local, RoundMode::CAST_NONE, this->rmsNormElemPerSlice);
        PipeBarrier<PIPE_V>();
        this->inQueueX.FreeTensor(xB16Local);

        // load gamma [this->RMS_NORM_LENGTH]
        LocalTensor<KV_DTYPE> gammaLocal = this->inQueueGamma.template AllocTensor<KV_DTYPE>();
        DataCopy(gammaLocal, this->gammaGm, this->RMS_NORM_LENGTH);
        this->inQueueGamma.EnQue(gammaLocal);
        gammaLocal = this->inQueueGamma.template DeQue<KV_DTYPE>();

        // Calc: xSquare = xFp32 * xFp32, xSquare shape is [this->rowsPerBlock, this->RMS_NORM_LENGTH]
        LocalTensor<float> xSquareLocal = this->bufferXSquare.template Get<float>();
        Mul(xSquareLocal, xFp32Local, xFp32Local, this->rmsNormElemPerSlice);
        PipeBarrier<PIPE_V>();

        // Calc: xSquare[this->rowsPerBlock, 512]
        //          --> [this->rowsPerBlock, 256]
        //          --> [this->rowsPerBlock, 128]
        //          --> [this->rowsPerBlock, 64]
        uint64_t mask[NUM_TWO] = {UINT64_MAX, UINT64_MAX};
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[NUM_SIXTY_FOUR], mask, NUM_FOUR * this->rowsPerBlock,
            {1, 1, 1, NUM_EIGHT, NUM_SIXTEEN, NUM_SIXTEEN}); // [this->rowsPerBlock, 256]
        PipeBarrier<PIPE_V>();
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[NUM_SIXTY_FOUR], mask, NUM_TWO * this->rowsPerBlock,
            {1, 1, 1, NUM_EIGHT, NUM_SIXTEEN, NUM_SIXTEEN}); // [this->rowsPerBlock, 128]
        PipeBarrier<PIPE_V>();
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[NUM_SIXTY_FOUR], mask, this->rowsPerBlock,
            {1, 1, 1, NUM_EIGHT, NUM_SIXTEEN, NUM_SIXTEEN}); // [this->rowsPerBlock, 64]
        PipeBarrier<PIPE_V>();

        // Calc: xSum = [this->rowsPerBlock, 1]
        LocalTensor<float> xSumLocal = this->bufferSum.template Get<float>();
        WholeReduceSum(xSumLocal, xSquareLocal, NUM_SIXTY_FOUR, this->rowsPerBlock, 1, 1, NUM_EIGHT);
        PipeBarrier<PIPE_V>();

        // Calc: xSum = xSum * reciprocal
        Muls<float>(xSumLocal, xSumLocal, this->tilingData_->reciprocal, this->rowsPerBlock);
        PipeBarrier<PIPE_V>();

        // Calc: xSum = xSum + epsilon
        Adds<float>(xSumLocal, xSumLocal, this->tilingData_->epsilon, this->rowsPerBlock);
        PipeBarrier<PIPE_V>();

        // Calc: xSum = sqrt(xSum)
        Sqrt(xSumLocal, xSumLocal, this->rowsPerBlock);
        PipeBarrier<PIPE_V>();

        // Calc: xSquare[this->rowsPerBlock, 8] = brc(xSum[this->rowsPerBlock, 1])
        int64_t block_count = (this->rowsPerBlock + NUM_EIGHT - 1) / NUM_EIGHT;
        Brcb(xSquareLocal, xSumLocal, block_count, {1, NUM_EIGHT});
        PipeBarrier<PIPE_V>();

        // Calc: xFp32Local = xFp32Local / xSquareLocal
        for (int64_t rowId = 0; rowId < this->rowsPerBlock; rowId++) {
            Div(xFp32Local[rowId * this->RMS_NORM_LENGTH], xFp32Local[rowId * this->RMS_NORM_LENGTH],
                xSquareLocal[rowId * NUM_EIGHT], mask, NUM_EIGHT, {1, 1, 0, NUM_EIGHT, NUM_EIGHT, 0});
        }
        PipeBarrier<PIPE_V>();

        // Cast gammaLocal to xSquareLocal (b16 -> fp32) [this->RMS_NORM_LENGTH]
        Cast(xSquareLocal, gammaLocal, RoundMode::CAST_NONE, this->RMS_NORM_LENGTH);
        PipeBarrier<PIPE_V>();
        this->inQueueGamma.FreeTensor(gammaLocal);

        // Calc: xFp32Local = xFp32Local * xSquareLocal
        for (int64_t rowId = 0; rowId < this->rowsPerBlock; rowId++) {
            Mul(xFp32Local[rowId * this->RMS_NORM_LENGTH], xFp32Local[rowId * this->RMS_NORM_LENGTH], xSquareLocal,
                this->RMS_NORM_LENGTH);
        }
        PipeBarrier<PIPE_V>();

        // Cast xFp32 to outLocal
        LocalTensor<KV_DTYPE> outLocal = this->outQueueV.template AllocTensor<KV_DTYPE>();
        if constexpr (std::is_same<KV_DTYPE, bfloat16_t>::value) {
            Cast(outLocal, xFp32Local, RoundMode::CAST_RINT, this->rmsNormElemPerSlice);
        } else if constexpr (std::is_same<KV_DTYPE, half>::value) {
            Cast(outLocal, xFp32Local, RoundMode::CAST_NONE, this->rmsNormElemPerSlice);
        }
        this->outQueueV.EnQue(outLocal);
    }

    template<typename T>
    __aicore__ inline void RmsNormV2(const LocalTensor<float>& xFp32Local, const LocalTensor<T>& gammaLocal)
    {
        // Calc: xSquare = xFp32 * xFp32, xSquare shape is [this->rowsPerBlock, this->RMS_NORM_LENGTH]
        LocalTensor<float> xSquareLocal = this->bufferXSquare.template Get<float>();
        Mul(xSquareLocal, xFp32Local, xFp32Local, this->rmsNormElemPerSlice);
        PipeBarrier<PIPE_V>();

        // Calc: xSquare[this->rowsPerBlock, 192]
        //          --> [this->rowsPerBlock, 96]
        //          --> [this->rowsPerBlock, 48]
        uint64_t mask[NUM_TWO] = {UINT64_MAX, UINT64_MAX};
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[NUM_FORTY_EIGHT],
            NUM_FORTY_EIGHT, NUM_TWO * this->rowsPerBlock,
            {1, 1, 1, NUM_SIX, NUM_TWELVE, NUM_TWELVE}); // [this->rowsPerBlock, 96]
        PipeBarrier<PIPE_V>();
        Add(xSquareLocal[0], xSquareLocal[0], xSquareLocal[NUM_FORTY_EIGHT],
            NUM_FORTY_EIGHT, this->rowsPerBlock,
            {1, 1, 1, NUM_SIX, NUM_TWELVE, NUM_TWELVE}); // [this->rowsPerBlock, 48]
        PipeBarrier<PIPE_V>();

        // Calc: xSum = [this->rowsPerBlock, 1]
        LocalTensor<float> xSumLocal = this->bufferSum.template Get<float>();
        WholeReduceSum(xSumLocal, xSquareLocal, NUM_FORTY_EIGHT, this->rowsPerBlock, 1, 1, NUM_SIX);
        PipeBarrier<PIPE_V>();

        // Calc: xSum = xSum * reciprocal
        Muls<float>(xSumLocal, xSumLocal, this->tilingData_->reciprocal, this->rowsPerBlock);
        PipeBarrier<PIPE_V>();

        // Calc: xSum = xSum + epsilon
        Adds<float>(xSumLocal, xSumLocal, this->tilingData_->epsilon, this->rowsPerBlock);
        PipeBarrier<PIPE_V>();

        // Calc: xSum = sqrt(xSum)
        Sqrt(xSumLocal, xSumLocal, this->rowsPerBlock);
        PipeBarrier<PIPE_V>();

        // Calc: xSquare[this->rowsPerBlock, 8] = brc(xSum[this->rowsPerBlock, 1])
        int64_t block_count = (this->rowsPerBlock + NUM_EIGHT - 1) / NUM_EIGHT;
        Brcb(xSquareLocal, xSumLocal, block_count, {1, NUM_EIGHT});
        PipeBarrier<PIPE_V>();

        // Calc: xFp32Local = xFp32Local / xSquareLocal
        for (int64_t rowId = 0; rowId < this->rowsPerBlock; rowId++) {
            Div(xFp32Local[rowId * this->RMS_NORM_LENGTH], xFp32Local[rowId * this->RMS_NORM_LENGTH],
                xSquareLocal[rowId * NUM_EIGHT], mask, (this->RMS_NORM_LENGTH / NUM_SIXTY_FOUR), {1, 1, 0, NUM_EIGHT, NUM_EIGHT, 0});
        }
        PipeBarrier<PIPE_V>();

        // Cast gammaLocal to xSquareLocal (b16 -> fp32) [this->RMS_NORM_LENGTH]
        Cast(xSquareLocal, gammaLocal, RoundMode::CAST_NONE, this->RMS_NORM_LENGTH);
        PipeBarrier<PIPE_V>();

        // Calc: xFp32Local = xFp32Local * xSquareLocal
        for (int64_t rowId = 0; rowId < this->rowsPerBlock; rowId++) {
            Mul(xFp32Local[rowId * this->RMS_NORM_LENGTH], xFp32Local[rowId * this->RMS_NORM_LENGTH], xSquareLocal,
                this->RMS_NORM_LENGTH);
        }
        PipeBarrier<PIPE_V>();

        // Cast xFp32 to outLocal
        LocalTensor<KV_DTYPE> xLocal = this->inQueueX.template AllocTensor<KV_DTYPE>();
        if constexpr (std::is_same<KV_DTYPE, bfloat16_t>::value) {
            Cast(xLocal, xFp32Local, RoundMode::CAST_RINT, this->rmsNormElemPerSlice);
        } else if constexpr (std::is_same<KV_DTYPE, half>::value) {
            Cast(xLocal, xFp32Local, RoundMode::CAST_NONE, this->rmsNormElemPerSlice);
        }
        this->inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void ScatterUpdateK(int64_t seqId)
    {
        LocalTensor<KV_DTYPE> outLocal = this->outQueueK.template DeQue<KV_DTYPE>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->ROPE_LENGTH * sizeof(KV_DTYPE)), 0, 0, 0};
        for (int64_t i = 0; i < this->rowsPerBlock; ++i) {
            if constexpr (isPagedAttention) {
                int64_t tokenId = GetBlockIdx() * this->rowsPerBlock * this->tilingData_->seqLength +
                                  i * this->tilingData_->seqLength + seqId;
                int64_t offset = this->indexGm(tokenId);
                PipeBarrier<PIPE_ALL>();
                if (offset < 0) {
                    continue;
                }
                DataCopyPad(this->kCacheGm[offset * this->ROPE_LENGTH], outLocal[i * this->ROPE_LENGTH], copyParams);
            } else {
                int64_t offset = this->indexGm(i * this->tilingData_->seqLength + seqId);
                PipeBarrier<PIPE_ALL>();
                if (offset < 0) {
                    continue;
                }
                DataCopyPad(
                    this->kCacheGm[i * this->tilingData_->cacheLength * this->ROPE_LENGTH + offset * this->ROPE_LENGTH],
                    outLocal[i * this->ROPE_LENGTH], copyParams);
            }
        }
        this->outQueueK.FreeTensor(outLocal);
    }

    __aicore__ inline void ScatterUpdateV(int64_t seqId)
    {
        LocalTensor<KV_DTYPE> outLocal = this->outQueueV.template DeQue<KV_DTYPE>();
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->RMS_NORM_LENGTH * sizeof(KV_DTYPE)), 0, 0, 0};
        for (int64_t i = 0; i < this->rowsPerBlock; ++i) {
            if constexpr (isPagedAttention) {
                int64_t tokenId = GetBlockIdx() * this->rowsPerBlock * this->tilingData_->seqLength +
                                  i * this->tilingData_->seqLength + seqId;
                int64_t offset = this->indexGm(tokenId);
                PipeBarrier<PIPE_ALL>();
                if (offset < 0) {
                    continue;
                }
                DataCopyPad(this->vCacheGm[offset * this->RMS_NORM_LENGTH], outLocal[i * this->RMS_NORM_LENGTH], copyParams);
            } else {
                int64_t offset = this->indexGm(i * this->tilingData_->seqLength + seqId);
                PipeBarrier<PIPE_ALL>();
                if (offset < 0) {
                    continue;
                }
                DataCopyPad(
                    this->vCacheGm[i * this->tilingData_->cacheLength * this->RMS_NORM_LENGTH + offset * this->RMS_NORM_LENGTH],
                    outLocal[i * this->RMS_NORM_LENGTH], copyParams);
            }
        }
        this->outQueueV.FreeTensor(outLocal);
    }

    template<typename T>
    __aicore__ inline void ScatterUpdate(
        int64_t nIdx, int64_t seqId, const GlobalTensor<T>& dst, const LocalTensor<T>& outLocal, int64_t kvLength)
    {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(kvLength * sizeof(KV_DTYPE)), 0, 0, 0};
        for (int64_t i = 0; i < this->rowsPerBlock; ++i) {
            if constexpr (isPagedAttention) {
                int64_t tokenId = GetBlockIdx() * this->rowsPerBlock * this->tilingData_->seqLength +
                                    i * this->tilingData_->seqLength + seqId;
                int64_t offset = this->indexGm(tokenId);
                PipeBarrier<PIPE_ALL>();
                if (offset < 0) {
                    continue;
                }
                int64_t dstOffset = offset * this->tilingData_->numHead * kvLength + nIdx * kvLength;
                DataCopyPad(dst[dstOffset], outLocal[i * kvLength], copyParams);
            } else {
                int64_t offset = this->indexGm(i * this->tilingData_->seqLength + seqId);
                PipeBarrier<PIPE_ALL>();
                if (offset < 0) {
                    continue;
                }
                int64_t dstOffset = i * this->tilingData_->numHead * this->tilingData_->cacheLength * kvLength +
                                    nIdx * this->tilingData_->cacheLength * kvLength +
                                    offset * kvLength;
                DataCopyPad(
                    dst[dstOffset], outLocal[i * kvLength], copyParams);
            }
        }
    }

private:
    #if _KV_RNRC_UNLEASH_D_SIZE
        int64_t ROPE_LENGTH_HALF{32};
    #else
        constexpr static int64_t ROPE_LENGTH_HALF = 32;
    #endif
    uint64_t methodMode = 0;
    constexpr static int64_t NUM_SIX = 6;
    constexpr static int64_t NUM_TWELVE = 12;
    constexpr static int64_t NUM_FORTY_EIGHT = 48;

    GlobalTensor<KV_DTYPE> vGm;
    TQue<QuePosition::VECIN, 1> CosSin; // MTE2 buffer for rope terms

    uint32_t rowsPerBlock{0};
    uint32_t rmsNormElemPerSlice{0};    // rmsNorm elements operated in a single process flow
    uint32_t ropeElemPerSlice{0};       // rope elements operated in a single process flow
    uint32_t kElemPerSlice{0};          // k cache elements written out per MTE call
    uint32_t vElemPerSlice{0};          // v cache elements written out per MTE call
};
} // namespace KvRmsNormRopeCache

#endif // _KV_RMS_NORM_ROPE_CACHE_B16_MTP_H_