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
 * \file kv_rms_norm_rope_cache_regbase_dk_split.h
 * \brief Dk切分模板：Dk过大UB放不下、Dv较小UB可全载的场景。
 *        RmsNorm侧(Dv)沿用全载模板的实现，Rope侧(Dk)在UB上按dkUbFactor分块循环。
 */

#ifndef KV_RMS_NORM_ROPE_CACHE_REGBASE_DK_SPLIT_H_
#define KV_RMS_NORM_ROPE_CACHE_REGBASE_DK_SPLIT_H_
#include "kv_rms_norm_rope_cache_regbase_base.h"

namespace KvRmsNormRopeCache {

using namespace AscendC;

template <typename T>
struct IsQuantCacheType {
    static constexpr bool value = IsSameType<T, int8_t>::value || IsSameType<T, hifloat8_t>::value ||
                                  IsSameType<T, fp8_e5m2_t>::value || IsSameType<T, fp8_e4m3fn_t>::value;
};

/*
 * Rope的VF尾块以maskAll无条件读满一个VL(见RopeBasicComputeVF中x的load)，
 * 每个输入buffer的最后一行都可能越过有效数据尾部，各自预留一段读余量。
 */
constexpr static int64_t ROPE_X_UB_HEADROOM_ELEM = CONST_TWO * VL_FP32;
constexpr static int64_t ROPE_COS_SIN_UB_HEADROOM_ELEM = VL_FP32;
constexpr static int64_t ROPE_K_SCALE_UB_HEADROOM_ELEM = VL_FP32;

template <typename T_KV, typename T_K_CACHE, typename T_V_CACHE>
class KvRmsNormRopeCacheRegbaseDkSplit : public KvRmsNormRopeCacheRegbase<T_KV, T_K_CACHE, T_V_CACHE>
{
    static constexpr bool K_QUANT = IsQuantCacheType<T_K_CACHE>::value;
    static constexpr bool V_QUANT = IsQuantCacheType<T_V_CACHE>::value;

public:
    __aicore__ inline KvRmsNormRopeCacheRegbaseDkSplit(
        TPipe* pipe, const KvRmsNormRopeCacheRegbaseDkSplitTilingData* tiling)
    {
        tilingData_ = tiling;
        pipe_ = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR kv, GM_ADDR gamma, GM_ADDR cos, GM_ADDR sin, GM_ADDR index, GM_ADDR k_cache, GM_ADDR v_cache,
        GM_ADDR k_scale, GM_ADDR v_scale, GM_ADDR k_offset, GM_ADDR v_offset, GM_ADDR k_out, GM_ADDR v_out)
    {
        // 获取分块操作中的单个块的大小。判断是否是最后一块，是最后一块，则等于剩余元素的数量，否则等于固定的单核处理的行数
        int64_t currentBlockFactor = tilingData_->blockFactor;
        if (this->blockIdx == (this->useCoreNum - 1)) {
            currentBlockFactor =
                tilingData_->batchSize * tilingData_->seqLength - (this->useCoreNum - 1) * tilingData_->blockFactor;
        }
        this->reciprocal = tilingData_->reciprocal;
        this->epsilon = tilingData_->epsilon;
        this->dv = tilingData_->dv;
        this->dvAlign = tilingData_->dvAlign;
        this->dvB8Align = tilingData_->dvB8Align;
        this->dk = tilingData_->dk;
        this->ubFactor = tilingData_->ubFactor;

        halfDk = tilingData_->halfDk;
        dkUbFactor = tilingData_->dkUbFactor;
        dkUbFactorAlign = tilingData_->dkUbFactorAlign;
        dkUbFactorB8Align = tilingData_->dkUbFactorB8Align;
        halfDkUbFactor = tilingData_->halfDkUbFactor;
        halfDkUbFactorAlign = tilingData_->halfDkUbFactorAlign;
        dkLoop = tilingData_->dkLoop;
        dkTail = tilingData_->dkTail;
        blockElem = BLOCK_SIZE / sizeof(T_KV);

        // NZ参数
        dk0 = BLOCK_SIZE / sizeof(T_K_CACHE);
        dk1 = this->dk / dk0;
        dv0 = BLOCK_SIZE / sizeof(T_V_CACHE);
        dv1 = this->dv / dv0;

        copyOutVParams.blockCount = 1;
        copyOutVParams.blockLen = tilingData_->dv * sizeof(T_V_CACHE);
        copyOutVParams.srcStride = 0;
        copyOutVParams.dstStride = 0;

        copyOutVParamsNz.blockCount = dv1;
        copyOutVParamsNz.blockLen = dv0 * sizeof(T_V_CACHE);
        copyOutVParamsNz.srcStride = 0;
        copyOutVParamsNz.dstStride = (tilingData_->blockSize - 1) * dv0 * sizeof(T_V_CACHE);

        // ubLoop: 表示需要多少个子块来覆盖singleA大小的数据
        this->ubLoop = (currentBlockFactor + this->ubFactor - 1) / this->ubFactor;
        // ubLoop：最后一块
        this->ubTail = currentBlockFactor - (this->ubLoop - 1) * this->ubFactor;
        dKV = (tilingData_->dk + tilingData_->dv);
        vScaleCopyLen = tilingData_->vScaleType == 1 ? 1 : tilingData_->dv;
        vOffsetCopyLen = tilingData_->vOffsetType == 1 ? 1 : tilingData_->dv;

        this->dvLoopCount = ops::CeilDiv(tilingData_->dv, static_cast<int64_t>(VL_FP32));

        // init global memory
        this->kvGm.SetGlobalBuffer((__gm__ T_KV*)kv + this->blockIdx * tilingData_->blockFactor * dKV);
        this->gammaGm.SetGlobalBuffer((__gm__ T_KV*)gamma);
        this->cosGm.SetGlobalBuffer((__gm__ T_KV*)cos);
        this->sinGm.SetGlobalBuffer((__gm__ T_KV*)sin);
        this->indexGm.SetGlobalBuffer((__gm__ int64_t*)index);
        this->kCacheGm.SetGlobalBuffer((__gm__ T_K_CACHE*)k_cache);
        this->vCacheGm.SetGlobalBuffer((__gm__ T_V_CACHE*)v_cache);
        this->kScaleGm.SetGlobalBuffer((__gm__ float*)k_scale);
        this->vScaleGm.SetGlobalBuffer((__gm__ float*)v_scale);
        this->kOffsetGm.SetGlobalBuffer((__gm__ float*)k_offset);
        this->vOffsetGm.SetGlobalBuffer((__gm__ float*)v_offset);
        this->kOutGm.SetGlobalBuffer((__gm__ T_KV*)k_out + this->blockIdx * tilingData_->blockFactor * this->dk);
        this->vOutGm.SetGlobalBuffer((__gm__ T_KV*)v_out + this->blockIdx * tilingData_->blockFactor * this->dv);

        // init pipe
        pipe_->InitBuffer(inQueueGamma, BUFFER_COUNT_SINGLE, tilingData_->dvAlign * sizeof(T_KV));
        pipe_->InitBuffer(
            inQueueCosSin, BUFFER_COUNT_DOUBLE,
            this->ubFactor * COS_SIN_CHANNEL_COUNT * halfDkUbFactorAlign * sizeof(T_KV) +
                ROPE_COS_SIN_UB_HEADROOM_ELEM * sizeof(T_KV));
        pipe_->InitBuffer(
            inQueueX, BUFFER_COUNT_DOUBLE,
            this->ubFactor * tilingData_->inUbSize + ROPE_X_UB_HEADROOM_ELEM * sizeof(T_KV));
        pipe_->InitBuffer(outQueue, BUFFER_COUNT_DOUBLE, this->ubFactor * tilingData_->outUbSize);
        pipe_->InitBuffer(wsBuffer0, WORKSPACE_FLOAT_VECTOR_COUNT * VL_FP32 * sizeof(float));
        pipe_->InitBuffer(wsBuffer1, this->ubFactor * tilingData_->rmsNormWspSize);
        if constexpr (K_QUANT) {
            pipe_->InitBuffer(
                kScaleOffsetQueue, BUFFER_COUNT_SINGLE,
                K_SCALE_OFFSET_CHANNEL_COUNT * halfDkUbFactorAlign * sizeof(float) +
                    ROPE_K_SCALE_UB_HEADROOM_ELEM * sizeof(float));
        }
        if constexpr (V_QUANT) {
            pipe_->InitBuffer(
                vScaleOffsetQueue, BUFFER_COUNT_SINGLE,
                V_SCALE_OFFSET_CHANNEL_COUNT * tilingData_->dvAlign * sizeof(float));
        }
    }

    /*
    ub->gm 拷出一段列区间：srcUb 行间距为 srcRowStride，每行拷出 copyLen 个元素
    行间空洞不是32B整数倍时退化为逐行搬运
    */
    template <typename T>
    __aicore__ inline void CopyOutRowsSeg(
        const GlobalTensor<T>& dstGm, const LocalTensor<T>& srcUb, int64_t rows, int64_t dstLineLength,
        int64_t copyLen, int64_t srcRowStride)
    {
        int64_t gapBytes = (srcRowStride - copyLen) * sizeof(T);
        DataCopyExtParams copyOutParams;
        if (gapBytes % BLOCK_SIZE == 0) {
            copyOutParams.blockCount = rows;
            copyOutParams.blockLen = copyLen * sizeof(T);
            copyOutParams.srcStride = gapBytes / BLOCK_SIZE;
            copyOutParams.dstStride = (dstLineLength - copyLen) * sizeof(T);
            DataCopyPad(dstGm, srcUb, copyOutParams);
            return;
        }
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = copyLen * sizeof(T);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
        for (int64_t i = 0; i < rows; i++) {
            DataCopyPad(dstGm[i * dstLineLength], srcUb[i * srcRowStride], copyOutParams);
        }
    }

    /*
    把一行内 [halfChunkStart, halfChunkStart + halfChunk) 这段实部/虚部写进 kCache
    connected 排布下实部与虚部在GM上相距 halfDk，因此每行需要两次搬运
    */
    __aicore__ inline void CopyOutKCacheSeg(
        const LocalTensor<T_K_CACHE>& src, int64_t ubRowOffset, int64_t gmRowOffset, int64_t halfChunkStart,
        int64_t halfChunk)
    {
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = halfChunk * sizeof(T_K_CACHE);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
        DataCopyPad(this->kCacheGm[gmRowOffset + halfChunkStart], src[ubRowOffset], copyOutParams);
        DataCopyPad(
            this->kCacheGm[gmRowOffset + halfDk + halfChunkStart], src[ubRowOffset + halfDkUbFactor], copyOutParams);
    }

    /*
    NZ排布下的一段列区间拷出：GM上每dk0个元素为一个内轴block，block间距为blockSize*dk0
    halfChunkStart与halfDk都保证是dk0的整数倍(tiling校验)，尾块的halfChunk可能不足dk0的整数倍
    */
    __aicore__ inline void CopyOutKCacheSegNz(
        const LocalTensor<T_K_CACHE>& src, int64_t ubRowOffset, int64_t gmPageBase, int64_t halfChunkStart,
        int64_t halfChunk)
    {
        int64_t fullBlocks = halfChunk / dk0;
        int64_t remain = halfChunk - fullBlocks * dk0;
        int64_t gmFront = gmPageBase + (halfChunkStart / dk0) * tilingData_->blockSize * dk0;
        int64_t gmBack = gmPageBase + ((halfDk + halfChunkStart) / dk0) * tilingData_->blockSize * dk0;
        int64_t ubBack = ubRowOffset + halfDkUbFactor;

        DataCopyExtParams copyOutParams;
        if (fullBlocks > 0) {
            copyOutParams.blockCount = fullBlocks;
            copyOutParams.blockLen = dk0 * sizeof(T_K_CACHE);
            copyOutParams.srcStride = 0;
            copyOutParams.dstStride = (tilingData_->blockSize - 1) * dk0 * sizeof(T_K_CACHE);
            DataCopyPad(this->kCacheGm[gmFront], src[ubRowOffset], copyOutParams);
            DataCopyPad(this->kCacheGm[gmBack], src[ubBack], copyOutParams);
        }
        if (remain > 0) {
            int64_t blockGm = fullBlocks * tilingData_->blockSize * dk0;
            int64_t blockUb = fullBlocks * dk0;
            copyOutParams.blockCount = 1;
            copyOutParams.blockLen = remain * sizeof(T_K_CACHE);
            copyOutParams.srcStride = 0;
            copyOutParams.dstStride = 0;
            DataCopyPad(this->kCacheGm[gmFront + blockGm], src[ubRowOffset + blockUb], copyOutParams);
            DataCopyPad(this->kCacheGm[gmBack + blockGm], src[ubBack + blockUb], copyOutParams);
        }
    }

    __aicore__ inline void ScatterUpdateK(
        const LocalTensor<T_K_CACHE>& src, int64_t rows, int64_t startIdx, int64_t ubRowStride,
        int64_t halfChunkStart, int64_t halfChunk)
    {
        eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        int64_t seqIndex = 0;
        int64_t batchId = 0;
        for (int64_t i = 0; i < rows; i++) {
            seqIndex = this->indexGm(startIdx + i);
            SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            if (seqIndex < 0) {
                continue;
            }
            int64_t ubRowOffset = i * ubRowStride;
            if (tilingData_->cacheMode == NORM_CACHE_MODE) {
                batchId = (startIdx + i) / tilingData_->seqLength;
                int64_t gmRowOffset = (batchId * tilingData_->cacheLength + seqIndex) * tilingData_->dk;
                CopyOutKCacheSeg(src, ubRowOffset, gmRowOffset, halfChunkStart, halfChunk);
            } else if (tilingData_->cacheMode == PA_CACHE_MODE) {
                CopyOutKCacheSeg(src, ubRowOffset, seqIndex * tilingData_->dk, halfChunkStart, halfChunk);
            } else if (tilingData_->cacheMode == PA_NZ_CACHE_MODE) {
                int64_t pageId = seqIndex / tilingData_->blockSize;
                int64_t tokenOffsetInCurrentPage = seqIndex % tilingData_->blockSize;
                int64_t gmPageBase = (pageId * dk1 * tilingData_->blockSize + tokenOffsetInCurrentPage) * dk0;
                CopyOutKCacheSegNz(src, ubRowOffset, gmPageBase, halfChunkStart, halfChunk);
            } else {
                return;
            }
        }
    }

    __aicore__ inline void ScatterBlkUpdateK(
        const LocalTensor<T_K_CACHE>& src, int64_t rows, int64_t startIdx, int64_t ubRowStride,
        int64_t halfChunkStart, int64_t halfChunk)
    {
        eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        int64_t indexPageLength = (tilingData_->seqLength + tilingData_->blockSize - 1) / tilingData_->blockSize;
        for (int64_t i = 0; i < rows; i++) {
            int64_t tokenId = startIdx + i;
            int64_t batchId = tokenId / tilingData_->seqLength;
            int64_t tokenIdInCurrentBatch = tokenId % tilingData_->seqLength;
            int64_t indexPageId = tokenIdInCurrentBatch / tilingData_->blockSize;
            int64_t indexOffset = batchId * indexPageLength + indexPageId;
            int64_t pageOffset = this->indexGm(indexOffset);
            SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            if (pageOffset < 0) {
                continue;
            }
            int64_t tokenOffsetInCurrentPage = tokenIdInCurrentBatch % tilingData_->blockSize;
            int64_t pageId = pageOffset / tilingData_->blockSize;
            int64_t ubRowOffset = i * ubRowStride;
            if (tilingData_->cacheMode == PA_BLK_NZ_CACHE_MODE) {
                int64_t gmPageBase = (pageId * dk1 * tilingData_->blockSize + tokenOffsetInCurrentPage) * dk0;
                CopyOutKCacheSegNz(src, ubRowOffset, gmPageBase, halfChunkStart, halfChunk);
            } else {
                int64_t gmRowOffset =
                    (pageId * tilingData_->blockSize + tokenOffsetInCurrentPage) * tilingData_->dk;
                CopyOutKCacheSeg(src, ubRowOffset, gmRowOffset, halfChunkStart, halfChunk);
            }
        }
    }

    __aicore__ inline void ScatterUpdateV(int64_t rows, int64_t startIdx)
    {
        DataCopyExtParams copyOutParams =
            tilingData_->cacheMode == PA_NZ_CACHE_MODE ? copyOutVParamsNz : copyOutVParams;
        eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        int64_t seqIndex = 0;
        int64_t batchId = 0;
        int64_t gmOffset = 0;
        for (int64_t i = 0; i < rows; i++) {
            seqIndex = this->indexGm(startIdx + i);
            SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            if (seqIndex >= 0) {
                if (tilingData_->cacheMode == NORM_CACHE_MODE) {
                    batchId = (startIdx + i) / tilingData_->seqLength;
                    gmOffset = (batchId * tilingData_->cacheLength + seqIndex) * tilingData_->dv;
                } else if (tilingData_->cacheMode == PA_CACHE_MODE) {
                    gmOffset = seqIndex * tilingData_->dv;
                } else if (tilingData_->cacheMode == PA_NZ_CACHE_MODE) {
                    int64_t pageId = seqIndex / tilingData_->blockSize;
                    int64_t tokenOffsetInCurrentPage = seqIndex % tilingData_->blockSize;
                    gmOffset = (pageId * dv1 * tilingData_->blockSize + tokenOffsetInCurrentPage) * dv0;
                } else {
                    return;
                }
                if constexpr (V_QUANT) {
                    DataCopyPad(this->vCacheGm[gmOffset], vQuantLocal[i * tilingData_->dvB8Align], copyOutParams);
                } else {
                    DataCopyPad(this->vCacheGm[gmOffset], vOutLocal[i * tilingData_->dvAlign], copyOutParams);
                }
            }
        }
    }

    __aicore__ inline void ScatterBlkUpdateV(int64_t rows, int64_t startIdx)
    {
        DataCopyExtParams copyOutParams =
            tilingData_->cacheMode == PA_BLK_NZ_CACHE_MODE ? copyOutVParamsNz : copyOutVParams;
        eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        int64_t indexPageLength = (tilingData_->seqLength + tilingData_->blockSize - 1) / tilingData_->blockSize;
        int64_t gmOffset = 0;
        for (int64_t i = 0; i < rows; i++) {
            int64_t tokenId = startIdx + i;
            int64_t batchId = tokenId / tilingData_->seqLength;
            int64_t tokenIdInCurrentBatch = tokenId % tilingData_->seqLength;
            int64_t indexPageId = tokenIdInCurrentBatch / tilingData_->blockSize;
            int64_t indexOffset = batchId * indexPageLength + indexPageId;
            int64_t pageOffset = this->indexGm(indexOffset);
            SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
            int64_t tokenOffsetInCurrentPage = tokenIdInCurrentBatch % tilingData_->blockSize;
            int64_t pageId = pageOffset / tilingData_->blockSize;
            if (pageOffset >= 0) {
                if (tilingData_->cacheMode == PA_BLK_NZ_CACHE_MODE) {
                    gmOffset = (pageId * dv1 * tilingData_->blockSize + tokenOffsetInCurrentPage) * dv0;
                } else {
                    gmOffset = (pageId * tilingData_->blockSize + tokenOffsetInCurrentPage) * tilingData_->dv;
                }
                if constexpr (V_QUANT) {
                    DataCopyPad(this->vCacheGm[gmOffset], vQuantLocal[i * tilingData_->dvB8Align], copyOutParams);
                } else {
                    DataCopyPad(this->vCacheGm[gmOffset], vOutLocal[i * tilingData_->dvAlign], copyOutParams);
                }
            }
        }
    }

    __aicore__ inline void RopeBasicComputeVF(
        __local_mem__ T_KV* x, AscendC::MicroAPI::RegTensor<float>& x_0, AscendC::MicroAPI::RegTensor<float>& x_1,
        AscendC::MicroAPI::RegTensor<float>& cos_0, AscendC::MicroAPI::RegTensor<float>& cos_1,
        AscendC::MicroAPI::RegTensor<float>& sin_0, AscendC::MicroAPI::RegTensor<float>& sin_1,
        __local_mem__ T_KV* realCos, __local_mem__ T_KV* imgCos, __local_mem__ T_KV* realSin,
        __local_mem__ T_KV* imgSin, __local_mem__ float* ws, uint32_t xOffset, uint32_t cosSinOffset, uint16_t rowId,
        uint16_t colId, AscendC::MicroAPI::MaskReg mask, AscendC::MicroAPI::MaskReg maskAll)
    {
        // cast to float32
        LoadTensorForDtypeT<T_KV>(x, x_0, maskAll, colId * CONST_TWO * VL_FP32 + rowId * xOffset);
        LoadTensorForDtypeT<T_KV>(x, x_1, maskAll, (colId * CONST_TWO + CONST_ONE) * VL_FP32 + rowId * xOffset);
        StoreTensorForDtypeTOut<float>(ws, x_0, maskAll, 0);
        StoreTensorForDtypeTOut<float>(ws, x_1, maskAll, VL_FP32);
        AscendC::MicroAPI::LocalMemBar<AscendC::MicroAPI::MemType::VEC_STORE, AscendC::MicroAPI::MemType::VEC_LOAD>();
        // get x
        DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_DINTLV_B32>(x_0, x_1, ((__local_mem__ float*)(ws)));
        LoadTensorForDtypeT(realCos, cos_0, mask, colId * VL_FP32 + rowId * cosSinOffset);
        LoadTensorForDtypeT(imgCos, cos_1, mask, colId * VL_FP32 + rowId * cosSinOffset);
        LoadTensorForDtypeT(realSin, sin_0, mask, colId * VL_FP32 + rowId * cosSinOffset);
        LoadTensorForDtypeT(imgSin, sin_1, mask, colId * VL_FP32 + rowId * cosSinOffset);
        AscendC::MicroAPI::Mul(cos_0, x_0, cos_0, mask);
        AscendC::MicroAPI::Mul(cos_1, x_1, cos_1, mask);
        AscendC::MicroAPI::Muls(x_1, x_1, -1, mask);
        AscendC::MicroAPI::Mul(sin_0, x_1, sin_0, mask);
        AscendC::MicroAPI::Mul(sin_1, x_0, sin_1, mask);
        AscendC::MicroAPI::Add(sin_0, cos_0, sin_0, mask);
        AscendC::MicroAPI::Add(sin_1, cos_1, sin_1, mask);
    }

    __aicore__ inline void RopeVF(
        __local_mem__ T_KV* outFront, __local_mem__ T_KV* outBack, __local_mem__ T_KV* x, __local_mem__ T_KV* realCos,
        __local_mem__ T_KV* imgCos, __local_mem__ T_KV* realSin, __local_mem__ T_KV* imgSin, __local_mem__ float* ws,
        uint16_t row, uint16_t colLoopCount, uint32_t colTail, uint32_t colCosSin, uint32_t xOffset,
        uint32_t outOffset, uint32_t cosSinOffset)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> x_0, x_1, cos_0, cos_1, sin_0, sin_1;
            AscendC::MicroAPI::UnalignReg UReg0;
            AscendC::MicroAPI::UnalignReg UReg1;
            AscendC::MicroAPI::MaskReg mask;
            AscendC::MicroAPI::MaskReg maskAll =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            for (uint16_t rowId = 0; rowId < row; rowId++) {
                uint32_t sreg = colCosSin;
                // 跨行地址重刷新
                __local_mem__ T_KV* out0 = outFront + rowId * outOffset;
                __local_mem__ T_KV* out1 = outBack + rowId * outOffset;
                // 整块
                for (uint16_t i = 0; i < colLoopCount; i++) {
                    mask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                    RopeBasicComputeVF(
                        x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws, xOffset,
                        cosSinOffset, rowId, i, mask, maskAll);
                    StoreUnAlignOneTensor<T_KV>(out0, sin_0, UReg0, mask, VL_FP32);
                    StoreUnAlignOneTensor<T_KV>(out1, sin_1, UReg1, mask, VL_FP32);
                }
                // 尾块
                mask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                RopeBasicComputeVF(
                    x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws, xOffset,
                    cosSinOffset, rowId, colLoopCount, mask, maskAll);
                StoreUnAlignOneTensor<T_KV>(out0, sin_0, UReg0, mask, colTail);
                StoreUnAlignOneTensor<T_KV>(out1, sin_1, UReg1, mask, colTail);
                AscendC::MicroAPI::DataCopyUnAlignPost(out0, UReg0, 0);
                AscendC::MicroAPI::DataCopyUnAlignPost(out1, UReg1, 0);
            }
        }
    }

    __aicore__ inline void RopeSymQuantVF(
        __local_mem__ T_K_CACHE* quantFront, __local_mem__ T_K_CACHE* quantBack, __local_mem__ T_KV* x,
        __local_mem__ T_KV* realCos, __local_mem__ T_KV* imgCos, __local_mem__ T_KV* realSin,
        __local_mem__ T_KV* imgSin, __local_mem__ float* realKScale, __local_mem__ float* imgKScale,
        __local_mem__ float* ws, uint16_t row, uint16_t colLoopCount, uint32_t colTail, uint32_t colCosSin,
        uint32_t xOffset, uint32_t cosSinOffset, uint32_t quantOutOffset)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> x_0, x_1, cos_0, cos_1, sin_0, sin_1, scale_0, scale_1;
            AscendC::MicroAPI::UnalignReg UReg2, UReg3;
            AscendC::MicroAPI::MaskReg mask;
            AscendC::MicroAPI::MaskReg maskAll =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            for (uint16_t rowId = 0; rowId < row; rowId++) {
                uint32_t sreg = colCosSin;
                __local_mem__ T_K_CACHE* out2 = quantFront + rowId * quantOutOffset;
                __local_mem__ T_K_CACHE* out3 = quantBack + rowId * quantOutOffset;
                for (uint16_t i = 0; i < colLoopCount; i++) {
                    mask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                    RopeBasicComputeVF(
                        x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws, xOffset,
                        cosSinOffset, rowId, i, mask, maskAll);
                    LoadTensorForDtypeT(realKScale, scale_0, mask, i * VL_FP32);
                    LoadTensorForDtypeT(imgKScale, scale_1, mask, i * VL_FP32);
                    AscendC::MicroAPI::Mul(scale_0, sin_0, scale_0, mask);
                    AscendC::MicroAPI::Mul(scale_1, sin_1, scale_1, mask);
                    StoreUnAlignOneTensor<T_K_CACHE>(out2, scale_0, UReg2, mask, VL_FP32);
                    StoreUnAlignOneTensor<T_K_CACHE>(out3, scale_1, UReg3, mask, VL_FP32);
                }
                mask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                RopeBasicComputeVF(
                    x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws, xOffset,
                    cosSinOffset, rowId, colLoopCount, mask, maskAll);
                LoadTensorForDtypeT(realKScale, scale_0, mask, colLoopCount * VL_FP32);
                LoadTensorForDtypeT(imgKScale, scale_1, mask, colLoopCount * VL_FP32);
                AscendC::MicroAPI::Mul(scale_0, sin_0, scale_0, mask);
                AscendC::MicroAPI::Mul(scale_1, sin_1, scale_1, mask);
                StoreUnAlignOneTensor<T_K_CACHE>(out2, scale_0, UReg2, mask, colTail);
                StoreUnAlignOneTensor<T_K_CACHE>(out3, scale_1, UReg3, mask, colTail);
                AscendC::MicroAPI::DataCopyUnAlignPost(out2, UReg2, 0);
                AscendC::MicroAPI::DataCopyUnAlignPost(out3, UReg3, 0);
            }
        }
    }

    __aicore__ inline void RopeAsymQuantVF(
        __local_mem__ T_K_CACHE* quantFront, __local_mem__ T_K_CACHE* quantBack, __local_mem__ T_KV* x,
        __local_mem__ T_KV* realCos, __local_mem__ T_KV* imgCos, __local_mem__ T_KV* realSin,
        __local_mem__ T_KV* imgSin, __local_mem__ float* realKScale, __local_mem__ float* imgKScale,
        __local_mem__ float* realKOffset, __local_mem__ float* imgKOffset, __local_mem__ float* ws, uint16_t row,
        uint16_t colLoopCount, uint32_t colTail, uint32_t colCosSin, uint32_t xOffset, uint32_t cosSinOffset,
        uint32_t quantOutOffset)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> x_0, x_1, cos_0, cos_1, sin_0, sin_1, scale_0, scale_1, offset_0,
                offset_1;
            AscendC::MicroAPI::UnalignReg UReg2, UReg3;
            AscendC::MicroAPI::MaskReg mask;
            AscendC::MicroAPI::MaskReg maskAll =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            for (uint16_t rowId = 0; rowId < row; rowId++) {
                uint32_t sreg = colCosSin;
                __local_mem__ T_K_CACHE* out2 = quantFront + rowId * quantOutOffset;
                __local_mem__ T_K_CACHE* out3 = quantBack + rowId * quantOutOffset;
                for (uint16_t i = 0; i < colLoopCount; i++) {
                    mask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                    RopeBasicComputeVF(
                        x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws, xOffset,
                        cosSinOffset, rowId, i, mask, maskAll);
                    LoadTensorForDtypeT(realKScale, scale_0, mask, i * VL_FP32);
                    LoadTensorForDtypeT(imgKScale, scale_1, mask, i * VL_FP32);
                    AscendC::MicroAPI::Mul(scale_0, sin_0, scale_0, mask);
                    AscendC::MicroAPI::Mul(scale_1, sin_1, scale_1, mask);
                    LoadTensorForDtypeT(realKOffset, offset_0, mask, i * VL_FP32);
                    LoadTensorForDtypeT(imgKOffset, offset_1, mask, i * VL_FP32);
                    AscendC::MicroAPI::Add(offset_0, offset_0, scale_0, mask);
                    AscendC::MicroAPI::Add(offset_1, offset_1, scale_1, mask);
                    StoreUnAlignOneTensor<T_K_CACHE>(out2, offset_0, UReg2, mask, VL_FP32);
                    StoreUnAlignOneTensor<T_K_CACHE>(out3, offset_1, UReg3, mask, VL_FP32);
                }
                mask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                RopeBasicComputeVF(
                    x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws, xOffset,
                    cosSinOffset, rowId, colLoopCount, mask, maskAll);
                LoadTensorForDtypeT(realKScale, scale_0, mask, colLoopCount * VL_FP32);
                LoadTensorForDtypeT(imgKScale, scale_1, mask, colLoopCount * VL_FP32);
                AscendC::MicroAPI::Mul(scale_0, sin_0, scale_0, mask);
                AscendC::MicroAPI::Mul(scale_1, sin_1, scale_1, mask);
                LoadTensorForDtypeT(realKOffset, offset_0, mask, colLoopCount * VL_FP32);
                LoadTensorForDtypeT(imgKOffset, offset_1, mask, colLoopCount * VL_FP32);
                AscendC::MicroAPI::Add(offset_0, offset_0, scale_0, mask);
                AscendC::MicroAPI::Add(offset_1, offset_1, scale_1, mask);
                StoreUnAlignOneTensor<T_K_CACHE>(out2, offset_0, UReg2, mask, colTail);
                StoreUnAlignOneTensor<T_K_CACHE>(out3, offset_1, UReg3, mask, colTail);
                AscendC::MicroAPI::DataCopyUnAlignPost(out2, UReg2, 0);
                AscendC::MicroAPI::DataCopyUnAlignPost(out3, UReg3, 0);
            }
        }
    }

    __aicore__ inline void RopeSymQuantWithKvVF(
        __local_mem__ T_KV* outFront, __local_mem__ T_KV* outBack, __local_mem__ T_K_CACHE* quantFront,
        __local_mem__ T_K_CACHE* quantBack, __local_mem__ T_KV* x, __local_mem__ T_KV* realCos,
        __local_mem__ T_KV* imgCos, __local_mem__ T_KV* realSin, __local_mem__ T_KV* imgSin,
        __local_mem__ float* realKScale, __local_mem__ float* imgKScale, __local_mem__ float* ws, uint16_t row,
        uint16_t colLoopCount, uint32_t colTail, uint32_t colCosSin, uint32_t xOffset, uint32_t outOffset,
        uint32_t cosSinOffset, uint32_t quantOutOffset)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> x_0, x_1, cos_0, cos_1, sin_0, sin_1, scale_0, scale_1;
            AscendC::MicroAPI::UnalignReg UReg0, UReg1, UReg2, UReg3;
            AscendC::MicroAPI::MaskReg mask;
            AscendC::MicroAPI::MaskReg maskAll =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            for (uint16_t rowId = 0; rowId < row; rowId++) {
                uint32_t sreg = colCosSin;
                __local_mem__ T_KV* out0 = outFront + rowId * outOffset;
                __local_mem__ T_KV* out1 = outBack + rowId * outOffset;
                __local_mem__ T_K_CACHE* out2 = quantFront + rowId * quantOutOffset;
                __local_mem__ T_K_CACHE* out3 = quantBack + rowId * quantOutOffset;
                for (uint16_t i = 0; i < colLoopCount; i++) {
                    mask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                    RopeBasicComputeVF(
                        x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws, xOffset,
                        cosSinOffset, rowId, i, mask, maskAll);
                    StoreUnAlignOneTensor<T_KV>(out0, sin_0, UReg0, mask, VL_FP32);
                    StoreUnAlignOneTensor<T_KV>(out1, sin_1, UReg1, mask, VL_FP32);
                    LoadTensorForDtypeT(realKScale, scale_0, mask, i * VL_FP32);
                    LoadTensorForDtypeT(imgKScale, scale_1, mask, i * VL_FP32);
                    AscendC::MicroAPI::Mul(scale_0, sin_0, scale_0, mask);
                    AscendC::MicroAPI::Mul(scale_1, sin_1, scale_1, mask);
                    StoreUnAlignOneTensor<T_K_CACHE>(out2, scale_0, UReg2, mask, VL_FP32);
                    StoreUnAlignOneTensor<T_K_CACHE>(out3, scale_1, UReg3, mask, VL_FP32);
                }
                mask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                RopeBasicComputeVF(
                    x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws, xOffset,
                    cosSinOffset, rowId, colLoopCount, mask, maskAll);
                StoreUnAlignOneTensor<T_KV>(out0, sin_0, UReg0, mask, colTail);
                StoreUnAlignOneTensor<T_KV>(out1, sin_1, UReg1, mask, colTail);
                LoadTensorForDtypeT(realKScale, scale_0, mask, colLoopCount * VL_FP32);
                LoadTensorForDtypeT(imgKScale, scale_1, mask, colLoopCount * VL_FP32);
                AscendC::MicroAPI::Mul(scale_0, sin_0, scale_0, mask);
                AscendC::MicroAPI::Mul(scale_1, sin_1, scale_1, mask);
                StoreUnAlignOneTensor<T_K_CACHE>(out2, scale_0, UReg2, mask, colTail);
                StoreUnAlignOneTensor<T_K_CACHE>(out3, scale_1, UReg3, mask, colTail);
                AscendC::MicroAPI::DataCopyUnAlignPost(out0, UReg0, 0);
                AscendC::MicroAPI::DataCopyUnAlignPost(out1, UReg1, 0);
                AscendC::MicroAPI::DataCopyUnAlignPost(out2, UReg2, 0);
                AscendC::MicroAPI::DataCopyUnAlignPost(out3, UReg3, 0);
            }
        }
    }

    __aicore__ inline void RopeAsymQuantWithKvVF(
        __local_mem__ T_KV* outFront, __local_mem__ T_KV* outBack, __local_mem__ T_K_CACHE* quantFront,
        __local_mem__ T_K_CACHE* quantBack, __local_mem__ T_KV* x, __local_mem__ T_KV* realCos,
        __local_mem__ T_KV* imgCos, __local_mem__ T_KV* realSin, __local_mem__ T_KV* imgSin,
        __local_mem__ float* realKScale, __local_mem__ float* imgKScale, __local_mem__ float* realKOffset,
        __local_mem__ float* imgKOffset, __local_mem__ float* ws, uint16_t row, uint16_t colLoopCount,
        uint32_t colTail, uint32_t colCosSin, uint32_t xOffset, uint32_t outOffset, uint32_t cosSinOffset,
        uint32_t quantOutOffset)
    {
        __VEC_SCOPE__
        {
            AscendC::MicroAPI::RegTensor<float> x_0, x_1, cos_0, cos_1, sin_0, sin_1, scale_0, scale_1, offset_0,
                offset_1;
            AscendC::MicroAPI::UnalignReg UReg0, UReg1, UReg2, UReg3;
            AscendC::MicroAPI::MaskReg mask;
            AscendC::MicroAPI::MaskReg maskAll =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            for (uint16_t rowId = 0; rowId < row; rowId++) {
                uint32_t sreg = colCosSin;
                __local_mem__ T_KV* out0 = outFront + rowId * outOffset;
                __local_mem__ T_KV* out1 = outBack + rowId * outOffset;
                __local_mem__ T_K_CACHE* out2 = quantFront + rowId * quantOutOffset;
                __local_mem__ T_K_CACHE* out3 = quantBack + rowId * quantOutOffset;
                for (uint16_t i = 0; i < colLoopCount; i++) {
                    mask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                    RopeBasicComputeVF(
                        x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws, xOffset,
                        cosSinOffset, rowId, i, mask, maskAll);
                    StoreUnAlignOneTensor<T_KV>(out0, sin_0, UReg0, mask, VL_FP32);
                    StoreUnAlignOneTensor<T_KV>(out1, sin_1, UReg1, mask, VL_FP32);
                    LoadTensorForDtypeT(realKScale, scale_0, mask, i * VL_FP32);
                    LoadTensorForDtypeT(imgKScale, scale_1, mask, i * VL_FP32);
                    AscendC::MicroAPI::Mul(scale_0, sin_0, scale_0, mask);
                    AscendC::MicroAPI::Mul(scale_1, sin_1, scale_1, mask);
                    LoadTensorForDtypeT(realKOffset, offset_0, mask, i * VL_FP32);
                    LoadTensorForDtypeT(imgKOffset, offset_1, mask, i * VL_FP32);
                    AscendC::MicroAPI::Add(offset_0, offset_0, scale_0, mask);
                    AscendC::MicroAPI::Add(offset_1, offset_1, scale_1, mask);
                    StoreUnAlignOneTensor<T_K_CACHE>(out2, offset_0, UReg2, mask, VL_FP32);
                    StoreUnAlignOneTensor<T_K_CACHE>(out3, offset_1, UReg3, mask, VL_FP32);
                }
                mask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                RopeBasicComputeVF(
                    x, x_0, x_1, cos_0, cos_1, sin_0, sin_1, realCos, imgCos, realSin, imgSin, ws, xOffset,
                    cosSinOffset, rowId, colLoopCount, mask, maskAll);
                StoreUnAlignOneTensor<T_KV>(out0, sin_0, UReg0, mask, colTail);
                StoreUnAlignOneTensor<T_KV>(out1, sin_1, UReg1, mask, colTail);
                LoadTensorForDtypeT(realKScale, scale_0, mask, colLoopCount * VL_FP32);
                LoadTensorForDtypeT(imgKScale, scale_1, mask, colLoopCount * VL_FP32);
                AscendC::MicroAPI::Mul(scale_0, sin_0, scale_0, mask);
                AscendC::MicroAPI::Mul(scale_1, sin_1, scale_1, mask);
                LoadTensorForDtypeT(realKOffset, offset_0, mask, colLoopCount * VL_FP32);
                LoadTensorForDtypeT(imgKOffset, offset_1, mask, colLoopCount * VL_FP32);
                AscendC::MicroAPI::Add(offset_0, offset_0, scale_0, mask);
                AscendC::MicroAPI::Add(offset_1, offset_1, scale_1, mask);
                StoreUnAlignOneTensor<T_K_CACHE>(out2, offset_0, UReg2, mask, colTail);
                StoreUnAlignOneTensor<T_K_CACHE>(out3, offset_1, UReg3, mask, colTail);
                AscendC::MicroAPI::DataCopyUnAlignPost(out0, UReg0, 0);
                AscendC::MicroAPI::DataCopyUnAlignPost(out1, UReg1, 0);
                AscendC::MicroAPI::DataCopyUnAlignPost(out2, UReg2, 0);
                AscendC::MicroAPI::DataCopyUnAlignPost(out3, UReg3, 0);
            }
        }
    }

    // 当前dk子块的cos/sin搬入，realCosLocal为四段(realCos/imgCos/realSin/imgSin)的首地址
    __aicore__ inline void CopyInCosSinChunk(
        const LocalTensor<T_KV>& realCosLocal, const LocalTensor<T_KV>& imgCosLocal,
        const LocalTensor<T_KV>& realSinLocal, const LocalTensor<T_KV>& imgSinLocal, int64_t loopIdx, int64_t calcRow,
        int64_t startIdx, int64_t halfChunkStart, int64_t halfChunk, int64_t halfChunkAlign)
    {
        if (tilingData_->cosSinNeedBrc == 0) {
            int64_t freqOffset = (this->blockIdx * tilingData_->blockFactor + loopIdx * this->ubFactor) * this->dk +
                                 halfChunkStart;
            CopyInLineAlign<T_KV>(realCosLocal, this->cosGm[freqOffset], calcRow, this->dk, halfChunk);
            CopyInLineAlign<T_KV>(imgCosLocal, this->cosGm[freqOffset + halfDk], calcRow, this->dk, halfChunk);
            CopyInLineAlign<T_KV>(realSinLocal, this->sinGm[freqOffset], calcRow, this->dk, halfChunk);
            CopyInLineAlign<T_KV>(imgSinLocal, this->sinGm[freqOffset + halfDk], calcRow, this->dk, halfChunk);
            return;
        }
        for (int64_t i = 0; i < calcRow; i++) {
            int64_t batchId = (startIdx + i) / tilingData_->seqLength;
            int64_t freqOffset = batchId * this->dk + halfChunkStart;
            int64_t ubOffset = i * halfChunkAlign;
            CopyInLineAlign<T_KV>(realCosLocal[ubOffset], this->cosGm[freqOffset], 1, this->dk, halfChunk);
            CopyInLineAlign<T_KV>(imgCosLocal[ubOffset], this->cosGm[freqOffset + halfDk], 1, this->dk, halfChunk);
            CopyInLineAlign<T_KV>(realSinLocal[ubOffset], this->sinGm[freqOffset], 1, this->dk, halfChunk);
            CopyInLineAlign<T_KV>(imgSinLocal[ubOffset], this->sinGm[freqOffset + halfDk], 1, this->dk, halfChunk);
        }
    }

    // 当前dk子块的k scale/offset搬入。scaleType==1为标量，需要Brc成一整段方便VF按列取数
    __aicore__ inline void CopyInKScaleOffsetChunk(int64_t halfChunkStart, int64_t halfChunk)
    {
        realKScaleLocal = kScaleOffsetQueue.AllocTensor<float>();
        imgKScaleLocal = realKScaleLocal[halfDkUbFactorAlign];
        realKOffsetLocal = imgKScaleLocal[halfDkUbFactorAlign];
        imgKOffsetLocal = realKOffsetLocal[halfDkUbFactorAlign];

        int64_t kScaleCopyLen = tilingData_->kScaleType == 1 ? 1 : halfChunk;
        int64_t kScaleRealOffset = tilingData_->kScaleType == 1 ? 0 : halfChunkStart;
        int64_t kScaleImgOffset = tilingData_->kScaleType == 1 ? 0 : halfDk + halfChunkStart;
        CopyInLineAlign<float>(realKScaleLocal, this->kScaleGm[kScaleRealOffset], 1, kScaleCopyLen, kScaleCopyLen);
        CopyInLineAlign<float>(imgKScaleLocal, this->kScaleGm[kScaleImgOffset], 1, kScaleCopyLen, kScaleCopyLen);
        if (tilingData_->kOffsetType > 0) {
            int64_t kOffsetCopyLen = tilingData_->kOffsetType == 1 ? 1 : halfChunk;
            int64_t kOffsetRealOffset = tilingData_->kOffsetType == 1 ? 0 : halfChunkStart;
            int64_t kOffsetImgOffset = tilingData_->kOffsetType == 1 ? 0 : halfDk + halfChunkStart;
            CopyInLineAlign<float>(
                realKOffsetLocal, this->kOffsetGm[kOffsetRealOffset], 1, kOffsetCopyLen, kOffsetCopyLen);
            CopyInLineAlign<float>(
                imgKOffsetLocal, this->kOffsetGm[kOffsetImgOffset], 1, kOffsetCopyLen, kOffsetCopyLen);
        }
        kScaleOffsetQueue.EnQue(realKScaleLocal);
        realKScaleLocal = kScaleOffsetQueue.DeQue<float>();
        // 将scale和offset Brc方便VF处理
        if (tilingData_->kScaleType == 1) {
            Duplicate<float>(realKScaleLocal, realKScaleLocal, halfDkUbFactorAlign);
            Duplicate<float>(imgKScaleLocal, imgKScaleLocal, halfDkUbFactorAlign);
        }
        if (tilingData_->kOffsetType == 1) {
            Duplicate<float>(realKOffsetLocal, realKOffsetLocal, halfDkUbFactorAlign);
            Duplicate<float>(imgKOffsetLocal, imgKOffsetLocal, halfDkUbFactorAlign);
        }
    }

    __aicore__ inline void Rope(
        const LocalTensor<T_KV>& xLocal, const LocalTensor<T_KV>& realCosLocal, const LocalTensor<T_KV>& imgCosLocal,
        const LocalTensor<T_KV>& realSinLocal, const LocalTensor<T_KV>& imgSinLocal, const LocalTensor<float>& wsLocal,
        uint16_t rowNum, int64_t halfChunk, int64_t chunkDkAlign, int64_t halfChunkAlign)
    {
        uint16_t row = rowNum;
        uint32_t colCosSin = halfChunk;
        uint16_t colLoopCount = static_cast<uint16_t>(halfChunk / VL_FP32);
        uint32_t colTail = static_cast<uint32_t>(halfChunk % VL_FP32);
        uint32_t xOffset = chunkDkAlign;
        uint32_t outOffset = dkUbFactorAlign;
        uint32_t cosSinOffset = halfChunkAlign;
        uint32_t quantOutOffset = dkUbFactorB8Align;

        __local_mem__ T_KV* x = (__local_mem__ T_KV*)xLocal.GetPhyAddr();
        __local_mem__ T_KV* realCos = (__local_mem__ T_KV*)realCosLocal.GetPhyAddr();
        __local_mem__ T_KV* imgCos = (__local_mem__ T_KV*)imgCosLocal.GetPhyAddr();
        __local_mem__ T_KV* realSin = (__local_mem__ T_KV*)realSinLocal.GetPhyAddr();
        __local_mem__ T_KV* imgSin = (__local_mem__ T_KV*)imgSinLocal.GetPhyAddr();
        __local_mem__ float* ws = (__local_mem__ float*)wsLocal.GetPhyAddr();

        if constexpr (K_QUANT) {
            __local_mem__ float* realKScale = (__local_mem__ float*)realKScaleLocal.GetPhyAddr();
            __local_mem__ float* imgKScale = (__local_mem__ float*)imgKScaleLocal.GetPhyAddr();
            __local_mem__ float* realKOffset = (__local_mem__ float*)realKOffsetLocal.GetPhyAddr();
            __local_mem__ float* imgKOffset = (__local_mem__ float*)imgKOffsetLocal.GetPhyAddr();
            if (tilingData_->isOutputKv > 0) {
                kOutLocal = outQueue.AllocTensor<T_KV>();
                kQuantLocal =
                    kOutLocal.template ReinterpretCast<T_K_CACHE>()[this->ubFactor * dkUbFactorB8Align * sizeof(T_KV)];
                __local_mem__ T_KV* outFront = (__local_mem__ T_KV*)kOutLocal.GetPhyAddr();
                __local_mem__ T_KV* outBack = (__local_mem__ T_KV*)kOutLocal.GetPhyAddr() + halfDkUbFactor;
                __local_mem__ T_K_CACHE* quantFront = (__local_mem__ T_K_CACHE*)kQuantLocal.GetPhyAddr();
                __local_mem__ T_K_CACHE* quantBack =
                    (__local_mem__ T_K_CACHE*)kQuantLocal.GetPhyAddr() + halfDkUbFactor;
                if (tilingData_->kOffsetType > 0) {
                    RopeAsymQuantWithKvVF(
                        outFront, outBack, quantFront, quantBack, x, realCos, imgCos, realSin, imgSin, realKScale,
                        imgKScale, realKOffset, imgKOffset, ws, row, colLoopCount, colTail, colCosSin, xOffset,
                        outOffset, cosSinOffset, quantOutOffset);
                } else {
                    RopeSymQuantWithKvVF(
                        outFront, outBack, quantFront, quantBack, x, realCos, imgCos, realSin, imgSin, realKScale,
                        imgKScale, ws, row, colLoopCount, colTail, colCosSin, xOffset, outOffset, cosSinOffset,
                        quantOutOffset);
                }
                outQueue.EnQue(kOutLocal);
                kOutLocal = outQueue.DeQue<T_KV>();
            } else {
                kQuantLocal = outQueue.AllocTensor<T_K_CACHE>();
                __local_mem__ T_K_CACHE* quantFront = (__local_mem__ T_K_CACHE*)kQuantLocal.GetPhyAddr();
                __local_mem__ T_K_CACHE* quantBack =
                    (__local_mem__ T_K_CACHE*)kQuantLocal.GetPhyAddr() + halfDkUbFactor;
                if (tilingData_->kOffsetType > 0) {
                    RopeAsymQuantVF(
                        quantFront, quantBack, x, realCos, imgCos, realSin, imgSin, realKScale, imgKScale, realKOffset,
                        imgKOffset, ws, row, colLoopCount, colTail, colCosSin, xOffset, cosSinOffset, quantOutOffset);
                } else {
                    RopeSymQuantVF(
                        quantFront, quantBack, x, realCos, imgCos, realSin, imgSin, realKScale, imgKScale, ws, row,
                        colLoopCount, colTail, colCosSin, xOffset, cosSinOffset, quantOutOffset);
                }
                outQueue.EnQue(kQuantLocal);
                kQuantLocal = outQueue.DeQue<T_K_CACHE>();
            }
        } else {
            kOutLocal = outQueue.AllocTensor<T_KV>();
            __local_mem__ T_KV* outFront = (__local_mem__ T_KV*)kOutLocal.GetPhyAddr();
            __local_mem__ T_KV* outBack = (__local_mem__ T_KV*)kOutLocal.GetPhyAddr() + halfDkUbFactor;
            RopeVF(
                outFront, outBack, x, realCos, imgCos, realSin, imgSin, ws, row, colLoopCount, colTail, colCosSin,
                xOffset, outOffset, cosSinOffset);
            outQueue.EnQue(kOutLocal);
            kOutLocal = outQueue.DeQue<T_KV>();
        }
    }

    // Rope的dk方向UB循环：每次处理[calcRow, chunkDk]，实部/虚部分别落到kCache与kOut的两段
    __aicore__ inline void RopeDkLoop(
        int64_t loopIdx, int64_t calcRow, int64_t startIdx, int64_t kOutGmOffset,
        const LocalTensor<float>& workspaceBuffer0)
    {
        for (int64_t dkIdx = 0; dkIdx < dkLoop; ++dkIdx) {
            int64_t chunkDk = (dkIdx == dkLoop - 1) ? dkTail : dkUbFactor;
            int64_t halfChunk = chunkDk / CONST_TWO;
            int64_t dkChunkStart = dkIdx * dkUbFactor;
            int64_t halfChunkStart = dkChunkStart / CONST_TWO;
            int64_t chunkDkAlign = ops::Aligned(chunkDk, blockElem);
            int64_t halfChunkAlign = ops::Aligned(halfChunk, blockElem);

            LocalTensor<T_KV> ropeLocal = inQueueX.AllocTensor<T_KV>();
            LocalTensor<T_KV> realCosLocal = inQueueCosSin.AllocTensor<T_KV>();
            LocalTensor<T_KV> imgCosLocal = realCosLocal[this->ubFactor * halfDkUbFactorAlign];
            LocalTensor<T_KV> realSinLocal = imgCosLocal[this->ubFactor * halfDkUbFactorAlign];
            LocalTensor<T_KV> imgSinLocal = realSinLocal[this->ubFactor * halfDkUbFactorAlign];

            CopyInLineAlign<T_KV>(
                ropeLocal, this->kvGm[loopIdx * this->ubFactor * dKV + this->dv + dkChunkStart], calcRow, dKV,
                chunkDk);
            CopyInCosSinChunk(
                realCosLocal, imgCosLocal, realSinLocal, imgSinLocal, loopIdx, calcRow, startIdx, halfChunkStart,
                halfChunk, halfChunkAlign);
            inQueueX.EnQue(ropeLocal);
            inQueueCosSin.EnQue(realCosLocal);
            ropeLocal = inQueueX.DeQue<T_KV>();
            realCosLocal = inQueueCosSin.DeQue<T_KV>();
            if constexpr (K_QUANT) {
                CopyInKScaleOffsetChunk(halfChunkStart, halfChunk);
            }

            Rope(ropeLocal, realCosLocal, imgCosLocal, realSinLocal, imgSinLocal, workspaceBuffer0, calcRow, halfChunk,
                chunkDkAlign, halfChunkAlign);
            inQueueX.FreeTensor(ropeLocal);
            inQueueCosSin.FreeTensor(realCosLocal);
            if constexpr (K_QUANT) {
                kScaleOffsetQueue.FreeTensor(realKScaleLocal);
            }

            // 拷出到kCache
            int64_t kCacheRowStride = K_QUANT ? dkUbFactorB8Align : dkUbFactorAlign;
            LocalTensor<T_K_CACHE> kCacheSrc;
            if constexpr (K_QUANT) {
                kCacheSrc = kQuantLocal;
            } else {
                kCacheSrc = kOutLocal.template ReinterpretCast<T_K_CACHE>();
            }
            if (tilingData_->cacheMode <= PA_NZ_CACHE_MODE) {
                ScatterUpdateK(kCacheSrc, calcRow, startIdx, kCacheRowStride, halfChunkStart, halfChunk);
            } else {
                ScatterBlkUpdateK(kCacheSrc, calcRow, startIdx, kCacheRowStride, halfChunkStart, halfChunk);
            }

            // 拷出到kOut
            if (tilingData_->isOutputKv > 0) {
                CopyOutRowsSeg<T_KV>(
                    this->kOutGm[kOutGmOffset + halfChunkStart], kOutLocal, calcRow, this->dk, halfChunk,
                    dkUbFactorAlign);
                CopyOutRowsSeg<T_KV>(
                    this->kOutGm[kOutGmOffset + halfDk + halfChunkStart], kOutLocal[halfDkUbFactor], calcRow, this->dk,
                    halfChunk, dkUbFactorAlign);
            }

            // FreeTensor 区分场景
            if constexpr (K_QUANT) {
                if (tilingData_->isOutputKv > 0) {
                    outQueue.FreeTensor(kOutLocal);
                } else {
                    outQueue.FreeTensor(kQuantLocal);
                }
            } else {
                outQueue.FreeTensor(kOutLocal);
            }
        }
    }

    __aicore__ inline void RmsNorm(
        const LocalTensor<T_KV>& outVLocal, const LocalTensor<T_KV>& xLocal, const LocalTensor<T_KV>& gammaLocal,
        const LocalTensor<float>& wsLocal, const LocalTensor<float>& vScaleTensor,
        const LocalTensor<float>& vOffsetTensor, int64_t calcRow)
    {
        // 需要量化
        if constexpr (V_QUANT) {
            // 需要输出中间结果
            if (tilingData_->isOutputKv > 0) {
                vQuantLocal =
                    outVLocal
                        .template ReinterpretCast<T_V_CACHE>()[this->ubFactor * tilingData_->dvB8Align * sizeof(T_KV)];
                // 需要量化+偏移
                if (tilingData_->vOffsetType > 0) {
                    this->RmsNormAsymQuantWithKvVF(
                        outVLocal, vQuantLocal, xLocal, gammaLocal, wsLocal, vScaleTensor, vOffsetTensor, calcRow);
                } else { // 只需要量化
                    this->RmsNormSymQuantWithKvVF(
                        outVLocal, vQuantLocal, xLocal, gammaLocal, wsLocal, vScaleTensor, calcRow);
                }
            } else {
                vQuantLocal = outVLocal.template ReinterpretCast<T_V_CACHE>();
                if (tilingData_->vOffsetType > 0) {
                    this->RmsNormAsymQuantVF(
                        vQuantLocal, xLocal, gammaLocal, wsLocal, vScaleTensor, vOffsetTensor, calcRow);
                } else {
                    this->RmsNormSymQuantVF(vQuantLocal, xLocal, gammaLocal, wsLocal, vScaleTensor, calcRow);
                }
            }
        } else {
            // 普通情况， outLocal的类型需要指定
            this->RmsNormVF(outVLocal, xLocal, gammaLocal, wsLocal, calcRow);
        }
    }

    __aicore__ inline void Process()
    {
        // CopyIn gamma
        LocalTensor<T_KV> gammaLocal = inQueueGamma.AllocTensor<T_KV>();
        CopyInLineAlign<T_KV>(gammaLocal, this->gammaGm, 1, tilingData_->dv, tilingData_->dv);
        inQueueGamma.EnQue(gammaLocal);
        gammaLocal = inQueueGamma.DeQue<T_KV>();
        LocalTensor<float> workspaceBuffer0 = wsBuffer0.Get<float>();
        LocalTensor<float> workspaceBuffer1 = wsBuffer1.Get<float>();
        // 行，有几行相当于ubA
        int64_t calcRow = this->ubFactor;
        // v侧Dv全载，scale/offset只需搬入一次
        if constexpr (V_QUANT) {
            vScaleLocal = vScaleOffsetQueue.AllocTensor<float>();
            vOffsetLocal = vScaleLocal[tilingData_->dvAlign];
            CopyInLineAlign<float>(vScaleLocal, this->vScaleGm, 1, vScaleCopyLen, vScaleCopyLen);
            if (tilingData_->vOffsetType > 0) {
                CopyInLineAlign<float>(vOffsetLocal, this->vOffsetGm, 1, vOffsetCopyLen, vOffsetCopyLen);
            }
            vScaleOffsetQueue.EnQue(vScaleLocal);
            vScaleLocal = vScaleOffsetQueue.DeQue<float>();
            // 将scale和offset Brc方便VF处理
            if (tilingData_->vScaleType == 1) {
                Duplicate<float>(vScaleLocal, vScaleLocal, tilingData_->dvAlign);
            }
            if (tilingData_->vOffsetType == 1) {
                Duplicate<float>(vOffsetLocal, vOffsetLocal, tilingData_->dvAlign);
            }
        }
        for (int64_t loopIdx = 0; loopIdx < this->ubLoop; ++loopIdx) {
            if (loopIdx == this->ubLoop - 1) {
                calcRow = this->ubTail;
            }
            kvGlobalMemoryOffset = loopIdx * this->ubFactor * dKV;
            kOutGmOffset = loopIdx * this->ubFactor * this->dk;
            vOutGmOffset = loopIdx * this->ubFactor * this->dv;
            // 当前子块的偏移量
            int64_t startIdx = this->blockIdx * tilingData_->blockFactor + loopIdx * this->ubFactor;

            // Calc: RoPE, Dk方向按dkUbFactor切分循环
            RopeDkLoop(loopIdx, calcRow, startIdx, kOutGmOffset, workspaceBuffer0);

            // CopyIn x: [ubFactor, RmsLength]
            LocalTensor<T_KV> xLocal = inQueueX.AllocTensor<T_KV>();
            CopyInLineAlign<T_KV>(xLocal, this->kvGm[kvGlobalMemoryOffset], calcRow, dKV, tilingData_->dv);
            inQueueX.EnQue(xLocal);
            xLocal = inQueueX.DeQue<T_KV>();

            // Calc: RmsNorm，Dv全载
            vOutLocal = outQueue.AllocTensor<T_KV>();
            this->RmsNorm(vOutLocal, xLocal, gammaLocal, workspaceBuffer1, vScaleLocal, vOffsetLocal, calcRow);
            inQueueX.FreeTensor(xLocal);
            outQueue.EnQue(vOutLocal);
            vOutLocal = outQueue.DeQue<T_KV>();
            // 拷出到vCache
            if (tilingData_->cacheMode <= PA_NZ_CACHE_MODE) {
                ScatterUpdateV(calcRow, startIdx);
            } else {
                ScatterBlkUpdateV(calcRow, startIdx);
            }
            // 拷出到vOut
            if (tilingData_->isOutputKv > 0) {
                CopyLineAlignOut<T_KV>(this->vOutGm[vOutGmOffset], vOutLocal, calcRow, this->dv, this->dv);
            }
            // FreeTensor 不论什么场景，都是申请的vOutLocal
            outQueue.FreeTensor(vOutLocal);
        }
        inQueueGamma.FreeTensor(gammaLocal);
        if constexpr (V_QUANT) {
            vScaleOffsetQueue.FreeTensor(vScaleLocal);
        }
    }

private:
    const KvRmsNormRopeCacheRegbaseDkSplitTilingData* tilingData_;
    TQue<QuePosition::VECIN, 1> inQueueX, inQueueGamma, inQueueCosSin;
    TQue<QuePosition::VECIN, 1> kScaleOffsetQueue, vScaleOffsetQueue;
    TQue<QuePosition::VECOUT, 1> outQueue;
    TBuf<TPosition::VECCALC> wsBuffer0, wsBuffer1;
    TPipe* pipe_ = nullptr;

    LocalTensor<T_KV> kOutLocal;
    LocalTensor<T_K_CACHE> kQuantLocal;
    LocalTensor<T_KV> vOutLocal;
    LocalTensor<T_V_CACHE> vQuantLocal;

    LocalTensor<float> realKScaleLocal;
    LocalTensor<float> imgKScaleLocal;
    LocalTensor<float> realKOffsetLocal;
    LocalTensor<float> imgKOffsetLocal;
    LocalTensor<float> vScaleLocal;
    LocalTensor<float> vOffsetLocal;

    event_t eventIDSToMTE3;

    int64_t dKV = 0;
    int64_t vScaleCopyLen = 0;
    int64_t vOffsetCopyLen = 0;
    int64_t dk0 = 0;
    int64_t dk1 = 0;
    int64_t dv0 = 0;
    int64_t dv1 = 0;
    int64_t blockElem = 0;

    int64_t halfDk = 0;
    int64_t dkUbFactor = 0;
    int64_t dkUbFactorAlign = 0;
    int64_t dkUbFactorB8Align = 0;
    int64_t halfDkUbFactor = 0;
    int64_t halfDkUbFactorAlign = 0;
    int64_t dkLoop = 0;
    int64_t dkTail = 0;

    DataCopyExtParams copyOutVParams;
    DataCopyExtParams copyOutVParamsNz;

    int64_t kvGlobalMemoryOffset = 0;
    int64_t kOutGmOffset = 0;
    int64_t vOutGmOffset = 0;
};
} // namespace KvRmsNormRopeCache

#endif // KV_RMS_NORM_ROPE_CACHE_REGBASE_DK_SPLIT_H_
