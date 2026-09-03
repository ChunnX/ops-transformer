/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_pa_kv_cache_with_k_scale_simd.h
 * \brief SIMD template: split cores by num_tokens, scatter key/value/key_scale with DataCopyPad
 * \note SIMD 模板要求 kHeadSize/vHeadSize 不小于 128B（tiling 侧 IsSimdCapable 保证）
 */

#ifndef SCATTER_PA_KV_CACHE_WITH_K_SCALE_SIMD_H_
#define SCATTER_PA_KV_CACHE_WITH_K_SCALE_SIMD_H_

#include "kernel_operator.h"
#include "scatter_pa_kv_cache_with_k_scale_tiling_data.h"

namespace NsScatterPaKvCacheWithKScale {

using namespace AscendC;

constexpr int64_t SIMD_DOUBLE_BUFFER = 2;
constexpr int64_t SIMD_ONE_BLK_SIZE = 32;
constexpr int64_t SIMD_KV_QUEUE_NUM = 2; // key / value

template <typename T, typename IndexDtype>
class ScatterPaKvCacheWithKScaleSimd {
public:
    __aicore__ inline ScatterPaKvCacheWithKScaleSimd(TPipe *pipe,
                                                     const ScatterPaKvCacheWithKScaleTilingData *__restrict tiling)
        : pipe_(pipe),
          tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR key, GM_ADDR value, GM_ADDR slotMapping, GM_ADDR keyScale, GM_ADDR keyCacheOut,
                                GM_ADDR valueCacheOut, GM_ADDR keyScaleCacheOut);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInKey(int64_t tokenIdx, int64_t headIdx, int64_t sizeOffset, int64_t headNum,
                                     int64_t headSize);
    __aicore__ inline void CopyOutKey(int64_t blockIdx, int64_t blockOffset, int64_t headIdx, int64_t sizeOffset,
                                      int64_t headNum, int64_t headSize);
    __aicore__ inline void CopyInValue(int64_t tokenIdx, int64_t headIdx, int64_t sizeOffset, int64_t headNum,
                                       int64_t headSize);
    __aicore__ inline void CopyOutValue(int64_t blockIdx, int64_t blockOffset, int64_t headIdx, int64_t sizeOffset,
                                        int64_t headNum, int64_t headSize);
    __aicore__ inline int64_t RoundUp(int64_t x);
    __aicore__ inline void InitLoopInfo();
    __aicore__ inline void ProcessKey(int64_t tokenIdx, int64_t blockIdx, int64_t blockOffset);
    __aicore__ inline void ProcessValue(int64_t tokenIdx, int64_t blockIdx, int64_t blockOffset);
    __aicore__ inline void ProcessKeyScale(int64_t tokenIdx, int64_t blockIdx, int64_t blockOffset);

private:
    TPipe *pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, SIMD_DOUBLE_BUFFER> keyQue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, SIMD_DOUBLE_BUFFER> valueQue_;
    const ScatterPaKvCacheWithKScaleTilingData *__restrict tilingData_;

    GlobalTensor<T> keyGm_;
    GlobalTensor<T> valueGm_;
    GlobalTensor<T> keyCacheOutGm_;
    GlobalTensor<T> valueCacheOutGm_;
    GlobalTensor<float> keyScaleGm_;
    GlobalTensor<float> keyScaleCacheGm_;
    GlobalTensor<IndexDtype> slotMappingGm_;

    int64_t blockIdx_ = 0;
    int64_t blockFactorOffset_ = 0;
    int64_t maxSlot_ = 0;
    int64_t maxHandleSize_ = 0;

    bool kFullyLoad_ = false;
    int64_t kHeadSizeIn_ = 0;
    int64_t kHeadSizeloop_ = 0;
    int64_t kHeadSizeTail_ = 0;
    int64_t kNumHeadIn_ = 0;
    int64_t kNumHeadLoop_ = 0;
    int64_t kNumHeadTail_ = 0;

    bool vFullyLoad_ = false;
    int64_t vHeadSizeIn_ = 0;
    int64_t vHeadSizeloop_ = 0;
    int64_t vHeadSizeTail_ = 0;
    int64_t vNumHeadIn_ = 0;
    int64_t vNumHeadLoop_ = 0;
    int64_t vNumHeadTail_ = 0;
};

template <typename T, typename IndexDtype>
__aicore__ inline void ScatterPaKvCacheWithKScaleSimd<T, IndexDtype>::Init(GM_ADDR key, GM_ADDR value,
                                                                           GM_ADDR slotMapping, GM_ADDR keyScale,
                                                                           GM_ADDR keyCacheOut, GM_ADDR valueCacheOut,
                                                                           GM_ADDR keyScaleCacheOut)
{
    blockIdx_ = GetBlockIdx();
    blockFactorOffset_ = blockIdx_ * tilingData_->blockFactor;
    maxSlot_ = tilingData_->maxSlot;

    keyGm_.SetGlobalBuffer((__gm__ T *)(key));
    valueGm_.SetGlobalBuffer((__gm__ T *)(value));
    slotMappingGm_.SetGlobalBuffer((__gm__ IndexDtype *)(slotMapping));
    keyScaleGm_.SetGlobalBuffer((__gm__ float *)(keyScale));
    keyCacheOutGm_.SetGlobalBuffer((__gm__ T *)(keyCacheOut));
    valueCacheOutGm_.SetGlobalBuffer((__gm__ T *)(valueCacheOut));
    keyScaleCacheGm_.SetGlobalBuffer((__gm__ float *)(keyScaleCacheOut));

    int64_t ubSize = tilingData_->ubSize;
    int64_t maxHandleSize = ubSize / SIMD_KV_QUEUE_NUM / SIMD_DOUBLE_BUFFER;
    maxHandleSize = maxHandleSize / SIMD_ONE_BLK_SIZE * SIMD_ONE_BLK_SIZE;
    if (maxHandleSize < SIMD_ONE_BLK_SIZE) {
        maxHandleSize = SIMD_ONE_BLK_SIZE;
    }

    pipe_->InitBuffer(keyQue_, SIMD_DOUBLE_BUFFER, maxHandleSize);
    pipe_->InitBuffer(valueQue_, SIMD_DOUBLE_BUFFER, maxHandleSize);
    maxHandleSize_ = maxHandleSize;
}

template <typename T, typename IndexDtype>
__aicore__ inline int64_t ScatterPaKvCacheWithKScaleSimd<T, IndexDtype>::RoundUp(int64_t x)
{
    int64_t elemNum = SIMD_ONE_BLK_SIZE / static_cast<int64_t>(sizeof(T));
    return (x + elemNum - 1) / elemNum * elemNum;
}

template <typename T, typename IndexDtype>
__aicore__ inline void ScatterPaKvCacheWithKScaleSimd<T, IndexDtype>::CopyInKey(int64_t tokenIdx, int64_t headIdx,
                                                                                int64_t sizeOffset, int64_t headNum,
                                                                                int64_t headSize)
{
    LocalTensor<T> inputKeyLocal = keyQue_.AllocTensor<T>();

    int64_t keyOffset = tokenIdx * tilingData_->keyStride[0] + headIdx * tilingData_->keyStride[1] + sizeOffset;

    DataCopyExtParams keyParams = {static_cast<uint16_t>(headNum), static_cast<uint32_t>(headSize * sizeof(T)),
                                   static_cast<uint32_t>((tilingData_->keyStride[1] - headSize) * sizeof(T)),
                                   static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPadExtParams<T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0)};
    DataCopyPad(inputKeyLocal, keyGm_[keyOffset], keyParams, padParams);

    keyQue_.EnQue(inputKeyLocal);
}

template <typename T, typename IndexDtype>
__aicore__ inline void ScatterPaKvCacheWithKScaleSimd<T, IndexDtype>::CopyOutKey(int64_t blockIdx, int64_t blockOffset,
                                                                                 int64_t headIdx, int64_t sizeOffset,
                                                                                 int64_t headNum, int64_t headSize)
{
    LocalTensor<T> inputKeyLocal = keyQue_.DeQue<T>();

    int64_t keyCacheOffset = blockIdx * tilingData_->keyCacheStride[0] + headIdx * tilingData_->keyCacheStride[1] +
                             blockOffset * tilingData_->keyCacheStride[2] + sizeOffset;

    DataCopyExtParams outKeyCacheParams = {
        static_cast<uint16_t>(headNum), static_cast<uint32_t>(headSize * sizeof(T)), static_cast<uint32_t>(0),
        static_cast<uint32_t>((tilingData_->keyCacheStride[1] - headSize) * sizeof(T)), static_cast<uint32_t>(0)};

    DataCopyPad(keyCacheOutGm_[keyCacheOffset], inputKeyLocal, outKeyCacheParams);

    keyQue_.FreeTensor(inputKeyLocal);
}

template <typename T, typename IndexDtype>
__aicore__ inline void ScatterPaKvCacheWithKScaleSimd<T, IndexDtype>::CopyInValue(int64_t tokenIdx, int64_t headIdx,
                                                                                  int64_t sizeOffset, int64_t headNum,
                                                                                  int64_t headSize)
{
    LocalTensor<T> inputValueLocal = valueQue_.AllocTensor<T>();

    int64_t valueOffset = tokenIdx * tilingData_->valueStride[0] + headIdx * tilingData_->valueStride[1] + sizeOffset;

    DataCopyExtParams valueParams = {static_cast<uint16_t>(headNum), static_cast<uint32_t>(headSize * sizeof(T)),
                                     static_cast<uint32_t>((tilingData_->valueStride[1] - headSize) * sizeof(T)),
                                     static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPadExtParams<T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0)};
    DataCopyPad(inputValueLocal, valueGm_[valueOffset], valueParams, padParams);

    valueQue_.EnQue(inputValueLocal);
}

template <typename T, typename IndexDtype>
__aicore__ inline void ScatterPaKvCacheWithKScaleSimd<T, IndexDtype>::CopyOutValue(int64_t blockIdx,
                                                                                   int64_t blockOffset, int64_t headIdx,
                                                                                   int64_t sizeOffset, int64_t headNum,
                                                                                   int64_t headSize)
{
    LocalTensor<T> inputValueLocal = valueQue_.DeQue<T>();

    int64_t valueCacheOffset = blockIdx * tilingData_->valueCacheStride[0] +
                               headIdx * tilingData_->valueCacheStride[1] +
                               blockOffset * tilingData_->valueCacheStride[2] + sizeOffset;

    DataCopyExtParams outValueCacheParams = {
        static_cast<uint16_t>(headNum), static_cast<uint32_t>(headSize * sizeof(T)), static_cast<uint32_t>(0),
        static_cast<uint32_t>((tilingData_->valueCacheStride[1] - headSize) * sizeof(T)), static_cast<uint32_t>(0)};

    DataCopyPad(valueCacheOutGm_[valueCacheOffset], inputValueLocal, outValueCacheParams);

    valueQue_.FreeTensor(inputValueLocal);
}

template <typename T, typename IndexDtype>
__aicore__ inline void ScatterPaKvCacheWithKScaleSimd<T, IndexDtype>::InitLoopInfo()
{
    int64_t ubHandleSize = maxHandleSize_ / SIMD_ONE_BLK_SIZE * SIMD_ONE_BLK_SIZE;
    ubHandleSize = ubHandleSize / static_cast<int64_t>(sizeof(T));
    int64_t alignKHeadSize = RoundUp(tilingData_->kHeadSize);

    kFullyLoad_ = (alignKHeadSize <= ubHandleSize);

    if (kFullyLoad_) {
        kHeadSizeIn_ = tilingData_->kHeadSize;
        kHeadSizeloop_ = 1;
        kHeadSizeTail_ = 0;
        kNumHeadIn_ = ubHandleSize / alignKHeadSize;
        if (kNumHeadIn_ > tilingData_->numHead) {
            kNumHeadIn_ = tilingData_->numHead;
        }
        if (kNumHeadIn_ < 1) {
            kNumHeadIn_ = 1;
        }
        kNumHeadLoop_ = tilingData_->numHead / kNumHeadIn_;
        kNumHeadTail_ = tilingData_->numHead - kNumHeadIn_ * kNumHeadLoop_;
    } else {
        kHeadSizeIn_ = ubHandleSize;
        kHeadSizeloop_ = tilingData_->kHeadSize / kHeadSizeIn_;
        kHeadSizeTail_ = tilingData_->kHeadSize - kHeadSizeIn_ * kHeadSizeloop_;
        kNumHeadIn_ = 1;
        kNumHeadLoop_ = tilingData_->numHead;
        kNumHeadTail_ = 0;
    }

    int64_t alignVHeadSize = RoundUp(tilingData_->vHeadSize);

    vFullyLoad_ = (alignVHeadSize <= ubHandleSize);

    if (vFullyLoad_) {
        vHeadSizeIn_ = tilingData_->vHeadSize;
        vHeadSizeloop_ = 1;
        vHeadSizeTail_ = 0;
        vNumHeadIn_ = ubHandleSize / alignVHeadSize;
        if (vNumHeadIn_ > tilingData_->numHead) {
            vNumHeadIn_ = tilingData_->numHead;
        }
        if (vNumHeadIn_ < 1) {
            vNumHeadIn_ = 1;
        }
        vNumHeadLoop_ = tilingData_->numHead / vNumHeadIn_;
        vNumHeadTail_ = tilingData_->numHead - vNumHeadIn_ * vNumHeadLoop_;
    } else {
        vHeadSizeIn_ = ubHandleSize;
        vHeadSizeloop_ = tilingData_->vHeadSize / vHeadSizeIn_;
        vHeadSizeTail_ = tilingData_->vHeadSize - vHeadSizeIn_ * vHeadSizeloop_;
        vNumHeadIn_ = 1;
        vNumHeadLoop_ = tilingData_->numHead;
        vNumHeadTail_ = 0;
    }
}

template <typename T, typename IndexDtype>
__aicore__ inline void ScatterPaKvCacheWithKScaleSimd<T, IndexDtype>::ProcessKey(int64_t tokenIdx, int64_t blockIdx,
                                                                                 int64_t blockOffset)
{
    if (kFullyLoad_) {
        for (int64_t headLoop = 0; headLoop < kNumHeadLoop_; headLoop++) {
            int64_t headIdx = headLoop * kNumHeadIn_;
            CopyInKey(tokenIdx, headIdx, 0, kNumHeadIn_, kHeadSizeIn_);
            CopyOutKey(blockIdx, blockOffset, headIdx, 0, kNumHeadIn_, kHeadSizeIn_);
        }
        if (kNumHeadTail_ > 0) {
            int64_t headIdx = kNumHeadLoop_ * kNumHeadIn_;
            CopyInKey(tokenIdx, headIdx, 0, kNumHeadTail_, kHeadSizeIn_);
            CopyOutKey(blockIdx, blockOffset, headIdx, 0, kNumHeadTail_, kHeadSizeIn_);
        }
    } else {
        for (int64_t headLoop = 0; headLoop < kNumHeadLoop_; headLoop++) {
            int64_t headIdx = headLoop;
            for (int64_t sizeLoop = 0; sizeLoop < kHeadSizeloop_; sizeLoop++) {
                int64_t sizeOffset = sizeLoop * kHeadSizeIn_;
                CopyInKey(tokenIdx, headIdx, sizeOffset, 1, kHeadSizeIn_);
                CopyOutKey(blockIdx, blockOffset, headIdx, sizeOffset, 1, kHeadSizeIn_);
            }
            if (kHeadSizeTail_ > 0) {
                int64_t sizeOffset = kHeadSizeloop_ * kHeadSizeIn_;
                CopyInKey(tokenIdx, headIdx, sizeOffset, 1, kHeadSizeTail_);
                CopyOutKey(blockIdx, blockOffset, headIdx, sizeOffset, 1, kHeadSizeTail_);
            }
        }
    }
}

template <typename T, typename IndexDtype>
__aicore__ inline void ScatterPaKvCacheWithKScaleSimd<T, IndexDtype>::ProcessValue(int64_t tokenIdx, int64_t blockIdx,
                                                                                   int64_t blockOffset)
{
    if (vFullyLoad_) {
        for (int64_t headLoop = 0; headLoop < vNumHeadLoop_; headLoop++) {
            int64_t headIdx = headLoop * vNumHeadIn_;
            CopyInValue(tokenIdx, headIdx, 0, vNumHeadIn_, vHeadSizeIn_);
            CopyOutValue(blockIdx, blockOffset, headIdx, 0, vNumHeadIn_, vHeadSizeIn_);
        }
        if (vNumHeadTail_ > 0) {
            int64_t headIdx = vNumHeadLoop_ * vNumHeadIn_;
            CopyInValue(tokenIdx, headIdx, 0, vNumHeadTail_, vHeadSizeIn_);
            CopyOutValue(blockIdx, blockOffset, headIdx, 0, vNumHeadTail_, vHeadSizeIn_);
        }
    } else {
        for (int64_t headLoop = 0; headLoop < vNumHeadLoop_; headLoop++) {
            int64_t headIdx = headLoop;
            for (int64_t sizeLoop = 0; sizeLoop < vHeadSizeloop_; sizeLoop++) {
                int64_t sizeOffset = sizeLoop * vHeadSizeIn_;
                CopyInValue(tokenIdx, headIdx, sizeOffset, 1, vHeadSizeIn_);
                CopyOutValue(blockIdx, blockOffset, headIdx, sizeOffset, 1, vHeadSizeIn_);
            }
            if (vHeadSizeTail_ > 0) {
                int64_t sizeOffset = vHeadSizeloop_ * vHeadSizeIn_;
                CopyInValue(tokenIdx, headIdx, sizeOffset, 1, vHeadSizeTail_);
                CopyOutValue(blockIdx, blockOffset, headIdx, sizeOffset, 1, vHeadSizeTail_);
            }
        }
    }
}

template <typename T, typename IndexDtype>
__aicore__ inline void ScatterPaKvCacheWithKScaleSimd<T, IndexDtype>::ProcessKeyScale(int64_t tokenIdx,
                                                                                      int64_t blockIdx,
                                                                                      int64_t blockOffset)
{
    // key_scale 每个 (token, head) 仅一个标量，直接标量读写，无需 DataCopyPad
    for (int64_t headIdx = 0; headIdx < tilingData_->numHead; headIdx++) {
        int64_t keyScaleOffset = tokenIdx * tilingData_->keyScaleStride[0] + headIdx * tilingData_->keyScaleStride[1];
        int64_t keyScaleCacheOffset = blockIdx * tilingData_->keyScaleCacheStride[0] +
                                      headIdx * tilingData_->keyScaleCacheStride[1] +
                                      blockOffset * tilingData_->keyScaleCacheStride[2];
        keyScaleCacheGm_.SetValue(static_cast<uint64_t>(keyScaleCacheOffset),
                                  keyScaleGm_.GetValue(static_cast<uint64_t>(keyScaleOffset)));
    }
}

template <typename T, typename IndexDtype>
__aicore__ inline void ScatterPaKvCacheWithKScaleSimd<T, IndexDtype>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }
    int64_t curBlockFactor =
        (blockIdx_ == tilingData_->usedCoreNum - 1) ? tilingData_->tailBlockFactor : tilingData_->blockFactor;

    InitLoopInfo();
    for (int64_t idx = 0; idx < curBlockFactor; idx++) {
        int64_t tokenIdx = blockFactorOffset_ + idx;
        int64_t slotIdx = static_cast<int64_t>(
            slotMappingGm_.GetValue(static_cast<uint64_t>(tokenIdx * tilingData_->slotMappingStride[0])));
        if (slotIdx < 0 || slotIdx >= maxSlot_) {
            continue;
        }
        int64_t blockIdx = slotIdx / tilingData_->blockSize;
        int64_t blockOffset = slotIdx % tilingData_->blockSize;

        ProcessKey(tokenIdx, blockIdx, blockOffset);
        ProcessValue(tokenIdx, blockIdx, blockOffset);
        ProcessKeyScale(tokenIdx, blockIdx, blockOffset);
    }
}

} // namespace NsScatterPaKvCacheWithKScale
#endif // SCATTER_PA_KV_CACHE_WITH_K_SCALE_SIMD_H_
