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
 * \file scatter_pa_kv_cache_normal_contiguous.h
 * \brief
 */

#ifndef SCATTER_PA_KV_CACHE_NORMAL_CONTIGUOUS_H_
#define SCATTER_PA_KV_CACHE_NORMAL_CONTIGUOUS_H_

#include "kernel_operator.h"
#include "common.h"

namespace ScatterPaKvCache {
using namespace AscendC;

template <typename T, typename IndexDtype, int64_t InOutMode>
class ScatterPaKvCacheNormalContiguous {
public:
    __aicore__ inline ScatterPaKvCacheNormalContiguous(TPipe *pipe,
                                                         const ScatterPaKvCacheTilingData *__restrict tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR key, GM_ADDR key_cache_in, GM_ADDR slot_mapping, GM_ADDR value,
                                GM_ADDR value_cache_in, GM_ADDR key_cache_out, GM_ADDR value_cache_out);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInKey(int64_t tokenIdx, int64_t headIdx, int64_t sizeOffset, int64_t headNum, int64_t headSize);
    __aicore__ inline void CopyOutKey(int64_t blockIdx, int64_t blockOffset, int64_t headIdx, int64_t sizeOffset, int64_t headNum, int64_t headSize);
    __aicore__ inline void CopyInValue(int64_t tokenIdx, int64_t headIdx, int64_t sizeOffset, int64_t headNum, int64_t headSize);
    __aicore__ inline void CopyOutValue(int64_t blockIdx, int64_t blockOffset, int64_t headIdx, int64_t sizeOffset, int64_t headNum, int64_t headSize);
    __aicore__ inline int64_t RoundUp(int64_t x);
    __aicore__ inline void InitLoopInfo();

private:
    TPipe *pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> que_;
    const ScatterPaKvCacheTilingData *tilingData_;

    GlobalTensor<T> keyGm_;
    GlobalTensor<T> valueGm_;
    GlobalTensor<T> keyCacheOutGm_;
    GlobalTensor<T> valueCacheOutGm_;
    GlobalTensor<IndexDtype> slotMappingGm_;

    int64_t blockIdx_{0};
    int64_t maxTokens_ = 0;
    int64_t blockFactorOffset_ = 0;
    int64_t alignKHeadSize_ = 0;
    int64_t kHeadSizeIn_ = 0;
    int64_t kHeadSizeloop_ = 0;
    int64_t kHeadSizeTail_ = 0;
    int64_t kNumHeadIn_ = 0;
    int64_t kNumHeadLoop_ = 0;
    int64_t kNumHeadTail_ = 0;
    int64_t alignVHeadSize_ = 0;
    int64_t vHeadSizeIn_ = 0;
    int64_t vHeadSizeloop_ = 0;
    int64_t vHeadSizeTail_ = 0;
    int64_t vNumHeadIn_ = 0;
    int64_t vNumHeadLoop_ = 0;
    int64_t vNumHeadTail_ = 0;
};

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheNormalContiguous<T, IndexDtype, InOutMode>::Init(
    GM_ADDR key, GM_ADDR key_cache_in, GM_ADDR slot_mapping, GM_ADDR value, GM_ADDR value_cache_in,
    GM_ADDR key_cache_out, GM_ADDR value_cache_out)
{
    blockIdx_ = GetBlockIdx();
    blockFactorOffset_ = blockIdx_ * tilingData_->blockFactor;
    keyGm_.SetGlobalBuffer((__gm__ T *)(key));
    slotMappingGm_.SetGlobalBuffer((__gm__ IndexDtype *)(slot_mapping) + blockIdx_ * tilingData_->blockFactor);
    keyCacheOutGm_.SetGlobalBuffer((__gm__ T *)(key_cache_out));

    if constexpr (InOutMode == DUAL_IN_OUT) {
        valueGm_.SetGlobalBuffer((__gm__ T *)(value));
        valueCacheOutGm_.SetGlobalBuffer((__gm__ T *)(value_cache_out));
    }
    pipe_->InitBuffer(que_, 1, tilingData_->ubSize);
    maxTokens_ = tilingData_->numBlocks * tilingData_->blockSize;
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline int64_t ScatterPaKvCacheNormalContiguous<T, IndexDtype, InOutMode>::RoundUp(int64_t x)
{
    int64_t elemNum = ONE_BLK_SIZE / sizeof(T);
    return (x + elemNum - 1) / elemNum * elemNum;
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheNormalContiguous<T, IndexDtype, InOutMode>::CopyInKey(int64_t tokenIdx,
                                                                                             int64_t headIdx,
                                                                                             int64_t sizeOffset,
                                                                                             int64_t headNum,
                                                                                             int64_t headSize)
{
    LocalTensor<T> inputKeyLocal = que_.AllocTensor<T>();

    int64_t keyOffset = tokenIdx * tilingData_->keyStride0 + headIdx * tilingData_->keyStride1 + sizeOffset;

    DataCopyExtParams keyParams = {static_cast<uint16_t>(headNum), static_cast<uint32_t>(headSize * sizeof(T)),
                                   static_cast<uint32_t>(tilingData_->keyStride1 * sizeof(T)),
                                   static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPadExtParams<T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0)};
    DataCopyPad(inputKeyLocal, keyGm_[keyOffset], keyParams, padParams);

    que_.EnQue(inputKeyLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheNormalContiguous<T, IndexDtype, InOutMode>::CopyOutKey(int64_t blockIdx,
                                                                                              int64_t blockOffset,
                                                                                              int64_t headIdx,
                                                                                              int64_t sizeOffset,
                                                                                              int64_t headNum,
                                                                                              int64_t headSize)
{
    LocalTensor<T> inputKeyLocal = que_.DeQue<T>();

    int64_t keyCacheOffset = blockIdx * tilingData_->keyCacheStride0 +
                             blockOffset * tilingData_->keyCacheStride1 +
                             headIdx * tilingData_->keyCacheStride2 + sizeOffset;

    DataCopyExtParams outKeyCacheParams = {
        static_cast<uint16_t>(headNum), static_cast<uint32_t>(headSize * sizeof(T)),
        static_cast<uint32_t>(tilingData_->keyCacheStride2 * sizeof(T)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    event_t eventIdV2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdV2ToS);
    WaitFlag<HardEvent::V_S>(eventIdV2ToS);

    DataCopyPad(keyCacheOutGm_[keyCacheOffset], inputKeyLocal, outKeyCacheParams);

    que_.FreeTensor(inputKeyLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheNormalContiguous<T, IndexDtype, InOutMode>::CopyInValue(int64_t tokenIdx,
                                                                                               int64_t headIdx,
                                                                                               int64_t sizeOffset,
                                                                                               int64_t headNum,
                                                                                               int64_t headSize)
{
    LocalTensor<T> inputValueLocal = que_.AllocTensor<T>();

    int64_t valueOffset = tokenIdx * tilingData_->valueStride0 + headIdx * tilingData_->valueStride1 + sizeOffset;

    DataCopyExtParams valueParams = {static_cast<uint16_t>(headNum), static_cast<uint32_t>(headSize * sizeof(T)),
                                     static_cast<uint32_t>(tilingData_->valueStride1 * sizeof(T)),
                                     static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPadExtParams<T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0)};
    DataCopyPad(inputValueLocal, valueGm_[valueOffset], valueParams, padParams);

    que_.EnQue(inputValueLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheNormalContiguous<T, IndexDtype, InOutMode>::CopyOutValue(int64_t blockIdx,
                                                                                                int64_t blockOffset,
                                                                                                int64_t headIdx,
                                                                                                int64_t sizeOffset,
                                                                                                int64_t headNum,
                                                                                                int64_t headSize)
{
    LocalTensor<T> inputValueLocal = que_.DeQue<T>();

    int64_t valueCacheOffset = blockIdx * tilingData_->valueCacheStride0 +
                               blockOffset * tilingData_->valueCacheStride1 +
                               headIdx * tilingData_->valueCacheStride2 + sizeOffset;

    DataCopyExtParams outValueCacheParams = {
        static_cast<uint16_t>(headNum), static_cast<uint32_t>(headSize * sizeof(T)),
        static_cast<uint32_t>(tilingData_->valueCacheStride2 * sizeof(T)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    event_t eventIdV2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdV2ToS);
    WaitFlag<HardEvent::V_S>(eventIdV2ToS);

    DataCopyPad(valueCacheOutGm_[valueCacheOffset], inputValueLocal, outValueCacheParams);

    que_.FreeTensor(inputValueLocal);
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheNormalContiguous<T, IndexDtype, InOutMode>::InitLoopInfo() {
    int64_t ubHandleSize = tilingData_->ubSize / sizeof(T);
    alignKHeadSize_ = RoundUp(tilingData_->kHeadSize);
    kHeadSizeIn_ = tilingData_->kHeadSize;
    kHeadSizeloop_ = 1;
    kHeadSizeTail_ = 0;
    kNumHeadIn_ = 1;
    kNumHeadLoop_ = tilingData_->numHead;
    if (alignKHeadSize_ > ubHandleSize) {
        kHeadSizeIn_ = RoundUp(ubHandleSize);
        kHeadSizeloop_ = tilingData_->kHeadSize / kHeadSizeIn_;
        kHeadSizeTail_ = tilingData_->kHeadSize - kHeadSizeIn_ * kHeadSizeloop_;
    } else {
        kHeadSizeIn_ = alignKHeadSize_;
        kNumHeadIn_ = ubHandleSize / kHeadSizeIn_;
        kNumHeadLoop_ = tilingData_->numHead / kNumHeadIn_;
        kNumHeadTail_ = tilingData_->numHead - kNumHeadIn_ * kNumHeadLoop_;
    }

    if constexpr (InOutMode == DUAL_IN_OUT) {
        alignVHeadSize_ = RoundUp(tilingData_->vHeadSize);
        vHeadSizeIn_ = tilingData_->vHeadSize;
        vHeadSizeloop_ = 1;
        vHeadSizeTail_ = 0;
        vNumHeadIn_ = 1;
        vNumHeadLoop_ = tilingData_->numHead;
        if (alignVHeadSize_ > ubHandleSize) {
            vHeadSizeIn_ = RoundUp(ubHandleSize);
            vHeadSizeloop_ = tilingData_->vHeadSize / vHeadSizeIn_;
            vHeadSizeTail_ = tilingData_->vHeadSize - vHeadSizeIn_ * vHeadSizeloop_;
        } else {
            vHeadSizeIn_ = alignVHeadSize_;
            vNumHeadIn_ = ubHandleSize / vHeadSizeIn_;
            vNumHeadLoop_ = tilingData_->numHead / vNumHeadIn_;
            vNumHeadTail_ = tilingData_->numHead - vNumHeadIn_ * vNumHeadLoop_;
        }
    }
}

template <typename T, typename IndexDtype, int64_t InOutMode>
__aicore__ inline void ScatterPaKvCacheNormalContiguous<T, IndexDtype, InOutMode>::Process()
{
    if (blockIdx_ >= tilingData_->usedCoreNum) {
        return;
    }
    int64_t curBlockFactor =
        (blockIdx_ == tilingData_->usedCoreNum - 1) ? tilingData_->tailBlockFactor : tilingData_->blockFactor;

    InitLoopInfo();
    for (int64_t idx = 0; idx < curBlockFactor; idx++) {
        int64_t slotIdx = slotMappingGm_.GetValue(idx);
        if (slotIdx < 0 || slotIdx >= maxTokens_) {
            continue;
        }
        int64_t blockIdx = slotIdx / tilingData_->blockSize;
        int64_t blockOffset = slotIdx % tilingData_->blockSize;
        int64_t tokenIdx = blockFactorOffset_ + idx;

        for (int64_t headLoop = 0; headLoop < kNumHeadLoop_; headLoop++) {
            int64_t headIdx = headLoop * kNumHeadIn_;
            for (int64_t sizeLoop = 0; sizeLoop < kHeadSizeloop_; sizeLoop++) {
                int64_t sizeOffset = sizeLoop * kHeadSizeIn_;
                CopyInKey(tokenIdx, headIdx, sizeOffset, kNumHeadIn_, kHeadSizeIn_);
                CopyOutKey(blockIdx, blockOffset, headIdx, sizeOffset, kNumHeadIn_, kHeadSizeIn_);
            }
            if (kHeadSizeTail_ > 0) {
                int64_t sizeOffset = kHeadSizeloop_ * kHeadSizeIn_;
                CopyInKey(tokenIdx, headIdx, sizeOffset, kNumHeadIn_, kHeadSizeTail_);
                CopyOutKey(blockIdx, blockOffset, headIdx, sizeOffset, kNumHeadIn_, kHeadSizeTail_);
            }
        }
        if (kNumHeadTail_ > 0) {
            int64_t headIdx = kNumHeadLoop_ * kNumHeadIn_;
            for (int64_t sizeLoop = 0; sizeLoop < kHeadSizeloop_; sizeLoop++) {
                int64_t sizeOffset = sizeLoop * kHeadSizeIn_;
                CopyInKey(tokenIdx, headIdx, sizeOffset, kNumHeadTail_, kHeadSizeIn_);
                CopyOutKey(blockIdx, blockOffset, headIdx, sizeOffset, kNumHeadTail_, kHeadSizeIn_);
            }
            if (kHeadSizeTail_ > 0) {
                int64_t sizeOffset = kHeadSizeloop_ * kHeadSizeIn_;
                CopyInKey(tokenIdx, headIdx, sizeOffset, kNumHeadTail_, kHeadSizeTail_);
                CopyOutKey(blockIdx, blockOffset, headIdx, sizeOffset, kNumHeadTail_, kHeadSizeTail_);
            }
        }

        if constexpr (InOutMode == DUAL_IN_OUT) {
            for (int64_t headLoop = 0; headLoop < vNumHeadLoop_; headLoop++) {
                int64_t headIdx = headLoop * vNumHeadIn_;
                for (int64_t sizeLoop = 0; sizeLoop < vHeadSizeloop_; sizeLoop++) {
                    int64_t sizeOffset = sizeLoop * vHeadSizeIn_;
                    CopyInValue(tokenIdx, headIdx, sizeOffset, vNumHeadIn_, vHeadSizeIn_);
                    CopyOutValue(blockIdx, blockOffset, headIdx, sizeOffset, vNumHeadIn_, vHeadSizeIn_);
                }
                if (vHeadSizeTail_ > 0) {
                    int64_t sizeOffset = vHeadSizeloop_ * vHeadSizeIn_;
                    CopyInValue(tokenIdx, headIdx, sizeOffset, vNumHeadIn_, vHeadSizeTail_);
                    CopyOutValue(blockIdx, blockOffset, headIdx, sizeOffset, vNumHeadIn_, vHeadSizeTail_);
                }
            }
            if (vNumHeadTail_ > 0) {
                int64_t headIdx = vNumHeadLoop_ * vNumHeadIn_;
                for (int64_t sizeLoop = 0; sizeLoop < vHeadSizeloop_; sizeLoop++) {
                    int64_t sizeOffset = sizeLoop * vHeadSizeIn_;
                    CopyInValue(tokenIdx, headIdx, sizeOffset, vNumHeadTail_, vHeadSizeIn_);
                    CopyOutValue(blockIdx, blockOffset, headIdx, sizeOffset, vNumHeadTail_, vHeadSizeIn_);
                }
                if (vHeadSizeTail_ > 0) {
                    int64_t sizeOffset = vHeadSizeloop_ * vHeadSizeIn_;
                    CopyInValue(tokenIdx, headIdx, sizeOffset, vNumHeadTail_, vHeadSizeTail_);
                    CopyOutValue(blockIdx, blockOffset, headIdx, sizeOffset, vNumHeadTail_, vHeadSizeTail_);
                }
            }
        }
    }
}
} // namespace ScatterPaKvCache
#endif // SCATTER_PA_KV_CACHE_NORMAL_CONTIGUOUS_H_