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
 * \file scatter_pa_kv_cache.h
 * \brief
 */

#ifndef ASCEND_SCATTER_PA_KV_CACHE_ND_H
#define ASCEND_SCATTER_PA_KV_CACHE_ND_H
#include "scatter_pa_kv_cache_base.h"

namespace ScatterPaKvCache {
template <typename T>
class ScatterPaKvCacheNd : public ScatterPaKvCacheBase {
public:
    __aicore__ inline ScatterPaKvCacheNd()
    {
    }

    template <bool NCT = false>
    __aicore__ inline void ProcessKVUnequal(GM_ADDR keyIn, GM_ADDR valueIn, GM_ADDR keyCacheIn, GM_ADDR valueCacheIn,
                                            GM_ADDR slotMapping, GM_ADDR keyCacheOut, GM_ADDR valueCacheOut)
    {
        InitGlobalTensor<T>(keyInputGt_, keyIn);
        InitGlobalTensor<T>(valueInputGt_, valueIn);
        InitGlobalTensor<int32_t>(slotInputGt_, slotMapping);
        InitGlobalTensor<T>(keyOutputGt_, keyCacheOut);
        InitGlobalTensor<T>(valueOutputGt_, valueCacheOut);

        tokenSizeK_ = numHeads_ * headSizeK_;
        tokenSizeV_ = numHeads_ * headSizeV_;
        PrepareCopy<T>(tokenSizeK_, loopK_, tailK_, MAX_UB_USED, queBindK_);
        PrepareCopy<T>(tokenSizeV_, loopV_, tailV_, MAX_UB_USED, queBindV_);
        AllocateTask();
        for (uint32_t i = 0; i < perCoreTaskNum_; i++) {
            AscendC::GlobalTensor<T> keyInputGtOffset_ = keyInputGt_[offsetK_];
            AscendC::GlobalTensor<T> valueInputGtOffset_ = valueInputGt_[offsetV_];
            if constexpr (NCT) {
                CopyToCache<T, true>(i, tokenSizeK_, loopK_, tailK_, keyInputGtOffset_, keyOutputGt_, slotInputGt_,
                                     MAX_UB_USED, queBindK_);
                CopyToCache<T, true>(i, tokenSizeV_, loopV_, tailV_, valueInputGtOffset_, valueOutputGt_, slotInputGt_,
                                     MAX_UB_USED, queBindV_);
            } else {
                CopyToCache<T>(i, tokenSizeK_, loopK_, tailK_, keyInputGt_, keyOutputGt_, slotInputGt_, MAX_UB_USED,
                               queBindK_);
                CopyToCache<T>(i, tokenSizeV_, loopV_, tailV_, valueInputGt_, valueOutputGt_, slotInputGt_, MAX_UB_USED,
                               queBindV_);
            }
        }
    }

    template <bool NCT = false>
    __aicore__ inline void ProcessKVUnequalIncrement(GM_ADDR keyIn, GM_ADDR valueIn, GM_ADDR keyCacheIn,
                                                     GM_ADDR valueCacheIn, GM_ADDR slotMapping, GM_ADDR keyCacheOut,
                                                     GM_ADDR valueCacheOut)
    {
        InitGlobalTensor<T>(keyInputGt_, keyIn);
        InitGlobalTensor<T>(valueInputGt_, valueIn);
        InitGlobalTensor<int32_t>(slotInputGt_, slotMapping);
        InitGlobalTensor<T>(keyOutputGt_, keyCacheOut);
        InitGlobalTensor<T>(valueOutputGt_, valueCacheOut);

        tokenSizeK_ = numHeads_ * headSizeK_;
        tokenSizeV_ = numHeads_ * headSizeV_;
        numBlocksK_ = tokenSizeK_ * sizeof(T) / BLOCK_SIZE;
        numBlocksV_ = tokenSizeV_ * sizeof(T) / BLOCK_SIZE;
        uint32_t tokenSize = tokenSizeK_ > tokenSizeV_ ? tokenSizeK_ : tokenSizeV_;
        InitTBuf<T>(tokenBuf_, tokenSize);
        tokenLocal_ = tokenBuf_.Get<T>();
        AllocateTask();
        AscendC::DataCopyParams copyParamsK = {1, static_cast<uint16_t>(numBlocksK_), 0, 0};
        AscendC::DataCopyParams copyParamsV = {1, static_cast<uint16_t>(numBlocksV_), 0, 0};
        for (uint32_t i = 0; i < perCoreTaskNum_; i++) {
            uint32_t startK = (i + startTaskId_) * tokenSizeK_;
            uint32_t startV = (i + startTaskId_) * tokenSizeV_;
            if constexpr (NCT) {
                startK = (i + startTaskId_) * strideK_ + offsetK_;
                startV = (i + startTaskId_) * strideK_ + offsetV_;
            }
            int64_t slotValue = (int64_t)(slotInputGt_.GetValue(i + startTaskId_));
            if (slotValue < 0)
                continue;
            uint64_t cacheStartK = static_cast<uint64_t>(slotValue) * static_cast<uint64_t>(tokenSizeK_);
            uint64_t cacheStartV = static_cast<uint64_t>(slotValue) * static_cast<uint64_t>(tokenSizeV_);
            CopyKvCache(keyInputGt_, tokenLocal_, keyOutputGt_, startK, cacheStartK, copyParamsK, copyParamsK);
            CopyKvCache(valueInputGt_, tokenLocal_, valueOutputGt_, startV, cacheStartV, copyParamsV, copyParamsV);
        }
    }

    template <bool NCT = false>
    __aicore__ inline void ProcessKVEqual(GM_ADDR keyIn, GM_ADDR valueIn, GM_ADDR keyCacheIn, GM_ADDR valueCacheIn,
                                          GM_ADDR slotMapping, GM_ADDR keyCacheOut, GM_ADDR valueCacheOut)
    {
        InitGlobalTensor<T>(keyInputGt_, keyIn);
        InitGlobalTensor<T>(valueInputGt_, valueIn);
        InitGlobalTensor<int32_t>(slotInputGt_, slotMapping);
        InitGlobalTensor<T>(keyOutputGt_, keyCacheOut);
        InitGlobalTensor<T>(valueOutputGt_, valueCacheOut);

        tokenSizeK_ = numHeads_ * headSizeK_;
        numBlocksK_ = tokenSizeK_ * sizeof(T) / BLOCK_SIZE;
        InitTBuf<T>(tokenBuf_, tokenSizeK_);
        tokenLocal_ = tokenBuf_.Get<T>();

        AllocateTask();
        AscendC::DataCopyParams copyParams = {1, static_cast<uint16_t>(numBlocksK_), 0, 0};
        for (uint32_t i = 0; i < perCoreTaskNum_; i++) {
            uint32_t start = (i + startTaskId_) * tokenSizeK_;
            uint32_t startK = (i + startTaskId_) * strideK_ + offsetK_;
            uint32_t startV = (i + startTaskId_) * strideK_ + offsetV_;
            int64_t slotValue = (int64_t)(slotInputGt_.GetValue(i + startTaskId_));
            if (slotValue < 0)
                continue;
            uint64_t cacheStart = static_cast<uint64_t>(slotValue) * static_cast<uint64_t>(tokenSizeK_);
            if constexpr (NCT) {
                CopyKvCache(keyInputGt_, tokenLocal_, keyOutputGt_, startK, cacheStart, copyParams, copyParams);
                CopyKvCache(valueInputGt_, tokenLocal_, valueOutputGt_, startV, cacheStart, copyParams, copyParams);
            } else {
                CopyKvCache(keyInputGt_, tokenLocal_, keyOutputGt_, start, cacheStart, copyParams, copyParams);
                CopyKvCache(valueInputGt_, tokenLocal_, valueOutputGt_, start, cacheStart, copyParams, copyParams);
            }
        }
    }

private:
    uint32_t tokenSizeK_ = 0;
    uint32_t tokenSizeV_ = 0;
    uint32_t numBlocksK_ = 0;
    uint32_t numBlocksV_ = 0;
    uint32_t loopK_ = 0;
    uint32_t tailK_ = 0;
    uint32_t loopV_ = 0;
    uint32_t tailV_ = 0;

    AscendC::LocalTensor<T> tokenLocal_;
    AscendC::GlobalTensor<T> keyInputGt_;
    AscendC::GlobalTensor<T> keyOutputGt_;
    AscendC::GlobalTensor<T> valueInputGt_;
    AscendC::GlobalTensor<T> valueOutputGt_;
    AscendC::GlobalTensor<int32_t> slotInputGt_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tokenBuf_;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, BUFFER_NUM> queBindK_;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, BUFFER_NUM> queBindV_;
};
} // namespace ScatterPaKvCache
#endif