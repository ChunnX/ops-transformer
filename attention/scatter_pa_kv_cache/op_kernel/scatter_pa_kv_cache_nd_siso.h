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
 * \file scatter_pa_kv_cache_nd_siso.h
 * \brief
 */

#ifndef ASCEND_SCATTER_PA_KV_CACHE_ND_SISO_H
#define ASCEND_SCATTER_PA_KV_CACHE_ND_SISO_H
#include "scatter_pa_kv_cache_base.h"

namespace ScatterPaKvCache {
template <typename T>
class ScatterPaKvCacheNdSiso : public ScatterPaKvCacheBase {
public:
    __aicore__ inline ScatterPaKvCacheNdSiso()
    {
    }

    template <bool NCT = false>
    __aicore__ inline void ProcessKey(GM_ADDR keyIn, GM_ADDR keyCacheIn, GM_ADDR slotMapping, GM_ADDR keyCacheOut)
    {
        InitGlobalTensor<T>(keyInputGt_, keyIn);
        InitGlobalTensor<int32_t>(slotInputGt_, slotMapping);
        InitGlobalTensor<T>(keyOutputGt_, keyCacheOut);
        tokenSizeK_ = numHeads_ * headSizeK_;
        PrepareCopy<T>(tokenSizeK_, loopK_, tailK_, MAX_UB_USED, queBindK_);
        AllocateTask();
        for (uint32_t i = 0; i < perCoreTaskNum_; i++) {
            AscendC::GlobalTensor<T> keyInputGtOffset_ = keyInputGt_[offsetK_];
            if constexpr (NCT) {
                CopyToCache<T, true>(i, tokenSizeK_, loopK_, tailK_, keyInputGtOffset_, keyOutputGt_, slotInputGt_,
                                     MAX_UB_USED, queBindK_);
            } else {
                CopyToCache<T, false>(i, tokenSizeK_, loopK_, tailK_, keyInputGt_, keyOutputGt_, slotInputGt_,
                                      MAX_UB_USED, queBindK_);
            }
        }
    }

    template <bool NCT = false>
    __aicore__ inline void ProcessKeyIncrement(GM_ADDR keyIn, GM_ADDR keyCacheIn, GM_ADDR slotMapping,
                                               GM_ADDR keyCacheOut)
    {
        InitGlobalTensor<T>(keyInputGt_, keyIn);
        InitGlobalTensor<int32_t>(slotInputGt_, slotMapping);
        InitGlobalTensor<T>(keyOutputGt_, keyCacheOut);
        tokenSizeK_ = numHeads_ * headSizeK_;
        numBlocksK_ = tokenSizeK_ * sizeof(T) / BLOCK_SIZE;
        InitTBuf<T>(tokenBuf_, tokenSizeK_);
        tokenLocal_ = tokenBuf_.Get<T>();
        AllocateTask();
        AscendC::DataCopyParams copyParams = {1, static_cast<uint16_t>(numBlocksK_), 0, 0};
        for (uint32_t i = 0; i < perCoreTaskNum_; i++) {
            uint32_t start = (i + startTaskId_) * tokenSizeK_;
            if constexpr (NCT) {
                start = (i + startTaskId_) * strideK_ + offsetK_;
            }
            int64_t slotValue = (int64_t)(slotInputGt_.GetValue(i + startTaskId_));
            if (slotValue < 0)
                continue;
            uint64_t cacheStart = static_cast<uint64_t>(slotValue) * static_cast<uint64_t>(tokenSizeK_);
            CopyKvCache(keyInputGt_, tokenLocal_, keyOutputGt_, start, cacheStart, copyParams, copyParams);
        }
    }

private:
    uint32_t tokenSizeK_ = 0;
    uint32_t numBlocksK_ = 0;
    uint32_t numBlocksRowsK_ = 0;
    uint32_t loopK_ = 0;
    uint32_t tailK_ = 0;

    AscendC::LocalTensor<T> tokenLocal_;
    AscendC::GlobalTensor<T> keyInputGt_;
    AscendC::GlobalTensor<T> keyOutputGt_;
    AscendC::GlobalTensor<int32_t> slotInputGt_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tokenBuf_;
    AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, BUFFER_NUM> queBindK_;
};
} // namespace ScatterPaKvCache
#endif