/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_VEC_H_
#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_VEC_H_

#include "qkv_rms_norm_rope_cache_with_k_scale_common.h"

namespace QkvRmsNormRopeCacheWithKScale {

class D128FastVec {
public:
    __aicore__ inline void Init(const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData, uint32_t aivLocalId,
        const GlobalTensors &tensors)
    {
        tilingData_ = tilingData;
        aivLocalId_ = aivLocalId;
        BindGlobalTensors(tensors);
    }

    __aicore__ inline void ComputeTile(const TileParam &tile, const StaticLocalTensors &localTensors)
    {
        const AivSlice slice = MakeAivSlice(tile, tilingData_->qkHeadTile, aivLocalId_);
        const uint32_t bufferId = MakeDataBufferId(tile.bufferId);
        (void)globalTensors_;
        (void)localTensors.aRotL1Nz[bufferId];
        (void)localTensors.vUb[bufferId];
        (void)localTensors.qkAfterCubeUb[bufferId];
        (void)localTensors.qkFp8StagingUb[bufferId];
        (void)localTensors.gammaUb;
        (void)localTensors.cosSinUb[bufferId];
        (void)localTensors.slotMappingUb[bufferId];
        (void)localTensors.vScaleUb;
        (void)localTensors.vOutUb[bufferId];
        (void)slice;
    }

private:
    __aicore__ inline void BindGlobalTensors(const GlobalTensors &tensors)
    {
        globalTensors_.qkv.SetGlobalBuffer((__gm__ bfloat16_t *)tensors.qkv);
        globalTensors_.qGamma.SetGlobalBuffer((__gm__ float *)tensors.qGamma);
        globalTensors_.kGamma.SetGlobalBuffer((__gm__ float *)tensors.kGamma);
        globalTensors_.cosSin.SetGlobalBuffer((__gm__ float *)tensors.cosSin);
        globalTensors_.slotMapping.SetGlobalBuffer((__gm__ int32_t *)tensors.slotMapping);
        globalTensors_.kCacheIn.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)tensors.kCache);
        globalTensors_.vCacheIn.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)tensors.vCache);
        globalTensors_.kScaleCacheIn.SetGlobalBuffer((__gm__ float *)tensors.kScaleCache);
        globalTensors_.cuSeqLen.SetGlobalBuffer((__gm__ int32_t *)tensors.cuSeqLen);
        globalTensors_.vScale.SetGlobalBuffer((__gm__ float *)tensors.vScale);
        globalTensors_.qOut.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)tensors.qOut);
        globalTensors_.qScale.SetGlobalBuffer((__gm__ float *)tensors.qScale);
        globalTensors_.kCacheOut.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)tensors.kCacheOut);
        globalTensors_.vCacheOut.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)tensors.vCacheOut);
        globalTensors_.kScaleCacheOut.SetGlobalBuffer((__gm__ float *)tensors.kScaleCacheOut);
        globalTensors_.workspace.SetGlobalBuffer((__gm__ uint8_t *)tensors.workspace);
    }

    const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData_;
    uint32_t aivLocalId_;
    VecGlobalTensors globalTensors_;
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_VEC_H_
