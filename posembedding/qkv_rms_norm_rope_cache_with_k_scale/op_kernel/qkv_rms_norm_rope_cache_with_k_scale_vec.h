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
        BindLocalTensors();
        BindGlobalTensors(tensors);
    }

    __aicore__ inline void ComputeTile(const TileParam &tile)
    {
        const AivSlice slice = MakeAivSlice(tile, tilingData_->qkHeadTile, aivLocalId_);
        const uint32_t bufferId = MakeDataBufferId(tile.bufferId);
        LocalTensor<bfloat16_t> aRotL1Nz = aRotL1Pool_[bufferId * QKV_RRC_A_ROT_L1_LOGICAL_BUFFER_ELEMENTS];
        LocalTensor<bfloat16_t> vUb = inputUb_[bufferId * QKV_RRC_INPUT_ONE_BUFFER_ELEMENTS];
        LocalTensor<float> qkAfterCubeUb = outputUb_[bufferId * QKV_RRC_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS];
        LocalTensor<fp8_e4m3fn_t> qkFp8StagingUb = qkAfterCubeUb.template ReinterpretCast<fp8_e4m3fn_t>();
        LocalTensor<float> cosSinUb = cosSinUb_[bufferId * QKV_RRC_COS_SIN_ONE_BUFFER_ELEMENTS];
        LocalTensor<int32_t> slotMappingUb = slotMappingUb_[bufferId * QKV_RRC_SLOT_MAPPING_ONE_BUFFER_ELEMENTS];
        LocalTensor<fp8_e4m3fn_t> vOutUb = vOutUb_[bufferId * QKV_RRC_V_OUT_ONE_BUFFER_ELEMENTS];
        (void)globalTensors_;
        (void)aRotL1Nz;
        (void)vUb;
        (void)qkAfterCubeUb;
        (void)qkFp8StagingUb;
        (void)gammaUb_;
        (void)cosSinUb;
        (void)slotMappingUb;
        (void)vScaleUb_;
        (void)vOutUb;
        (void)slice;
    }

private:
    __aicore__ inline void BindLocalTensors()
    {
        aRotL1Pool_ =
            LocalTensor<bfloat16_t>(TPosition::TSCM, QKV_RRC_A_ROT_L1_POOL_OFFSET, QKV_RRC_A_ROT_L1_POOL_ELEMENTS);
        inputUb_ =
            LocalTensor<bfloat16_t>(TPosition::LCM, QKV_RRC_INPUT_DB_POOL_OFFSET, QKV_RRC_INPUT_DB_POOL_ELEMENTS);
        outputUb_ =
            LocalTensor<float>(TPosition::LCM, QKV_RRC_OUTPUT_DB_POOL_OFFSET, QKV_RRC_OUTPUT_DB_POOL_FLOAT_ELEMENTS);
        BindReserveUbTensors();
    }

    __aicore__ inline void BindReserveUbTensors()
    {
        gammaUb_ = LocalTensor<float>(TPosition::LCM,
            QKV_RRC_RESERVE_UB_OFFSET + QKV_RRC_GAMMA_UB_OFFSET,
            QKV_RRC_GAMMA_UB_ELEMENTS);
        cosSinUb_ = LocalTensor<float>(TPosition::LCM,
            QKV_RRC_RESERVE_UB_OFFSET + QKV_RRC_COS_SIN_DB_POOL_OFFSET,
            QKV_RRC_COS_SIN_DB_POOL_ELEMENTS);
        slotMappingUb_ = LocalTensor<int32_t>(TPosition::LCM,
            QKV_RRC_RESERVE_UB_OFFSET + QKV_RRC_SLOT_MAPPING_DB_POOL_OFFSET,
            QKV_RRC_SLOT_MAPPING_DB_POOL_ELEMENTS);
        vScaleUb_ = LocalTensor<float>(TPosition::LCM,
            QKV_RRC_RESERVE_UB_OFFSET + QKV_RRC_V_SCALE_UB_OFFSET,
            QKV_RRC_V_SCALE_UB_ELEMENTS);
        vOutUb_ = LocalTensor<fp8_e4m3fn_t>(TPosition::LCM,
            QKV_RRC_RESERVE_UB_OFFSET + QKV_RRC_V_OUT_DB_POOL_OFFSET,
            QKV_RRC_V_OUT_DB_POOL_ELEMENTS);
    }

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
    LocalTensor<bfloat16_t> aRotL1Pool_;
    LocalTensor<bfloat16_t> inputUb_;
    LocalTensor<float> outputUb_;
    LocalTensor<float> gammaUb_;
    LocalTensor<float> cosSinUb_;
    LocalTensor<int32_t> slotMappingUb_;
    LocalTensor<float> vScaleUb_;
    LocalTensor<fp8_e4m3fn_t> vOutUb_;
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_VEC_H_
