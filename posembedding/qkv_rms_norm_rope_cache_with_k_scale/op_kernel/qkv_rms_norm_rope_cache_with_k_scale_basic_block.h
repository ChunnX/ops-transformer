/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_BASIC_BLOCK_H_
#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_BASIC_BLOCK_H_

#include "qkv_rms_norm_rope_cache_with_k_scale_cube.h"
#include "qkv_rms_norm_rope_cache_with_k_scale_vec.h"

namespace QkvRmsNormRopeCacheWithKScale {

class D128FastBasicBlock {
public:
    __aicore__ inline void Init(const GlobalTensors &tensors,
        const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData)
    {
        const uint32_t cubeIndex = GetCubeIndex();
        const uint32_t aivLocalId = g_coreType == AIC ? 0U : GetSubBlockIdx();
        BindGlobalTensors(tensors);
        InitStaticBuffers();
        cube_.Init(cubeIndex);
        vec_.Init(tilingData, aivLocalId);
        InitIntraCoreEvents();
    }

    __aicore__ inline void ComputeTile(const TileParam &tile)
    {
        if (g_coreType == AIC) {
            cube_.ComputeTile(tile, globalTensors_, localTensors_);
        } else {
            vec_.ComputeTile(tile, globalTensors_, localTensors_);
        }
    }

    __aicore__ inline void End()
    {
        EndIntraCoreEvents();
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
        globalTensors_.rotation.SetGlobalBuffer((__gm__ bfloat16_t *)tensors.rotation);
        globalTensors_.vScale.SetGlobalBuffer((__gm__ float *)tensors.vScale);
        globalTensors_.qOut.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)tensors.qOut);
        globalTensors_.qScale.SetGlobalBuffer((__gm__ float *)tensors.qScale);
        globalTensors_.kCacheOut.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)tensors.kCacheOut);
        globalTensors_.vCacheOut.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)tensors.vCacheOut);
        globalTensors_.kScaleCacheOut.SetGlobalBuffer((__gm__ float *)tensors.kScaleCacheOut);
        globalTensors_.workspace.SetGlobalBuffer((__gm__ uint8_t *)tensors.workspace);
    }

    __aicore__ inline void InitStaticBuffers()
    {
        InitStaticL1Tensors();
        if (g_coreType != AIC) {
            InitAivStaticBuffers();
        }
    }

    __aicore__ inline void InitStaticL1Tensors()
    {
        localTensors_.rotationL1Nz = LocalTensor<bfloat16_t>(
            TPosition::TSCM, QKV_RRC_ROTATION_L1_OFFSET, QKV_RRC_ROTATION_L1_BYTES / sizeof(bfloat16_t));
        for (uint32_t i = 0U; i < QKV_RRC_DOUBLE_BUFFER_NUM; ++i) {
            localTensors_.aRotL1Nz[i] = LocalTensor<bfloat16_t>(TPosition::TSCM,
                QKV_RRC_A_ROT_L1_POOL_OFFSET + i * QKV_RRC_A_ROT_L1_LOGICAL_BUFFER_BYTES,
                QKV_RRC_A_ROT_L1_LOGICAL_BUFFER_BYTES / sizeof(bfloat16_t));
        }
    }

    __aicore__ inline void InitAivStaticBuffers()
    {
        constexpr uint32_t inputOneBufferElements = QKV_RRC_INPUT_ONE_BUFFER_BYTES / sizeof(bfloat16_t);
        for (uint32_t i = 0U; i < QKV_RRC_DOUBLE_BUFFER_NUM; ++i) {
            localTensors_.vUb[i] = LocalTensor<bfloat16_t>(TPosition::LCM,
                QKV_RRC_INPUT_DB_POOL_OFFSET + i * QKV_RRC_INPUT_ONE_BUFFER_BYTES,
                inputOneBufferElements);
        }

        constexpr uint32_t outputOneBufferFloatElements = QKV_RRC_OUTPUT_ONE_BUFFER_BYTES / sizeof(float);
        for (uint32_t i = 0U; i < QKV_RRC_DOUBLE_BUFFER_NUM; ++i) {
            localTensors_.qkAfterCubeUb[i] = LocalTensor<float>(TPosition::LCM,
                QKV_RRC_OUTPUT_DB_POOL_OFFSET + i * QKV_RRC_OUTPUT_ONE_BUFFER_BYTES,
                outputOneBufferFloatElements);
            localTensors_.qkFp8StagingUb[i] =
                localTensors_.qkAfterCubeUb[i].template ReinterpretCast<fp8_e4m3fn_t>();
        }

        BindReserveUbTensors();
    }

    __aicore__ inline void BindReserveUbTensors()
    {
        localTensors_.gammaUb = LocalTensor<float>(TPosition::LCM,
            QKV_RRC_RESERVE_UB_OFFSET + QKV_RRC_GAMMA_UB_OFFSET,
            QKV_RRC_GAMMA_UB_BYTES / sizeof(float));

        for (uint32_t i = 0U; i < QKV_RRC_DOUBLE_BUFFER_NUM; ++i) {
            localTensors_.cosSinUb[i] = LocalTensor<float>(TPosition::LCM,
                QKV_RRC_RESERVE_UB_OFFSET + QKV_RRC_COS_SIN_DB_POOL_OFFSET +
                    i * QKV_RRC_COS_SIN_ONE_BUFFER_BYTES,
                QKV_RRC_COS_SIN_ONE_BUFFER_BYTES / sizeof(float));
            localTensors_.slotMappingUb[i] = LocalTensor<int32_t>(TPosition::LCM,
                QKV_RRC_RESERVE_UB_OFFSET + QKV_RRC_SLOT_MAPPING_DB_POOL_OFFSET +
                    i * QKV_RRC_SLOT_MAPPING_ONE_BUFFER_BYTES,
                QKV_RRC_SLOT_MAPPING_ONE_BUFFER_BYTES / sizeof(int32_t));
            localTensors_.vOutUb[i] = LocalTensor<fp8_e4m3fn_t>(TPosition::LCM,
                QKV_RRC_RESERVE_UB_OFFSET + QKV_RRC_V_OUT_DB_POOL_OFFSET +
                    i * QKV_RRC_V_OUT_ONE_BUFFER_BYTES,
                QKV_RRC_V_OUT_ONE_BUFFER_BYTES / sizeof(fp8_e4m3fn_t));
        }

        localTensors_.vScaleUb = LocalTensor<float>(TPosition::LCM,
            QKV_RRC_RESERVE_UB_OFFSET + QKV_RRC_V_SCALE_UB_OFFSET,
            QKV_RRC_V_SCALE_UB_BYTES / sizeof(float));
    }

    __aicore__ inline void InitIntraCoreEvents()
    {
        // T05 will add the real event-id initialization. T01 keeps lifecycle hooks in place.
    }

    __aicore__ inline void EndIntraCoreEvents()
    {
        // T05 will add the matching event-id waits.
    }

    BoundGlobalTensors globalTensors_;
    StaticLocalTensors localTensors_;
    D128FastCube cube_;
    D128FastVec vec_;
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_BASIC_BLOCK_H_
