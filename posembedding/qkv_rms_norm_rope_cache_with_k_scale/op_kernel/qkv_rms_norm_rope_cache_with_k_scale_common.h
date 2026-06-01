/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_COMMON_H_
#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_COMMON_H_

#include "kernel_operator.h"
#include "qkv_rms_norm_rope_cache_with_k_scale_tiling_data.h"

#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_D128_FAST 350128

namespace QkvRmsNormRopeCacheWithKScale {
using namespace AscendC;
using QkvRmsNormRopeCacheWithKScaleKernelTiling::QkvRmsNormRopeCacheWithKScaleTilingData;

constexpr uint32_t QKV_RRC_MIX_AIV_PER_AIC = 2U;
constexpr uint32_t QKV_RRC_DOUBLE_BUFFER_NUM = 2U;
constexpr uint32_t QKV_RRC_HEAD_DIM_D128 = 128U;
constexpr uint32_t QKV_RRC_KIB = 1024U;

constexpr uint8_t QKV_RRC_SYNC_MODE4 = 4U;
constexpr uint64_t QKV_RRC_FLAG_ID_MAX = 16U;
constexpr uint64_t SYNC_A_READY = 0U;
constexpr uint64_t SYNC_C_READY = 1U;
constexpr uint64_t SYNC_C_CONSUMED = 2U;

static_assert(SYNC_A_READY != SYNC_C_READY && SYNC_A_READY != SYNC_C_CONSUMED &&
        SYNC_C_READY != SYNC_C_CONSUMED,
    "QkvRmsNormRopeCacheWithKScale CrossCore flags must keep unique semantic ids.");
static_assert(SYNC_C_CONSUMED < QKV_RRC_FLAG_ID_MAX,
    "QkvRmsNormRopeCacheWithKScale CrossCore base flag ids must not overlap AIV1 offset.");

constexpr uint32_t QKV_RRC_INPUT_ONE_BUFFER_BYTES = 40U * QKV_RRC_KIB;
constexpr uint32_t QKV_RRC_OUTPUT_ONE_BUFFER_BYTES = 64U * QKV_RRC_KIB;
constexpr uint32_t QKV_RRC_INPUT_DB_POOL_BYTES = QKV_RRC_DOUBLE_BUFFER_NUM * QKV_RRC_INPUT_ONE_BUFFER_BYTES;
constexpr uint32_t QKV_RRC_OUTPUT_DB_POOL_BYTES = QKV_RRC_DOUBLE_BUFFER_NUM * QKV_RRC_OUTPUT_ONE_BUFFER_BYTES;

constexpr uint32_t QKV_RRC_ROTATION_L1_BYTES =
    QKV_RRC_HEAD_DIM_D128 * QKV_RRC_HEAD_DIM_D128 * sizeof(bfloat16_t);
constexpr uint32_t QKV_RRC_A_ROT_L1_ONE_BUFFER_BYTES = 64U * QKV_RRC_KIB;
constexpr uint32_t QKV_RRC_A_ROT_L1_BUFFER_NUM = QKV_RRC_DOUBLE_BUFFER_NUM * QKV_RRC_MIX_AIV_PER_AIC;
constexpr uint32_t QKV_RRC_A_ROT_L1_POOL_BYTES =
    QKV_RRC_A_ROT_L1_BUFFER_NUM * QKV_RRC_A_ROT_L1_ONE_BUFFER_BYTES;
constexpr uint32_t QKV_RRC_ROTATION_L1_OFFSET = 0U;
constexpr uint32_t QKV_RRC_A_ROT_L1_POOL_OFFSET = QKV_RRC_ROTATION_L1_OFFSET + QKV_RRC_ROTATION_L1_BYTES;
constexpr uint32_t QKV_RRC_A_ROT_L1_LOGICAL_BUFFER_BYTES =
    QKV_RRC_A_ROT_L1_POOL_BYTES / QKV_RRC_DOUBLE_BUFFER_NUM;

constexpr uint32_t QKV_RRC_RESERVE_UB_BYTES = 40U * QKV_RRC_KIB;
constexpr uint32_t QKV_RRC_GAMMA_UB_BYTES = 1U * QKV_RRC_KIB;
constexpr uint32_t QKV_RRC_COS_SIN_ONE_BUFFER_BYTES = 4U * QKV_RRC_KIB;
constexpr uint32_t QKV_RRC_SLOT_MAPPING_ONE_BUFFER_BYTES = 512U;
constexpr uint32_t QKV_RRC_V_SCALE_UB_BYTES = 512U;
constexpr uint32_t QKV_RRC_V_OUT_ONE_BUFFER_BYTES = 10U * QKV_RRC_KIB;

constexpr uint32_t QKV_RRC_GAMMA_UB_OFFSET = 0U;
constexpr uint32_t QKV_RRC_COS_SIN_DB_POOL_OFFSET = QKV_RRC_GAMMA_UB_OFFSET + QKV_RRC_GAMMA_UB_BYTES;
constexpr uint32_t QKV_RRC_SLOT_MAPPING_DB_POOL_OFFSET =
    QKV_RRC_COS_SIN_DB_POOL_OFFSET + QKV_RRC_DOUBLE_BUFFER_NUM * QKV_RRC_COS_SIN_ONE_BUFFER_BYTES;
constexpr uint32_t QKV_RRC_V_SCALE_UB_OFFSET =
    QKV_RRC_SLOT_MAPPING_DB_POOL_OFFSET + QKV_RRC_DOUBLE_BUFFER_NUM * QKV_RRC_SLOT_MAPPING_ONE_BUFFER_BYTES;
constexpr uint32_t QKV_RRC_V_OUT_DB_POOL_OFFSET = QKV_RRC_V_SCALE_UB_OFFSET + QKV_RRC_V_SCALE_UB_BYTES;
constexpr uint32_t QKV_RRC_RESERVE_USED_BYTES =
    QKV_RRC_V_OUT_DB_POOL_OFFSET + QKV_RRC_DOUBLE_BUFFER_NUM * QKV_RRC_V_OUT_ONE_BUFFER_BYTES;

static_assert(QKV_RRC_RESERVE_USED_BYTES <= QKV_RRC_RESERVE_UB_BYTES,
    "QkvRmsNormRopeCacheWithKScale reserve UB layout exceeds 40 KiB.");

constexpr uint32_t QKV_RRC_INPUT_DB_POOL_OFFSET = 0U;
constexpr uint32_t QKV_RRC_OUTPUT_DB_POOL_OFFSET = QKV_RRC_INPUT_DB_POOL_OFFSET + QKV_RRC_INPUT_DB_POOL_BYTES;
constexpr uint32_t QKV_RRC_RESERVE_UB_OFFSET = QKV_RRC_OUTPUT_DB_POOL_OFFSET + QKV_RRC_OUTPUT_DB_POOL_BYTES;
constexpr uint32_t QKV_RRC_UB_USED_BYTES = QKV_RRC_RESERVE_UB_OFFSET + QKV_RRC_RESERVE_UB_BYTES;

static_assert(QKV_RRC_UB_USED_BYTES <= 248U * QKV_RRC_KIB,
    "QkvRmsNormRopeCacheWithKScale static UB layout exceeds 248 KiB.");

struct TileParam {
    uint32_t tileId;
    uint32_t bufferId;
    uint32_t tokenOffset;
    uint32_t tokenSize;
    uint32_t qkHeadOffset;
    uint32_t qkHeadSize;
    uint32_t vHeadOffset;
    uint32_t vHeadSize;
    uint32_t rowOffset;
    uint32_t rowSize;
};

struct TokenRange {
    uint32_t begin;
    uint32_t end;
};

struct AivSlice {
    uint32_t tokenOffset;
    uint32_t tokenSize;
    uint32_t rowOffset;
    uint32_t rowSize;
};

struct GlobalTensors {
    GM_ADDR qkv;
    GM_ADDR qGamma;
    GM_ADDR kGamma;
    GM_ADDR cosSin;
    GM_ADDR slotMapping;
    GM_ADDR kCache;
    GM_ADDR vCache;
    GM_ADDR kScaleCache;
    GM_ADDR cuSeqLen;
    GM_ADDR rotation;
    GM_ADDR vScale;
    GM_ADDR qOut;
    GM_ADDR qScale;
    GM_ADDR kCacheOut;
    GM_ADDR vCacheOut;
    GM_ADDR kScaleCacheOut;
    GM_ADDR workspace;
};

struct CubeGlobalTensors {
    GlobalTensor<bfloat16_t> rotation;
    GlobalTensor<uint8_t> workspace;
};

struct VecGlobalTensors {
    GlobalTensor<bfloat16_t> qkv;
    GlobalTensor<float> qGamma;
    GlobalTensor<float> kGamma;
    GlobalTensor<float> cosSin;
    GlobalTensor<int32_t> slotMapping;
    GlobalTensor<fp8_e4m3fn_t> kCacheIn;
    GlobalTensor<fp8_e4m3fn_t> vCacheIn;
    GlobalTensor<float> kScaleCacheIn;
    GlobalTensor<int32_t> cuSeqLen;
    GlobalTensor<float> vScale;
    GlobalTensor<fp8_e4m3fn_t> qOut;
    GlobalTensor<float> qScale;
    GlobalTensor<fp8_e4m3fn_t> kCacheOut;
    GlobalTensor<fp8_e4m3fn_t> vCacheOut;
    GlobalTensor<float> kScaleCacheOut;
    GlobalTensor<uint8_t> workspace;
};

struct StaticLocalTensors {
    LocalTensor<bfloat16_t> rotationL1Nz;
    LocalTensor<bfloat16_t> aRotL1Nz[QKV_RRC_DOUBLE_BUFFER_NUM];
    LocalTensor<bfloat16_t> vUb[QKV_RRC_DOUBLE_BUFFER_NUM];
    LocalTensor<float> qkAfterCubeUb[QKV_RRC_DOUBLE_BUFFER_NUM];
    LocalTensor<fp8_e4m3fn_t> qkFp8StagingUb[QKV_RRC_DOUBLE_BUFFER_NUM];
    LocalTensor<float> gammaUb;
    LocalTensor<float> cosSinUb[QKV_RRC_DOUBLE_BUFFER_NUM];
    LocalTensor<int32_t> slotMappingUb[QKV_RRC_DOUBLE_BUFFER_NUM];
    LocalTensor<float> vScaleUb;
    LocalTensor<fp8_e4m3fn_t> vOutUb[QKV_RRC_DOUBLE_BUFFER_NUM];
};

__aicore__ inline uint32_t CeilDiv(uint32_t value, uint32_t factor)
{
    if (factor == 0U) {
        return 0U;
    }
    return (value + factor - 1U) / factor;
}

__aicore__ inline uint32_t MinU32(uint32_t lhs, uint32_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline uint32_t MakeDataBufferId(uint32_t tileBufferId)
{
    return tileBufferId & (QKV_RRC_DOUBLE_BUFFER_NUM - 1U);
}

template <const pipe_t pipe>
__aicore__ inline void SetAivToAic(uint64_t syncFlag)
{
#ifndef __CCE_KT_TEST__
    CrossCoreSetFlag<QKV_RRC_SYNC_MODE4, pipe>(syncFlag);
#else
    (void)syncFlag;
#endif
}

template <const pipe_t pipe>
__aicore__ inline void WaitAivToAic(uint64_t syncFlag)
{
#ifndef __CCE_KT_TEST__
    CrossCoreWaitFlag<QKV_RRC_SYNC_MODE4, pipe>(syncFlag + QKV_RRC_FLAG_ID_MAX);
    CrossCoreWaitFlag<QKV_RRC_SYNC_MODE4, pipe>(syncFlag);
#else
    (void)syncFlag;
#endif
}

template <const pipe_t pipe>
__aicore__ inline void SetAicToAiv(uint64_t syncFlag)
{
#ifndef __CCE_KT_TEST__
    CrossCoreSetFlag<QKV_RRC_SYNC_MODE4, pipe>(syncFlag + QKV_RRC_FLAG_ID_MAX);
    CrossCoreSetFlag<QKV_RRC_SYNC_MODE4, pipe>(syncFlag);
#else
    (void)syncFlag;
#endif
}

template <const pipe_t pipe>
__aicore__ inline void WaitAicToAiv(uint64_t syncFlag)
{
#ifndef __CCE_KT_TEST__
    CrossCoreWaitFlag<QKV_RRC_SYNC_MODE4, pipe>(syncFlag);
#else
    (void)syncFlag;
#endif
}

__aicore__ inline void NotifyAReadyToAic()
{
    SetAivToAic<PIPE_MTE3>(SYNC_A_READY);
}

__aicore__ inline void WaitAReadyFromAiv()
{
    WaitAivToAic<PIPE_MTE1>(SYNC_A_READY);
}

__aicore__ inline void NotifyCReadyToAiv()
{
    SetAicToAiv<PIPE_FIX>(SYNC_C_READY);
}

__aicore__ inline void WaitCReadyFromAic()
{
    WaitAicToAiv<PIPE_V>(SYNC_C_READY);
}

__aicore__ inline void NotifyCConsumedToAic()
{
    SetAivToAic<PIPE_MTE3>(SYNC_C_CONSUMED);
}

__aicore__ inline void WaitCConsumedFromAiv()
{
    WaitAivToAic<PIPE_FIX>(SYNC_C_CONSUMED);
}

__aicore__ inline uint32_t GetAicCubeIndex()
{
    return GetBlockIdx();
}

__aicore__ inline uint32_t GetAivCubeIndex()
{
    return GetBlockIdx() / QKV_RRC_MIX_AIV_PER_AIC;
}

__aicore__ inline AivSlice MakeAivSlice(const TileParam &tile, uint32_t qkHeadTile, uint32_t aivLocalId)
{
    AivSlice slice = {0U, 0U, 0U, 0U};
    const uint32_t aiv0TokenSize = CeilDiv(tile.tokenSize, QKV_RRC_MIX_AIV_PER_AIC);
    const uint32_t tokenPrefix = aivLocalId == 0U ? 0U : aiv0TokenSize;
    const uint32_t tokenSize = aivLocalId == 0U ? aiv0TokenSize : tile.tokenSize - aiv0TokenSize;

    slice.tokenOffset = tile.tokenOffset + tokenPrefix;
    slice.tokenSize = tokenSize;
    slice.rowOffset = tokenPrefix * qkHeadTile;
    slice.rowSize = tokenSize * qkHeadTile;
    return slice;
}

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_COMMON_H_
