/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_D128_FAST_H_
#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_D128_FAST_H_

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

struct TokenRange {
    uint32_t begin;
    uint32_t end;
};

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

struct BoundGlobalTensors {
    GlobalTensor<bfloat16_t> qkv;
    GlobalTensor<float> qGamma;
    GlobalTensor<float> kGamma;
    GlobalTensor<float> cosSin;
    GlobalTensor<int32_t> slotMapping;
    GlobalTensor<fp8_e4m3fn_t> kCacheIn;
    GlobalTensor<fp8_e4m3fn_t> vCacheIn;
    GlobalTensor<float> kScaleCacheIn;
    GlobalTensor<int32_t> cuSeqLen;
    GlobalTensor<bfloat16_t> rotation;
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

__aicore__ inline uint32_t GetCubeIndex()
{
    if (g_coreType == AIC) {
        return GetBlockIdx();
    }
    return GetBlockIdx() / QKV_RRC_MIX_AIV_PER_AIC;
}

__aicore__ inline uint32_t GetCubeGroupNum()
{
    if (g_coreType == AIC) {
        return GetBlockNum();
    }
    return CeilDiv(GetBlockNum(), QKV_RRC_MIX_AIV_PER_AIC);
}

__aicore__ inline TokenRange MakeCoreTokenRange(uint32_t totalTokens, uint32_t cubeIndex, uint32_t cubeGroupNum)
{
    TokenRange range = {0U, 0U};
    if (cubeGroupNum == 0U || cubeIndex >= cubeGroupNum || totalTokens == 0U) {
        return range;
    }

    const uint32_t baseTokenSize = totalTokens / cubeGroupNum;
    const uint32_t extraTokenNum = totalTokens - baseTokenSize * cubeGroupNum;
    range.begin = cubeIndex * baseTokenSize + MinU32(cubeIndex, extraTokenNum);
    range.end = range.begin + baseTokenSize + (cubeIndex < extraTokenNum ? 1U : 0U);
    return range;
}

__aicore__ inline TileParam MakeTileParam(uint32_t tileId, uint32_t tokenOffset, uint32_t tokenSize,
    const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData)
{
    TileParam tile = {0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U, 0U};
    tile.tileId = tileId;
    tile.bufferId = tileId & 1U;
    tile.tokenOffset = tokenOffset;
    if (tile.tokenOffset >= tilingData->totalTokens || tokenSize == 0U) {
        return tile;
    }

    tile.tokenSize = MinU32(MinU32(tokenSize, tilingData->tokenTile), tilingData->totalTokens - tile.tokenOffset);
    tile.qkHeadOffset = 0U;
    tile.qkHeadSize = tilingData->qkHeadTile;
    tile.vHeadOffset = 0U;
    tile.vHeadSize = tilingData->vHeadTile;
    tile.rowOffset = 0U;
    tile.rowSize = tile.tokenSize * tilingData->qkHeadTile;
    return tile;
}

__aicore__ inline AivSlice MakeAivSlice(const TileParam &tile, uint32_t qkHeadTile,
    uint32_t aivLocalId)
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

class D128FastBasicBlock {
public:
    __aicore__ inline void Init(const GlobalTensors &tensors,
        const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData)
    {
        tensors_ = tensors;
        tilingData_ = tilingData;
        cubeIndex_ = GetCubeIndex();
        aivLocalId_ = g_coreType == AIC ? 0U : GetSubBlockIdx();
        BindGlobalTensors();
        InitStaticBuffers();
        InitIntraCoreEvents();
    }

    __aicore__ inline void ComputeTile(const TileParam &tile)
    {
        if (g_coreType == AIC) {
            ComputeAicTile(tile);
        } else {
            ComputeAivTile(tile);
        }
    }

    __aicore__ inline void End()
    {
        EndIntraCoreEvents();
    }

private:
    __aicore__ inline void BindGlobalTensors()
    {
        globalTensors_.qkv.SetGlobalBuffer((__gm__ bfloat16_t *)tensors_.qkv);
        globalTensors_.qGamma.SetGlobalBuffer((__gm__ float *)tensors_.qGamma);
        globalTensors_.kGamma.SetGlobalBuffer((__gm__ float *)tensors_.kGamma);
        globalTensors_.cosSin.SetGlobalBuffer((__gm__ float *)tensors_.cosSin);
        globalTensors_.slotMapping.SetGlobalBuffer((__gm__ int32_t *)tensors_.slotMapping);
        globalTensors_.kCacheIn.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)tensors_.kCache);
        globalTensors_.vCacheIn.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)tensors_.vCache);
        globalTensors_.kScaleCacheIn.SetGlobalBuffer((__gm__ float *)tensors_.kScaleCache);
        globalTensors_.cuSeqLen.SetGlobalBuffer((__gm__ int32_t *)tensors_.cuSeqLen);
        globalTensors_.rotation.SetGlobalBuffer((__gm__ bfloat16_t *)tensors_.rotation);
        globalTensors_.vScale.SetGlobalBuffer((__gm__ float *)tensors_.vScale);
        globalTensors_.qOut.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)tensors_.qOut);
        globalTensors_.qScale.SetGlobalBuffer((__gm__ float *)tensors_.qScale);
        globalTensors_.kCacheOut.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)tensors_.kCacheOut);
        globalTensors_.vCacheOut.SetGlobalBuffer((__gm__ fp8_e4m3fn_t *)tensors_.vCacheOut);
        globalTensors_.kScaleCacheOut.SetGlobalBuffer((__gm__ float *)tensors_.kScaleCacheOut);
        globalTensors_.workspace.SetGlobalBuffer((__gm__ uint8_t *)tensors_.workspace);
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

    __aicore__ inline void ComputeAivTile(const TileParam &tile)
    {
        const AivSlice slice = MakeAivSlice(tile, tilingData_->qkHeadTile, aivLocalId_);
        const uint32_t bufferId = MakeDataBufferId(tile.bufferId);
        (void)localTensors_.aRotL1Nz[bufferId];
        (void)localTensors_.vUb[bufferId];
        (void)localTensors_.qkAfterCubeUb[bufferId];
        (void)localTensors_.qkFp8StagingUb[bufferId];
        (void)localTensors_.gammaUb;
        (void)localTensors_.cosSinUb[bufferId];
        (void)localTensors_.slotMappingUb[bufferId];
        (void)localTensors_.vScaleUb;
        (void)localTensors_.vOutUb[bufferId];
        (void)slice;
    }

    __aicore__ inline void ComputeAicTile(const TileParam &tile)
    {
        const uint32_t bufferId = MakeDataBufferId(tile.bufferId);
        (void)localTensors_.rotationL1Nz;
        (void)localTensors_.aRotL1Nz[bufferId];
        (void)tile;
        (void)cubeIndex_;
    }

    GlobalTensors tensors_;
    BoundGlobalTensors globalTensors_;
    StaticLocalTensors localTensors_;
    const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData_;
    uint32_t cubeIndex_;
    uint32_t aivLocalId_;
};

class D128FastController {
public:
    __aicore__ inline void Init(const GlobalTensors &tensors,
        const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData)
    {
        tensors_ = tensors;
        tilingData_ = tilingData;
    }

    __aicore__ inline void Process()
    {
        const TokenRange tokenRange = MakeCoreTokenRange(tilingData_->totalTokens, GetCubeIndex(), GetCubeGroupNum());

        D128FastBasicBlock basicBlock;
        basicBlock.Init(tensors_, tilingData_);
        uint32_t localTileId = 0U;
        for (uint32_t tokenOffset = tokenRange.begin; tokenOffset < tokenRange.end;) {
            const uint32_t tokenSize = MinU32(tilingData_->tokenTile, tokenRange.end - tokenOffset);
            if (tokenSize == 0U) {
                break;
            }
            const TileParam tile = MakeTileParam(localTileId, tokenOffset, tokenSize, tilingData_);
            basicBlock.ComputeTile(tile);
            tokenOffset += tokenSize;
            ++localTileId;
        }
        basicBlock.End();
    }

private:
    GlobalTensors tensors_;
    const QkvRmsNormRopeCacheWithKScaleTilingData *tilingData_;
};

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_D128_FAST_H_
