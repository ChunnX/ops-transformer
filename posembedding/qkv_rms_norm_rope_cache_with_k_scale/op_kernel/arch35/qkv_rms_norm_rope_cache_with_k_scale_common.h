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

#ifndef QKV_RRC_LAYOUT_NTD
#define QKV_RRC_LAYOUT_NTD 0
#endif
#ifndef QKV_RRC_LAYOUT_TND
#define QKV_RRC_LAYOUT_TND 1
#endif

namespace QkvRmsNormRopeCacheWithKScale {
using AscendC::CrossCoreSetFlag;
using AscendC::CrossCoreWaitFlag;
using AscendC::CO2Layout;
using AscendC::DataCopy;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyParams;
using AscendC::DataCopyPad;
using AscendC::DataCopyPadExtParams;
using AscendC::Fixpipe;
using AscendC::FixpipeConfig;
using AscendC::FixpipeParamsC310;
using AscendC::GetBlockIdx;
using AscendC::GetSubBlockIdx;
using AscendC::GlobalTensor;
using AscendC::HardEvent;
using AscendC::LoadData;
using AscendC::LoadData2DParamsV2;
using AscendC::LocalTensor;
using AscendC::Mmad;
using AscendC::MmadParams;
using AscendC::Nd2NzParams;
using AscendC::RoundMode;
using AscendC::SetFlag;
using AscendC::TPosition;
using AscendC::WaitFlag;
using QkvRmsNormRopeCacheWithKScaleKernelTiling::QkvRmsNormRopeCacheWithKScaleTilingData;

constexpr uint32_t QKV_RRC_MIX_AIV_PER_AIC = 2U;
constexpr uint32_t QKV_RRC_DOUBLE_BUFFER_NUM = 2U;
constexpr uint32_t QKV_RRC_HEAD_DIM_D128 = 128U;
constexpr uint32_t QKV_RRC_MAX_TOKEN_TILE_PER_AIV = 8U;
constexpr uint32_t QKV_RRC_NZ_C0 = 16U;
constexpr uint32_t QKV_RRC_BLOCK_BYTES = 32U;
constexpr uint32_t QKV_RRC_KIB = 1024U;

constexpr uint8_t QKV_RRC_SYNC_MODE4 = 4U;
constexpr uint64_t QKV_RRC_FLAG_ID_MAX = 16U;
constexpr uint64_t SYNC_A_READY = 0U;
constexpr uint64_t SYNC_C_READY = 1U;
constexpr uint64_t SYNC_C_CONSUMED = 2U;
constexpr uint64_t SYNC_A_CONSUMED = 3U;

static_assert(SYNC_A_READY != SYNC_C_READY && SYNC_A_READY != SYNC_C_CONSUMED && SYNC_A_READY != SYNC_A_CONSUMED &&
                  SYNC_C_READY != SYNC_C_CONSUMED && SYNC_C_READY != SYNC_A_CONSUMED &&
                  SYNC_C_CONSUMED != SYNC_A_CONSUMED,
              "QkvRmsNormRopeCacheWithKScale CrossCore flags must keep unique semantic ids.");
static_assert(SYNC_A_READY < QKV_RRC_FLAG_ID_MAX && SYNC_C_READY < QKV_RRC_FLAG_ID_MAX &&
                  SYNC_C_CONSUMED < QKV_RRC_FLAG_ID_MAX && SYNC_A_CONSUMED < QKV_RRC_FLAG_ID_MAX,
              "QkvRmsNormRopeCacheWithKScale CrossCore base flag ids must not overlap AIV1 offset.");

constexpr uint32_t QKV_RRC_INPUT_ONE_BUFFER_BYTES = 40U * QKV_RRC_KIB;
constexpr uint32_t QKV_RRC_OUTPUT_ONE_BUFFER_BYTES = 64U * QKV_RRC_KIB;
constexpr uint32_t QKV_RRC_INPUT_DB_POOL_BYTES = QKV_RRC_DOUBLE_BUFFER_NUM * QKV_RRC_INPUT_ONE_BUFFER_BYTES;
constexpr uint32_t QKV_RRC_OUTPUT_DB_POOL_BYTES = QKV_RRC_DOUBLE_BUFFER_NUM * QKV_RRC_OUTPUT_ONE_BUFFER_BYTES;
constexpr uint32_t QKV_RRC_INPUT_ONE_BUFFER_ELEMENTS = QKV_RRC_INPUT_ONE_BUFFER_BYTES / sizeof(bfloat16_t);
constexpr uint32_t QKV_RRC_INPUT_DB_POOL_ELEMENTS = QKV_RRC_INPUT_DB_POOL_BYTES / sizeof(bfloat16_t);
constexpr uint32_t QKV_RRC_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS = QKV_RRC_OUTPUT_ONE_BUFFER_BYTES / sizeof(float);
constexpr uint32_t QKV_RRC_OUTPUT_DB_POOL_FLOAT_ELEMENTS = QKV_RRC_OUTPUT_DB_POOL_BYTES / sizeof(float);

constexpr uint32_t QKV_RRC_ROTATION_ONE_L1_BYTES = QKV_RRC_HEAD_DIM_D128 * QKV_RRC_HEAD_DIM_D128 * sizeof(bfloat16_t);
constexpr uint32_t QKV_RRC_A_ROT_L1_ONE_BUFFER_BYTES = 64U * QKV_RRC_KIB;
constexpr uint32_t QKV_RRC_A_ROT_L1_BUFFER_NUM = QKV_RRC_DOUBLE_BUFFER_NUM * QKV_RRC_MIX_AIV_PER_AIC;
constexpr uint32_t QKV_RRC_A_ROT_L1_POOL_BYTES = QKV_RRC_A_ROT_L1_BUFFER_NUM * QKV_RRC_A_ROT_L1_ONE_BUFFER_BYTES;
constexpr uint32_t QKV_RRC_Q_ROTATION_L1_OFFSET = 0U;
constexpr uint32_t QKV_RRC_K_ROTATION_L1_OFFSET = QKV_RRC_Q_ROTATION_L1_OFFSET + QKV_RRC_ROTATION_ONE_L1_BYTES;
constexpr uint32_t QKV_RRC_A_ROT_L1_POOL_OFFSET = QKV_RRC_K_ROTATION_L1_OFFSET + QKV_RRC_ROTATION_ONE_L1_BYTES;
constexpr uint32_t QKV_RRC_A_ROT_L1_LOGICAL_BUFFER_BYTES = QKV_RRC_A_ROT_L1_POOL_BYTES / QKV_RRC_DOUBLE_BUFFER_NUM;
constexpr uint32_t QKV_RRC_ROTATION_ONE_L1_ELEMENTS = QKV_RRC_ROTATION_ONE_L1_BYTES / sizeof(bfloat16_t);
constexpr uint32_t QKV_RRC_A_ROT_L1_LOGICAL_BUFFER_ELEMENTS =
    QKV_RRC_A_ROT_L1_LOGICAL_BUFFER_BYTES / sizeof(bfloat16_t);
constexpr uint32_t QKV_RRC_A_ROT_L1_POOL_ELEMENTS = QKV_RRC_A_ROT_L1_POOL_BYTES / sizeof(bfloat16_t);

constexpr uint32_t QKV_RRC_RESERVE_UB_BYTES = 40U * QKV_RRC_KIB;
constexpr uint32_t QKV_RRC_GAMMA_UB_BYTES = 1U * QKV_RRC_KIB;
constexpr uint32_t QKV_RRC_COS_SIN_ONE_BUFFER_BYTES = 4U * QKV_RRC_KIB;
constexpr uint32_t QKV_RRC_SLOT_MAPPING_ONE_BUFFER_BYTES = 512U;
constexpr uint32_t QKV_RRC_V_SCALE_UB_BYTES = 512U;
constexpr uint32_t QKV_RRC_QK_SCALE_MTE3_ALIGN_ELEMENTS = QKV_RRC_BLOCK_BYTES / sizeof(float);
constexpr uint32_t QKV_RRC_QK_SCALE_MAX_ROWS =
    QKV_RRC_OUTPUT_ONE_BUFFER_BYTES / (QKV_RRC_HEAD_DIM_D128 * sizeof(float));
constexpr uint32_t QKV_RRC_QK_SCALE_UB_BYTES =
    QKV_RRC_QK_SCALE_MAX_ROWS * QKV_RRC_QK_SCALE_MTE3_ALIGN_ELEMENTS * sizeof(float);
constexpr uint32_t QKV_RRC_QK_NZ_SCATTER_INDEX_UB_BYTES = 512U;
constexpr uint32_t QKV_RRC_V_OUT_ONE_BUFFER_BYTES = 10U * QKV_RRC_KIB;

constexpr uint32_t QKV_RRC_GAMMA_UB_OFFSET = 0U;
constexpr uint32_t QKV_RRC_COS_SIN_DB_POOL_OFFSET = QKV_RRC_GAMMA_UB_OFFSET + QKV_RRC_GAMMA_UB_BYTES;
constexpr uint32_t QKV_RRC_SLOT_MAPPING_DB_POOL_OFFSET =
    QKV_RRC_COS_SIN_DB_POOL_OFFSET + QKV_RRC_DOUBLE_BUFFER_NUM * QKV_RRC_COS_SIN_ONE_BUFFER_BYTES;
constexpr uint32_t QKV_RRC_V_SCALE_UB_OFFSET =
    QKV_RRC_SLOT_MAPPING_DB_POOL_OFFSET + QKV_RRC_DOUBLE_BUFFER_NUM * QKV_RRC_SLOT_MAPPING_ONE_BUFFER_BYTES;
constexpr uint32_t QKV_RRC_QK_NZ_SCATTER_INDEX_UB_OFFSET = QKV_RRC_V_SCALE_UB_OFFSET + QKV_RRC_V_SCALE_UB_BYTES;
constexpr uint32_t QKV_RRC_V_OUT_DB_POOL_OFFSET =
    QKV_RRC_QK_NZ_SCATTER_INDEX_UB_OFFSET + QKV_RRC_QK_NZ_SCATTER_INDEX_UB_BYTES;
constexpr uint32_t QKV_RRC_RESERVE_USED_BYTES =
    QKV_RRC_V_OUT_DB_POOL_OFFSET + QKV_RRC_DOUBLE_BUFFER_NUM * QKV_RRC_V_OUT_ONE_BUFFER_BYTES;
constexpr uint32_t QKV_RRC_GAMMA_UB_ELEMENTS = QKV_RRC_GAMMA_UB_BYTES / sizeof(float);
constexpr uint32_t QKV_RRC_COS_SIN_ONE_BUFFER_ELEMENTS = QKV_RRC_COS_SIN_ONE_BUFFER_BYTES / sizeof(float);
constexpr uint32_t QKV_RRC_COS_SIN_DB_POOL_ELEMENTS = QKV_RRC_DOUBLE_BUFFER_NUM * QKV_RRC_COS_SIN_ONE_BUFFER_ELEMENTS;
constexpr uint32_t QKV_RRC_SLOT_MAPPING_ONE_BUFFER_ELEMENTS = QKV_RRC_SLOT_MAPPING_ONE_BUFFER_BYTES / sizeof(int32_t);
constexpr uint32_t QKV_RRC_SLOT_MAPPING_DB_POOL_ELEMENTS =
    QKV_RRC_DOUBLE_BUFFER_NUM * QKV_RRC_SLOT_MAPPING_ONE_BUFFER_ELEMENTS;
constexpr uint32_t QKV_RRC_V_SCALE_UB_ELEMENTS = QKV_RRC_V_SCALE_UB_BYTES / sizeof(float);
constexpr uint32_t QKV_RRC_QK_SCALE_UB_ELEMENTS = QKV_RRC_QK_SCALE_UB_BYTES / sizeof(float);
constexpr uint32_t QKV_RRC_QK_NZ_SCATTER_INDEX_UB_ELEMENTS = QKV_RRC_QK_NZ_SCATTER_INDEX_UB_BYTES / sizeof(uint16_t);
constexpr uint32_t QKV_RRC_V_OUT_ONE_BUFFER_ELEMENTS = QKV_RRC_V_OUT_ONE_BUFFER_BYTES / sizeof(fp8_e4m3fn_t);
constexpr uint32_t QKV_RRC_V_OUT_DB_POOL_ELEMENTS = QKV_RRC_DOUBLE_BUFFER_NUM * QKV_RRC_V_OUT_ONE_BUFFER_ELEMENTS;

static_assert(QKV_RRC_RESERVE_USED_BYTES <= QKV_RRC_RESERVE_UB_BYTES,
              "QkvRmsNormRopeCacheWithKScale reserve UB layout exceeds 40 KiB.");

constexpr uint32_t QKV_RRC_INPUT_DB_POOL_OFFSET = 0U;
constexpr uint32_t QKV_RRC_OUTPUT_DB_POOL_OFFSET = QKV_RRC_INPUT_DB_POOL_OFFSET + QKV_RRC_INPUT_DB_POOL_BYTES;
constexpr uint32_t QKV_RRC_RESERVE_UB_OFFSET = QKV_RRC_OUTPUT_DB_POOL_OFFSET + QKV_RRC_OUTPUT_DB_POOL_BYTES;
constexpr uint32_t QKV_RRC_UB_USED_BYTES = QKV_RRC_RESERVE_UB_OFFSET + QKV_RRC_RESERVE_UB_BYTES;

static_assert(QKV_RRC_UB_USED_BYTES <= 248U * QKV_RRC_KIB,
              "QkvRmsNormRopeCacheWithKScale static UB layout exceeds 248 KiB.");

struct TileParam {
    uint64_t tokenOffset;
    uint64_t tokenSize;
    uint64_t cubeTokenSize;
    uint64_t cubeHalfTokenSize;
    uint64_t aivTokenOffset;
    uint64_t aivTokenSize;
    uint64_t aivBlockTokenOffset;
    uint64_t vHeadOffset;
    uint64_t vHeadSize;
    uint64_t cacheBaseOffset[QKV_RRC_MAX_TOKEN_TILE_PER_AIV];
    uint64_t scaleCacheBaseOffset[QKV_RRC_MAX_TOKEN_TILE_PER_AIV];
};

struct TokenRange {
    uint64_t begin;
    uint64_t end;
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
    GM_ADDR actualSeqLens;
    GM_ADDR qRotation;
    GM_ADDR kRotation;
    GM_ADDR vScale;
    GM_ADDR qOut;
    GM_ADDR qScale;
    GM_ADDR kCacheOut;
    GM_ADDR vCacheOut;
    GM_ADDR kScaleCacheOut;
    GM_ADDR workspace;
};

struct DataCopy2DParams {
    uint64_t rowCount;
    uint64_t colCount;
    uint64_t srcStride;
};

struct DataCopyUbToGm2DParams {
    uint64_t rowCount;
    uint64_t colCount;
    uint64_t srcStride;
    uint64_t dstStride;
};

__aicore__ inline uint64_t CeilDiv(uint64_t value, uint64_t factor)
{
    if (factor == 0U) {
        return 0U;
    }
    return (value + factor - 1U) / factor;
}

__aicore__ inline uint64_t AlignUp(uint64_t value, uint64_t align)
{
    if (align == 0U) {
        return value;
    }
    return ((value + align - 1U) / align) * align;
}

__aicore__ inline uint64_t MinU64(uint64_t lhs, uint64_t rhs)
{
    return lhs < rhs ? lhs : rhs;
}

__aicore__ inline uint64_t MaxU64(uint64_t lhs, uint64_t rhs)
{
    return lhs > rhs ? lhs : rhs;
}

__aicore__ inline uint64_t NzRowAligned(uint64_t rowCount)
{
    return AlignUp(rowCount, QKV_RRC_NZ_C0);
}

__aicore__ inline uint64_t NzMatrixElements(uint64_t rowCount)
{
    return (QKV_RRC_HEAD_DIM_D128 / QKV_RRC_NZ_C0) * NzRowAligned(rowCount) * QKV_RRC_NZ_C0;
}

__aicore__ inline uint64_t NzMatrixElements(uint64_t rowCount, uint64_t colCount)
{
    return CeilDiv(colCount, QKV_RRC_NZ_C0) * NzRowAligned(rowCount) * QKV_RRC_NZ_C0;
}

template <typename T>
__aicore__ inline void DataCopyGmToUb2D(const LocalTensor<T> &dst, const GlobalTensor<T> &src,
                                        const DataCopy2DParams &params2d)
{
    if (params2d.rowCount == 0U || params2d.colCount == 0U) {
        return;
    }

    DataCopyExtParams params;
    params.blockCount = static_cast<uint16_t>(params2d.rowCount);
    params.blockLen = static_cast<uint32_t>(params2d.colCount * sizeof(T));
    params.srcStride = static_cast<decltype(params.srcStride)>((params2d.srcStride - params2d.colCount) * sizeof(T));
    params.dstStride = 0U;
    params.rsv = 0U;
    DataCopyPadExtParams<T> padParams{false, 0U, 0U, 0U};
    const uint32_t blockElements = QKV_RRC_BLOCK_BYTES / sizeof(T);
    const uint64_t alignedColCount = AlignUp(params2d.colCount, blockElements);
    if (alignedColCount != params2d.colCount) {
        padParams.isPad = true;
        padParams.rightPadding = static_cast<uint8_t>(alignedColCount - params2d.colCount);
    }
    DataCopyPad(dst, src, params, padParams);
}

template <typename T>
__aicore__ inline void DataCopyUbToGm2D(const GlobalTensor<T> &dst, const LocalTensor<T> &src,
                                        const DataCopyUbToGm2DParams &params2d)
{
    if (params2d.rowCount == 0U || params2d.colCount == 0U) {
        return;
    }

    DataCopyExtParams params;
    params.blockCount = static_cast<uint16_t>(params2d.rowCount);
    params.blockLen = static_cast<uint32_t>(params2d.colCount * sizeof(T));
    params.srcStride =
        static_cast<decltype(params.srcStride)>((params2d.srcStride - params2d.colCount) * sizeof(T) /
                                                QKV_RRC_BLOCK_BYTES);
    params.dstStride = static_cast<decltype(params.dstStride)>((params2d.dstStride - params2d.colCount) * sizeof(T));
    params.rsv = 0U;
    DataCopyPad(dst, src, params);
}

template <const pipe_t pipe>
__aicore__ inline void SetAivToAic(uint64_t syncFlag)
{
    CrossCoreSetFlag<QKV_RRC_SYNC_MODE4, pipe>(syncFlag);
}

template <const pipe_t pipe>
__aicore__ inline void WaitAivToAic(uint64_t syncFlag)
{
    CrossCoreWaitFlag<QKV_RRC_SYNC_MODE4, pipe>(syncFlag + QKV_RRC_FLAG_ID_MAX);
    CrossCoreWaitFlag<QKV_RRC_SYNC_MODE4, pipe>(syncFlag);
}

template <const pipe_t pipe>
__aicore__ inline void SetAicToAiv(uint64_t syncFlag)
{
    CrossCoreSetFlag<QKV_RRC_SYNC_MODE4, pipe>(syncFlag + QKV_RRC_FLAG_ID_MAX);
    CrossCoreSetFlag<QKV_RRC_SYNC_MODE4, pipe>(syncFlag);
}

template <const pipe_t pipe>
__aicore__ inline void WaitAicToAiv(uint64_t syncFlag)
{
    CrossCoreWaitFlag<QKV_RRC_SYNC_MODE4, pipe>(syncFlag);
}

__aicore__ inline uint32_t GetAicCubeIndex()
{
    return GetBlockIdx();
}

__aicore__ inline uint32_t GetAivCubeIndex()
{
    return GetBlockIdx() / QKV_RRC_MIX_AIV_PER_AIC;
}

__aicore__ inline uint32_t GetCubeIndex()
{
    if ASCEND_IS_AIC {
        return GetAicCubeIndex();
    }
    return GetAivCubeIndex();
}

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_COMMON_H_
