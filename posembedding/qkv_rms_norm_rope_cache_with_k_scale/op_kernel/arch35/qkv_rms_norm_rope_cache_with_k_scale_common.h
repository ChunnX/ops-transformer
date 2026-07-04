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

#ifndef QKV_K_SCALE_LAYOUT_NTD
#define QKV_K_SCALE_LAYOUT_NTD 0
#endif
#ifndef QKV_K_SCALE_LAYOUT_TND
#define QKV_K_SCALE_LAYOUT_TND 1
#endif

namespace QkvRmsNormRopeCacheWithKScale {
using AscendC::CO2Layout;
using AscendC::DataCopy;
using AscendC::DataCopyExtParams;
using AscendC::DataCopyPad;
using AscendC::DataCopyPadExtParams;
using AscendC::DataCopyParams;
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

constexpr uint32_t QKV_K_SCALE_MIX_AIV_PER_AIC = 2U;
constexpr uint32_t QKV_K_SCALE_DOUBLE_BUFFER_NUM = 2U;
constexpr uint32_t QKV_K_SCALE_HEAD_DIM_D128 = 128U;
constexpr uint32_t QKV_K_SCALE_MAX_TOKEN_TILE_PER_AIV = 8U;
constexpr uint32_t QKV_K_SCALE_NZ_C0 = 16U;
constexpr uint32_t QKV_K_SCALE_QK_PREPROCESS_UB_NZ_STRIDE_ALIGN = QKV_K_SCALE_NZ_C0;
constexpr uint32_t QKV_K_SCALE_BLOCK_BYTES = 32U;
constexpr uint32_t QKV_K_SCALE_KIB = 1024U;

constexpr uint32_t QKV_K_SCALE_INPUT_ONE_BUFFER_BYTES = 40U * QKV_K_SCALE_KIB;
constexpr uint32_t QKV_K_SCALE_OUTPUT_ONE_BUFFER_BYTES = 64U * QKV_K_SCALE_KIB;
constexpr uint32_t QKV_K_SCALE_INPUT_DB_POOL_BYTES = QKV_K_SCALE_DOUBLE_BUFFER_NUM * QKV_K_SCALE_INPUT_ONE_BUFFER_BYTES;
constexpr uint32_t QKV_K_SCALE_OUTPUT_DB_POOL_BYTES =
    QKV_K_SCALE_DOUBLE_BUFFER_NUM * QKV_K_SCALE_OUTPUT_ONE_BUFFER_BYTES;
constexpr uint32_t QKV_K_SCALE_INPUT_ONE_BUFFER_ELEMENTS = QKV_K_SCALE_INPUT_ONE_BUFFER_BYTES / sizeof(bfloat16_t);
constexpr uint32_t QKV_K_SCALE_INPUT_DB_POOL_ELEMENTS = QKV_K_SCALE_INPUT_DB_POOL_BYTES / sizeof(bfloat16_t);
constexpr uint32_t QKV_K_SCALE_OUTPUT_ONE_BUFFER_FLOAT_ELEMENTS = QKV_K_SCALE_OUTPUT_ONE_BUFFER_BYTES / sizeof(float);
constexpr uint32_t QKV_K_SCALE_OUTPUT_DB_POOL_FLOAT_ELEMENTS = QKV_K_SCALE_OUTPUT_DB_POOL_BYTES / sizeof(float);

constexpr uint32_t QKV_K_SCALE_ROTATION_ONE_L1_BYTES =
    QKV_K_SCALE_HEAD_DIM_D128 * QKV_K_SCALE_HEAD_DIM_D128 * sizeof(bfloat16_t);
constexpr uint32_t QKV_K_SCALE_ROTATION_L1_OFFSET = 0U;
constexpr uint32_t QKV_K_SCALE_ROTATION_RESERVED_L1_BYTES = 2U * QKV_K_SCALE_ROTATION_ONE_L1_BYTES;
constexpr uint32_t QKV_K_SCALE_A_ROT_L1_POOL_OFFSET = QKV_K_SCALE_ROTATION_RESERVED_L1_BYTES;
constexpr uint32_t QKV_K_SCALE_ROTATION_ONE_L1_ELEMENTS = QKV_K_SCALE_ROTATION_ONE_L1_BYTES / sizeof(bfloat16_t);
constexpr uint32_t QKV_K_SCALE_A_ROT_L1_LOGICAL_BUFFER_ELEMENTS =
    QKV_K_SCALE_MIX_AIV_PER_AIC * 64U * QKV_K_SCALE_KIB / sizeof(bfloat16_t);
constexpr uint32_t QKV_K_SCALE_A_ROT_L1_POOL_ELEMENTS =
    QKV_K_SCALE_DOUBLE_BUFFER_NUM * QKV_K_SCALE_A_ROT_L1_LOGICAL_BUFFER_ELEMENTS;

constexpr uint32_t QKV_K_SCALE_RESERVE_UB_BYTES = 40U * QKV_K_SCALE_KIB;
constexpr uint32_t QKV_K_SCALE_GAMMA_UB_BYTES = 1U * QKV_K_SCALE_KIB;
constexpr uint32_t QKV_K_SCALE_COS_SIN_ONE_BUFFER_BYTES = 4U * QKV_K_SCALE_KIB;
constexpr uint32_t QKV_K_SCALE_SLOT_MAPPING_ONE_BUFFER_BYTES = 512U;
constexpr uint32_t QKV_K_SCALE_V_SCALE_UB_BYTES = 512U;
constexpr uint32_t QKV_K_SCALE_QK_SCALE_MTE3_ALIGN_ELEMENTS = QKV_K_SCALE_BLOCK_BYTES / sizeof(float);
constexpr uint32_t QKV_K_SCALE_QK_NZ_SCATTER_INDEX_TABLE_ELEMENTS = QKV_K_SCALE_HEAD_DIM_D128;
constexpr uint32_t QKV_K_SCALE_QK_NZ_SCATTER_INDEX_UB_BYTES =
    2U * QKV_K_SCALE_QK_NZ_SCATTER_INDEX_TABLE_ELEMENTS * sizeof(uint16_t);
constexpr uint32_t QKV_K_SCALE_V_OUT_ONE_BUFFER_BYTES = 10U * QKV_K_SCALE_KIB;

constexpr uint32_t QKV_K_SCALE_GAMMA_UB_OFFSET = 0U;
constexpr uint32_t QKV_K_SCALE_COS_SIN_DB_POOL_OFFSET = QKV_K_SCALE_GAMMA_UB_OFFSET + QKV_K_SCALE_GAMMA_UB_BYTES;
constexpr uint32_t QKV_K_SCALE_SLOT_MAPPING_DB_POOL_OFFSET =
    QKV_K_SCALE_COS_SIN_DB_POOL_OFFSET + QKV_K_SCALE_DOUBLE_BUFFER_NUM * QKV_K_SCALE_COS_SIN_ONE_BUFFER_BYTES;
constexpr uint32_t QKV_K_SCALE_V_SCALE_UB_OFFSET =
    QKV_K_SCALE_SLOT_MAPPING_DB_POOL_OFFSET + QKV_K_SCALE_DOUBLE_BUFFER_NUM * QKV_K_SCALE_SLOT_MAPPING_ONE_BUFFER_BYTES;
constexpr uint32_t QKV_K_SCALE_QK_NZ_SCATTER_INDEX_UB_OFFSET =
    QKV_K_SCALE_V_SCALE_UB_OFFSET + QKV_K_SCALE_V_SCALE_UB_BYTES;
constexpr uint32_t QKV_K_SCALE_V_OUT_DB_POOL_OFFSET =
    QKV_K_SCALE_QK_NZ_SCATTER_INDEX_UB_OFFSET + QKV_K_SCALE_QK_NZ_SCATTER_INDEX_UB_BYTES;
constexpr uint32_t QKV_K_SCALE_GAMMA_UB_ELEMENTS = QKV_K_SCALE_GAMMA_UB_BYTES / sizeof(float);
constexpr uint32_t QKV_K_SCALE_COS_SIN_ONE_BUFFER_ELEMENTS = QKV_K_SCALE_COS_SIN_ONE_BUFFER_BYTES / sizeof(float);
constexpr uint32_t QKV_K_SCALE_COS_SIN_DB_POOL_ELEMENTS =
    QKV_K_SCALE_DOUBLE_BUFFER_NUM * QKV_K_SCALE_COS_SIN_ONE_BUFFER_ELEMENTS;
constexpr uint32_t QKV_K_SCALE_SLOT_MAPPING_ONE_BUFFER_ELEMENTS =
    QKV_K_SCALE_SLOT_MAPPING_ONE_BUFFER_BYTES / sizeof(int32_t);
constexpr uint32_t QKV_K_SCALE_SLOT_MAPPING_DB_POOL_ELEMENTS =
    QKV_K_SCALE_DOUBLE_BUFFER_NUM * QKV_K_SCALE_SLOT_MAPPING_ONE_BUFFER_ELEMENTS;
constexpr uint32_t QKV_K_SCALE_V_SCALE_UB_ELEMENTS = QKV_K_SCALE_V_SCALE_UB_BYTES / sizeof(float);
constexpr uint32_t QKV_K_SCALE_QK_NZ_SCATTER_INDEX_UB_ELEMENTS =
    QKV_K_SCALE_QK_NZ_SCATTER_INDEX_UB_BYTES / sizeof(uint16_t);
constexpr uint32_t QKV_K_SCALE_V_OUT_ONE_BUFFER_ELEMENTS = QKV_K_SCALE_V_OUT_ONE_BUFFER_BYTES / sizeof(fp8_e4m3fn_t);
constexpr uint32_t QKV_K_SCALE_V_OUT_DB_POOL_ELEMENTS =
    QKV_K_SCALE_DOUBLE_BUFFER_NUM * QKV_K_SCALE_V_OUT_ONE_BUFFER_ELEMENTS;

constexpr uint32_t QKV_K_SCALE_INPUT_DB_POOL_OFFSET = 0U;
constexpr uint32_t QKV_K_SCALE_OUTPUT_DB_POOL_OFFSET =
    QKV_K_SCALE_INPUT_DB_POOL_OFFSET + QKV_K_SCALE_INPUT_DB_POOL_BYTES;
constexpr uint32_t QKV_K_SCALE_RESERVE_UB_OFFSET = QKV_K_SCALE_OUTPUT_DB_POOL_OFFSET + QKV_K_SCALE_OUTPUT_DB_POOL_BYTES;

struct TileParam {
    uint64_t tokenOffset;
    uint64_t tokenSize;
    uint64_t cubeTokenSize;
    uint64_t cubeHalfTokenSize;
    uint64_t aivTokenOffset;
    uint64_t aivTokenSize;
    uint64_t aivBlockTokenOffset;
    uint64_t vHeadSize;
    uint64_t cacheBaseOffset[QKV_K_SCALE_MAX_TOKEN_TILE_PER_AIV];
    uint64_t scaleCacheBaseOffset[QKV_K_SCALE_MAX_TOKEN_TILE_PER_AIV];
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
    GM_ADDR queryStartLoc;
    GM_ADDR seqLens;
    GM_ADDR rotation;
    GM_ADDR vScale;
    GM_ADDR qOut;
    GM_ADDR qScale;
    GM_ADDR kCacheOut;
    GM_ADDR vCacheOut;
    GM_ADDR kScaleCacheOut;
    GM_ADDR workspace;
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

__aicore__ inline uint64_t NzMatrixElements(uint64_t rowCount)
{
    return (QKV_K_SCALE_HEAD_DIM_D128 / QKV_K_SCALE_NZ_C0) * AlignUp(rowCount, QKV_K_SCALE_NZ_C0) * QKV_K_SCALE_NZ_C0;
}

template <typename T>
__aicore__ inline void DataCopyGmToUb2D(const LocalTensor<T> &dst, const GlobalTensor<T> &src, uint64_t rowCount,
                                        uint64_t colCount, uint64_t srcStride)
{
    if (rowCount == 0U || colCount == 0U) {
        return;
    }

    DataCopyExtParams params;
    params.blockCount = static_cast<uint16_t>(rowCount);
    params.blockLen = static_cast<uint32_t>(colCount * sizeof(T));
    params.srcStride = static_cast<decltype(params.srcStride)>((srcStride - colCount) * sizeof(T));
    params.dstStride = 0U;
    params.rsv = 0U;
    DataCopyPadExtParams<T> padParams{false, 0U, 0U, 0U};
    const uint32_t blockElements = QKV_K_SCALE_BLOCK_BYTES / sizeof(T);
    const uint64_t alignedColCount = AlignUp(colCount, blockElements);
    if (alignedColCount != colCount) {
        padParams.isPad = true;
        padParams.rightPadding = static_cast<uint8_t>(alignedColCount - colCount);
    }
    DataCopyPad(dst, src, params, padParams);
}

template <typename T>
__aicore__ inline void DataCopyUbToGm2D(const GlobalTensor<T> &dst, const LocalTensor<T> &src, uint64_t rowCount,
                                        uint64_t colCount, uint64_t srcStride, uint64_t dstStride)
{
    if (rowCount == 0U || colCount == 0U) {
        return;
    }

    DataCopyExtParams params;
    params.blockCount = static_cast<uint16_t>(rowCount);
    params.blockLen = static_cast<uint32_t>(colCount * sizeof(T));
    params.srcStride =
        static_cast<decltype(params.srcStride)>((srcStride - colCount) * sizeof(T) / QKV_K_SCALE_BLOCK_BYTES);
    params.dstStride = static_cast<decltype(params.dstStride)>((dstStride - colCount) * sizeof(T));
    params.rsv = 0U;
    DataCopyPad(dst, src, params);
}

} // namespace QkvRmsNormRopeCacheWithKScale

#endif // QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_COMMON_H_
