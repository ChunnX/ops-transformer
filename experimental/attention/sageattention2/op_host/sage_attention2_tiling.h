/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software and is licensed under the CANN Open Software License Agreement
 * Version 2.0. THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OF ANY KIND.
 */

#ifndef SAGE_ATTENTION2_TILING_H_
#define SAGE_ATTENTION2_TILING_H_

#include <cstdint>
#include "register/tilingdata_base.h"

namespace optiling {

constexpr uint32_t SAGE_Q_INDEX = 0U;
constexpr uint32_t SAGE_K_INDEX = 1U;
constexpr uint32_t SAGE_V_INDEX = 2U;
constexpr uint32_t SAGE_OUT_INDEX = 0U;
constexpr uint32_t SAGE_LSE_INDEX = 1U;

constexpr uint32_t SAGE_LAYOUT_ATTR = 0U;
constexpr uint32_t SAGE_CAUSAL_ATTR = 1U;
constexpr uint32_t SAGE_GRAN_ATTR = 2U;
constexpr uint32_t SAGE_SCALE_ATTR = 3U;
constexpr uint32_t SAGE_ACCUM_ATTR = 4U;
constexpr uint32_t SAGE_SMOOTH_K_ATTR = 5U;
constexpr uint32_t SAGE_SMOOTH_V_ATTR = 6U;
constexpr uint32_t SAGE_RETURN_LSE_ATTR = 7U;

enum class SageLayout : uint32_t { HND = 0U, NHD = 1U };
enum class SageQuantGran : uint32_t { PER_WARP = 0U, PER_THREAD = 1U };
enum class SageAccumMode : uint32_t { FP32 = 0U, FP32_FP32 = 1U, FP32_FP16 = 2U };

BEGIN_TILING_DATA_DEF(SageAttention2ShapeParams)
TILING_DATA_FIELD_DEF(uint32_t, batch)
TILING_DATA_FIELD_DEF(uint32_t, qHeads)
TILING_DATA_FIELD_DEF(uint32_t, kvHeads)
TILING_DATA_FIELD_DEF(uint32_t, qSeq)
TILING_DATA_FIELD_DEF(uint32_t, kvSeq)
TILING_DATA_FIELD_DEF(uint32_t, headDim)
TILING_DATA_FIELD_DEF(uint32_t, headDimPad)
TILING_DATA_FIELD_DEF(uint32_t, gqaRatio)
TILING_DATA_FIELD_DEF(uint32_t, qBlocks)
TILING_DATA_FIELD_DEF(uint32_t, kvBlocks)
TILING_DATA_FIELD_DEF(uint32_t, kvSeqPad)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(SageAttention2ShapeParamsOp, SageAttention2ShapeParams)

BEGIN_TILING_DATA_DEF(SageAttention2AttrParams)
TILING_DATA_FIELD_DEF(uint32_t, layout)
TILING_DATA_FIELD_DEF(uint32_t, causal)
TILING_DATA_FIELD_DEF(uint32_t, quantGran)
TILING_DATA_FIELD_DEF(uint32_t, accumMode)
TILING_DATA_FIELD_DEF(uint32_t, smoothK)
TILING_DATA_FIELD_DEF(uint32_t, smoothV)
TILING_DATA_FIELD_DEF(uint32_t, returnLse)
TILING_DATA_FIELD_DEF(uint32_t, inputDtype)
TILING_DATA_FIELD_DEF(float, smScale)
TILING_DATA_FIELD_DEF(float, vScaleMax)
TILING_DATA_FIELD_DEF(float, quantEpsilon)
TILING_DATA_FIELD_DEF(uint32_t, accumFlushInterval)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(SageAttention2AttrParamsOp, SageAttention2AttrParams)

BEGIN_TILING_DATA_DEF(SageAttention2CoreParams)
TILING_DATA_FIELD_DEF(uint32_t, coreNum)
TILING_DATA_FIELD_DEF(uint32_t, vecCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, totalAttentionTasks)
TILING_DATA_FIELD_DEF(uint32_t, qTile)
TILING_DATA_FIELD_DEF(uint32_t, kvTile)
TILING_DATA_FIELD_DEF(uint32_t, qScaleGroupsPerBlock)
TILING_DATA_FIELD_DEF(uint32_t, kScaleGroupsPerBlock)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(SageAttention2CoreParamsOp, SageAttention2CoreParams)

BEGIN_TILING_DATA_DEF(SageAttention2WorkspaceParams)
TILING_DATA_FIELD_DEF(uint64_t, q8Offset)
TILING_DATA_FIELD_DEF(uint64_t, k8Offset)
TILING_DATA_FIELD_DEF(uint64_t, v8Offset)
TILING_DATA_FIELD_DEF(uint64_t, qScaleOffset)
TILING_DATA_FIELD_DEF(uint64_t, kScaleOffset)
TILING_DATA_FIELD_DEF(uint64_t, vScaleOffset)
TILING_DATA_FIELD_DEF(uint64_t, kMeanOffset)
TILING_DATA_FIELD_DEF(uint64_t, vMeanOffset)
TILING_DATA_FIELD_DEF(uint64_t, workspaceBytes)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(SageAttention2WorkspaceParamsOp, SageAttention2WorkspaceParams)

BEGIN_TILING_DATA_DEF(SageAttention2TilingData)
TILING_DATA_FIELD_DEF_STRUCT(SageAttention2ShapeParams, shape)
TILING_DATA_FIELD_DEF_STRUCT(SageAttention2AttrParams, attrs)
TILING_DATA_FIELD_DEF_STRUCT(SageAttention2CoreParams, cores)
TILING_DATA_FIELD_DEF_STRUCT(SageAttention2WorkspaceParams, workspace)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(SageAttention2, SageAttention2TilingData)

} // namespace optiling
#endif // SAGE_ATTENTION2_TILING_H_

