/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GENERIC_LIGHTNING_INDEXER_TILING_H
#define GENERIC_LIGHTNING_INDEXER_TILING_H

#include "err/ops_err.h"
#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_util.h"

namespace optiling {
enum class GliDataLayout : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BBND = 2
};

constexpr uint32_t GLI_QUERY_INDEX = 0;
constexpr uint32_t GLI_KEY_INDEX = 1;
constexpr uint32_t GLI_WEIGHTS_INDEX = 2;
constexpr uint32_t GLI_ACTUAL_SEQ_Q_INDEX = 3;
constexpr uint32_t GLI_ACTUAL_SEQ_K_INDEX = 4;
constexpr uint32_t GLI_SEQUSED_Q_INDEX = 5;
constexpr uint32_t GLI_SEQUSED_K_INDEX = 6;
constexpr uint32_t GLI_CMP_RESIDUAL_K_INDEX = 7;
constexpr uint32_t GLI_BLOCK_TABLE_INDEX = 8;
constexpr uint32_t GLI_Q_DESCALE_INDEX = 9;
constexpr uint32_t GLI_K_DESCALE_INDEX = 10;
constexpr uint32_t GLI_OUTPUT_IDX_OFFSET_INDEX = 11;
constexpr uint32_t GLI_METADATA_INDEX = 12;
constexpr uint32_t GLI_START_LOC_INDEX = 13;

constexpr uint32_t GLI_ATTR_TOPK = 0;
constexpr uint32_t GLI_ATTR_CMP_RATIO = 1;
constexpr uint32_t GLI_ATTR_LAYOUT_Q = 2;
constexpr uint32_t GLI_ATTR_LAYOUT_K = 3;
constexpr uint32_t GLI_ATTR_SPARSE_MODE = 4;
constexpr uint32_t GLI_ATTR_ACT_TYPE = 5;
constexpr uint32_t GLI_ATTR_HAS_WEIGHTED_SUM = 6;
constexpr uint32_t GLI_ATTR_EXPAND_INDICES = 7;
constexpr uint32_t GLI_ATTR_APPEND_TAIL = 8;
constexpr uint32_t GLI_ATTR_QUANT_MODE = 9;
constexpr uint32_t GLI_ATTR_RETURN_VALUE = 10;
constexpr uint32_t GLI_ATTR_SELECT_MODE = 11;
constexpr uint32_t GLI_ATTR_KEY_REDUCE_MODE = 12;
constexpr uint32_t GLI_ATTR_SCORE_SCALE = 13;
constexpr uint32_t GLI_ATTR_KEY_STRIDE0 = 14;
constexpr uint32_t GLI_ATTR_K_DESCALE_STRIDE0 = 15;
constexpr uint32_t GLI_ATTR_MAX_SEQLEN_Q = 16;
constexpr uint32_t GLI_ATTR_INIT_BLOCKS = 17;
constexpr uint32_t GLI_ATTR_LOCAL_BLOCKS = 18;

constexpr uint32_t GLI_HEAD_DIM = 128;
constexpr uint32_t GLI_N2_FIXED = 1;
constexpr uint32_t GLI_N1_LIMIT = 64;
constexpr uint32_t GLI_CMP_RATIO_MAX = 128;
constexpr uint32_t GLI_S1_BASE_SIZE = 4;
constexpr uint32_t GLI_S1_BASE_SIZE_SMALL = 2;
constexpr uint32_t GLI_S2_BASE_SIZE = 128;
constexpr uint32_t GLI_M_BASE_SIZE = 256;
constexpr uint32_t GLI_TRUNK_LEN_8K = 8192;
constexpr uint32_t GLI_GM_ALIGN_BYTES = 512;
constexpr uint32_t GLI_SCORE_T_SIZE = 4;
constexpr uint32_t GLI_BLOCK_SIZE_FACTOR = 16;
constexpr uint32_t GLI_BLOCK_SIZE_LIMIT = 2048;

constexpr uint32_t GLI_SEQ_Q_I64 = 1U;
constexpr uint32_t GLI_SEQ_K_I64 = 2U;
constexpr uint32_t GLI_SEQUSED_Q_I64 = 4U;
constexpr uint32_t GLI_SEQUSED_K_I64 = 8U;
constexpr uint32_t GLI_RESIDUAL_I64 = 16U;
constexpr uint32_t GLI_START_LOC_I64 = 32U;

// Kernel 侧沿用 PKI 字段名: poolSize=cmp_ratio, maskMode=sparse_mode, sparseCount=打分轴 topk
BEGIN_TILING_DATA_DEF(GenericLightningIndexerTilingData)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, gSize)
TILING_DATA_FIELD_DEF(uint32_t, s1Size)
TILING_DATA_FIELD_DEF(int64_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, sparseCount)
TILING_DATA_FIELD_DEF(uint32_t, topk)
TILING_DATA_FIELD_DEF(uint32_t, poolSize)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, s1BaseSize)
TILING_DATA_FIELD_DEF(uint32_t, mBaseSize)
TILING_DATA_FIELD_DEF(uint32_t, mBaseSizeMax)
TILING_DATA_FIELD_DEF(uint32_t, s2BaseSize)
TILING_DATA_FIELD_DEF(uint32_t, trunkLen)
TILING_DATA_FIELD_DEF(uint32_t, maskMode)
TILING_DATA_FIELD_DEF(uint32_t, quantMode)
TILING_DATA_FIELD_DEF(uint32_t, returnValue)
TILING_DATA_FIELD_DEF(uint32_t, layoutQ)
TILING_DATA_FIELD_DEF(uint32_t, layoutK)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, keyStride0)
TILING_DATA_FIELD_DEF(uint32_t, keyDequantScaleStride0)
TILING_DATA_FIELD_DEF(uint64_t, wsOffScore)
TILING_DATA_FIELD_DEF(uint64_t, wsOffLdScore)
TILING_DATA_FIELD_DEF(uint64_t, wsOffLdIdx)
TILING_DATA_FIELD_DEF(float, qkScale)
TILING_DATA_FIELD_DEF(uint32_t, expandMode)
TILING_DATA_FIELD_DEF(uint32_t, hasWeightedSum)
TILING_DATA_FIELD_DEF(uint32_t, actType)
TILING_DATA_FIELD_DEF(uint32_t, selectMode)
TILING_DATA_FIELD_DEF(uint32_t, keyReduceMode)
TILING_DATA_FIELD_DEF(uint32_t, initBlocks)
TILING_DATA_FIELD_DEF(uint32_t, localBlocks)
TILING_DATA_FIELD_DEF(uint32_t, hasOutputIdxOffset)
TILING_DATA_FIELD_DEF(int32_t, maxSeqlenQ)
TILING_DATA_FIELD_DEF(uint32_t, seqQCuFormat)
TILING_DATA_FIELD_DEF(uint32_t, seqKCuFormat)
TILING_DATA_FIELD_DEF(uint32_t, seqDtypeFlags)
TILING_DATA_FIELD_DEF(uint32_t, msaScoreNT)
TILING_DATA_FIELD_DEF(uint64_t, wsOffKCast)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(GenericLightningIndexer, GenericLightningIndexerTilingData)

struct GenericLightningIndexerCompileInfo {
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND950;
    NpuArch npuArch = NpuArch::DAV_3510;
    uint64_t aicNum = 0;
    uint64_t aivNum = 0;
    uint64_t ubSize = 0;
    uint64_t l1Size = 0;
};

struct GenericLightningIndexerTilingInfo {
    const char *opName = nullptr;
    uint32_t bSize = 0;
    uint32_t n1Size = 0;
    uint32_t gSize = 0;
    uint32_t s1Size = 0;
    int64_t s2Size = 0;
    uint32_t headDim = GLI_HEAD_DIM;
    uint32_t topk = 2048;
    uint32_t sparseCount = 2048;
    uint32_t cmpRatio = 1;
    int32_t sparseMode = 0;
    int32_t actType = 0;
    bool hasWeightedSum = true;
    bool expandIndices = false;
    bool appendTail = false;
    uint32_t expandMode = 0;
    int32_t quantMode = -1;
    bool returnValue = false;
    int32_t selectMode = 0;
    int32_t keyReduceMode = 0;
    uint32_t initBlocks = 0;
    uint32_t localBlocks = 0;
    uint32_t hasOutputIdxOffset = 0;
    int32_t maxSeqlenQ = -1;
    uint32_t seqQCuFormat = 0;
    uint32_t seqKCuFormat = 0;
    uint32_t seqDtypeFlags = 0;
    uint32_t msaScoreNT = 0;
    float scoreScale = 1.0f;
    GliDataLayout layoutQ = GliDataLayout::BSND;
    GliDataLayout layoutK = GliDataLayout::BSND;
    bool pageAttention = false;
    bool paBnbd = false;
    uint32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    uint32_t keyStride0 = 0;
    uint32_t keyDequantScaleStride0 = 0;
    uint32_t usedCoreNum = 0;
    uint32_t aicNum = 0;
    uint32_t aivNum = 0;
    uint64_t ubSize = 0;
    ge::DataType queryDtype = ge::DT_FLOAT16;
    ge::DataType keyDtype = ge::DT_FLOAT16;
};

class GenericLightningIndexerTiling {
public:
    explicit GenericLightningIndexerTiling(gert::TilingContext *context)
        : context_(context)
    {}
    ge::graphStatus DoTiling(GenericLightningIndexerTilingInfo *tilingInfo);

private:
    gert::TilingContext *context_ = nullptr;
    GenericLightningIndexerTilingData tilingData_;

    ge::graphStatus ParseAndCheckParams(GenericLightningIndexerTilingInfo &info);
    ge::graphStatus CalcTilingParams(GenericLightningIndexerTilingInfo &info);
    ge::graphStatus CalcWorkspaceSize(GenericLightningIndexerTilingInfo &info, uint64_t &workspaceSize);
    void SetTilingData(const GenericLightningIndexerTilingInfo &info);
};

} // namespace optiling
#endif
