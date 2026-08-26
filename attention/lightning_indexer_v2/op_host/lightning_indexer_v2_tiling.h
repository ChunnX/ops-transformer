/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
  */

/*!
 * \file lightning_indexer_v2_tiling.h
 * \brief
 */

#ifndef LIGHTNING_INDEXER_V2_TILING_H_
#define LIGHTNING_INDEXER_V2_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "err/ops_err.h"
#include "platform/platform_info.h"
#include "op_host/tiling_util.h"
#include "checkers/base_checker_indexer_v2.h"

namespace optiling {
enum class DataLayout : uint32_t {
    BSND = 0,
    TND = 1,
    PA_BBND = 2
};

// ------------------算子原型索引常量定义----------------
// Inputs Index
constexpr uint32_t QUERY_INDEX = 0;
constexpr uint32_t KEY_INDEX = 1;
constexpr uint32_t CU_SEQLENS_Q_INDEX = 3;
constexpr uint32_t BLOCK_TABLE_INDEX = 8;
// Outputs Index
constexpr uint32_t LIGHTNING_INDEXER = 0;
// Attributes Index
constexpr uint32_t ATTR_TOPK_INDEX = 0;
constexpr uint32_t ATTR_QUERY_LAYOUT_INDEX = 2;
constexpr uint32_t ATTR_KEY_LAYOUT_INDEX = 3;
constexpr uint32_t ATTR_MASK_MODE_INDEX = 4;
constexpr uint32_t ATTR_CMP_RATIO_INDEX = 5;
constexpr uint32_t ATTR_RETURN_VALUE_INDEX = 6;
// Dim Index
constexpr uint32_t DIM_IDX_ZERO = 0;
constexpr uint32_t DIM_IDX_ONE = 1;
constexpr uint32_t DIM_IDX_TWO = 2;
constexpr uint32_t DIM_IDX_THREE = 3;
// -----------算子TilingData定义---------------
BEGIN_TILING_DATA_DEF(LIV2TilingData)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, n2Size)
TILING_DATA_FIELD_DEF(uint32_t, gSize)
TILING_DATA_FIELD_DEF(uint32_t, s1Size)
TILING_DATA_FIELD_DEF(uint32_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, topk)
TILING_DATA_FIELD_DEF(uint32_t, maxSeqlenQ)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, maskMode)
TILING_DATA_FIELD_DEF(int64_t, preTokens)
TILING_DATA_FIELD_DEF(int64_t, nextTokens)
TILING_DATA_FIELD_DEF(int64_t, cmpRatio)
TILING_DATA_FIELD_DEF(uint32_t, batchSupperFlag)
TILING_DATA_FIELD_DEF(uint32_t, keyStride0)
TILING_DATA_FIELD_DEF(uint32_t, returnValue)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(LightningIndexerV2, LIV2TilingData)

// -----------算子CompileInfo定义-------------------
struct LIV2CompileInfo {};

// -----------算子Tiling入参信息类---------------
class LIV2TilingInfo {
public:
    indexer_v2_checker::IndexerV2CheckerInfo checkerInfo;
    fe::PlatFormInfos *platformInfo = nullptr;
    // Base Param
    uint32_t bSize = 0;
    uint32_t s1Size = 0;
    int64_t s2Size = 0;
    uint32_t gSize = 0;
    // PageAttention
    bool pageAttentionFlag = false;
    int32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    // Mask
    int32_t maskMode = 0;
    // Others Flag
    uint32_t topk = 0;
    int64_t preTokens = INT64_MAX;
    int64_t nextTokens = INT64_MAX;
    int64_t cmpRatio = 1;
    uint32_t batchSupperFlag = 0;
    uint32_t returnValue = 0;
    uint32_t keyStride0 = 0;

    // DType
    ge::DataType inputQType = ge::DT_FLOAT16;
    ge::DataType inputKType = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_INT32;
    // Layout
    DataLayout inputQLayout = DataLayout::BSND;
    DataLayout inputKLayout = DataLayout::PA_BBND;
};

// -----------算子Tiling入参信息解析及Check类---------------
class LIV2InfoParser {
public:
    explicit LIV2InfoParser(gert::TilingContext *context)
        : context_(context)
    {}
    ~LIV2InfoParser() = default;

    ge::graphStatus Parse(LIV2TilingInfo &liInfo) const;

private:
    gert::TilingContext *context_ = nullptr;
};

// ---------------算子Tiling类---------------
class LightningIndexerV2Tiling {
public:
    explicit LightningIndexerV2Tiling(gert::TilingContext *context)
        : context_(context) {};
    ge::graphStatus DoTiling(LIV2TilingInfo *tilingInfo);

private:
    gert::TilingContext *context_ = nullptr;
    LIV2TilingData tilingData_;
};
} // namespace optiling
#endif // LIGHTNING_INDEXER_V2_TILING_H_
