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
 * \file quant_lightning_indexer_v2_tiling.h
 * \brief
 */

#ifndef QUANT_LIGHTNING_INDEXER_V2_TILING_H
#define QUANT_LIGHTNING_INDEXER_V2_TILING_H

#include "err/ops_err.h"
#include "exe_graph/runtime/tiling_context.h"
#include "platform/platform_info.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "op_host/tiling_util.h"
#include "../../lightning_indexer_v2/op_host/checkers/base_checker_indexer_v2.h"

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
constexpr uint32_t KEY_DEQUANT_SCALE_INDEX = 4;
constexpr uint32_t CU_SEQLENS_Q_INDEX = 5;
constexpr uint32_t BLOCK_TABLE_INDEX = 10;
constexpr uint32_t SPARSE_INDICES_INDEX = 0;
// Attributes Index
constexpr uint32_t ATTR_TOPK_INDEX = 0;
constexpr uint32_t ATTR_QUANT_MODE_INDEX = 1;
constexpr uint32_t ATTR_MAX_SEQLEN_Q_INDEX = 2;
constexpr uint32_t ATTR_QUERY_LAYOUT_INDEX = 3;
constexpr uint32_t ATTR_KEY_LAYOUT_INDEX = 4;
constexpr uint32_t ATTR_MASK_MODE_INDEX = 5;
constexpr uint32_t ATTR_CMP_RATIO_INDEX = 6;
constexpr uint32_t ATTR_RETURN_VALUE_INDEX = 7;

constexpr uint32_t DIM_IDX_ONE = 1;
// 入参限制常量
constexpr int32_t QUANT_MODE_MXFP8 = 3;
constexpr int32_t QUANT_MODE_MXFP4 = 5;
constexpr uint32_t MXFP4_PACK_NUM = 2;       // 每个uint8承载2个FP4 E2M1逻辑元素
constexpr uint32_t MX_SCALE_GROUP_SIZE = 32; // MX量化每32个D维元素对应1个E8M0 scale

// -----------算子TilingData定义---------------
BEGIN_TILING_DATA_DEF(QLIV2TilingData)
TILING_DATA_FIELD_DEF(uint32_t, bSize)
TILING_DATA_FIELD_DEF(uint32_t, n2Size)
TILING_DATA_FIELD_DEF(uint32_t, gSize)
TILING_DATA_FIELD_DEF(uint32_t, s1Size)
TILING_DATA_FIELD_DEF(uint32_t, s2Size)
TILING_DATA_FIELD_DEF(uint32_t, sparseCount)
TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum)
TILING_DATA_FIELD_DEF(uint32_t, blockSize)
TILING_DATA_FIELD_DEF(uint32_t, maxBlockNumPerBatch)
TILING_DATA_FIELD_DEF(uint32_t, sparseMode)
TILING_DATA_FIELD_DEF(uint32_t, cmpRatio)
TILING_DATA_FIELD_DEF(uint32_t, returnValue)
TILING_DATA_FIELD_DEF(int32_t, maxSeqlenQ)
TILING_DATA_FIELD_DEF(uint32_t, keyStride0)
TILING_DATA_FIELD_DEF(uint32_t, keyDequantScaleStride0)
TILING_DATA_FIELD_DEF(uint32_t, quantMode)
END_TILING_DATA_DEF
REGISTER_TILING_DATA_CLASS(QuantLightningIndexerV2, QLIV2TilingData)

// -----------算子CompileInfo定义-------------------
struct QLIV2CompileInfo {};

// -----------算子Tiling入参信息类---------------
class QLIV2TilingInfo {
public:
    indexer_v2_checker::IndexerV2CheckerInfo checkerInfo;
    fe::PlatFormInfos *platformInfo = nullptr;
    // Base Param
    uint32_t bSize = 0;
    uint32_t s1Size = 0;
    int64_t s2Size = 0;
    uint32_t qkHeadDim = 0;
    uint32_t gSize = 0;
    // PageAttention
    bool pageAttentionFlag = false;
    int32_t blockSize = 0;
    uint32_t maxBlockNumPerBatch = 0;
    // Mask
    int32_t sparseMode = 0;
    // Others Flag
    uint32_t sparseCount = 0;
    uint32_t quantMode = 0;
    uint32_t cmpRatio = 1;
    bool returnValue = false;
    uint32_t keyStride0 = 0;
    uint32_t keyDequantScaleStride0 = 0;
    int32_t maxSeqlenQ = -1;
    // DType
    ge::DataType inputQType = ge::DT_FLOAT16;
    ge::DataType inputKType = ge::DT_FLOAT16;
    ge::DataType outputType = ge::DT_INT32;
    // Layout
    DataLayout inputQLayout = DataLayout::BSND;
    DataLayout inputKLayout = DataLayout::PA_BBND;
};

// -----------算子Tiling入参信息解析及Check类---------------
class QLIV2InfoParser {
public:
    explicit QLIV2InfoParser(gert::TilingContext *context)
        : context_(context)
    {}
    ~QLIV2InfoParser() = default;

    ge::graphStatus Parse(QLIV2TilingInfo &qliv2Info) const;

private:
    gert::TilingContext *context_ = nullptr;
};

// ---------------算子Tiling类---------------
class QuantLightningIndexerV2Tiling {
public:
    explicit QuantLightningIndexerV2Tiling(gert::TilingContext *context)
        : context_(context) {};
    ge::graphStatus DoTiling(QLIV2TilingInfo *tilingInfo);

private:
    gert::TilingContext *context_ = nullptr;
    QLIV2TilingData tilingData_;
};

} // namespace optiling
#endif // QUANT_LIGHTNING_INDEXER_V2_TILING_H
