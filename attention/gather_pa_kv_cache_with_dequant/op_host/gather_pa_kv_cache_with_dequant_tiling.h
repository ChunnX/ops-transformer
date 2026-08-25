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
 * \file gather_pa_kv_cache_with_dequant_tiling.h
 * \brief Tiling data structures for fused GatherPaKvCache + Dequant operator
 */

#ifndef GATHER_PA_KV_CACHE_WITH_DEQUANT_TILING_H
#define GATHER_PA_KV_CACHE_WITH_DEQUANT_TILING_H

#include <string>
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "op_host/tiling_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(GatherPaKvCacheWithDequantTilingDataV35)
TILING_DATA_FIELD_DEF(uint32_t, seqLenAccumSize);
TILING_DATA_FIELD_DEF(int64_t, batchCount);
TILING_DATA_FIELD_DEF(int64_t, batchPerCore);
TILING_DATA_FIELD_DEF(uint32_t, needCoreNum);
TILING_DATA_FIELD_DEF(int64_t, kvCacheBlockSize);
TILING_DATA_FIELD_DEF(int64_t, blockTableWidth);
TILING_DATA_FIELD_DEF(int64_t, numBlocks);
TILING_DATA_FIELD_DEF(uint64_t, hiddenSizeK);
TILING_DATA_FIELD_DEF(uint64_t, hiddenSizeV);
TILING_DATA_FIELD_DEF(int64_t, numTokens);
TILING_DATA_FIELD_DEF(uint64_t, maxUbHiddenSizeK);
TILING_DATA_FIELD_DEF(uint64_t, maxUbHiddenSizeV);
TILING_DATA_FIELD_DEF(uint64_t, maxUbHiddenSize);
TILING_DATA_FIELD_DEF(uint64_t, numHeadsK);
TILING_DATA_FIELD_DEF(uint64_t, numHeadsV);
TILING_DATA_FIELD_DEF(uint64_t, headDimK);
TILING_DATA_FIELD_DEF(uint64_t, headDimV);
END_TILING_DATA_DEF;

constexpr uint64_t TILING_KEY_1111 = 1111;
constexpr uint64_t TILING_KEY_1110 = 1110;
constexpr uint64_t TILING_KEY_1101 = 1101;
constexpr uint64_t TILING_KEY_1100 = 1100;

struct GatherPaKvCacheWithDequantTilingKeyMap {
    uint64_t tilingKey;
    bool isSeqLenCumSum;
    bool hasSeqOffset;
};

constexpr std::array<GatherPaKvCacheWithDequantTilingKeyMap, 4> tilingKeyTable = {{
    {TILING_KEY_1111, true, true},
    {TILING_KEY_1110, true, false},
    {TILING_KEY_1101, false, true},
    {TILING_KEY_1100, false, false},
}};

REGISTER_TILING_DATA_CLASS(GatherPaKvCacheWithDequant_1111, GatherPaKvCacheWithDequantTilingDataV35);
REGISTER_TILING_DATA_CLASS(GatherPaKvCacheWithDequant_1110, GatherPaKvCacheWithDequantTilingDataV35);
REGISTER_TILING_DATA_CLASS(GatherPaKvCacheWithDequant_1101, GatherPaKvCacheWithDequantTilingDataV35);
REGISTER_TILING_DATA_CLASS(GatherPaKvCacheWithDequant_1100, GatherPaKvCacheWithDequantTilingDataV35);

struct GatherPaKvCacheWithDequantCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

class GatherPaKvCacheWithDequantTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit GatherPaKvCacheWithDequantTiling(gert::TilingContext *context) : TilingBaseClass(context)
    {
    }
    ~GatherPaKvCacheWithDequantTiling() override
    {
    }
    int64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;
    uint32_t needCoreNum_ = 0;
    uint64_t workspaceSize_ = 0;

protected:
    bool IsCapable() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    int64_t batchCount_ = 0;
    int64_t blockTableWidth_ = 0;
    int64_t blockSize_ = 0;
    uint64_t hiddenSizeK_ = 0;
    uint64_t hiddenSizeV_ = 0;
    int64_t numBlocks_ = 0;
    int64_t numTokens_ = 0;
    uint64_t numHeadsK_ = 0;
    uint64_t numHeadsV_ = 0;
    uint64_t headDimK_ = 0;
    uint64_t headDimV_ = 0;
    bool isSeqLenCumSum_;
    bool hasSeqOffset_;

private:
    ge::graphStatus GetAttrs();
    ge::graphStatus GetInputKeyCache();
    ge::graphStatus GetInputKeyScaleCache();
    ge::graphStatus GetInputValueCache();
    ge::graphStatus GetInputValueScale();
    ge::graphStatus GetInputBlockTables();
    ge::graphStatus GetInputSeqLens();
    ge::graphStatus GetInputOutputKey();
    ge::graphStatus GetInputOutputValue();
    ge::graphStatus GetInputSeqOffset();

    uint64_t tilingKey_ = 0;
    GatherPaKvCacheWithDequantTilingDataV35 tilingData_;
};
} // namespace optiling
#endif
