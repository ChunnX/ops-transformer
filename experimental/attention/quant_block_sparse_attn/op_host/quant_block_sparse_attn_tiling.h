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
 * \file quant_block_sparse_attn_tiling.h
 * \brief QuantBlockSparseAttn tiling data definitions.
 */

#ifndef QUANT_BLOCK_SPARSE_ATTN_TILING_H_
#define QUANT_BLOCK_SPARSE_ATTN_TILING_H_

#include <cstdint>
#include <limits>
#include <string>

#include <exe_graph/runtime/tiling_context.h>
#include <register/op_impl_registry.h>
#include "../op_kernel/quant_block_sparse_attn_const.h"
#include "../op_kernel/arch35/quant_block_sparse_attn_tiling_data.h"
#include "quant_block_sparse_attn_info_parser.h"

namespace optiling {
constexpr uint32_t QBSA_BLOCK_SIZE = 128U;
constexpr uint32_t QBSA_D_SIZE = 128U;
constexpr uint32_t QBSA_QUANT_MODE_FP8 = 1U;
constexpr uint32_t QBSA_QUANT_MODE_MXFP8_FULL_QUANT = 2U;
constexpr uint32_t QBSA_MASK_MODE_NONE = 0U;
constexpr uint32_t QBSA_MASK_MODE_CAUSAL = 3U;
constexpr uint32_t QBSA_MASK_MODE_MAX = 4U;
constexpr uint32_t QBSA_MXFP8_S2_BASE_SIZE = 512U;
constexpr uint32_t QBSA_MXFP8_MAX_PA_BLOCK_SIZE = 1024U;
constexpr uint32_t QBSA_MXFP8_SPARSE_BLOCK_SIZE_128 = 128U;
constexpr uint32_t QBSA_MXFP8_SPARSE_BLOCK_SIZE_64 = 64U;
constexpr uint32_t QBSA_MXFP8_SCALE_GROUP_SIZE = 64U;
constexpr uint32_t QBSA_MXFP8_SCALE_LAST_DIM = 2U;
constexpr uint32_t QBSA_MXFP8_PER_TOKEN_GROUP_MODE = 6U;
constexpr uint32_t QBSA_MXFP8_PER_CHANNEL_GROUP_MODE = 8U;
constexpr uint32_t QBSA_MAX_BATCH_SIZE = 65536U;
constexpr uint32_t QBSA_MAX_N2_SIZE = 8U;
constexpr uint32_t QBSA_MAX_G_SIZE = 16U;

inline uint32_t QBSACeilDiv(uint32_t value, uint32_t divisor)
{
    return divisor == 0U ? 0U : (value + divisor - 1U) / divisor;
}

inline bool QBSAGetDimAsU32(const gert::Shape &shape, size_t dimIndex, uint32_t &value)
{
    if (dimIndex >= shape.GetDimNum()) {
        return false;
    }
    const auto dim = shape.GetDim(dimIndex);
    if (dim <= 0 || dim > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    value = static_cast<uint32_t>(dim);
    return true;
}

inline uint32_t QBSAGetPositiveAttr(const gert::RuntimeAttrs *attrs, uint32_t index, uint32_t defaultValue)
{
    const int64_t *attrPtr = attrs->GetAttrPointer<int64_t>(index);
    if (attrPtr == nullptr || *attrPtr <= 0) {
        return defaultValue;
    }
    if (*attrPtr > std::numeric_limits<uint32_t>::max()) {
        return defaultValue;
    }
    return static_cast<uint32_t>(*attrPtr);
}

inline uint32_t QBSAGetUintAttr(const gert::RuntimeAttrs *attrs, uint32_t index, uint32_t defaultValue)
{
    const int64_t *attrPtr = attrs->GetAttrPointer<int64_t>(index);
    if (attrPtr == nullptr || *attrPtr < 0) {
        return defaultValue;
    }
    if (*attrPtr > std::numeric_limits<uint32_t>::max()) {
        return defaultValue;
    }
    return static_cast<uint32_t>(*attrPtr);
}

inline float QBSAGetFloatAttr(const gert::RuntimeAttrs *attrs, uint32_t index, float defaultValue)
{
    const float *attrPtr = attrs->GetAttrPointer<float>(index);
    return attrPtr == nullptr ? defaultValue : *attrPtr;
}

inline bool QBSAGetBoolAttr(const gert::RuntimeAttrs *attrs, uint32_t index, bool defaultValue)
{
    const bool *attrPtr = attrs->GetAttrPointer<bool>(index);
    return attrPtr == nullptr ? defaultValue : *attrPtr;
}

inline std::string QBSAGetStringAttr(const gert::RuntimeAttrs *attrs, uint32_t index, const char *defaultValue)
{
    const char *attrPtr = attrs->GetAttrPointer<char>(index);
    return attrPtr == nullptr ? std::string(defaultValue) : std::string(attrPtr);
}

ge::graphStatus TilingQuantBlockSparseAttn(gert::TilingContext *context);

class QuantBlockSparseAttnTiling {
public:
    explicit QuantBlockSparseAttnTiling(gert::TilingContext *context);
    ~QuantBlockSparseAttnTiling() = default;

    ge::graphStatus DoOpTiling(QuantBlockSparseAttnTilingInfo *tilingInfo);

private:
    void FillCommonParams();
    void FillFp8Params();
    void FillMxParams();
    ge::graphStatus SelectFeature();
    void PrintTilingData();
    ge::graphStatus SaveTilingData();

    gert::TilingContext *context_ = nullptr;
    QuantBlockSparseAttnTilingInfo *tilingInfo_ = nullptr;
    QuantBlockSparseAttnTilingData tilingData_;
    uint32_t usedCoreNum_ = 0;
    uint64_t totalTaskNum_ = 0;
    uint64_t tilingKey_ = 0;
    uint64_t workspaceSize_ = 0;
};

} // namespace optiling

#endif // QUANT_BLOCK_SPARSE_ATTN_TILING_H_
