/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QBSA_OP_CONTRACT_H_
#define QBSA_OP_CONTRACT_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include <exe_graph/runtime/tiling_context.h>

namespace optiling {
// Inputs, outputs and attributes follow the order declared by QuantBlockSparseAttn's OpDef.
constexpr uint32_t QBSA_QUERY_INDEX = 0U;
constexpr uint32_t QBSA_KEY_INDEX = 1U;
constexpr uint32_t QBSA_VALUE_INDEX = 2U;
constexpr uint32_t QBSA_Q_DESCALE_INDEX = 3U;
constexpr uint32_t QBSA_K_DESCALE_INDEX = 4U;
constexpr uint32_t QBSA_V_DESCALE_INDEX = 5U;
constexpr uint32_t QBSA_P_SCALE_INDEX = 6U;
constexpr uint32_t QBSA_CU_SEQLENS_Q_INDEX = 7U;
constexpr uint32_t QBSA_CU_SEQLENS_KV_INDEX = 8U;
constexpr uint32_t QBSA_SEQUSED_Q_INDEX = 9U;
constexpr uint32_t QBSA_SEQUSED_KV_INDEX = 10U;
constexpr uint32_t QBSA_SPARSE_INDICES_INDEX = 11U;
constexpr uint32_t QBSA_SPARSE_SEQ_LEN_INDEX = 12U;
constexpr uint32_t QBSA_BLOCK_TABLE_INDEX = 13U;
constexpr uint32_t QBSA_ATTEN_MASK_INDEX = 14U;
constexpr uint32_t QBSA_METADATA_INDEX = 15U;

constexpr uint32_t QBSA_ATTENTION_OUT_INDEX = 0U;
constexpr uint32_t QBSA_SOFTMAX_LSE_INDEX = 1U;

constexpr uint32_t QBSA_SOFTMAX_SCALE_ATTR_INDEX = 0U;
constexpr uint32_t QBSA_SPARSE_Q_BLOCK_SIZE_ATTR_INDEX = 1U;
constexpr uint32_t QBSA_SPARSE_KV_BLOCK_SIZE_ATTR_INDEX = 2U;
constexpr uint32_t QBSA_LAYOUT_KV_ATTR_INDEX = 3U;
constexpr uint32_t QBSA_LAYOUT_Q_ATTR_INDEX = 4U;
constexpr uint32_t QBSA_LAYOUT_SPARSE_INDICES_ATTR_INDEX = 5U;
constexpr uint32_t QBSA_LAYOUT_OUT_ATTR_INDEX = 6U;
constexpr uint32_t QBSA_QUANT_MODE_ATTR_INDEX = 7U;
constexpr uint32_t QBSA_MASK_MODE_ATTR_INDEX = 8U;
constexpr uint32_t QBSA_RETURN_SOFTMAX_LSE_ATTR_INDEX = 9U;

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

inline bool IsQbsaQuantModeSupported(uint32_t quantMode)
{
    return quantMode == QBSA_QUANT_MODE_FP8 || quantMode == QBSA_QUANT_MODE_MXFP8_FULL_QUANT;
}

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
    if (attrPtr == nullptr || *attrPtr <= 0 || *attrPtr > std::numeric_limits<uint32_t>::max()) {
        return defaultValue;
    }
    return static_cast<uint32_t>(*attrPtr);
}

inline uint32_t QBSAGetUintAttr(const gert::RuntimeAttrs *attrs, uint32_t index, uint32_t defaultValue)
{
    const int64_t *attrPtr = attrs->GetAttrPointer<int64_t>(index);
    if (attrPtr == nullptr || *attrPtr < 0 || *attrPtr > std::numeric_limits<uint32_t>::max()) {
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
} // namespace optiling

#endif // QBSA_OP_CONTRACT_H_
