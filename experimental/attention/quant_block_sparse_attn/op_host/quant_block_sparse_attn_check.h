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
 * \file quant_block_sparse_attn_check.h
 * \brief QuantBlockSparseAttn parameter validation class.
 */

#ifndef QUANT_BLOCK_SPARSE_ATTN_CHECK_H_
#define QUANT_BLOCK_SPARSE_ATTN_CHECK_H_

#include <exe_graph/runtime/tiling_context.h>
#include "quant_block_sparse_attn_info_parser.h"

namespace optiling {

struct QbsaShapeLimits {
    uint32_t maxBatchSize;
    uint32_t maxQueryHeads;
    uint32_t maxKvHeads;
    uint32_t maxGroupSize;
    uint32_t headDim;
    uint32_t valueHeadDim;
};

class QuantBlockSparseAttnCheck {
public:
    explicit QuantBlockSparseAttnCheck(const QuantBlockSparseAttnTilingInfo &tilingInfo);
    ~QuantBlockSparseAttnCheck() = default;

    ge::graphStatus Process();

private:
    // Reusable validation blocks; each mode explicitly selects the applicable checks.
    ge::graphStatus CheckLayouts(const char *kvLayout, const char *sparseLayout) const;
    ge::graphStatus CheckSoftmaxScale(float minExclusive, float maxInclusive) const;
    ge::graphStatus CheckDtype(ge::DataType qkvDtype, ge::DataType outputDtype) const;
    ge::graphStatus CheckFormat(ge::Format expectedFormat) const;
    ge::graphStatus CheckActualSeqLen() const;
    ge::graphStatus CheckMetadata() const;
    ge::graphStatus CheckPaInputs() const;
    ge::graphStatus CheckReservedInputs() const;
    ge::graphStatus CheckShapeConsistency() const;
    ge::graphStatus CheckShapeLimits(const QbsaShapeLimits &limits) const;
    ge::graphStatus CheckKeyValueShape(uint32_t expectedBlockSize) const;
    ge::graphStatus CheckNoneOrCausalMask() const;
    // Mode-specific policies and their validation entry points.
    ge::graphStatus CheckFp8Attrs() const;
    ge::graphStatus CheckMxAttrs() const;
    ge::graphStatus CheckFp8Inputs() const;
    ge::graphStatus CheckFp8PaStride() const;
    ge::graphStatus CheckFp8ScaleDtype() const;
    ge::graphStatus CheckMxScaleDtype() const;
    ge::graphStatus CheckFp8BlockSize() const;
    ge::graphStatus CheckMxBlockSize() const;
    ge::graphStatus CheckFp8QuantShape() const;
    ge::graphStatus CheckMXFP8FullQuantShape() const;
    ge::graphStatus CheckFp8Feature() const;
    ge::graphStatus CheckMxFeature() const;

    const QuantBlockSparseAttnTilingInfo &tilingInfo_;
};

} // namespace optiling

#endif // QUANT_BLOCK_SPARSE_ATTN_CHECK_H_
