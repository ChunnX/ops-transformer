/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QBSA_COMMON_CHECK_H_
#define QBSA_COMMON_CHECK_H_

#include <exe_graph/runtime/tiling_context.h>
#include "qbsa_tiling_context.h"

namespace optiling {
struct QbsaShapeLimits {
    uint32_t maxBatchSize;
    uint32_t maxQueryHeads;
    uint32_t maxKvHeads;
    uint32_t maxGroupSize;
    uint32_t headDim;
    uint32_t valueHeadDim;
};

// Parameterized validation blocks. A mode strategy explicitly chooses which blocks apply.
class QbsaCommonCheck {
public:
    explicit QbsaCommonCheck(const QuantBlockSparseAttnTilingInfo &tilingInfo);
    ~QbsaCommonCheck() = default;

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

private:
    const QuantBlockSparseAttnTilingInfo &tilingInfo_;
};
} // namespace optiling

#endif // QBSA_COMMON_CHECK_H_
