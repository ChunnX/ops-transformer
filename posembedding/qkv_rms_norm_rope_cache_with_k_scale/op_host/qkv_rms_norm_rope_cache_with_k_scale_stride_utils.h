/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_STRIDE_UTILS_H_
#define QKV_RMS_NORM_ROPE_CACHE_WITH_K_SCALE_STRIDE_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <limits>

namespace QkvRmsNormRopeCacheWithKScaleStrideUtils {

// Validate a semantic-order positive-stride view. Padding is allowed, but an
// outer-axis step must cover the complete span of every inner axis. ACLNN and
// graph tiling share this checked formula so their cache contracts cannot drift.
template <typename DimAt, typename StrideAt>
inline bool IsNonOverlappingPositiveStrides(size_t rank, DimAt dimAt, StrideAt strideAt)
{
    uint64_t innerSpan = 1U;
    for (size_t offset = 0U; offset < rank; ++offset) {
        const size_t index = rank - 1U - offset;
        const int64_t dim = dimAt(index);
        const int64_t stride = strideAt(index);
        if (dim <= 0 || stride <= 0) {
            return false;
        }
        const uint64_t uDim = static_cast<uint64_t>(dim);
        const uint64_t uStride = static_cast<uint64_t>(stride);
        if (uDim > 1U && uStride < innerSpan) {
            return false;
        }
        const uint64_t extent = uDim - 1U;
        if (extent != 0U && uStride > (std::numeric_limits<uint64_t>::max() - innerSpan) / extent) {
            return false;
        }
        innerSpan += extent * uStride;
    }
    return true;
}

} // namespace QkvRmsNormRopeCacheWithKScaleStrideUtils

#endif
