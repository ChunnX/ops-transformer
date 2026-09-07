/**
 * Copyright (c) 2026 fk2323
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file fmm016_streaming_top_k_blocked_v4_tiling.cpp
 * \brief
 */
#include "../op_kernel/fmm016_router_tiling_v4.h"
#include <register/op_impl_registry.h>
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
namespace optiling {
static ge::graphStatus StreamingTopKBlockedTiling(gert::TilingContext *context)
{
    const auto *qDesc = context->GetInputDesc(0);
    const auto *kDesc = context->GetInputDesc(1);
    const auto *weightDesc = context->GetInputDesc(2);
    const auto *indexDesc = context->GetOutputDesc(0);
    const auto *validDesc = context->GetOutputDesc(1);
    if (qDesc == nullptr || kDesc == nullptr || weightDesc == nullptr || indexDesc == nullptr || validDesc == nullptr ||
        qDesc->GetDataType() != ge::DT_FLOAT || kDesc->GetDataType() != ge::DT_FLOAT ||
        weightDesc->GetDataType() != ge::DT_FLOAT || indexDesc->GetDataType() != ge::DT_INT32 ||
        validDesc->GetDataType() != ge::DT_UINT8) {
        return ge::GRAPH_FAILED;
    }
    const auto *qShape = context->GetInputShape(0);
    const auto *kShape = context->GetInputShape(1);
    const auto *weightShape = context->GetInputShape(2);
    const auto *indexShape = context->GetOutputShape(0);
    const auto *validShape = context->GetOutputShape(1);
    if (qShape == nullptr || kShape == nullptr || weightShape == nullptr || indexShape == nullptr ||
        validShape == nullptr)
        return ge::GRAPH_FAILED;
    const auto &q = qShape->GetStorageShape();
    const auto &kTranspose = kShape->GetStorageShape();
    const auto &weight = weightShape->GetStorageShape();
    if (q.GetDimNum() != 5 || kTranspose.GetDimNum() != 4 || weight.GetDimNum() != 4 ||
        indexShape->GetStorageShape().GetDimNum() != 4 || validShape->GetStorageShape().GetDimNum() != 4)
        return ge::GRAPH_FAILED;
    const int64_t batch = q.GetDim(0), heads = q.GetDim(1), leaves = q.GetDim(2);
    const int64_t groups = q.GetDim(3), indexDim = q.GetDim(4);
    const int64_t paddedLeaves = kTranspose.GetDim(3);
    const int64_t *topKAttr = context->GetAttrs()->GetInt(0);
    const int64_t topK = topKAttr == nullptr ? 1 : *topKAttr;
    const auto &indexOut = indexShape->GetStorageShape();
    const auto &validOut = validShape->GetStorageShape();
    if (batch <= 0 || heads <= 0 || leaves <= 0 || groups <= 0 || groups > 4 || indexDim <= 0 || indexDim > 16 ||
        topK <= 0 || topK > 16 || paddedLeaves < leaves || paddedLeaves % 1024 != 0 || kTranspose.GetDim(0) != batch ||
        kTranspose.GetDim(1) != heads || kTranspose.GetDim(2) != indexDim || weight.GetDim(0) != batch ||
        weight.GetDim(1) != heads || weight.GetDim(2) != leaves || weight.GetDim(3) != groups ||
        indexOut.GetDim(0) != batch || indexOut.GetDim(1) != heads || indexOut.GetDim(2) != leaves ||
        indexOut.GetDim(3) != topK || validOut.GetDim(0) != batch || validOut.GetDim(1) != heads ||
        validOut.GetDim(2) != leaves || validOut.GetDim(3) != topK) {
        return ge::GRAPH_FAILED;
    }
    const uint64_t bh64 = static_cast<uint64_t>(batch) * heads;
    const uint64_t rows64 = bh64 * leaves;
    if (rows64 > std::numeric_limits<uint32_t>::max())
        return ge::GRAPH_FAILED;
    auto *t = context->GetTilingData<Fmm016StreamingTopKBlockedTilingData>();
    t->bhCount = static_cast<uint32_t>(bh64);
    t->leaves = static_cast<uint32_t>(leaves);
    t->paddedLeaves = static_cast<uint32_t>(paddedLeaves);
    t->indexHeads = static_cast<uint32_t>(groups);
    t->indexDim = static_cast<uint32_t>(indexDim);
    t->topK = static_cast<uint32_t>(topK);
    t->totalRows = static_cast<uint32_t>(rows64);
    t->invSqrtIndexDim = 1.0F / std::sqrt(static_cast<float>(indexDim));
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    const uint32_t tasks = (t->totalRows + 31U) / 32U;
    context->SetBlockDim(std::min(tasks, std::max(1U, platform.GetCoreNumAiv())));
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace optiling {
IMPL_OP_OPTILING(Fmm016StreamingTopKBlockedV4).Tiling(StreamingTopKBlockedTiling);
}
