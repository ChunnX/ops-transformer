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
 * \file allto_all_matmul_apace_infershape.cpp
 * \brief InferShape for AlltoAllMatmulApace
 */

#include <register/op_impl_registry.h>

#include "common/utils/op_mc2.h"
#include "mc2_log.h"
#include "op_host/mc2_common_infershape.h"
#include "util/math_util.h"

namespace ops {

using Ops::Base::CeilDiv;

namespace {

constexpr size_t INDEX_IN_X1 = 0;
constexpr size_t INDEX_IN_X2 = 1;
constexpr size_t INDEX_IN_X1_SCALE = 2;
constexpr size_t INDEX_IN_X2_SCALE = 3;
constexpr size_t INDEX_OUT_Y = 0;
constexpr size_t INDEX_ATTR_WORLD_SIZE = 1;

constexpr int64_t HEAD_M_SIZE = 512;
constexpr int64_t MX_SCALE_BLOCK_SIZE = 64;
constexpr int64_t MX_SCALE_LAST_DIM = 2;
constexpr int64_t DIM_MINUS_ONE = -1;

} // namespace

ge::graphStatus AlltoAllMatmulApaceInferShape(gert::InferShapeContext *context)
{
    const auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "AlltoAllMatmulApace InferShape start");

    // ---- Get input shapes -----------------------------------------------
    const auto x1Shape = context->GetInputShape(INDEX_IN_X1);
    const auto x2Shape = context->GetInputShape(INDEX_IN_X2);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    OPS_CHECK(x1Shape->GetDimNum() < 2 || x2Shape->GetDimNum() < 2,
              OP_LOGE(nodeName, "x1 and x2 must have at least 2 dimensions"), return ge::GRAPH_FAILED);

    int64_t m = x1Shape->GetDim(0U);
    int64_t k = x1Shape->GetDim(1U);
    int64_t n = x2Shape->GetDim(0U);
    int64_t k2 = x2Shape->GetDim(1U);

    OPS_CHECK(k != k2, OP_LOGE(nodeName, "K dimension mismatch: x1.K=%ld != x2.K=%ld", k, k2), return ge::GRAPH_FAILED);

    // ---- Validate dtypes ------------------------------------------------
    auto x1Dtype = context->GetInputDesc(INDEX_IN_X1)->GetDataType();
    auto x2Dtype = context->GetInputDesc(INDEX_IN_X2)->GetDataType();

    if ((x1Dtype != ge::DT_FLOAT8_E4M3FN && x1Dtype != ge::DT_FLOAT8_E5M2) ||
        (x2Dtype != ge::DT_FLOAT8_E4M3FN && x2Dtype != ge::DT_FLOAT8_E5M2)) {
        OP_LOGE(nodeName, "Unsupported dtype. Only FP8 supported.");
        return ge::GRAPH_FAILED;
    }

    // ---- Get world_size attr --------------------------------------------
    const auto attrs = context->GetAttrs();
    const int64_t *wsPtr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_WORLD_SIZE);
    OPS_CHECK(wsPtr == nullptr, OP_LOGE(nodeName, "world_size attr not found"), return ge::GRAPH_FAILED);
    int64_t worldSize = *wsPtr;

    // ---- Validate constraints (skip if dynamic shape) -------------------
    if (m != DIM_MINUS_ONE && k != DIM_MINUS_ONE && n != DIM_MINUS_ONE) {
        OPS_CHECK(m % worldSize != 0, OP_LOGE(nodeName, "M=%ld not divisible by world_size=%ld", m, worldSize),
                  return ge::GRAPH_FAILED);
        OPS_CHECK(k % worldSize != 0, OP_LOGE(nodeName, "K=%ld not divisible by world_size=%ld", k, worldSize),
                  return ge::GRAPH_FAILED);
        int64_t ka = k / worldSize;
        int64_t ceilDivKa = CeilDiv(ka, MX_SCALE_BLOCK_SIZE);
        OPS_CHECK(ceilDivKa % 2 != 0, OP_LOGE(nodeName, "CeilDiv(Ka,64)=%ld must be even, Ka=%ld", ceilDivKa, ka),
                  return ge::GRAPH_FAILED);
        OPS_CHECK(m % HEAD_M_SIZE != 0, OP_LOGE(nodeName, "M=%ld not divisible by headMSize=%ld", m, HEAD_M_SIZE),
                  return ge::GRAPH_FAILED);

        // Validate scale shapes
        auto x1ScaleShape = context->GetInputShape(INDEX_IN_X1_SCALE);
        auto x2ScaleShape = context->GetInputShape(INDEX_IN_X2_SCALE);
        if (x1ScaleShape != nullptr && x2ScaleShape != nullptr) {
            int64_t expectedScaleK = CeilDiv(k, MX_SCALE_BLOCK_SIZE);
            OPS_CHECK(x1ScaleShape->GetDim(1U) != static_cast<uint64_t>(expectedScaleK) ||
                          x1ScaleShape->GetDim(2U) != static_cast<uint64_t>(MX_SCALE_LAST_DIM),
                      OP_LOGE(nodeName, "x1_scale shape mismatch"), return ge::GRAPH_FAILED);
            OPS_CHECK(x2ScaleShape->GetDim(1U) != static_cast<uint64_t>(expectedScaleK) ||
                          x2ScaleShape->GetDim(2U) != static_cast<uint64_t>(MX_SCALE_LAST_DIM),
                      OP_LOGE(nodeName, "x2_scale shape mismatch"), return ge::GRAPH_FAILED);
        }
    }

    // ---- Derive output shape: y = [M / world_size, N] -------------------
    auto outShape = context->GetOutputShape(INDEX_OUT_Y);
    OPS_CHECK_NULL_WITH_CONTEXT(context, outShape);

    outShape->SetDimNum(2);
    if (m != DIM_MINUS_ONE && n != DIM_MINUS_ONE) {
        outShape->SetDim(0U, m / worldSize);
        outShape->SetDim(1U, n);
    } else {
        outShape->SetDim(0U, DIM_MINUS_ONE);
        outShape->SetDim(1U, DIM_MINUS_ONE);
    }

    // Dtype and format are inferred from the proto definition

    OP_LOGD(nodeName, "AlltoAllMatmulApace InferShape success: M=%ld K=%ld N=%ld WS=%ld", m, k, n, worldSize);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AlltoAllMatmulApace).InferShape(AlltoAllMatmulApaceInferShape);

} // namespace ops
