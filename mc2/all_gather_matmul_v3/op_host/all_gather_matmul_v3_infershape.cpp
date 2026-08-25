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
 * \file all_gather_matmul_v3_infershape.cpp
 * \brief InferShape for AllGatherMatmulV3
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
constexpr size_t INDEX_ATTR_RANK_SIZE = 4;

constexpr int64_t MX_SCALE_BLOCK_SIZE = 64;
constexpr int64_t SCALE_LAST_DIM = 2;
constexpr int64_t DIM_MINUS_ONE = -1;

} // namespace

ge::graphStatus AllGatherMatmulV3InferShape(gert::InferShapeContext *context)
{
    const auto nodeName = context->GetNodeName();
    OP_LOGD(nodeName, "InferShape start");

    // ---- Get input shapes -----------------------------------------------
    const auto x1Shape = context->GetInputShape(INDEX_IN_X1);
    const auto x2Shape = context->GetInputShape(INDEX_IN_X2);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x1Shape);
    OPS_CHECK_NULL_WITH_CONTEXT(context, x2Shape);
    OPS_CHECK(x1Shape->GetDimNum() < 2 || x2Shape->GetDimNum() < 2,
              OP_LOGE(nodeName, "x1 and x2 must have at least 2D, got x1=%luD x2=%luD.",
                      x1Shape->GetDimNum(), x2Shape->GetDimNum()), return ge::GRAPH_FAILED);

    int64_t m = x1Shape->GetDim(0U);   // per-rank M
    int64_t k = x1Shape->GetDim(1U);   // full K (not split across ranks)
    // x2 shape: is_trans_b=true → [N, K]
    constexpr size_t IDX_ATTR_IS_TRANS_B = 3;
    auto *tx2 = context->GetAttrs()->GetAttrPointer<bool>(IDX_ATTR_IS_TRANS_B);
    bool transX2 = tx2 ? *tx2 : true;
    int64_t n = transX2 ? x2Shape->GetDim(0U) : x2Shape->GetDim(1U);
    int64_t kX2 = transX2 ? x2Shape->GetDim(1U) : x2Shape->GetDim(0U);

    OP_LOGD(nodeName, "InferShape: x1[%ld,%ld] x2 K=%ld N=%ld", m, k, kX2, n);

    // ---- Validate dtypes -------------------------------------------------
    auto x1Dtype = context->GetInputDesc(INDEX_IN_X1)->GetDataType();
    auto x2Dtype = context->GetInputDesc(INDEX_IN_X2)->GetDataType();

    if ((x1Dtype != ge::DT_FLOAT8_E4M3FN && x1Dtype != ge::DT_FLOAT8_E5M2 && x1Dtype != ge::DT_FLOAT4_E2M1) ||
        (x2Dtype != ge::DT_FLOAT8_E4M3FN && x2Dtype != ge::DT_FLOAT8_E5M2 && x2Dtype != ge::DT_FLOAT4_E2M1)) {
        OP_LOGE(nodeName, "Unsupported dtype, x1_dtype=%d x2_dtype=%d, only FP8/FP4 supported.",
                static_cast<int>(x1Dtype), static_cast<int>(x2Dtype));
        return ge::GRAPH_FAILED;
    }
    if ((x1Dtype == ge::DT_FLOAT4_E2M1) != (x2Dtype == ge::DT_FLOAT4_E2M1)) {
        OP_LOGE(nodeName, "FP4 requires both x1 and x2 to be FP4_E2M1, got x1_dtype=%d x2_dtype=%d.",
                static_cast<int>(x1Dtype), static_cast<int>(x2Dtype));
        return ge::GRAPH_FAILED;
    }

    // ---- Get world_size attr ---------------------------------------------
    const auto attrs = context->GetAttrs();
    const int64_t *wsPtr = attrs->GetAttrPointer<int64_t>(INDEX_ATTR_RANK_SIZE);
    OPS_CHECK(wsPtr == nullptr, OP_LOGE(nodeName, "rank_size attr not found"), return ge::GRAPH_FAILED);
    int64_t worldSize = *wsPtr;

    // ---- Validate K dimension cross-check: x1.K == x2.K (full K) ---------
    OPS_CHECK(kX2 != k,
              OP_LOGE(nodeName, "K dimension mismatch: x1.K=%ld != x2.K=%ld", k, kX2),
              return ge::GRAPH_FAILED);

    // ---- Validate constraints (skip if dynamic shape) --------------------
    if (m != DIM_MINUS_ONE && k != DIM_MINUS_ONE && n != DIM_MINUS_ONE) {
        OPS_CHECK(k < 256 || k >= 65535,
                  OP_LOGE(nodeName, "K=%ld must be in range [256, 65535)", k),
                  return ge::GRAPH_FAILED);

        // fp4 要求 K 为偶数
        if (x1Dtype == ge::DT_FLOAT4_E2M1) {
            OPS_CHECK(k % 2 != 0,
                      OP_LOGE(nodeName, "K=%ld must be even in mxfp4 scene", k),
                      return ge::GRAPH_FAILED);
        }

        // Validate scale shapes
        // x1_scale: [M, Ceil(K/64), 2]; x2_scale: [N, Ceil(K/64), 2]
        auto x1ScaleShape = context->GetInputShape(INDEX_IN_X1_SCALE);
        auto x2ScaleShape = context->GetInputShape(INDEX_IN_X2_SCALE);
        if (x1ScaleShape != nullptr && x2ScaleShape != nullptr) {
            int64_t expectedScaleK = CeilDiv(k, MX_SCALE_BLOCK_SIZE);
            OPS_CHECK(x1ScaleShape->GetDim(1U) != static_cast<uint64_t>(expectedScaleK) ||
                          x1ScaleShape->GetDim(2U) != static_cast<uint64_t>(SCALE_LAST_DIM),
                      OP_LOGE(nodeName, "x1_scale shape mismatch, expected dim1=%ld dim2=%ld got dim1=%lu dim2=%lu.",
                              expectedScaleK, SCALE_LAST_DIM,
                              x1ScaleShape->GetDim(1U), x1ScaleShape->GetDim(2U)), return ge::GRAPH_FAILED);
            OPS_CHECK(x2ScaleShape->GetDim(1U) != static_cast<uint64_t>(expectedScaleK) ||
                          x2ScaleShape->GetDim(2U) != static_cast<uint64_t>(SCALE_LAST_DIM),
                      OP_LOGE(nodeName, "x2_scale shape mismatch, expected dim1=%ld dim2=%ld got dim1=%lu dim2=%lu.",
                              expectedScaleK, SCALE_LAST_DIM,
                              x2ScaleShape->GetDim(1U), x2ScaleShape->GetDim(2U)), return ge::GRAPH_FAILED);
        }
    }

    // ---- Derive output shape: y = [world_size * M_per_rank, N] -----------
    auto outShape = context->GetOutputShape(INDEX_OUT_Y);
    OPS_CHECK_NULL_WITH_CONTEXT(context, outShape);

    outShape->SetDimNum(2);
    if (m != DIM_MINUS_ONE && n != DIM_MINUS_ONE) {
        outShape->SetDim(0U, m * worldSize); // gathered total M
        outShape->SetDim(1U, n);
    } else {
        outShape->SetDim(0U, DIM_MINUS_ONE);
        outShape->SetDim(1U, DIM_MINUS_ONE);
    }

    // ---- gather_out = [world_size * M_per_rank, K] (REQUIRED) ----
    constexpr size_t INDEX_OUT_GATHER_OUT = 1;
    auto gatherOutShape = context->GetOutputShape(INDEX_OUT_GATHER_OUT);
    if (gatherOutShape != nullptr) {
        gatherOutShape->SetDimNum(2);
        if (m != DIM_MINUS_ONE && k != DIM_MINUS_ONE) {
            gatherOutShape->SetDim(0U, m * worldSize);
            gatherOutShape->SetDim(1U, k);
        } else {
            gatherOutShape->SetDim(0U, DIM_MINUS_ONE);
            gatherOutShape->SetDim(1U, DIM_MINUS_ONE);
        }
    }

    // Dtype and format are inferred from the proto definition

    OP_LOGD(nodeName,
            "InferShape success: M_per_rank=%ld K=%ld N=%ld world_size=%ld, y=[%ld,%ld] gather_out=[%ld,%ld]",
            m, k, n, worldSize,
            outShape->GetDim(0U), outShape->GetDim(1U),
            gatherOutShape != nullptr ? gatherOutShape->GetDim(0U) : 0L,
            gatherOutShape != nullptr ? gatherOutShape->GetDim(1U) : 0L);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(AllGatherMatmulV3).InferShape(AllGatherMatmulV3InferShape);

} // namespace ops
