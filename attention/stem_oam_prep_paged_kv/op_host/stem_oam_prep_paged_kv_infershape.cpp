/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stem_oam_prep_paged_kv_infershape.cpp
 * \brief StemOamPrepPagedKV InferShape implementation
 */
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "log/log.h"
#include "log/error_code.h"

using namespace ge;

namespace ops {

static constexpr uint32_t KCACHE_INDEX = 0;
static constexpr uint32_t VCACHE_INDEX = 1;
static constexpr uint32_t KV_INDICES_INDEX = 2;
static constexpr uint32_t KV_SEQ_LENS_INDEX = 3;
static constexpr uint32_t KSCALE_CACHE_INDEX = 4;
static constexpr uint32_t VSCALE_INDEX = 5;
static constexpr uint32_t KFLAT_INDEX = 0;
static constexpr uint32_t VBIAS_INDEX = 1;

static constexpr int32_t UNKNOWN_DIMS = -2;

static ge::graphStatus InferShapeStemOamPrepPagedKV(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        OP_LOGE("StemOamPrepPagedKV", "context is nullptr!");
        return ge::GRAPH_FAILED;
    }

    OP_LOGD(context->GetNodeName(), "Enter StemOamPrepPagedKV InferShape impl.");

    const gert::Shape *kcacheShape = context->GetInputShape(KCACHE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, kcacheShape);

    const gert::Shape *kvSeqLensShape = context->GetInputShape(KV_SEQ_LENS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, kvSeqLensShape);

    gert::Shape *kflatShape = context->GetOutputShape(KFLAT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, kflatShape);

    gert::Shape *vBiasShape = context->GetOutputShape(VBIAS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, vBiasShape);

    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);

    int64_t kvLayout = attrs->GetAttrValue<int64_t>("CacheLayout");
    int64_t StemBlockSize = attrs->GetAttrValue<int64_t>("StemBlockSize");
    int64_t StemStride = attrs->GetAttrValue<int64_t>("StemStride");

    // UNKNOWN DIM handling
    if ((kcacheShape->GetDimNum() == 1 && kcacheShape->GetDim(0) == UNKNOWN_DIMS) ||
        (kvSeqLensShape->GetDimNum() == 1 && kvSeqLensShape->GetDim(0) == UNKNOWN_DIMS)) {
        kflatShape->SetDimNum(1);
        (*kflatShape)[0] = UNKNOWN_DIMS;
        vBiasShape->SetDimNum(1);
        (*vBiasShape)[0] = UNKNOWN_DIMS;
        return ge::GRAPH_SUCCESS;
    }

    // Determine H_kv based on layout
    int64_t H_kv;
    if (kvLayout == 0) { // Layout A
        H_kv = kcacheShape->GetStorageShape().GetDim(2);
    } else { // Layout B
        H_kv = kcacheShape->GetStorageShape().GetDim(1);
    }

    // batch = kvSeqLens first dim
    int64_t batch = kvSeqLensShape->GetStorageShape().GetDim(0);

    // Derive output shapes
    int64_t kflat_dim = stemStride * 128; // S * D

    // max_kv_len from kvSeqLens (runtime) — placeholder for static shape
    // For infershape we use -1 for dynamic dimensions
    kflatShape->SetDimNum(4);
    (*kflatShape)[0] = batch;
    (*kflatShape)[1] = H_kv;
    (*kflatShape)[2] = -1; // max_Kb (dynamic)
    (*kflatShape)[3] = kflat_dim;

    vBiasShape->SetDimNum(3);
    (*vBiasShape)[0] = batch;
    (*vBiasShape)[1] = H_kv;
    (*vBiasShape)[2] = -1; // max_Kb (dynamic)

    OP_LOGD(context->GetNodeName(), "StemOamPrepPagedKV InferShape success.");
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeStemOamPrepPagedKV(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    context->SetOutputDataType(KFLAT_INDEX, ge::DT_BFLOAT16);
    context->SetOutputDataType(VBIAS_INDEX, ge::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(StemOamPrepPagedKV)
    .InferShape(InferShapeStemOamPrepPagedKV)
    .InferDataType(InferDataTypeStemOamPrepPagedKV);

} // namespace ops