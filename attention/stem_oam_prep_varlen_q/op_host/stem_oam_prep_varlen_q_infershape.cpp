/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <exe_graph/runtime/infer_shape_context.h>
#include <register/op_impl_registry.h>

namespace ops {

static constexpr int64_t DIM_QK = 128;
static constexpr int64_t INPUT_Q = 0;
static constexpr int64_t INPUT_Q_SEQ_LENS = 1;
static constexpr int64_t OUTPUT_Q_FLAT = 0;
static constexpr int64_t ATTR_STEM_BLOCK_SIZE = 0;
static constexpr int64_t ATTR_STEM_STRIDE = 1;

static ge::graphStatus InferShapeForStemOamPrepVarlenQ(gert::InferShapeContext* context) {
    auto attrs = context->GetAttrs();
    const int64_t *stemStridePtr = attrs->GetInt(ATTR_STEM_STRIDE);
    if (stemStridePtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t stemStride = *stemStridePtr;
    int64_t kflat_dim = stemStride * DIM_QK;

    const gert::Shape *qSeqLensShape = context->GetInputShape(INPUT_Q_SEQ_LENS);
    if (qSeqLensShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t batch = qSeqLensShape->GetDim(0);

    const gert::Shape *qShape = context->GetInputShape(INPUT_Q);
    if (qShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    int64_t H_q = qShape->GetDim(1);

    gert::Shape *qFlatShape = context->GetOutputShape(OUTPUT_Q_FLAT);
    if (qFlatShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    qFlatShape->SetDimNum(4);
    qFlatShape->SetDim(0, batch);
    qFlatShape->SetDim(1, H_q);
    qFlatShape->SetDim(2, -1);
    qFlatShape->SetDim(3, kflat_dim);

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForStemOamPrepVarlenQ(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(OUTPUT_Q_FLAT, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(StemOamPrepVarlenQ)
    .InferShape(InferShapeForStemOamPrepVarlenQ)
    .InferDataType(InferDataTypeForStemOamPrepVarlenQ);

}  // namespace ops
