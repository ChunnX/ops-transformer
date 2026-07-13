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
#include "log/log.h"

namespace ops {

static constexpr int64_t DIM_QK = 128;
static constexpr int64_t INPUT_Q = 0;
static constexpr int64_t INPUT_Q_SEQ_LENS = 1;
static constexpr int64_t OUTPUT_Q_FLAT = 0;
static constexpr int64_t ATTR_STEM_BLOCK_SIZE = 0;
static constexpr int64_t ATTR_STEM_STRIDE = 1;

static ge::graphStatus InferShapeForStemOamPrepVarlenQ(gert::InferShapeContext* context) {
    OP_LOGI(context, "Enter StemOamPrepVarlenQ runtime infershape impl.");
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const gert::Shape *qShape = context->GetInputShape(INPUT_Q);
    OP_CHECK_NULL_WITH_CONTEXT(context, qShape);
    int64_t H_q = qShape->GetDim(1);

    const gert::Shape *qSeqLensShape = context->GetInputShape(INPUT_Q_SEQ_LENS);
    OP_CHECK_NULL_WITH_CONTEXT(context, qSeqLensShape);
    int64_t batch = qSeqLensShape->GetDim(0);

    const gert::Tensor *qSeqLensTensor = context->GetInputTensor(INPUT_Q_SEQ_LENS);
    OP_CHECK_NULL_WITH_CONTEXT(context, qSeqLensTensor);
    const int64_t *qSeqLensData = qSeqLensTensor->GetData<int64_t>();
    OP_CHECK_NULL_WITH_CONTEXT(context, qSeqLensData);

    const gert::RuntimeAttrs *attrs = context->GetAttrs();
    auto stemBlockSizePtr = attrs->GetAttrPointer<int64_t>(ATTR_STEM_BLOCK_SIZE);
    OP_CHECK_NULL_WITH_CONTEXT(context, stemBlockSizePtr);
    int64_t stemBlockSize = *stemBlockSizePtr;

    auto stemStridePtr = attrs->GetAttrPointer<int64_t>(ATTR_STEM_STRIDE);
    OP_CHECK_NULL_WITH_CONTEXT(context, stemStridePtr);
    int64_t stemStride = *stemStridePtr;
    int64_t kflat_dim = stemStride * DIM_QK;

    int64_t maxQLen = 0;
    for (int64_t i = 0; i < batch; i++) {
        if (qSeqLensData[i] > maxQLen) {
            maxQLen = qSeqLensData[i];
        }
    }
    int64_t maxQPadded = ((maxQLen + stemBlockSize - 1) / stemBlockSize) * stemBlockSize;
    int64_t maxQb = maxQPadded / stemBlockSize;

    gert::Shape *qFlatShape = context->GetOutputShape(OUTPUT_Q_FLAT);
    OP_CHECK_NULL_WITH_CONTEXT(context, qFlatShape);
    *qFlatShape = gert::Shape({batch, H_q, maxQb, kflat_dim});

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDataTypeForStemOamPrepVarlenQ(gert::InferDataTypeContext* context) {
    auto qDtype = context->GetInputDataType(INPUT_Q);
    if (qDtype == ge::DT_FLOAT16) {
        context->SetOutputDataType(OUTPUT_Q_FLAT, ge::DT_FLOAT16);
    } else {
        context->SetOutputDataType(OUTPUT_Q_FLAT, ge::DT_BF16);
    }
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(StemOamPrepVarlenQ)
    .InferShape(InferShapeForStemOamPrepVarlenQ)
    .InferDataType(InferDataTypeForStemOamPrepVarlenQ);

}  // namespace ops
