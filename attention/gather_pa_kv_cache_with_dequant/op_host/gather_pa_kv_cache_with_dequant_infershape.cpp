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
 * \file gather_pa_kv_cache_with_dequant_infershape.cpp
 * \brief InferShape implementation for fused GatherPaKvCache + Dequant
 */

#include "util/shape_util.h"
#include "register/op_impl_registry.h"
#include "op_common/log/log.h"

using namespace ge;
static constexpr size_t DIM_0 = 0;
static constexpr size_t DIM_1 = 1;
static constexpr size_t DIM_2 = 2;
static constexpr size_t DIM_3 = 3;
static constexpr size_t DIM_4 = 4;
static constexpr size_t DIM_5 = 5;
namespace ops {
namespace {

static constexpr size_t KEY_IN_INDEX = 6;
static constexpr size_t VALUE_IN_INDEX = 7;
static constexpr size_t KEY_OUT_INDEX = 0;
static constexpr size_t VALUE_OUT_INDEX = 1;
static constexpr size_t DIM_0 = 0;
static constexpr size_t DIM_1 = 1;
static constexpr size_t DIM_2 = 2;
static constexpr size_t DIM_3 = 3;
static constexpr size_t IS_SEQ_LENS_CUMSUM_INDEX = 1;
static constexpr size_t CACHE_MODE_INDEX = 0;

static ge::graphStatus CheckCommonPagedCacheLoad(gert::InferShapeContext *context)
{
    auto blockTables_shape = context->GetInputShape(DIM_2);
    OP_CHECK_NULL_WITH_CONTEXT(context, blockTables_shape);
    auto seqLens_shape = context->GetInputShape(DIM_3);
    OP_CHECK_NULL_WITH_CONTEXT(context, seqLens_shape);
    
    if (blockTables_shape->GetDimNum() != DIM_2) {
        OP_LOGE(context, "blockTables's dim num should be 2.");
        return ge::GRAPH_FAILED;
    }
    if (seqLens_shape->GetDimNum() != DIM_1) {
        OP_LOGE(context, "seqLens's dim num should be 1.");
        return ge::GRAPH_FAILED;
    }

    auto attrs = context->GetAttrs();
    auto is_seq_lens_cumsum = attrs->GetAttrPointer<bool>(IS_SEQ_LENS_CUMSUM_INDEX);
    if (is_seq_lens_cumsum == nullptr || *is_seq_lens_cumsum) {
        if ((blockTables_shape->GetDim(DIM_0) + 1) != seqLens_shape->GetDim(DIM_0)) {
            OP_LOGE(context, "seqLens's dim_0 %ld must equal blockTables's dim_0+1 %ld.", 
                    seqLens_shape->GetDim(DIM_0), blockTables_shape->GetDim(DIM_0) + 1);
            return ge::GRAPH_FAILED;
        }
    } else {
        if (blockTables_shape->GetDim(DIM_0) != seqLens_shape->GetDim(DIM_0)) {
            OP_LOGE(context, "seqLens's dim_0 %ld must equal blockTables's dim_0 %ld.", 
                    seqLens_shape->GetDim(DIM_0), blockTables_shape->GetDim(DIM_0));
            return ge::GRAPH_FAILED;
        }
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferShape4GatherPaKvCacheWithDequant(gert::InferShapeContext *context)
{
    OP_LOGD(context, "Begin to do InferShape4GatherPaKvCacheWithDequant");
    
    auto tensorKeyIn_shape = context->GetInputShape(KEY_IN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensorKeyIn_shape);
    auto tensorValueIn_shape = context->GetInputShape(VALUE_IN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, tensorValueIn_shape);
    auto output0_shape = context->GetOutputShape(KEY_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output0_shape);
    auto output1_shape = context->GetOutputShape(VALUE_OUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, output1_shape);
    
    *output0_shape = *tensorKeyIn_shape;
    *output1_shape = *tensorValueIn_shape;

    for (uint32_t i = 0; i < context->GetComputeNodeInputNum(); i++) {
        const gert::Shape *shape = context->GetInputShape(i);
        if (Ops::Base::IsUnknownRank(*shape)) {
            Ops::Base::SetUnknownRank(*output0_shape);
            Ops::Base::SetUnknownRank(*output1_shape);
            OP_LOGD(context, "input[%u].shape is UnknownRank, set output key.shape and value.shape to -2.", i);
            return ge::GRAPH_SUCCESS;
        }
    }
    
    for (uint32_t i = 0; i < context->GetComputeNodeInputNum(); i++) {
        auto shape = context->GetInputShape(i);
        if (Ops::Base::IsUnknownShape(*shape)) {
            OP_LOGD(context, "input[%u].shape is UnknownShape, set output key.shape and value.shape "
                    "same as input key.shape and value.shape respectively.", i);
            return ge::GRAPH_SUCCESS;
        }
    }

    OP_CHECK_IF(CheckCommonPagedCacheLoad(context) == ge::GRAPH_FAILED,
                OP_LOGE(context, "Failed to check common launch param."), return ge::GRAPH_FAILED);

    auto keyCacheShape = context->GetInputShape(DIM_0);
    auto valueCacheShape = context->GetInputShape(DIM_1);
    OP_CHECK_NULL_WITH_CONTEXT(context, keyCacheShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, valueCacheShape);

    if (keyCacheShape->GetDimNum() != DIM_4 || valueCacheShape->GetDimNum() != DIM_4) {
        OP_LOGE(context, "KCache&VCache's dim num should be 4.");
        return ge::GRAPH_FAILED;
    }
    if (output0_shape->GetDimNum() != DIM_3 || output1_shape->GetDimNum() != DIM_3) {
        OP_LOGE(context, "K&V's dim num should be 3.");
        return ge::GRAPH_FAILED;
    }
    
    return ge::GRAPH_SUCCESS;
}

static graphStatus InferDataType4GatherPaKvCacheWithDequant(gert::InferDataTypeContext *context)
{
    // Output is always BF16 regardless of input types
    context->SetOutputDataType(KEY_OUT_INDEX, ge::DT_BF16);
    context->SetOutputDataType(VALUE_OUT_INDEX, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}

} // namespace

IMPL_OP_INFERSHAPE(GatherPaKvCacheWithDequant)
    .InferShape(InferShape4GatherPaKvCacheWithDequant)
    .InferDataType(InferDataType4GatherPaKvCacheWithDequant);

} // namespace ops
